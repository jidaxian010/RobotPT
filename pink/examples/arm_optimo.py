#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Optimo arm replaying a recorded gripper trajectory from a fiducial marker."""

import argparse
import csv
import time
from pathlib import Path
import numpy as np
import qpsolvers
from scipy.spatial.transform import Rotation as ScipyRotation, Slerp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

import sys
sys.path = [p for p in sys.path if '/python3.10/' not in p] + \
           [p for p in sys.path if '/python3.10/' in p]

import meshcat_shapes
import pink
import pinocchio as pin
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask, DampingTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer

try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, "
        "try `[conda|pip] install loop-rate-limiters`"
    ) from exc


EE_NAME      = "link8"
_DATA_DIR    = Path(__file__).resolve().parent.parent / "data" / "roboligent_optimo_description"
URDF_PATH    = str(_DATA_DIR / "roboligent_optimo.urdf")
PACKAGE_DIRS = [str(_DATA_DIR)]
ROOT_JOINT   = None
DATA_NAME  = "P2-A4"  # for default CSV path; edit one value here

RUN_MODE = "loop"  # "loop" or "once"
TIME_SCALE = 0.5  # < 1.0 slows replay; 1.0 = original recorded speed
SOLVER_HZ = 200.0  # IK solver frequency (higher = smoother tracking)
REPARAMETRIZE = True  # arc-length reparametrization for uniform speed



# Map recorded gripper-frame data into the simulator EE frame:
# gripper +y -> EE +x
# gripper +x -> EE -y
# gripper +z -> EE +z
R_GRIPPER_TO_EE = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])

# CSV produced by RosbagGripperPoseTracker.save_poses_csv()
# Columns: t, frame, pos_x, pos_y, pos_z, orient_x, orient_y, orient_z
#   pos_xyz    – displacement in mm from first valid frame, in initial gripper frame
#   orient_xyz – Euler XYZ angles (rad) relative to initial gripper orientation
_REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CSV = str(_REPO_ROOT / "posEstimate" / "data" / DATA_NAME / f"{DATA_NAME}.csv")

JOINT_TRAJ_OUT = None  # None -> save in same folder as _DEFAULT_CSV
PLOT_SAVED_JOINT_TRAJ = True
WARMUP_SKIP_SEC = 0.3  # in RUN_MODE="once", don't save the first N seconds of replay
TRJ_HZ = 50.0  # output .trj frequency (frames are dropped/interpolated from solver Hz)

# ── Start configuration ──────────────────────────────────────────────────────
# Two ways to define where the robot's EE starts before replaying the trajectory:
#
#   Mode A – joint angles (current default):
#     Set START_EE_POSITION = None.  q_ref joint angles are used directly.
#
#   Mode B – world-frame EE position:
#     Set START_EE_POSITION to an [x, y, z] vector (meters, robot world frame).
#     IK pre-solve will find joint angles that place the EE at that position.
#     Workflow:
#       1. Run once with START_EE_POSITION=None — note the printed "Initial EE position".
#       2. Decide where you want the EE to start and set START_EE_POSITION.
#       3. Optionally set START_EE_ROTATION (3×3 matrix) to fix orientation too.
#          Leave None to keep orientation from q_ref.
#
# To use camera-frame gripper data as a start hint (requires calibration):
#   T_cam_to_world = <your camera→robot-base 4×4 transform>
#   gripper_pos_cam = <row-0 absolute camera-frame position from CSV, in meters>
#   START_EE_POSITION = (T_cam_to_world[:3, :3] @ gripper_pos_cam + T_cam_to_world[:3, 3])
START_EE_POSITION = None        # None → use q_ref directly; or np.array([x, y, z])
START_EE_ROTATION = None        # None → keep orientation from q_ref; or np.ndarray(3,3)
IK_PRESOLVE_ITERS = 800         # IK gradient steps to converge before replay



def read_poses(path):
    """
    Load gripper CoM poses from a CSV saved by save_poses_csv().

    Positions are converted mm → m.
    Euler XYZ angles are converted to rotation matrices.
    Loaded poses are remapped from the recorded gripper frame into the
    simulator end-effector frame.

    Returns
    -------
    list of {"t": float, "position": np.ndarray(3,), "rotation": np.ndarray(3,3)}
    """
    poses = []
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pos_m = np.array([
                    float(row["pos_x"]),
                    float(row["pos_y"]),
                    float(row["pos_z"]),
                ]) / 1000.0   # mm → m

                euler = np.array([
                    float(row["orient_x"]),
                    float(row["orient_y"]),
                    float(row["orient_z"]),
                ])
                R_gripper = ScipyRotation.from_euler("xyz", euler).as_matrix()
                pos_ee = R_GRIPPER_TO_EE @ pos_m
                R_ee = R_GRIPPER_TO_EE @ R_gripper @ R_GRIPPER_TO_EE.T

                poses.append({
                    "t":        float(row["t"]),
                    "position": pos_ee,
                    "rotation": R_ee,
                })
    except FileNotFoundError:
        print(f"Error: CSV not found at {path}")
    except Exception as exc:
        print(f"Error reading CSV: {exc}")
    return poses


def save_joint_trajectory_csv(csv_path, records):
    """Save solved joint trajectory records to CSV."""
    if not records:
        print("No solved joint trajectory records to save.")
        return None

    q_dim = len(records[0]["q"])
    header = ["t", "sample_idx"] + [f"joint{i+1}" for i in range(q_dim)]
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for rec in records:
            writer.writerow(
                [round(rec["t"], 6), rec["sample_idx"]]
                + [round(float(v), 8) for v in rec["q"]]
            )
    print(f"Saved solved joint trajectory: {csv_path}")
    return csv_path


def save_trj(trj_path, records, target_hz=50.0):
    """Save solved joint trajectory as a .trj file, resampled to target_hz."""
    if not records:
        print("No solved joint trajectory records to save.")
        return None

    t_arr = np.array([r["t"] for r in records], dtype=float)
    q_arr = np.array([r["q"] for r in records], dtype=float)

    duration = t_arr[-1] - t_arr[0]
    if duration <= 0:
        print("Warning: zero-duration trajectory, skipping .trj save.")
        return None

    n_frames = int(duration * target_hz) + 1
    t_new = np.linspace(t_arr[0], t_arr[-1], n_frames)

    n_joints = q_arr.shape[1]
    q_new = np.zeros((n_frames, n_joints))
    for j in range(n_joints):
        q_new[:, j] = np.interp(t_new, t_arr, q_arr[:, j])

    actual_hz = (n_frames - 1) / duration

    trj_path = Path(trj_path)
    trj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trj_path, "w") as f:
        f.write(f"{n_frames} [size]\n")
        f.write(f"{duration:.6f} [sec]\n")
        f.write(f"{actual_hz:.6f} [Hz]\n")
        f.write("1 [type]\n")
        for row in q_new:
            line = " ".join(f"{v:.6f}" for v in row) + " ,"
            f.write(line + "\n")

    print(f"Saved .trj: {trj_path}  ({n_frames} frames @ {actual_hz:.3f} Hz)")
    return trj_path


def plot_joint_trajectory(records):
    """Plot solved joint trajectories over time."""
    if not records:
        print("No solved joint trajectory records to plot.")
        return

    t = np.array([r["t"] for r in records], dtype=float)
    q = np.array([r["q"] for r in records], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, q, linewidth=1.2)
    ax.set_title("Solved Joint Trajectory")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint position (rad)")
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()


def print_joint_trajectory_debug(records, q_init, n=5):
    """Print raw solved joint samples and initial-step deltas for debugging."""
    if not records:
        print("No solved joint records to print.")
        return

    q_init = np.asarray(q_init, dtype=float).copy()
    print("\nRaw joint debug")
    print(f"Initial q (before replay): {np.round(q_init, 6)}")

    n_show = min(n, len(records))
    for i in range(n_show):
        rec = records[i]
        q = np.asarray(rec["q"], dtype=float)
        dq_from_init = q - q_init
        print(
            f"  sample {i:02d}  t={rec['t']:.4f}s"
            f"  q={np.round(q, 6)}"
            f"  dq_init={np.round(dq_from_init, 6)}"
        )

    if len(records) >= 2:
        q0 = np.asarray(records[0]["q"], dtype=float)
        q1 = np.asarray(records[1]["q"], dtype=float)
        print(f"First-step delta (sample1 - sample0): {np.round(q1 - q0, 6)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay a gripper trajectory on the Optimo arm."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=_DEFAULT_CSV,
        help="Path to gripper pose CSV (columns: t,pos_x,pos_y,pos_z,orient_x,orient_y,orient_z)",
    )
    args = parser.parse_args()
    PATH_TO_POSES = args.csv_path

    robot = pin.RobotWrapper.BuildFromURDF(
        filename=URDF_PATH,
        package_dirs=PACKAGE_DIRS,
        root_joint=ROOT_JOINT,
    )
    viz    = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    meshcat_shapes.frame(viewer["end_effector_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["end_effector"],        opacity=1.0)

    # Tasks
    end_effector_task = FrameTask(
        EE_NAME,
        position_cost=1.0,
        orientation_cost=0.5,
        lm_damping=1.0,
    )
    posture_task = PostureTask(cost=1e-3)
    damping_task = DampingTask(cost=1e-3)

    # q_ref = custom_configuration_vector(
    #     robot,
    #     joint1=0.0,
    #     joint2=2.85,
    #     joint3=0.0,
    #     joint4=-1.25,
    #     joint5=0.0,
    #     joint6=-1.66,
    #     joint7=0.0,
    # )

    # q_ref = custom_configuration_vector(
    #     robot,
    #     joint1=0.0,
    #     joint2=2.85,
    #     joint3=0.0,
    #     joint4=-1.58,
    #     joint5=0.0,
    #     joint6=-1.43,
    #     joint7=0.0,
    # )

    q_ref = custom_configuration_vector( #aba
        robot,
        joint1=0.0,
        joint2=2.69,
        joint3=0.0,
        joint4=-1.87,
        joint5=0.0,
        joint6=0.69,
        joint7=0.0,
    )
    
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    configuration = pink.Configuration(robot.model, robot.data, q_ref)

    # ── Mode B: IK pre-solve to reach a desired world-frame EE position ──────
    if START_EE_POSITION is not None:
        pre_target = configuration.get_transform_frame_to_world(EE_NAME)
        pre_target = pin.SE3(
            START_EE_ROTATION if START_EE_ROTATION is not None else pre_target.rotation.copy(),
            np.asarray(START_EE_POSITION, dtype=float),
        )
        pre_ee_task = FrameTask(
            EE_NAME,
            position_cost=1.0,
            orientation_cost=0.1 if START_EE_ROTATION is not None else 0.0,
            lm_damping=1.0,
        )
        pre_ee_task.set_target(pre_target)
        pre_posture = PostureTask(cost=1e-3)
        pre_posture.set_target(q_ref)
        pre_tasks = [pre_ee_task, pre_posture, DampingTask(cost=1e-3)]
        dt_pre = 1.0 / 200.0
        for _ in range(IK_PRESOLVE_ITERS):
            vel = solve_ik(configuration, pre_tasks, dt_pre, solver=solver)
            configuration.integrate_inplace(vel, dt_pre)
        actual_ee = configuration.get_transform_frame_to_world(EE_NAME).translation
        print(f"Pre-solve: target={np.asarray(START_EE_POSITION).round(4)}, "
              f"actual EE={actual_ee.round(4)}")
        print(f"Pre-solved joints (copy to q_ref to skip pre-solve next time):")
        jnames = [robot.model.names[i] for i in range(1, robot.model.njoints)]
        for name, val in zip(jnames, configuration.q[7:] if len(configuration.q) > 7 else configuration.q):
            print(f"  {name}={val:.6f}")
        viz.display(configuration.q)
    # ─────────────────────────────────────────────────────────────────────────

    q_init_for_debug = configuration.q.copy()
    tasks = [end_effector_task, posture_task]
    for task in tasks:
        task.set_target_from_configuration(configuration)
    tasks.append(damping_task)
    viz.display(configuration.q)

    rate = RateLimiter(frequency=SOLVER_HZ, warn=False)
    dt   = rate.period

    # Load trajectory
    ee_poses = read_poses(PATH_TO_POSES)
    print(f"Loaded {len(ee_poses)} poses from {PATH_TO_POSES}")
    if not ee_poses:
        raise RuntimeError(f"No poses loaded — check path: {PATH_TO_POSES}")

    # Pre-build interpolators — cubic spline for C2-continuous position
    traj_t   = np.array([p["t"] for p in ee_poses])
    traj_pos = np.array([p["position"] for p in ee_poses])   # (N, 3) m
    traj_rot = ScipyRotation.from_matrix(
        np.stack([p["rotation"] for p in ee_poses])
    )
    pos_spline = CubicSpline(traj_t, traj_pos, bc_type="clamped")
    slerp      = Slerp(traj_t, traj_rot)
    T_total    = traj_t[-1]

    # Arc-length reparameterization: resample so the EE moves at constant speed
    # along the same geometric path. This eliminates sudden fast/slow sections.
    if REPARAMETRIZE:
        n_resample = max(len(traj_t) * 4, 2000)
        t_dense = np.linspace(traj_t[0], traj_t[-1], n_resample)
        pos_dense = pos_spline(t_dense)
        ds = np.linalg.norm(np.diff(pos_dense, axis=0), axis=1)
        arc = np.zeros(n_resample)
        arc[1:] = np.cumsum(ds)
        total_arc = arc[-1]
        # Map: uniform arc-length → original time
        arc_uniform = np.linspace(0, total_arc, n_resample)
        t_reparam = np.interp(arc_uniform, arc, t_dense)
        # Rebuild position and rotation on the new time grid
        traj_pos_rp = pos_spline(t_reparam)
        traj_rot_rp = slerp(np.clip(t_reparam, traj_t[0], traj_t[-1]))
        # Replace interpolators with reparameterized ones
        traj_t_rp = np.linspace(0, T_total, n_resample)
        pos_spline = CubicSpline(traj_t_rp, traj_pos_rp, bc_type="clamped")
        slerp      = Slerp(traj_t_rp, traj_rot_rp)
        # Verify
        speeds_rp = np.linalg.norm(np.diff(traj_pos_rp, axis=0), axis=1) / np.diff(traj_t_rp)
        print(f"  Reparameterized: speed min={speeds_rp.min():.1f} max={speeds_rp.max():.1f} "
              f"mean={speeds_rp.mean():.1f} mm/s  (path={total_arc*1000:.1f} mm)")

    # Anchor trajectory to the robot's current EE pose.
    # CSV starts at (0,0,0) / identity, so relative motion is applied directly
    # on top of the initial EE position and orientation — matching arm_optimo_vel.py.
    initial_transform = configuration.get_transform_frame_to_world(EE_NAME)
    initial_position  = initial_transform.translation.copy()
    initial_rotation  = initial_transform.rotation.copy()
    print(f"Initial EE position: {initial_position.round(3)}")
    print(f"Solver: {SOLVER_HZ} Hz  |  Reparametrize: {REPARAMETRIZE}")
    # Frame-axis offset: maps CSV gripper frame → robot EE frame.
    # CSV frame: y=up, z=out-of-surface (forward).
    # EE frame (with current q_ref): y=up, z=forward → frames are aligned.
    R_frame_offset = np.eye(3)

    # Replay loop — interpolates at controller rate.
    # Use a simulation clock (dt accumulation) so the saved joint trajectory matches
    # exactly what the simulator executed, independent of wall-clock jitter.
    sim_time = 0.0
    sim_time_offset = None   # wall-clock time at first saved record (after warmup)
    solved_joint_records = []
    if RUN_MODE == "once" and WARMUP_SKIP_SEC <= 0:
        # Include the true initial simulator state so saved/plot trajectories
        # start from the actual robot configuration before the first IK step.
        solved_joint_records.append(
            {
                "t": 0.0,
                "sample_idx": 0,
                "q": configuration.q.copy(),
            }
        )
    while True:
        replay_time = sim_time * TIME_SCALE
        if RUN_MODE == "once":
            if replay_time > T_total:
                break
            t_elapsed = replay_time
        else:
            t_elapsed = replay_time % T_total

        pos = pos_spline(t_elapsed)
        rot = slerp(np.clip(t_elapsed, traj_t[0] if not REPARAMETRIZE else 0.0,
                            traj_t[-1] if not REPARAMETRIZE else T_total)).as_matrix()

        target = end_effector_task.transform_target_to_world
        # Apply the recorded displacement in the EE's initial frame, so the
        # motion directions rotate with the EE initial orientation too.
        target.translation = initial_position + initial_rotation @ R_frame_offset @ pos
        # Anchor raw relative CSV rotation to the current EE orientation so
        # the target frame starts aligned with the EE frame at t=0.
        target.rotation    = initial_rotation @ R_frame_offset @ rot

        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)
        viz.display(configuration.q)

        if RUN_MODE == "once" and float(t_elapsed) >= float(WARMUP_SKIP_SEC):
            if sim_time_offset is None:
                sim_time_offset = sim_time
            solved_joint_records.append(
                {
                    "t": float(sim_time - sim_time_offset),
                    "sample_idx": len(solved_joint_records),
                    "q": configuration.q.copy(),
                }
            )

        viewer["end_effector_target"].set_transform(target.np)
        ee_tf = configuration.get_transform_frame_to_world(EE_NAME)
        ee_tf_vis = pin.SE3(ee_tf.rotation @ R_frame_offset, ee_tf.translation)
        viewer["end_effector"].set_transform(ee_tf_vis.np)

        rate.sleep()
        sim_time += dt

    if RUN_MODE == "once":
        default_save_dir = Path(_DEFAULT_CSV).parent
        out_path = (
            Path(JOINT_TRAJ_OUT)
            if JOINT_TRAJ_OUT is not None
            else default_save_dir / f"{Path(PATH_TO_POSES).stem}_solved_qs.csv"
        )
        print_joint_trajectory_debug(solved_joint_records, q_init_for_debug, n=8)
        save_joint_trajectory_csv(out_path, solved_joint_records)
        trj_path = default_save_dir / f"{Path(PATH_TO_POSES).stem}.trj"
        save_trj(trj_path, solved_joint_records, target_hz=TRJ_HZ)
        if PLOT_SAVED_JOINT_TRAJ:
            plot_joint_trajectory(solved_joint_records)


        
