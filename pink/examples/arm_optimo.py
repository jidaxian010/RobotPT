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
import matplotlib.pyplot as plt

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
URDF_PATH    = "/home/jdx/Documents/1.0LatentAct/pink/data/roboligent_optimo_description/roboligent_optimo.urdf"
PACKAGE_DIRS = ["/home/jdx/Documents/1.0LatentAct/pink/data/roboligent_optimo_description"]
ROOT_JOINT   = None

# CSV produced by RosbagGripperPoseTracker.save_poses_csv()
# Columns: t, frame, pos_x, pos_y, pos_z, orient_x, orient_y, orient_z
#   pos_xyz    – displacement in mm from first valid frame, in initial gripper frame
#   orient_xyz – Euler XYZ angles (rad) relative to initial gripper orientation
_DEFAULT_CSV = (
    "/home/jdx/Documents/1.0LatentAct/RobotPT/posEstimate/data"
    "/pose3.csv"
)

# Run behavior flags (edit in code; no CLI flags needed)
RUN_MODE = "once"  # "loop" or "once"
JOINT_TRAJ_OUT = None  # None -> save in same folder as _DEFAULT_CSV
PLOT_SAVED_JOINT_TRAJ = True
TIME_SCALE = 0.8  # < 1.0 slows replay; 1.0 = original recorded speed



def read_poses(path):
    """
    Load gripper CoM poses from a CSV saved by save_poses_csv().

    Positions are converted mm → m.
    Euler XYZ angles are converted to rotation matrices.

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
                R = ScipyRotation.from_euler("xyz", euler).as_matrix()

                poses.append({
                    "t":        float(row["t"]),
                    "position": pos_m,
                    "rotation": R,
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

    q_ref = custom_configuration_vector(
        robot,
        joint1=0.0,
        joint2=2.85,
        joint3=0.0,
        joint4=-1.25,
        joint5=0.0,
        joint6=-1.66,
        joint7=0.0,
    )

    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    tasks = [end_effector_task, posture_task]
    for task in tasks:
        task.set_target_from_configuration(configuration)
    tasks.append(damping_task)
    viz.display(configuration.q)

    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=60.0, warn=False)
    dt   = rate.period

    # Load trajectory
    ee_poses = read_poses(PATH_TO_POSES)
    print(f"Loaded {len(ee_poses)} poses from {PATH_TO_POSES}")
    if not ee_poses:
        raise RuntimeError(f"No poses loaded — check path: {PATH_TO_POSES}")

    # Pre-build interpolators
    traj_t   = np.array([p["t"] for p in ee_poses])
    traj_pos = np.array([p["position"] for p in ee_poses])   # (N, 3) m
    traj_rot = ScipyRotation.from_matrix(
        np.stack([p["rotation"] for p in ee_poses])
    )
    slerp   = Slerp(traj_t, traj_rot)
    T_total = traj_t[-1]

    # Anchor trajectory to the robot's current EE pose.
    # CSV starts at (0,0,0) / identity, so relative motion is applied directly
    # on top of the initial EE position and orientation — matching arm_optimo_vel.py.
    initial_transform = configuration.get_transform_frame_to_world(EE_NAME)
    initial_position  = initial_transform.translation.copy()
    initial_rotation  = initial_transform.rotation.copy()
    print(f"Initial EE position: {initial_position.round(3)}")
    # Frame-axis offset for both EE and target visuals/motion:
    # desired: x points left, y points front  (from x=front, y=right)
    # This corresponds to a local-frame rotation of -90 deg about z.
    R_frame_offset = ScipyRotation.from_euler("z", -90.0, degrees=True).as_matrix()

    # Replay loop — interpolates at controller rate.
    # Use a simulation clock (dt accumulation) so the saved joint trajectory matches
    # exactly what the simulator executed, independent of wall-clock jitter.
    sim_time = 0.0
    solved_joint_records = []
    while True:
        replay_time = sim_time * TIME_SCALE
        if RUN_MODE == "once":
            if replay_time > T_total:
                break
            t_elapsed = replay_time
        else:
            t_elapsed = replay_time % T_total

        pos = np.array([np.interp(t_elapsed, traj_t, traj_pos[:, i])
                        for i in range(3)])
        rot = slerp(t_elapsed).as_matrix()

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

        if RUN_MODE == "once":
            solved_joint_records.append(
                {
                    # Save the exact replay-time index used to generate the target
                    # for this displayed simulator step.
                    "t": float(t_elapsed),
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
        save_joint_trajectory_csv(out_path, solved_joint_records)
        if PLOT_SAVED_JOINT_TRAJ:
            plot_joint_trajectory(solved_joint_records)


        
