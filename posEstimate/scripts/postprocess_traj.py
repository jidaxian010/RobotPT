#!/usr/bin/env python3
"""
Plot joint states from a rosbag, optionally overlaid with a .trj file.

Topics:
  /optimo/joint_states  -> joint positions over time
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as ScipyRotation
import yaml

sys.path = [p for p in sys.path if '/python3.10/' not in p] + \
           [p for p in sys.path if '/python3.10/' in p]

import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores
from data_processing.mocap_process import MocapGripperPoseProcessor

# ==========================================
# --- 1. USER CONFIGURATION ---
# ==========================================

TRJ_NAMES = ["p9-a2-g1", "p9-a2-g2", "p9-a2-g3"]   # <-- add/remove entries here

DATA_DIR  = Path(__file__).resolve().parents[1] / "data"
DATA_NAME = "p9-a2-g"          # <-- change this (or set to None to skip)
TRJ_PATH  = DATA_DIR / DATA_NAME / f"{DATA_NAME}.trj" if DATA_NAME else None
FORCE_RULE_PATH = DATA_DIR / DATA_NAME / "force_rule.yaml" if DATA_NAME else None

TOPIC_JOINT_STATES = "/optimo/joint_states"
TOPIC_EXT_WRENCH_EE = "/optimo/safety_monitor/ext_wrench_ee"

JOINT_LABELS = ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]

# Align bag to trj by matching the global minimum of this joint.
ALIGN_JOINT = "joint7"   # <-- change to any joint name

# Gripper ground-truth (MoCap) settings
CROP_START_S = 22   # <-- seconds from bag start, or None
CROP_END_S   = 50   # <-- seconds from bag start, or None

# URDF / FK settings (mirrors arm_optimo.py)
_URDF_DIR  = Path(__file__).resolve().parents[2] / "pink" / "data" / "roboligent_optimo_description"
URDF_PATH  = str(_URDF_DIR / "roboligent_optimo.urdf")
URDF_PKGS  = [str(_URDF_DIR)]
EE_FRAME   = "link8"

# Inverse of arm_optimo.py R_COMBINED: maps EE frame → gripper frame
_R_SIDE_TO_DEFAULT = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0],
    [ 0.0, -1.0,  0.0],
])
_R_GRIPPER_TO_EE = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
])
R_COMBINED     = _R_GRIPPER_TO_EE @ _R_SIDE_TO_DEFAULT   # gripper → EE
R_EE_TO_GRIPPER = R_COMBINED.T                            # EE → gripper

# ==========================================
# --- 2. HELPERS ---
# ==========================================

typestore = get_typestore(Stores.LATEST)


def _stamp_sec(msg, fallback_ns):
    try:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    except Exception:
        return fallback_ns * 1e-9


def read_joint_states(bag_path, topic):
    """Return (timestamps, joint_names, positions [rad], t0_abs) sorted by joint number."""
    times_abs, positions, names = [], [], None

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            available = [c.topic for c in reader.connections]
            raise RuntimeError(f"Topic {topic!r} not found.\nAvailable: {available}")

        for conn, ts, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            t = _stamp_sec(msg, ts)
            if names is None:
                names = list(msg.name)
            times_abs.append(t)
            positions.append(list(msg.position))

    if not times_abs:
        raise RuntimeError(f"No messages in {topic!r}")

    import re
    def _joint_key(name):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else name

    sorted_idx = sorted(range(len(names)), key=lambda i: _joint_key(names[i]))
    names = [names[i] for i in sorted_idx]

    times_abs = np.array(times_abs)
    t0_abs = float(times_abs[0])
    times = times_abs - t0_abs
    positions = np.array(positions)[:, sorted_idx]

    print(f"[joint_states] {len(times)} msgs, {len(names)} joints: {names}")
    return times, names, positions, t0_abs


def read_ext_wrench_ee(bag_path, topic, t0_abs=None):
    """Return (timestamps, wrench [Fx,Fy,Fz,Tx,Ty,Tz]) with optional alignment to t0_abs."""
    times_abs, wrench = [], []

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            available = [c.topic for c in reader.connections]
            raise RuntimeError(f"Topic {topic!r} not found.\nAvailable: {available}")

        for conn, ts, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            t = _stamp_sec(msg, ts)
            f = msg.wrench.force
            tor = msg.wrench.torque
            times_abs.append(t)
            wrench.append([f.x, f.y, f.z, tor.x, tor.y, tor.z])

    if not times_abs:
        raise RuntimeError(f"No messages in {topic!r}")

    times_abs = np.array(times_abs)
    if t0_abs is None:
        t0_abs = float(times_abs[0])
    times = times_abs - t0_abs
    wrench = np.array(wrench)

    print(f"[ext_wrench_ee] {len(times)} msgs from {topic}")
    return times, wrench



def align_by_global_min(bag_times, bag_positions, names,
                        trj_times, trj_data, ref_joint):
    """
    Shift trj_times so the global minimum of ref_joint aligns with the bag.
    Returns shifted trj_times.
    """
    if ref_joint not in names:
        raise ValueError(f"ALIGN_JOINT {ref_joint!r} not in bag joints: {names}")

    j = names.index(ref_joint)
    t_min_bag = bag_times[int(np.argmin(bag_positions[:, j]))]
    t_min_trj = trj_times[int(np.argmin(trj_data[:, j]))]

    offset = t_min_bag - t_min_trj
    print(f"[align] {ref_joint} global min — bag t={t_min_bag:.2f}s, "
          f"trj t={t_min_trj:.2f}s → shift trj by {offset:+.2f}s")
    return trj_times + offset


def compute_ee_in_initial_frame(robot, frame_id, positions):
    """
    Run FK on each row of positions (N, 7) rad.
    Returns (pos_mm, euler_deg) in the initial EE frame.
      pos_mm   : (N, 3)  — displacement from first pose, in mm
      euler_deg: (N, 3)  — XYZ Euler angles relative to first pose, in deg
    """
    pos_world = np.empty((len(positions), 3))
    rot_world = np.empty((len(positions), 3, 3))

    for i, q_joints in enumerate(positions):
        q = np.zeros(robot.model.nq)
        q[:len(q_joints)] = q_joints
        pin.forwardKinematics(robot.model, robot.data, q)
        pin.updateFramePlacements(robot.model, robot.data)
        tf = robot.data.oMf[frame_id]
        pos_world[i] = tf.translation
        rot_world[i] = tf.rotation

    R0, t0 = rot_world[0], pos_world[0]
    # relative pose in initial EE frame, then rotate to gripper frame
    pos_mm    = np.array([R_EE_TO_GRIPPER @ (R0.T @ (p - t0)) for p in pos_world]) * 1000.0
    euler_deg = np.degrees(np.array([
        ScipyRotation.from_matrix(R_EE_TO_GRIPPER @ (R0.T @ R) @ R_COMBINED).as_euler("xyz")
        for R in rot_world
    ]))
    return pos_mm, euler_deg


def parse_trj(path: Path):
    """Parse .trj file. Returns (times, data)."""
    with open(path) as f:
        lines = f.readlines()
    hz = float(lines[2].split()[0])
    frames = []
    for line in lines[4:]:
        line = line.strip().rstrip(",").strip()
        if line:
            frames.append([float(v) for v in line.split()])
    data = np.array(frames)
    t = np.arange(len(frames)) / hz
    print(f"[trj] {len(frames)} frames @ {hz:.1f} Hz  ({path.name})")
    return t, data


def load_force_rules(path: Path):
    """Load force min/max from force_rule.yaml. Returns dict or None."""
    if path is None or not path.exists():
        print(f"Warning: FORCE_RULE_PATH not found: {path}")
        return None
    with open(path) as f:
        doc = yaml.safe_load(f)
    force_rules = doc.get("force_N", None) if isinstance(doc, dict) else None
    if force_rules is None:
        print(f"Warning: no force_N in rulebook: {path}")
        return None
    return force_rules


# ==========================================
# --- 3. PLOTTING ---
# ==========================================


def plot_joint_states(bags, names, trj_times=None, trj_data=None):
    """
    bags: list of (label, times, positions) — one entry per bag.
    Plots mean ± std across bags as a shaded area, plus the .trj overlay.
    """
    n = bags[0][2].shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]


    # Common time grid: finest resolution, spanning shortest bag
    t_end = min(b[1][-1] for b in bags)
    n_pts = max(len(b[1]) for b in bags)
    t_common = np.linspace(0, t_end, n_pts)

    # Interpolate all bags onto common grid: shape (n_bags, n_pts, n_joints)
    interp_all = np.stack([
        np.column_stack([
            np.interp(t_common, times, positions[:, j])
            for j in range(n)
        ])
        for _, times, positions in bags
    ])  # (n_bags, n_pts, n_joints)

    mean_pos = np.degrees(interp_all.mean(axis=0))   # (n_pts, n_joints)
    std_pos  = np.degrees(interp_all.std(axis=0))    # (n_pts, n_joints)

    for i, ax in enumerate(axes):
        joint_label = names[i] if i < len(names) else f"joint_{i+1}"

        ax.plot(t_common, mean_pos[:, i], linewidth=1.4, label="actual trajectory", color="red")
        ax.fill_between(t_common,
                        mean_pos[:, i] - std_pos[:, i],
                        mean_pos[:, i] + std_pos[:, i],
                        alpha=0.3, color="red", label="±std")

        if trj_times is not None and trj_data is not None and i < trj_data.shape[1]:
            ax.plot(trj_times, np.degrees(trj_data[:, i]),
                    label="commanded trajectory", linewidth=1.2, color="black")

        ax.set_ylabel(f"{joint_label}\n(deg)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.4)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

def plot_ee_comparison(bags_ee, gt_gripper=None):
    """
    6 subplots (x, y, z, roll, pitch, yaw) — one per channel.
    bags_ee   : list of (label, times, pos_mm, euler_deg) — FK from TRJ_NAMES bags
    gt_gripper: (times_s, pos_mm, euler_rad) — MoCap ground truth, or None
    """
    labels_pos   = ["x", "y", "z"]
    labels_euler = ["roll", "pitch", "yaw"]
    units        = ["mm"] * 3 + ["deg"] * 3

    t_end    = min(b[1][-1] for b in bags_ee)
    n_pts    = max(len(b[1]) for b in bags_ee)
    t_common = np.linspace(0, t_end, n_pts)

    pos_all   = np.stack([
        np.column_stack([np.interp(t_common, t, pos[:, j])   for j in range(3)])
        for _, t, pos, _ in bags_ee
    ])  # (n_bags, n_pts, 3)
    euler_all = np.stack([
        np.column_stack([np.interp(t_common, t, euler[:, j]) for j in range(3)])
        for _, t, _, euler in bags_ee
    ])

    pos_mean,   pos_std   = pos_all.mean(axis=0),   pos_all.std(axis=0)
    euler_mean, euler_std = euler_all.mean(axis=0), euler_all.std(axis=0)

    fig, axes = plt.subplots(6, 1, figsize=(12, 2.2 * 6), sharex=True)
    gt_pos_mm   = gt_gripper[1]                  if gt_gripper is not None else None
    gt_euler_deg = np.degrees(gt_gripper[2])     if gt_gripper is not None else None
    gt_times     = gt_gripper[0]                 if gt_gripper is not None else None

    for i in range(6):
        ax = axes[i]
        if i < 3:
            mean, std = pos_mean[:, i],   pos_std[:, i]
            label_ch  = labels_pos[i]
            gt_data   = gt_pos_mm[:, i]   if gt_pos_mm is not None else None
        else:
            mean, std = euler_mean[:, i - 3], euler_std[:, i - 3]
            label_ch  = labels_euler[i - 3]
            gt_data   = gt_euler_deg[:, i - 3] if gt_euler_deg is not None else None

        ax.plot(t_common, mean, linewidth=1.4, color="C0", label="actual trajectory")
        ax.fill_between(t_common, mean - std, mean + std,
                        alpha=0.3, color="C0", label="±std")

        if gt_data is not None:
            ax.plot(gt_times, gt_data, linewidth=1.2, color="black", label="mocap trajectory")

        ax.set_ylabel(f"{label_ch}\n({units[i]})", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.4)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()


def plot_ext_wrench_ee(bags_wrench, force_rules=None, danger_tol_n=20.0):
    """
    3 subplots (Fx, Fy, Fz) — mean ± std across bags.
    bags_wrench: list of (label, times, wrench) — wrench in EE frame
    """
    labels = ["Fx", "Fy", "Fz"]
    units  = ["N", "N", "N"]
    axis_keys = ["x", "y", "z"]

    t_end    = min(b[1][-1] for b in bags_wrench)
    n_pts    = max(len(b[1]) for b in bags_wrench)
    t_common = np.linspace(0, t_end, n_pts)

    wrench_all = np.stack([
        np.column_stack([np.interp(t_common, t, w[:, j]) for j in range(3)])
        for _, t, w in bags_wrench
    ])  # (n_bags, n_pts, 3)

    mean_w, std_w = wrench_all.mean(axis=0), wrench_all.std(axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 2.2 * 3), sharex=True)
    for i in range(3):
        ax = axes[i]
        ax.plot(t_common, mean_w[:, i], linewidth=1.4, color="green",
                label="actual force")
        ax.fill_between(t_common, mean_w[:, i] - std_w[:, i],
                        mean_w[:, i] + std_w[:, i],
                        alpha=0.3, color="green", label="±std")

        if force_rules is not None:
            rule = force_rules.get(axis_keys[i], None)
            if rule is not None:
                warn_min = float(rule["min"])
                warn_max = float(rule["max"])
                danger_min = warn_min - danger_tol_n
                danger_max = warn_max + danger_tol_n
                ax.axhline(warn_min, linestyle="--", color="black", linewidth=1.0,
                           label="warning min/max")
                ax.axhline(warn_max, linestyle="--", color="black", linewidth=1.0)
                ax.axhline(danger_min, linestyle="-.", color="black", linewidth=1.0,
                           label=f"danger min/max")
                ax.axhline(danger_max, linestyle="-.", color="black", linewidth=1.0)

        ax.set_ylabel(f"{labels[i]}\n({units[i]})", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(False)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()


# ==========================================
# --- 4. MAIN ---
# ==========================================

def main():
    trj_times, trj_data = None, None
    if TRJ_PATH is not None:
        if not TRJ_PATH.exists():
            print(f"Warning: TRJ_PATH not found: {TRJ_PATH}")
        else:
            trj_times, trj_data = parse_trj(TRJ_PATH)
    force_rules = load_force_rules(FORCE_RULE_PATH)

    # --- MoCap ground-truth gripper pose (from DATA_NAME bag) ---
    gt_gripper = None   # (times_s, pos_mm, euler_rad)
    if DATA_NAME is not None:
        trj_bag_path = Path(f"~/Downloads/{DATA_NAME}").expanduser()
        print(f"\nReading MoCap gripper from: {trj_bag_path}")
        proc = MocapGripperPoseProcessor(
            bag_path=trj_bag_path,
            output_dir=DATA_DIR / DATA_NAME,
            crop_start_s=CROP_START_S,
            crop_end_s=CROP_END_S,
            show_image=False,
        )
        proc.load()
        proc.smooth_poses()
        g_times, g_pos, g_euler = proc.get_pose_arrays()
        print(f"[gripper] {len(g_times)} poses in initial gripper frame")
        gt_gripper = (g_times, g_pos, g_euler)

    robot    = pin.RobotWrapper.BuildFromURDF(URDF_PATH, URDF_PKGS, None)
    frame_id = robot.model.getFrameId(EE_FRAME)
    print(f"[FK] built robot, EE frame '{EE_FRAME}' id={frame_id}")

    bags_joint  = []   # (label, times, positions) — for joint plot
    bags_ee     = []   # (label, times, pos_mm, euler_deg) — FK EE in initial EE frame
    bags_wrench = []   # (label, times, wrench) — ext_wrench_ee plot
    first_names = None

    for name in TRJ_NAMES:
        bag_path = Path(f"~/Downloads/{name}").expanduser()
        print(f"\nReading bag: {bag_path}")
        times_j, names, positions, t0_abs = read_joint_states(bag_path, TOPIC_JOINT_STATES)
        times_w, wrench = read_ext_wrench_ee(bag_path, TOPIC_EXT_WRENCH_EE, t0_abs=t0_abs)

        if first_names is None:
            first_names = names

        if trj_times is not None:
            trj_times_aligned = align_by_global_min(
                times_j, positions, names, trj_times, trj_data, ALIGN_JOINT
            )
            t1, t2 = trj_times_aligned[0], trj_times_aligned[-1]
            mask_j = (times_j >= t1) & (times_j <= t2)
            times_j_plot   = times_j[mask_j] - t1
            positions_plot = positions[mask_j]
            trj_times_plot = trj_times_aligned - t1

            mask_w = (times_w >= t1) & (times_w <= t2)
            times_w_plot = times_w[mask_w] - t1
            wrench_plot  = wrench[mask_w]
        else:
            times_j_plot, positions_plot = times_j, positions
            trj_times_plot = None
            times_w_plot, wrench_plot = times_w, wrench

        bags_joint.append((name, times_j_plot, positions_plot))
        bags_wrench.append((name, times_w_plot, wrench_plot))

        pos_mm, euler_deg = compute_ee_in_initial_frame(robot, frame_id, positions_plot)
        bags_ee.append((name, times_j_plot, pos_mm, euler_deg))

    plot_joint_states(bags_joint, first_names, trj_times_plot, trj_data)
    plot_ee_comparison(bags_ee, gt_gripper)
    plot_ext_wrench_ee(bags_wrench, force_rules=force_rules)

    plt.show()


if __name__ == "__main__":
    main()
