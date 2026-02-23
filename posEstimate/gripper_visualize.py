import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def plot_all_markers_3d(all_odometry):
    """
    Fig 1 — interactive 3D trajectory for all detected marker centroids.

    Args:
        all_odometry: Dict[int, np.ndarray]  marker_id -> (M, 4) [x, y, z, frame_idx]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    marker_ids = sorted(all_odometry.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(marker_ids), 1)))

    for color, marker_id in zip(colors, marker_ids):
        odo = all_odometry[marker_id]
        if len(odo) == 0:
            continue
        ax.plot(odo[:, 0], odo[:, 2], -odo[:, 1],
                color=color, label=f"Marker {marker_id}")

    ax.scatter(0, 0, 0, color="black", s=100, marker="*", label="Camera")
    ax.legend()
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Trajectory – All ArUco Markers")
    ax.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_com_smooth_3d(pos_smooth):
    """
    Fig 2 — interactive 3D plot of the smoothed gripper CoM trajectory.

    Args:
        pos_smooth: np.ndarray (N, 3)  relative positions in mm
                    (first row = [0,0,0]), as returned by traj_smooth.smooth_trajectory()
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Same axis convention: x lateral, depth as Y, -y as Z
    ax.plot(pos_smooth[:, 0], pos_smooth[:, 2], -pos_smooth[:, 1],
            color="tab:blue", linewidth=1.8, label="CoM (smoothed)")
    ax.scatter(*[pos_smooth[0,  0], pos_smooth[0,  2], -pos_smooth[0,  1]],
               color="tab:green", marker="o", s=80, zorder=5, label="start")
    ax.scatter(*[pos_smooth[-1, 0], pos_smooth[-1, 2], -pos_smooth[-1, 1]],
               color="tab:red",   marker="^", s=80, zorder=5, label="end")

    ax.legend()
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Gripper CoM Trajectory (smoothed)")
    ax.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_individual_marker_poses(all_raw_poses, timestamps=None):
    """
    Fig — per-marker 6-DoF pose from raw solvePnP (no T matrix).

    3 × 2 grid, all markers overlaid with different colours:
      Left column  — position relative to each marker's first valid frame (mm)
      Right column — orientation relative to each marker's first valid frame (rad)

    If all markers show the same rotation curves, detection + solvePnP is clean
    and any remaining errors are in the T matrices.

    Args:
        all_raw_poses: Dict[marker_id, List[dict | None]]
                       as returned by GripperOdometry.compute_marker_poses_raw().
        timestamps:    np.ndarray (T,) optional; when None uses frame index.
    """
    marker_ids = sorted(all_raw_poses.keys())
    if not marker_ids:
        print("plot_individual_marker_poses: no data.")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(marker_ids), 1)))

    pos_labels = ["pos_x (mm)", "pos_y (mm)", "pos_z (mm)"]
    ori_labels = ["roll (rad)", "pitch (rad)", "yaw (rad)"]

    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    fig.suptitle("Per-Marker Raw solvePnP Pose — Verification (no T matrix)", fontsize=11)
    axes[0, 0].set_title("Translation relative to each marker's t=0 (mm)", fontsize=9)
    axes[0, 1].set_title("Rotation in camera frame relative to t=0  ← should overlap", fontsize=9)

    for color, marker_id in zip(colors, marker_ids):
        poses = all_raw_poses[marker_id]
        valid = [(p["frame_idx"], p) for p in poses if p is not None]
        if not valid:
            continue

        # x-axis
        if timestamps is not None:
            t0 = float(timestamps[valid[0][0]])
            x  = np.array([float(timestamps[fi]) - t0 for fi, _ in valid])
        else:
            x = np.array([fi for fi, _ in valid], dtype=float)

        # position relative to first valid frame
        pos0 = valid[0][1]["position"]
        pos  = np.array([p["position"] - pos0 for _, p in valid])

        # Rotation in camera frame, relative to first valid frame:
        #   R_body = R_i @ R0_i^T
        # This is the same quantity for all markers on a rigid body,
        # so the curves should overlap when detection is correct.
        R0    = valid[0][1]["rotation"]
        quats = np.array([
            Rotation.from_matrix(p["rotation"] @ R0.T).as_quat()
            for _, p in valid
        ])
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i - 1]) < 0:
                quats[i] = -quats[i]
        euler = Rotation.from_quat(quats).as_euler("xyz")

        label = f"M{marker_id}"
        for i in range(3):
            axes[i, 0].plot(x, pos[:, i],   color=color, linewidth=1.2,
                            label=label if i == 0 else "_")
            axes[i, 0].set_ylabel(pos_labels[i], fontsize=8)
            axes[i, 0].grid(True, linewidth=0.4)

            axes[i, 1].plot(x, euler[:, i], color=color, linewidth=1.2,
                            label=label if i == 0 else "_")
            axes[i, 1].set_ylabel(ori_labels[i], fontsize=8)
            axes[i, 1].grid(True, linewidth=0.4)

    xlabel = "time (s)" if timestamps is not None else "frame"
    axes[2, 0].set_xlabel(xlabel, fontsize=9)
    axes[2, 1].set_xlabel(xlabel, fontsize=9)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[0, 1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_fused_orientation(fused_poses, timestamps=None):
    """
    Fig 3 — full 6-DoF pose of gripper frame f0 over time (fused solvePnP).

    3 × 2 grid (all subplots share the x-axis):
      Left column  — translation relative to first valid frame (mm):
        [0,0] pos_x   [1,0] pos_y   [2,0] pos_z
      Right column — orientation relative to first valid frame (rad, Euler XYZ):
        [0,1] roll    [1,1] pitch   [2,1] yaw

    Args:
        fused_poses: List[dict | None]  as returned by
                     GripperOdometry.compute_gripper_poses_fused().
        timestamps:  np.ndarray (T,) optional per-frame timestamps in seconds.
                     When None, frame index is used as the x-axis.
    """
    valid = [(p["frame_idx"], p) for p in fused_poses if p is not None]
    if not valid:
        print("plot_fused_orientation: no valid poses to plot.")
        return

    frame_indices = np.array([fi for fi, _ in valid])

    # --- translation: relative to first valid frame ---
    pos0 = valid[0][1]["position"]
    pos  = np.array([p["position"] - pos0 for _, p in valid])   # (N, 3) mm

    # --- orientation: relative to first valid frame, sign-continuous ---
    R0    = valid[0][1]["rotation"]
    quats = np.array([
        Rotation.from_matrix(R0.T @ p["rotation"]).as_quat() for _, p in valid
    ])
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    euler = Rotation.from_quat(quats).as_euler("xyz")   # (N, 3) rad

    # --- x-axis ---
    if timestamps is not None:
        t0     = float(timestamps[valid[0][0]])
        x      = np.array([float(timestamps[fi]) - t0 for fi, _ in valid])
        xlabel = "time (s)"
    else:
        x      = frame_indices.astype(float)
        xlabel = "frame"

    pos_labels  = ["pos_x (mm)", "pos_y (mm)", "pos_z (mm)"]
    ori_labels  = ["roll (rad)", "pitch (rad)", "yaw (rad)"]
    pos_colors  = ["tab:blue",   "tab:orange",  "tab:green"]
    ori_colors  = ["tab:red",    "tab:purple",  "tab:brown"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    fig.suptitle("Gripper Frame f0 — 6-DoF Pose over Time (fused PnP)", fontsize=12)

    for i in range(3):
        axes[i, 0].plot(x, pos[:, i],   color=pos_colors[i], linewidth=1.4)
        axes[i, 0].set_ylabel(pos_labels[i], fontsize=8)
        axes[i, 0].grid(True, linewidth=0.4)

        axes[i, 1].plot(x, euler[:, i], color=ori_colors[i], linewidth=1.4)
        axes[i, 1].set_ylabel(ori_labels[i], fontsize=8)
        axes[i, 1].grid(True, linewidth=0.4)

    axes[0, 0].set_title("Translation", fontsize=9)
    axes[0, 1].set_title("Orientation", fontsize=9)
    axes[2, 0].set_xlabel(xlabel, fontsize=9)
    axes[2, 1].set_xlabel(xlabel, fontsize=9)

    plt.tight_layout()
    plt.show()
