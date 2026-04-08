import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


# -----------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------

def _build_x(valid, timestamps):
    """Return (x array, xlabel string) for the given valid-pose list."""
    if timestamps is not None:
        t0 = float(timestamps[valid[0][0]])
        return np.array([float(timestamps[fi]) - t0 for fi, _ in valid]), "time (s)"
    return np.array([fi for fi, _ in valid], dtype=float), "frame"


def _sign_continuous_euler(quats):
    """Enforce quaternion sign continuity then convert to Euler XYZ."""
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    return Rotation.from_quat(quats).as_euler("xyz")


def _subplot_grid(title, left_title, right_title):
    """Create a 3×2 subplot grid and return (fig, axes)."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(title, fontsize=11)
    axes[0, 0].set_title(left_title,  fontsize=9)
    axes[0, 1].set_title(right_title, fontsize=9)
    return fig, axes


def _fill_grid(axes, x, pos, euler, pos_labels, ori_labels,
               pos_color, ori_color, xlabel, label=None):
    for i in range(3):
        axes[i, 0].plot(x, pos[:, i],   color=pos_color, linewidth=1.3,
                        label=label if (label and i == 0) else "_")
        axes[i, 0].set_ylabel(pos_labels[i], fontsize=8)
        axes[i, 0].grid(True, linewidth=0.4)
        axes[i, 1].plot(x, euler[:, i], color=ori_color, linewidth=1.3,
                        label=label if (label and i == 0) else "_")
        axes[i, 1].set_ylabel(ori_labels[i], fontsize=8)
        axes[i, 1].grid(True, linewidth=0.4)
    axes[2, 0].set_xlabel(xlabel, fontsize=9)
    axes[2, 1].set_xlabel(xlabel, fontsize=9)


_POS_LABELS = ["pos_x (mm)", "pos_y (mm)", "pos_z (mm)"]
_ORI_LABELS = ["roll (rad)", "pitch (rad)", "yaw (rad)"]


# -----------------------------------------------------------------------
# Plot 1 — markers in camera frame
# -----------------------------------------------------------------------

def plot_markers_camera_frame(all_raw_poses, timestamps=None):
    """
    Plot 1 — each marker's 6-DoF pose in the camera frame, relative to its t=0.

    3×2 grid, all markers overlaid:
      Left  — Δposition in camera frame (mm)
      Right — relative rotation in camera frame (rad)
              Formula: R_t @ R0.T  ← same for all markers on a rigid body,
              so curves should overlap when detection is clean.
    """
    marker_ids = sorted(all_raw_poses.keys())
    if not marker_ids:
        print("plot_markers_camera_frame: no data.")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(marker_ids), 1)))
    _, axes = _subplot_grid(
        "Marker Pose in Camera Frame (relative to t=0)",
        "Δposition — camera frame (mm)",
        "Δorientation — camera frame (rad)  ← should overlap",
    )

    for color, marker_id in zip(colors, marker_ids):
        poses = all_raw_poses[marker_id]
        valid = [(p["frame_idx"], p) for p in poses if p is not None]
        if not valid:
            continue

        x, xlabel = _build_x(valid, timestamps)

        pos0 = valid[0][1]["position"]
        R0   = valid[0][1]["rotation"]

        pos   = np.array([p["position"] - pos0 for _, p in valid])
        quats = np.array([Rotation.from_matrix(p["rotation"] @ R0.T).as_quat()
                          for _, p in valid])
        euler = _sign_continuous_euler(quats)

        _fill_grid(axes, x, pos, euler, _POS_LABELS, _ORI_LABELS,
                   color, color, xlabel, label=f"M{marker_id}")

    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.show(block=False)


# -----------------------------------------------------------------------
# Plot 2 — gripper in camera frame  (after applying T matrix)
# -----------------------------------------------------------------------

def plot_gripper_camera_frame(gripper_poses, timestamps=None):
    """
    Plot 2 — gripper CoM pose in the camera frame, relative to t=0.

    Input: List[dict | None] from compute_gripper_poses()[marker_id]
    3×2 grid:
      Left  — Δposition in camera frame (mm)
      Right — relative rotation in camera frame (rad)
              Formula: R_t @ R0.T
    """
    valid = [(p["frame_idx"], p) for p in gripper_poses if p is not None]
    if not valid:
        print("plot_gripper_camera_frame: no valid poses.")
        return

    x, xlabel = _build_x(valid, timestamps)

    pos0  = valid[0][1]["position"]
    R0    = valid[0][1]["rotation"]
    pos   = np.array([p["position"] - pos0 for _, p in valid])
    quats = np.array([Rotation.from_matrix(p["rotation"] @ R0.T).as_quat()
                      for _, p in valid])
    euler = _sign_continuous_euler(quats)

    pos_colors = ["tab:blue",  "tab:orange", "tab:green"]
    ori_colors = ["tab:red",   "tab:purple", "tab:brown"]

    _, axes = _subplot_grid(
        "Gripper Pose in Camera Frame (relative to t=0)",
        "Δposition — camera frame (mm)",
        "Δorientation — camera frame (rad)",
    )
    for i in range(3):
        axes[i, 0].plot(x, pos[:, i],   color=pos_colors[i], linewidth=1.4)
        axes[i, 0].set_ylabel(_POS_LABELS[i], fontsize=8)
        axes[i, 0].grid(True, linewidth=0.4)
        axes[i, 1].plot(x, euler[:, i], color=ori_colors[i], linewidth=1.4)
        axes[i, 1].set_ylabel(_ORI_LABELS[i], fontsize=8)
        axes[i, 1].grid(True, linewidth=0.4)
    axes[2, 0].set_xlabel(xlabel, fontsize=9)
    axes[2, 1].set_xlabel(xlabel, fontsize=9)
    plt.tight_layout()
    plt.show(block=False)


# -----------------------------------------------------------------------
# Plot 3 — gripper in initial gripper frame
# -----------------------------------------------------------------------

def plot_gripper_body_frame(gripper_poses, timestamps=None):
    """
    Plot 3 — gripper CoM motion expressed in the initial gripper frame.

    Input: List[dict | None] from compute_gripper_poses()[marker_id]
    3×2 grid:
      Left  — Δposition in initial gripper frame (mm)
              Formula: R0.T @ (t_t − t_0)
      Right — Δorientation in initial gripper frame (rad)
              Formula: R0.T @ R_t
    """
    valid = [(p["frame_idx"], p) for p in gripper_poses if p is not None]
    if not valid:
        print("plot_gripper_body_frame: no valid poses.")
        return

    x, xlabel = _build_x(valid, timestamps)

    R0   = valid[0][1]["rotation"]
    pos0 = valid[0][1]["position"]
    pos  = np.array([R0.T @ (p["position"] - pos0) for _, p in valid])
    quats = np.array([Rotation.from_matrix(R0.T @ p["rotation"]).as_quat()
                      for _, p in valid])
    euler = _sign_continuous_euler(quats)

    pos_colors = ["tab:blue",  "tab:orange", "tab:green"]
    ori_colors = ["tab:red",   "tab:purple", "tab:brown"]

    _, axes = _subplot_grid(
        "Gripper Motion in Initial Gripper Frame",
        "Δposition — gripper frame (mm)",
        "Δorientation — gripper frame (rad)",
    )
    for i in range(3):
        axes[i, 0].plot(x, pos[:, i],   color=pos_colors[i], linewidth=1.4)
        axes[i, 0].set_ylabel(_POS_LABELS[i], fontsize=8)
        axes[i, 0].grid(True, linewidth=0.4)
        axes[i, 1].plot(x, euler[:, i], color=ori_colors[i], linewidth=1.4)
        axes[i, 1].set_ylabel(_ORI_LABELS[i], fontsize=8)
        axes[i, 1].grid(True, linewidth=0.4)
    axes[2, 0].set_xlabel(xlabel, fontsize=9)
    axes[2, 1].set_xlabel(xlabel, fontsize=9)
    plt.tight_layout()
    plt.show(block=False)
