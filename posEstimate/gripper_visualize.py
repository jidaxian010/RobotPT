import numpy as np
import matplotlib.pyplot as plt


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
