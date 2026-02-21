import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path


def plot_all_markers_3d(all_odometry):
    """
    Pop up an interactive 3D trajectory plot for all detected markers.

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


def save_trajectory_animation(all_odometry, output_path, rgb_timestamps, dpi=100):
    """
    Save an animated 3D trajectory of all ArUco markers as an MP4.

    Playback speed matches the original video: fps is derived from the rosbag
    RGB timestamps so each animation frame covers the same wall-clock duration
    as the corresponding video frame.

    Args:
        all_odometry:    Dict[int, np.ndarray]  marker_id -> (M, 4) [x, y, z, frame_idx]
        output_path:     Path to save the MP4
        rgb_timestamps:  np.ndarray (T,)  per-frame timestamps in seconds
        dpi:             DPI of the rendered figure
    """
    output_path = Path(output_path)

    all_pts = np.vstack([odo[:, :3] for odo in all_odometry.values() if len(odo) > 0])
    margin = 50  # mm

    all_frame_idxs = np.concatenate(
        [odo[:, 3] for odo in all_odometry.values() if len(odo) > 0]
    ).astype(int)
    min_fi, max_fi = int(all_frame_idxs.min()), int(all_frame_idxs.max())
    frame_range = range(min_fi, max_fi + 1)

    time_span = float(rgb_timestamps[max_fi] - rgb_timestamps[min_fi])
    fps = (max_fi - min_fi + 1) / time_span if time_span > 0 else 30.0
    interval_ms = 1000.0 / fps

    marker_ids = sorted(all_odometry.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(marker_ids), 1)))

    # frame_idx -> (x, y, z) per marker
    marker_lookup = {
        mid: {int(row[3]): row[:3] for row in odo}
        for mid, odo in all_odometry.items() if len(odo) > 0
    }

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)
    ax.set_zlim(-all_pts[:, 1].max() - margin, -all_pts[:, 1].min() + margin)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Trajectory – All ArUco Markers")
    ax.scatter(0, 0, 0, color="black", s=100, marker="*", zorder=5)

    lines, dots = {}, {}
    for color, marker_id in zip(colors, marker_ids):
        (line,) = ax.plot([], [], [], color=color, linewidth=1.5,
                          label=f"Marker {marker_id}")
        (dot,) = ax.plot([], [], [], color=color, marker="o", markersize=6,
                         linestyle="None")
        lines[marker_id] = line
        dots[marker_id] = dot

    ax.legend(loc="upper left", fontsize=8)
    history = {mid: ([], [], []) for mid in marker_ids}
    frame_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=8)

    def update(frame_idx):
        for marker_id in marker_ids:
            pos = marker_lookup.get(marker_id, {}).get(frame_idx)
            if pos is not None:
                hx, hy, hz = history[marker_id]
                hx.append(pos[0])
                hy.append(pos[2])
                hz.append(-pos[1])
            hx, hy, hz = history[marker_id]
            if hx:
                lines[marker_id].set_data(hx, hy)
                lines[marker_id].set_3d_properties(hz)
                dots[marker_id].set_data([hx[-1]], [hy[-1]])
                dots[marker_id].set_3d_properties([hz[-1]])
        frame_text.set_text(f"frame {frame_idx}")
        return list(lines.values()) + list(dots.values()) + [frame_text]

    anim = FuncAnimation(fig, update, frames=frame_range,
                         interval=interval_ms, blit=False)
    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])

    print(f"Saving animation to {output_path} ...")
    anim.save(str(output_path), writer=writer, dpi=dpi)
    print(f"Saved: {output_path}")
    plt.close(fig)
