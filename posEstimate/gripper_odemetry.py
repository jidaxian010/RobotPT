import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from rosbag_reader import RosbagVideoReader
from object_detect.select_marker import SelectMarker
from odometry import convert_pixel_array_to_depth_format


# Camera intrinsics (848x480)
FX = 602.6597900390625
FY = 602.2169799804688
CX = 423.1910400390625
CY = 249.92578125


def pixel_depth_to_3d(pixel_depth_array):
    """
    Convert pixel depth array to 3D positions with depth smoothing.

    Args:
        pixel_depth_array: shape (N, 4) with columns [u, v, depth_mm, frame_idx]

    Returns:
        odometry_array: shape (M, 4) with columns [x, y, z, frame_idx]
                        where z = depth, and x/y are lateral offsets in mm.
    """
    from scipy.signal import medfilt

    depths = pixel_depth_array[:, 2].copy()

    # Clamp jumps larger than 30% of the previous value
    max_change_percent = 30
    for i in range(1, len(depths)):
        if depths[i] > 0 and depths[i - 1] > 0:
            pct = abs(depths[i] - depths[i - 1]) / depths[i - 1] * 100
            if pct > max_change_percent:
                depths[i] = depths[i - 1]

    smoothed_depths = medfilt(depths, kernel_size=7)

    odometry_array = []
    for i in range(len(pixel_depth_array)):
        u = pixel_depth_array[i, 0]
        v = pixel_depth_array[i, 1]
        d = smoothed_depths[i]
        frame_idx = pixel_depth_array[i, 3]

        if d == 0:
            continue

        x = (u - CX) * d / FX
        y = (v - CY) * d / FY
        odometry_array.append([x, y, d, frame_idx])

    return np.array(odometry_array) if odometry_array else np.empty((0, 4))


def plot_all_markers_3d(all_odometry):
    """
    Plot 3D trajectories for all detected markers on a single figure.

    Args:
        all_odometry: Dict[int, np.ndarray] mapping marker_id to (M, 4) array
                      with columns [x, y, z, frame_idx]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    marker_ids = sorted(all_odometry.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(marker_ids), 1)))

    for color, marker_id in zip(colors, marker_ids):
        odometry_array = all_odometry[marker_id]
        if len(odometry_array) == 0:
            continue
        posx = odometry_array[:, 0]
        posy = odometry_array[:, 2]   # depth as Y axis
        posz = -odometry_array[:, 1]  # flip image-v to world-Z
        ax.plot(posx, posy, posz, color=color, label=f"Marker {marker_id}")

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
    RGB timestamps so each animation frame covers exactly the same wall-clock
    duration as the corresponding video frame.

    Args:
        all_odometry:    Dict[int, np.ndarray] mapping marker_id to (M, 4) array
                         with columns [x, y, z, frame_idx]
        output_path:     Path to save the MP4
        rgb_timestamps:  np.ndarray shape (T,) of per-frame timestamps in
                         seconds, as returned by RosbagVideoReader.get_rgb_timestamps()
        dpi:             DPI of the rendered figure
    """
    output_path = Path(output_path)

    # Collect all 3D points to compute axis limits
    all_pts = np.vstack([
        odo[:, :3] for odo in all_odometry.values() if len(odo) > 0
    ])
    margin = 50  # mm

    # Global frame index range across all markers
    all_frame_idxs = np.concatenate([
        odo[:, 3] for odo in all_odometry.values() if len(odo) > 0
    ]).astype(int)
    min_fi, max_fi = int(all_frame_idxs.min()), int(all_frame_idxs.max())
    frame_range = range(min_fi, max_fi + 1)

    # Derive fps from actual rosbag timestamps so playback matches the video
    n_frames = max_fi - min_fi + 1
    time_span = float(rgb_timestamps[max_fi] - rgb_timestamps[min_fi])
    fps = n_frames / time_span if time_span > 0 else 30.0
    interval_ms = 1000.0 / fps

    # Build per-marker lookup: frame_idx -> (x, y, z)
    marker_ids = sorted(all_odometry.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(marker_ids), 1)))

    marker_lookup = {}
    for marker_id, odo in all_odometry.items():
        if len(odo) == 0:
            continue
        marker_lookup[marker_id] = {int(row[3]): row[:3] for row in odo}

    # Set up figure
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

    # One line + one dot per marker
    lines, dots = {}, {}
    for color, marker_id in zip(colors, marker_ids):
        (line,) = ax.plot([], [], [], color=color, linewidth=1.5,
                          label=f"Marker {marker_id}")
        (dot,) = ax.plot([], [], [], color=color, marker="o", markersize=6,
                         linestyle="None")
        lines[marker_id] = line
        dots[marker_id] = dot

    ax.legend(loc="upper left", fontsize=8)

    # Cumulative history per marker (world coords)
    history = {mid: ([], [], []) for mid in marker_ids}

    frame_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=8)

    def update(frame_idx):
        for marker_id in marker_ids:
            if marker_id not in marker_lookup:
                continue
            pos = marker_lookup[marker_id].get(frame_idx)
            if pos is not None:
                hx, hy, hz = history[marker_id]
                hx.append(pos[0])
                hy.append(pos[2])   # depth as Y
                hz.append(-pos[1])  # flip image-v to world-Z

            hx, hy, hz = history[marker_id]
            if hx:
                lines[marker_id].set_data(hx, hy)
                lines[marker_id].set_3d_properties(hz)
                dots[marker_id].set_data([hx[-1]], [hy[-1]])
                dots[marker_id].set_3d_properties([hz[-1]])

        frame_text.set_text(f"frame {frame_idx}")
        return list(lines.values()) + list(dots.values()) + [frame_text]

    anim = FuncAnimation(
        fig, update, frames=frame_range,
        interval=interval_ms, blit=False,
    )

    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])

    print(f"Saving animation to {output_path} ...")
    anim.save(str(output_path), writer=writer, dpi=dpi)
    print(f"Saved: {output_path}")
    plt.close(fig)


def find_gripper_odometry(bagpath, video_path, dict_name="DICT_4X4_50"):
    """
    Detect all ArUco markers in the video and compute 3D odometry for each
    marker's centroid (mean of its 4 corners).

    Args:
        bagpath:    Path to rosbag directory / file
        video_path: Path to output video file (MP4)
        dict_name:  ArUco dictionary name (default: "DICT_4X4_50")

    Returns:
        Tuple of four items:
            all_pixel_arrays       – Dict[marker_id -> (1, 2, T)]  centroid (u, v) per frame
            all_pixel_depth_arrays – Dict[marker_id -> (M, 4)]     [u, v, depth_mm, frame_idx]
            all_odometry_arrays    – Dict[marker_id -> (M, 4)]     [x, y, z, frame_idx]
            rgb_timestamps         – np.ndarray shape (T,)          per-frame timestamps (seconds)
    """
    # Extract RGB and depth video from rosbag
    video_reader = RosbagVideoReader(
        Path(bagpath), Path(video_path),
        is_third_person=True, skip_first_n=0, skip_last_n=0,
    )
    video_reader.process_data()
    video_reader.save_depth_video()

    # Detect all ArUco markers across every frame
    video_path_obj = Path(video_path)
    output_path = video_path_obj.parent / f"{video_path_obj.stem}_all_markers.mp4"

    marker_tracker = SelectMarker(
        input_path=video_path,
        output_path=str(output_path),
        dict_name=dict_name,
    )

    # all_marker_trajectories: Dict[marker_id -> (4, 2, T)]
    all_marker_trajectories = marker_tracker.run()

    rgb_timestamps = video_reader.get_rgb_timestamps()

    if not all_marker_trajectories:
        print("No ArUco markers detected in video.")
        return {}, {}, {}, rgb_timestamps

    print(f"Detected {len(all_marker_trajectories)} marker(s): {sorted(all_marker_trajectories.keys())}")

    all_pixel_arrays = {}
    all_pixel_depth_arrays = {}
    all_odometry_arrays = {}

    for marker_id, corners_array in all_marker_trajectories.items():
        # corners_array: (4, 2, T) → centroid over 4 corners → (1, 2, T)
        centroid_array = np.nanmean(corners_array, axis=0, keepdims=True)
        all_pixel_arrays[marker_id] = centroid_array

        # Convert to (N, 3) [u, v, frame_idx] for depth lookup
        pixel_array_for_depth = convert_pixel_array_to_depth_format(centroid_array)

        if len(pixel_array_for_depth) == 0:
            print(f"  Marker {marker_id}: no valid pixel data, skipping.")
            continue

        # Look up depth → (N, 4) [u, v, depth_mm, frame_idx]
        pixel_depth_array = video_reader.find_depth(pixel_array_for_depth)
        all_pixel_depth_arrays[marker_id] = pixel_depth_array

        # Project to 3D with smoothing
        odometry_array = pixel_depth_to_3d(pixel_depth_array)
        all_odometry_arrays[marker_id] = odometry_array

        print(f"  Marker {marker_id}: {len(odometry_array)} valid 3D points")

    return all_pixel_arrays, all_pixel_depth_arrays, all_odometry_arrays, rgb_timestamps


def main():
    bagpath = "/home/jdx/Downloads/gripper1"
    video_path = "posEstimate/data/gripper1.mp4"

    mp4_path = Path(video_path).parent / f"{Path(video_path).stem}_trajectory.mp4"

    print("=== Gripper ArUco Marker Odometry ===")
    try:
        all_pixel_arrays, all_pixel_depth_arrays, all_odometry_arrays, rgb_timestamps = \
            find_gripper_odometry(bagpath, video_path)

        if all_odometry_arrays:
            save_trajectory_animation(all_odometry_arrays, mp4_path, rgb_timestamps)
        else:
            print("No odometry data to plot.")
    except Exception as e:
        print(f"Gripper odometry failed: {e}")
        raise


if __name__ == "__main__":
    main()
