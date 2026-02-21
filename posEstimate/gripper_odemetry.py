import numpy as np
from pathlib import Path
from scipy.signal import medfilt

from rosbag_reader import RosbagVideoReader
from object_detect.select_marker import SelectMarker
from odometry import convert_pixel_array_to_depth_format
from gripper_visualize import plot_all_markers_3d, save_trajectory_animation


# Camera intrinsics (848x480)
_FX = 602.6597900390625
_FY = 602.2169799804688
_CX = 423.1910400390625
_CY = 249.92578125


def _pixel_depth_to_3d(pixel_depth_array):
    """
    Convert pixel+depth array to 3D positions with depth smoothing.

    Args:
        pixel_depth_array: (N, 4)  [u, v, depth_mm, frame_idx]

    Returns:
        (M, 4)  [x, y, z, frame_idx]  in mm, camera frame
    """
    depths = pixel_depth_array[:, 2].copy()

    # Clamp jumps larger than 30% of the previous value
    for i in range(1, len(depths)):
        if depths[i] > 0 and depths[i - 1] > 0:
            if abs(depths[i] - depths[i - 1]) / depths[i - 1] > 0.30:
                depths[i] = depths[i - 1]

    smoothed = medfilt(depths, kernel_size=7)

    result = []
    for i in range(len(pixel_depth_array)):
        u, v, _, frame_idx = pixel_depth_array[i]
        d = smoothed[i]
        if d == 0:
            continue
        x = (u - _CX) * d / _FX
        y = (v - _CY) * d / _FY
        result.append([x, y, d, frame_idx])

    return np.array(result) if result else np.empty((0, 4))


class GripperOdometry:
    """
    Full pipeline: extract RGB+depth video from a rosbag, detect all ArUco
    markers, and compute 3D odometry for each marker's centroid.

    Usage:
        g = GripperOdometry(bagpath, video_path)
        g.run()
        g.plot()
        g.save_animation()

    Results after run():
        g.all_odometry    – Dict[marker_id -> (M, 4)  [x, y, z, frame_idx]]
        g.all_pixel_depth – Dict[marker_id -> (N, 4)  [u, v, depth_mm, frame_idx]]
        g.rgb_timestamps  – np.ndarray (T,)  per-frame timestamps in seconds
    """

    def __init__(self, bagpath, video_path, dict_name="DICT_4X4_50"):
        self.bagpath = Path(bagpath)
        self.video_path = Path(video_path)
        self.dict_name = dict_name

        self.all_odometry: dict = {}
        self.all_pixel_depth: dict = {}
        self.rgb_timestamps: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Run the full pipeline. Returns self for optional chaining."""
        video_reader = self._build_video_reader()
        video_reader.process_data()
        video_reader.save_depth_video()

        all_marker_trajectories = self._detect_markers()
        self.rgb_timestamps = video_reader.get_rgb_timestamps()

        if not all_marker_trajectories:
            print("No ArUco markers detected.")
            return self

        print(f"Detected {len(all_marker_trajectories)} marker(s): "
              f"{sorted(all_marker_trajectories.keys())}")

        for marker_id, corners in all_marker_trajectories.items():
            self._process_marker(marker_id, corners, video_reader)

        return self

    def plot(self):
        """Pop up an interactive 3D trajectory window."""
        plot_all_markers_3d(self.all_odometry)

    def save_animation(self, output_path=None):
        """
        Save the animated trajectory as MP4.
        Defaults to <video_stem>_trajectory.mp4 next to the video file.
        """
        if output_path is None:
            output_path = (self.video_path.parent /
                           f"{self.video_path.stem}_trajectory.mp4")
        save_trajectory_animation(self.all_odometry, output_path,
                                  self.rgb_timestamps)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_video_reader(self):
        return RosbagVideoReader(
            self.bagpath, self.video_path,
            is_third_person=True, skip_first_n=0, skip_last_n=0,
        )

    def _detect_markers(self):
        annotated_path = (self.video_path.parent /
                          f"{self.video_path.stem}_all_markers.mp4")
        tracker = SelectMarker(
            input_path=str(self.video_path),
            output_path=str(annotated_path),
            dict_name=self.dict_name,
        )
        return tracker.run()  # Dict[marker_id -> (4, 2, T)]

    def _process_marker(self, marker_id, corners, video_reader):
        """Compute centroid pixel trajectory → depth lookup → 3D odometry."""
        centroid = np.nanmean(corners, axis=0, keepdims=True)  # (1, 2, T)
        pixel_for_depth = convert_pixel_array_to_depth_format(centroid)

        if len(pixel_for_depth) == 0:
            print(f"  Marker {marker_id}: no valid pixels, skipping.")
            return

        pixel_depth = video_reader.find_depth(pixel_for_depth)  # (N, 4)
        self.all_pixel_depth[marker_id] = pixel_depth

        odometry = _pixel_depth_to_3d(pixel_depth)             # (M, 4)
        self.all_odometry[marker_id] = odometry

        print(f"  Marker {marker_id}: {len(odometry)} valid 3D points")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    g = GripperOdometry(
        bagpath="/home/jdx/Downloads/gripper2",
        video_path="posEstimate/data/gripper2.mp4",
    )
    g.run()

    if g.all_odometry:
        g.plot()
        g.save_animation()
    else:
        print("No odometry data.")


if __name__ == "__main__":
    main()
