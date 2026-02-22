import csv
import sys
import cv2
import numpy as np
from pathlib import Path
from scipy.signal import medfilt
from scipy.spatial.transform import Rotation as ScipyRotation

# Allow imports from posEstimate/ regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosbag_reader import RosbagVideoReader
from object_detect.select_marker import SelectMarker
from odometry import convert_pixel_array_to_depth_format
from gripper_config import MARKER_CONFIGS
from gripper_visualize import plot_all_markers_3d, plot_com_smooth_3d


# Camera intrinsics (848x480)
_FX = 602.6597900390625
_FY = 602.2169799804688
_CX = 423.1910400390625
_CY = 249.92578125
_K  = np.array([[_FX, 0, _CX], [0, _FY, _CY], [0, 0, 1]], dtype=np.float32)
_DIST = np.zeros((4, 1), dtype=np.float32)   # assume undistorted; update if needed


def _pixel_depth_to_3d(pixel_depth_array):
    """
    Convert pixel+depth array to 3D positions with depth smoothing.

    Args:
        pixel_depth_array: (N, 4)  [u, v, depth_mm, frame_idx]

    Returns:
        (M, 4)  [x, y, z, frame_idx]  in mm, camera frame
    """
    depths = pixel_depth_array[:, 2].copy()

    # Clamp jumps larger than 30 % of the previous value
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


def _solvepnp_corners(obj_pts, corners_4_2_T):
    """
    Run solvePnP on every frame that has valid (non-NaN) corners.

    Args:
        obj_pts:       (4, 3) float32  3-D corner coords in marker-local frame
        corners_4_2_T: (4, 2, T)       pixel corners per frame (NaN when missing)

    Returns:
        List of length T, each entry either
            {"rvec": (3,1), "tvec": (3,1), "R": (3,3)}  or  None
    """
    T = corners_4_2_T.shape[2]
    poses = []
    for t in range(T):
        img_pts = corners_4_2_T[:, :, t].astype(np.float32)
        if np.any(np.isnan(img_pts)):
            poses.append(None)
            continue
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, _K, _DIST,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            poses.append(None)
            continue
        R, _ = cv2.Rodrigues(rvec)
        poses.append({"rvec": rvec, "tvec": tvec, "R": R})
    return poses


class GripperOdometry:
    """
    Full pipeline: extract RGB+depth video from a rosbag, detect all ArUco
    markers, compute 3D odometry for each marker's centroid, and (optionally)
    recover the full 6-DoF pose of the gripper CoM via per-marker SE(3) offsets.

    Usage
    -----
        g = GripperOdometry(bagpath, video_path)
        g.run()

        # Position-only trajectory (from depth)
        g.plot()
        g.save_animation()

        # Full 6-DoF gripper pose (position + orientation)
        gripper_poses = g.compute_gripper_poses()
        # gripper_poses[6][t] -> {"position": (3,) mm, "rotation": (3,3), "frame_idx": int}

    Attributes populated after run()
    ---------------------------------
        all_odometry           Dict[marker_id -> (M, 4)  [x, y, z, frame_idx]]
        all_pixel_depth        Dict[marker_id -> (N, 4)  [u, v, depth_mm, frame_idx]]
        all_marker_trajectories Dict[marker_id -> (4, 2, T)]  raw corner pixels
        rgb_timestamps         np.ndarray (T,)  per-frame timestamps in seconds
    """

    def __init__(self, bagpath, video_path, dict_name="DICT_4X4_50"):
        self.bagpath    = Path(bagpath)
        self.video_path = Path(video_path)
        self.dict_name  = dict_name

        self.all_odometry:            dict = {}
        self.all_pixel_depth:         dict = {}
        self.all_marker_trajectories: dict = {}   # raw corners (4, 2, T)
        self.rgb_timestamps: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Run the full pipeline. Returns self for optional chaining."""
        video_reader = self._build_video_reader()
        video_reader.process_data()
        video_reader.save_depth_video()

        self.all_marker_trajectories = self._detect_markers()
        self.rgb_timestamps = video_reader.get_rgb_timestamps()

        if not self.all_marker_trajectories:
            print("No ArUco markers detected.")
            return self

        print(f"Detected {len(self.all_marker_trajectories)} marker(s): "
              f"{sorted(self.all_marker_trajectories.keys())}")

        for marker_id, corners in self.all_marker_trajectories.items():
            self._process_marker(marker_id, corners, video_reader)

        return self

    def compute_gripper_poses(self, marker_configs=None):
        """
        Recover the full 6-DoF pose of the gripper CoM from detected markers.

        For each marker that has a MarkerConfig, solvePnP is run on every frame
        to get the marker's SE(3) pose in the camera frame.  The fixed
        marker→CoM transform is then applied to express the gripper CoM pose
        in the camera frame.

        Args:
            marker_configs: Dict[int, MarkerConfig].  Defaults to MARKER_CONFIGS
                            from gripper_config.py.

        Returns:
            Dict[marker_id, list[dict | None]]
            Each per-frame entry (when detection succeeded):
                {
                    "position": np.ndarray (3,)   # CoM in camera frame (mm)
                    "rotation": np.ndarray (3, 3) # CoM orientation in camera frame
                    "frame_idx": int
                }
            None for frames where the marker was not detected.
        """
        if marker_configs is None:
            marker_configs = MARKER_CONFIGS

        all_gripper_poses = {}

        for marker_id, corners_4_2_T in self.all_marker_trajectories.items():
            cfg = marker_configs.get(marker_id)
            if cfg is None:
                continue  # no config for this marker yet

            # 3-D corner coords in marker-local frame (Z = 0 plane)
            h = cfg.marker_size_mm / 2
            obj_pts = np.array(
                [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                dtype=np.float32,
            )

            marker_poses = _solvepnp_corners(obj_pts, corners_4_2_T)

            gripper_poses = []
            for frame_idx, mp in enumerate(marker_poses):
                if mp is None:
                    gripper_poses.append(None)
                    continue

                # T_cam←marker  (from solvePnP)
                R_cm = mp["R"]
                t_cm = mp["tvec"].flatten()

                # Apply T_marker←com  (from MarkerConfig)
                # R_cam←com = R_cm @ R_mc
                # t_cam←com = R_cm @ t_mc + t_cm
                R_cam_com = R_cm @ cfg.R_marker_to_com
                t_cam_com = R_cm @ cfg.t_marker_to_com + t_cm

                gripper_poses.append({
                    "position":  t_cam_com,   # (3,)  mm, camera frame
                    "rotation":  R_cam_com,   # (3, 3)
                    "frame_idx": frame_idx,
                })

            all_gripper_poses[marker_id] = gripper_poses
            valid = sum(1 for p in gripper_poses if p is not None)
            print(f"  Marker {marker_id}: {valid} gripper poses computed")

        return all_gripper_poses

    def save_poses_csv(self, gripper_poses, output_dir=None):
        """
        Save the gripper CoM trajectory relative to the first valid frame.

        A single file: <video_stem>_com_trajectory.csv
        Uses the first marker that has valid poses (currently marker 6).
        When multiple markers are calibrated, this can be extended to fuse them.

        Columns
        -------
        t            elapsed seconds since the first valid frame
        frame        video frame index
        pos_x/y/z    position relative to first valid pose (mm); first row = 0,0,0
        orient_x/y/z Euler XYZ angles relative to first valid pose (rad);
                     first row = 0,0,0
        """
        if output_dir is None:
            output_dir = self.video_path.parent
        output_dir = Path(output_dir)

        # Use the first marker that has valid data
        poses = None
        for marker_id, p in gripper_poses.items():
            valid = [(i, e) for i, e in enumerate(p) if e is not None]
            if valid:
                poses = valid
                source_marker = marker_id
                break

        if not poses:
            print("  No valid gripper poses to save.")
            return

        _, first = poses[0]
        R0  = first["rotation"]
        t0  = first["position"]
        ts0 = float(self.rgb_timestamps[first["frame_idx"]])

        csv_path = output_dir / f"{self.video_path.stem}_com_trajectory.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "frame",
                              "pos_x", "pos_y", "pos_z",
                              "orient_x", "orient_y", "orient_z"])
            for _, pose in poses:
                ts    = float(self.rgb_timestamps[pose["frame_idx"]]) - ts0
                pos   = pose["position"] - t0
                R_rel = R0.T @ pose["rotation"]
                euler = ScipyRotation.from_matrix(R_rel).as_euler("xyz")
                writer.writerow([
                    round(ts, 6),
                    pose["frame_idx"],
                    round(pos[0], 3), round(pos[1], 3), round(pos[2], 3),
                    round(euler[0], 6), round(euler[1], 6), round(euler[2], 6),
                ])

        print(f"  Saved CoM trajectory (from marker {source_marker}): {csv_path}")
        return csv_path

    def plot(self):
        """Pop up fig1: marker centroid trajectories."""
        plot_all_markers_3d(self.all_odometry)

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
        """Centroid pixel trajectory → depth lookup → 3D odometry."""
        centroid = np.nanmean(corners, axis=0, keepdims=True)  # (1, 2, T)
        pixel_for_depth = convert_pixel_array_to_depth_format(centroid)

        if len(pixel_for_depth) == 0:
            print(f"  Marker {marker_id}: no valid pixels, skipping.")
            return

        pixel_depth = video_reader.find_depth(pixel_for_depth)   # (N, 4)
        self.all_pixel_depth[marker_id] = pixel_depth

        odometry = _pixel_depth_to_3d(pixel_depth)               # (M, 4)
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

    if not g.all_odometry:
        print("No odometry data.")
        return

    g.plot()  # fig1: marker centroid trajectories

    gripper_poses = g.compute_gripper_poses()
    if gripper_poses:
        csv_path = g.save_poses_csv(gripper_poses)
        if csv_path is not None:
            from traj_smooth import smooth_trajectory
            _, _, _, _, pos_smooth, _ = smooth_trajectory(str(csv_path))
            plot_com_smooth_3d(pos_smooth)  # fig2: smoothed CoM trajectory


if __name__ == "__main__":
    main()
