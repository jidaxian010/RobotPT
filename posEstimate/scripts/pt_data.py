"""
ultimate version to process collected data

Reads /aruco/gripper_pose_four_pose (PoseStamped) directly from a MoCap rosbag
and expresses the trajectory in the initial gripper frame.
"""

import csv
import sys
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as ScipyRotation

# Allow imports from posEstimate/ regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosbag_reader import AnyReader, typestore
from gripper_visualize import (
    plot_gripper_camera_frame,
    plot_gripper_body_frame,
)
from denoise import denoise_pose_list

# ==========================================
# --- 1. USER CONFIGURATION ---
# ==========================================

DATA_NAME = "jeff"
DATA_DIR  = Path(__file__).resolve().parents[1] / "data"
BAG_PATH  = DATA_DIR / DATA_NAME

GRIPPER_POSE_TOPIC = "/aruco/gripper_pose_four_pose"  # PoseStamped, in camera frame, mm + rot matrix

CROP_START_S = 11.9  # seconds from bag start, or None to keep from beginning
CROP_END_S   = 17.6  # seconds from bag start, or None to keep to end

SHOW_IMAGE = True
SMOOTH_GRIPPER_POSE = True
SMOOTH_MED_KERNEL = 5   # odd, frames
SMOOTH_SIGMA = 5        # frames


# ==========================================
# --- 2. ROSBAG READER ---
# ==========================================

def _get_stamp_sec(msg, fallback_ts_ns):
    try:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    except Exception:
        return fallback_ts_ns * 1e-9


def read_gripper_poses_from_bag(bag_path, crop_start_s=None, crop_end_s=None):
    """
    Read /aruco/gripper_pose_four_pose (PoseStamped) from rosbag.

    Args:
        crop_start_s : keep poses with t >= t_bag_start + crop_start_s (None = no trim)
        crop_end_s   : keep poses with t <= t_bag_start + crop_end_s   (None = no trim)

    Returns:
        poses      : List[dict]  — each dict has
                       "position"  : np.ndarray (3,) in mm  (camera frame)
                       "rotation"  : np.ndarray (3,3)        (camera frame)
                       "frame_idx" : int
        timestamps : List[float] — ROS stamp in seconds
    """
    bag_path = Path(bag_path)
    poses = []
    timestamps = []

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conn = None
        for c in reader.connections:
            if c.topic == GRIPPER_POSE_TOPIC:
                conn = c
                break
        if conn is None:
            raise RuntimeError(
                f"Topic {GRIPPER_POSE_TOPIC!r} not found in {bag_path}.\n"
                f"Available topics: {[c.topic for c in reader.connections]}"
            )

        frame_idx = 0
        t_bag_start = None

        for _, ts, raw in reader.messages(connections=[conn]):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            t = _get_stamp_sec(msg, ts)

            if t_bag_start is None:
                t_bag_start = t

            t_rel = t - t_bag_start
            if crop_start_s is not None and t_rel < crop_start_s:
                continue
            if crop_end_s is not None and t_rel > crop_end_s:
                continue

            try:
                p = msg.pose.position
                q = msg.pose.orientation
                position_m = np.array([p.x, p.y, p.z], dtype=np.float64)
                quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
            except AttributeError:
                # flat Pose (no header) — try direct access
                p = msg.position
                q = msg.orientation
                position_m = np.array([p.x, p.y, p.z], dtype=np.float64)
                quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

            rotation = ScipyRotation.from_quat(quat).as_matrix()

            poses.append({
                "position": position_m * 1000.0,   # metres -> mm
                "rotation": rotation,
                "frame_idx": frame_idx,
            })
            timestamps.append(t)
            frame_idx += 1

    if not poses:
        raise RuntimeError(f"No messages read from {GRIPPER_POSE_TOPIC!r}")

    crop_info = (
        f"crop=[{crop_start_s}s, {crop_end_s}s]" if (crop_start_s or crop_end_s) else "no crop"
    )
    print(f"Read {len(poses)} gripper poses from {GRIPPER_POSE_TOPIC!r} ({crop_info})")
    return poses, timestamps


# ==========================================
# --- 3. PROCESSOR CLASS ---
# ==========================================

class MocapGripperPoseProcessor:
    def __init__(self, bag_path=BAG_PATH, show_image=SHOW_IMAGE):
        self.bag_path = Path(bag_path)
        self.show_image = bool(show_image)
        self.gripper_poses_raw = []
        self.gripper_poses = []
        self.timestamps = []

    def load(self):
        self.gripper_poses_raw, self.timestamps = read_gripper_poses_from_bag(
            self.bag_path,
            crop_start_s=CROP_START_S,
            crop_end_s=CROP_END_S,
        )

    def smooth(self):
        if SMOOTH_GRIPPER_POSE:
            self.gripper_poses = denoise_pose_list(
                self.gripper_poses_raw,
                med_kernel=SMOOTH_MED_KERNEL,
                sigma=SMOOTH_SIGMA,
            )
            print(
                f"Applied smoothing: med_kernel={SMOOTH_MED_KERNEL}, sigma={SMOOTH_SIGMA}"
            )
        else:
            self.gripper_poses = list(self.gripper_poses_raw)

    def save_poses_csv(self, output_dir=None):
        """
        Save gripper CoM trajectory relative to the first valid frame.

        Poses are expressed in the initial gripper frame:
            pos   = R0.T @ (t_cam_t - t_cam_0)   [mm], first row = 0,0,0
            euler = Euler XYZ of R0.T @ R_cam_t   [rad], first row = 0,0,0

        The CSV is directly readable by pink/examples/arm_optimo.py.

        Returns the Path of the written file, or None on failure.
        """
        if output_dir is None:
            output_dir = DATA_DIR / self.bag_path.stem
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        valid = [(i, p) for i, p in enumerate(self.gripper_poses) if p is not None]
        if not valid:
            print("  No valid gripper poses to save.")
            return None

        _, first = valid[0]
        R0  = first["rotation"]
        t0  = first["position"]
        ts0 = float(self.timestamps[first["frame_idx"]])

        csv_path = output_dir / f"{self.bag_path.stem}.csv"
        first_valid_idx = valid[0][0]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t", "frame",
                "pos_x", "pos_y", "pos_z",
                "orient_x", "orient_y", "orient_z",
            ])
            for idx, pose in valid:
                ts    = float(self.timestamps[pose["frame_idx"]]) - ts0
                pos   = R0.T @ (pose["position"] - t0)
                R_rel = R0.T @ pose["rotation"]
                euler = ScipyRotation.from_matrix(R_rel).as_euler("xyz")

                if idx == first_valid_idx:
                    ts    = 0.0
                    pos   = np.zeros(3, dtype=np.float64)
                    euler = np.zeros(3, dtype=np.float64)

                writer.writerow([
                    round(ts, 6),
                    pose["frame_idx"],
                    round(pos[0], 3), round(pos[1], 3), round(pos[2], 3),
                    round(euler[0], 6), round(euler[1], 6), round(euler[2], 6),
                ])

        print(f"  Saved gripper 6D motion in initial gripper frame: {csv_path}")
        return csv_path

    def plot_results(self):
        if not self.show_image:
            print("Skipping plots (SHOW_IMAGE=False).")
            return

        if any(p is not None for p in self.gripper_poses):
            plot_gripper_camera_frame(self.gripper_poses, self.timestamps)
            plot_gripper_body_frame(self.gripper_poses, self.timestamps)
        else:
            print("No gripper poses available for plotting.")

        plt.show()

    def run(self):
        self.load()
        self.smooth()
        csv_path = self.save_poses_csv()
        if csv_path is not None:
            arm_script = (
                Path(__file__).resolve().parents[2] / "pink/examples/arm_optimo.py"
            )
            print(f"\nTo replay on the robot arm, run:")
            print(f"  python {arm_script} {csv_path}")
        self.plot_results()


# ==========================================
# --- 4. ENTRY POINT ---
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Process MoCap rosbag and extract gripper pose trajectory."
    )
    parser.add_argument(
        "bag_path",
        nargs="?",
        default=str(BAG_PATH),
        help=f"Path to the rosbag (default: {BAG_PATH})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plots.",
    )
    args = parser.parse_args()

    MocapGripperPoseProcessor(
        bag_path=Path(args.bag_path).expanduser(),
        show_image=(not args.no_plot),
    ).run()


if __name__ == "__main__":
    main()
