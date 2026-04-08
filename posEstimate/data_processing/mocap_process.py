"""
MocapGripperPoseProcessor — loads, smooths, saves, and plots
gripper pose trajectories from MoCap rosbags.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as ScipyRotation

from data_processing.rosbag_reader import read_pose_stamped_from_bag
from data_processing.gripper_visualize import (
    plot_gripper_camera_frame,
    plot_gripper_body_frame,
)
from data_processing.denoise import denoise_pose_list

GRIPPER_POSE_TOPIC = "/aruco/gripper_pose_four_pose"

SMOOTH_GRIPPER_POSE = True
SMOOTH_MED_KERNEL = 5
SMOOTH_SIGMA = 5


class MocapGripperPoseProcessor:
    def __init__(
        self,
        bag_path,
        output_dir,
        crop_start_s=None,
        crop_end_s=None,
        show_image=True,
        smooth=SMOOTH_GRIPPER_POSE,
        med_kernel=SMOOTH_MED_KERNEL,
        sigma=SMOOTH_SIGMA,
    ):
        self.bag_path = Path(bag_path)
        self.output_dir = Path(output_dir)
        self.crop_start_s = crop_start_s
        self.crop_end_s = crop_end_s
        self.show_image = bool(show_image)
        self.smooth = smooth
        self.med_kernel = med_kernel
        self.sigma = sigma

        self.gripper_poses_raw = []
        self.gripper_poses = []
        self.timestamps = []

    def load(self):
        self.gripper_poses_raw, self.timestamps = read_pose_stamped_from_bag(
            self.bag_path,
            topic=GRIPPER_POSE_TOPIC,
            crop_start_s=self.crop_start_s,
            crop_end_s=self.crop_end_s,
        )

    def smooth_poses(self):
        if self.smooth:
            self.gripper_poses = denoise_pose_list(
                self.gripper_poses_raw,
                med_kernel=self.med_kernel,
                sigma=self.sigma,
            )
            print(
                f"Applied smoothing: med_kernel={self.med_kernel}, sigma={self.sigma}"
            )
        else:
            self.gripper_poses = list(self.gripper_poses_raw)

    def save_poses_csv(self):
        """
        Save gripper CoM trajectory relative to the first valid frame.

        Poses are expressed in the initial gripper frame:
            pos   = R0.T @ (t_cam_t - t_cam_0)   [mm], first row = 0,0,0
            euler = Euler XYZ of R0.T @ R_cam_t   [rad], first row = 0,0,0

        The CSV is directly readable by pink/examples/arm_optimo.py.

        Returns the Path of the written file, or None on failure.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        valid = [(i, p) for i, p in enumerate(self.gripper_poses) if p is not None]
        if not valid:
            print("  No valid gripper poses to save.")
            return None

        _, first = valid[0]
        R0  = first["rotation"]
        t0  = first["position"]
        ts0 = float(self.timestamps[first["frame_idx"]])

        csv_path = self.output_dir / f"{self.bag_path.stem}.csv"
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
        self.smooth_poses()
        csv_path = self.save_poses_csv()
        if csv_path is not None:
            arm_script = (
                self.bag_path.resolve().parents[1] / "pink/examples/arm_optimo.py"
            )
            print(f"\nTo replay on the robot arm, run:")
            print(f"  python {arm_script} {csv_path}")
        self.plot_results()
