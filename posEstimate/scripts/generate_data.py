"""
ultimate version to process collected data

Reads /aruco/gripper_pose_four_pose (PoseStamped) directly from a MoCap rosbag
and expresses the trajectory in the initial gripper frame.
"""

import sys
from pathlib import Path
import argparse

# Allow imports from posEstimate/ regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_processing.mocap_process import MocapGripperPoseProcessor

# ==========================================
# --- 1. USER CONFIGURATION ---
# ==========================================

DATA_NAME = "p9-c1-g"
DATA_DIR  = Path(__file__).resolve().parents[1] / "data"
BAG_PATH  = Path(f"~/Downloads/{DATA_NAME}").expanduser()

CROP_START_S = 28  # seconds from bag start, or None to keep from beginning
CROP_END_S   = 33.5  # seconds from bag start, or None to keep to end

SHOW_IMAGE = True
SMOOTH_GRIPPER_POSE = True
SMOOTH_MED_KERNEL = 5   # odd, frames
SMOOTH_SIGMA = 5        # frames


# ==========================================
# --- 2. ENTRY POINT ---
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

    bag = Path(args.bag_path).expanduser()
    MocapGripperPoseProcessor(
        bag_path=bag,
        output_dir=DATA_DIR / bag.stem,
        crop_start_s=CROP_START_S,
        crop_end_s=CROP_END_S,
        show_image=(not args.no_plot),
        smooth=SMOOTH_GRIPPER_POSE,
        med_kernel=SMOOTH_MED_KERNEL,
        sigma=SMOOTH_SIGMA,
    ).run()


if __name__ == "__main__":
    main()
