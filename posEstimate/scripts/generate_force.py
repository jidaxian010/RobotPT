"""
Process a MoCap rosbag: calibrate and plot the full 6-DOF wrench (Fx/Fy/Fz + Tx/Ty/Tz).
"""

import sys
from pathlib import Path
import argparse

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_processing.force_process import ForceProcessor
from data_processing.wrench_calibration import WrenchCalibrator

# ==========================================
# --- 1. USER CONFIGURATION ---
# ==========================================

DATA_NAME      = "p8-c2-w"
BAG_PATH       = Path(f"~/Downloads/{DATA_NAME}").expanduser()
DATA_DIR       = Path(__file__).resolve().parents[1] / "data"
CALIB_BAG_PATH = Path("~/Downloads/bota_bag_test_wo_gripper").expanduser()

SHOW_IMAGE = True


# ==========================================
# --- 2. ENTRY POINT ---
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Read 6-DOF wrench from a rosbag, apply calibration offset, and plot."
    )
    parser.add_argument(
        "bag_path",
        nargs="?",
        default=str(BAG_PATH),
        help=f"Path to the rosbag (default: {BAG_PATH})",
    )
    parser.add_argument(
        "--calib-bag",
        default=str(CALIB_BAG_PATH),
        help=f"Path to static calibration rosbag (default: {CALIB_BAG_PATH})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plots.",
    )
    args = parser.parse_args()

    offset = WrenchCalibrator(bag_path=args.calib_bag).compute_offset()

    bag = Path(args.bag_path).expanduser()
    out_name = bag.stem[:-1] + "g"   # replace last letter with 'g' (e.g. p6-a2-w → p6-a2-g)
    ForceProcessor(
        bag_path=bag,
        output_dir=DATA_DIR / out_name,
        show_image=(not args.no_plot),
        offset=offset,
    ).run()

    if not args.no_plot:
        plt.show()


if __name__ == "__main__":
    main()
