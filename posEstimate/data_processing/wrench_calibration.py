"""
Wrench calibration: compute per-axis force/torque offsets from a static bag.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_processing.rosbag_reader import AnyReader, typestore
from data_processing.force_process import remove_spikes

TOPIC_NAME = "/rokubi/wrench"
BAG_PATH = Path("~/Downloads/bota_bag_test_wo_gripper").expanduser()


class WrenchCalibrator:
    """
    Reads a static (no-load) rosbag and computes the mean wrench offset
    (force xyz + torque xyz) to subtract from subsequent recordings.
    """

    def __init__(self, bag_path=BAG_PATH, topic=TOPIC_NAME):
        self.bag_path = Path(bag_path)
        self.topic = topic
        self.offset = None  # np.ndarray shape (6,): [Fx, Fy, Fz, Tx, Ty, Tz]

    def compute_offset(self):
        """Load the bag and return the mean wrench as a (6,) offset array."""
        samples = []

        with AnyReader([self.bag_path], default_typestore=typestore) as reader:
            conn = next(
                (c for c in reader.connections if c.topic == self.topic), None
            )
            if conn is None:
                raise RuntimeError(
                    f"Topic {self.topic!r} not found in {self.bag_path}.\n"
                    f"Available topics: {[c.topic for c in reader.connections]}"
                )

            for conn_, _, raw in reader.messages(connections=[conn]):
                msg = typestore.deserialize_cdr(raw, conn_.msgtype)
                f = msg.wrench.force
                t = msg.wrench.torque
                samples.append([
                    float(f.x), float(f.y), float(f.z),
                    float(t.x), float(t.y), float(t.z),
                ])

        if not samples:
            raise RuntimeError(
                f"No messages on {self.topic!r} in {self.bag_path}"
            )

        # prepend a dummy timestamp column so remove_spikes can operate on columns 1:
        raw = np.asarray(samples, dtype=np.float64)
        dummy_t = np.zeros((len(raw), 1), dtype=np.float64)
        data_with_t = np.hstack([dummy_t, raw])
        clean = remove_spikes(data_with_t)
        self.offset = np.nanmean(clean[:, 1:], axis=0)

        labels = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
        print(f"Calibration offsets from {self.bag_path.name}:")
        for label, val in zip(labels, self.offset):
            print(f"  {label}: {val:.4f}")

        return self.offset


def main():
    parser = argparse.ArgumentParser(
        description="Compute wrench calibration offsets from a static rosbag."
    )
    parser.add_argument(
        "bag_path", nargs="?", default=str(BAG_PATH), help="Path to static rosbag"
    )
    parser.add_argument("--topic", default=TOPIC_NAME, help="Wrench topic")
    args = parser.parse_args()

    calibrator = WrenchCalibrator(bag_path=args.bag_path, topic=args.topic)
    calibrator.compute_offset()


if __name__ == "__main__":
    main()
