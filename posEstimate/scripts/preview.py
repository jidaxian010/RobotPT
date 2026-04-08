"""
preview.py — save /aruco/gripper_pose_four (image topic) from a rosbag to MP4.

Playback is real-time: FPS is derived from the actual message timestamps.

Usage:
    python preview.py [bag_path] [--out preview.mp4]
    python preview.py ~/Downloads/jeff --crop-start 11.9 --crop-end 17.6
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_processing.rosbag_reader import AnyReader, typestore

# ── config ────────────────────────────────────────────────────────────────────
DATA_NAME = "p9-c1-g"
DATA_DIR  = Path(__file__).resolve().parents[1] / "data"
BAG_PATH  = Path(f"~/Downloads/{DATA_NAME}").expanduser()
IMAGE_TOPIC = "/aruco/gripper_pose_four"


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_stamp_sec(msg, fallback_ts_ns):
    try:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    except Exception:
        return fallback_ts_ns * 1e-9


def _decode_image(msg):
    """Convert a sensor_msgs/Image or CompressedImage to a BGR numpy array."""
    if hasattr(msg, "format"):
        # CompressedImage
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Raw Image
    enc = msg.encoding.lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    if enc in ("bgr8", "bgr"):
        return buf.reshape(msg.height, msg.width, 3)
    if enc in ("rgb8", "rgb"):
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
    if enc in ("mono8", "8uc1"):
        return cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    if enc in ("bgra8",):
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 4), cv2.COLOR_BGRA2BGR)
    if enc in ("rgba8",):
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 4), cv2.COLOR_RGBA2BGR)
    if enc in ("16uc1", "mono16"):
        img16 = buf.view(np.uint16).reshape(msg.height, msg.width)
        img8  = (img16 >> 8).astype(np.uint8)
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Save image topic from rosbag to MP4 at real-time speed."
    )
    parser.add_argument("bag_path", nargs="?", default=str(BAG_PATH),
                        help=f"Path to the rosbag (default: {BAG_PATH})")
    parser.add_argument("--out", default=None,
                        help="Output MP4 path (default: data/<stem>/<stem>_preview.mp4)")
    parser.add_argument("--crop-start", type=float, default=None, dest="crop_start",
                        help="Skip messages before bag_start + N seconds")
    parser.add_argument("--crop-end",   type=float, default=None, dest="crop_end",
                        help="Skip messages after  bag_start + N seconds")
    args = parser.parse_args()

    bag_path = Path(args.bag_path).expanduser()
    out_dir  = DATA_DIR / bag_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"{bag_path.stem}_preview.mp4"

    # ── read all frames ───────────────────────────────────────────────────────
    frames     = []
    timestamps = []

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conn = next((c for c in reader.connections if c.topic == IMAGE_TOPIC), None)
        if conn is None:
            avail = [c.topic for c in reader.connections]
            raise RuntimeError(
                f"Topic {IMAGE_TOPIC!r} not found.\nAvailable: {avail}"
            )

        t_bag_start = None
        for _, ts, raw in reader.messages(connections=[conn]):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            t   = _get_stamp_sec(msg, ts)

            if t_bag_start is None:
                t_bag_start = t
            t_rel = t - t_bag_start

            if args.crop_start is not None and t_rel < args.crop_start:
                continue
            if args.crop_end is not None and t_rel > args.crop_end:
                break

            bgr = _decode_image(msg)
            if bgr is None:
                continue
            frames.append(bgr)
            timestamps.append(t)

    if not frames:
        raise RuntimeError(f"No frames read from {IMAGE_TOPIC!r}")

    n      = len(frames)
    dur    = timestamps[-1] - timestamps[0]
    fps    = n / dur if dur > 0 else 30.0
    h, w   = frames[0].shape[:2]

    print(f"Read {n} frames  ({dur:.2f} s  →  {fps:.2f} fps)")
    print(f"Saving: {out_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for bgr in frames:
        writer.write(bgr)

    writer.release()
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
