"""
Data extraction pipeline:

    1. Read camera rosbag → save videos + depth to disk
    2. Crop saved videos/depth by CROP_VIDEO seconds  (0 = skip)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow imports from posEstimate/ and posEstimate/scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rosbag_reader import AnyReader, typestore

# ==========================================
# --- USER CONFIGURATION ---
# ==========================================

DATA_NAME  = "P3-B2"
BAG_PATH   = Path(f"~/Downloads/{DATA_NAME}").expanduser()
OUT_BASE   = Path("posEstimate/data")

# ── Crop ──────────────────────────────────────────────────────────────
#   0 = keep full video; N > 0 = discard first N seconds and overwrite
CROP_VIDEO = 6

LEFT_TOPIC        = "/left_camera/camera/camera/color/image_raw"
RIGHT_TOPIC       = "/right_camera/camera/camera/color/image_raw"
LEFT_DEPTH_TOPIC  = "/left_camera/camera/camera/aligned_depth_to_color/image_raw"
RIGHT_DEPTH_TOPIC = "/right_camera/camera/camera/aligned_depth_to_color/image_raw"

# False = skip depth (saves disk space; ArUco falls back to PnP-estimated depth)
SAVE_DEPTH = True

# "mp4v" is widely compatible; "avc1" gives smaller H.264 files if supported
VIDEO_CODEC = "mp4v"

# ==========================================
# --- HELPERS ---
# ==========================================

def _get_stamp(msg, fallback_ns):
    try:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    except Exception:
        return fallback_ns * 1e-9


def _ros_image_to_bgr(msg):
    h, w = msg.height, msg.width
    data = msg.data if isinstance(msg.data, bytes) else bytes(msg.data)
    img  = np.frombuffer(data, dtype=np.uint8)
    enc  = msg.encoding
    if enc == "bgr8":
        return img.reshape((h, w, 3))
    elif enc == "rgb8":
        return cv2.cvtColor(img.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
    elif enc == "mono8":
        return cv2.cvtColor(img.reshape((h, w)), cv2.COLOR_GRAY2BGR)
    elif enc == "bgra8":
        return cv2.cvtColor(img.reshape((h, w, 4)), cv2.COLOR_BGRA2BGR)
    elif enc == "rgba8":
        return cv2.cvtColor(img.reshape((h, w, 4)), cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError(f"Unsupported encoding: {enc}")


# ==========================================
# --- EXTRACTION ---
# ==========================================

def extract():
    out_dir = OUT_BASE / DATA_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting bag : {BAG_PATH}")
    print(f"Output dir     : {out_dir}")

    topics = {LEFT_TOPIC, RIGHT_TOPIC}
    if SAVE_DEPTH:
        topics |= {LEFT_DEPTH_TOPIC, RIGHT_DEPTH_TOPIC}
        (out_dir / "left_depth").mkdir(exist_ok=True)
        (out_dir / "right_depth").mkdir(exist_ok=True)

    # Per-topic state
    writers    = {}                         # topic → cv2.VideoWriter
    ts_lists   = {t: [] for t in topics}   # topic → [timestamp, ...]
    depth_idx  = {LEFT_DEPTH_TOPIC: 0, RIGHT_DEPTH_TOPIC: 0}

    n_msg = 0

    with AnyReader([BAG_PATH], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        if not conns:
            raise RuntimeError(f"No matching topics found in bag: {topics}")

        # Compute true fps per RGB topic from bag metadata (no extra pass needed)
        try:
            bag_dur_s = (reader.end_time - reader.start_time) * 1e-9
            topic_fps = {
                c.topic: c.msgcount / bag_dur_s
                for c in reader.connections
                if c.topic in (LEFT_TOPIC, RIGHT_TOPIC) and c.msgcount > 1 and bag_dur_s > 0
            }
            print(f"  Bag duration   : {bag_dur_s:.2f} s")
            for t, f in topic_fps.items():
                print(f"  FPS ({t.split('/')[-4]}): {f:.3f}")
        except Exception:
            topic_fps = {}

        for conn, ts_ns, raw in reader.messages(connections=conns):
            topic = conn.topic
            try:
                msg   = typestore.deserialize_cdr(raw, conn.msgtype)
                stamp = _get_stamp(msg, ts_ns)
            except Exception as e:
                print(f"  Warning: deserialize failed ({topic}): {e}")
                continue

            # ── RGB cameras ─────────────────────────────────────────────
            if topic in (LEFT_TOPIC, RIGHT_TOPIC):
                try:
                    frame = _ros_image_to_bgr(msg)
                except Exception as e:
                    print(f"  Warning: RGB skip ({topic}): {e}")
                    continue

                if topic not in writers:
                    h, w = frame.shape[:2]
                    side = "left" if topic == LEFT_TOPIC else "right"
                    path = str(out_dir / f"{side}.mp4")
                    fps  = topic_fps.get(topic, 30.0)
                    writers[topic] = cv2.VideoWriter(
                        path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (w, h)
                    )
                    print(f"  Opened writer: {path}  ({w}x{h})  @ {fps:.3f} fps")

                writers[topic].write(frame)
                ts_lists[topic].append(stamp)

            # ── Depth (16-bit PNG, one file per frame) ───────────────────
            elif SAVE_DEPTH and topic in (LEFT_DEPTH_TOPIC, RIGHT_DEPTH_TOPIC):
                try:
                    h, w = msg.height, msg.width
                    data = msg.data if isinstance(msg.data, bytes) else bytes(msg.data)
                    dimg = np.frombuffer(data, dtype=np.uint16)
                    dimg = dimg.reshape((h, msg.step // 2))[:, :w]

                    side = "left" if topic == LEFT_DEPTH_TOPIC else "right"
                    idx  = depth_idx[topic]
                    cv2.imwrite(str(out_dir / f"{side}_depth" / f"{idx:06d}.png"), dimg)

                    ts_lists[topic].append(stamp)
                    depth_idx[topic] += 1
                except Exception as e:
                    print(f"  Warning: depth skip ({topic}): {e}")

            n_msg += 1
            if n_msg % 3000 == 0:
                print(f"  {n_msg} messages | "
                      f"left={len(ts_lists[LEFT_TOPIC])} "
                      f"right={len(ts_lists[RIGHT_TOPIC])} frames")

    # --- Release video writers ---
    for w in writers.values():
        w.release()

    # --- Save timestamps ---
    np.save(str(out_dir / "left_ts.npy"),  np.array(ts_lists[LEFT_TOPIC],  dtype=np.float64))
    np.save(str(out_dir / "right_ts.npy"), np.array(ts_lists[RIGHT_TOPIC], dtype=np.float64))
    if SAVE_DEPTH:
        np.save(str(out_dir / "left_depth_ts.npy"),
                np.array(ts_lists[LEFT_DEPTH_TOPIC],  dtype=np.float64))
        np.save(str(out_dir / "right_depth_ts.npy"),
                np.array(ts_lists[RIGHT_DEPTH_TOPIC], dtype=np.float64))

    print("\n=== Extraction complete ===")
    print(f"  Left RGB   : {len(ts_lists[LEFT_TOPIC])} frames  → {out_dir / 'left.mp4'}")
    print(f"  Right RGB  : {len(ts_lists[RIGHT_TOPIC])} frames  → {out_dir / 'right.mp4'}")
    if SAVE_DEPTH:
        print(f"  Left depth : {depth_idx[LEFT_DEPTH_TOPIC]} frames → {out_dir / 'left_depth/'}")
        print(f"  Right depth: {depth_idx[RIGHT_DEPTH_TOPIC]} frames → {out_dir / 'right_depth/'}")


def crop_saved_videos():
    """Trim the first CROP_VIDEO seconds from saved mp4s and timestamp arrays (overwrite)."""
    if CROP_VIDEO <= 0:
        print("CROP_VIDEO = 0 — nothing to crop.")
        return

    out_dir = OUT_BASE / DATA_NAME
    sides   = ["left", "right"]

    for side in sides:
        mp4_path = out_dir / f"{side}.mp4"
        ts_path  = out_dir / f"{side}_ts.npy"
        if not mp4_path.exists():
            print(f"  Skipping {side}: {mp4_path} not found")
            continue

        ts = np.load(str(ts_path))
        t0 = ts[0] + CROP_VIDEO
        sf = int(np.searchsorted(ts, t0))
        if sf >= len(ts):
            print(f"  {side}: CROP_VIDEO exceeds video duration — skipping")
            continue

        print(f"  Cropping {side}.mp4: keeping frames {sf}–{len(ts)-1} ({len(ts)-sf} frames)")

        cap = cv2.VideoCapture(str(mp4_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

        ret, first = cap.read()
        if not ret:
            print(f"  {side}: could not seek to frame {sf}")
            cap.release()
            continue
        h, w = first.shape[:2]

        tmp_path = out_dir / f"{side}_crop_tmp.mp4"
        writer   = cv2.VideoWriter(
            str(tmp_path), cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (w, h)
        )
        writer.write(first)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()
        cap.release()

        tmp_path.replace(mp4_path)
        np.save(str(ts_path), ts[sf:])
        print(f"    → {mp4_path} overwritten  ({len(ts)-sf} frames)")

        # Depth timestamps (optional)
        depth_ts_path = out_dir / f"{side}_depth_ts.npy"
        if depth_ts_path.exists():
            dts = np.load(str(depth_ts_path))
            d0  = ts[sf]           # use same wall-clock cut point
            dsf = int(np.searchsorted(dts, d0))
            np.save(str(depth_ts_path), dts[dsf:])
            print(f"    → {side}_depth_ts.npy trimmed to {len(dts)-dsf} entries")


if __name__ == "__main__":
    extract()           # 1. camera rosbag → videos + depth
    crop_saved_videos() # 2. trim first CROP_VIDEO seconds (no-op if 0)
