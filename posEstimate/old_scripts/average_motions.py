"""
average_motions.py — Average repeated motion cycles from a gripper trajectory CSV.

Cycle boundaries are detected automatically: whenever the gripper position
returns close to its starting position (Euclidean distance < RETURN_THRESH_MM),
a new cycle begins.

Usage:
    1. Set CROP (must match read_two.py so time bases agree).
    2. Adjust RETURN_THRESH_MM and MIN_CYCLE_SEC if needed.
    3. Run:  python posEstimate/scripts/average_motions.py

Output:
    <DATA_NAME>_avg.csv   — averaged single-cycle trajectory
    Overlay plot of all cycles + average with ±1σ bands.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp

# ===========================================================
# USER CONFIGURATION
# ===========================================================

DATA_NAME = "P2-A4"
DATA_DIR  = Path("posEstimate/data") / DATA_NAME
CSV_PATH  = DATA_DIR / f"{DATA_NAME}.csv"
VIDEO_IN  = DATA_DIR / f"{DATA_NAME}_right.mp4"   # annotated video from read_two.py

CROP = (18, 43)  # must match read_two.py so we know the CSV timebase

# Auto-detection parameters
RETURN_THRESH_MM = 30.0   # distance from start position to count as "returned"
MIN_CYCLE_SEC    = 6.0    # minimum cycle duration — ignores brief returns

# Video annotation
FLASH_FRAMES = 15         # how many frames to show the "CYCLE N" banner

# ===========================================================
# I/O helpers
# ===========================================================

def _read_csv(path):
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    t     = np.array([float(r["t"])       for r in rows])
    frame = np.array([int(float(r["frame"])) for r in rows])
    pos   = np.array([[float(r["pos_x"]),
                        float(r["pos_y"]),
                        float(r["pos_z"])]  for r in rows])
    euler = np.array([[float(r["orient_x"]),
                        float(r["orient_y"]),
                        float(r["orient_z"])] for r in rows])
    return t, frame, pos, euler


def _write_csv(path, t, frames, pos, euler):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "frame", "pos_x", "pos_y", "pos_z",
                     "orient_x", "orient_y", "orient_z"])
        for i in range(len(t)):
            w.writerow([
                round(float(t[i]), 6), int(frames[i]),
                round(float(pos[i, 0]), 3),
                round(float(pos[i, 1]), 3),
                round(float(pos[i, 2]), 3),
                round(float(euler[i, 0]), 6),
                round(float(euler[i, 1]), 6),
                round(float(euler[i, 2]), 6),
            ])
    print(f"Saved: {path}")


def _interpolate_full(t, frames, pos, euler, fps):
    """
    Interpolate sparse CSV (missing frames) into a complete trajectory
    with one row per video frame.  Position is linearly interpolated;
    orientation is Slerp-interpolated in quaternion space.
    """
    f_min, f_max = int(frames[0]), int(frames[-1])
    all_frames = np.arange(f_min, f_max + 1)
    dt = 1.0 / fps
    all_t = all_frames * dt

    # Position: linear interp per axis
    full_pos = np.zeros((len(all_frames), 3))
    for ax in range(3):
        full_pos[:, ax] = np.interp(all_frames, frames, pos[:, ax])

    # Orientation: Slerp
    rots = Rotation.from_euler("xyz", euler)
    quats = rots.as_quat()
    quats = _ensure_quat_continuity(quats)
    rots = Rotation.from_quat(quats)
    slerp = Slerp(frames, rots)
    full_euler = slerp(all_frames).as_euler("xyz")

    n_filled = len(all_frames) - len(frames)
    print(f"Interpolated: {len(frames)} → {len(all_frames)} frames "
          f"({n_filled} filled)")

    return all_t, all_frames.astype(float), full_pos, full_euler


# ===========================================================
# Cycle detection
# ===========================================================

def _detect_cycles(t, pos, thresh_mm, min_sec):
    """
    Find cycle start indices by detecting when the gripper height (Z)
    returns close to its starting height.

    For each return period (contiguous run of frames within thresh_mm of
    start height), picks the frame closest to start_z as the cycle start.

    Returns list of start indices (always includes 0).
    """
    start_z = pos[0, 2]
    dist = np.abs(pos[:, 2] - start_z)
    close_mask = dist < thresh_mm

    # Find contiguous "close" regions, pick the minimum-distance frame in each
    starts = [0]
    in_close = False
    region_best_i = None
    region_best_d = np.inf

    for i in range(1, len(t)):
        if close_mask[i]:
            if not in_close:
                # entering a close region
                in_close = True
                region_best_i = i
                region_best_d = dist[i]
            elif dist[i] < region_best_d:
                region_best_i = i
                region_best_d = dist[i]
        else:
            if in_close:
                # leaving the close region — commit the best frame
                if (t[region_best_i] - t[starts[-1]]) >= min_sec:
                    starts.append(region_best_i)
                in_close = False
                region_best_i = None
                region_best_d = np.inf

    # Handle if the trajectory ends inside a close region
    if in_close and region_best_i is not None:
        if (t[region_best_i] - t[starts[-1]]) >= min_sec:
            starts.append(region_best_i)

    return starts


# ===========================================================
# Core logic
# ===========================================================

def _ensure_quat_continuity(quats):
    """Flip quaternions so consecutive ones are on the same hemisphere."""
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    return quats


def _resample_pos(t_in, pos_in, t_out):
    """Linear interpolation of position to new time grid."""
    out = np.zeros((len(t_out), 3))
    for ax in range(3):
        out[:, ax] = np.interp(t_out, t_in, pos_in[:, ax])
    return out


def _resample_orient(t_in, euler_in, t_out):
    """Resample orientation via Slerp in quaternion space."""
    rots = Rotation.from_euler("xyz", euler_in)
    quats = rots.as_quat()
    quats = _ensure_quat_continuity(quats)
    rots = Rotation.from_quat(quats)
    slerp = Slerp(t_in, rots)
    return slerp(t_out).as_euler("xyz")


def _average_orientations(euler_stack):
    """
    Average K orientation trajectories (each N×3 Euler).
    Uses normalized quaternion mean — valid when orientations are close.
    """
    K, N, _ = euler_stack.shape
    quat_stack = np.zeros((K, N, 4))
    for k in range(K):
        q = Rotation.from_euler("xyz", euler_stack[k]).as_quat()
        q = _ensure_quat_continuity(q)
        if k > 0 and np.dot(q[0], quat_stack[0, 0]) < 0:
            q = -q
        quat_stack[k] = q

    mean_q = quat_stack.mean(axis=0)
    mean_q /= np.linalg.norm(mean_q, axis=1, keepdims=True)
    return Rotation.from_quat(mean_q).as_euler("xyz")


# ===========================================================
# Video annotation
# ===========================================================

CYCLE_COLORS = [
    (0, 255, 0),    # green
    (0, 200, 255),  # orange
    (255, 100, 0),  # blue
    (200, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]


def _annotate_video(video_path, csv_frames, cycle_start_indices, out_suffix="_cycles"):
    """
    Read the annotated video and overlay cycle markers.
    - Thin colored bar at the top showing current cycle
    - "CYCLE N" banner that flashes at each cycle start
    - Ongoing small cycle label in the corner
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = video_path.stem
    out_path = video_path.parent / f"{stem}{out_suffix}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Map video frame index → cycle number using the CSV 'frame' column.
    # The CSV skips frames with no ArUco detection, so we must use the
    # frame column (which matches the video frame index) for the lookup.
    cycle_start_frame_to_num = {}
    for ci, si in enumerate(cycle_start_indices):
        video_frame = int(csv_frames[si])
        cycle_start_frame_to_num[video_frame] = ci + 1

    current_cycle = 0
    flash_remaining = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if this frame starts a new cycle
        if frame_idx in cycle_start_frame_to_num:
            current_cycle = cycle_start_frame_to_num[frame_idx]
            flash_remaining = FLASH_FRAMES

        if current_cycle > 0:
            color = CYCLE_COLORS[(current_cycle - 1) % len(CYCLE_COLORS)]

            # Colored bar at top
            cv2.rectangle(frame, (0, 0), (w, 6), color, -1)

            # Small persistent label
            label = f"Cycle {current_cycle}"
            cv2.putText(frame, label, (w - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Flash banner at cycle start
            if flash_remaining > 0:
                banner = f"CYCLE {current_cycle} START"
                text_size = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                tx = (w - text_size[0]) // 2
                ty = h // 2

                # Semi-transparent dark background
                overlay = frame.copy()
                cv2.rectangle(overlay, (tx - 15, ty - text_size[1] - 15),
                              (tx + text_size[0] + 15, ty + 15), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                cv2.putText(frame, banner, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                flash_remaining -= 1

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved annotated video: {out_path}  ({frame_idx} frames)")


def main():
    # --- Load sparse CSV ---
    t_sparse, frames_sparse, pos_sparse, euler_sparse = _read_csv(CSV_PATH)

    # --- Get video FPS ---
    cap = cv2.VideoCapture(str(VIDEO_IN))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Video FPS: {fps:.3f}")

    # --- Interpolate to complete trajectory (one row per frame) ---
    t_raw, frames_raw, pos_raw, euler_raw = _interpolate_full(
        t_sparse, frames_sparse, pos_sparse, euler_sparse, fps)

    # Save the complete interpolated trajectory
    full_csv = CSV_PATH.parent / f"{DATA_NAME}_full.csv"
    _write_csv(full_csv, t_raw, frames_raw, pos_raw, euler_raw)

    # --- Detect cycle boundaries (on complete trajectory) ---
    starts = _detect_cycles(t_raw, pos_raw, RETURN_THRESH_MM, MIN_CYCLE_SEC)

    print(f"Detected {len(starts)} cycles (thresh={RETURN_THRESH_MM} mm, "
          f"min_cycle={MIN_CYCLE_SEC} s)")
    for ci, si in enumerate(starts):
        print(f"  Cycle {ci+1}: starts at t={t_raw[si]:.2f}s  "
              f"(frame {int(frames_raw[si])})")

    # --- Annotate video with cycle markers ---
    if VIDEO_IN.exists():
        _annotate_video(VIDEO_IN, frames_raw, starts)
    else:
        print(f"Video not found ({VIDEO_IN}), skipping annotation.")

    # Build segment slices: [start0:start1], [start1:start2], ..., [startN:]
    segments_pos   = []
    segments_euler = []
    segments_dur   = []

    for i in range(len(starts)):
        i0 = starts[i]
        i1 = starts[i + 1] if i + 1 < len(starts) else len(t_raw)

        t_seg     = t_raw[i0:i1] - t_raw[i0]
        pos_seg   = pos_raw[i0:i1].copy()
        euler_seg = euler_raw[i0:i1].copy()

        # zero position to start at origin
        pos_seg -= pos_seg[0]

        segments_pos.append((t_seg, pos_seg))
        segments_euler.append((t_seg, euler_seg))
        segments_dur.append(t_seg[-1])

    n_cycles = len(segments_pos)
    print(f"Detected {n_cycles} cycles at t = "
          + ", ".join(f"{t_raw[s]:.2f}s" for s in starts))
    print(f"Durations: " + ", ".join(f"{d:.2f}s" for d in segments_dur))

    if n_cycles < 2:
        print("Only 1 cycle detected — try increasing RETURN_THRESH_MM "
              "or decreasing MIN_CYCLE_SEC.")
        return

    # --- Resample all cycles to common length ---
    # Use the shortest cycle's duration so no segment is extrapolated
    min_dur = min(segments_dur)
    median_len = int(np.median([len(s[0]) for s in segments_pos]))
    t_common = np.linspace(0, min_dur, median_len)

    pos_stack   = np.zeros((n_cycles, median_len, 3))
    euler_stack = np.zeros((n_cycles, median_len, 3))

    for k in range(n_cycles):
        t_seg, p_seg = segments_pos[k]
        _, e_seg     = segments_euler[k]
        pos_stack[k]   = _resample_pos(t_seg, p_seg, t_common)
        euler_stack[k] = _resample_orient(t_seg, e_seg, t_common)

    # --- Average ---
    pos_mean = pos_stack.mean(axis=0)
    pos_std  = pos_stack.std(axis=0)
    euler_mean = _average_orientations(euler_stack)

    # --- Save ---
    out_path = CSV_PATH.parent / f"{DATA_NAME}_avg.csv"
    avg_frames = np.arange(len(t_common), dtype=float)
    _write_csv(out_path, t_common, avg_frames, pos_mean, euler_mean)

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    fig.suptitle(f"{DATA_NAME}: {n_cycles} cycles averaged  "
                 f"(thresh={RETURN_THRESH_MM} mm, min_cycle={MIN_CYCLE_SEC} s)",
                 fontsize=11)

    pos_labels = ["pos_x (mm)", "pos_y (mm)", "pos_z (mm)"]
    ori_labels = ["orient_x (rad)", "orient_y (rad)", "orient_z (rad)"]

    for ax_i in range(3):
        # Position
        ax = axes[0, ax_i]
        for k in range(n_cycles):
            ax.plot(t_common, pos_stack[k, :, ax_i],
                    alpha=0.35, linewidth=0.8, label=f"cycle {k+1}")
        ax.plot(t_common, pos_mean[:, ax_i],
                color="black", linewidth=2, label="avg")
        ax.fill_between(t_common,
                         pos_mean[:, ax_i] - pos_std[:, ax_i],
                         pos_mean[:, ax_i] + pos_std[:, ax_i],
                         color="black", alpha=0.12)
        ax.set_title(pos_labels[ax_i], fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.4)

        # Orientation
        ax = axes[1, ax_i]
        for k in range(n_cycles):
            ax.plot(t_common, euler_stack[k, :, ax_i],
                    alpha=0.35, linewidth=0.8)
        ax.plot(t_common, euler_mean[:, ax_i],
                color="black", linewidth=2, label="avg")
        ax.set_title(ori_labels[ax_i], fontsize=9)
        ax.set_xlabel("t (s)")
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
