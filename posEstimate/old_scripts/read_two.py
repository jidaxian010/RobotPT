"""
Annotate pre-extracted RealSense camera videos with human pose (MediaPipe)
and gripper tracking (ArUco + depth), then save a gripper 6DoF CSV and
a joint-angle rule book.

Prerequisites:
    Run read_rosbag.py first to extract the bag to posEstimate/data/<DATA_NAME>/.

Processing pipeline:
    Pass 1  — ArUco detection on gripper-cam video (sequential, fast).
    Pass 2  — MediaPipe annotation on both cameras IN PARALLEL (2 processes).

Outputs (in posEstimate/data/<DATA_NAME>/):
    <DATA_NAME>_left.mp4           annotated left camera
    <DATA_NAME>_right.mp4          annotated right camera
    <DATA_NAME>.csv                gripper 6DoF trajectory
    <DATA_NAME>_pose_rules.yaml    patient joint-angle rule book
    <DATA_NAME>_force_rules.yaml   Fx/Fy/Fz stats from cropped force_raw.npy
"""
import bisect
import contextlib
import csv
import datetime
import multiprocessing as mp
import sys
from pathlib import Path

import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mpipe
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as ScipyRotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from denoise import denoise_pose_list

# MediaPipe aliases
BaseOptions           = mpipe.tasks.BaseOptions
PoseLandmarker        = mpipe.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mpipe.tasks.vision.PoseLandmarkerOptions
RunningMode           = mpipe.tasks.vision.RunningMode

# ==========================================
# --- USER CONFIGURATION ---
# ==========================================

DATA_NAME   = "P3-A3"
GRIPPER_CAM = "right"       # "left" or "right": which camera provides ArUco 3D tracking
CROP        = (15, 21)       # (start, end); negative values = seconds from end of stream
# CROP      = None          # None = no extra crop (process everything in the extracted video)
CROP_UNIT   = "s"           # "s" (seconds) or "frame"

HUMAN_POSE  = True          # Toggle MediaPipe human pose

GRIPPER_PROCESS = "pick"    # "pick" = single pass (no cycle detection)
                            # "cycle" = detect cycles and average them
IF_MIRROR       = False      # Only used when GRIPPER_PROCESS="pick":
                            # append a time-reversed (mirrored) copy of the trajectory

DATA_DIR  = Path("posEstimate/data") / DATA_NAME   # folder created by read_rosbag.py
OUT_DIR   = DATA_DIR                               # outputs go into the same folder

LEFT_TOPIC        = "/left_camera/camera/camera/color/image_raw"
RIGHT_TOPIC       = "/right_camera/camera/camera/color/image_raw"
LEFT_DEPTH_TOPIC  = "/left_camera/camera/camera/aligned_depth_to_color/image_raw"
RIGHT_DEPTH_TOPIC = "/right_camera/camera/camera/aligned_depth_to_color/image_raw"

SMOOTH_GRIPPER_POSE = True
SMOOTH_MED_KERNEL   = 15   # must be odd
SMOOTH_SIGMA        = 10

MARKER_SIZE_METERS = 0.0725

# --- Human pose rule book ---
POSE_RULE_JOINTS = [11, 12, 13, 14, 23, 24]
VIS_THRESHOLD    = 0.3

JOINT_ANGLE_TRIPLETS = {
    "right_arm_torso": (14, 12, 24),
    "left_arm_torso":  (13, 11, 23),
}

_TRIPLET_DESCRIPTIONS = {
    "right_arm_torso": "right elbow – right shoulder – right hip (arm elevation vs trunk)",
    "left_arm_torso":  "left elbow – left shoulder – left hip (arm elevation vs trunk)",
}

MODEL_PATH = str(Path(__file__).resolve().parent.parent /
                 "data" / "pose_landmarker_full.task")

# Cam: 335222070270
_FX = 602.6598510742188
_FY = 602.2169799804688
_CX = 319.1910400390625
_CY = 249.92578125

# # ?
# _FX = 602.6597900390625
# _FY = 602.2169799804688
# _CX = 423.1910400390625
# _CY = 249.92578125

# # Cam: 018322070277
# _FX = 914.2034301757812
# _FY = 914.743896484375
# _CX = 646.1510620117188
# _CY = 363.29620361328125


CAMERA_MATRIX = np.array(
    [[_FX, 0.0, _CX], [0.0, _FY, _CY], [0.0, 0.0, 1.0]], dtype=np.float32
)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float32)


# ==========================================
# --- GRIPPER GEOMETRY ---
# ==========================================

def build_T(t, R):
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = np.asarray(R, dtype=np.float64)
    T[0:3, 3:4] = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return T


def deproject_pixel_depth_to_point_m(u, v, depth_m):
    x = (float(u) - _CX) * float(depth_m) / _FX
    y = (float(v) - _CY) * float(depth_m) / _FY
    return np.array([x, y, float(depth_m)], dtype=np.float64)


R_10 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
R_20 = np.array([[0.7071, 0.0, 0.7071], [0.0, 1.0, 0.0], [-0.7071, 0.0, 0.7071]], dtype=np.float64)
R_30 = np.array([[0.0, 0.0, 1.0], [-0.6654, 0.7465, 0.0], [-0.7465, -0.6654, 0.0]], dtype=np.float64)
R_40 = np.array([[-0.7071, 0.0, 0.7071], [0.0, 1.0, 0.0], [-0.7071, 0.0, -0.7071]], dtype=np.float64)
R_50 = np.array([[0.0, 0.0, 1.0], [0.7465, 0.6654, 0.0], [-0.6654, 0.7465, 0.0]], dtype=np.float64)

TAG_TRANSFORMS = {
    1: build_T([0.3927,  0.0225, -0.2142], R_10),
    2: build_T([0.3641,  0.0225,  0.0993], R_20),
    3: build_T([0.3927, -0.0592, -0.2003], R_30),
    4: build_T([0.1912,  0.0225, -0.4561], R_40),
    5: build_T([0.3927,  0.0928, -0.1703], R_50),
}

_half = MARKER_SIZE_METERS / 2.0
marker_3d_edges = np.array(
    [[-_half,  _half, 0.0], [ _half,  _half, 0.0],
     [ _half, -_half, 0.0], [-_half, -_half, 0.0]], dtype=np.float32
)


# ==========================================
# --- MODULE-LEVEL HELPERS ---
# ==========================================

def _angle_at_joint_deg(A, B, C):
    BA = np.asarray(A, dtype=np.float64) - np.asarray(B, dtype=np.float64)
    BC = np.asarray(C, dtype=np.float64) - np.asarray(B, dtype=np.float64)
    cos_t = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


def fps_from_timestamps(timestamps, fallback=30.0):
    if len(timestamps) < 2:
        return fallback
    dts = np.diff(np.asarray(timestamps, dtype=np.float64))
    dts = dts[dts > 0]
    return float(1.0 / np.median(dts)) if len(dts) else fallback


def _crop_frame_range(timestamps, crop, crop_unit):
    """Return (start_idx, end_idx) slice for the crop window."""
    if crop is None:
        return 0, len(timestamps)
    total = len(timestamps)
    if total == 0:
        return 0, 0
    start, end = crop
    t_rel = np.asarray(timestamps, dtype=np.float64) - timestamps[0]
    if crop_unit == "s":
        dur     = float(t_rel[-1])
        start_s = float(start) if float(start) >= 0 else dur + float(start)
        end_s   = float(end)   if float(end)   >= 0 else dur + float(end)
        sf = int(np.searchsorted(t_rel, max(0.0, start_s), side="left"))
        ef = int(np.searchsorted(t_rel, max(0.0, end_s),   side="left"))
    else:
        sf = int(start) if int(start) >= 0 else total + int(start)
        ef = int(end)   if int(end)   >= 0 else total + int(end)
    sf = int(np.clip(sf, 0, total))
    ef = int(np.clip(ef, sf, total))
    print(f"  Crop: frames [{sf}, {ef}) of {total}")
    return sf, ef


def _find_depth_idx(depth_ts_arr, query_ts, max_dt=0.05):
    """Binary-search for nearest depth frame index, or None if beyond max_dt."""
    if len(depth_ts_arr) == 0:
        return None
    i = bisect.bisect_left(depth_ts_arr, query_ts)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(depth_ts_arr)]
    if not candidates:
        return None
    best = min(candidates, key=lambda j: abs(depth_ts_arr[j] - query_ts))
    return best if abs(depth_ts_arr[best] - query_ts) <= max_dt else None


def _annotate_gcam_frame(out, dets, gp):
    """Overlay ArUco markers and fused gripper frame axes onto a frame (in-place)."""
    if not dets:
        return
    draw_corners = [d["corners_draw"] for d in dets]
    draw_ids     = np.array([[d["tag_id"]] for d in dets], dtype=np.int32)
    cv2.aruco.drawDetectedMarkers(out, draw_corners, draw_ids)

    k_positions = []
    for det in dets:
        cv2.drawFrameAxes(out, CAMERA_MATRIX, DIST_COEFFS,
                          det["rvec"], det["tvec_rgb"], MARKER_SIZE_METERS)
        cx, cy = det["center"]
        cv2.circle(out, (cx, cy), 4, (0, 255, 255), -1)
        if det["depth_m"] > 0:
            cv2.putText(out,
                        f"{det['tag_id']}:{det['depth_m']*1000:.0f}mm",
                        (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        if "T_ik" in det:
            k_positions.append(det["T_ik"][0:3, 3:4])

    if gp is not None:
        r_m = np.asarray(gp["position"], dtype=np.float64) / 1000.0
        rvec_k, _ = cv2.Rodrigues(np.asarray(gp["rotation"], dtype=np.float64))
        cv2.drawFrameAxes(out, CAMERA_MATRIX, DIST_COEFFS,
                          rvec_k, r_m.reshape(3, 1), MARKER_SIZE_METERS * 1.5)
        label = f"Tags: {len(k_positions)}" + (" (smoothed)" if SMOOTH_GRIPPER_POSE else "")
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


# ==========================================
# --- PARALLEL WORKER FUNCTIONS (module-level for multiprocessing) ---
# ==========================================

def _gcam_worker(rgb_path, ts_arr, crop_sf, crop_ef,
                 per_frame_dets, gripper_poses,
                 fps, out_path, human_pose, model_path):
    """
    Process A (gripper camera): MediaPipe + ArUco annotation → output video.
    Runs in a separate process.
    """
    import cv2
    import numpy as np
    import mediapipe as mpipe
    from joint_tracker import PoseTracker
    from posEstimate.old_scripts.human_pose import _draw_pose, _draw_vis_legend

    BaseOptions           = mpipe.tasks.BaseOptions
    PoseLandmarker        = mpipe.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mpipe.tasks.vision.PoseLandmarkerOptions
    RunningMode           = mpipe.tasks.vision.RunningMode

    n_frames   = crop_ef - crop_sf
    cap        = cv2.VideoCapture(str(rgb_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, crop_sf)
    writer     = None
    t0_ms      = None
    prev_ts    = None
    tracker    = PoseTracker(alpha=0.35, dead_band=3.0) if human_pose else None
    written    = 0
    crop_idx   = 0

    mp_options = None
    if human_pose:
        mp_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    lm_ctx = PoseLandmarker.create_from_options(mp_options) if human_pose else contextlib.nullcontext(None)

    with lm_ctx as landmarker:
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            stamp = float(ts_arr[crop_idx])
            h_f, w_f = frame.shape[:2]
            out_frame = frame.copy()

            if landmarker is not None:
                if t0_ms is None:
                    t0_ms = int(stamp * 1000)
                ts_ms  = max(0, int(stamp * 1000) - t0_ms)
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mpipe.Image(image_format=mpipe.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, ts_ms)
                dt     = (stamp - prev_ts) if prev_ts is not None else 0.0
                prev_ts = stamp
                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    pos = np.array([[lm.x * w_f, lm.y * h_f, lm.z * w_f] for lm in lms])
                    vis = np.array([lm.visibility for lm in lms])
                    tracked = tracker.update(pos, vis, dt)
                    _draw_pose(out_frame, lms, tracked, h_f, w_f)
                else:
                    cv2.putText(out_frame, "No pose", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    tracker.update(np.zeros((33, 3)), np.zeros(33), dt)
                _draw_vis_legend(out_frame, h_f, w_f)

            dets = per_frame_dets[crop_idx] if crop_idx < len(per_frame_dets) else []
            gp   = gripper_poses[crop_idx]  if crop_idx < len(gripper_poses)  else None
            _annotate_gcam_frame(out_frame, dets, gp)

            cv2.putText(out_frame, f"Frame: {crop_sf + crop_idx}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_f, h_f)
                )
            writer.write(out_frame)
            crop_idx += 1
            written  += 1
            if written % 200 == 0:
                print(f"  [gcam]  {written}/{n_frames} frames written")

    cap.release()
    if writer:
        writer.release()
    print(f"  [gcam]  Done: {written} frames → {out_path}")


def _other_worker(rgb_path, ts_arr, crop_sf, crop_ef,
                  fps, out_path, lm_save_path,
                  human_pose, model_path):
    """
    Process B (patient camera): MediaPipe annotation → output video + landmarks.
    Saves landmarks as (N, 33, 4) float32 array [x, y, z, visibility]; NaN = no detection.
    Runs in a separate process.
    """
    import cv2
    import numpy as np
    import mediapipe as mpipe
    from joint_tracker import PoseTracker
    from posEstimate.old_scripts.human_pose import _draw_pose, _draw_vis_legend

    BaseOptions           = mpipe.tasks.BaseOptions
    PoseLandmarker        = mpipe.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mpipe.tasks.vision.PoseLandmarkerOptions
    RunningMode           = mpipe.tasks.vision.RunningMode

    n_frames = crop_ef - crop_sf
    cap      = cv2.VideoCapture(str(rgb_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, crop_sf)
    writer   = None
    t0_ms    = None
    prev_ts  = None
    tracker  = PoseTracker(alpha=0.35, dead_band=3.0) if human_pose else None
    written  = 0
    crop_idx = 0

    lm_arr = np.full((n_frames, 33, 4), np.nan, dtype=np.float32)

    mp_options = None
    if human_pose:
        mp_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    lm_ctx = PoseLandmarker.create_from_options(mp_options) if human_pose else contextlib.nullcontext(None)

    with lm_ctx as landmarker:
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            stamp = float(ts_arr[crop_idx])
            h_f, w_f = frame.shape[:2]
            out_frame = frame.copy()

            if landmarker is not None:
                if t0_ms is None:
                    t0_ms = int(stamp * 1000)
                ts_ms  = max(0, int(stamp * 1000) - t0_ms)
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mpipe.Image(image_format=mpipe.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, ts_ms)
                dt     = (stamp - prev_ts) if prev_ts is not None else 0.0
                prev_ts = stamp
                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    pos = np.array([[lm.x * w_f, lm.y * h_f, lm.z * w_f] for lm in lms])
                    vis = np.array([lm.visibility for lm in lms])
                    tracked = tracker.update(pos, vis, dt)
                    _draw_pose(out_frame, lms, tracked, h_f, w_f)
                    for j, lm in enumerate(lms):
                        lm_arr[crop_idx, j] = [lm.x, lm.y, lm.z, lm.visibility]
                else:
                    cv2.putText(out_frame, "No pose", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    tracker.update(np.zeros((33, 3)), np.zeros(33), dt)
                _draw_vis_legend(out_frame, h_f, w_f)

            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_f, h_f)
                )
            writer.write(out_frame)
            crop_idx += 1
            written  += 1
            if written % 200 == 0:
                print(f"  [other] {written}/{n_frames} frames written")

    cap.release()
    if writer:
        writer.release()

    np.save(str(lm_save_path), lm_arr[:crop_idx])   # trim to actual frames written
    print(f"  [other] Done: {written} frames → {out_path}")
    print(f"  [other] Landmarks saved → {lm_save_path}")


# ==========================================
# --- TRAJECTORY POST-PROCESSING ---
# ==========================================

RETURN_THRESH_MM = 30.0   # Z-distance from start to count as "returned"
MIN_CYCLE_SEC    = 6.0    # minimum cycle duration — ignores brief returns


def _ensure_quat_continuity(quats):
    """Flip quaternions so consecutive ones are on the same hemisphere."""
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    return quats


def _interpolate_full(t, frames, pos, euler, fps):
    """
    Interpolate sparse CSV (missing frames due to ArUco occlusion) into a
    complete trajectory with one row per video frame.
    Uses cubic spline (C2 smooth) instead of linear interp to avoid
    velocity discontinuities at gap boundaries.
    """
    f_min, f_max = int(frames[0]), int(frames[-1])
    all_frames = np.arange(f_min, f_max + 1)
    all_t = all_frames / fps

    # Cubic spline for position — C2 continuous across gaps
    full_pos = np.zeros((len(all_frames), 3))
    for ax in range(3):
        cs = CubicSpline(frames, pos[:, ax], bc_type="clamped")
        full_pos[:, ax] = cs(all_frames)

    rots = ScipyRotation.from_euler("xyz", euler)
    quats = rots.as_quat()
    quats = _ensure_quat_continuity(quats)
    rots = ScipyRotation.from_quat(quats)
    slerp = Slerp(frames, rots)
    full_euler = slerp(all_frames).as_euler("xyz")

    n_filled = len(all_frames) - len(frames)
    print(f"  Interpolated: {len(frames)} → {len(all_frames)} frames "
          f"({n_filled} filled)")
    return all_t, all_frames.astype(float), full_pos, full_euler


def _detect_cycles(t, pos, thresh_mm, min_sec):
    """
    Find cycle start indices by detecting when the gripper height (Z)
    returns close to its starting height.
    """
    start_z = pos[0, 2]
    dist = np.abs(pos[:, 2] - start_z)
    close_mask = dist < thresh_mm

    starts = [0]
    in_close = False
    region_best_i = None
    region_best_d = np.inf

    for i in range(1, len(t)):
        if close_mask[i]:
            if not in_close:
                in_close = True
                region_best_i = i
                region_best_d = dist[i]
            elif dist[i] < region_best_d:
                region_best_i = i
                region_best_d = dist[i]
        else:
            if in_close:
                if (t[region_best_i] - t[starts[-1]]) >= min_sec:
                    starts.append(region_best_i)
                in_close = False
                region_best_i = None
                region_best_d = np.inf

    if in_close and region_best_i is not None:
        if (t[region_best_i] - t[starts[-1]]) >= min_sec:
            starts.append(region_best_i)

    return starts


def _resample_pos(t_in, pos_in, t_out):
    out = np.zeros((len(t_out), 3))
    for ax in range(3):
        out[:, ax] = np.interp(t_out, t_in, pos_in[:, ax])
    return out


def _resample_orient(t_in, euler_in, t_out):
    rots = ScipyRotation.from_euler("xyz", euler_in)
    quats = rots.as_quat()
    quats = _ensure_quat_continuity(quats)
    rots = ScipyRotation.from_quat(quats)
    slerp = Slerp(t_in, rots)
    return slerp(t_out).as_euler("xyz")


def _average_orientations(euler_stack):
    K, N, _ = euler_stack.shape
    quat_stack = np.zeros((K, N, 4))
    for k in range(K):
        q = ScipyRotation.from_euler("xyz", euler_stack[k]).as_quat()
        q = _ensure_quat_continuity(q)
        if k > 0 and np.dot(q[0], quat_stack[0, 0]) < 0:
            q = -q
        quat_stack[k] = q
    mean_q = quat_stack.mean(axis=0)
    mean_q /= np.linalg.norm(mean_q, axis=1, keepdims=True)
    return ScipyRotation.from_quat(mean_q).as_euler("xyz")


def _write_traj_csv(path, t, frames, pos, euler):
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
    print(f"  Saved: {path}")


CYCLE_COLORS = [
    (0, 255, 0),    # green
    (0, 200, 255),  # orange
    (255, 100, 0),  # blue
    (200, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]
FLASH_FRAMES = 15


def _annotate_video_cycles(video_path, cycle_start_frames):
    """
    Re-read the annotated video, overlay cycle markers, and overwrite it.
    cycle_start_frames: dict mapping video frame index → cycle number.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open video for cycle annotation: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write to a temp file, then replace original
    tmp_path = Path(video_path).with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    current_cycle = 0
    flash_remaining = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in cycle_start_frames:
            current_cycle = cycle_start_frames[frame_idx]
            flash_remaining = FLASH_FRAMES

        if current_cycle > 0:
            color = CYCLE_COLORS[(current_cycle - 1) % len(CYCLE_COLORS)]

            # Colored bar at top
            cv2.rectangle(frame, (0, 0), (w, 6), color, -1)

            # Persistent label
            cv2.putText(frame, f"Cycle {current_cycle}", (w - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Flash banner
            if flash_remaining > 0:
                banner = f"CYCLE {current_cycle} START"
                ts = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                tx = (w - ts[0]) // 2
                ty = h // 2
                overlay = frame.copy()
                cv2.rectangle(overlay, (tx - 15, ty - ts[1] - 15),
                              (tx + ts[0] + 15, ty + 15), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, banner, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                flash_remaining -= 1

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Replace original with annotated version
    import shutil
    shutil.move(str(tmp_path), str(video_path))
    print(f"  Annotated video with cycle markers: {video_path}  ({frame_idx} frames)")


def _read_sparse_csv(sparse_csv_path):
    """Read sparse gripper CSV and return (t, frame, pos, euler) arrays."""
    rows = []
    with open(sparse_csv_path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    t     = np.array([float(r["t"])       for r in rows])
    frame = np.array([int(float(r["frame"])) for r in rows])
    pos   = np.array([[float(r["pos_x"]), float(r["pos_y"]),
                        float(r["pos_z"])]  for r in rows])
    euler = np.array([[float(r["orient_x"]), float(r["orient_y"]),
                        float(r["orient_z"])] for r in rows])
    return t, frame, pos, euler


MIRROR_DWELL_SEC = 0.5  # pause at the turnaround point (seconds)


def _mirror_trajectory(t, frames, pos, euler, fps):
    """Append a time-reversed copy so the trajectory goes A→B→A.

    Inserts a short dwell (MIRROR_DWELL_SEC) at the turnaround point so
    the cubic spline in arm_optimo.py sees a flat segment and naturally
    produces zero velocity at the junction — no discontinuous motion.
    """
    dt_frame = 1.0 / fps
    n_dwell  = max(1, int(MIRROR_DWELL_SEC * fps))

    # Dwell: repeat the endpoint for n_dwell frames
    dwell_t      = t[-1] + dt_frame * np.arange(1, n_dwell + 1)
    dwell_frames = frames[-1] + np.arange(1, n_dwell + 1).astype(float)
    dwell_pos    = np.tile(pos[-1], (n_dwell, 1))
    dwell_euler  = np.tile(euler[-1], (n_dwell, 1))

    # Reversed segment (skip first point = same as last dwell point)
    t_after_dwell = dwell_t[-1]
    t_rev     = t_after_dwell + (t[-1] - t[::-1]) + dt_frame
    f_rev     = dwell_frames[-1] + np.arange(1, len(t) + 1).astype(float)
    pos_rev   = pos[::-1].copy()
    euler_rev = euler[::-1].copy()

    t_out     = np.concatenate([t,      dwell_t,      t_rev[1:]])
    f_out     = np.concatenate([frames, dwell_frames,  f_rev[1:]])
    pos_out   = np.concatenate([pos,    dwell_pos,     pos_rev[1:]], axis=0)
    euler_out = np.concatenate([euler,  dwell_euler,   euler_rev[1:]], axis=0)
    print(f"  Mirror: {len(t)} → {len(t_out)} frames "
          f"(dwell={MIRROR_DWELL_SEC}s, duration {t_out[-1]:.2f}s)")
    return t_out, f_out, pos_out, euler_out


def interpolate_trajectory(sparse_csv_path, fps, out_dir, data_name, mirror=False):
    """
    Interpolate sparse CSV to one row per frame, optionally mirror,
    and save as <data_name>.traj.
    """
    t, frame, pos, euler = _read_sparse_csv(sparse_csv_path)
    t_full, frames_full, pos_full, euler_full = _interpolate_full(
        t, frame, pos, euler, fps)

    if mirror:
        t_full, frames_full, pos_full, euler_full = _mirror_trajectory(
            t_full, frames_full, pos_full, euler_full, fps)

    main_path = Path(out_dir) / f"{data_name}.csv"
    _write_traj_csv(main_path, t_full, frames_full, pos_full, euler_full)


def average_trajectory(sparse_csv_path, fps, out_dir, data_name, video_path=None):
    """
    Read the sparse gripper CSV, interpolate to full, detect cycles,
    average them, save full as <data_name>_full.csv and averaged as
    <data_name>.csv, and annotate the video with cycle markers.
    """
    t, frame, pos, euler = _read_sparse_csv(sparse_csv_path)

    # Interpolate to one row per video frame
    t_full, frames_full, pos_full, euler_full = _interpolate_full(
        t, frame, pos, euler, fps)

    # Save full trajectory
    full_path = Path(out_dir) / f"{data_name}_full.csv"
    _write_traj_csv(full_path, t_full, frames_full, pos_full, euler_full)

    # Detect cycles
    starts = _detect_cycles(t_full, pos_full, RETURN_THRESH_MM, MIN_CYCLE_SEC)

    print(f"\n  Detected {len(starts)} cycles "
          f"(thresh={RETURN_THRESH_MM} mm, min_cycle={MIN_CYCLE_SEC} s)")
    print(f"  {'Cycle':<8} {'t (s)':<10} {'frame':<8} {'pos_z (mm)':<12}")
    print(f"  {'-'*40}")
    for ci, si in enumerate(starts):
        print(f"    {ci+1:<6} {t_full[si]:<10.2f} {int(frames_full[si]):<8} "
              f"{pos_full[si, 2]:<12.2f}")

    if len(starts) < 2:
        print("  Only 1 cycle detected — saving full trajectory as the main traj")
        avg_path = Path(out_dir) / f"{data_name}.csv"
        _write_traj_csv(avg_path, t_full, frames_full, pos_full, euler_full)
        return

    # Annotate video with cycle markers
    if video_path is not None and Path(video_path).exists():
        cycle_frame_map = {}
        for ci, si in enumerate(starts):
            cycle_frame_map[int(frames_full[si])] = ci + 1
        _annotate_video_cycles(video_path, cycle_frame_map)

    segments_pos, segments_euler, segments_dur = [], [], []
    for i in range(len(starts) - 1):
        i0 = starts[i]
        i1 = starts[i + 1]
        t_seg     = t_full[i0:i1] - t_full[i0]
        pos_seg   = pos_full[i0:i1].copy()
        euler_seg = euler_full[i0:i1].copy()
        pos_seg  -= pos_seg[0]
        segments_pos.append((t_seg, pos_seg))
        segments_euler.append((t_seg, euler_seg))
        segments_dur.append(t_seg[-1])

    n_cycles = len(segments_pos)
    print(f"  Using {n_cycles} complete cycles (dropped last incomplete segment)")
    print(f"  Durations: " + ", ".join(f"{d:.2f}s" for d in segments_dur))

    # Resample to common length (shortest duration)
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

    # Average
    pos_mean   = pos_stack.mean(axis=0)
    euler_mean = _average_orientations(euler_stack)

    # Save averaged trajectory as the main traj
    avg_path = Path(out_dir) / f"{data_name}.csv"
    avg_frames = np.arange(len(t_common), dtype=float)
    _write_traj_csv(avg_path, t_common, avg_frames, pos_mean, euler_mean)
    print(f"  End position (mm): {pos_mean[-1].round(2)}")
    print(f"  Averaged {n_cycles} cycles → {avg_path}")


# ==========================================
# --- TRAJECTORY VISUALIZATION ---
# ==========================================

def _extract_cam_frame_trajectory(gripper_poses, timestamps):
    """Extract position (mm) and time arrays from gripper_poses list (camera frame)."""
    t_list, pos_list = [], []
    t0 = None
    for i, p in enumerate(gripper_poses):
        if p is None:
            continue
        if t0 is None:
            t0 = float(timestamps[i])
        t_list.append(float(timestamps[i]) - t0)
        pos_list.append(p["position"].copy())  # already in mm, camera frame
    return np.array(t_list), np.array(pos_list)


def plot_3d_trajectory(gripper_poses, timestamps, gripper_poses_raw=None):
    """3D plot of gripper trajectory in camera frame with camera origin."""
    t, pos = _extract_cam_frame_trajectory(gripper_poses, timestamps)
    if len(pos) < 2:
        print("  Not enough poses for 3D plot.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # If raw poses provided and different from main, show both
    if gripper_poses_raw is not None and gripper_poses_raw is not gripper_poses:
        t_raw, pos_raw = _extract_cam_frame_trajectory(gripper_poses_raw, timestamps)
        if len(pos_raw) >= 2:
            ax.plot(pos_raw[:, 0], pos_raw[:, 1], pos_raw[:, 2],
                    color="gray", linewidth=0.6, alpha=0.5, label="Raw")
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", linewidth=1.2, label="Smoothed")
    else:
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", linewidth=1.2, label="Trajectory")

    ax.scatter(*pos[0], c="green", s=80, marker="o", label="Start")
    ax.scatter(*pos[-1], c="red", s=80, marker="x", label="End")

    # Draw camera at origin
    ax.scatter(0, 0, 0, c="black", s=120, marker="^", label="Camera")
    cam_len = 30.0  # mm
    ax.quiver(0, 0, 0, cam_len, 0, 0, color="r", arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, cam_len, 0, color="g", arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, 0, cam_len, color="b", arrow_length_ratio=0.15)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"{DATA_NAME} — Gripper trajectory (camera frame)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_xyz_vs_time(gripper_poses, timestamps, gripper_poses_raw=None):
    """3-subplot figure: X, Y, Z position vs time in initial gripper frame."""
    t, pos = _extract_cam_frame_trajectory(gripper_poses, timestamps)
    if len(pos) < 2:
        print("  Not enough poses for XYZ plot.")
        return

    # Transform into initial gripper frame (R0, t0)
    R0 = None
    for p in gripper_poses:
        if p is not None:
            R0 = p["rotation"]
            break
    t0 = pos[0]
    pos_gripper = np.array([R0.T @ (p - t0) for p in pos])

    # Raw poses (if smoothing is on, overlay raw underneath)
    has_raw = (gripper_poses_raw is not None
               and gripper_poses_raw is not gripper_poses)
    if has_raw:
        t_raw, pos_raw_cam = _extract_cam_frame_trajectory(gripper_poses_raw, timestamps)
        if len(pos_raw_cam) >= 2:
            pos_raw_gripper = np.array([R0.T @ (p - t0) for p in pos_raw_cam])
        else:
            has_raw = False

    labels = ("X", "Y", "Z")
    colors = ("tab:red", "tab:green", "tab:blue")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for idx, ax in enumerate(axes):
        if has_raw:
            ax.plot(t_raw, pos_raw_gripper[:, idx], color="gray", linewidth=0.6,
                    alpha=0.5, label=f"{labels[idx]} raw")
            ax.plot(t, pos_gripper[:, idx], color=colors[idx], linewidth=1.2,
                    label=f"{labels[idx]} smoothed")
        else:
            ax.plot(t, pos_gripper[:, idx], color=colors[idx], linewidth=1.2,
                    label=f"{labels[idx]}")
        ax.set_ylabel(f"{labels[idx]} (mm)")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{DATA_NAME} — Gripper position (initial gripper frame)")
    plt.tight_layout()
    plt.show()


def plot_per_marker_pose(per_frame_dets, timestamps):
    """Plot each ArUco marker's position (relative to its own first detection) vs time.
    3 subplots (X, Y, Z), each with one line per marker ID."""
    # Collect per-marker trajectories
    marker_data = {}  # tag_id -> (times, positions_mm)
    t0 = None
    for i, dets in enumerate(per_frame_dets):
        for det in dets:
            if "t_ij_mm" not in det:
                continue
            tid = det["tag_id"]
            if t0 is None:
                t0 = float(timestamps[0])
            if tid not in marker_data:
                marker_data[tid] = {"t": [], "pos": [], "pos0": det["t_ij_mm"].copy()}
            marker_data[tid]["t"].append(float(timestamps[i]) - t0)
            marker_data[tid]["pos"].append(det["t_ij_mm"] - marker_data[tid]["pos0"])

    if not marker_data:
        print("  No marker data for per-marker plot.")
        return

    labels = ("X", "Y", "Z")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{DATA_NAME} — Per-marker position (relative to own start)")
    for ax_idx, ax in enumerate(axes):
        for tid in sorted(marker_data.keys()):
            t_arr = np.array(marker_data[tid]["t"])
            pos_arr = np.array(marker_data[tid]["pos"])
            ax.plot(t_arr, pos_arr[:, ax_idx], linewidth=0.8,
                    label=f"Tag {tid}", alpha=0.8)
        ax.set_ylabel(f"{labels[ax_idx]} (mm)")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_per_marker_gripper(per_frame_dets, timestamps, gripper_poses):
    """Plot gripper position estimated by each marker independently.
    Transforms each marker's T_ik into the initial gripper frame.
    3 subplots (X, Y, Z), one line per marker ID."""
    # Find initial gripper frame (R0, t0) from first valid fused pose
    R0, t0_pos = None, None
    for p in gripper_poses:
        if p is not None:
            R0 = p["rotation"]
            t0_pos = p["position"]
            break
    if R0 is None:
        print("  No valid gripper pose for per-marker gripper plot.")
        return

    # Collect per-marker gripper estimates
    marker_data = {}  # tag_id -> (times, gripper_pos_in_initial_frame)
    t0 = float(timestamps[0])
    for i, dets in enumerate(per_frame_dets):
        for det in dets:
            if "T_ik" not in det:
                continue
            tid = det["tag_id"]
            gripper_mm = det["T_ik"][0:3, 3] * 1000.0
            pos_rel = R0.T @ (gripper_mm - t0_pos)
            if tid not in marker_data:
                marker_data[tid] = {"t": [], "pos": []}
            marker_data[tid]["t"].append(float(timestamps[i]) - t0)
            marker_data[tid]["pos"].append(pos_rel)

    if not marker_data:
        print("  No marker data for per-marker gripper plot.")
        return

    labels = ("X", "Y", "Z")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{DATA_NAME} — Gripper estimate by each marker (initial gripper frame)")
    for ax_idx, ax in enumerate(axes):
        for tid in sorted(marker_data.keys()):
            t_arr = np.array(marker_data[tid]["t"])
            pos_arr = np.array(marker_data[tid]["pos"])
            ax.plot(t_arr, pos_arr[:, ax_idx], linewidth=0.8,
                    label=f"Tag {tid}", alpha=0.8)
        ax.set_ylabel(f"{labels[ax_idx]} (mm)")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_debug_marker3(per_frame_dets, timestamps):
    """Decompose marker 3's gripper estimate to find the Z-drift source.

    Gripper position = R_ij @ t_tag + t_ij
      - t_ij:          marker translation in camera frame (from depth or PnP)
      - R_ij @ t_tag:  rotation-amplified offset from marker to gripper

    Plots:
      Row 1: raw inputs — u (px), v (px), depth (mm)
      Row 2: marker translation t_ij (mm, relative)
      Row 3: PnP tvec_rgb (mm, relative) — for comparison with depth-based t_ij
      Row 4: R_ij euler angles (deg)
      Row 5: R_ij @ t_tag (mm, relative)
      Row 6: final gripper T_ik (mm, relative)
    """
    TAG_ID = 3
    t_tag = TAG_TRANSFORMS[TAG_ID][0:3, 3:4]  # (3,1) in meters

    times = []
    u_list, v_list, depth_list = [], [], []
    t_ij_list, tvec_pnp_list = [], []
    euler_list, rot_contrib_list, gripper_list = [], [], []
    t0 = float(timestamps[0])
    first = True
    t_ij_0, tvec_pnp_0, rot_contrib_0, gripper_0 = None, None, None, None

    for i, dets in enumerate(per_frame_dets):
        for det in dets:
            if det["tag_id"] != TAG_ID or "T_ik" not in det:
                continue

            R_ij = det["R_ij"]
            t_ik = det["T_ik"][0:3, 3:4]           # (3,1) meters
            rot_contrib = R_ij @ t_tag               # (3,1) meters
            t_ij = t_ik - rot_contrib                # (3,1) meters

            t_ij_mm = t_ij.flatten() * 1000.0
            tvec_pnp_mm = np.asarray(det["tvec_rgb"], dtype=np.float64).flatten() * 1000.0
            rot_contrib_mm = rot_contrib.flatten() * 1000.0
            gripper_mm = t_ik.flatten() * 1000.0
            euler = ScipyRotation.from_matrix(R_ij).as_euler("xyz", degrees=True)

            cx, cy = det["center"]
            depth_mm = det["depth_m"] * 1000.0

            if first:
                t_ij_0 = t_ij_mm.copy()
                tvec_pnp_0 = tvec_pnp_mm.copy()
                rot_contrib_0 = rot_contrib_mm.copy()
                gripper_0 = gripper_mm.copy()
                first = False

            times.append(float(timestamps[i]) - t0)
            u_list.append(cx)
            v_list.append(cy)
            depth_list.append(depth_mm)
            t_ij_list.append(t_ij_mm - t_ij_0)
            tvec_pnp_list.append(tvec_pnp_mm - tvec_pnp_0)
            euler_list.append(euler)
            rot_contrib_list.append(rot_contrib_mm - rot_contrib_0)
            gripper_list.append(gripper_mm - gripper_0)

    if not times:
        print("  No marker 3 detections for debug plot.")
        return

    times = np.array(times)
    u_arr = np.array(u_list)
    v_arr = np.array(v_list)
    depth_arr = np.array(depth_list)
    t_ij_arr = np.array(t_ij_list)
    tvec_pnp_arr = np.array(tvec_pnp_list)
    euler_arr = np.array(euler_list)
    rot_arr = np.array(rot_contrib_list)
    grip_arr = np.array(gripper_list)

    fig, axes = plt.subplots(6, 3, figsize=(16, 18), sharex=True)
    fig.suptitle(f"{DATA_NAME} — Marker 3 full debug", fontsize=13)

    # Row 0: raw inputs (u, v, depth)
    axes[0][0].plot(times, u_arr, linewidth=0.8, color="purple")
    axes[0][0].set_ylabel("Raw inputs", fontsize=8)
    axes[0][0].set_title("u (px)")
    axes[0][1].plot(times, v_arr, linewidth=0.8, color="purple")
    axes[0][1].set_title("v (px)")
    axes[0][2].plot(times, depth_arr, linewidth=0.8, color="purple")
    axes[0][2].set_title("depth (mm)")

    # Rows 1-5: decomposition
    row_titles = [
        "t_ij depth-based (mm, rel)",
        "tvec PnP (mm, rel)",
        "R_ij Euler (deg)",
        "R_ij @ t_tag (mm, rel)",
        "Gripper T_ik (mm, rel)",
    ]
    data_rows = [t_ij_arr, tvec_pnp_arr, euler_arr, rot_arr, grip_arr]
    axis_labels = ("X", "Y", "Z")

    for row_i, (title, data) in enumerate(zip(row_titles, data_rows), start=1):
        for col in range(3):
            ax = axes[row_i][col]
            ax.plot(times, data[:, col], linewidth=0.8)
            ax.grid(True, linewidth=0.4, alpha=0.5)
            if col == 0:
                ax.set_ylabel(title, fontsize=8)
            ax.set_title(f"{axis_labels[col]}", fontsize=9)

    for ax in axes[0]:
        ax.grid(True, linewidth=0.4, alpha=0.5)
    for col in range(3):
        axes[-1][col].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\n  Marker 3 debug summary:")
    print(f"  {'Raw input':<20} {'min':>10} {'max':>10} {'range':>10}")
    print(f"  {'-'*52}")
    print(f"  {'u (px)':<20} {u_arr.min():>10.1f} {u_arr.max():>10.1f} {u_arr.max()-u_arr.min():>10.1f}")
    print(f"  {'v (px)':<20} {v_arr.min():>10.1f} {v_arr.max():>10.1f} {v_arr.max()-v_arr.min():>10.1f}")
    print(f"  {'depth (mm)':<20} {depth_arr.min():>10.1f} {depth_arr.max():>10.1f} {depth_arr.max()-depth_arr.min():>10.1f}")
    n_no_depth = np.sum(depth_arr == 0)
    print(f"  Frames with no depth: {n_no_depth}/{len(depth_arr)}")
    print()
    print(f"  {'Component':<30} {'X range':>10} {'Y range':>10} {'Z range':>10} mm")
    print(f"  {'-'*62}")
    for name, arr in [("t_ij (depth-based)", t_ij_arr),
                       ("tvec PnP", tvec_pnp_arr),
                       ("R_ij @ t_tag (rot contrib)", rot_arr),
                       ("T_ik (gripper total)", grip_arr)]:
        ranges = arr.max(axis=0) - arr.min(axis=0)
        print(f"  {name:<30} {ranges[0]:>10.1f} {ranges[1]:>10.1f} {ranges[2]:>10.1f}")

    # Save debug data to CSV
    debug_csv_path = OUT_DIR / f"{DATA_NAME}_marker3_debug.csv"
    with open(debug_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t", "u_px", "v_px", "depth_mm",
            "t_ij_x", "t_ij_y", "t_ij_z",
            "tvec_pnp_x", "tvec_pnp_y", "tvec_pnp_z",
            "euler_x_deg", "euler_y_deg", "euler_z_deg",
            "rot_contrib_x", "rot_contrib_y", "rot_contrib_z",
            "gripper_x", "gripper_y", "gripper_z",
        ])
        for i in range(len(times)):
            w.writerow([
                round(float(times[i]), 6),
                int(u_arr[i]), int(v_arr[i]), round(float(depth_arr[i]), 1),
                round(float(t_ij_arr[i, 0]), 3), round(float(t_ij_arr[i, 1]), 3), round(float(t_ij_arr[i, 2]), 3),
                round(float(tvec_pnp_arr[i, 0]), 3), round(float(tvec_pnp_arr[i, 1]), 3), round(float(tvec_pnp_arr[i, 2]), 3),
                round(float(euler_arr[i, 0]), 4), round(float(euler_arr[i, 1]), 4), round(float(euler_arr[i, 2]), 4),
                round(float(rot_arr[i, 0]), 3), round(float(rot_arr[i, 1]), 3), round(float(rot_arr[i, 2]), 3),
                round(float(grip_arr[i, 0]), 3), round(float(grip_arr[i, 1]), 3), round(float(grip_arr[i, 2]), 3),
            ])
    print(f"  Saved debug CSV: {debug_csv_path}")


# ==========================================
# --- MAIN PROCESSOR ---
# ==========================================

class TwoCamProcessor:
    def __init__(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if not DATA_DIR.exists():
            raise FileNotFoundError(
                f"Data directory not found: {DATA_DIR}\n"
                "Run read_rosbag.py first to extract the bag."
            )
        if HUMAN_POSE and not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {MODEL_PATH}\n"
                "Download pose_landmarker_full.task to posEstimate/data/"
            )

        gcam  = "left"  if GRIPPER_CAM == "left" else "right"
        other = "right" if GRIPPER_CAM == "left" else "left"

        self._gcam_rgb    = DATA_DIR / f"{gcam}.mp4"
        self._other_rgb   = DATA_DIR / f"{other}.mp4"
        self._gcam_ts     = np.load(str(DATA_DIR / f"{gcam}_ts.npy"))
        self._other_ts    = np.load(str(DATA_DIR / f"{other}_ts.npy"))
        self._gcam_depth_dir = DATA_DIR / f"{gcam}_depth"
        self._gcam_depth_ts  = (
            np.load(str(DATA_DIR / f"{gcam}_depth_ts.npy"))
            if (DATA_DIR / f"{gcam}_depth_ts.npy").exists() else np.array([])
        )
        self._other_cam_name = other

        aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.adaptiveThreshWinSizeMin = 3
        aruco_params.adaptiveThreshWinSizeMax = 53
        aruco_params.adaptiveThreshWinSizeStep = 4
        aruco_params.adaptiveThreshConstant = 7
        aruco_params.minMarkerPerimeterRate = 0.01
        aruco_params.maxMarkerPerimeterRate = 4.0
        aruco_params.polygonalApproxAccuracyRate = 0.05
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        aruco_params.cornerRefinementWinSize = 5
        aruco_params.cornerRefinementMaxIterations = 50
        aruco_params.cornerRefinementMinAccuracy = 0.01
        self._aruco_det = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        self._prev_rvec = {}  # tag_id -> (rvec, tvec) for PnP temporal consistency

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self):
        gcam_ts  = self._gcam_ts
        other_ts = self._other_ts

        # Apply crop to both cameras independently
        sf_gcam,  ef_gcam  = _crop_frame_range(gcam_ts,  CROP, CROP_UNIT)
        sf_other, ef_other = _crop_frame_range(other_ts, CROP, CROP_UNIT)
        gcam_ts  = gcam_ts[sf_gcam:ef_gcam]
        other_ts = other_ts[sf_other:ef_other]
        print(f"  Gripper cam: {len(gcam_ts)} frames  |  Other cam: {len(other_ts)} frames")

        fps_gcam  = fps_from_timestamps(gcam_ts)  if len(gcam_ts)  >= 2 else 30.0
        fps_other = fps_from_timestamps(other_ts) if len(other_ts) >= 2 else 30.0
        print(f"  FPS — gcam: {fps_gcam:.3f}  other: {fps_other:.3f}")

        # ── Force data: crop by same time window in seconds ───────────
        force_cropped = self._load_and_crop_force()

        # ── Pass 1: ArUco detection (sequential, fast) ────────────────
        print("Pass 1: ArUco detection on gripper camera...")
        per_frame_dets_all = self._pass1_aruco(gcam_ts, sf_gcam)

        gripper_poses_raw = self._build_gripper_poses(per_frame_dets_all)
        gripper_poses     = self._smooth_gripper_poses(gripper_poses_raw)

        # ── Visualize gripper trajectory in camera frame ────────────────
        plot_3d_trajectory(gripper_poses, gcam_ts, gripper_poses_raw)
        plot_xyz_vs_time(gripper_poses, gcam_ts, gripper_poses_raw)

        # ── Per-marker diagnostic plots ───────────────────────────────
        plot_per_marker_pose(per_frame_dets_all, gcam_ts)
        plot_per_marker_gripper(per_frame_dets_all, gcam_ts, gripper_poses)
        plot_debug_marker3(per_frame_dets_all, gcam_ts)

        sparse_csv = self._save_csv(gripper_poses, gcam_ts)

        # ── Pass 2: MediaPipe annotation, two cameras in parallel ─────
        print("Pass 2: Parallel MediaPipe annotation...")
        stem = DATA_NAME
        if GRIPPER_CAM == "left":
            gcam_out_path  = OUT_DIR / f"{stem}_left.mp4"
            other_out_path = OUT_DIR / f"{stem}_right.mp4"
        else:
            gcam_out_path  = OUT_DIR / f"{stem}_right.mp4"
            other_out_path = OUT_DIR / f"{stem}_left.mp4"

        lm_tmp_path = OUT_DIR / "_tmp_landmarks.npy"

        # Determine other-cam frame dimensions for rule book
        other_cap = cv2.VideoCapture(str(self._other_rgb))
        other_hw  = (int(other_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(other_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        other_cap.release()

        p_gcam = mp.Process(
            target=_gcam_worker,
            args=(
                str(self._gcam_rgb),
                gcam_ts, sf_gcam, ef_gcam,
                per_frame_dets_all, gripper_poses,
                fps_gcam, str(gcam_out_path),
                False, MODEL_PATH,
            ),
        )
        p_other = mp.Process(
            target=_other_worker,
            args=(
                str(self._other_rgb),
                other_ts, sf_other, ef_other,
                fps_other, str(other_out_path),
                str(lm_tmp_path),
                HUMAN_POSE, MODEL_PATH,
            ),
        )

        p_gcam.start()
        p_other.start()
        p_gcam.join()
        p_other.join()

        if p_gcam.exitcode != 0:
            print(f"  WARNING: gcam worker exited with code {p_gcam.exitcode}")
        if p_other.exitcode != 0:
            print(f"  WARNING: other worker exited with code {p_other.exitcode}")

        # ── Rule book (from landmarks saved by other worker) ──────────
        if HUMAN_POSE and lm_tmp_path.exists():
            print("Computing patient joint-angle rule book...")
            lm_arr = np.load(str(lm_tmp_path))
            self._compute_and_save_rule_book(lm_arr, other_hw, self._other_cam_name)
            lm_tmp_path.unlink(missing_ok=True)

        # ── Force rule book ───────────────────────────────────────────
        if force_cropped is not None and len(force_cropped) > 0:
            self._save_force_rules(force_cropped)

        # ── Interpolate + cycle-average trajectory + annotate video ───
        if sparse_csv is not None:
            if GRIPPER_PROCESS == "cycle":
                average_trajectory(sparse_csv, fps_gcam, str(OUT_DIR),
                                   DATA_NAME, str(gcam_out_path))
            else:  # "pick"
                interpolate_trajectory(sparse_csv, fps_gcam, str(OUT_DIR),
                                       DATA_NAME, mirror=IF_MIRROR)

    # ------------------------------------------------------------------
    # Pass 1: ArUco detection from video file
    # ------------------------------------------------------------------

    def _pass1_aruco(self, gcam_ts, start_frame):
        """
        Read gripper-cam video frame-by-frame, detect ArUco per frame.
        Depth is loaded from 16-bit PNG files when available.
        Returns per_frame_dets: list[list[dict]] aligned to gcam_ts.
        """
        n_frames = len(gcam_ts)
        depth_ts = self._gcam_depth_ts
        depth_dir = self._gcam_depth_dir
        has_depth = depth_dir.exists() and len(depth_ts) > 0

        cap = cv2.VideoCapture(str(self._gcam_rgb))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        per_frame_dets = []

        for crop_idx in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                per_frame_dets.append([])
                continue

            depth_img = None
            if has_depth:
                di = _find_depth_idx(depth_ts, float(gcam_ts[crop_idx]))
                if di is not None:
                    dp = depth_dir / f"{di:06d}.png"
                    if dp.exists():
                        depth_img = cv2.imread(str(dp), cv2.IMREAD_ANYDEPTH)

            dets = self._detect_aruco_frame(frame, depth_img)
            per_frame_dets.append(dets)

            if (crop_idx + 1) % 500 == 0:
                print(f"  Pass 1: {crop_idx + 1}/{n_frames} frames")

        cap.release()
        total_dets = sum(len(d) for d in per_frame_dets)
        print(f"  ArUco: {total_dets} detections across {n_frames} frames")
        return per_frame_dets

    # ------------------------------------------------------------------
    # ArUco detection (single frame)
    # ------------------------------------------------------------------

    def _detect_aruco_frame(self, color_image, depth_img):
        blur = cv2.GaussianBlur(color_image, (0, 0), sigmaX=5)
        sharp = cv2.addWeighted(color_image, 2.0, blur, -1.0, 0)
        corners, ids, _ = self._aruco_det.detectMarkers(sharp)
        frame_dets = []
        if ids is None:
            return frame_dets

        for i in range(len(ids)):
            tag_id = int(ids[i][0])
            if tag_id not in TAG_TRANSFORMS or tag_id != 3:
                continue
            marker_2d = corners[i][0].astype(np.float32)

            # Use previous frame's solution as initial guess for stability
            prev = self._prev_rvec.get(tag_id)
            if prev is not None:
                ok, rvec, tvec_rgb = cv2.solvePnP(
                    marker_3d_edges, marker_2d, CAMERA_MATRIX, DIST_COEFFS,
                    rvec=prev[0].copy(), tvec=prev[1].copy(),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            else:
                ok, rvec, tvec_rgb = cv2.solvePnP(
                    marker_3d_edges, marker_2d, CAMERA_MATRIX, DIST_COEFFS
                )
            if not ok:
                continue

            # Reject rotation flips: if rvec suddenly jumps too far from
            # previous frame, keep the previous rvec instead
            R_ij, _ = cv2.Rodrigues(rvec)
            if prev is not None:
                R_prev, _ = cv2.Rodrigues(prev[0])
                # Angle between current and previous rotation
                R_diff = R_prev.T @ R_ij
                angle_diff = abs(np.arccos(np.clip(
                    (np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0
                )))
                if np.degrees(angle_diff) > 30.0:  # max 30°/frame
                    rvec = prev[0].copy()
                    tvec_rgb = prev[1].copy()
                    R_ij, _ = cv2.Rodrigues(rvec)

            self._prev_rvec[tag_id] = (rvec.copy(), tvec_rgb.copy())
            cx = int(np.mean(marker_2d[:, 0]))
            cy = int(np.mean(marker_2d[:, 1]))
            depth_m = 0.0
            if depth_img is not None:
                if 0 <= cy < depth_img.shape[0] and 0 <= cx < depth_img.shape[1]:
                    raw_mm = depth_img[cy, cx]
                    if raw_mm > 0:
                        depth_m = float(raw_mm) / 1000.0
            frame_dets.append({
                "tag_id":       tag_id,
                "corners_draw": corners[i],
                "marker_2d":    marker_2d,
                "rvec":         rvec,
                "tvec_rgb":     tvec_rgb,
                "R_ij":         R_ij,
                "center":       (cx, cy),
                "depth_m":      depth_m,
            })
        return frame_dets

    # ------------------------------------------------------------------
    # Gripper pose computation
    # ------------------------------------------------------------------

    def _build_gripper_poses(self, per_frame_dets):
        n = len(per_frame_dets)
        poses = [None] * n
        for f_idx, frame_dets in enumerate(per_frame_dets):
            k_pos, k_rot = [], []
            for det in frame_dets:
                if det["depth_m"] > 0:
                    t_ij = deproject_pixel_depth_to_point_m(
                        det["center"][0], det["center"][1], det["depth_m"]
                    ).reshape(3, 1)
                else:
                    t_ij = np.asarray(det["tvec_rgb"], dtype=np.float64).reshape(3, 1)
                T_ij = build_T(t_ij, det["R_ij"])
                T_ik = T_ij @ TAG_TRANSFORMS[det["tag_id"]]
                det["T_ik"] = T_ik
                det["t_ij_mm"] = t_ij.flatten() * 1000.0  # marker pos in cam frame (mm)
                k_pos.append(T_ik[0:3, 3:4])
                k_rot.append(T_ik[0:3, 0:3])
            if k_pos:
                r_avg = np.mean(np.stack(k_pos, axis=0), axis=0)
                poses[f_idx] = {
                    "position":  r_avg[:, 0] * 1000.0,
                    "rotation":  k_rot[0].copy(),
                    "frame_idx": f_idx,
                }
        valid = sum(1 for p in poses if p is not None)
        print(f"  Gripper poses: {valid}/{n} frames with valid pose")
        return poses

    def _smooth_gripper_poses(self, poses_raw):
        if SMOOTH_GRIPPER_POSE:
            poses = denoise_pose_list(poses_raw,
                                      med_kernel=SMOOTH_MED_KERNEL, sigma=SMOOTH_SIGMA)
            print(f"  Smoothing: med_kernel={SMOOTH_MED_KERNEL}, sigma={SMOOTH_SIGMA}")
        else:
            poses = poses_raw
        return poses

    # ------------------------------------------------------------------
    # Rule book
    # ------------------------------------------------------------------

    def _compute_and_save_rule_book(self, lm_arr, frame_hw, cam_name):
        """
        lm_arr: (N, 33, 4) float32 — columns [x, y, z, visibility].
                NaN rows = frames with no MediaPipe detection.
        """
        if lm_arr.ndim != 3 or lm_arr.shape[1:] != (33, 4):
            print("  Rule book: unexpected landmark array shape, skipping.")
            return

        h, w = frame_hw
        all_joints = set(POSE_RULE_JOINTS)

        triplet_samples = {n: [] for n in JOINT_ANGLE_TRIPLETS}

        for frame_data in lm_arr:
            if np.all(np.isnan(frame_data)):
                continue

            joint_3d = {}
            for j in sorted(all_joints):
                x, y, z, vis = frame_data[j]
                if np.isnan(x) or float(vis) < VIS_THRESHOLD:
                    joint_3d[j] = None
                    continue
                joint_3d[j] = np.array([float(x) * w, float(y) * h, float(z) * w],
                                        dtype=np.float64)

            for name, (ja, jb, jc) in JOINT_ANGLE_TRIPLETS.items():
                A, B, C = joint_3d.get(ja), joint_3d.get(jb), joint_3d.get(jc)
                if A is None or B is None or C is None:
                    continue
                triplet_samples[name].append(_angle_at_joint_deg(A, B, C))

        def _stats(samples, joints_field, desc_field):
            if not samples:
                return None
            arr = np.asarray(samples, dtype=np.float64)
            return {
                "joints":      joints_field,
                "description": desc_field,
                "n_samples":   int(len(arr)),
                "min_deg":     round(float(arr.min()),  2),
                "max_deg":     round(float(arr.max()),  2),
                "mean_deg":    round(float(arr.mean()), 2),
                "std_deg":     round(float(arr.std()),  2),
            }

        triplet_stats = {
            n: s for n, samps in triplet_samples.items()
            if (s := _stats(samps, list(JOINT_ANGLE_TRIPLETS[n]), _TRIPLET_DESCRIPTIONS[n]))
        }

        if not triplet_stats:
            print("  Rule book: no valid angle samples found.")
            return

        n_valid = int(np.sum(~np.all(np.isnan(lm_arr), axis=(1, 2))))
        doc = {
            "source_bag":        DATA_NAME,
            "patient_camera":    cam_name,
            "generated":         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "invariant_note":    (
                "Angles are computed from 3D bone vectors (dot products). "
                "Camera-rotation invariant — rule holds regardless of camera placement."
            ),
            "joints_monitored":  POSE_RULE_JOINTS,
            "n_frames_analyzed": n_valid,
            "joint_angles":      triplet_stats,
        }
        yaml_path = OUT_DIR / f"{DATA_NAME}_pose_rules.yaml"
        with open(yaml_path, "w") as f:
            f.write("# Human pose joint-angle rule book\n")
            f.write("# Angles in degrees. Use min_deg/max_deg as safe-range limits.\n")
            f.write("# Load in ROS:  import yaml; rules = yaml.safe_load(open(path))\n\n")
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)

        total = len(triplet_stats)
        print(f"  Saved rule book ({total} angles, {n_valid} frames): {yaml_path}")

    # ------------------------------------------------------------------
    # Force helpers
    # ------------------------------------------------------------------

    def _load_and_crop_force(self):
        """Load force_raw.npy and slice to the same relative time window as CROP.

        force_raw[:, 0] is relative time (zeroed at first force sample).
        CROP is applied as the same number of seconds from each stream's own start/end.
        Returns cropped (N, 4) array or None if file missing or CROP_UNIT != "s".
        """
        force_path = DATA_DIR / "force_raw.npy"
        if not force_path.exists():
            print("  Force: force_raw.npy not found, skipping force rule book.")
            return None
        if CROP_UNIT != "s":
            print("  Force: CROP_UNIT != 's', skipping time-based force crop.")
            return np.load(str(force_path))

        force_raw = np.load(str(force_path))  # (N, 4): [t_rel, Fx, Fy, Fz]
        if CROP is None:
            print(f"  Force: {len(force_raw)} samples (no crop)")
            return force_raw

        force_t = force_raw[:, 0]
        dur_f   = float(force_t[-1])
        t_start = float(CROP[0]) if float(CROP[0]) >= 0 else dur_f + float(CROP[0])
        t_end   = float(CROP[1]) if float(CROP[1]) >= 0 else dur_f + float(CROP[1])
        sf_f = int(np.searchsorted(force_t, max(0.0, t_start), side="left"))
        ef_f = int(np.searchsorted(force_t, max(0.0, t_end),   side="left"))
        cropped = force_raw[sf_f:ef_f]
        print(f"  Force: {len(cropped)} samples kept (t={t_start:.2f}s → {t_end:.2f}s)")
        return cropped

    def _save_force_rules(self, force_data):
        """Save per-axis Fx/Fy/Fz min/max/mean/std from cropped force data."""
        axis_stats = {}
        for col, name in enumerate(("Fx", "Fy", "Fz"), start=1):
            vals = force_data[:, col]
            axis_stats[name] = {
                "n_samples": int(len(vals)),
                "min_N":     round(float(np.min(vals)),  2),
                "max_N":     round(float(np.max(vals)),  2),
                "mean_N":    round(float(np.mean(vals)), 2),
                "std_N":     round(float(np.std(vals)),  2),
            }
        doc = {
            "source":     DATA_NAME,
            "generated":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "crop":       list(CROP) if CROP is not None else None,
            "n_samples":  int(len(force_data)),
            "duration_s": round(float(force_data[-1, 0] - force_data[0, 0]), 3),
            "force_axes": axis_stats,
        }
        yaml_path = OUT_DIR / f"{DATA_NAME}_force_rules.yaml"
        with open(yaml_path, "w") as f:
            f.write("# Force rule book (Fx, Fy, Fz from /rokubi/wrench)\n")
            f.write("# Units: Newtons. Cropped to same time window as trajectory.\n\n")
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
        print(f"  Saved force rule book: {yaml_path}")

    # ------------------------------------------------------------------
    # CSV saving
    # ------------------------------------------------------------------

    def _save_csv(self, gripper_poses, timestamps):
        valid = [(i, p) for i, p in enumerate(gripper_poses) if p is not None]
        if not valid:
            print("  No valid gripper poses to save.")
            return None

        first_i, first = valid[0]
        R0  = first["rotation"]
        t0  = first["position"]
        ts0 = float(timestamps[first_i])

        csv_path = OUT_DIR / f"{DATA_NAME}_sparse.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "frame",
                             "pos_x", "pos_y", "pos_z",
                             "orient_x", "orient_y", "orient_z"])
            for idx, pose in valid:
                ts    = float(timestamps[idx]) - ts0
                pos   = R0.T @ (pose["position"] - t0)
                euler = ScipyRotation.from_matrix(R0.T @ pose["rotation"]).as_euler("xyz")
                if idx == first_i:
                    ts, pos, euler = 0.0, np.zeros(3), np.zeros(3)
                writer.writerow([
                    round(ts, 6), pose["frame_idx"],
                    round(pos[0], 3), round(pos[1], 3), round(pos[2], 3),
                    round(euler[0], 6), round(euler[1], 6), round(euler[2], 6),
                ])

        print(f"  Saved sparse gripper trajectory: {csv_path}")
        return csv_path


# ==========================================
# --- ENTRY POINT ---
# ==========================================

if __name__ == "__main__":
    TwoCamProcessor().run()
