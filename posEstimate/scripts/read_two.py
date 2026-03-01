"""
Extract RGB videos from left and right RealSense cameras stored in a rosbag,
annotate with human pose (MediaPipe) and gripper tracking (ArUco + depth),
and save a gripper 6DoF pose CSV and a joint-angle rule book.

Outputs:
    <DATA_NAME>_left.mp4           annotated left camera
    <DATA_NAME>_right.mp4          annotated right camera
    <DATA_NAME>.csv                gripper 6DoF trajectory
    <DATA_NAME>_pose_rules.yaml    patient joint-angle rule book
"""
import csv
import sys
import bisect
import datetime
from pathlib import Path

import yaml
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.transform import Rotation as ScipyRotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosbag_reader import AnyReader, typestore
from joint_tracker import PoseTracker
from human_pose import _draw_pose, _draw_vis_legend
from denoise import denoise_pose_list

# MediaPipe aliases
BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode

# ==========================================
# --- USER CONFIGURATION ---
# ==========================================

DATA_NAME = "aba"
BAG_PATH  = Path(f"/home/jdx/Downloads/{DATA_NAME}")
OUT_DIR   = Path("posEstimate/data")

LEFT_TOPIC        = "/left_camera/camera/camera/color/image_raw"
RIGHT_TOPIC       = "/right_camera/camera/camera/color/image_raw"
LEFT_DEPTH_TOPIC  = "/left_camera/camera/camera/aligned_depth_to_color/image_raw"
RIGHT_DEPTH_TOPIC = "/right_camera/camera/camera/aligned_depth_to_color/image_raw"

GRIPPER_CAM  = "left"   # "left" or "right": which camera provides ArUco 3D tracking
HUMAN_POSE   = True     # Toggle MediaPipe human pose

CROP         = (-42, -33)  # (start, end); negative values = seconds from end of stream
CROP_UNIT    = "s"      # "s" (seconds) or "frame"

SMOOTH_GRIPPER_POSE = True
SMOOTH_MED_KERNEL   = 15   # must be odd
SMOOTH_SIGMA        = 20

MARKER_SIZE_METERS = 0.0725

# --- Human pose rule book ---
# Joints monitored: shoulders (11,12), elbows (13,14), wrists (15,16), hips (23,24)
# NOTE: All angles are computed from 3D bone vectors (dot products), so they are
# camera-position invariant — the rule holds regardless of where the camera is placed.
POSE_RULE_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24]
VIS_THRESHOLD    = 0.3   # skip joint if visibility below this

# --- Triplet angles: angle at the MIDDLE joint (A, B, C) → angle at B ---
JOINT_ANGLE_TRIPLETS = {
    # Elbow flexion (arm chain)
    "left_elbow":       (11, 13, 15),   # l.shoulder – l.elbow – l.wrist
    "right_elbow":      (12, 14, 16),   # r.shoulder – r.elbow – r.wrist
    # Shoulder — arm angle relative to the shoulder line
    "left_shoulder":    (12, 11, 13),   # r.shoulder – l.shoulder – l.elbow
    "right_shoulder":   (11, 12, 14),   # l.shoulder – r.shoulder – r.elbow
    # Arm angle relative to trunk (hip-anchored)
    "left_arm_torso":   (23, 11, 13),   # l.hip – l.shoulder – l.elbow
    "right_arm_torso":  (24, 12, 14),   # r.hip – r.shoulder – r.elbow
    # Torso shape: hip-line bending toward each shoulder
    "torso_left":       (24, 23, 11),   # r.hip – l.hip – l.shoulder
    "torso_right":      (23, 24, 12),   # l.hip – r.hip – r.shoulder
}

_TRIPLET_DESCRIPTIONS = {
    "left_elbow":       "left shoulder – left elbow – left wrist",
    "right_elbow":      "right shoulder – right elbow – right wrist",
    "left_shoulder":    "right shoulder – left shoulder – left elbow (arm vs shoulder line)",
    "right_shoulder":   "left shoulder – right shoulder – right elbow (arm vs shoulder line)",
    "left_arm_torso":   "left hip – left shoulder – left elbow (arm vs trunk)",
    "right_arm_torso":  "right hip – right shoulder – right elbow (arm vs trunk)",
    "torso_left":       "right hip – left hip – left shoulder (hip bends to left side)",
    "torso_right":      "left hip – right hip – right shoulder (hip bends to right side)",
}

# --- Segment angles: angle between two separate bone vectors ---
# Each entry: ((ja, jb), (jc, jd)) → angle between vector (jb-ja) and (jd-jc)
# Used for torso twist: ~0° means no twist, larger value = more twist.
SEGMENT_ANGLE_PAIRS = {
    "torso_twist": ((11, 12), (23, 24)),  # shoulder axis vs hip axis
}

_SEGMENT_DESCRIPTIONS = {
    "torso_twist": "angle between shoulder axis (11→12) and hip axis (23→24); ~0° = no twist",
}

MODEL_PATH = str(Path(__file__).resolve().parent.parent /
                 "data" / "pose_landmarker_full.task")

# Camera intrinsics (848×480 RealSense stream)
_FX = 602.6597900390625
_FY = 602.2169799804688
_CX = 423.1910400390625
_CY = 249.92578125

CAMERA_MATRIX = np.array(
    [[_FX, 0.0, _CX], [0.0, _FY, _CY], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float32)


# ==========================================
# --- GRIPPER GEOMETRY (from realsense_pose.py) ---
# ==========================================

def build_T(t, R):
    """Build 4×4 homogeneous transform from translation t (3,) and rotation R (3,3)."""
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    T[0:3, 3:4] = t
    return T


def deproject_pixel_depth_to_point_m(u, v, depth_m):
    """Back-project aligned depth pixel to camera-frame 3D point (meters)."""
    x = (float(u) - _CX) * float(depth_m) / _FX
    y = (float(v) - _CY) * float(depth_m) / _FY
    z = float(depth_m)
    return np.array([x, y, z], dtype=np.float64)


R_10 = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)
R_20 = np.array(
    [[0.7071, 0.0, 0.7071], [0.0, 1.0, 0.0], [-0.7071, 0.0, 0.7071]],
    dtype=np.float64,
)
R_30 = np.array(
    [[0.0, 0.0, 1.0], [-0.6654, 0.7465, 0.0], [-0.7465, -0.6654, 0.0]],
    dtype=np.float64,
)
R_40 = np.array(
    [[-0.7071, 0.0, 0.7071], [0.0, 1.0, 0.0], [-0.7071, 0.0, -0.7071]],
    dtype=np.float64,
)
R_50 = np.array(
    [[0.0, 0.0, 1.0], [0.7465, 0.6654, 0.0], [-0.6654, 0.7465, 0.0]],
    dtype=np.float64,
)

TAG_TRANSFORMS = {
    1: build_T(t=[0.3927,  0.0225, -0.2142], R=R_10),
    2: build_T(t=[0.3641,  0.0225,  0.0993], R=R_20),
    3: build_T(t=[0.3927, -0.0592, -0.2003], R=R_30),
    4: build_T(t=[0.1912,  0.0225, -0.4561], R=R_40),
    5: build_T(t=[0.3927,  0.0928, -0.1703], R=R_50),
}

_half_size = MARKER_SIZE_METERS / 2.0
marker_3d_edges = np.array(
    [
        [-_half_size,  _half_size, 0.0],
        [ _half_size,  _half_size, 0.0],
        [ _half_size, -_half_size, 0.0],
        [-_half_size, -_half_size, 0.0],
    ],
    dtype=np.float32,
)


# ==========================================
# --- HELPERS ---
# ==========================================

def _angle_at_joint_deg(A, B, C):
    """Angle in degrees at joint B, between vectors B→A and B→C."""
    BA = np.asarray(A, dtype=np.float64) - np.asarray(B, dtype=np.float64)
    BC = np.asarray(C, dtype=np.float64) - np.asarray(B, dtype=np.float64)
    denom = np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-8
    cos_t = np.dot(BA, BC) / denom
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))


def ros_image_to_cv2(msg):
    height, width, encoding = msg.height, msg.width, msg.encoding
    data = msg.data if isinstance(msg.data, bytes) else bytes(msg.data)
    img = np.frombuffer(data, dtype=np.uint8)

    if encoding in ("bgr8",):
        img = img.reshape((height, width, 3))
    elif encoding in ("rgb8",):
        img = img.reshape((height, width, 3))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding in ("mono8",):
        img = img.reshape((height, width))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif encoding in ("bgra8",):
        img = img.reshape((height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif encoding in ("rgba8",):
        img = img.reshape((height, width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")
    return img


def get_stamp(msg, fallback_ns):
    try:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    except Exception:
        return fallback_ns * 1e-9


def fps_from_timestamps(timestamps, fallback=30.0):
    if len(timestamps) < 2:
        return fallback
    dts = np.diff(np.asarray(timestamps, dtype=np.float64))
    dts = dts[dts > 0]
    return float(1.0 / np.median(dts)) if len(dts) else fallback


def save_video(frames, timestamps, out_path, fallback_fps=30.0):
    if not frames:
        print(f"  No frames to save for {out_path}")
        return
    h, w = frames[0].shape[:2]
    fps = fps_from_timestamps(timestamps, fallback=fallback_fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {out_path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"  Saved {len(frames)} frames @ {fps:.2f} fps -> {out_path}")


def _apply_crop(frames, timestamps, crop, crop_unit):
    """Crop frames/timestamps by time window or frame indices."""
    if crop is None:
        return frames, timestamps
    total = len(timestamps)
    if total == 0:
        return frames, timestamps

    start, end = crop
    t_rel = np.asarray(timestamps, dtype=np.float64) - timestamps[0]

    if crop_unit == "s":
        duration = float(t_rel[-1])
        start_s = float(start) if float(start) >= 0 else duration + float(start)
        end_s   = float(end)   if float(end)   >= 0 else duration + float(end)
        sf = int(np.searchsorted(t_rel, max(0.0, start_s), side="left"))
        ef = int(np.searchsorted(t_rel, max(0.0, end_s),   side="left"))
    else:
        sf = int(start) if int(start) >= 0 else total + int(start)
        ef = int(end)   if int(end)   >= 0 else total + int(end)

    sf = int(np.clip(sf, 0, total))
    ef = int(np.clip(ef, sf, total))
    print(f"  Crop: frames [{sf}, {ef}) of {total} total")
    return frames[sf:ef], timestamps[sf:ef]


# ==========================================
# --- MAIN PROCESSOR ---
# ==========================================

class TwoCamProcessor:
    def __init__(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if HUMAN_POSE and not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {MODEL_PATH}\n"
                "Download pose_landmarker_full.task to posEstimate/data/"
            )

    # ------------------------------------------------------------------
    # Top-level orchestrator
    # ------------------------------------------------------------------

    def run(self):
        # Step 1: read all streams from bag in one pass
        print("Reading bag...")
        left_frames, left_ts, right_frames, right_ts, gcam_depth_data, other_depth_data = \
            self._read_both_cameras()
        print(f"  Left:  {len(left_frames)} frames")
        print(f"  Right: {len(right_frames)} frames")
        print(f"  Gripper-cam depth: {len(gcam_depth_data)} frames")
        print(f"  Other-cam depth:   {len(other_depth_data)} frames")

        # Step 2: crop both cameras
        left_frames, left_ts   = _apply_crop(left_frames,  left_ts,  CROP, CROP_UNIT)
        right_frames, right_ts = _apply_crop(right_frames, right_ts, CROP, CROP_UNIT)
        print(f"  After crop — Left: {len(left_frames)}, Right: {len(right_frames)}")

        # Identify gripper cam vs other cam
        if GRIPPER_CAM == "left":
            gcam_frames, gcam_ts   = left_frames,  left_ts
            other_frames, other_ts = right_frames, right_ts
            other_cam_name = "right"
        else:
            gcam_frames, gcam_ts   = right_frames, right_ts
            other_frames, other_ts = left_frames,  left_ts
            other_cam_name = "left"

        # Step 3: human pose annotation on both cameras; other cam also returns landmarks
        if HUMAN_POSE:
            print("Running MediaPipe on gripper camera...")
            gcam_annotated, _             = self._run_human_pose(gcam_frames,  gcam_ts)
            print("Running MediaPipe on other camera (patient)...")
            other_annotated, other_lms    = self._run_human_pose(other_frames, other_ts)
        else:
            gcam_annotated  = [f.copy() for f in gcam_frames]
            other_annotated = [f.copy() for f in other_frames]
            other_lms       = [None] * len(other_frames)

        # Step 4: ArUco + depth on GRIPPER_CAM
        print("Running ArUco detection on gripper camera...")
        gcam_depth_lookup = self._match_depth(gcam_ts, gcam_depth_data)
        per_frame_dets    = self._run_aruco(gcam_frames, gcam_depth_lookup)
        gripper_poses_raw = self._build_gripper_poses(per_frame_dets)

        # Step 5: smooth gripper poses
        gripper_poses = self._smooth_gripper_poses(gripper_poses_raw)

        # Step 6: overlay ArUco annotations on gripper cam annotated frames
        self._annotate_gripper_cam_frames(gcam_annotated, per_frame_dets, gripper_poses)

        # Step 7: save videos (left always → _left.mp4, right → _right.mp4)
        stem = Path(BAG_PATH).name
        if GRIPPER_CAM == "left":
            save_video(gcam_annotated,  gcam_ts,  OUT_DIR / f"{stem}_left.mp4")
            save_video(other_annotated, other_ts, OUT_DIR / f"{stem}_right.mp4")
        else:
            save_video(other_annotated, other_ts, OUT_DIR / f"{stem}_left.mp4")
            save_video(gcam_annotated,  gcam_ts,  OUT_DIR / f"{stem}_right.mp4")

        # Step 8: save gripper CSV
        self._save_csv(gripper_poses, gcam_ts)

        # Step 9: compute and save joint-angle rule book from other (patient) camera
        if HUMAN_POSE:
            print("Computing patient joint-angle rule book...")
            other_depth_lookup = self._match_depth(other_ts, other_depth_data)
            self._compute_and_save_rule_book(
                other_lms, other_frames, other_ts, other_depth_lookup, other_cam_name
            )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _read_both_cameras(self):
        gcam_depth_topic  = (LEFT_DEPTH_TOPIC  if GRIPPER_CAM == "left" else RIGHT_DEPTH_TOPIC)
        other_depth_topic = (RIGHT_DEPTH_TOPIC if GRIPPER_CAM == "left" else LEFT_DEPTH_TOPIC)
        topics_needed = {LEFT_TOPIC, RIGHT_TOPIC, gcam_depth_topic, other_depth_topic}

        left_frames, left_ts   = [], []
        right_frames, right_ts = [], []
        gcam_depth_data  = []
        other_depth_data = []

        with AnyReader([BAG_PATH], default_typestore=typestore) as reader:
            conns = {c.topic: c for c in reader.connections
                     if c.topic in topics_needed}
            missing = topics_needed - set(conns)
            if missing:
                color_missing = missing - {gcam_depth_topic, other_depth_topic}
                if color_missing:
                    raise RuntimeError(f"Color topics not found in bag: {color_missing}")
                for t in missing & {gcam_depth_topic, other_depth_topic}:
                    print(f"  Warning: depth topic not found in bag: {t}")

            for conn, ts, raw in reader.messages(connections=list(conns.values())):
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                stamp = get_stamp(msg, ts)

                if conn.topic == LEFT_TOPIC:
                    try:
                        left_frames.append(ros_image_to_cv2(msg))
                        left_ts.append(stamp)
                    except Exception as e:
                        print(f"  Warning: skipping left frame: {e}")

                elif conn.topic == RIGHT_TOPIC:
                    try:
                        right_frames.append(ros_image_to_cv2(msg))
                        right_ts.append(stamp)
                    except Exception as e:
                        print(f"  Warning: skipping right frame: {e}")

                elif conn.topic in (gcam_depth_topic, other_depth_topic):
                    try:
                        h = msg.height
                        w = msg.width
                        step = msg.step
                        data = msg.data if isinstance(msg.data, bytes) else bytes(msg.data)
                        dimg = np.frombuffer(data, dtype=np.uint16)
                        dimg = dimg.reshape((h, step // 2))[:, :w]
                        if conn.topic == gcam_depth_topic:
                            gcam_depth_data.append((stamp, dimg))
                        else:
                            other_depth_data.append((stamp, dimg))
                    except Exception as e:
                        print(f"  Warning: skipping depth frame: {e}")

        return left_frames, left_ts, right_frames, right_ts, gcam_depth_data, other_depth_data

    # ------------------------------------------------------------------
    # Depth matching
    # ------------------------------------------------------------------

    def _match_depth(self, rgb_timestamps, depth_data, max_dt_s=0.05):
        """
        Return dict[frame_idx -> depth_img] by nearest-timestamp matching.
        """
        if not depth_data:
            return {}
        depth_ts = [t for t, _ in depth_data]
        result = {}
        for frame_idx, t_rgb in enumerate(rgb_timestamps):
            t_rgb = float(t_rgb)
            j = bisect.bisect_left(depth_ts, t_rgb)
            candidates = []
            if j > 0:
                candidates.append(j - 1)
            if j < len(depth_ts):
                candidates.append(j)
            if not candidates:
                continue
            best = min(candidates, key=lambda k: abs(depth_ts[k] - t_rgb))
            if abs(depth_ts[best] - t_rgb) <= max_dt_s:
                result[frame_idx] = depth_data[best][1]
        return result

    # ------------------------------------------------------------------
    # Human pose
    # ------------------------------------------------------------------

    def _run_human_pose(self, frames, timestamps):
        """Run MediaPipe PoseLandmarker on frames; return list of annotated BGR images."""
        if not frames:
            return []

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        tracker = PoseTracker(alpha=0.35, dead_band=3.0)
        t0_ms = int(timestamps[0] * 1000)
        dts = [0.0] + list(np.diff(np.asarray(timestamps, dtype=np.float64)))
        h, w = frames[0].shape[:2]
        annotated   = []
        all_landmarks = []   # list[list[NormalizedLandmark] | None]

        with PoseLandmarker.create_from_options(options) as landmarker:
            for i, (bgr, ts, dt) in enumerate(zip(frames, timestamps, dts)):
                rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms  = max(0, int(ts * 1000) - t0_ms)

                result = landmarker.detect_for_video(mp_img, ts_ms)
                out = bgr.copy()

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    positions    = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms])
                    visibilities = np.array([lm.visibility for lm in lms])
                    tracked = tracker.update(positions, visibilities, dt)
                    _draw_pose(out, lms, tracked, h, w)
                    all_landmarks.append(lms)
                else:
                    cv2.putText(out, "No pose", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    tracker.update(np.zeros((33, 3)), np.zeros(33), dt)
                    all_landmarks.append(None)

                _draw_vis_legend(out, h, w)
                annotated.append(out)

                if (i + 1) % 50 == 0:
                    print(f"    MediaPipe: {i+1}/{len(frames)} frames")

        return annotated, all_landmarks

    # ------------------------------------------------------------------
    # ArUco detection
    # ------------------------------------------------------------------

    def _run_aruco(self, frames, depth_lookup):
        """Detect ArUco markers per frame; return per_frame_detections."""
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector   = cv2.aruco.ArucoDetector(dictionary, parameters)

        per_frame_dets = []
        for frame_idx, color_image in enumerate(frames):
            corners, ids, _ = detector.detectMarkers(color_image)
            frame_dets = []

            if ids is not None:
                for i in range(len(ids)):
                    tag_id = int(ids[i][0])
                    if tag_id not in TAG_TRANSFORMS:
                        continue

                    marker_2d = corners[i][0].astype(np.float32)
                    ok, rvec, tvec_rgb = cv2.solvePnP(
                        marker_3d_edges, marker_2d, CAMERA_MATRIX, DIST_COEFFS
                    )
                    if not ok:
                        continue

                    R_ij, _ = cv2.Rodrigues(rvec)
                    cx = int(np.mean(marker_2d[:, 0]))
                    cy = int(np.mean(marker_2d[:, 1]))

                    depth_m = 0.0
                    depth_img = depth_lookup.get(frame_idx)
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

            per_frame_dets.append(frame_dets)

        total_dets = sum(len(d) for d in per_frame_dets)
        print(f"  ArUco: {total_dets} marker detections across {len(frames)} frames")
        return per_frame_dets

    # ------------------------------------------------------------------
    # Gripper pose computation
    # ------------------------------------------------------------------

    def _build_gripper_poses(self, per_frame_dets):
        """Fuse multi-tag detections into gripper CoM pose per frame."""
        num_frames = len(per_frame_dets)
        gripper_poses_raw = [None] * num_frames

        for f_idx, frame_dets in enumerate(per_frame_dets):
            k_positions = []
            k_rotations = []

            for det in frame_dets:
                tag_id    = det["tag_id"]
                R_ij      = det["R_ij"]
                depth_m   = det["depth_m"]
                cx, cy    = det["center"]

                if depth_m > 0:
                    t_ij = deproject_pixel_depth_to_point_m(cx, cy, depth_m).reshape(3, 1)
                else:
                    t_ij = np.asarray(det["tvec_rgb"], dtype=np.float64).reshape(3, 1)

                T_ij = build_T(t_ij, R_ij)
                T_ik = T_ij @ TAG_TRANSFORMS[tag_id]
                det["T_ik"] = T_ik

                k_positions.append(T_ik[0:3, 3:4])
                k_rotations.append(T_ik[0:3, 0:3])

            if k_positions:
                r_avg = np.mean(np.stack(k_positions, axis=0), axis=0)
                gripper_poses_raw[f_idx] = {
                    "position":  r_avg[:, 0] * 1000.0,   # meters → mm
                    "rotation":  k_rotations[0].copy(),
                    "frame_idx": f_idx,
                }

        valid_count = sum(1 for p in gripper_poses_raw if p is not None)
        print(f"  Gripper poses: {valid_count}/{num_frames} frames with valid pose")
        return gripper_poses_raw

    def _smooth_gripper_poses(self, poses_raw):
        if SMOOTH_GRIPPER_POSE:
            poses = denoise_pose_list(
                poses_raw,
                med_kernel=SMOOTH_MED_KERNEL,
                sigma=SMOOTH_SIGMA,
            )
            print(f"  Applied gripper smoothing: med_kernel={SMOOTH_MED_KERNEL}, sigma={SMOOTH_SIGMA}")
        else:
            poses = poses_raw
        return poses

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotate_gripper_cam_frames(self, annotated_frames, per_frame_dets, gripper_poses):
        """Overlay ArUco markers and gripper frame axes onto already-annotated frames."""
        for frame_idx, (out, frame_dets) in enumerate(zip(annotated_frames, per_frame_dets)):
            if not frame_dets:
                continue

            # Draw detected marker outlines and IDs
            draw_corners = [d["corners_draw"] for d in frame_dets]
            draw_ids     = np.array([[d["tag_id"]] for d in frame_dets], dtype=np.int32)
            cv2.aruco.drawDetectedMarkers(out, draw_corners, draw_ids)

            # Draw per-tag frame axes + center dot + depth label
            k_positions = []
            for det in frame_dets:
                cv2.drawFrameAxes(
                    out, CAMERA_MATRIX, DIST_COEFFS,
                    det["rvec"], det["tvec_rgb"], MARKER_SIZE_METERS
                )
                cx, cy = det["center"]
                cv2.circle(out, (cx, cy), 4, (0, 255, 255), -1)
                if det["depth_m"] > 0:
                    cv2.putText(
                        out,
                        f"{det['tag_id']}:{det['depth_m']*1000:.0f}mm",
                        (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                    )
                if "T_ik" in det:
                    k_positions.append(det["T_ik"][0:3, 3:4])

            # Draw fused gripper CoM frame
            gp = gripper_poses[frame_idx] if frame_idx < len(gripper_poses) else None
            if gp is not None:
                r_m = np.asarray(gp["position"], dtype=np.float64) / 1000.0
                rvec_k, _ = cv2.Rodrigues(np.asarray(gp["rotation"], dtype=np.float64))
                cv2.drawFrameAxes(
                    out, CAMERA_MATRIX, DIST_COEFFS,
                    rvec_k, r_m.reshape(3, 1), MARKER_SIZE_METERS * 1.5,
                )
                label = (f"Tags: {len(k_positions)}"
                         + (" (smoothed)" if SMOOTH_GRIPPER_POSE else ""))
                cv2.putText(out, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.putText(out, f"Frame: {frame_idx}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ------------------------------------------------------------------
    # Patient joint-angle rule book
    # ------------------------------------------------------------------

    def _compute_and_save_rule_book(self, all_landmarks, frames, timestamps,
                                    depth_lookup, cam_name):
        """
        Compute per-frame 3D joint angles (joints 11-16, 23, 24) and aggregate
        min/max/mean/std into a YAML rule book.

        All angles are derived from 3D bone vectors (dot products), making them
        camera-position invariant — the same rule holds regardless of camera placement.

        3D position:  aligned depth → deproject;  fallback: MediaPipe lm.z * frame_width
        """
        if not frames:
            return

        h, w = frames[0].shape[:2]
        all_joints = set(POSE_RULE_JOINTS)
        for (ja, jb), (jc, jd) in SEGMENT_ANGLE_PAIRS.values():
            all_joints.update([ja, jb, jc, jd])

        triplet_samples: dict[str, list[float]] = {n: [] for n in JOINT_ANGLE_TRIPLETS}
        segment_samples: dict[str, list[float]] = {n: [] for n in SEGMENT_ANGLE_PAIRS}

        for frame_idx, lms in enumerate(all_landmarks):
            if lms is None:
                continue

            depth_img = depth_lookup.get(frame_idx)

            # Build 3D position for every required joint
            joint_3d: dict[int, np.ndarray | None] = {}
            for j in sorted(all_joints):
                lm = lms[j]
                if lm.visibility < VIS_THRESHOLD:
                    joint_3d[j] = None
                    continue
                cx = int(np.clip(int(lm.x * w), 0, w - 1))
                cy = int(np.clip(int(lm.y * h), 0, h - 1))

                depth_m = 0.0
                if depth_img is not None:
                    raw_mm = depth_img[cy, cx]
                    if raw_mm > 0:
                        depth_m = float(raw_mm) / 1000.0

                if depth_m > 0:
                    pos = deproject_pixel_depth_to_point_m(cx, cy, depth_m)
                else:
                    # MediaPipe z shares the same normalised scale as x
                    pos = np.array([lm.x * w, lm.y * h, lm.z * w], dtype=np.float64)

                joint_3d[j] = pos

            # Triplet angles (angle at middle joint)
            for name, (ja, jb, jc) in JOINT_ANGLE_TRIPLETS.items():
                A, B, C = joint_3d.get(ja), joint_3d.get(jb), joint_3d.get(jc)
                if A is None or B is None or C is None:
                    continue
                triplet_samples[name].append(_angle_at_joint_deg(A, B, C))

            # Segment angles (angle between two separate bone vectors)
            for name, ((ja, jb), (jc, jd)) in SEGMENT_ANGLE_PAIRS.items():
                A, B = joint_3d.get(ja), joint_3d.get(jb)
                C, D = joint_3d.get(jc), joint_3d.get(jd)
                if A is None or B is None or C is None or D is None:
                    continue
                seg1 = np.asarray(B, dtype=np.float64) - np.asarray(A, dtype=np.float64)
                seg2 = np.asarray(D, dtype=np.float64) - np.asarray(C, dtype=np.float64)
                denom = np.linalg.norm(seg1) * np.linalg.norm(seg2) + 1e-8
                cos_t = np.dot(seg1, seg2) / denom
                segment_samples[name].append(
                    float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
                )

        def _stats(name, samples, joints_field, desc_field):
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

        triplet_stats = {}
        for name, samples in triplet_samples.items():
            s = _stats(name, samples,
                       list(JOINT_ANGLE_TRIPLETS[name]),
                       _TRIPLET_DESCRIPTIONS[name])
            if s:
                triplet_stats[name] = s

        segment_stats = {}
        for name, samples in segment_samples.items():
            (ja, jb), (jc, jd) = SEGMENT_ANGLE_PAIRS[name]
            s = _stats(name, samples,
                       [[ja, jb], [jc, jd]],
                       _SEGMENT_DESCRIPTIONS[name])
            if s:
                segment_stats[name] = s

        if not triplet_stats and not segment_stats:
            print("  Rule book: no valid angle samples found.")
            return

        n_valid = int(sum(1 for lm in all_landmarks if lm is not None))
        doc = {
            "source_bag":        DATA_NAME,
            "patient_camera":    cam_name,
            "generated":         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "invariant_note":    (
                "Angles are computed from 3D bone vectors (dot products). "
                "They are camera-rotation invariant — the rule holds regardless "
                "of camera placement."
            ),
            "joints_monitored":  POSE_RULE_JOINTS,
            "n_frames_analyzed": n_valid,
            "joint_angles":      triplet_stats,   # angle at a single joint
            "torso_angles":      segment_stats,   # angle between two body segments
        }

        yaml_path = OUT_DIR / f"{DATA_NAME}_pose_rules.yaml"
        with open(yaml_path, "w") as f:
            f.write("# Human pose joint-angle rule book\n")
            f.write("# Angles in degrees. Use min_deg/max_deg as safe-range limits.\n")
            f.write("# torso_twist ~0 deg = no twist; larger = more twisting.\n")
            f.write("# Load in ROS:  import yaml; rules = yaml.safe_load(open(path))\n\n")
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)

        total = len(triplet_stats) + len(segment_stats)
        print(f"  Saved rule book ({total} angles, {n_valid} frames): {yaml_path}")

    # ------------------------------------------------------------------
    # CSV saving
    # ------------------------------------------------------------------

    def _save_csv(self, gripper_poses, timestamps):
        """
        Save gripper CoM trajectory relative to the first valid frame.

        Poses are expressed in the initial gripper frame:
            pos   = R0.T @ (t_cam_t - t_cam_0)   [mm], first row = 0,0,0
            euler = Euler XYZ of R0.T @ R_cam_t   [rad], first row = 0,0,0
        """
        valid = [(i, p) for i, p in enumerate(gripper_poses) if p is not None]
        if not valid:
            print("  No valid gripper poses to save.")
            return None

        _, first = valid[0]
        R0  = first["rotation"]
        t0  = first["position"]
        ts0 = float(timestamps[first["frame_idx"]])

        csv_path = OUT_DIR / f"{DATA_NAME}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "frame",
                             "pos_x", "pos_y", "pos_z",
                             "orient_x", "orient_y", "orient_z"])
            first_valid_idx = valid[0][0]
            for idx, pose in valid:
                ts    = float(timestamps[pose["frame_idx"]]) - ts0
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


# ==========================================
# --- ENTRY POINT ---
# ==========================================

if __name__ == "__main__":
    TwoCamProcessor().run()
