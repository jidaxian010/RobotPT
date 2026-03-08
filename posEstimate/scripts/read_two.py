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
import mediapipe as mpipe
from scipy.spatial.transform import Rotation as ScipyRotation

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

DATA_NAME = "P2-A4"
DATA_DIR  = Path("posEstimate/data") / DATA_NAME   # folder created by read_rosbag.py
OUT_DIR   = DATA_DIR                               # outputs go into the same folder

LEFT_TOPIC        = "/left_camera/camera/camera/color/image_raw"
RIGHT_TOPIC       = "/right_camera/camera/camera/color/image_raw"
LEFT_DEPTH_TOPIC  = "/left_camera/camera/camera/aligned_depth_to_color/image_raw"
RIGHT_DEPTH_TOPIC = "/right_camera/camera/camera/aligned_depth_to_color/image_raw"

GRIPPER_CAM  = "right"  # "left" or "right": which camera provides ArUco 3D tracking
HUMAN_POSE   = True     # Toggle MediaPipe human pose

CROP = (18, 42)  # (start, end); negative values = seconds from end of stream
# CROP      = None    # None = no extra crop (process everything in the extracted video)
CROP_UNIT = "s"     # "s" (seconds) or "frame"

SMOOTH_GRIPPER_POSE = True
SMOOTH_MED_KERNEL   = 15   # must be odd
SMOOTH_SIGMA        = 20

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

_FX = 602.6597900390625
_FY = 602.2169799804688
_CX = 423.1910400390625
_CY = 249.92578125

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
    from human_pose import _draw_pose, _draw_vis_legend

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
    from human_pose import _draw_pose, _draw_vis_legend

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
        self._aruco_det = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

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

        self._save_csv(gripper_poses, gcam_ts)

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
        corners, ids, _ = self._aruco_det.detectMarkers(color_image)
        frame_dets = []
        if ids is None:
            return frame_dets

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

        csv_path = OUT_DIR / f"{DATA_NAME}.csv"
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

        print(f"  Saved gripper 6D trajectory: {csv_path}")
        return csv_path


# ==========================================
# --- ENTRY POINT ---
# ==========================================

if __name__ == "__main__":
    TwoCamProcessor().run()
