"""
Single-marker debug script.

Runs ArUco detection on the gripper camera for ONE selected marker,
decomposes the gripper pose estimate into its components, and plots them.

Replicates the detection + pose pipeline from read_two.py but stripped
down to focus on understanding a single marker's contribution.
"""
import bisect
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as ScipyRotation

# ==========================================
# --- USER CONFIGURATION ---
# ==========================================

DATA_NAME   = "P3-A3"
GRIPPER_CAM = "right"
CROP        = (15, 21)
CROP_UNIT   = "s"

DEBUG_TAG_ID = 3          # <-- which marker to debug

DATA_DIR = Path("posEstimate/data") / DATA_NAME

MARKER_SIZE_METERS = 0.0725

_FX = 602.6598510742188
_FY = 602.2169799804688
_CX = 319.1910400390625
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


def deproject(u, v, depth_m):
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
# --- HELPERS ---
# ==========================================

def _crop_frame_range(timestamps, crop, crop_unit):
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
    if len(depth_ts_arr) == 0:
        return None
    i = bisect.bisect_left(depth_ts_arr, query_ts)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(depth_ts_arr)]
    if not candidates:
        return None
    best = min(candidates, key=lambda j: abs(depth_ts_arr[j] - query_ts))
    return best if abs(depth_ts_arr[best] - query_ts) <= max_dt else None


# ==========================================
# --- MAIN ---
# ==========================================

def main():
    tag_id = DEBUG_TAG_ID

    print(f"Debugging marker {tag_id}")

    # Load data
    gcam = "left" if GRIPPER_CAM == "left" else "right"
    rgb_path  = DATA_DIR / f"{gcam}.mp4"
    ts_all    = np.load(str(DATA_DIR / f"{gcam}_ts.npy"))
    depth_dir = DATA_DIR / f"{gcam}_depth"
    depth_ts  = (
        np.load(str(DATA_DIR / f"{gcam}_depth_ts.npy"))
        if (DATA_DIR / f"{gcam}_depth_ts.npy").exists() else np.array([])
    )
    has_depth = depth_dir.exists() and len(depth_ts) > 0

    sf, ef = _crop_frame_range(ts_all, CROP, CROP_UNIT)
    ts = ts_all[sf:ef]
    n_frames = len(ts)
    print(f"  Frames: {n_frames}")

    # ArUco detector
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
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Video output
    debug_video_path = DATA_DIR / f"{DATA_NAME}_debug.mp4"
    writer = None

    # FPS from timestamps
    if len(ts) >= 2:
        dts = np.diff(ts.astype(np.float64))
        dts = dts[dts > 0]
        fps = float(1.0 / np.median(dts)) if len(dts) else 30.0
    else:
        fps = 30.0

    # Process frames
    cap = cv2.VideoCapture(str(rgb_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

    prev_rvec = None
    prev_tvec = None

    # Collected data
    times, u_list, v_list, depth_list = [], [], [], []
    tvec_pnp_list, rvec_pnp_list = [], []
    t_ij_list, r_ij_list = [], []
    gripper_pos_cam_list, gripper_rot_cam_list = [], []
    gripper_pos_init_list = []
    gripper_uv_list = []  # gripper reprojected to 2D pixels
    t0 = float(ts[0])
    first = True
    R0_gripper = None   # initial gripper rotation (for initial-frame transform)
    t0_gripper = None   # initial gripper position
    n_detected = 0
    n_flips_rejected = 0

    for crop_idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        out_frame = frame.copy()
        h_f, w_f = frame.shape[:2]

        # Init video writer on first frame
        if writer is None:
            writer = cv2.VideoWriter(
                str(debug_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (w_f, h_f),
            )

        # Sharpen + detect
        blur  = cv2.GaussianBlur(frame, (0, 0), sigmaX=5)
        sharp = cv2.addWeighted(frame, 2.0, blur, -1.0, 0)
        corners, ids, _ = detector.detectMarkers(sharp)

        detected_this_frame = False

        if ids is not None:
            for i in range(len(ids)):
                if int(ids[i][0]) != tag_id:
                    continue

                marker_2d = corners[i][0].astype(np.float32)

                # SolvePnP with extrinsic guess
                if prev_rvec is not None:
                    ok, rvec, tvec_rgb = cv2.solvePnP(
                        marker_3d_edges, marker_2d, CAMERA_MATRIX, DIST_COEFFS,
                        rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                else:
                    ok, rvec, tvec_rgb = cv2.solvePnP(
                        marker_3d_edges, marker_2d, CAMERA_MATRIX, DIST_COEFFS
                    )
                if not ok:
                    continue

                # Flip rejection
                R_ij, _ = cv2.Rodrigues(rvec)
                if prev_rvec is not None:
                    R_prev, _ = cv2.Rodrigues(prev_rvec)
                    R_diff = R_prev.T @ R_ij
                    angle_diff = abs(np.arccos(np.clip(
                        (np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0
                    )))
                    if np.degrees(angle_diff) > 30.0:
                        rvec = prev_rvec.copy()
                        tvec_rgb = prev_tvec.copy()
                        R_ij, _ = cv2.Rodrigues(rvec)
                        n_flips_rejected += 1

                prev_rvec = rvec.copy()
                prev_tvec = tvec_rgb.copy()

                # Depth (patch median to reduce noise)
                DEPTH_PATCH = 11  # patch half-size in pixels
                cx = int(np.mean(marker_2d[:, 0]))
                cy = int(np.mean(marker_2d[:, 1]))
                depth_m = 0.0
                if has_depth:
                    di = _find_depth_idx(depth_ts, float(ts[crop_idx]))
                    if di is not None:
                        dp = depth_dir / f"{di:06d}.png"
                        if dp.exists():
                            depth_img = cv2.imread(str(dp), cv2.IMREAD_ANYDEPTH)
                            if depth_img is not None:
                                h_d, w_d = depth_img.shape[:2]
                                y0 = max(0, cy - DEPTH_PATCH)
                                y1 = min(h_d, cy + DEPTH_PATCH + 1)
                                x0 = max(0, cx - DEPTH_PATCH)
                                x1 = min(w_d, cx + DEPTH_PATCH + 1)
                                patch = depth_img[y0:y1, x0:x1].astype(np.float64)
                                valid = patch[patch > 0]
                                if len(valid) > 0:
                                    depth_m = float(np.median(valid)) / 1000.0

                # t_ij: marker position in camera frame
                if depth_m > 0:
                    t_ij = deproject(cx, cy, depth_m).reshape(3, 1)
                else:
                    t_ij = np.asarray(tvec_rgb, dtype=np.float64).reshape(3, 1)

                # T_ik: gripper pose in camera frame
                T_ij = build_T(t_ij, R_ij)
                T_ik = T_ij @ TAG_TRANSFORMS[tag_id]
                gripper_pos_cam = T_ik[0:3, 3].flatten() * 1000.0  # mm
                R_ik = T_ik[0:3, 0:3]

                if first:
                    R0_gripper = R_ik.copy()
                    t0_gripper = gripper_pos_cam.copy()
                    first = False

                # Gripper in initial gripper frame
                gripper_pos_init = R0_gripper.T @ (gripper_pos_cam - t0_gripper)

                times.append(float(ts[crop_idx]) - t0)
                u_list.append(cx)
                v_list.append(cy)
                depth_list.append(depth_m * 1000.0)
                tvec_pnp_list.append(np.asarray(tvec_rgb, dtype=np.float64).flatten() * 1000.0)
                rvec_pnp_list.append(ScipyRotation.from_matrix(R_ij).as_euler("xyz", degrees=True))
                t_ij_list.append(t_ij.flatten() * 1000.0)
                r_ij_list.append(ScipyRotation.from_matrix(R_ij).as_euler("xyz", degrees=True))
                gripper_pos_cam_list.append(gripper_pos_cam)
                gripper_rot_cam_list.append(ScipyRotation.from_matrix(R_ik).as_euler("xyz", degrees=True))
                gripper_pos_init_list.append(gripper_pos_init)
                # Reproject gripper 3D (meters) to pixel coords
                gp_m = T_ik[0:3, 3]  # meters
                grip_u = _FX * gp_m[0] / gp_m[2] + _CX
                grip_v = _FY * gp_m[1] / gp_m[2] + _CY
                gripper_uv_list.append((grip_u, grip_v))
                n_detected += 1
                detected_this_frame = True

                # --- Annotate frame: marker detection ---
                cv2.aruco.drawDetectedMarkers(
                    out_frame, [corners[i]], np.array([[tag_id]], dtype=np.int32))
                cv2.drawFrameAxes(out_frame, CAMERA_MATRIX, DIST_COEFFS,
                                  rvec, tvec_rgb, MARKER_SIZE_METERS)
                cv2.circle(out_frame, (cx, cy), 4, (0, 255, 255), -1)
                if depth_m > 0:
                    cv2.putText(out_frame,
                                f"{tag_id}:{depth_m*1000:.0f}mm",
                                (cx + 6, cy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # --- Annotate frame: gripper axes ---
                r_m = (gripper_pos_cam / 1000.0).reshape(3, 1)
                rvec_k, _ = cv2.Rodrigues(R_ik)
                cv2.drawFrameAxes(out_frame, CAMERA_MATRIX, DIST_COEFFS,
                                  rvec_k, r_m, MARKER_SIZE_METERS * 1.5)

        # Frame number overlay
        cv2.putText(out_frame, f"Frame: {sf + crop_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if detected_this_frame:
            cv2.putText(out_frame, f"Tag {tag_id} detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(out_frame, "No detection", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        writer.write(out_frame)

        if (crop_idx + 1) % 500 == 0:
            print(f"  {crop_idx + 1}/{n_frames} frames processed")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"  Debug video saved: {debug_video_path}")
    print(f"  Detected marker {tag_id} in {n_detected}/{n_frames} frames")
    print(f"  Rotation flips rejected: {n_flips_rejected}")

    if not times:
        print("  No detections. Exiting.")
        return

    # Convert to arrays
    times          = np.array(times)
    u_arr          = np.array(u_list)
    v_arr          = np.array(v_list)
    depth_arr      = np.array(depth_list)
    tvec_pnp_arr   = np.array(tvec_pnp_list)
    rvec_pnp_arr   = np.array(rvec_pnp_list)
    t_ij_arr       = np.array(t_ij_list)
    r_ij_arr       = np.array(r_ij_list)
    grip_pos_cam   = np.array(gripper_pos_cam_list)
    grip_rot_cam   = np.array(gripper_rot_cam_list)
    grip_pos_init  = np.array(gripper_pos_init_list)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n  Raw inputs:")
    print(f"    u:     {u_arr.min():.0f} - {u_arr.max():.0f} px  (range {u_arr.max()-u_arr.min():.0f})")
    print(f"    v:     {v_arr.min():.0f} - {v_arr.max():.0f} px  (range {v_arr.max()-v_arr.min():.0f})")
    print(f"    depth: {depth_arr.min():.0f} - {depth_arr.max():.0f} mm  (range {depth_arr.max()-depth_arr.min():.0f})")
    print(f"    no depth: {np.sum(depth_arr == 0)}/{len(depth_arr)}")

    # ── Plot ──────────────────────────────────────────────────────────
    axis_labels = ("X", "Y", "Z")

    fig, axes = plt.subplots(7, 3, figsize=(16, 20), sharex=True)
    fig.suptitle(f"{DATA_NAME} — Marker {tag_id} debug", fontsize=13)

    # Row 0: u, v, depth
    axes[0][0].plot(times, u_arr, lw=0.8, color="purple")
    axes[0][0].set_ylabel("Raw inputs", fontsize=8)
    axes[0][0].set_title("u (px)")
    axes[0][1].plot(times, v_arr, lw=0.8, color="purple")
    axes[0][1].set_title("v (px)")
    axes[0][2].plot(times, depth_arr, lw=0.8, color="purple")
    axes[0][2].set_title("depth (mm)")

    # Row 1: PnP tvec (mm)
    row_data = [
        ("PnP tvec (mm)", tvec_pnp_arr),
        ("PnP rvec / R_ij Euler (deg)", rvec_pnp_arr),
        ("t_ij depth-based (mm)", t_ij_arr),
        ("Gripper rot cam (Euler deg)", grip_rot_cam),
        ("Gripper pos cam (mm)", grip_pos_cam),
        ("Gripper pos init frame (mm)", grip_pos_init),
    ]
    for row_i, (title, data) in enumerate(row_data, start=1):
        for col in range(3):
            ax = axes[row_i][col]
            ax.plot(times, data[:, col], lw=0.8)
            ax.grid(True, lw=0.4, alpha=0.5)
            if col == 0:
                ax.set_ylabel(title, fontsize=8)
            ax.set_title(f"{axis_labels[col]}", fontsize=9)

    for ax in axes[0]:
        ax.grid(True, lw=0.4, alpha=0.5)
    for col in range(3):
        axes[-1][col].set_xlabel("Time (s)")

    plt.tight_layout()

    # ── Plot 2: marker vs gripper in 2D pixel space ────────────────────
    gripper_uv = np.array(gripper_uv_list)
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig2.suptitle(f"{DATA_NAME} — Marker {tag_id} vs Gripper (pixel coords)", fontsize=13)

    axes2[0].plot(times, u_arr, lw=0.8, label=f"Marker {tag_id} u")
    axes2[0].plot(times, gripper_uv[:, 0], lw=0.8, label="Gripper u (reprojected)")
    axes2[0].set_ylabel("u (px)")
    axes2[0].legend(loc="upper right")
    axes2[0].grid(True, lw=0.4, alpha=0.5)

    axes2[1].plot(times, v_arr, lw=0.8, label=f"Marker {tag_id} v")
    axes2[1].plot(times, gripper_uv[:, 1], lw=0.8, label="Gripper v (reprojected)")
    axes2[1].set_ylabel("v (px)")
    axes2[1].set_xlabel("Time (s)")
    axes2[1].legend(loc="upper right")
    axes2[1].grid(True, lw=0.4, alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
