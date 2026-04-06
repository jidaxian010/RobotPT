import pyrealsense2 as rs
import numpy as np
import cv2

# ==========================================
# --- 1. USER CONFIGURATION ---
# ==========================================

MARKER_SIZE_METERS = 0.0725


def build_T(t, R):
    """Build 4x4 homogeneous transform from translation t (3,) or (3,1) and rotation R (3,3)."""
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    T[0:3, 3:4] = t
    return T


# Define the known r_jk and R_jk for EACH tag ID on your 3D print.
R_10 = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

R_20 = np.array(
    [
        [0.7071, 0.0, 0.7071],
        [0.0, 1.0, 0.0],
        [-0.7071, 0.0, 0.7071],
    ],
    dtype=np.float64,
)

R_30 = np.array(
    [
        [0.0, 0.0, 1.0],
        [-0.6654, 0.7465, 0.0],
        [-0.7465, -0.6654, 0.0],
    ],
    dtype=np.float64,
)

R_40 = np.array(
    [
        [-0.7071, 0.0, 0.7071],
        [0.0, 1.0, 0.0],
        [-0.7071, 0.0, -0.7071],
    ],
    dtype=np.float64,
)

R_50 = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.7465, 0.6654, 0.0],
        [-0.6654, 0.7465, 0.0],
    ],
    dtype=np.float64,
)

TAG_TRANSFORMS = {
    1: build_T(t=[0.3927, 0.0225, -0.2142], R=R_10),  # Tag 1 -> k
    2: build_T(t=[0.3641, 0.0225, 0.0993], R=R_20),   # Tag 2 -> k
    3: build_T(t=[0.3927, -0.0592, -0.2003], R=R_30), # Tag 3 -> k
    4: build_T(t=[0.1912, 0.0225, -0.4561], R=R_40),  # Tag 4 -> k
    5: build_T(t=[0.3927, 0.0928, -0.1703], R=R_50),  # Tag 5 -> k
}

# ==========================================
# --- 2. CAMERA & ARUCO SETUP ---
# ==========================================

half_size = MARKER_SIZE_METERS / 2.0
marker_3d_edges = np.array(
    [
        [-half_size,  half_size, 0.0],
        [ half_size,  half_size, 0.0],
        [ half_size, -half_size, 0.0],
        [-half_size, -half_size, 0.0],
    ],
    dtype=np.float32,
)


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    print("Starting pipeline...")
    profile = pipeline.start(config)

    # Lock FPS / AE priority off (best-effort)
    device = profile.get_device()
    for sensor in device.query_sensors():
        try:
            if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                sensor.set_option(rs.option.auto_exposure_priority, 0)
        except Exception:
            pass

    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.01
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    print("Tracking multi-tag surface...")

    # ==========================================
    # --- 3. MAIN TRACKING LOOP ---
    # ==========================================
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            blur = cv2.GaussianBlur(color_image, (0, 0), sigmaX=5)
            sharp = cv2.addWeighted(color_image, 2.0, blur, -1.0, 0)
            corners, ids, rejected = detector.detectMarkers(sharp)

            # Lists to hold the estimates for frame k from all visible tags
            k_positions = []
            k_rotations = []

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

                for i in range(len(ids)):
                    tag_id = int(ids[i][0])

                    # Only process tags that we have mapped in our dictionary
                    if tag_id not in TAG_TRANSFORMS:
                        continue

                    marker_2d_corners = corners[i][0]  # (4,2)

                    # A. Pose Estimation (i -> j)
                    success, rvec, tvec_rgb = cv2.solvePnP(
                        marker_3d_edges, marker_2d_corners, camera_matrix, dist_coeffs
                    )

                    if not success:
                        continue

                    R_ij, _ = cv2.Rodrigues(rvec)
                    cv2.drawFrameAxes(
                        color_image, camera_matrix, dist_coeffs, rvec, tvec_rgb, MARKER_SIZE_METERS
                    )

                    # Depth refinement (center pixel)
                    center_x = int(np.mean(marker_2d_corners[:, 0]))
                    center_y = int(np.mean(marker_2d_corners[:, 1]))
                    depth_meters = aligned_depth_frame.get_distance(center_x, center_y)

                    if depth_meters > 0:
                        tvec_depth = rs.rs2_deproject_pixel_to_point(
                            intrinsics, [center_x, center_y], depth_meters
                        )
                        t_ij = np.asarray(tvec_depth, dtype=np.float64).reshape(3, 1)
                    else:
                        t_ij = np.asarray(tvec_rgb, dtype=np.float64).reshape(3, 1)

                    T_ij = build_T(t_ij, R_ij)

                    # B. Chain transformations (i -> j -> k)
                    T_jk = TAG_TRANSFORMS[tag_id]
                    T_ik = T_ij @ T_jk

                    # Store this tag's estimate for the k frame
                    k_positions.append(T_ik[0:3, 3:4])  # (3,1)
                    k_rotations.append(T_ik[0:3, 0:3])  # (3,3)

            # ==========================================
            # --- 4. AVERAGE AND DRAW FRAME K ---
            # ==========================================
            if len(k_positions) > 0:
                # Average positions for stability
                r_ik_avg = np.mean(np.stack(k_positions, axis=0), axis=0)  # (3,1)

                # Rotation averaging is non-trivial; use first tag's rotation
                R_ik_final = k_rotations[0]

                rvec_k, _ = cv2.Rodrigues(R_ik_final)

                cv2.drawFrameAxes(
                    color_image,
                    camera_matrix,
                    dist_coeffs,
                    rvec_k,
                    r_ik_avg.astype(np.float64),
                    MARKER_SIZE_METERS * 1.5,
                )

                cv2.putText(
                    color_image,
                    f"Tags Tracking: {len(k_positions)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("RealSense - Multi-Tag Surface Tracker", color_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()