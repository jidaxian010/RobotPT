import sys
from pathlib import Path
import bisect

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Allow imports from posEstimate/ regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosbag_reader import RosbagVideoReader, AnyReader, typestore
from gripper_visualize import (
    plot_markers_camera_frame,
    plot_gripper_camera_frame,
    plot_gripper_body_frame,
)

# ==========================================
# --- 1. USER CONFIGURATION ---
# ==========================================

MARKER_SIZE_METERS = 0.0725

# Rosbag/video config (same RosbagVideoReader path/topic conventions as gripper_pose.py)
BAG_PATH = Path("/home/jdx/Downloads/pose1")
VIDEO_PATH = Path("posEstimate/data/pose1.mp4")
IS_THIRD_PERSON = True
SKIP_FIRST_N = 0
SKIP_LAST_N = 0
EXTRACT_RGB_VIDEO = True  # Set False to reuse an existing VIDEO_PATH

# Camera intrinsics (copied from gripper_pose.py; 848x480 RealSense stream)
_FX = 602.6597900390625
_FY = 602.2169799804688
_CX = 423.1910400390625
_CY = 249.92578125

CAMERA_MATRIX = np.array(
    [[_FX, 0.0, _CX], [0.0, _FY, _CY], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float32)


def build_T(t, R):
    """Build 4x4 homogeneous transform from translation t (3,) or (3,1) and rotation R (3,3)."""
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


def _get_depth_topic_name(is_third_person):
    if is_third_person:
        return "/third_person_cam/camera/camera/aligned_depth_to_color/image_raw"
    return "/left_camera/camera/camera/aligned_depth_to_color/image_raw"


def _extract_depth_frames_with_timestamps(video_reader):
    """
    Read aligned depth frames from rosbag and return [(timestamp_sec, depth_img_uint16), ...]
    using the same topic/skip settings as RosbagVideoReader.
    """
    topic_name = _get_depth_topic_name(video_reader.is_third_person)
    depth_data = []
    frame_count = 0

    with AnyReader([video_reader.bagpath], default_typestore=typestore) as reader:
        depth_conn = None
        for conn in reader.connections:
            if conn.topic == topic_name:
                depth_conn = conn
                break
        if depth_conn is None:
            raise RuntimeError(f"Depth topic {topic_name} not found in bag file")

        for conn, ts, raw in reader.messages(connections=[depth_conn]):
            if frame_count < video_reader.skip_first_n:
                frame_count += 1
                continue

            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            try:
                try:
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                except Exception:
                    timestamp = ts * 1e-9

                height = msg.height
                width = msg.width
                step = msg.step
                data_bytes = msg.data if isinstance(msg.data, bytes) else bytes(msg.data)
                depth_img = np.frombuffer(data_bytes, dtype=np.uint16)
                pixels_per_row = step // 2
                depth_img = depth_img.reshape((height, pixels_per_row))[:, :width]
                depth_data.append((timestamp, depth_img))
            except Exception as e:
                print(f"Warning: Failed to convert depth frame: {e}")

            frame_count += 1

    if video_reader.skip_last_n > 0 and len(depth_data) > video_reader.skip_last_n:
        depth_data = depth_data[:-video_reader.skip_last_n]

    if not depth_data:
        raise RuntimeError("No valid depth frames found")

    return depth_data


def find_depth_by_rgb_timestamp(video_reader, pixel_array, max_dt_s=0.05):
    """
    Timestamp-based depth lookup for [u, v, rgb_frame_idx] queries.
    Matches each RGB frame index to the nearest depth frame by timestamp.
    Returns (N,4) => [u, v, depth_mm, rgb_frame_idx]
    """
    pixel_array = np.asarray(pixel_array, dtype=np.float64)
    if pixel_array.ndim != 2 or pixel_array.shape[1] != 3:
        raise ValueError(f"Expected pixel_array shape (N,3), got {pixel_array.shape}")

    rgb_timestamps = video_reader.get_rgb_timestamps()
    depth_data = _extract_depth_frames_with_timestamps(video_reader)
    depth_timestamps = [t for t, _ in depth_data]

    result = np.zeros((len(pixel_array), 4), dtype=np.float64)

    for i, (u, v, rgb_frame_idx_f) in enumerate(pixel_array):
        u_int, v_int = int(u), int(v)
        rgb_frame_idx = int(rgb_frame_idx_f)

        depth_value = 0.0
        if 0 <= rgb_frame_idx < len(rgb_timestamps):
            target_t = float(rgb_timestamps[rgb_frame_idx])
            j = bisect.bisect_left(depth_timestamps, target_t)
            candidates = []
            if j > 0:
                candidates.append(j - 1)
            if j < len(depth_timestamps):
                candidates.append(j)

            if candidates:
                best_j = min(candidates, key=lambda k: abs(depth_timestamps[k] - target_t))
                dt = abs(depth_timestamps[best_j] - target_t)
                if dt <= max_dt_s:
                    depth_img = depth_data[best_j][1]
                    if 0 <= v_int < depth_img.shape[0] and 0 <= u_int < depth_img.shape[1]:
                        depth_value = float(depth_img[v_int, u_int])

        result[i] = [u, v, depth_value, rgb_frame_idx]

    return result


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
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    video_reader = RosbagVideoReader(
        BAG_PATH,
        VIDEO_PATH,
        is_third_person=IS_THIRD_PERSON,
        skip_first_n=SKIP_FIRST_N,
        skip_last_n=SKIP_LAST_N,
    )

    if EXTRACT_RGB_VIDEO or not VIDEO_PATH.exists():
        print(f"Extracting RGB video from rosbag: {BAG_PATH}")
        video_reader.process_data()

    # ------------------------------------------
    # Pass 1: detect markers + collect depth queries
    # ------------------------------------------
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing extracted video: {VIDEO_PATH} ({frame_count_est} frames @ {fps:.2f} fps)")

    per_frame_detections = []
    depth_queries = []  # [u, v, frame_idx] for RosbagVideoReader.find_depth
    query_refs = []     # (frame_idx, det_idx_within_frame)

    frame_idx = 0
    while True:
        ok, color_image = cap.read()
        if not ok:
            break

        corners, ids, _ = detector.detectMarkers(color_image)
        frame_dets = []

        if ids is not None:
            for i in range(len(ids)):
                tag_id = int(ids[i][0])
                if tag_id not in TAG_TRANSFORMS:
                    continue

                marker_2d_corners = corners[i][0].astype(np.float32)  # (4,2)
                success, rvec, tvec_rgb = cv2.solvePnP(
                    marker_3d_edges, marker_2d_corners, CAMERA_MATRIX, DIST_COEFFS
                )
                if not success:
                    continue

                R_ij, _ = cv2.Rodrigues(rvec)
                center_x = int(np.mean(marker_2d_corners[:, 0]))
                center_y = int(np.mean(marker_2d_corners[:, 1]))

                det = {
                    "tag_id": tag_id,
                    "corners_draw": corners[i],  # shape (1,4,2), for drawDetectedMarkers
                    "marker_2d_corners": marker_2d_corners,
                    "rvec": rvec,
                    "tvec_rgb": tvec_rgb,
                    "R_ij": R_ij,
                    "center": (center_x, center_y),
                    "depth_m": 0.0,
                }
                frame_dets.append(det)

                depth_queries.append([center_x, center_y, frame_idx])
                query_refs.append((frame_idx, len(frame_dets) - 1))

        per_frame_detections.append(frame_dets)
        frame_idx += 1

    cap.release()

    # ------------------------------------------
    # Batch depth lookup from rosbag (same topics as gripper_pose.py)
    # ------------------------------------------
    if depth_queries:
        depth_results = find_depth_by_rgb_timestamp(
            video_reader,
            np.asarray(depth_queries, dtype=np.float64),
            max_dt_s=0.05,
        )
        for q_idx, (f_idx, det_idx) in enumerate(query_refs):
            depth_mm = float(depth_results[q_idx, 2])
            per_frame_detections[f_idx][det_idx]["depth_m"] = depth_mm / 1000.0
        print(f"Depth lookup complete for {len(depth_queries)} marker detections.")
    else:
        print("No mapped markers detected in video.")

    # Build pose outputs for plotting (same data shape expected by gripper_visualize.py)
    num_frames = len(per_frame_detections)
    all_raw_poses = {}
    gripper_poses = [None] * num_frames

    for f_idx, frame_dets in enumerate(per_frame_detections):
        k_positions = []
        k_rotations = []
        for det in frame_dets:
            tag_id = det["tag_id"]
            R_ij = det["R_ij"]
            tvec_rgb = det["tvec_rgb"]
            center_x, center_y = det["center"]
            depth_meters = det["depth_m"]

            if depth_meters > 0:
                t_ij = deproject_pixel_depth_to_point_m(center_x, center_y, depth_meters).reshape(3, 1)
            else:
                t_ij = np.asarray(tvec_rgb, dtype=np.float64).reshape(3, 1)

            T_ij = build_T(t_ij, R_ij)
            T_ik = T_ij @ TAG_TRANSFORMS[tag_id]

            det["t_ij"] = t_ij
            det["T_ik"] = T_ik

            if tag_id not in all_raw_poses:
                all_raw_poses[tag_id] = [None] * num_frames
            all_raw_poses[tag_id][f_idx] = {
                "position": (t_ij[:, 0] * 1000.0),  # mm
                "rotation": R_ij.copy(),
                "frame_idx": f_idx,
            }

            k_positions.append(T_ik[0:3, 3:4])
            k_rotations.append(T_ik[0:3, 0:3])

        if k_positions:
            r_ik_avg = np.mean(np.stack(k_positions, axis=0), axis=0)
            R_ik_final = k_rotations[0]
            gripper_poses[f_idx] = {
                "position": (r_ik_avg[:, 0] * 1000.0),  # mm
                "rotation": R_ik_final.copy(),
                "frame_idx": f_idx,
            }

    rgb_timestamps = video_reader.get_rgb_timestamps()
    rgb_timestamps = rgb_timestamps[:num_frames]

    # ------------------------------------------
    # Pass 2: render playback with gripper frame estimate
    # ------------------------------------------
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    annotated_tmp_path = VIDEO_PATH.with_name(f"{VIDEO_PATH.stem}.annotated_tmp{VIDEO_PATH.suffix}")
    writer = cv2.VideoWriter(
        str(annotated_tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {annotated_tmp_path}")

    print(f"Playing back gripper-frame estimates (press 'q' to quit). Saving annotated video to {VIDEO_PATH}")
    frame_idx = 0
    try:
        while True:
            ok, color_image = cap.read()
            if not ok or frame_idx >= len(per_frame_detections):
                break

            frame_dets = per_frame_detections[frame_idx]

            if frame_dets:
                draw_corners = [d["corners_draw"] for d in frame_dets]
                draw_ids = np.array([[d["tag_id"]] for d in frame_dets], dtype=np.int32)
                cv2.aruco.drawDetectedMarkers(color_image, draw_corners, draw_ids)

            # Lists to hold the estimates for frame k from all visible tags
            k_positions = []
            k_rotations = []

            for det in frame_dets:
                tag_id = det["tag_id"]
                rvec = det["rvec"]
                tvec_rgb = det["tvec_rgb"]
                R_ij = det["R_ij"]
                center_x, center_y = det["center"]
                depth_meters = det["depth_m"]

                cv2.drawFrameAxes(
                    color_image, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec_rgb, MARKER_SIZE_METERS
                )

                # Use precomputed depth-refined pose for consistency with plots
                T_ik = det["T_ik"]

                k_positions.append(T_ik[0:3, 3:4])  # (3,1)
                k_rotations.append(T_ik[0:3, 0:3])  # (3,3)

                cv2.circle(color_image, (center_x, center_y), 4, (0, 255, 255), -1)
                if depth_meters > 0:
                    cv2.putText(
                        color_image,
                        f"{tag_id}:{depth_meters*1000:.0f}mm",
                        (center_x + 6, center_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                    )

            if k_positions:
                r_ik_avg = np.mean(np.stack(k_positions, axis=0), axis=0)  # (3,1)
                R_ik_final = k_rotations[0]  # simple fallback rotation choice
                rvec_k, _ = cv2.Rodrigues(R_ik_final)

                cv2.drawFrameAxes(
                    color_image,
                    CAMERA_MATRIX,
                    DIST_COEFFS,
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

            cv2.putText(
                color_image,
                f"Frame: {frame_idx}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Rosbag - Multi-Tag Surface Tracker", color_image)
            writer.write(color_image)

            # Approximate realtime playback using extracted video FPS
            wait_ms = max(1, int(round(1000.0 / fps)))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # Replace source video with annotated output (user-requested VIDEO_PATH target)
    if annotated_tmp_path.exists():
        annotated_tmp_path.replace(VIDEO_PATH)
        print(f"Saved annotated playback video to {VIDEO_PATH}")

    # Plots (same style/helpers as gripper_pose.py)
    poses_for_plot = {mid: poses for mid, poses in all_raw_poses.items() if any(p is not None for p in poses)}
    if poses_for_plot:
        plot_markers_camera_frame(poses_for_plot, rgb_timestamps)
    else:
        print("No marker poses available for plot 1.")

    if any(p is not None for p in gripper_poses):
        plot_gripper_camera_frame(gripper_poses, rgb_timestamps)
        plot_gripper_body_frame(gripper_poses, rgb_timestamps)
    else:
        print("No gripper poses available for plot 2/3.")

    plt.show()


if __name__ == "__main__":
    main()
