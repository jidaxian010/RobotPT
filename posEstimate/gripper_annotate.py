"""
Annotated video utility for gripper pose tracking.

Renders coordinate frame axes (via cv2.drawFrameAxes) onto each detected
ArUco marker and onto the fused gripper reference frame, then writes the
result to an MP4 file.
"""

from pathlib import Path

import cv2
import numpy as np

from gripper_config import TAG_MARKER_SIZE_MM, MARKER_CONFIGS


def save_annotated_video(
    video_path,
    all_marker_trajectories,
    gripper_poses,
    K,
    dist,
    marker_configs=None,
    output_path=None,
):
    """
    Write a copy of *video_path* with coordinate frame axes overlaid on every
    detected ArUco marker and on the averaged gripper frame.

    Args:
        video_path:               str | Path — source (already-cropped) video.
        all_marker_trajectories:  Dict[int, np.ndarray (4, 2, T)] — pixel
                                  corners per marker per frame (NaN = missing).
        gripper_poses:            List[dict | None] length T — output of
                                  compute_gripper_poses_averaged().  Each dict
                                  has keys "position" (3,), "rotation" (3,3),
                                  "frame_idx", "n_markers".
        K:                        (3, 3) float32 camera matrix.
        dist:                     (4,1) or (5,1) distortion coefficients.
        marker_configs:           Dict[int, MarkerConfig] — used for per-marker
                                  size lookup.  Falls back to TAG_MARKER_SIZE_MM.
        output_path:              str | Path — destination file.  Defaults to
                                  <video_stem>_annotated.mp4 beside source.

    Returns:
        Path of the written video, or None on failure.
    """
    if marker_configs is None:
        marker_configs = MARKER_CONFIGS

    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_annotated.mp4"
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  save_annotated_video: cannot open {video_path}")
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Pre-compute 3-D corner coordinates in each marker's local frame (Z=0).
    # OpenCV ArUco order: top-left, top-right, bottom-right, bottom-left.
    def _obj_pts(size_mm):
        h = size_mm / 2.0
        return np.array(
            [[-h,  h, 0.0],
             [ h,  h, 0.0],
             [ h, -h, 0.0],
             [-h, -h, 0.0]],
            dtype=np.float32,
        )

    marker_obj_pts = {}
    for mid in all_marker_trajectories:
        cfg = marker_configs.get(mid)
        size = cfg.marker_size_mm if cfg else TAG_MARKER_SIZE_MM
        marker_obj_pts[mid] = _obj_pts(size)

    gripper_axis_len = 100.0   # mm — larger than any individual marker

    frame_t = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Draw each detected marker's coordinate frame ──────────────────
        for mid, corners_4_2_T in all_marker_trajectories.items():
            if frame_t >= corners_4_2_T.shape[2]:
                continue
            img_pts = corners_4_2_T[:, :, frame_t].astype(np.float32)
            if np.any(np.isnan(img_pts)):
                continue

            ok, rvec, tvec = cv2.solvePnP(
                marker_obj_pts[mid], img_pts, K, dist,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if not ok:
                continue

            cfg = marker_configs.get(mid)
            axis_len = (cfg.marker_size_mm if cfg else TAG_MARKER_SIZE_MM) * 0.8

            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, axis_len)

            # Label the marker ID at its centroid
            cx = int(np.mean(img_pts[:, 0]))
            cy = int(np.mean(img_pts[:, 1]))
            cv2.putText(
                frame, f"#{mid}",
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA,
            )

        # ── Draw the fused gripper frame ───────────────────────────────────
        if frame_t < len(gripper_poses) and gripper_poses[frame_t] is not None:
            gp = gripper_poses[frame_t]
            rvec_g, _ = cv2.Rodrigues(gp["rotation"].astype(np.float64))
            tvec_g    = gp["position"].reshape(3, 1).astype(np.float64)

            cv2.drawFrameAxes(frame, K, dist, rvec_g, tvec_g, gripper_axis_len)

            n = gp.get("n_markers", "?")
            cv2.putText(
                frame, f"Gripper ({n} tags)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2, cv2.LINE_AA,
            )

        writer.write(frame)
        frame_t += 1

    cap.release()
    writer.release()
    print(f"  Saved annotated video ({frame_t} frames): {output_path}")
    return output_path
