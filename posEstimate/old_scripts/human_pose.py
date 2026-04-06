import sys
from pathlib import Path

# Allow imports from posEstimate/ regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import mediapipe as mp
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores
from rosbag_reader import RosbagVideoReader
from joint_tracker import PoseTracker, VIS_THRESHOLD

typestore = get_typestore(Stores.LATEST)

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode
_CONNECTIONS          = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

MODEL_PATH = str(Path(__file__).resolve().parent.parent /
                 "data" / "pose_landmarker_full.task")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _vis_color(vis: float):
    """
    Map visibility [0, 1] → BGR color:
        1.0 → green  (0, 255, 0)
        0.5 → yellow (0, 255, 255)
        0.0 → red    (0, 0, 255)
    """
    v = max(0.0, min(1.0, vis))
    if v >= 0.5:
        r = int(255 * 2 * (1.0 - v))
        return (0, 255, r)
    else:
        g = int(255 * 2 * v)
        return (0, g, 255)


def _draw_pose(frame, landmarks, tracked_states, h, w):
    """
    Draw skeleton using Kalman-smoothed positions coloured by raw visibility.

    Position  → from tracker (smoothed, outlier-rejected)
    Colour    → from raw MediaPipe visibility (green=confident, red=uncertain)
    Fallback  → raw landmark pixel if joint not yet initialised in tracker
    """
    def _pt(i):
        """Return (x, y) pixel for joint i: tracked if available, else raw."""
        s = tracked_states[i] if tracked_states else None
        if s is not None:
            return (int(s.position[0]), int(s.position[1]))
        lm = landmarks[i]
        return (int(lm.x * w), int(lm.y * h))

    # Connections
    for conn in _CONNECTIONS:
        a, b = landmarks[conn.start], landmarks[conn.end]
        vis = min(a.visibility, b.visibility)
        if vis < 0.15:
            continue
        cv2.line(frame, _pt(conn.start), _pt(conn.end), _vis_color(vis), 2)

    # Joints
    for i, lm in enumerate(landmarks):
        if lm.visibility < 0.15:
            continue
        cv2.circle(frame, _pt(i), 5, _vis_color(lm.visibility), -1)



def _draw_vis_legend(frame, h, w):
    """Draw a small green→yellow→red colorbar in the top-right corner."""
    bar_w, bar_h = 12, 80
    x0 = w - bar_w - 8
    y0 = 8
    for row in range(bar_h):
        vis = 1.0 - row / bar_h
        color = _vis_color(vis)
        cv2.rectangle(frame, (x0, y0 + row), (x0 + bar_w, y0 + row + 1), color, -1)
    cv2.putText(frame, "vis", (x0 - 2, y0 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(frame, "1.0", (x0 + bar_w + 2, y0 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.putText(frame, "0.0", (x0 + bar_w + 2, y0 + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HumanPose:
    """
    Read RGB frames from a rosbag, run MediaPipe PoseLandmarker on each
    frame, apply per-joint Kalman filtering, and save an annotated video.

    Visualisation layers
    --------------------
    Raw (visibility-colored):  green = fully visible, yellow = partial, red = occluded
    Tracked (cyan):            Kalman-smoothed positions; sigma-circles show uncertainty
    """

    def __init__(self, bagpath, output_path, is_third_person=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 ema_alpha=0.35, ema_dead_band=3.0):
        self.bagpath      = Path(bagpath)
        self.output_path  = Path(output_path)
        self.is_third_person = is_third_person
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence  = min_tracking_confidence
        self.ema_alpha     = ema_alpha
        self.ema_dead_band = ema_dead_band

    def run(self):
        topic = (
            "/third_person_cam/camera/camera/color/image_raw"
            if self.is_third_person
            else "/left_camera/camera/camera/color/image_raw"
        )

        frames, timestamps = self._read_frames(topic)
        if not frames:
            print("No frames found — check bag path and topic.")
            return

        fps = self._compute_fps(timestamps)
        h, w = frames[0].shape[:2]

        print(f"Running MediaPipe PoseLandmarker on {len(frames)} frames ...")
        annotated = self._annotate_frames(frames, timestamps, h, w)

        self._write_video(annotated, fps, w, h)
        print(f"Saved annotated video: {self.output_path}")

    # ------------------------------------------------------------------

    def _read_frames(self, topic):
        print(f"Reading '{topic}' from {self.bagpath}")
        frames, timestamps = [], []

        with AnyReader([self.bagpath], default_typestore=typestore) as reader:
            conns = [c for c in reader.connections if c.topic == topic]
            if not conns:
                raise RuntimeError(f"Topic '{topic}' not found in bag.")

            for conn, ts, raw in reader.messages(connections=conns):
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                try:
                    img = RosbagVideoReader.ros_image_to_cv2(msg)
                    frames.append(img)
                    try:
                        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    except Exception:
                        t = ts * 1e-9
                    timestamps.append(t)
                except Exception as e:
                    print(f"  Warning: skipping frame — {e}")

        print(f"Extracted {len(frames)} frames")
        return frames, timestamps

    @staticmethod
    def _compute_fps(timestamps):
        if len(timestamps) > 1:
            fps = 1.0 / np.median(np.diff(timestamps))
            print(f"FPS: {fps:.2f}")
            return fps
        return 30.0

    def _annotate_frames(self, frames, timestamps, h, w):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        # EMA smoother in pixel-space (x*w, y*h, z*w)
        tracker = PoseTracker(alpha=self.ema_alpha, dead_band=self.ema_dead_band)
        dts = [0.0] + list(np.diff(timestamps))

        t0_ms    = int(timestamps[0] * 1000)
        annotated = []

        with PoseLandmarker.create_from_options(options) as landmarker:
            for i, (bgr, ts, dt) in enumerate(zip(frames, timestamps, dts)):
                rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms  = int(ts * 1000) - t0_ms

                result = landmarker.detect_for_video(mp_img, ts_ms)

                out = bgr.copy()

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]

                    # Build (33, 3) pixel-space positions and (33,) visibilities
                    positions    = np.array([[lm.x * w, lm.y * h, lm.z * w]
                                             for lm in lms])
                    visibilities = np.array([lm.visibility for lm in lms])

                    tracked = tracker.update(positions, visibilities, dt)
                    _draw_pose(out, lms, tracked, h, w)
                else:
                    cv2.putText(out, "No pose detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # Still run tracker in predict-only mode
                    dummy_vis = np.zeros(33)
                    dummy_pos = np.zeros((33, 3))
                    tracker.update(dummy_pos, dummy_vis, dt)

                _draw_vis_legend(out, h, w)
                annotated.append(out)

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(frames)} frames")

        return annotated

    def _write_video(self, frames, fps, w, h):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    h = HumanPose(
        bagpath="/home/jdx/Downloads/pose1",
        output_path="posEstimate/data/pose1_human.mp4",
    )
    h.run()


if __name__ == "__main__":
    main()
