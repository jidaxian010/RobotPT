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

typestore = get_typestore(Stores.LATEST)

BaseOptions          = mp.tasks.BaseOptions
PoseLandmarker       = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode          = mp.tasks.vision.RunningMode
_CONNECTIONS         = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

MODEL_PATH = str(Path(__file__).resolve().parent.parent /
                 "data" / "pose_landmarker_full.task")


def _draw_pose(frame, landmarks, h, w):
    """Draw skeleton on frame using normalised landmarks."""
    # Draw connections
    for conn in _CONNECTIONS:
        a = landmarks[conn.start]
        b = landmarks[conn.end]
        if a.visibility < 0.3 or b.visibility < 0.3:
            continue
        pt1 = (int(a.x * w), int(a.y * h))
        pt2 = (int(b.x * w), int(b.y * h))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Draw joints
    for lm in landmarks:
        if lm.visibility < 0.3:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 4, (0, 128, 255),   1)


class HumanPose:
    """
    Read RGB frames from a rosbag, run MediaPipe PoseLandmarker on each
    frame, and save an annotated video with skeleton overlays.

    Usage
    -----
        h = HumanPose(bagpath, output_path)
        h.run()
    """

    def __init__(self, bagpath, output_path, is_third_person=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.bagpath      = Path(bagpath)
        self.output_path  = Path(output_path)
        self.is_third_person = is_third_person
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence  = min_tracking_confidence

    def run(self):
        """Stream frames from the bag, annotate with pose, save video."""
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
    # Private helpers
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

        t0_ms = int(timestamps[0] * 1000)
        annotated = []

        with PoseLandmarker.create_from_options(options) as landmarker:
            for i, (bgr, ts) in enumerate(zip(frames, timestamps)):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms  = int(ts * 1000) - t0_ms

                result = landmarker.detect_for_video(mp_img, ts_ms)

                out = bgr.copy()
                if result.pose_landmarks:
                    _draw_pose(out, result.pose_landmarks[0], h, w)
                else:
                    cv2.putText(out, "No pose detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
        bagpath="/home/jdx/Downloads/gripper2",
        output_path="posEstimate/data/gripper2_pose.mp4",
    )
    h.run()


if __name__ == "__main__":
    main()
