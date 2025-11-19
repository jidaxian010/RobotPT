#!/usr/bin/env python3
"""
Find and track ArUco/AprilTag markers in a video.

- Detects markers frame-by-frame.
- Draws green boxes + red center dot + ID on frames.
- Saves annotated video.
- Returns 4-corner (u,v) trajectories per marker as NumPy arrays
  with shape (T, 4, 2), where T = number of processed frames.
  Missing frames are filled with NaNs.

Usage: just run this file. Edit the config in main() as needed.
"""

import sys
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional


class SelectMarker:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        dict_name: str = "DICT_4X4_50",
        refine_corners: bool = True,
        max_frames: int = 0,
    ):
        """
        Args:
            input_path: path to input video
            output_path: path for annotated output video
            dict_name: OpenCV aruco/apriltag dictionary name
                       e.g., "DICT_4X4_50", "DICT_6X6_250",
                             "DICT_APRILTAG_36h11"
            refine_corners: subpixel refinement for better stability
            max_frames: 0 = process all frames; otherwise limit
        """
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV aruco module not found. Install with:\n  pip install opencv-contrib-python"
            )

        self.input_path = input_path
        self.output_path = output_path
        self.dict_name = dict_name
        self.refine_corners = refine_corners
        self.max_frames = max_frames

        self.dictionary = self._get_dictionary(dict_name)
        self.params = self._build_detector_params(refine_corners)
        self.detect_fn = self._make_detector(self.dictionary, self.params)

        # Storage for trajectories: id -> list of (4,2) arrays per frame, or None
        self._traj: Dict[int, List[Optional[np.ndarray]]] = {}

    # ---------- ArUco helpers ----------

    def _get_dictionary(self, name: str):
        mapping = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_APRILTAG_16H5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25H9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36H10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36H11": cv2.aruco.DICT_APRILTAG_36h11,
        }
        key = name.upper()
        if key not in mapping:
            raise ValueError(
                f"Unknown dictionary '{name}'. Valid: {', '.join(mapping.keys())}"
            )
        return cv2.aruco.getPredefinedDictionary(mapping[key])

    def _build_detector_params(self, refine: bool):
        params = cv2.aruco.DetectorParameters()
        if refine:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return params

    def _make_detector(self, dictionary, params):
        """
        Return a callable detect(gray) -> (corners, ids, rejected)
        that works across OpenCV versions.
        """
        if hasattr(cv2.aruco, "ArucoDetector"):
            det = cv2.aruco.ArucoDetector(dictionary, params)

            def detect(gray):
                return det.detectMarkers(gray)

            return detect
        else:
            def detect(gray):
                return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
            return detect

    # ---------- Drawing ----------

    @staticmethod
    def _draw_marker(frame: np.ndarray, corners: np.ndarray, marker_id: int):
        """
        corners: array (1,4,2) of float -> we reshape to (4,2)
        """
        pts = corners.reshape(4, 2).astype(int)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        center = pts.mean(axis=0).astype(int)
        cv2.circle(frame, tuple(center), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"ID {int(marker_id)}",
            tuple(pts[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # ---------- Trajectory bookkeeping ----------

    def _append_frame_placeholders(self, ids_in_frame: List[int], frame_idx: int):
        """
        Ensure that for every known marker id, we append a placeholder for this frame
        (None) if it wasn't detected. For newly seen ids, backfill previous frames with None.
        """
        # First, backfill new ids
        for mid in ids_in_frame:
            if mid not in self._traj:
                self._traj[mid] = [None] * frame_idx

        # Then, append placeholder for all known ids; actual detections overwrite below
        for mid in self._traj.keys():
            self._traj[mid].append(None)

    def _store_corners(self, marker_id: int, corners: np.ndarray, frame_idx: int):
        """
        Store (4,2) float32 corners for given id at current frame index.
        """
        pts = corners.reshape(4, 2).astype(np.float32)
        self._traj[marker_id][frame_idx] = pts

    def _finalize_trajectories(self) -> Dict[int, np.ndarray]:
        """
        Convert lists (with Nones) to arrays (N,2,T), filling missing with NaNs.
        N=4 corners, 2=(u,v), T=frames
        """
        result: Dict[int, np.ndarray] = {}
        # All lists should have the same length T
        T = max((len(lst) for lst in self._traj.values()), default=0)
        for mid, lst in self._traj.items():
            # Create array with shape (N, 2, T) where N=4 corners
            arr = np.empty((4, 2, T), dtype=np.float32)
            arr[:] = np.nan
            for t, val in enumerate(lst):
                if val is not None:
                    # val has shape (4, 2), assign directly to (4, 2, t)
                    arr[:, :, t] = val
            result[mid] = arr
        return result


    def run(self) -> Dict[int, np.ndarray]:
        """
        Process the video, save annotated output, and return per-marker 4-corner trajectories.

        Returns:
            Dict[int, np.ndarray]: mapping from marker ID to corner trajectory
            Each trajectory has shape (N, 2, T) where N=4 corners, 2=(u,v), T=frames
        """
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.input_path}")

        # Video properties (fallback if unknown)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # If W/H are zero, grab first frame to infer size
        first_frame = None
        if w == 0 or h == 0:
            ok, first_frame = cap.read()
            if not ok:
                raise IOError("Could not read first frame to determine size.")
            h, w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise IOError(f"Cannot open VideoWriter for '{self.output_path}'")

        print(f"Processing {self.input_path} -> {self.output_path}")
        frame_idx = 0

        try:
            # If we peeked, process it first
            if first_frame is not None:
                self._process_one_frame(first_frame, writer, frame_idx)
                frame_idx += 1

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self._process_one_frame(frame, writer, frame_idx)
                frame_idx += 1
                if self.max_frames and frame_idx >= self.max_frames:
                    break
        finally:
            cap.release()
            writer.release()

        # Convert to arrays with NaNs for missing frames
        trajectories = self._finalize_trajectories()
        print(f"Done. Annotated video saved to: {self.output_path}")
        print("Tracked marker IDs:", sorted(trajectories.keys()))
        return trajectories

    def _process_one_frame(self, frame: np.ndarray, writer, frame_idx: int):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detect_fn(gray)

        # Prepare placeholders for this frame for all known ids and any new ones
        ids_in_frame = [] if ids is None else [int(i) for i in ids.flatten()]
        self._append_frame_placeholders(ids_in_frame, frame_idx)

        # Draw + store
        if ids is not None and len(ids) > 0:
            for mid, mc in zip(ids.flatten(), corners):
                mid = int(mid)
                self._draw_marker(frame, mc, mid)
                self._store_corners(mid, mc, frame_idx)

        writer.write(frame)


def main():
    # Simple config (adjust as needed)
    config = {
        "input_path": "posEstimate/data/f1.mp4",
        "output_path": "posEstimate/data/mark_f1.mp4",
        "dict_name": "DICT_5X5_100",   
        "refine_corners": True,
        "max_frames": 0,              
    }

    fm = SelectMarker(
        input_path=config["input_path"],
        output_path=config["output_path"],
        dict_name=config["dict_name"],
        refine_corners=config["refine_corners"],
        max_frames=config["max_frames"],
    )

    trajectories = fm.run()  # Dict[int, np.ndarray(4,2,T)]

    # Example: print shapes
    for mid, arr in trajectories.items():
        print(f"Marker {mid}: traj array shape = {arr.shape}  (4 x 2 x T)")

    # If you need the trajectories programmatically, you have them in `trajectories` now.


if __name__ == "__main__":
    main()
