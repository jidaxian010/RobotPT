import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

class SelectColor:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        color: str,
            rgb_lower: Optional[np.ndarray] = None,
            rgb_upper: Optional[np.ndarray] = None,
        min_area: int = 100,
        max_frames: int = 0,
    ):
        """
        Initialize color-based object tracker.
        
        Args:
            input_path: Path to input video
            output_path: Path for annotated output video
            color: Color name (e.g., "red", "blue", "green", "yellow") or "custom"
            rgb_lower: Lower RGB bound in BGR format (only used if color="custom")
            rgb_upper: Upper RGB bound in BGR format (only used if color="custom")
            min_area: Minimum area in pixels for a valid color blob
            max_frames: 0 = process all frames; otherwise limit
        """
        self.input_path = input_path
        self.output_path = output_path
        self.color = color.lower()
        self.min_area = min_area
        self.max_frames = max_frames
        
        # Get RGB color range (BGR format for OpenCV)
        if color.lower() == "custom":
            if rgb_lower is None or rgb_upper is None:
                raise ValueError("rgb_lower and rgb_upper must be provided when color='custom'")
            self.rgb_lower = rgb_lower
            self.rgb_upper = rgb_upper
        else:
            self.rgb_lower, self.rgb_upper = self._get_color_range(color)
        
        # Storage for center point trajectory: list of (u, v) or None per frame
        self._traj: List[Optional[Tuple[float, float]]] = []
    
    def _get_color_range(self, color: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get RGB color range (BGR format for OpenCV) for common colors.
        
        Returns:
            Tuple of (lower_bound, upper_bound) as numpy arrays in BGR format
        """
        color_ranges = {
            "red": (
                np.array([0, 0, 100]),      # Lower bound (B, G, R)
                np.array([50, 50, 255])    # Upper bound
            ),
            "green": (
                np.array([100, 200, 0]),       # Lower bound (B, G, R) - RGB(0, 194, 107) = BGR(107, 194, 0) ±20
                np.array([180, 255, 40])     # Upper bound
            ),
            "blue": (
                np.array([100, 0, 0]),      # Lower bound (B, G, R)
                np.array([255, 50, 50])     # Upper bound
            ),
            "pink": (
                np.array([150, 68, 235]),   # Lower bound (B, G, R) - RGB(255, 88, 155) = BGR(155, 88, 255) ±20
                np.array([210, 150, 255])   # Upper bound
            ),
            "yellow": (
                np.array([0, 200, 200]),     # Lower bound (B, G, R)
                np.array([50, 255, 255])    # Upper bound
            ),
            "orange": (
                np.array([0, 100, 200]),     # Lower bound (B, G, R)
                np.array([50, 200, 255])    # Upper bound
            ),
            "purple": (
                np.array([100, 0, 100]),    # Lower bound (B, G, R)
                np.array([200, 50, 200])    # Upper bound
            ),
        }

        color_range = color_ranges.get(color, None)
        if color_range is None:
            raise ValueError(
                f"Unknown color '{color}'. Valid colors: {', '.join(color_ranges.keys())}, or 'custom'"
            )
        
        # Return RGB ranges directly (in BGR format for OpenCV)
        return color_range[0], color_range[1]
    
    def _find_color_center(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Find the center point of the largest color blob in the frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            (u, v) center coordinates, or None if no color found
        """
        # Create mask for the color using RGB (BGR format)
        mask = cv2.inRange(frame, self.rgb_lower, self.rgb_upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < self.min_area:
            return None
        
        # Calculate centroid using moments
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        return (float(cx), float(cy))
    
    def _draw_color_blob(self, frame: np.ndarray, center: Optional[Tuple[float, float]], mask: np.ndarray):
        """
        Draw the color blob and center point on the frame.
        
        Args:
            frame: Frame to draw on
            center: Center point (u, v) or None
            mask: Color mask for visualization
        """
        # Draw mask overlay (semi-transparent)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame_overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)
        frame[:] = frame_overlay[:]
        
        if center is not None:
            cx, cy = int(center[0]), int(center[1])
            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"({cx}, {cy})",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    
    def _process_one_frame(self, frame: np.ndarray, writer, frame_idx: int):
        """
        Process a single frame: detect color, find center, draw, and store.
        
        Args:
            frame: Frame to process
            writer: VideoWriter to write annotated frame
            frame_idx: Current frame index
        """
        # Check if frame is grayscale and convert to BGR if needed
        if len(frame.shape) == 2:
            # Grayscale frame - convert to BGR for color tracking
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Find color center
        center = self._find_color_center(frame)
        
        # Create mask for visualization using RGB (BGR format)
        mask = cv2.inRange(frame, self.rgb_lower, self.rgb_upper)
        
        # Draw visualization
        self._draw_color_blob(frame, center, mask)
        
        # Store center point (or None if not found)
        self._traj.append(center)
        
        writer.write(frame)
    
    def run(self):
        """
        Process the video, save annotated output, and return center point trajectory.
        
        Returns:
            List of (u, v) tuples or None for each frame
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
        print(f"Tracking color: {self.color}")
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
        
        print(f"Done. Annotated video saved to: {self.output_path}")
        valid_frames = sum(1 for c in self._traj if c is not None)
        print(f"Color detected in {valid_frames}/{len(self._traj)} frames")
        return self._traj
    
    def process_data(self) -> np.ndarray:
        """
        Process the video and return pixel coordinates for the color blob center.
        Similar interface to SelectItem.process_data() and SelectMarker.process_data().
        
        Returns:
            Numpy array of shape (1, 2, T) where 1=one point, 2=(u,v), T=frames
        """
        # Process video and get trajectory
        trajectory = self.run()  # List of (u, v) or None
        
        if not trajectory:
            raise RuntimeError("No frames processed")
        
        # Convert to array format (1, 2, T), filling missing with NaNs
        T = len(trajectory)
        pixel_array = np.empty((1, 2, T), dtype=np.float32)
        pixel_array[:] = np.nan
        
        for t, center in enumerate(trajectory):
            if center is not None:
                u, v = center
                pixel_array[0, 0, t] = u
                pixel_array[0, 1, t] = v
        
        return pixel_array


def main():
    # Example usage
    config = {
        "input_path": "posEstimate/data/demo.mp4",
        "output_path": "posEstimate/data/demo_color_tracked.mp4",
        "color": "red",  # or "blue", "green", "yellow", etc.
        "min_area": 100,
        "max_frames": 0,
    }
    
    tracker = SelectColor(
        input_path=config["input_path"],
        output_path=config["output_path"],
        color=config["color"],
        min_area=config["min_area"],
        max_frames=config["max_frames"],
    )
    
    pixel_array = tracker.process_data()  # Shape: (1, 2, T)
    print(f"Pixel array shape: {pixel_array.shape}")


if __name__ == "__main__":
    main()
