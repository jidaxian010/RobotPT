import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from rosbag_reader import RosbagVideoReader
from object_detect.select_pixel import SelectItem
from object_detect.select_marker import SelectMarker
from object_detect.select_color import SelectColor


class MultiColorTracker:
    """Track multiple colors simultaneously and create combined output video."""
    
    def __init__(self, input_path: str, output_path: str, colors: list, min_area: int = 100):
        """
        Initialize multi-color tracker.
        
        Args:
            input_path: Path to input video
            output_path: Path for annotated output video
            colors: List of color names to track (e.g., ["green", "pink"])
            min_area: Minimum area in pixels for a valid color blob
        """
        self.input_path = input_path
        self.output_path = output_path
        self.colors = colors
        self.min_area = min_area
        
        # Initialize individual color trackers to get color ranges
        self.color_trackers = {}
        for color in colors:
            tracker = SelectColor(input_path, "", color=color, min_area=min_area)
            self.color_trackers[color] = tracker
        
        # Storage for trajectories: color -> list of (u, v) or None per frame
        self._traj: dict = {color: [] for color in colors}
    
    def _find_color_center(self, frame: np.ndarray, color: str) -> tuple:
        """Find center of color blob in frame."""
        tracker = self.color_trackers[color]
        
        # Check if frame is grayscale and convert to BGR if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Create mask for the color using RGB (BGR format)
        mask = cv2.inRange(frame, tracker.rgb_lower, tracker.rgb_upper)
        
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
    
    def _draw_color_blob(self, frame: np.ndarray, center: tuple, color_name: str, color_bgr: tuple):
        """Draw color blob and center point on frame."""
        if center is not None:
            cx, cy = int(center[0]), int(center[1])
            # Draw center point with color-specific color
            cv2.circle(frame, (cx, cy), 5, color_bgr, -1)
            cv2.circle(frame, (cx, cy), 10, color_bgr, 2)
            cv2.putText(
                frame,
                f"{color_name} ({cx}, {cy})",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_bgr,
                1,
                cv2.LINE_AA,
            )
    
    def _process_one_frame(self, frame: np.ndarray, writer, frame_idx: int):
        """Process a single frame: detect all colors, draw, and store."""
        # Check if frame is grayscale and convert to BGR if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Color-specific BGR colors for drawing (B, G, R)
        color_draw_colors = {
            "green": (0, 255, 0),      # Green in BGR
            "pink": (203, 192, 255),   # Pink in BGR
            "red": (0, 0, 255),       # Red in BGR
            "blue": (255, 0, 0),      # Blue in BGR
            "yellow": (0, 255, 255),   # Yellow in BGR
        }
        
        # Find and draw each color
        for color in self.colors:
            center = self._find_color_center(frame, color)
            draw_color = color_draw_colors.get(color, (255, 255, 255))
            self._draw_color_blob(frame, center, color, draw_color)
            
            # Store center point (or None if not found)
            self._traj[color].append(center)
        
        writer.write(frame)
    
    def run(self):
        """
        Process the video, save annotated output, and return center point trajectories.
        
        Returns:
            Dict mapping color name to list of (u, v) tuples or None for each frame
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
        print(f"Tracking colors: {', '.join(self.colors)}")
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
        finally:
            cap.release()
            writer.release()
        
        print(f"Done. Annotated video saved to: {self.output_path}")
        for color in self.colors:
            valid_frames = sum(1 for c in self._traj[color] if c is not None)
            print(f"  {color}: detected in {valid_frames}/{len(self._traj[color])} frames")
        
        return self._traj
    
    def process_data(self) -> dict:
        """
        Process the video and return pixel coordinates for all tracked colors.
        
        Returns:
            Dict mapping color name to numpy array of shape (1, 2, T)
        """
        # Process video and get trajectories
        trajectories = self.run()  # Dict[color] -> List of (u, v) or None
        
        if not trajectories:
            raise RuntimeError("No frames processed")
        
        # Convert to array format for each color
        result = {}
        for color, trajectory in trajectories.items():
            if not trajectory:
                continue
            
            T = len(trajectory)
            pixel_array = np.empty((1, 2, T), dtype=np.float32)
            pixel_array[:] = np.nan
            
            for t, center in enumerate(trajectory):
                if center is not None:
                    u, v = center
                    pixel_array[0, 0, t] = u
                    pixel_array[0, 1, t] = v
            
            result[color] = pixel_array
        
        return result


class CameraOdometry:
    def __init__(self, pixel_depth_array):
        # 848x480x60
        self.fx = 602.6597900390625
        self.fy = 602.2169799804688
        self.cx = 423.1910400390625
        self.cy = 249.92578125
        self.pixel_depth_array = pixel_depth_array # (u, v, d, frame_idx)

    def plot_3d_trajectory(self, odometry_array):
        """
        Input: odometry_array = (x, y, z, frame_idx)
        Plot: 3D trajectory with equal numeric scaling on all axes
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory

        posx = odometry_array[:, 0]
        posy = odometry_array[:, 2]
        posz = -odometry_array[:, 1]
        ax.plot(posx, posy, posz)
        ax.scatter(0, 0, 0, color='red', s=100, label='Camera')
        
        ax.axis('equal')

        ax.legend()
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Trajectory (Equal Axis Scale)')

        plt.show()
    
    def plot_3d_trajectory_multi(self, odometry_arrays: dict, colors_dict: dict):
        """
        Plot multiple 3D trajectories with different colors.
        
        Args:
            odometry_arrays: Dict mapping color name to odometry_array (x, y, z, frame_idx)
            colors_dict: Dict mapping color name to matplotlib color string
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot each trajectory
        for color_name, odometry_array in odometry_arrays.items():
            if len(odometry_array) == 0:
                continue
            
            posx = odometry_array[:, 0]
            posy = odometry_array[:, 2]
            posz = -odometry_array[:, 1]
            
            plot_color = colors_dict.get(color_name, 'blue')
            ax.plot(posx, posy, posz, color=plot_color, label=color_name, linewidth=2)
            # Plot start point
            if len(posx) > 0:
                ax.scatter(posx[0], posy[0], posz[0], color=plot_color, s=100, marker='o')
        
        ax.scatter(0, 0, 0, color='red', s=100, label='Camera', marker='x')
        
        ax.axis('equal')
        ax.legend()
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Trajectory - Multi-Color Tracking')
        
        plt.show()

    def process_data(self):
        """
        Input: pixel_depth_array = (u, v, d, frame_idx)
        Output: odometry_array = (x, y, z, frame_idx)
        """
        from scipy.signal import medfilt
        
        # Extract depth values
        depths = self.pixel_depth_array[:, 2].copy()
        
        # Remove jumps based on percentage change
        max_change_percent = 30  # Max 30% change between consecutive frames
        
        for i in range(1, len(depths)):
            if depths[i] > 0 and depths[i-1] > 0:
                change_percent = abs(depths[i] - depths[i-1]) / depths[i-1] * 100
                if change_percent > max_change_percent:
                    depths[i] = depths[i-1]  # Keep previous value
        
        # Smooth with median filter
        smoothed_depths = medfilt(depths, kernel_size=7)
        
        odometry_array = []
        for i in range(len(self.pixel_depth_array)):
            u = self.pixel_depth_array[i, 0]
            v = self.pixel_depth_array[i, 1]
            d = smoothed_depths[i]
            frame_idx = self.pixel_depth_array[i, 3]
            
            # Skip rows where depth is 0
            if d == 0:
                continue
            
            x = (u - self.cx) * d / self.fx
            y = (v - self.cy) * d / self.fy
            odometry_array.append([x, y, d, frame_idx])
        return np.array(odometry_array) if odometry_array else np.array([]).reshape(0, 4)


def convert_pixel_array_to_depth_format(pixel_array):
    """
    Convert pixel array from (N, 2, T) format to (N, 3) format expected by find_depth.
    
    Args:
        pixel_array: Numpy array of shape (N, 2, T) where N=points, 2=(u,v), T=frames
        
    Returns:
        Numpy array of shape (N*T, 3) with columns [u, v, frame_idx]
    """
    N, _, T = pixel_array.shape
    result = []
    
    for t in range(T):
        for n in range(N):
            u, v = pixel_array[n, :, t]
            if not np.isnan(u) and not np.isnan(v):  # Skip NaN values
                result.append([u, v, t])
    
    return np.array(result)


def find_odometry_data(bagpath, video_path, select_method="manual", marker_id=None, corner_idx=0, playback_speed=1, color="red", min_area=100):
    """
    Find odometry data using manual point selection, ArUco marker tracking, or color-based tracking.
    
    Args:
        bagpath: Path to rosbag file
        video_path: Path to video file
        select_method: "manual" for manual point selection, "marker" for ArUco marker tracking, "color" for color-based tracking
        marker_id: Specific marker ID to track (only used when select_method="marker")
        corner_idx: Which corner to track (0-3, only used when select_method="marker")
        playback_speed: Speed multiplier for video playback (only used when select_method="manual")
        color: Color name to track (only used when select_method="color"). Options: "red", "blue", "green", "yellow", "orange", "purple", or "custom"
        min_area: Minimum area in pixels for a valid color blob (only used when select_method="color")
        
    Returns:
        Tuple of (pixel_array, pixel_depth_array, odometry_array)
    """
    # Read video and save video
    Videoreader = RosbagVideoReader(Path(bagpath), Path(video_path), is_third_person=True, skip_first_n=0, skip_last_n=0)
    Videoreader.process_data()
    Videoreader.save_depth_video()


    # Get pixel_array: (1, 2, T)
    if select_method == "manual":
        tracker = SelectItem(video_path, playback_speed=playback_speed)
        pixel_array = tracker.process_data(show_tracking=True)  # Shape: (1, 2, T)
        
    elif select_method == "marker":
        video_path_obj = Path(video_path)
        output_path = video_path_obj.parent / f"{video_path_obj.stem}_marked.mp4"
        
        marker_tracker = SelectMarker(
            input_path=video_path,
            output_path=str(output_path),
            dict_name="DICT_4X4_50"
        )
        pixel_array = marker_tracker.process_data(marker_id=marker_id, corner_idx=corner_idx)  # Shape: (1, 2, T)
    elif select_method == "color":
        video_path_obj = Path(video_path)
        output_path = video_path_obj.parent / f"{video_path_obj.stem}_color_tracked.mp4"
        
        color_tracker = SelectColor(
            input_path=video_path,
            output_path=str(output_path),
            color=color,
            min_area=min_area
        )
        pixel_array = color_tracker.process_data()  # Shape: (1, 2, T)
        
    else:
        raise ValueError(f"Invalid select_method: {select_method}. Must be 'manual', 'marker', or 'color'")
    
    # Calculate 3D position
    pixel_array_for_depth = convert_pixel_array_to_depth_format(pixel_array)
    pixel_depth_array = Videoreader.find_depth(pixel_array_for_depth)
    camera_odometry = CameraOdometry(pixel_depth_array)
    odometry_array = camera_odometry.process_data()

    return pixel_array, pixel_depth_array, odometry_array # (1, 2, T), (u, v, d, frame_idx), (x, y, z, frame_idx)


def find_multi_color_odometry(bagpath, video_path, colors: list, min_area: int = 100):
    """
    Find odometry data tracking multiple colors simultaneously.
    
    Args:
        bagpath: Path to rosbag file
        video_path: Path to video file
        colors: List of color names to track (e.g., ["green", "pink"])
        min_area: Minimum area in pixels for a valid color blob
        
    Returns:
        Tuple of (pixel_arrays_dict, pixel_depth_arrays_dict, odometry_arrays_dict)
        Each dict maps color name to the corresponding array
    """
    # Read video and save video
    Videoreader = RosbagVideoReader(Path(bagpath), Path(video_path), is_third_person=True, skip_first_n=0, skip_last_n=0)
    Videoreader.process_data()
    Videoreader.save_depth_video()

    # Create output video path
    video_path_obj = Path(video_path)
    output_path = video_path_obj.parent / f"{video_path_obj.stem}_multi_color_tracked.mp4"
    
    # Track multiple colors simultaneously
    multi_tracker = MultiColorTracker(
        input_path=video_path,
        output_path=str(output_path),
        colors=colors,
        min_area=min_area
    )
    pixel_arrays_dict = multi_tracker.process_data()  # Dict[color] -> (1, 2, T)
    
    # Calculate 3D positions for each color
    pixel_depth_arrays_dict = {}
    odometry_arrays_dict = {}
    
    for color, pixel_array in pixel_arrays_dict.items():
        # Convert to depth format
        pixel_array_for_depth = convert_pixel_array_to_depth_format(pixel_array)
        pixel_depth_array = Videoreader.find_depth(pixel_array_for_depth)
        pixel_depth_arrays_dict[color] = pixel_depth_array
        
        # Calculate odometry
        camera_odometry = CameraOdometry(pixel_depth_array)
        odometry_array = camera_odometry.process_data()
        odometry_arrays_dict[color] = odometry_array
    
    return pixel_arrays_dict, pixel_depth_arrays_dict, odometry_arrays_dict


def main():
    """Example usage - track multiple colors simultaneously"""
    
    bagpath = "/home/jdx/Downloads/color_motion"
    video_path = "posEstimate/data/color_motion.mp4"
    colors = ["green", "pink"]
    
    print(f"\n=== Multi-Color Tracking: {', '.join(colors)} ===")
    try:
        pixel_arrays_dict, pixel_depth_arrays_dict, odometry_arrays_dict = find_multi_color_odometry(
            bagpath, video_path, colors=colors, min_area=10
        )
        
        print("\nResults:")
        for color in colors:
            if color in pixel_arrays_dict:
                print(f"\n{color}:")
                print(f"  Pixel array shape: {pixel_arrays_dict[color].shape}")
                print(f"  Pixel depth array shape: {pixel_depth_arrays_dict[color].shape}")
                print(f"  Odometry array shape: {odometry_arrays_dict[color].shape}")
        
        # Plot 3D trajectories for both colors
        if len(odometry_arrays_dict) > 0:
            camera_odometry = CameraOdometry(pixel_depth_arrays_dict[list(odometry_arrays_dict.keys())[0]])  # Just for the plot method
            color_map = {
                "green": "green",
                "pink": "magenta",
                "red": "red",
                "blue": "blue",
            }
            camera_odometry.plot_3d_trajectory_multi(odometry_arrays_dict, color_map)
        
    except Exception as e:
        print(f"Multi-color tracking failed: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()

