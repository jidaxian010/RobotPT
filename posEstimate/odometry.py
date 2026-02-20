import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from rosbag_reader import RosbagVideoReader
from object_detect.select_pixel import SelectItem
from object_detect.select_marker import SelectMarker
from object_detect.select_color import SelectColor

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
        self.plot_3d_trajectory(np.array(odometry_array))
        return np.array(odometry_array)


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


def main():
    """Example usage - demonstrates both manual point selection and ArUco marker tracking"""
    
    bagpath = "/home/jdx/Downloads/color_motion"
    video_path = "posEstimate/data/color_motion.mp4"
    
    # print("=== Manual Point Selection ===")
    # try:
    #     pixel_array, pixel_depth_array, odometry_array = find_odometry_data(
    #         bagpath, video_path, select_method="manual"
    #     )
    #     print(f"Pixel array shape: {pixel_array.shape}  (N=1 point, 2=(u,v), T=frames)")
    #     print(f"Pixel depth array shape: {pixel_depth_array.shape}")
    #     print(f"Odometry array shape: {odometry_array.shape}")
    #     print(pixel_depth_array)
    # except Exception as e:
    #     print(f"Manual point selection failed: {e}")
    
    print("\n=== Color Tracking ===")
    try:
        # marker_pixel_array, marker_pixel_depth_array, marker_odometry_array = find_odometry_data(
        #     bagpath, video_path, select_method="marker", corner_idx=0
        # )
        marker_pixel_array, marker_pixel_depth_array, marker_odometry_array = find_odometry_data(
            bagpath, video_path, select_method="color", color="green", min_area=10
        )        
        print(f"Color pixel array shape: {marker_pixel_array.shape}  (N=1 corner, 2=(u,v), T=frames)")
        print(f"Color pixel depth array shape: {marker_pixel_depth_array.shape}")
        print(f"Color odometry array shape: {marker_odometry_array.shape}")
        print(marker_pixel_depth_array)
    except Exception as e:
        print(f"Color tracking failed: {e}")



if __name__ == "__main__":
    main()

