import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rosbag_reader import RosbagVideoReader
from object_detect.select_marker import SelectMarker
from object_detect.select_pixel import SelectItem


class OnboardOdometry:
    """
    Estimate camera pose (position and orientation) in the marker's reference frame.
    
    This class tracks a fiducial marker and computes the camera's 3D position
    relative to the marker, essentially providing onboard odometry from the 
    camera's perspective.
    """
    
    def __init__(self, marker_size_mm=100.0):
        """
        Initialize onboard odometry estimator.
        
        Args:
            marker_size_mm: Physical size of the ArUco marker in millimeters
        """
        # Camera intrinsics for 848x480x60
        self.fx = 602.6597900390625
        self.fy = 602.2169799804688
        self.cx = 423.1910400390625
        self.cy = 249.92578125
        
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
        # Assume no distortion (or use calibrated values)
        self.dist_coeffs = np.zeros(5)
        
        # Marker size in millimeters
        self.marker_size = marker_size_mm
        
        # Define 3D coordinates of marker corners in marker frame
        # Origin at marker center, Z pointing out of marker
        half_size = marker_size_mm / 2.0
        self.marker_3d_points = np.array([
            [-half_size, half_size, 0],   # Top-left
            [half_size, half_size, 0],    # Top-right
            [half_size, -half_size, 0],   # Bottom-right
            [-half_size, -half_size, 0]   # Bottom-left
        ], dtype=np.float32)
    
    def estimate_pose_from_corners(self, corners_2d):
        """
        Estimate camera pose from 2D marker corners using PnP.
        
        Args:
            corners_2d: Array of shape (4, 2) with marker corners in image coordinates
                       Order: top-left, top-right, bottom-right, bottom-left
        
        Returns:
            Tuple of (rvec, tvec, success) where:
                - rvec: Rotation vector (3,) - marker to camera
                - tvec: Translation vector (3,) - marker to camera in mm
                - success: Boolean indicating if pose estimation succeeded
        """
        if np.any(np.isnan(corners_2d)):
            return None, None, False
        
        corners_2d = corners_2d.astype(np.float32)
        
        # Solve PnP to get marker pose relative to camera
        success, rvec, tvec = cv2.solvePnP(
            self.marker_3d_points,
            corners_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if success:
            return rvec.flatten(), tvec.flatten(), True
        else:
            return None, None, False
    
    def invert_pose(self, rvec, tvec):
        """
        Invert the pose transformation to get camera position in marker frame.
        
        Args:
            rvec: Rotation vector from marker to camera
            tvec: Translation vector from marker to camera
            
        Returns:
            Tuple of (R_cam_in_marker, t_cam_in_marker) where:
                - R_cam_in_marker: 3x3 rotation matrix of camera in marker frame
                - t_cam_in_marker: 3D position of camera in marker frame (mm)
        """
        # Convert rotation vector to rotation matrix
        R_marker_to_cam, _ = cv2.Rodrigues(rvec)
        
        # Invert the transformation
        # If T_marker_to_cam = [R | t], then T_cam_to_marker = [R^T | -R^T * t]
        R_cam_to_marker = R_marker_to_cam.T
        t_cam_in_marker = -R_cam_to_marker @ tvec
        
        return R_cam_to_marker, t_cam_in_marker
    
    def process_marker_trajectory(self, marker_corners_trajectory):
        """
        Process marker corner trajectory to estimate camera trajectory in marker frame.
        
        Args:
            marker_corners_trajectory: Array of shape (4, 2, T) where:
                - 4 corners
                - 2 coordinates (u, v)
                - T frames
        
        Returns:
            Dictionary containing:
                - 'positions': Array of shape (T, 3) - camera positions in marker frame (mm)
                - 'rotations': Array of shape (T, 3, 3) - camera orientations in marker frame
                - 'valid_frames': Boolean array of shape (T,) indicating valid pose estimates
        """
        n_corners, n_coords, n_frames = marker_corners_trajectory.shape
        
        positions = []
        rotations = []
        valid_frames = []
        
        for frame_idx in range(n_frames):
            # Extract corners for this frame: shape (4, 2)
            corners = marker_corners_trajectory[:, :, frame_idx]
            
            # Estimate pose
            rvec, tvec, success = self.estimate_pose_from_corners(corners)
            
            if success:
                # Invert to get camera pose in marker frame
                R_cam, t_cam = self.invert_pose(rvec, tvec)
                
                positions.append(t_cam)
                rotations.append(R_cam)
                valid_frames.append(True)
            else:
                # Use NaN for invalid frames
                positions.append(np.array([np.nan, np.nan, np.nan]))
                rotations.append(np.full((3, 3), np.nan))
                valid_frames.append(False)
        
        return {
            'positions': np.array(positions),
            'rotations': np.array(rotations),
            'valid_frames': np.array(valid_frames)
        }
    
    def plot_3d_trajectory(self, positions, valid_frames=None, title="Camera Trajectory in Marker Frame"):
        """
        Plot 3D trajectory of camera in marker reference frame.
        
        Args:
            positions: Array of shape (T, 3) with camera positions
            valid_frames: Boolean array indicating which frames are valid
            title: Plot title
        """
        if valid_frames is not None:
            # Filter out invalid frames
            valid_positions = positions[valid_frames]
        else:
            # Remove NaN values
            valid_mask = ~np.any(np.isnan(positions), axis=1)
            valid_positions = positions[valid_mask]
        
        if len(valid_positions) == 0:
            print("No valid positions to plot!")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates (marker frame: X=right, Y=down, Z=forward from marker)
        x = valid_positions[:, 0]
        y = valid_positions[:, 1]
        z = valid_positions[:, 2]
        
        # Plot trajectory
        ax.plot(x, y, z, 'b-', linewidth=2, label='Camera trajectory')
        
        # Mark start and end
        ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start', marker='o')
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='End', marker='X')
        
        # Draw marker at origin
        marker_size = self.marker_size / 2
        marker_corners = np.array([
            [-marker_size, -marker_size, 0],
            [marker_size, -marker_size, 0],
            [marker_size, marker_size, 0],
            [-marker_size, marker_size, 0],
            [-marker_size, -marker_size, 0]  # Close the square
        ])
        ax.plot(marker_corners[:, 0], marker_corners[:, 1], marker_corners[:, 2], 
                'k-', linewidth=3, label='Marker')
        
        # Draw coordinate frame at marker origin
        axis_length = self.marker_size
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2, label='X (right)')
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, linewidth=2, label='Y (down)')
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, linewidth=2, label='Z (forward)')
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('X (mm) - Right')
        ax.set_ylabel('Y (mm) - Down')
        ax.set_zlabel('Z (mm) - Forward')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_trajectory_components(self, positions, valid_frames=None, timestamps=None):
        """
        Plot X, Y, Z components of camera trajectory over time.
        
        Args:
            positions: Array of shape (T, 3) with camera positions
            valid_frames: Boolean array indicating which frames are valid
            timestamps: Optional array of timestamps for x-axis
        """
        if valid_frames is not None:
            positions = positions.copy()
            positions[~valid_frames] = np.nan
        
        if timestamps is None:
            timestamps = np.arange(len(positions))
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        labels = ['X (right)', 'Y (down)', 'Z (forward)']
        colors = ['r', 'g', 'b']
        
        for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
            ax.plot(timestamps, positions[:, i], color=color, linewidth=2)
            ax.set_ylabel(f'{label} (mm)')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Camera Position - {label}')
        
        axes[-1].set_xlabel('Frame' if timestamps is None else 'Time (s)')
        plt.tight_layout()
        plt.show()
    
    def estimate_pose_from_3d_points(self, points_3d_world, points_2d_image):
        """
        Estimate camera pose from 3D-2D point correspondences using PnP.
        
        Args:
            points_3d_world: Array of shape (N, 3) with 3D points in world frame
            points_2d_image: Array of shape (N, 2) with corresponding 2D image points
        
        Returns:
            Tuple of (R_cam, t_cam, success) where:
                - R_cam: 3x3 rotation matrix of camera in world frame
                - t_cam: 3D position of camera in world frame (mm)
                - success: Boolean indicating if pose estimation succeeded
        """
        if len(points_3d_world) < 4:
            return None, None, False
        
        # Remove any NaN values
        valid_mask = ~(np.any(np.isnan(points_3d_world), axis=1) | np.any(np.isnan(points_2d_image), axis=1))
        if valid_mask.sum() < 4:
            return None, None, False
        
        points_3d_world = points_3d_world[valid_mask].astype(np.float32)
        points_2d_image = points_2d_image[valid_mask].astype(np.float32)
        
        # Solve PnP to get camera pose
        success, rvec, tvec = cv2.solvePnP(
            points_3d_world,
            points_2d_image,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Convert rotation vector to rotation matrix
            R_world_to_cam, _ = cv2.Rodrigues(rvec)
            tvec = tvec.flatten()
            
            # Invert to get camera pose in world frame
            R_cam = R_world_to_cam.T
            t_cam = -R_cam @ tvec
            
            return R_cam, t_cam, True
        else:
            return None, None, False


class VisualOdometry:
    """
    Robust visual odometry similar to RealSense T265 approach.
    
    Uses automatic feature detection, tracking, and RANSAC-based pose estimation.
    """
    
    def __init__(self, camera_matrix, dist_coeffs=None):
        """
        Initialize visual odometry.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients (default: no distortion)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        
        # Feature detector (FAST + ORB for robustness)
        self.feature_detector = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # RANSAC parameters for PnP
        self.ransac_threshold = 3.0  # pixels
        self.ransac_confidence = 0.99
        
        # State
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_points_3d = None
        self.world_points = {}  # track_id -> 3D position in world frame
        self.next_track_id = 0
        
        # Trajectory
        self.trajectory = []
        self.rotations = []
        
    def detect_features(self, frame):
        """Detect features in a frame."""
        keypoints = self.feature_detector.detect(frame, None)
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return points
    
    def track_features(self, prev_frame, curr_frame, prev_points):
        """Track features using optical flow."""
        if len(prev_points) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Track forward
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, prev_points, None, **self.lk_params
        )
        
        # Track backward for validation
        prev_points_back, status_back, err_back = cv2.calcOpticalFlowPyrLK(
            curr_frame, prev_frame, curr_points, None, **self.lk_params
        )
        
        # Keep only good tracks (forward-backward consistency)
        fb_error = np.linalg.norm(prev_points - prev_points_back, axis=1)
        good_mask = (status.flatten() == 1) & (status_back.flatten() == 1) & (fb_error < 1.0)
        
        return curr_points[good_mask], prev_points[good_mask], good_mask
    
    def triangulate_points(self, points_2d, depth_values):
        """
        Convert 2D points with depth to 3D points in camera frame.
        
        Args:
            points_2d: Array of shape (N, 2) with pixel coordinates
            depth_values: Array of shape (N,) with depth in mm
            
        Returns:
            Array of shape (N, 3) with 3D points in camera frame
        """
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        points_3d = []
        for (u, v), d in zip(points_2d, depth_values):
            if d > 0:
                x = (u - cx) * d / fx
                y = (v - cy) * d / fy
                z = d
                points_3d.append([x, y, z])
            else:
                points_3d.append([np.nan, np.nan, np.nan])
        
        return np.array(points_3d)
    
    def estimate_pose_ransac(self, points_3d_world, points_2d_image):
        """
        Estimate camera pose using PnP with RANSAC for outlier rejection.
        
        Args:
            points_3d_world: Array of shape (N, 3) with 3D points in world frame
            points_2d_image: Array of shape (N, 2) with corresponding 2D image points
            
        Returns:
            Tuple of (R_cam, t_cam, inliers, success)
        """
        if len(points_3d_world) < 6:  # Need at least 6 points for RANSAC
            return None, None, None, False
        
        # Remove NaN values
        valid_mask = ~(np.any(np.isnan(points_3d_world), axis=1) | np.any(np.isnan(points_2d_image), axis=1))
        if valid_mask.sum() < 6:
            return None, None, None, False
        
        points_3d = points_3d_world[valid_mask].astype(np.float32)
        points_2d = points_2d_image[valid_mask].astype(np.float32)
        
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            self.camera_matrix,
            self.dist_coeffs,
            reprojectionError=self.ransac_threshold,
            confidence=self.ransac_confidence,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success and inliers is not None and len(inliers) >= 6:
            # Convert to camera pose in world frame
            R_world_to_cam, _ = cv2.Rodrigues(rvec)
            tvec = tvec.flatten()
            
            R_cam = R_world_to_cam.T
            t_cam = -R_cam @ tvec
            
            return R_cam, t_cam, inliers.flatten(), True
        else:
            return None, None, None, False
    
    def process_frame(self, frame, depth_image, is_first_frame=False):
        """
        Process a single frame to estimate camera pose.
        
        Args:
            frame: Grayscale image
            depth_image: Depth image (same size as frame, values in mm)
            is_first_frame: Whether this is the first frame
            
        Returns:
            Tuple of (R_cam, t_cam, n_inliers, success)
        """
        if is_first_frame:
            # Initialize with first frame
            points = self.detect_features(frame)
            
            # Get depth for detected features
            depth_values = []
            valid_points = []
            for pt in points:
                u, v = int(pt[0]), int(pt[1])
                if 0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]:
                    d = depth_image[v, u]
                    if d > 0:
                        depth_values.append(d)
                        valid_points.append(pt)
            
            if len(valid_points) < 10:
                return None, None, 0, False
            
            valid_points = np.array(valid_points)
            depth_values = np.array(depth_values)
            
            # Convert to 3D in world frame (first frame defines world frame)
            points_3d = self.triangulate_points(valid_points, depth_values)
            
            # Store as world points
            for i, (pt, pt3d) in enumerate(zip(valid_points, points_3d)):
                if not np.any(np.isnan(pt3d)):
                    self.world_points[self.next_track_id] = pt3d
                    self.next_track_id += 1
            
            self.prev_frame = frame
            self.prev_keypoints = valid_points
            
            # First frame is at origin
            self.trajectory.append(np.zeros(3))
            self.rotations.append(np.eye(3))
            
            print(f"Initialized with {len(self.world_points)} 3D points")
            return np.eye(3), np.zeros(3), len(self.world_points), True
        
        # Track features from previous frame
        curr_points, prev_points, good_mask = self.track_features(
            self.prev_frame, frame, self.prev_keypoints
        )
        
        if len(curr_points) < 6:
            # Lost tracking - try to reinitialize with new features
            print(f"Warning: Only {len(curr_points)} tracked points, trying to detect new features")
            self.prev_frame = frame
            self.prev_keypoints = self.detect_features(frame)
            return None, None, 0, False
        
        # Match tracked points to world points
        points_3d_world = []
        points_2d_curr = []
        
        track_ids = list(self.world_points.keys())[:len(self.prev_keypoints)]
        for i, is_good in enumerate(good_mask):
            if is_good and i < len(track_ids):
                track_id = track_ids[i]
                if track_id in self.world_points:
                    points_3d_world.append(self.world_points[track_id])
                    points_2d_curr.append(curr_points[np.where(good_mask)[0].tolist().index(i)])
        
        if len(points_3d_world) < 6:
            print(f"Warning: Only {len(points_3d_world)} matched points")
            self.prev_frame = frame
            self.prev_keypoints = curr_points
            return None, None, 0, False
        
        points_3d_world = np.array(points_3d_world)
        points_2d_curr = np.array(points_2d_curr)
        
        # Estimate pose with RANSAC
        R_cam, t_cam, inliers, success = self.estimate_pose_ransac(
            points_3d_world, points_2d_curr
        )
        
        if success:
            self.trajectory.append(t_cam)
            self.rotations.append(R_cam)
            
            # Update previous frame and keypoints (only keep inliers)
            self.prev_frame = frame
            self.prev_keypoints = curr_points
            
            n_inliers = len(inliers)
            print(f"Frame processed: {n_inliers}/{len(points_3d_world)} inliers")
            
            # Detect and add new features if we're losing points
            if n_inliers < 20:
                new_points = self.detect_features(frame)
                # Add some new features to track
                # (In a full SLAM system, you'd triangulate these over time)
            
            return R_cam, t_cam, n_inliers, True
        else:
            print("Pose estimation failed")
            self.prev_frame = frame
            self.prev_keypoints = curr_points
            return None, None, 0, False


def find_onboard_odometry_visual(bagpath, video_path):
    """
    Robust visual odometry using automatic feature detection and tracking.
    
    Similar to RealSense T265 approach (without IMU fusion).
    
    Args:
        bagpath: Path to rosbag file
        video_path: Path to video file
        
    Returns:
        Dictionary containing:
            - 'positions': Camera positions in world frame (T, 3)
            - 'rotations': Camera orientations in world frame (T, 3, 3)
            - 'valid_frames': Boolean array of valid frames
    """
    print("=== Visual Odometry (Automatic Feature Tracking) ===")
    
    # Initialize camera
    fx = 602.6597900390625
    fy = 602.2169799804688
    cx = 423.1910400390625
    cy = 249.92578125
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    vo = VisualOdometry(camera_matrix)
    
    # Read video and depth
    print("Reading video and depth data...")
    video_reader = RosbagVideoReader(Path(bagpath), Path(video_path), skip_first_n=0, skip_last_n=0)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    frame_idx = 0
    positions = []
    rotations = []
    valid_frames = []
    
    print("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get depth image for this frame
        # Create a dummy depth request for the whole frame (we'll query specific pixels)
        # For efficiency, we should get the full depth image
        try:
            # This is a simplified approach - in practice, you'd want to cache depth images
            # For now, we'll process every Nth frame to make it manageable
            if frame_idx % 1 == 0:  # Process every frame
                # We need to get the depth image from rosbag
                # This is a limitation - we need a better way to access depth images
                # For now, let's create a placeholder
                depth_image = np.zeros_like(gray, dtype=np.uint16)  # Placeholder
                
                # Process frame
                is_first = (frame_idx == 0)
                R_cam, t_cam, n_inliers, success = vo.process_frame(gray, depth_image, is_first)
                
                if success:
                    positions.append(t_cam)
                    rotations.append(R_cam)
                    valid_frames.append(True)
                else:
                    positions.append(np.array([np.nan, np.nan, np.nan]))
                    rotations.append(np.full((3, 3), np.nan))
                    valid_frames.append(False)
            
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames...")
                
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            break
    
    cap.release()
    
    positions = np.array(positions)
    rotations = np.array(rotations)
    valid_frames = np.array(valid_frames)
    
    print(f"\nProcessed {frame_idx} frames")
    print(f"Valid poses: {valid_frames.sum()}/{len(valid_frames)}")
    
    return {
        'positions': positions,
        'rotations': rotations,
        'valid_frames': valid_frames,
        'odometry': OnboardOdometry()
    }


def find_onboard_odometry_multipoint(bagpath, video_path, n_points=10, tracking_method="auto", playback_speed=1):
    """
    Track multiple pixels and estimate camera trajectory using their 3D positions.
    
    This is more robust than single marker tracking as it uses multiple point correspondences.
    
    Args:
        bagpath: Path to rosbag file
        video_path: Path to video file
        n_points: Number of points to track (for manual selection)
        tracking_method: "manual" for manual point selection, "auto" for automatic feature detection
        playback_speed: Speed multiplier for video playback (manual mode only)
        
    Returns:
        Dictionary containing:
            - 'positions': Camera positions in world frame (T, 3)
            - 'rotations': Camera orientations in world frame (T, 3, 3)
            - 'valid_frames': Boolean array of valid frames
            - 'points_3d_initial': Initial 3D positions of tracked points
    """
    print(f"=== Multi-Point Onboard Odometry ===")
    print(f"Tracking method: {tracking_method}")
    
    # Step 1: Track pixels in 2D
    if tracking_method == "manual":
        print(f"Manual tracking: Select {n_points} points to track")
        tracker = SelectItem(video_path, playback_speed=playback_speed)
        pixel_trajectories = tracker.process_data(show_tracking=True)  # Shape: (N, 2, T)
    elif tracking_method == "auto":
        # TODO: Implement automatic feature detection and tracking
        raise NotImplementedError("Automatic feature tracking not yet implemented. Use 'manual' for now.")
    else:
        raise ValueError(f"Invalid tracking_method: {tracking_method}")
    
    n_points_tracked, _, n_frames = pixel_trajectories.shape
    print(f"Tracked {n_points_tracked} points across {n_frames} frames")
    
    # Step 2: Get depth for all tracked pixels
    print("Getting depth information from rosbag...")
    
    # Convert pixel trajectories to format for depth lookup
    pixel_depth_list = []
    for point_idx in range(n_points_tracked):
        for frame_idx in range(n_frames):
            u, v = pixel_trajectories[point_idx, :, frame_idx]
            if not np.isnan(u) and not np.isnan(v):
                pixel_depth_list.append([u, v, frame_idx])
    
    pixel_depth_array = np.array(pixel_depth_list)
    
    # Get depth values from rosbag
    video_reader = RosbagVideoReader(Path(bagpath), Path(video_path), skip_first_n=0, skip_last_n=0)
    pixel_depth_array = video_reader.find_depth(pixel_depth_array)  # Returns [u, v, d, frame_idx]
    
    print(f"Retrieved depth for {len(pixel_depth_array)} pixel observations")
    
    # Step 3: Convert to 3D points in camera frame for first frame (world frame)
    odometry = OnboardOdometry()
    
    # Organize data by frame
    frame_data = {}  # frame_idx -> list of (point_idx, u, v, d)
    for point_idx in range(n_points_tracked):
        for frame_idx in range(n_frames):
            u, v = pixel_trajectories[point_idx, :, frame_idx]
            if np.isnan(u) or np.isnan(v):
                continue
            
            # Find depth for this pixel
            mask = (pixel_depth_array[:, 3] == frame_idx) & \
                   (np.abs(pixel_depth_array[:, 0] - u) < 0.5) & \
                   (np.abs(pixel_depth_array[:, 1] - v) < 0.5)
            
            if mask.any():
                depth = pixel_depth_array[mask, 2][0]
                if depth > 0:
                    if frame_idx not in frame_data:
                        frame_data[frame_idx] = []
                    frame_data[frame_idx].append((point_idx, u, v, depth))
    
    print(f"Organized data for {len(frame_data)} frames")
    
    # Step 4: Establish world frame from first frame
    first_frame_idx = min(frame_data.keys())
    first_frame_points = frame_data[first_frame_idx]
    
    # Convert first frame pixels to 3D points (these define the world frame)
    points_3d_world = []
    point_indices_world = []
    
    for point_idx, u, v, d in first_frame_points:
        # Unproject to 3D
        x = (u - odometry.cx) * d / odometry.fx
        y = (v - odometry.cy) * d / odometry.fy
        z = d
        points_3d_world.append([x, y, z])
        point_indices_world.append(point_idx)
    
    points_3d_world = np.array(points_3d_world)
    point_indices_world = np.array(point_indices_world)
    
    print(f"Established world frame with {len(points_3d_world)} 3D points from frame {first_frame_idx}")
    print(f"World points range (mm):")
    print(f"  X: [{points_3d_world[:, 0].min():.1f}, {points_3d_world[:, 0].max():.1f}]")
    print(f"  Y: [{points_3d_world[:, 1].min():.1f}, {points_3d_world[:, 1].max():.1f}]")
    print(f"  Z: [{points_3d_world[:, 2].min():.1f}, {points_3d_world[:, 2].max():.1f}]")
    
    # Step 5: Estimate camera pose for each frame using PnP
    positions = []
    rotations = []
    valid_frames = []
    
    for frame_idx in sorted(frame_data.keys()):
        frame_points = frame_data[frame_idx]
        
        # Match points to world frame
        points_3d_matched = []
        points_2d_matched = []
        
        for point_idx, u, v, d in frame_points:
            # Find this point in world frame
            world_idx = np.where(point_indices_world == point_idx)[0]
            if len(world_idx) > 0:
                points_3d_matched.append(points_3d_world[world_idx[0]])
                points_2d_matched.append([u, v])
        
        if len(points_3d_matched) >= 4:
            points_3d_matched = np.array(points_3d_matched)
            points_2d_matched = np.array(points_2d_matched)
            
            # Estimate camera pose
            R_cam, t_cam, success = odometry.estimate_pose_from_3d_points(
                points_3d_matched, points_2d_matched
            )
            
            if success:
                positions.append(t_cam)
                rotations.append(R_cam)
                valid_frames.append(True)
            else:
                positions.append(np.array([np.nan, np.nan, np.nan]))
                rotations.append(np.full((3, 3), np.nan))
                valid_frames.append(False)
        else:
            positions.append(np.array([np.nan, np.nan, np.nan]))
            rotations.append(np.full((3, 3), np.nan))
            valid_frames.append(False)
    
    positions = np.array(positions)
    rotations = np.array(rotations)
    valid_frames = np.array(valid_frames)
    
    # Print statistics
    n_valid = valid_frames.sum()
    n_total = len(valid_frames)
    print(f"\nValid pose estimates: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    
    if n_valid > 0:
        valid_positions = positions[valid_frames]
        print(f"Camera position range (mm):")
        print(f"  X: [{valid_positions[:, 0].min():.1f}, {valid_positions[:, 0].max():.1f}]")
        print(f"  Y: [{valid_positions[:, 1].min():.1f}, {valid_positions[:, 1].max():.1f}]")
        print(f"  Z: [{valid_positions[:, 2].min():.1f}, {valid_positions[:, 2].max():.1f}]")
        
        # Calculate distance traveled
        distances = np.sqrt(np.sum(np.diff(valid_positions, axis=0)**2, axis=1))
        total_distance = distances.sum()
        print(f"Total distance traveled: {total_distance:.1f} mm ({total_distance/1000:.3f} m)")
    
    return {
        'positions': positions,
        'rotations': rotations,
        'valid_frames': valid_frames,
        'points_3d_initial': points_3d_world,
        'odometry': odometry
    }


def find_onboard_odometry(bagpath, video_path, marker_id=None, marker_size_mm=100.0, dict_name="DICT_4X4_50"):
    """
    Track fiducial marker and estimate camera pose in marker frame.
    
    Args:
        bagpath: Path to rosbag file (for generating video if needed)
        video_path: Path to video file
        marker_id: Specific marker ID to track (None = use first detected)
        marker_size_mm: Physical size of the marker in millimeters
        dict_name: ArUco dictionary name (e.g., "DICT_4X4_50")
        
    Returns:
        Dictionary containing:
            - 'positions': Camera positions in marker frame (T, 3)
            - 'rotations': Camera orientations in marker frame (T, 3, 3)
            - 'valid_frames': Boolean array of valid frames
            - 'marker_id': ID of tracked marker
    """
    # Track marker
    video_path_obj = Path(video_path)
    output_path = video_path_obj.parent / f"{video_path_obj.stem}_marked.mp4"
    
    marker_tracker = SelectMarker(
        input_path=video_path,
        output_path=str(output_path),
        dict_name=dict_name
    )
    
    # Get trajectories for all markers
    marker_trajectories = marker_tracker.run()  # Dict[int, np.ndarray(4,2,T)]
    
    if not marker_trajectories:
        raise RuntimeError("No ArUco markers detected in video")
    
    # Select marker to track
    if marker_id is None:
        marker_id = list(marker_trajectories.keys())[0]
        print(f"No marker_id specified, using first detected marker: {marker_id}")
    
    if marker_id not in marker_trajectories:
        raise RuntimeError(f"Marker ID {marker_id} not found. Available: {list(marker_trajectories.keys())}")
    
    print(f"Tracking marker ID {marker_id} (size: {marker_size_mm}mm)")
    
    # Get marker corner trajectory: shape (4, 2, T)
    marker_corners = marker_trajectories[marker_id]
    print(f"Marker trajectory shape: {marker_corners.shape}")
    
    # Estimate camera pose
    odometry = OnboardOdometry(marker_size_mm=marker_size_mm)
    result = odometry.process_marker_trajectory(marker_corners)
    
    # Add marker ID to result
    result['marker_id'] = marker_id
    result['odometry'] = odometry  # Keep reference for plotting
    
    # Print statistics
    n_valid = result['valid_frames'].sum()
    n_total = len(result['valid_frames'])
    print(f"Valid pose estimates: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    
    if n_valid > 0:
        valid_positions = result['positions'][result['valid_frames']]
        print(f"Position range (mm):")
        print(f"  X: [{valid_positions[:, 0].min():.1f}, {valid_positions[:, 0].max():.1f}]")
        print(f"  Y: [{valid_positions[:, 1].min():.1f}, {valid_positions[:, 1].max():.1f}]")
        print(f"  Z: [{valid_positions[:, 2].min():.1f}, {valid_positions[:, 2].max():.1f}]")
        
        # Calculate distance traveled
        valid_pos = valid_positions
        distances = np.sqrt(np.sum(np.diff(valid_pos, axis=0)**2, axis=1))
        total_distance = distances.sum()
        print(f"Total distance traveled: {total_distance:.1f} mm ({total_distance/1000:.3f} m)")
    
    return result


def main():
    """Example usage - demonstrates both marker-based and multi-point tracking"""
    
    # Configuration
    bagpath = "/home/jdx/Downloads/Grab_arm"
    video_path = "posEstimate/data/Grab_arm.mp4"
    
    # Choose method: "marker", "multipoint", or "visual"
    method = "multipoint"  # Change to use different tracking methods
    
    if method == "marker":
        print("=== Method 1: Marker-Based Odometry ===")
        marker_size_mm = 100.0  # Physical size of your ArUco marker in mm
        
        try:
            result = find_onboard_odometry(
                bagpath=bagpath,
                video_path=video_path,
                marker_id=None,  # Auto-select first marker
                marker_size_mm=marker_size_mm,
                dict_name="DICT_4X4_50"
            )
            
            print(f"\nTracked marker ID: {result['marker_id']}")
            print(f"Positions shape: {result['positions'].shape}")
            print(f"Rotations shape: {result['rotations'].shape}")
            
            # Plot 3D trajectory
            odometry = result['odometry']
            odometry.plot_3d_trajectory(
                result['positions'], 
                result['valid_frames'],
                title=f"Camera Trajectory in Marker {result['marker_id']} Frame"
            )
            
            # Plot position components over time
            odometry.plot_trajectory_components(
                result['positions'],
                result['valid_frames']
            )
            
        except Exception as e:
            print(f"Marker-based odometry failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif method == "multipoint":
        print("=== Method 2: Multi-Point Odometry (More Robust) ===")
        
        try:
            result = find_onboard_odometry_multipoint(
                bagpath=bagpath,
                video_path=video_path,
                n_points=10,  # Will use however many points you select
                tracking_method="manual",  # "manual" or "auto" (auto not yet implemented)
                playback_speed=1
            )
            
            print(f"\nPositions shape: {result['positions'].shape}")
            print(f"Rotations shape: {result['rotations'].shape}")
            print(f"Tracked {len(result['points_3d_initial'])} 3D points")
            
            # Plot 3D trajectory
            odometry = result['odometry']
            odometry.plot_3d_trajectory(
                result['positions'], 
                result['valid_frames'],
                title="Camera Trajectory (Multi-Point Tracking)"
            )
            
            # Plot position components over time
            odometry.plot_trajectory_components(
                result['positions'],
                result['valid_frames']
            )
            
        except Exception as e:
            print(f"Multi-point odometry failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif method == "visual":
        print("=== Method 3: Visual Odometry (T265-style, Automatic) ===")
        print("Note: This method uses automatic feature detection and tracking")
        print("Similar to RealSense T265 (without IMU fusion)")
        
        try:
            result = find_onboard_odometry_visual(
                bagpath=bagpath,
                video_path=video_path
            )
            
            print(f"\nPositions shape: {result['positions'].shape}")
            print(f"Rotations shape: {result['rotations'].shape}")
            
            # Plot 3D trajectory
            odometry = result['odometry']
            odometry.plot_3d_trajectory(
                result['positions'], 
                result['valid_frames'],
                title="Camera Trajectory (Visual Odometry)"
            )
            
            # Plot position components over time
            odometry.plot_trajectory_components(
                result['positions'],
                result['valid_frames']
            )
            
        except Exception as e:
            print(f"Visual odometry failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"Invalid method: {method}. Choose 'marker', 'multipoint', or 'visual'")


if __name__ == "__main__":
    main()

