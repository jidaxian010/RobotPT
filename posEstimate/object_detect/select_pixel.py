import cv2
import numpy as np
from pathlib import Path

class SelectItem:
    def __init__(self, video_path, playback_speed):
        """
        Initialize the point tracker.
        
        Args:
            video_path: Path to the MP4 video file (can be relative to data/ or absolute)
        """
        self.video_path = Path(video_path)
        self.playback_speed = playback_speed
        
        # If path doesn't exist, try prepending data/
        if not self.video_path.exists():
            data_path = Path("posEstimate/data") / video_path
            if data_path.exists():
                self.video_path = data_path
        
        self.picked = {"pt": None, "ready": False}
        # Reduced sensitivity: larger window, tighter criteria
        self.lk_params = dict(
            winSize=(35, 35),           # Larger window for more stable tracking
            maxLevel=2,                 # Fewer pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def on_mouse(self, event, x, y, flags, param):
        """Mouse callback for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.picked["pt"] = np.array([[x, y]], dtype=np.float32)
            self.picked["ready"] = True

    def select_point(self, first_frame):
        """
        Allow user to select a point on the first frame.
        
        Args:
            first_frame: The first frame of the video
            
        Returns:
            Selected point as numpy array (1,1,2)
        """
        cv2.namedWindow("Select point")
        cv2.setMouseCallback("Select point", self.on_mouse)

        # Show first frame until user clicks
        while True:
            vis = first_frame.copy()
            cv2.putText(vis, "Click a point on the object, then press any key",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Select point", vis)
            if cv2.waitKey(20) & 0xFF != 255 and self.picked["ready"]:
                break

        cv2.destroyWindow("Select point")
        return self.picked["pt"].reshape(-1, 1, 2)

    def process_data(self, show_tracking=True):
        """
        Track a selected point through the video and return pixel coordinates.
        
        Args:
            show_tracking: If True, display tracking visualization in real-time
            
        Returns:
            Numpy array of shape (N, 2, T) where N=1 point, 2=(u,v), T=frames
        """
        print(f"Opening video: {self.video_path}")
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Couldn't open video file: {self.video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: {fps:.2f} FPS, {total_frames} frames")
        
        # Read first frame
        ok, first = cap.read()
        if not ok:
            raise RuntimeError("Couldn't read first frame")
        
        # Let user select a point
        p0 = self.select_point(first)
        
        # Initialize tracking
        prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        trail = [tuple(map(int, self.picked["pt"][0]))]
        
        # Store pixel coordinates for each frame
        pixel_coords = []  # Will store [u, v] for each frame
        frame_idx = 0
        
        # Add first point
        x0, y0 = self.picked["pt"][0]
        pixel_coords.append([float(x0), float(y0)])
        
        # Track through video
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame_idx += 1
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Track the point
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk_params)
            
            if st is not None and st[0] == 1:
                x, y = p1[0, 0]
                p = (int(x), int(y))
                trail.append(p)
                
                # Store the tracked point
                pixel_coords.append([float(x), float(y)])
                
                if show_tracking:
                    # Draw trail
                    for i in range(1, len(trail)):
                        cv2.line(frame, trail[i-1], trail[i], (0, 255, 0), 2)
                    cv2.circle(frame, p, 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"({p[0]}, {p[1]})", (p[0]+8, p[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                p0 = p1  # update the point
            else:
                # Lost tracking - keep last known position
                if pixel_coords:
                    last = pixel_coords[-1]
                    pixel_coords.append([last[0], last[1]])
                
                if show_tracking:
                    cv2.putText(frame, "Lost point (occlusion/blur).", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if show_tracking:
                cv2.imshow("Tracking", frame)
                prev_gray = gray
                
                # Calculate delay based on fps and playback speed
                delay_ms = int((1000 / fps) * self.playback_speed)
                delay_ms = max(1, delay_ms)  # Ensure at least 1ms
                
                if cv2.waitKey(delay_ms) & 0xFF == 27:  # ESC to quit
                    print("Tracking interrupted by user")
                    break
            else:
                prev_gray = gray
        
        cap.release()
        if show_tracking:
            cv2.destroyAllWindows()
        
        # Convert to numpy array and reshape to (N, 2, T) format
        # N=1 point, 2=(u,v), T=frames
        pixel_coords_array = np.array(pixel_coords)  # Shape: (T, 2)
        pixel_array = pixel_coords_array.T.reshape(1, 2, -1)  # Shape: (1, 2, T)
        return pixel_array 

