from pathlib import Path
import numpy as np
import json
import bisect
from collections import defaultdict
import cv2

from rosbags.highlevel import AnyReader
from rosbags.serde import deserialize_cdr


class RosbagReader:
    """Read and synchronize multiple IMU topics from ROS bag files."""
    
    def __init__(self, bagpath, out_file, start_at_zero=True):
        """
        Initialize multi-IMU reader.
        
        Args:
            bagpath: Path to the ROS bag file
            out_file: Output JSONL file path
            start_at_zero: Whether to start timestamps from zero
        """
        self.bagpath = bagpath
        self.out_file = out_file
        self.start_at_zero = start_at_zero
        
        # Define the IMU topics to read
        self.imu_topics = {
            "/left_camera/camera/camera/imu": "imu_left",
            "/right_camera/camera/camera/imu": "imu_right", 
            "/vectornav/imu_uncompensated": "imu_vectornav"
        }
        self.other_topics = {
            "/vectornav/magnetic": "magnetic"
        }

    @staticmethod
    def get_stamp_sec(msg, fallback_ts_ns):
        """Extract timestamp from message."""
        try:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception:
            return fallback_ts_ns * 1e-9

    @staticmethod
    def parse_imu(msg):
        """Parse IMU message to extract acceleration and angular velocity."""
        return [
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
            float(msg.linear_acceleration.z),
            float(msg.angular_velocity.x),
            float(msg.angular_velocity.y),
            float(msg.angular_velocity.z),
        ]
    
    @staticmethod
    def parse_magnetic(msg):
        """Parse magnetic field message to extract x, y, z components."""
        return [
            float(msg.magnetic_field.x),
            float(msg.magnetic_field.y),
            float(msg.magnetic_field.z),
        ]

    def read_all_data(self, reader):
        """Read all data from specified topics (IMU + other sensors)."""
        # Find connections for all topics
        topic_connections = {}
        all_topics = {**self.imu_topics, **self.other_topics}
        
        for topic in all_topics.keys():
            conns = [c for c in reader.connections if c.topic == topic]
            if conns:
                topic_connections[topic] = conns[0]
                print(f"Found topic: {topic}")
            else:
                print(f"Warning: Topic {topic} not found in bag file")
        
        if not topic_connections:
            raise RuntimeError("No topics found in bag file")
        
        # Read all messages from all topics
        all_data = defaultdict(list)  # topic -> [(timestamp, data), ...]
        
        for conn, ts, raw in reader.messages(connections=list(topic_connections.values())):
            msg = deserialize_cdr(raw, conn.msgtype)
            t = self.get_stamp_sec(msg, ts)
            
            # Parse based on topic type
            if conn.topic in self.imu_topics:
                vals = self.parse_imu(msg)
            elif conn.topic in self.other_topics:
                vals = self.parse_magnetic(msg)
            else:
                continue
                
            all_data[conn.topic].append((t, vals))
        
        # Sort each topic's data by timestamp
        for topic in all_data:
            all_data[topic].sort(key=lambda x: x[0])
            
        
        return all_data

    def synchronize_all_data(self, all_data, sync_tolerance=0.01):
        """
        Synchronize all sensor data using timestamp matching.
        
        Args:
            all_data: Dictionary of topic -> [(timestamp, data), ...]
            sync_tolerance: Maximum time difference for synchronization (seconds)
        
        Returns:
            List of synchronized records
        """
        # Find the topic with the most samples as reference (prioritize IMU topics)
        imu_data = {k: v for k, v in all_data.items() if k in self.imu_topics}
        ref_topic = max(imu_data.keys(), key=lambda t: len(imu_data[t])) if imu_data else max(all_data.keys(), key=lambda t: len(all_data[t]))
        ref_data = all_data[ref_topic]
        
        print(f"Sync data {ref_topic} (most samples: {len(ref_data)})")
        
        synchronized_data = []
        all_topic_mappings = {**self.imu_topics, **self.other_topics}
        
        for ref_time, ref_vals in ref_data:
            sync_record = {
                "timestamp": ref_time,
                all_topic_mappings[ref_topic]: ref_vals
            }
            
            # Find closest matches in other topics
            for topic, topic_data in all_data.items():
                if topic == ref_topic:
                    continue
                
                # Find closest timestamp using binary search
                timestamps = [t for t, _ in topic_data]
                closest_idx = self.find_closest_timestamp(timestamps, ref_time)
                
                if closest_idx is not None:
                    closest_time, closest_vals = topic_data[closest_idx]
                    time_diff = abs(closest_time - ref_time)
                    
                    if time_diff <= sync_tolerance:
                        sync_record[all_topic_mappings[topic]] = closest_vals
                    else:
                        # Use None for missing data beyond tolerance
                        sync_record[all_topic_mappings[topic]] = None
                else:
                    sync_record[all_topic_mappings[topic]] = None
            
            synchronized_data.append(sync_record)
        
        # Filter out records with missing IMU data (magnetic can be missing)
        complete_records = []
        for record in synchronized_data:
            # Require all IMU data to be present
            if all(record.get(key) is not None for key in self.imu_topics.values()):
                complete_records.append(record)
        
        return complete_records

    @staticmethod
    def find_closest_timestamp(timestamps, target_time):
        """Find index of closest timestamp using binary search."""
        if not timestamps:
            return None
        
        # Use bisect to find insertion point
        idx = bisect.bisect_left(timestamps, target_time)
        
        # Check the closest candidates
        candidates = []
        if idx > 0:
            candidates.append((idx - 1, abs(timestamps[idx - 1] - target_time)))
        if idx < len(timestamps):
            candidates.append((idx, abs(timestamps[idx] - target_time)))
        
        if not candidates:
            return None
        
        # Return index with minimum time difference
        return min(candidates, key=lambda x: x[1])[0]

    def process_data(self, sync_tolerance=0.01, save_data=True):
        """
        Read all sensor data, synchronize, and optionally save to JSONL file.
        
        Args:
            sync_tolerance: Maximum time difference for synchronization (seconds)
            save_data: If True, save to file. If False, return data dictionary.
            
        Returns:
            If save_data=False, returns dictionary with sensor data arrays
            If save_data=True, returns None (data saved to file)
        """
        print(f"Reading sensor data from bag: {self.bagpath}")
        print(f"Target IMU topics: {list(self.imu_topics.keys())}")
        print(f"Target other topics: {list(self.other_topics.keys())}")
        
        with AnyReader([self.bagpath]) as reader:
            all_data = self.read_all_data(reader)
        
        if not all_data:
            raise RuntimeError("No sensor data found")
        
        # Synchronize the data
        synchronized_records = self.synchronize_all_data(all_data, sync_tolerance)
        
        if not synchronized_records:
            raise RuntimeError("No synchronized data found")
        
        # Adjust timestamps to start from zero if requested
        t0 = synchronized_records[0]["timestamp"] if self.start_at_zero else 0.0
        
        if len(synchronized_records) > 1:
            time_range = synchronized_records[-1]["timestamp"] - synchronized_records[0]["timestamp"]
            # print(f"Time range: {synchronized_records[0]['timestamp']:.3f}s to {synchronized_records[-1]['timestamp']:.3f}s ({time_range:.3f}s total)")
            
            # Calculate approximate sampling rate
            avg_dt = time_range / (len(synchronized_records) - 1)
            print(f"Average sampling rate: {1/avg_dt:.1f} Hz")
        
        # Create output records
        output_records = []
        for record in synchronized_records:
            # Adjust timestamp
            adjusted_timestamp = record["timestamp"] - t0 if self.start_at_zero else record["timestamp"]
            
            output_record = {
                "imu_left": record.get("imu_left"),
                "imu_right": record.get("imu_right"),
                "imu_vectornav": record.get("imu_vectornav"),
                "magnetic": record.get("magnetic"),
                "timestamp": adjusted_timestamp,
            }
            output_records.append(output_record)
        
        if save_data:
            # Forward-fill missing data before saving
            filled_records = self._forward_fill_missing_data(output_records)
            
            # Save to JSONL file
            with open(self.out_file, 'w') as f:
                for record in filled_records:
                    f.write(json.dumps(record, separators=(",", ":")) + "\n")
            
            print(f"[SAVE] {len(filled_records)} samples to {self.out_file}")
            print(f"Forward-filled {len(output_records) - len([r for r in output_records if all(r.get(k) is not None for k in ['imu_left', 'imu_right', 'imu_vectornav'])])} records with missing data")

            timestamps = np.array([record['timestamp'] for record in filled_records])
            imu_left = np.array([record['imu_left'] for record in filled_records])
            imu_right = np.array([record['imu_right'] for record in filled_records])
            imu_vectornav = np.array([record['imu_vectornav'] for record in filled_records])
            magnetic = np.array([record['magnetic'] for record in filled_records])
            data = {
                'timestamps': timestamps,
                'imu_left': imu_left,
                'imu_right': imu_right,
                'imu_vectornav': imu_vectornav,
                'magnetic': magnetic
            }
            estimated_size_mb = self._estimate_memory_usage(filled_records)
            print(f"[RETURN] {len(filled_records)} synchronized samples in memory ({estimated_size_mb:.1f} MB)")
            print(f"keys and shapes: {data.keys()} {[v.shape for v in data.values()]}")
            return data
        else:
            # Return data as dictionary with numpy arrays
            estimated_size_mb = self._estimate_memory_usage(output_records)
            
            # Forward-fill missing data instead of filtering out records
            filled_records = self._forward_fill_missing_data(output_records)
            
            print(f"Forward-filled {len(output_records) - len([r for r in output_records if all(r.get(k) is not None for k in ['imu_left', 'imu_right', 'imu_vectornav'])])} records with missing data")
            
            # Convert to numpy arrays (same format as load_sensor_data in main.py)
            timestamps = np.array([record['timestamp'] for record in filled_records])
            imu_left = np.array([record['imu_left'] for record in filled_records])
            imu_right = np.array([record['imu_right'] for record in filled_records])
            imu_vectornav = np.array([record['imu_vectornav'] for record in filled_records])
            magnetic = np.array([record['magnetic'] for record in filled_records])
            
            
            data = {
                'timestamps': timestamps,
                'imu_left': imu_left,
                'imu_right': imu_right,
                'imu_vectornav': imu_vectornav,
                'magnetic': magnetic
            }
            estimated_size_mb = self._estimate_memory_usage(filled_records)
            print(f"[RETURN] {len(filled_records)} synchronized samples in memory ({estimated_size_mb:.1f} MB)")
            print(f"[RETURN] keys and shapes: {list(data.keys())} {[v.shape for v in data.values()]}")
            return data

    def save_to_file(self, sync_tolerance=0.01):
        """Legacy method - calls process_data with save_data=True."""
        return self.process_data(sync_tolerance=sync_tolerance, save_data=True)

    def get_data(self, sync_tolerance=0.01):
        """Get data as numpy arrays without saving to file."""
        return self.process_data(sync_tolerance=sync_tolerance, save_data=False)

    def _forward_fill_missing_data(self, records):
        """
        Forward-fill missing sensor data with the last valid value.
        
        Args:
            records: List of records with potentially None values
            
        Returns:
            List of records with None values replaced by forward-filled data
        """
        if not records:
            return records
            
        filled_records = []
        
        # Track last valid values for each sensor
        last_valid = {
            'imu_left': None,
            'imu_right': None, 
            'imu_vectornav': None,
            'magnetic': [0.0, 0.0, 0.0]  # Default for magnetic if never seen
        }
        
        for record in records:
            filled_record = record.copy()
            
            # Forward-fill each sensor type
            for sensor_key in ['imu_left', 'imu_right', 'imu_vectornav', 'magnetic']:
                if filled_record.get(sensor_key) is not None:
                    # Update last valid value
                    last_valid[sensor_key] = filled_record[sensor_key]
                else:
                    # Use last valid value (forward-fill)
                    if last_valid[sensor_key] is not None:
                        filled_record[sensor_key] = last_valid[sensor_key]
                        
            filled_records.append(filled_record)
        
        # Handle case where the very first records have missing data
        # Find the first record with complete data and backfill
        first_complete_idx = None
        for i, record in enumerate(filled_records):
            if all(record.get(k) is not None for k in ['imu_left', 'imu_right', 'imu_vectornav']):
                first_complete_idx = i
                break
                
        if first_complete_idx is not None and first_complete_idx > 0:
            # Backfill the initial records
            complete_record = filled_records[first_complete_idx]
            for i in range(first_complete_idx):
                for sensor_key in ['imu_left', 'imu_right', 'imu_vectornav']:
                    if filled_records[i].get(sensor_key) is None:
                        filled_records[i][sensor_key] = complete_record[sensor_key]
                        
        return filled_records

    def _estimate_memory_usage(self, records):
        """Estimate memory usage of the dataset in MB."""
        if not records:
            return 0.0
        
        # Estimate based on data structure
        n_samples = len(records)
        
        # Each IMU record: 6 floats (8 bytes each) = 48 bytes
        # Magnetic: 3 floats = 24 bytes  
        # Timestamp: 1 float = 8 bytes
        # Total per sample ≈ 48*3 + 24 + 8 = 176 bytes
        
        estimated_bytes = n_samples * 176
        estimated_mb = estimated_bytes / (1024 * 1024)
        
        return estimated_mb

class RosbagVideoReader:
    def __init__(self, bagpath, out_file, skip_first_n=0, skip_last_n=0):
        self.bagpath = bagpath
        self.out_file = out_file
        self.skip_first_n = skip_first_n
        self.skip_last_n = skip_last_n
    @staticmethod
    def ros_image_to_cv2(msg):
        """
        Convert ROS Image message to OpenCV image.
        
        Args:
            msg: ROS Image message
            
        Returns:
            OpenCV image (numpy array)
        """
        # Get image dimensions
        height = msg.height
        width = msg.width
        encoding = msg.encoding
        
        # Convert data to numpy array
        if isinstance(msg.data, bytes):
            img_data = np.frombuffer(msg.data, dtype=np.uint8)
        else:
            img_data = np.array(msg.data, dtype=np.uint8)
        
        # Reshape based on encoding
        if encoding == "bgr8":
            img = img_data.reshape((height, width, 3))
        elif encoding == "rgb8":
            img = img_data.reshape((height, width, 3))
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif encoding == "mono8":
            img = img_data.reshape((height, width))
            # Convert grayscale to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif encoding == "bgra8":
            img = img_data.reshape((height, width, 4))
            # Convert BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif encoding == "rgba8":
            img = img_data.reshape((height, width, 4))
            # Convert RGBA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError(f"Unsupported image encoding: {encoding}")
        
        return img
        
    def get_rgb_timestamps(self):
        """
        Get timestamps for all RGB frames (after skipping first/last N).
        
        Returns:
            numpy array of timestamps in seconds (Unix epoch)
        """
        print(f"Reading RGB timestamps from bag: {self.bagpath}")
        topic_name = "/third_person_cam/camera/camera/color/image_raw"
        # topic_name = "/left_camera/camera/camera/color/image_raw"
        
        with AnyReader([self.bagpath]) as reader:
            camera_conn = None
            for conn in reader.connections:
                if conn.topic == topic_name:
                    camera_conn = conn
                    break
            
            if camera_conn is None:
                raise RuntimeError(f"Camera topic {topic_name} not found")
            
            timestamps = []
            frame_count = 0
            
            for conn, ts, raw in reader.messages(connections=[camera_conn]):
                if frame_count < self.skip_first_n:
                    frame_count += 1
                    continue
                
                msg = deserialize_cdr(raw, conn.msgtype)
                try:
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                except:
                    timestamp = ts * 1e-9
                timestamps.append(timestamp)
                frame_count += 1
            
            # Skip last N
            if self.skip_last_n > 0 and len(timestamps) > self.skip_last_n:
                timestamps = timestamps[:-self.skip_last_n]
            
            timestamps = np.array(timestamps)
            print(f"Got {len(timestamps)} RGB timestamps")
            print(f"Range: {timestamps[0]:.2f} to {timestamps[-1]:.2f} seconds")
            return timestamps
    
    def process_data(self):
        """
        Read camera images from ROS bag and save as MP4 video.
        
        Topic: third_person_cam/camera/camera/color/image_raw
        """
        print(f"Reading video data from bag: {self.bagpath}")
        topic_name = "/third_person_cam/camera/camera/color/image_raw"
        # topic_name = "/left_camera/camera/camera/color/image_raw"
        
        
        # Open the bag file and find the camera topic
        with AnyReader([self.bagpath]) as reader:
            # Find the camera connection
            camera_conn = None
            for conn in reader.connections:
                if conn.topic == topic_name:
                    camera_conn = conn
                    print(f"Found camera topic: {topic_name}")
                    print(f"Message type: {conn.msgtype}")
                    break
            
            if camera_conn is None:
                raise RuntimeError(f"Camera topic {topic_name} not found in bag file")
            
            # Read all messages and convert to OpenCV images
            frames = []
            timestamps = []
            frame_count = 0
            
            for conn, ts, raw in reader.messages(connections=[camera_conn]):
                msg = deserialize_cdr(raw, conn.msgtype)
                
                # Skip first N frames
                if frame_count < self.skip_first_n:
                    frame_count += 1
                    continue
                
                # Convert ROS image message to OpenCV image
                try:
                    img = self.ros_image_to_cv2(msg)
                    frames.append(img)
                    
                    # Get timestamp
                    try:
                        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    except:
                        timestamp = ts * 1e-9
                    timestamps.append(timestamp)
                    
                except Exception as e:
                    print(f"Warning: Failed to convert image: {e}")
                    continue
                
                frame_count += 1
            
            if not frames:
                raise RuntimeError("No valid frames found in bag file")
            
            # Skip last N frames by trimming the lists
            if self.skip_last_n > 0 and len(frames) > self.skip_last_n:
                frames = frames[:-self.skip_last_n]
                timestamps = timestamps[:-self.skip_last_n]
                print(f"Skipped last {self.skip_last_n} frames")
            
            if self.skip_first_n > 0:
                print(f"Skipped first {self.skip_first_n} frames")
            print(f"Extracted {len(frames)} frames")
            
            # Calculate video properties
            height, width = frames[0].shape[:2]
            
            # Calculate frame rate from timestamps using median interval
            if len(timestamps) > 1:
                # Calculate time differences between consecutive frames
                time_diffs = np.diff(timestamps)
                median_dt = np.median(time_diffs)
                mean_dt = np.mean(time_diffs)
                fps_median = 1.0 / median_dt if median_dt > 0 else 30.0
                fps_mean = 1.0 / mean_dt if mean_dt > 0 else 30.0
                
                time_range = timestamps[-1] - timestamps[0]
                fps_overall = len(timestamps) / time_range if time_range > 0 else 30.0
                
                # Use median-based FPS as it's more robust to outliers
                fps = fps_median
                
                print(f"Time range: {time_range:.2f}s, {len(timestamps)} frames")
                print(f"FPS (overall): {fps_overall:.2f}, FPS (median dt): {fps_median:.2f}, FPS (mean dt): {fps_mean:.2f}")
                print(f"Using FPS: {fps:.2f}")
            else:
                fps = 30.0  # Default fallback
                print(f"Only one frame, using default FPS: {fps}")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(self.out_file), fourcc, fps, (width, height))
            
            # Write all frames
            for frame in frames:
                out.write(frame)
            
            out.release()
            print(f"Saved video to {self.out_file}")
            print(f"Video properties: {width}x{height} @ {fps:.2f} fps")

    def find_depth(self, pixel_array):
        """
        Find depth values for given pixel coordinates and frame indices.
        
        Args:
            pixel_array: Numpy array of shape (N, 3) with columns [u, v, frame_idx]
                        u: x pixel coordinate
                        v: y pixel coordinate  
                        frame_idx: frame index (0, 1, 2, ...)
        
        Returns:
            Numpy array of shape (N, 4) with columns [u, v, d, frame_idx]
                        u: x pixel coordinate
                        v: y pixel coordinate
                        d: depth in millimeters
                        frame_idx: frame index (same as input)
        """
        print(f"Finding depth data from bag: {self.bagpath}")
        topic_name = "/third_person_cam/camera/camera/aligned_depth_to_color/image_raw"
        # topic_name = "/left_camera/camera/camera/color/image_raw"
        
        
        # Convert to numpy array if not already
        pixel_array = np.array(pixel_array)
        if pixel_array.shape[1] != 3:
            raise ValueError(f"Input array must have shape (N, 3), got {pixel_array.shape}")
        
        # Get RGB timestamps to ensure we have the right frame count
        rgb_timestamps = self.get_rgb_timestamps()
        print(f"RGB has {len(rgb_timestamps)} frames")
        
        # Open the bag file and find the depth topic
        with AnyReader([self.bagpath]) as reader:
            # Find the depth connection
            depth_conn = None
            for conn in reader.connections:
                if conn.topic == topic_name:
                    depth_conn = conn
                    print(f"Found depth topic: {topic_name}")
                    print(f"Message type: {conn.msgtype}")
                    break
            
            if depth_conn is None:
                raise RuntimeError(f"Depth topic {topic_name} not found in bag file")

            # Read and convert all depth messages
            depth_data = []  # List of (timestamp, depth_image_array)
            
            for conn, ts, raw in reader.messages(connections=[depth_conn]):
                msg = deserialize_cdr(raw, conn.msgtype)
                
                try:
                    # Get timestamp from message header
                    try:
                        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    except:
                        timestamp = ts * 1e-9
                    
                    # Get image dimensions
                    height = msg.height
                    width = msg.width
                    step = msg.step
                    
                    # Convert data to bytes
                    if isinstance(msg.data, bytes):
                        data_bytes = msg.data
                    else:
                        data_bytes = bytes(msg.data)
                    
                    # Read as uint16 (depth in mm) and reshape
                    depth_img = np.frombuffer(data_bytes, dtype=np.uint16)
                    
                    # Handle step (row stride) - extract valid pixels
                    pixels_per_row = step // 2  # step is in bytes, depth is uint16 (2 bytes)
                    depth_img = depth_img.reshape((height, pixels_per_row))
                    depth_img = depth_img[:, :width]  # Take only valid width
                    
                    depth_data.append((timestamp, depth_img))
                    
                except Exception as e:
                    print(f"Warning: Failed to convert depth frame: {e}")
                    continue
            
            if not depth_data:
                raise RuntimeError("No valid depth frames found")
            
            print(f"Extracted {len(depth_data)} depth frames ({depth_data[0][1].shape[0]}x{depth_data[0][1].shape[1]})")
            print(f"Timestamp range: {depth_data[0][0]:.2f} to {depth_data[-1][0]:.2f} seconds")
            
            # Match pixels to depth frames using frame indices
            result_array = np.zeros((len(pixel_array), 4))
            
            for i, (u, v, frame_idx) in enumerate(pixel_array):
                u_int, v_int = int(u), int(v)
                frame_idx = int(frame_idx)
                
                if 0 <= frame_idx < len(depth_data):
                    depth_img = depth_data[frame_idx][1]
                    
                    # Get depth value
                    if 0 <= v_int < depth_img.shape[0] and 0 <= u_int < depth_img.shape[1]:
                        depth_value = float(depth_img[v_int, u_int])
                    else:
                        depth_value = 0.0
                    
                    result_array[i] = [u, v, depth_value, frame_idx]
                    
                    if i < 3:  # Debug first 3
                        print(f"  Pixel ({u_int}, {v_int}) at frame {frame_idx} -> depth={depth_value:.0f}mm ({depth_value/1000:.2f}m)")
                        
                        # Show 3x3 neighborhood for context
                        v_start, v_end = max(0, v_int-1), min(depth_img.shape[0], v_int+2)
                        u_start, u_end = max(0, u_int-1), min(depth_img.shape[1], u_int+2)
                        neighborhood = depth_img[v_start:v_end, u_start:u_end]
                        print(f"    3x3 neighborhood (mm):\n{neighborhood}")
                        valid_neighbors = neighborhood[neighborhood > 0]
                        if len(valid_neighbors) > 0:
                            print(f"    Min: {valid_neighbors.min():.0f}mm, Max: {valid_neighbors.max():.0f}mm, Median: {np.median(valid_neighbors):.0f}mm")
                else:
                    result_array[i] = [u, v, 0.0, frame_idx]
                    if i < 3:
                        print(f"  Pixel ({u_int}, {v_int}) at frame {frame_idx} -> OUT OF RANGE (max: {len(depth_data)-1})")
            
            print(f"Matched {len(result_array)} pixels with depth values")
            return result_array
    
    def save_depth_video(self, out_file=None, colormap=cv2.COLORMAP_JET):
        """
        Save depth data as an MP4 video with color mapping.
        
        Args:
            out_file: Output video file path (if None, appends '_depth' to self.out_file)
            colormap: OpenCV colormap for depth visualization (default: COLORMAP_JET)
        """
        if out_file is None:
            # Create depth video filename
            out_path = Path(self.out_file)
            out_file = out_path.parent / (out_path.stem + '_depth' + out_path.suffix)
        
        print(f"Saving depth video to: {out_file}")
        topic_name = "/third_person_cam/camera/camera/aligned_depth_to_color/image_raw"
        # topic_name = "/left_camera/camera/camera/color/image_raw"
        
        
        with AnyReader([self.bagpath]) as reader:
            # Find the depth connection
            depth_conn = None
            for conn in reader.connections:
                if conn.topic == topic_name:
                    depth_conn = conn
                    print(f"Found depth topic: {topic_name}")
                    break
            
            if depth_conn is None:
                raise RuntimeError(f"Depth topic {topic_name} not found in bag file")
            
            # Read all depth frames
            depth_frames = []
            timestamps = []
            frame_count = 0
            
            for conn, ts, raw in reader.messages(connections=[depth_conn]):
                msg = deserialize_cdr(raw, conn.msgtype)
                
                # Skip first/last N frames (same as RGB)
                if frame_count < self.skip_first_n:
                    frame_count += 1
                    continue
                
                try:
                    # Get timestamp
                    try:
                        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    except:
                        timestamp = ts * 1e-9
                    
                    # Convert depth data
                    height = msg.height
                    width = msg.width
                    step = msg.step
                    
                    if isinstance(msg.data, bytes):
                        data_bytes = msg.data
                    else:
                        data_bytes = bytes(msg.data)
                    
                    depth_img = np.frombuffer(data_bytes, dtype=np.uint16)
                    pixels_per_row = step // 2
                    depth_img = depth_img.reshape((height, pixels_per_row))
                    depth_img = depth_img[:, :width]
                    
                    depth_frames.append(depth_img)
                    timestamps.append(timestamp)
                    
                except Exception as e:
                    print(f"Warning: Failed to convert depth frame: {e}")
                    continue
                
                frame_count += 1
            
            # Skip last N frames
            if self.skip_last_n > 0 and len(depth_frames) > self.skip_last_n:
                depth_frames = depth_frames[:-self.skip_last_n]
                timestamps = timestamps[:-self.skip_last_n]
                print(f"Skipped last {self.skip_last_n} frames")
            
            if not depth_frames:
                raise RuntimeError("No depth frames found")
            
            if self.skip_first_n > 0:
                print(f"Skipped first {self.skip_first_n} frames")
            print(f"Extracted {len(depth_frames)} depth frames")
            
            # Calculate FPS
            height, width = depth_frames[0].shape
            if len(timestamps) > 1:
                time_diffs = np.diff(timestamps)
                median_dt = np.median(time_diffs)
                fps = 1.0 / median_dt if median_dt > 0 else 30.0
                time_range = timestamps[-1] - timestamps[0]
                print(f"Time range: {time_range:.2f}s, FPS: {fps:.2f}")
            else:
                fps = 30.0
            
            # Normalize and colorize depth frames
            print("Converting depth to color...")
            color_frames = []
            
            # Find min/max depth for normalization (exclude zeros)
            all_depths = np.concatenate([d.flatten() for d in depth_frames])
            valid_depths = all_depths[all_depths > 0]
            min_depth = np.percentile(valid_depths, 1) if len(valid_depths) > 0 else 0
            max_depth = np.percentile(valid_depths, 99) if len(valid_depths) > 0 else 5000
            print(f"Depth range: {min_depth:.0f}mm to {max_depth:.0f}mm")
            
            for depth_img in depth_frames:
                # Normalize to 0-255 and INVERT (so close=255=red, far=0=blue)
                depth_normalized = np.clip(depth_img, min_depth, max_depth)
                depth_normalized = ((depth_normalized - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
                depth_normalized = 255 - depth_normalized  # INVERT: close objects = high value = red
                
                # Apply colormap (now red=close, blue=far)
                depth_color = cv2.applyColorMap(depth_normalized, colormap)
                color_frames.append(depth_color)
            
            # Write video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))
            
            for frame in color_frames:
                out.write(frame)
            
            out.release()
            print(f"Saved depth video: {out_file}")
            print(f"Video properties: {width}x{height} @ {fps:.2f} fps")
                

def main():

    # # Read IMU 
    # bagpath = Path("/home/jdx/Documents/1.0LatentAct/datasets/WearableData-09-24-25-21-16-04")
    # out_file = Path("posEstimate/data/multi_imu_raw.jsonl")
    # reader = RosbagReader(bagpath, out_file)
    # sensor_data = reader.process_data(save_data=False)
    # print("Test imu_left data shape", sensor_data['imu_left'].shape)

    # Read RGB
    bagpath = Path("/home/jdx/Downloads/demo")
    out_file = Path("posEstimate/data/demo.mp4")
    Videoreader = RosbagVideoReader(bagpath, out_file, skip_first_n=0, skip_last_n=0)
    
    Videoreader.process_data()
    Videoreader.save_depth_video()
    
    # Test depth lookup using frame indices (much simpler!)
    pixel_array = np.array([
        [262, 138, 0],   # pixel at frame 5
        [284, 155, 1],   # pixel at frame 3  
        [229, 113, 20]   # pixel at frame 20
    ])
    full_array = Videoreader.find_depth(pixel_array)
    print("\nu, v, d, frame:")
    print(full_array)
    print(f"\nDepths: {full_array[:, 2]} mm = {full_array[:, 2]/1000} meters")

if __name__ == "__main__":
    main()
