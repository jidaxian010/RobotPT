from pathlib import Path
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.transform import Rotation
from scipy.integrate import cumulative_trapezoid

from rosbags.highlevel import AnyReader
from rosbags.serde import deserialize_cdr

import torch
import pypose as pp

from ahrs.filters import Madgwick


class IMUReader:
    def __init__(self, bagpath, imu_topic, out_file, start_at_zero=True):
        self.bagpath = bagpath
        self.imu_topic = imu_topic
        self.out_file = out_file
        self.start_at_zero = start_at_zero

    @staticmethod
    def get_stamp_sec(msg, fallback_ts_ns):
        try:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception:
            return fallback_ts_ns * 1e-9

    @staticmethod
    def parse_imu(msg):
        return [
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
            float(msg.linear_acceleration.z),
            float(msg.angular_velocity.x),
            float(msg.angular_velocity.y),
            float(msg.angular_velocity.z),
        ]

    def read_imu_series(self, reader):
        """Read IMU data from the specified topic"""
        conns = [c for c in reader.connections if c.topic == self.imu_topic]
        if not conns:
            raise RuntimeError(f"Topic {self.imu_topic} not found")
        
        out = []
        for conn, ts, raw in reader.messages(connections=conns):
            msg = deserialize_cdr(raw, conn.msgtype)
            t = self.get_stamp_sec(msg, ts)
            vals = self.parse_imu(msg)
            out.append((t, vals))
        
        return sorted(out, key=lambda x: x[0])

    def save_to_file(self):
        """Read IMU data and save to JSONL file"""
        print(f"Reading IMU data from topic: {self.imu_topic}")
        print(f"Bag path: {self.bagpath}")
        
        with AnyReader([self.bagpath]) as reader:
            imu_series = self.read_imu_series(reader)
        
        if not imu_series:
            raise RuntimeError("No IMU data found")
        
        # Adjust timestamps to start from zero if requested
        t0 = imu_series[0][0] if self.start_at_zero else 0.0
        
        print(f"Found {len(imu_series)} IMU samples")
        print(f"Time range: {imu_series[0][0]:.3f}s to {imu_series[-1][0]:.3f}s")
        
        # Calculate sampling rate
        if len(imu_series) > 1:
            timestamps = [t for t, _ in imu_series]
            dts = np.diff(timestamps)
            median_dt = np.median(dts[dts > 0]) if np.any(dts > 0) else 0.0
            if median_dt > 0:
                print(f"Approximate sampling rate: {1/median_dt:.1f} Hz")
        
        # Save to JSONL file
        with open(self.out_file, 'w') as f:
            for t, imu_vals in imu_series:
                timestamp = t - t0 if self.start_at_zero else t
                rec = {
                    "imu_left": imu_vals,
                    "timestamp": timestamp,
                }
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        
        print(f"Saved {len(imu_series)} samples to {self.out_file}")


class ProcessRaw:
    def __init__(self, jsonl_file, imu_to_gripper_R=None, imu_to_gripper_T=None):
        """Initialize raw data processor"""
        self.jsonl_file = jsonl_file
        self.data = self.load_data()
        self.dt = self.calculate_dt()
        print(f"dt: {self.dt}")
        self.fs = 1.0 / self.dt  # sampling frequency
        
        # Set up IMU to world frame transformation
        if imu_to_gripper_R is None:
            # self.imu_to_gripper_R = np.eye(3)
            # print("Using identity transformation (no IMU to world frame rotation)")
            raise ValueError("imu_to_gripper_R is None")
        else:
            self.imu_to_gripper_R = np.array(imu_to_gripper_R)
            print(f"Using custom IMU to world frame transformation:")
            print(f"R = {self.imu_to_gripper_R}")
        
        # Create rotation object for easy application
        self.imu_to_gripper_rotation = Rotation.from_matrix(self.imu_to_gripper_R)

        if imu_to_gripper_T is None:
            # self.imu_to_gripper_T = np.zeros(3)
            # print("Using identity transformation (no IMU to world frame translation)")
            raise ValueError("imu_to_gripper_T is None")
        else:
            self.imu_to_gripper_T = np.array(imu_to_gripper_T)
            print(f"Using custom IMU to world frame translation:")
            print(f"T = {self.imu_to_gripper_T}")
        
    def load_data(self):
        """Load data from JSONL file"""
        data = []
        with open(self.jsonl_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def calculate_dt(self):
        """Calculate average time step"""
        timestamps = [d['timestamp'] for d in self.data]
        dts = np.diff(timestamps)
        return np.median(dts)
    
    def extract_sensor_data(self, start_timestep):
        """Extract left IMU data only, starting from specified timestep"""
        timestamps = []
        accel_data = []
        gyro_data = []
        
        # Skip the first start_timestep entries
        data_subset = self.data[start_timestep:]
        print(f"Starting from timestep {start_timestep}, processing {len(data_subset)} samples out of {len(self.data)} total samples")
        
        for entry in data_subset:
            timestamps.append(entry['timestamp'])
            
            # Extract left IMU data only
            imu_left = np.array(entry['imu_left'])   # [ax, ay, az, gx, gy, gz]
            
            accel_data.append(imu_left[:3])   # [ax, ay, az]
            gyro_data.append(imu_left[3:])    # [gx, gy, gz]
        
        return np.array(timestamps), np.array(accel_data), np.array(gyro_data)
    
    def remove_spikes(self, data, spike_threshold=3.0, window_size=1):
        """Remove spikes using median filter and z-score, then apply window averaging"""
        filtered_data = np.copy(data)
        
        for i in range(data.shape[1]):  # For each axis
            # Step 1: Apply median filter to detect spikes
            median_filtered = signal.medfilt(data[:, i], kernel_size=5)
            
            # Step 2: Calculate z-score of residuals
            residuals = data[:, i] - median_filtered
            z_scores = np.abs(residuals) / (np.std(residuals) + 1e-8)
            
            # Step 3: Replace spikes with median filtered values
            spike_mask = z_scores > spike_threshold
            temp_data = data[:, i].copy()
            temp_data[spike_mask] = median_filtered[spike_mask]
            
            if np.sum(spike_mask) > 0:
                print(f"  Removed {np.sum(spike_mask)} spikes from axis {i}")
            
            # Step 4: Apply window averaging
            if len(temp_data) >= window_size:
                # Apply moving average using convolution
                moving_avg = np.convolve(temp_data, np.ones(window_size)/window_size, mode='same')
                filtered_data[:, i] = moving_avg
                print(f"  Applied window averaging with window size {window_size}")
            else:
                filtered_data[:, i] = temp_data
                print(f"  Data too short for window averaging (length: {len(temp_data)}, window: {window_size})")
        
        return filtered_data
    

    def gyro_armframe(self, gyro_data):
        """Transform gyroscope data from IMU frame to arm frame"""
        gyro_arm = np.zeros_like(gyro_data)
        for i, gyro_imu in enumerate(gyro_data):
            gyro_arm[i] = self.imu_to_gripper_R @ gyro_imu
        # Apply light smoothing to reduce noise while preserving useful data
        # Option 1: Moving average with small window
        window_size = 100
        if len(gyro_arm) > window_size:
            # Apply moving average filter
            for i in range(gyro_arm.shape[1]):  # For each axis
                gyro_arm[:, i] = np.convolve(gyro_arm[:, i], np.ones(window_size)/window_size, mode='same')

        return gyro_arm

    def accel_armframe(self, accel_clean, gyro_clean):
        """
        Transform acceleration from IMU frame to arm frame.
        
        Complete transformation equation: a_arm = a_clean + angular_accel * T
        
      
        """
        R_matrix = self.imu_to_gripper_R
        T_vector = self.imu_to_gripper_T
        accel_clean_arm = accel_clean @ R_matrix.T
        gyro_clean_arm = gyro_clean @ R_matrix.T


        
        # Apply rotation to each row: (R @ a^T)^T = a @ R^T
        linear_accel = accel_clean_arm
        
        # Apply rotation to cross products
        angular_accel = np.gradient(gyro_clean_arm, axis=0)
        gyro_accel = np.cross(angular_accel, T_vector)
        
        # Cross product terms
        centripetal_accel = np.cross(gyro_clean_arm, np.cross(gyro_clean_arm, T_vector))
        
        accel_arm = linear_accel + gyro_accel + centripetal_accel
        print(f"[Debug] linear_accel: {np.mean(linear_accel, axis=0)} {np.std(linear_accel, axis=0)}")
        print(f"[Debug] gyro_accel: {np.mean(gyro_accel, axis=0)} {np.std(gyro_accel, axis=0)}")
        print(f"[Debug] centripetal_accel: {np.mean(centripetal_accel, axis=0)} {np.std(centripetal_accel, axis=0)}")
        print(f"[Debug] accel_arm: {np.mean(accel_arm, axis=0)} {np.std(accel_arm, axis=0)}")
        return accel_arm



    def accel_worldframe(self, accel_gripper, gyro_gripper=None, win_start=0, win_size=1500):
        """
        Transform acceleration from gripper frame to world frame.
        Uses gravity alignment to properly orient the coordinate frame.
        
        Args:
            accel_gripper (np.array): Acceleration in gripper frame [N, 3]
            gyro_gripper (np.array, optional): Gyroscope data (unused, kept for compatibility)
            win_start (int): Start index for calibration window (unused, kept for compatibility)
            win_size (int): Size of calibration window (unused, kept for compatibility)
            
        Returns:
            np.array: Acceleration in world frame [N, 3] where Z is up
        """
        # Use gravity alignment to find the proper rotation matrix
        R_arm_to_world = self.accel_align_gravity(accel_gripper)
        
        # Apply rotation to transform from arm frame to world frame
        accel_world = np.zeros_like(accel_gripper)
        for i, accel_g in enumerate(accel_gripper):
            accel_world[i] = R_arm_to_world @ accel_g
        
        print(f"accel_world at first timestep: {accel_world[0]}")
        
        return accel_world
    
    def accel_align_gravity(self, accel_arm):
        """
        Align the first acceleration measurement with world frame gravity [0, 0, 9.8].
        This function finds the rotation needed to align accel_arm[0] with [0, 0, 9.8].
        
        Args:
            accel_arm (np.array): Acceleration in arm frame [N, 3]
            
        Returns:
            np.array: Rotation matrix from arm frame to world frame [3, 3]
        """
        # Get the first acceleration measurement (should represent gravity in arm frame)
        accel_initial = accel_arm[0]
        
        # Target gravity vector in world frame
        gravity_world = np.array([0, 0, 9.8])
        
        print(f"Initial acceleration in arm frame: {accel_initial}")
        print(f"Target gravity in world frame: {gravity_world}")
        
        # Normalize both vectors
        accel_initial_norm = accel_initial / np.linalg.norm(accel_initial)
        gravity_world_norm = gravity_world / np.linalg.norm(gravity_world)
        
        print(f"Normalized initial acceleration: {accel_initial_norm}")
        print(f"Normalized target gravity: {gravity_world_norm}")
        
        # Calculate rotation axis using cross product
        rotation_axis = np.cross(accel_initial_norm, gravity_world_norm)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        # Check if vectors are already aligned (cross product is zero)
        if rotation_axis_norm < 1e-6:
            if np.dot(accel_initial_norm, gravity_world_norm) > 0:
                # Vectors are already aligned, return identity
                print("Vectors are already aligned")
                return np.eye(3)
            else:
                # Vectors are opposite, need 180-degree rotation
                print("Vectors are opposite, applying 180-degree rotation")
                # Find a perpendicular vector for rotation axis
                if abs(accel_initial_norm[0]) < 0.9:
                    rotation_axis = np.array([1, 0, 0])
                else:
                    rotation_axis = np.array([0, 1, 0])
                rotation_axis = rotation_axis - np.dot(rotation_axis, accel_initial_norm) * accel_initial_norm
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_angle = np.pi
        else:
            # Normalize rotation axis
            rotation_axis = rotation_axis / rotation_axis_norm
            
            # Calculate rotation angle using dot product
            rotation_angle = np.arccos(np.clip(np.dot(accel_initial_norm, gravity_world_norm), -1.0, 1.0))
        
        print(f"Rotation axis: {rotation_axis}")
        print(f"Rotation angle: {np.degrees(rotation_angle):.2f} degrees")
        
        # Create rotation matrix using Rodrigues' formula
        # R = I + sin(θ) * K + (1 - cos(θ)) * K²
        # where K is the skew-symmetric matrix of the rotation axis
        
        # Skew-symmetric matrix of rotation axis
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        # Rodrigues' rotation formula
        R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
        
        print(f"Rotation matrix:\n{R}")
        
        # Verify the rotation works
        accel_aligned = R @ accel_initial
        print(f"Acceleration after alignment: {accel_aligned}")
        print(f"Error magnitude: {np.linalg.norm(accel_aligned - gravity_world):.6f}")
        
        return R
    
    def shift_accel_to_gravity(self, accel_world, target_gravity=[0, 0, 9.8]):
        """
        Shift acceleration data so that the first timestep starts at target_gravity.
        This helps eliminate initial bias that causes drift.
        
        Args:
            accel_world (np.array): Acceleration in world frame [N, 3]
            target_gravity (list): Target gravity vector [x, y, z]
            
        Returns:
            np.array: Shifted acceleration data [N, 3]
        """
        # Calculate the shift needed
        initial_accel = accel_world[0]
        shift = np.array(target_gravity) - initial_accel
        
        print(f"Initial acceleration: {initial_accel}")
        print(f"Target gravity: {target_gravity}")
        print(f"Applying shift: {shift}")
        
        # Apply the shift to all data
        accel_shifted = accel_world + shift
        
        print(f"Shifted first timestep: {accel_shifted[0]}")
        
        return accel_shifted
    
    def _apply_highpass_filter(self, accel_world, cutoff_freq=0.1, fs=None):
        """
        Apply high-pass filter to remove DC bias and low-frequency drift.
        
        Args:
            accel_world (np.array): Acceleration data [N, 3]
            cutoff_freq (float): Cutoff frequency in Hz (default: 0.1 Hz)
            fs (float): Sampling frequency in Hz (if None, calculated from dt)
            
        Returns:
            np.array: High-pass filtered acceleration data [N, 3]
        """
        if fs is None:
            fs = 1.0 / self.dt  # sampling frequency
        
        print(f"Applying high-pass filter: cutoff={cutoff_freq} Hz, fs={fs:.1f} Hz")
        
        # Design high-pass Butterworth filter
        # Normalize cutoff frequency
        nyquist = fs / 2.0
        normal_cutoff = cutoff_freq / nyquist
        
        # Ensure cutoff is valid (must be < 1.0)
        if normal_cutoff >= 1.0:
            print(f"Warning: cutoff frequency {cutoff_freq} Hz is too high for sampling rate {fs:.1f} Hz")
            normal_cutoff = 0.99
        
        # Design 4th order Butterworth high-pass filter
        sos = signal.butter(4, normal_cutoff, btype='high', output='sos')
        
        # Apply filter to each axis
        accel_filtered = np.zeros_like(accel_world)
        for i in range(3):  # for each axis (x, y, z)
            accel_filtered[:, i] = signal.sosfilt(sos, accel_world[:, i])
        
        print(f"Before filtering - mean: {accel_world.mean(axis=0)}")
        print(f"After filtering - mean: {accel_filtered.mean(axis=0)}")
        
        return accel_filtered
    
    def integrate_accel_worldframe(self, test_accel_world):
        """Integrate acceleration in world frame using trapezoidal integration"""
        
        print(f"test_accel_world at first timestep: {test_accel_world[0]}")
        print(f"test_accel_world average: {test_accel_world.mean(axis=0)}")
        print(f"test_accel_world shape: {test_accel_world.shape}")

        # # Apply high-pass filter to remove DC bias and low-frequency drift
        # test_accel_world = self._apply_highpass_filter(test_accel_world, cutoff_freq=0.2)

        # Create time array
        time = np.arange(len(test_accel_world)) * self.dt
        print(f"time array length: {len(time)}")
        print(f"dt: {self.dt}")
        print(f"time range: {time[0]:.6f} to {time[-1]:.6f} seconds")

        # Integrate acceleration to get velocity (cumulative trapezoidal rule)
        velocity = np.zeros_like(test_accel_world)
        for i in range(3):  
            velocity[:, i] = cumulative_trapezoid(test_accel_world[:, i], x=time, initial=0)

        # Integrate velocity to get position (cumulative trapezoidal rule)
        position = np.zeros_like(test_accel_world)
        for i in range(3):  
            position[:, i] = cumulative_trapezoid(velocity[:, i], x=time, initial=0)

        return position, velocity

    
    

    def quaternion_to_rotation_matrix(self, qauternion):
        w, x, y, z = qauternion
        Rotation_matrix = np.array([
            [1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),         1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x**2 + y**2)]
        ])
        return Rotation_matrix


    
    def find_position(self, gyro_arm, accel_world):
        """Find pos using pypose"""
        # p = torch.zeros(3)    
        # r = pp.identity_SO3() 
        # v = torch.zeros(3)    
        # integrator = pp.module.IMUPreintegrator(p, r, v)

        # convert gyro, accel to torch
        gyro_arm_torch  = torch.from_numpy(gyro_arm).to(dtype=torch.double)
        accel_world_torch = torch.from_numpy(accel_world).to(dtype=torch.double)
        T = gyro_arm_torch.shape[0]

        # Build dt with matching time axis (and batch axis):
        dt_torch = torch.full((1, T, 1), float(self.dt), dtype=torch.double)   # [1,T,1]
        gyro_b   = gyro_arm_torch.unsqueeze(0)                                  # [1,T,3]
        acc_b    = accel_world_torch.unsqueeze(0)                               # [1,T,3]

        p0 = torch.zeros(1, 1, 3, dtype=torch.double)
        v0 = torch.zeros(1, 1, 3, dtype=torch.double)
        r0 = pp.so3(torch.zeros(1, 1, 3, dtype=torch.double)).Exp()  # identity SO3
        # R_total = estimate_Rtotal(acc_grip, gyro_grip)
        # r0 = pp.mat2SO3(torch.tensor(R_total, dtype=torch.float64)).lview(1,1)


        integrator = pp.module.IMUPreintegrator(p0, r0, v0, gravity=9.76005, reset=False).to(torch.double)
        states = integrator(dt=dt_torch, gyro=gyro_b, acc=acc_b)  # omit rot unless you have it


        # states = integrator(dt_torch, gyro_arm_torch, accel_clean_torch)
        for k, v in states.items():
            print(f"{k}: {v.shape}")


        return states


    
    
    def plot_results(self, gyro_clean, gyro_arm, accel_clean, accel_arm, accel_world, states, test_position, test_velocity, test_accel_world):
        """Plot IMU data, orientation, and 3D position trajectory"""
        fig = plt.figure(figsize=(20, 12))
        
        # plot 2 subplots for gyro_clean and gyro_arm
        ax1 = fig.add_subplot(6, 1, 1)
        ax1.plot(gyro_clean[:, 0], 'r-', label='Gyro X', linewidth=2)
        ax1.plot(gyro_clean[:, 1], 'g-', label='Gyro Y', linewidth=2)
        ax1.plot(gyro_clean[:, 2], 'b-', label='Gyro Z', linewidth=2)
        ax1.set_ylabel('Angular Velocity (rad/s)')
        ax1.set_title('Gyroscope Data (Clean)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(6, 1, 2)
        ax2.plot(gyro_arm[:, 0], 'r-', label='Gyro X', linewidth=2)
        ax2.plot(gyro_arm[:, 1], 'g-', label='Gyro Y', linewidth=2)
        ax2.plot(gyro_arm[:, 2], 'b-', label='Gyro Z', linewidth=2)
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.set_title('Gyroscope Data (Arm Frame)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax3 = fig.add_subplot(6, 1, 3)
        # convert the quaternion to euler angles
        # Try different sequences: 'xyz', 'zyx', 'yxz' - choose the most stable one
        gyro_arm_euler = np.array([Rotation.from_quat(states['rot'][0, i]).as_euler('zyx') for i in range(states['rot'].shape[1])])
        gyro_arm_euler_unwrapped = np.unwrap(gyro_arm_euler, axis=0)
        
        ax3.plot(gyro_arm_euler_unwrapped[:, 0], 'g-', label='Rotation x', linewidth=2)
        ax3.plot(gyro_arm_euler_unwrapped[:, 1], 'b-', label='Rotation y', linewidth=2)
        ax3.plot(gyro_arm_euler_unwrapped[:, 2], 'k-', label='Rotation z', linewidth=2)
        ax3.set_ylabel('Orientation (Euler Angles - Unwrapped)')
        ax3.set_title('Orientation (Euler Angles - Unwrapped)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(6, 1, 4)
        ax4.plot(accel_clean[:, 0], 'r-', label='Acceleration X', linewidth=2)
        ax4.plot(accel_clean[:, 1], 'g-', label='Acceleration Y', linewidth=2)
        ax4.plot(accel_clean[:, 2], 'b-', label='Acceleration Z', linewidth=2)
        ax4.set_ylabel('Acceleration (Clean) (m/s²)')
        ax4.set_title('Raw Acceleration')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(6, 1, 5)
        ax5.plot(accel_world[:, 0], 'r-', label='Acceleration X', linewidth=2)
        ax5.plot(accel_world[:, 1], 'g-', label='Acceleration Y', linewidth=2)
        ax5.plot(accel_world[:, 2], 'b-', label='Acceleration Z', linewidth=2)
        ax5.set_ylabel('Acceleration (Arm Frame) (m/s²)')
        ax5.set_title('Acceleration (Arm Frame)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(6, 1, 6)
        ax6.plot(states['pos'][0, :, 0], 'r-', label='Position X', linewidth=2)
        ax6.plot(states['pos'][0, :, 1], 'g-', label='Position Y', linewidth=2)
        ax6.plot(states['pos'][0, :, 2], 'b-', label='Position Z', linewidth=2)
        ax6.set_ylabel('Position (m)')
        ax6.set_title('Position')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        

        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(20, 12))


        ax1 = fig.add_subplot(5, 1, 1)
        ax1.plot(test_position[:, 0], 'r-', label='Position X', linewidth=2)
        ax1.plot(test_position[:, 1], 'g-', label='Position Y', linewidth=2)
        # ax1.plot(test_position[:, 2], 'b-', label='Position Z', linewidth=2)
        ax1.set_ylabel('Position (m)')
        ax1.set_title('Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(5, 1, 2)
        ax2.plot(test_velocity[:, 0], 'r-', label='Velocity X', linewidth=2)
        ax2.plot(test_velocity[:, 1], 'g-', label='Velocity Y', linewidth=2)
        # ax2.plot(test_velocity[:, 2], 'b-', label='Velocity Z', linewidth=2)
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(5, 1, 3)
        ax3.plot(test_accel_world[:, 0], 'r-', label='Acceleration X', linewidth=2)
        ax3.plot(test_accel_world[:, 1], 'g-', label='Acceleration Y', linewidth=2)
        ax3.plot(test_accel_world[:, 2], 'b-', label='Acceleration Z', linewidth=2)
        ax3.set_ylabel('World Acceleration (m/s²)')
        ax3.set_title('Acceleration')
        ax3.legend()
        ax3.grid(True, alpha=0.3)


        plt.tight_layout()
        plt.show()
    

    def process(self, start_timestep):
        # 1. Load sensor data
        timestamps, accel_data, gyro_data = self.extract_sensor_data(start_timestep)
        
        # 2. Remove spikes from both accelerometer and gyroscope data
        gyro_clean = self.remove_spikes(gyro_data)
        accel_clean = self.remove_spikes(accel_data)


        gyro_arm = self.gyro_armframe(gyro_clean)
        accel_arm = self.accel_armframe(accel_clean, gyro_clean)
        
        # Transform to world frame
        accel_world = self.accel_worldframe(accel_arm)
        
        # # Shift acceleration to start at [0, 0, 9.8] to eliminate initial bias
        # accel_world = self.shift_accel_to_gravity(accel_world, target_gravity=[0, 0, 9.8])
                
        # for test only
        test_accel_world = accel_world[:6000]
        # test_accel_world = accel_world
        test_position, test_velocity = self.integrate_accel_worldframe(test_accel_world)

        # 3. Implement pypose pose estimation
        states = self.find_position(gyro_arm, accel_world)
        
        # 4. Plot results
        self.plot_results(gyro_clean, gyro_arm, accel_clean, accel_arm, accel_world, states, test_position, test_velocity, test_accel_world)

        results = {
            'gyro_clean': gyro_clean,
            'gyro_arm': gyro_arm,
            'accel_clean': accel_clean,
            'accel_arm': accel_arm,
            'accel_world': accel_world,
            'position': states,
        }
        
        
        return results


class DualIMUProcessor:
    """Simple averaging of data from two IMUs"""
    
    def __init__(self, left_processor, right_processor):
        self.left_processor = left_processor
        self.right_processor = right_processor
        self.dt = left_processor.dt  # Assume same sampling rate
    
    def fuse_imu_data(self, left_results, right_results, fusion_weight=0.5):
        """
        Simple averaging of accelerometer and gyroscope data from both IMUs
        
        Args:
            left_results: Results from left IMU processor
            right_results: Results from right IMU processor
            fusion_weight: Weight for left IMU (0-1), right gets (1-weight)
        """
        print("Averaging dual IMU data...")
        
        # Extract data from both IMUs
        left_accel_world = left_results['accel_world']
        right_accel_world = right_results['accel_world']
        left_gyro_arm = left_results['gyro_arm']
        right_gyro_arm = right_results['gyro_arm']
        
        # Ensure same length (take minimum)
        min_length = min(len(left_accel_world), len(right_accel_world))
        left_accel_world = left_accel_world[:min_length]
        right_accel_world = right_accel_world[:min_length]
        left_gyro_arm = left_gyro_arm[:min_length]
        right_gyro_arm = right_gyro_arm[:min_length]
        
        # Simple weighted averaging
        fused_accel_world = (fusion_weight * left_accel_world + 
                            (1 - fusion_weight) * right_accel_world)
        
        fused_gyro_arm = (fusion_weight * left_gyro_arm + 
                         (1 - fusion_weight) * right_gyro_arm)
        
        print(f"Averaged data shape - Accel: {fused_accel_world.shape}, Gyro: {fused_gyro_arm.shape}")
        
        # Re-run pose estimation with averaged data
        print("Running pose estimation with averaged data...")
        fused_states = self.left_processor.find_position(fused_gyro_arm, fused_accel_world)
        
        return {
            'left_results': left_results,
            'right_results': right_results,
            'fused_accel_world': fused_accel_world,
            'fused_gyro_arm': fused_gyro_arm,
            'fused_states': fused_states
        }

    def plot_dual_imu_comparison(self, fused_results):
        """Plot comparison between individual IMUs and fused result"""
        left_results = fused_results['left_results']
        right_results = fused_results['right_results']
        fused_states = fused_results['fused_states']
        
        fig = plt.figure(figsize=(20, 15))
        
        # Extract positions for plotting
        if 'position' in left_results:
            left_pos = left_results['position']['pos'][0].numpy()  # PyPose format
        else:
            left_pos = np.zeros((100, 3))
            
        if 'position' in right_results:
            right_pos = right_results['position']['pos'][0].numpy()  # PyPose format
        else:
            right_pos = np.zeros((100, 3))
            
        fused_pos = fused_states['pos'][0].numpy()
        
        # Position comparison
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(left_pos[:, 0], 'r-', label='Left IMU X', alpha=0.7)
        ax1.plot(right_pos[:, 0], 'g-', label='Right IMU X', alpha=0.7)
        ax1.plot(fused_pos[:, 0], 'b-', label='Fused X', linewidth=2)
        ax1.set_ylabel('Position X (m)')
        ax1.set_title('Position X Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(left_pos[:, 1], 'r-', label='Left IMU Y', alpha=0.7)
        ax2.plot(right_pos[:, 1], 'g-', label='Right IMU Y', alpha=0.7)
        ax2.plot(fused_pos[:, 1], 'b-', label='Fused Y', linewidth=2)
        ax2.set_ylabel('Position Y (m)')
        ax2.set_title('Position Y Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(3, 2, 3)
        ax3.plot(left_pos[:, 2], 'r-', label='Left IMU Z', alpha=0.7)
        ax3.plot(right_pos[:, 2], 'g-', label='Right IMU Z', alpha=0.7)
        ax3.plot(fused_pos[:, 2], 'b-', label='Fused Z', linewidth=2)
        ax3.set_ylabel('Position Z (m)')
        ax3.set_title('Position Z Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 3D trajectory comparison
        ax4 = fig.add_subplot(3, 2, 4, projection='3d')
        ax4.plot(left_pos[:, 0], left_pos[:, 1], left_pos[:, 2], 'r-', label='Left IMU', alpha=0.7)
        ax4.plot(right_pos[:, 0], right_pos[:, 1], right_pos[:, 2], 'g-', label='Right IMU', alpha=0.7)
        ax4.plot(fused_pos[:, 0], fused_pos[:, 1], fused_pos[:, 2], 'b-', label='Fused', linewidth=2)
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Z (m)')
        ax4.set_title('3D Trajectory Comparison')
        ax4.legend()
        
        # Acceleration comparison
        ax5 = fig.add_subplot(3, 2, 5)
        left_accel = left_results['accel_world']
        right_accel = right_results['accel_world']
        fused_accel = fused_results['fused_accel_world']
        
        ax5.plot(np.linalg.norm(left_accel, axis=1), 'r-', label='Left IMU', alpha=0.7)
        ax5.plot(np.linalg.norm(right_accel, axis=1), 'g-', label='Right IMU', alpha=0.7)
        ax5.plot(np.linalg.norm(fused_accel, axis=1), 'b-', label='Fused', linewidth=2)
        ax5.axhline(y=9.81, color='k', linestyle='--', label='Gravity')
        ax5.set_ylabel('Acceleration Magnitude (m/s²)')
        ax5.set_title('Acceleration Magnitude Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Drift analysis
        ax6 = fig.add_subplot(3, 2, 6)
        left_drift = np.linalg.norm(left_pos, axis=1)
        right_drift = np.linalg.norm(right_pos, axis=1)
        fused_drift = np.linalg.norm(fused_pos, axis=1)
        
        ax6.plot(left_drift, 'r-', label=f'Left IMU (final: {left_drift[-1]:.2f}m)', alpha=0.7)
        ax6.plot(right_drift, 'g-', label=f'Right IMU (final: {right_drift[-1]:.2f}m)', alpha=0.7)
        ax6.plot(fused_drift, 'b-', label=f'Fused (final: {fused_drift[-1]:.2f}m)', linewidth=2)
        ax6.set_ylabel('Distance from Origin (m)')
        ax6.set_title('Position Drift Analysis')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print drift statistics
        print(f"\nDrift Analysis:")
        print(f"Left IMU final drift: {left_drift[-1]:.3f} m")
        print(f"Right IMU final drift: {right_drift[-1]:.3f} m")
        print(f"Fused IMU final drift: {fused_drift[-1]:.3f} m")
        print(f"Improvement over left: {((left_drift[-1] - fused_drift[-1]) / left_drift[-1] * 100):.1f}%")
        print(f"Improvement over right: {((right_drift[-1] - fused_drift[-1]) / right_drift[-1] * 100):.1f}%")


def main():
    # bagpath = Path("/home/jdx/Documents/1.0LatentAct/datasets/WearableData-09-04-25-19-42-36")
    # bagpath = Path("/home/jdx/Documents/1.0LatentAct/datasets/WearableData-09-04-25-21-50-24") # 3 individual rotate
    # bagpath = Path("/home/jdx/Documents/1.0LatentAct/datasets/WearableData-09-24-25-21-16-58") # turn right
    bagpath = Path("/home/jdx/Documents/1.0LatentAct/datasets/WearableData-09-24-25-21-16-04") # back forward

    imu_topic_left = "/left_camera/camera/camera/imu"
    imu_topic_right = "/right_camera/camera/camera/imu"
    out_file_left = Path("left_imu_raw.jsonl")
    out_file_right = Path("right_imu_raw.jsonl")
    start_at_zero = True

    reader = IMUReader(bagpath, imu_topic_left, out_file_left, start_at_zero)
    reader.save_to_file()
    reader = IMUReader(bagpath, imu_topic_right, out_file_right, start_at_zero)
    reader.save_to_file()

    imu_to_gripper_R_right = np.array([[0, 0.939693, -0.34202],
                                        [-1,  0, 0],
                                        [0,  0.34202, 0.939693]]) # left_camera
    imu_to_gripper_T_right = np.array([-0.098085, -0.0175, 0.017882])

    imu_to_gripper_R_left = np.array([[0, 0.939693, 0.34202],
                                        [-1,  0, 0],
                                        [0,  -0.34202, 0.939693]]) # right_camera
    imu_to_gripper_T_left = np.array([0.098085, -0.0175, 0.017882])

    # Process individual IMUs
    print("=" * 50)
    print("Processing Left IMU...")
    print("=" * 50)
    left_processor = ProcessRaw(out_file_left, imu_to_gripper_R_left, imu_to_gripper_T_left)
    left_results = left_processor.process(start_timestep=0)

    print("\n" + "=" * 50)
    print("Processing Right IMU...")
    print("=" * 50)
    right_processor = ProcessRaw(out_file_right, imu_to_gripper_R_right, imu_to_gripper_T_right)
    right_results = right_processor.process(start_timestep=0)

    # Fuse dual IMU data
    print("\n" + "=" * 50)
    print("Fusing Dual IMU Data...")
    print("=" * 50)
    
    dual_processor = DualIMUProcessor(left_processor, right_processor)
    fused_results = dual_processor.fuse_imu_data(left_results, right_results, fusion_weight=0.5)
    
    # Plot comparison
    print("\n" + "=" * 50)
    print("Plotting Results...")
    print("=" * 50)
    dual_processor.plot_dual_imu_comparison(fused_results)
    
    return fused_results

    
    

if __name__ == "__main__":
    main()