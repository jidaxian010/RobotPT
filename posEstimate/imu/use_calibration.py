import os
import json
import numpy as np
from scipy import signal
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from imu.calibration import calibrate_data

"""
1. Calculate and load calibration data
2. input: take in a 6 dim vector [acc, gyro], output: 6 dim calibratedvector [acc, gyro]
"""

class UseCalibration:
    def __init__(self, topic_name):
        self.topic_name = topic_name
        self.calibration = None

    def load_calibration(self):
        """Load calibration data from file, create if doesn't exist."""
        if self.calibration is None:
            if not os.path.exists("posEstimate/data/imu_calibration.json"):
                print("Calibration file not found, creating new calibration...")
                data_path = Path("/home/jdx/Documents/1.0LatentAct/datasets/Static_Orientation_Cali")
                calibrate_data(data_path, include_covariance=True)
            
            with open("posEstimate/data/imu_calibration.json", "r") as f:
                self.calibration = json.load(f)
                print(f"Loaded calibration for topic: {self.topic_name}")

    def use_calibration(self, imu):
        """Apply calibration to accelerometer and gyroscope data."""
        # Load calibration if not already loaded
        self.load_calibration()
        
        # Get bias values for this IMU topic
        acc_bias = np.array(self.calibration[self.topic_name]["acc_bias"])
        gyro_bias = np.array(self.calibration[self.topic_name]["gyro_bias"])

        acc = imu[:, :3]
        gyro = imu[:, 3:]
        # Apply calibration (subtract bias)
        acc_calibrated = acc - acc_bias
        gyro_calibrated = gyro - gyro_bias
        
        return acc_calibrated, gyro_calibrated

    def remove_spikes(self, data, spike_threshold=3.0, window_size=1):
        """Remove spikes using median filter and z-score, then apply window averaging"""
        # Input validation
        if data.shape[0] < 5:
            print(f"Warning: Data has only {data.shape[0]} samples, skipping spike removal")
            return data
        
        filtered_data = np.copy(data)
        
        for i in range(data.shape[1]):  # For each axis
            # Step 1: Apply median filter to detect spikes
            # Use adaptive kernel size based on data length
            kernel_size = min(5, data.shape[0] if data.shape[0] % 2 == 1 else data.shape[0] - 1)
            if kernel_size < 3:
                kernel_size = 3
            
            median_filtered = signal.medfilt(data[:, i], kernel_size=kernel_size)
            
            # Step 2: Calculate z-score of residuals
            residuals = data[:, i] - median_filtered
            residual_std = np.std(residuals)
            
            # Skip if no variation in data
            if residual_std < 1e-8:
                filtered_data[:, i] = data[:, i]
                continue
                
            z_scores = np.abs(residuals) / residual_std
            
            # Step 3: Replace spikes with median filtered values
            spike_mask = z_scores > spike_threshold
            temp_data = data[:, i].copy()
            temp_data[spike_mask] = median_filtered[spike_mask]
            
            # Step 4: Apply window averaging
            if len(temp_data) >= window_size and window_size > 1:
                # Apply moving average using convolution
                moving_avg = np.convolve(temp_data, np.ones(window_size)/window_size, mode='same')
                filtered_data[:, i] = moving_avg
            else:
                filtered_data[:, i] = temp_data
        
        return filtered_data

    def process_data(self, imu, spike_threshold=3.0, window_size=1):
        """Process data by removing spikes and applying calibration"""
        imu_clean = self.remove_spikes(imu, spike_threshold=spike_threshold, window_size=window_size)
        acc_calibrated, gyro_calibrated = self.use_calibration(imu_clean)
        return acc_calibrated, gyro_calibrated


def main():
    imu_single = np.array([[1, 0, 0, 0, 0, 0]])
    calibrator = UseCalibration("imu_left")
    acc_calibrated, gyro_calibrated = calibrator.process_data(imu_single, spike_threshold=3.0, window_size=1)
    print(f"acc_calibrated: {acc_calibrated}")
    print(f"gyro_calibrated: {gyro_calibrated}")
    
    
if __name__ == "__main__":
    main()