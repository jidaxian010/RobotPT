import numpy as np
import os
import json
from scipy.optimize import minimize
from pathlib import Path
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from rosbag_reader import RosbagReader

"""
A separate script refer to only Static Orientation Calibration Data
"""


class Calibration:
    """
    Calibrate the IMU data
    Return acc_bias, gyro_bias, acc_cov, gyro_cov, save to jsonl
    """

    def __init__(self, topic_name, data_path):
        self.topic_name = topic_name
        self.data_path = data_path

    def load_calibration_data(self, bag_name):
        """
        At each pos, return avg(acc), avg(gyro)
        """
        # Construct full path to the bag directory
        bag_path = self.data_path / bag_name
        reader = RosbagReader(bag_path, "not_saving.jsonl")
        sensor_data = reader.process_data(save_data=False)
        acc = sensor_data[self.topic_name][:, :3]
        gyro = sensor_data[self.topic_name][:, 3:]
        
        acc_mean = np.mean(acc, axis=0)
        gyro_mean = np.mean(gyro, axis=0)
        assert acc_mean.shape == (3,)
        assert gyro_mean.shape == (3,)

        return acc_mean, gyro_mean
    
    def find_bias(self, gravity_magnitude=9.81, validate=True):
        """
        Calculate the b_acc: b_acc = argmin SUM_all_pos(||acc_mean - b_acc||-g)^2
        gyro_bias: b_gyro = MEAN_all_pos(gyro_mean)
        """
        
        # for all the bag in data_path, calculate acc_mean
        acc_means = []
        gyro_means = []
        bag_files = []
        print(f"Loading calibration data from {self.data_path}")
        #check ls in data_path
        print(f"ls in {self.data_path}: {os.listdir(self.data_path)}")
        for bag_name in os.listdir(self.data_path):
            # Skip if not a directory
            bag_path = self.data_path / bag_name
            if not bag_path.is_dir():
                continue
                
            try:
                acc_mean, gyro_mean = self.load_calibration_data(bag_name)
                acc_means.append(acc_mean)
                gyro_means.append(gyro_mean)
                bag_files.append(bag_name)
                print(f"acc_mean for {bag_name}: {acc_mean}")
            except Exception as e:
                print(f"Error loading calibration data from {bag_name}: {e}")
                continue
        
        print(f"Found {len(acc_means)} valid measurements for calibration")
        acc_means = np.array(acc_means)  # Shape: [N, 3]

        def objective_function(acc_bias):
            """
            Objective function: SUM(||acc_mean - acc_bias|| - g)^2
            """
            total_error = 0.0
            for acc_mean in acc_means:
                # Subtract bias from measurement
                corrected_acc = acc_mean - acc_bias
                # Calculate magnitude
                magnitude = np.linalg.norm(corrected_acc)
                # Error from expected gravity magnitude
                error = (magnitude - gravity_magnitude) ** 2
                total_error += error
            return total_error
        
        # Initial guess for bias
        initial_bias = np.mean(acc_means, axis=0)
                
        # Minimize the objective function
        result = minimize(objective_function, initial_bias, method='BFGS')
        
        if result.success:
            optimal_acc_bias = result.x
            print(f"Optimal bias: {optimal_acc_bias}")
            print(f"Final error: {result.fun:.6f}")
            
            if validate:
                self._validate_bias(acc_means, optimal_acc_bias, gravity_magnitude)
            
        else:
            print(f"Accelerometer bias calibration failed: {result.message}")
            optimal_acc_bias = initial_bias

        # Calculate gyro bias: mean of all gyro readings (should be 0 when stationary)
        gyro_means = np.array(gyro_means)  # Shape: [N, 3]
        optimal_gyro_bias = np.mean(gyro_means, axis=0)
        
        print(f"\nGyroscope bias calibration:")
        print(f"✓ Optimal gyro bias: {optimal_gyro_bias}")
        
        if validate:
            self._validate_gyro_bias(gyro_means, optimal_gyro_bias)
        
        return optimal_acc_bias, optimal_gyro_bias
    
    def find_covariance(self, acc_bias, gyro_bias, gravity_magnitude=9.81):
        """
        Calculate covariance matrices for accelerometer and gyroscope.
        
        Args:
            acc_bias: Accelerometer bias vector [3,]
            gyro_bias: Gyroscope bias vector [3,]
            gravity_magnitude: Expected gravity magnitude
            
        Returns:
            acc_cov: Accelerometer covariance matrix [3,3]
            gyro_cov: Gyroscope covariance matrix [3,3]
        """
        
        # Collect residuals from all bags
        all_acc_residuals = []
        all_gyro_residuals = []
        
        for bag_name in os.listdir(self.data_path):
            bag_path = self.data_path / bag_name
            if not bag_path.is_dir():
                continue
                
            try:
                # Load raw data for this bag
                reader = RosbagReader(bag_path, "not_saving.jsonl")
                sensor_data = reader.process_data(save_data=False)
                
                # Get all raw measurements (not just means)
                raw_acc = sensor_data[self.topic_name][:, :3]  # Shape: [N, 3]
                raw_gyro = sensor_data[self.topic_name][:, 3:]  # Shape: [N, 3]
                
                # Apply bias correction
                corrected_acc = raw_acc - acc_bias  # Shape: [N, 3]
                corrected_gyro = raw_gyro - gyro_bias  # Shape: [N, 3]
                
                # For accelerometer: remove per-pose mean to avoid gravity inflation
                acc_pose_mean = np.mean(corrected_acc, axis=0)  # [3,] - gravity vector for this pose
                acc_residuals = corrected_acc - acc_pose_mean  # Remove pose-specific gravity
                
                # For gyroscope: already centered around zero, so residuals = corrected values
                gyro_residuals = corrected_gyro - np.mean(corrected_gyro, axis=0)  # Remove any remaining pose-specific bias
                
                all_acc_residuals.append(acc_residuals)
                all_gyro_residuals.append(gyro_residuals)
                
                print(f"Loaded {len(raw_acc)} samples from {bag_name}, pose gravity: {acc_pose_mean}")
                
            except Exception as e:
                print(f"Error loading covariance data from {bag_name}: {e}")
                continue
        
        if not all_acc_residuals:
            raise RuntimeError("No valid data found for covariance calculation")
        
        # Stack all residuals from all poses
        all_acc_residuals = np.vstack(all_acc_residuals)  # Shape: [Total_N, 3]
        all_gyro_residuals = np.vstack(all_gyro_residuals)  # Shape: [Total_N, 3]
        
        print(f"Total samples for covariance: {len(all_acc_residuals)}")
        
        # Calculate covariance matrices from residuals
        acc_cov = np.cov(all_acc_residuals.T)  # [3, 3] - noise covariance only
        gyro_cov = np.cov(all_gyro_residuals.T)  # [3, 3] - noise covariance only
        
        
        return acc_cov, gyro_cov
    
    def _validate_covariance(self, acc_cov, gyro_cov):
        """
        Display covariance matrix information for validation
        """
        print(f"\nCovariance Matrix Validation:")
        print(f"=" * 70)
        
        print(f"\nAccelerometer Noise Covariance Matrix (gravity removed per-pose):")
        print(f"Shape: {acc_cov.shape}")
        print(f"Units: (m/s²)²")
        print(acc_cov)
        
        # Extract diagonal (variances) and compute standard deviations
        acc_std = np.sqrt(np.diag(acc_cov))
        print(f"Noise standard deviations [X, Y, Z]: {acc_std} m/s²")
        print(f"Condition number: {np.linalg.cond(acc_cov):.2f}")
        
        print(f"\nGyroscope Noise Covariance Matrix:")
        print(f"Shape: {gyro_cov.shape}")
        print(f"Units: (rad/s)²")
        print(gyro_cov)
        
        gyro_std = np.sqrt(np.diag(gyro_cov))
        print(f"Noise standard deviations [X, Y, Z]: {gyro_std} rad/s")
        print(f"Condition number: {np.linalg.cond(gyro_cov):.2f}")
        print(f"=" * 70)
    
    def _validate_bias(self, acc_measurements, bias, gravity_magnitude):
        """
        Validate the calibration by showing corrected magnitudes
        """
        print(f"\nValidation Results:")
        print(f"Target gravity magnitude: {gravity_magnitude:.3f} m/s²")
        print(f"Orientation | Original Mag | Corrected Mag | Error")
        print(f"-" * 55)
        
        errors = []
        for i, acc_mean in enumerate(acc_measurements):
            original_mag = np.linalg.norm(acc_mean)
            corrected_acc = acc_mean - bias
            corrected_mag = np.linalg.norm(corrected_acc)
            error = corrected_mag - gravity_magnitude
            errors.append(error)
            
            print(f"Pos {i+1:2d}     | {original_mag:10.3f}   | {corrected_mag:11.3f}   | {error:+6.3f}")
        
        errors = np.array(errors)
        print(f"-" * 55)
        print(f"Mean error: {np.mean(errors):+6.3f} m/s²")
        print(f"Std error:  {np.std(errors):6.3f} m/s²")
        print(f"Max error:  {np.max(np.abs(errors)):6.3f} m/s²")

    def _validate_gyro_bias(self, gyro_measurements, bias):
        """
        Validate the gyro calibration by showing corrected readings (should be close to 0)
        """
        print(f"\nGyroscope Validation Results:")
        print(f"Target: [0.0, 0.0, 0.0] rad/s (stationary)")
        print(f"Orientation | Original X | Original Y | Original Z | Corrected X | Corrected Y | Corrected Z")
        print(f"-" * 85)
        
        corrected_readings = []
        for i, gyro_mean in enumerate(gyro_measurements):
            corrected_gyro = gyro_mean - bias
            corrected_readings.append(corrected_gyro)
            
            print(f"Pos {i+1:2d}     | {gyro_mean[0]:10.4f} | {gyro_mean[1]:10.4f} | {gyro_mean[2]:10.4f} | "
                  f"{corrected_gyro[0]:11.4f} | {corrected_gyro[1]:11.4f} | {corrected_gyro[2]:11.4f}")
        
        corrected_readings = np.array(corrected_readings)
        print(f"-" * 85)
        
        # Calculate statistics for each axis
        for axis, axis_name in enumerate(['X', 'Y', 'Z']):
            axis_data = corrected_readings[:, axis]
            print(f"{axis_name}-axis - Mean: {np.mean(axis_data):+8.4f}, Std: {np.std(axis_data):8.4f}, "
                  f"Max: {np.max(np.abs(axis_data)):8.4f} rad/s")

    def write_data(self, raw_data=None, include_covariance=True):
        """
        Calibrate the raw data using the calibration benchmark
        
        Args:
            raw_data: Unused (for compatibility)
            include_covariance: Whether to calculate covariance matrices
            
        Returns:
            If include_covariance=True: (acc_bias, gyro_bias, acc_cov, gyro_cov)
            If include_covariance=False: (acc_bias, gyro_bias)
        """
        acc_bias, gyro_bias = self.find_bias(validate=False)
        
        if include_covariance:
            acc_cov, gyro_cov = self.find_covariance(acc_bias, gyro_bias)
            return acc_bias, gyro_bias, acc_cov, gyro_cov
        else:
            return acc_bias, gyro_bias

def save_to_json(all_calibrations, output_file="posEstimate/data/imu_calibration.json"):
    """
    Save all IMU calibration data to a single JSON file.
    
    Args:
        all_calibrations: Dict with IMU topics as keys and calibration data as values
        output_file: Output JSON file path
    """
    calibration_data = {}
    
    for topic_name, cal_data in all_calibrations.items():
        acc_bias, gyro_bias, acc_cov, gyro_cov = cal_data
        calibration_data[topic_name] = {
            "acc_bias": acc_bias.tolist(),
            "gyro_bias": gyro_bias.tolist(),
            "acc_cov": acc_cov.tolist(),
            "gyro_cov": gyro_cov.tolist()
        }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"Saved all IMU calibrations to {output_file}")

def calibrate_data(data_path, include_covariance=True):
    """
    Calibrate all IMU topics and save results to a single JSON file.
    
    Args:
        data_path: Path to calibration data directory containing rosbag poses
        include_covariance: Whether to calculate covariance matrices
    """
    print("=== MULTI-IMU CALIBRATION ===")
    
    imu_topics = ["imu_left", "imu_right", "imu_vectornav"]
    all_calibrations = {}
    
    for topic_name in imu_topics:
        print(f"\nCalibrating {topic_name}...")
        try:
            calibration = Calibration(topic_name, data_path)
            acc_bias, gyro_bias, acc_cov, gyro_cov = calibration.write_data(include_covariance=include_covariance)
            all_calibrations[topic_name] = (acc_bias, gyro_bias, acc_cov, gyro_cov)
            print(f"Completed {topic_name}")
        except Exception as e:
            print(f"Failed {topic_name}: {e}")
            continue
    
    # Save all calibrations to single file
    save_to_json(all_calibrations)
    
    
    return all_calibrations

def main():
    """Calibrate all IMU topics and save to single JSON file."""
    data_path = Path("/home/jdx/Documents/1.0LatentAct/datasets/Static_Orientation_Cali")
    
    # Calibrate all IMUs
    calibrate_data(data_path, include_covariance=True)

if __name__ == "__main__":
    main()