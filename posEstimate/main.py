# -*- coding: utf-8 -*-
"""
1. read rosbag data
2. calibration
3. to wearable frame
4. pose estimation
"""

from pathlib import Path
import json
import numpy as np

from rosbag_reader import RosbagReader
from imu.use_calibration import UseCalibration
from imu.object_frame import ObjectFrame
from util import ConfigurablePlotter
from imu.find_orientation import MadgwickOrientation, PyPoseOrientation
from imu.find_position import FindPosition

def main():
    bagpath = Path("/home/jdx/Documents/1.0LatentAct/datasets/WearableData-09-24-25-21-16-04")
    out_file = Path("posEstimate/data/multi_imu_raw.jsonl")
    
    # 1. read rosbag data
    reader = RosbagReader(bagpath, out_file)
    sensor_data = reader.process_data(save_data=False)
    timestamps = sensor_data['timestamps']
    dt = timestamps[1] - timestamps[0]
    print(f"dt: {dt}")
    
    # 2. calibration
    left_imu = sensor_data['imu_left']
    acc_calibrated, gyro_calibrated = UseCalibration("imu_left").process_data(left_imu,window_size=1)
    
    # 3. to wearable frame
    acc_object, gyro_object = ObjectFrame("imu_left").process_data(acc_calibrated, gyro_calibrated)
    
    # plot
    plotter = ConfigurablePlotter(rows=4, cols=1, figsize=(12, 16))
    plotter.add_subplot("acc_all", 0, 0, "acc_calibrated", "Time (s)", "m/s²")
    plotter.add_subplot("gyro_all", 1, 0, "gyro_calibrated", "Time (s)", "rad/s")
    plotter.add_subplot("acc_object", 2, 0, "acc_object", "Time (s)", "m/s²")
    plotter.add_subplot("gyro_object", 3, 0, "gyro_object", "Time (s)", "rad/s")
    
    plotter.plot_data("acc_all", timestamps, acc_calibrated[:, 0], "Acc X", "red")
    plotter.plot_data("acc_all", timestamps, acc_calibrated[:, 1], "Acc Y", "green")
    plotter.plot_data("acc_all", timestamps, acc_calibrated[:, 2], "Acc Z", "blue")
    plotter.plot_data("gyro_all", timestamps, gyro_calibrated[:, 0], "Gyro X", "red")
    plotter.plot_data("gyro_all", timestamps, gyro_calibrated[:, 1], "Gyro Y", "green")
    plotter.plot_data("gyro_all", timestamps, gyro_calibrated[:, 2], "Gyro Z", "blue")
    plotter.plot_data("acc_object", timestamps, acc_object[:, 0], "Acc X", "red")
    plotter.plot_data("acc_object", timestamps, acc_object[:, 1], "Acc Y", "green")
    plotter.plot_data("acc_object", timestamps, acc_object[:, 2], "Acc Z", "blue")
    plotter.plot_data("gyro_object", timestamps, gyro_object[:, 0], "Gyro X", "red")
    plotter.plot_data("gyro_object", timestamps, gyro_object[:, 1], "Gyro Y", "green")
    plotter.plot_data("gyro_object", timestamps, gyro_object[:, 2], "Gyro Z", "blue")
    
    plotter.show()
    
    # 3.5 merge all imu data
    # 4. pose estimation
    # 4.1 Orientation
    

    orientation_R = MadgwickOrientation(acc_object, gyro_object, dt, data_format="rotation_matrix").find_orientation()
    orientation_R2 = PyPoseOrientation(acc_object, gyro_object, dt, data_format="rotation_matrix").find_orientation()
    
    # Validation: R * initial_acc should = [0, 0, 9.8] (gravity in world frame)
    print("\n=== ORIENTATION VALIDATION ===")
    initial_acc = acc_object[10]  # First accelerometer reading
    print(f"Initial accelerometer reading: {initial_acc}")
    
    # Test Madgwick
    R_madgwick_0 = orientation_R[1]  # First rotation matrix
    gravity_world_madgwick = R_madgwick_0 @ initial_acc
    print(f"Madgwick R[0] * acc[0] = {gravity_world_madgwick}")
    print(f"Expected: [0, 0, 9.8], Error: {np.linalg.norm(gravity_world_madgwick - np.array([0, 0, 9.8])):.4f}")
    
    # Test PyPose  
    R_pypose_0 = orientation_R2[1]  # First rotation matrix
    gravity_world_pypose = R_pypose_0 @ initial_acc
    print(f"PyPose R[0] * acc[0] = {gravity_world_pypose}")
    print(f"Expected: [0, 0, 9.8], Error: {np.linalg.norm(gravity_world_pypose - np.array([0, 0, 9.8])):.4f}")

    # 4.2 Position
    print("\n=== POSITION ESTIMATION ===")
    
    # With bias correction
    print("Computing position with bias correction...")
    position, velocity, acc_world = FindPosition(acc_object, orientation_R, dt).find_position()
    position2, velocity2, acc_world2 = FindPosition(acc_object, orientation_R2, dt).find_position()
    
    # Without bias correction for comparison
    print("\nComputing position without bias correction...")
    position_no_bias, velocity_no_bias, acc_world_no_bias = FindPosition(acc_object, orientation_R, dt, enable_bias_correction=False).find_position()
    
    # Compare final positions
    print(f"\n=== DRIFT COMPARISON ===")
    print(f"With bias correction - Final position: {position[-1]}")
    print(f"Without bias correction - Final position: {position_no_bias[-1]}")
    print(f"Drift reduction: {np.linalg.norm(position_no_bias[-1]) - np.linalg.norm(position[-1]):.3f} m")
    
    plotter = ConfigurablePlotter(rows=6, cols=1, figsize=(12, 16))
    plotter.add_subplot("acceleration", 0, 0, "acceleration", "Time (s)", "m/s²")
    plotter.add_subplot("velocity", 1, 0, "velocity", "Time (s)", "m/s")
    plotter.add_subplot("position", 2, 0, "position", "Time (s)", "m")
    plotter.add_subplot("acceleration2", 3, 0, "acceleration2", "Time (s)", "m/s²")
    plotter.add_subplot("velocity2", 4, 0, "velocity2", "Time (s)", "m/s")
    plotter.add_subplot("position2", 5, 0, "position2", "Time (s)", "m")
    
    plotter.plot_data("acceleration", timestamps, acc_world[:, 0], "Acceleration X", "red")
    plotter.plot_data("acceleration", timestamps, acc_world[:, 1], "Acceleration Y", "green")
    plotter.plot_data("acceleration", timestamps, acc_world[:, 2], "Acceleration Z", "blue")
    plotter.plot_data("velocity", timestamps, velocity[:, 0], "Velocity X", "red")
    plotter.plot_data("velocity", timestamps, velocity[:, 1], "Velocity Y", "green")
    plotter.plot_data("velocity", timestamps, velocity[:, 2], "Velocity Z", "blue")
    plotter.plot_data("position", timestamps, position[:, 0], "Position X", "red")
    plotter.plot_data("position", timestamps, position[:, 1], "Position Y", "green")
    plotter.plot_data("position", timestamps, position[:, 2], "Position Z", "blue")
    plotter.plot_data("acceleration2", timestamps, acc_world2[:, 0], "Acceleration X", "red")
    plotter.plot_data("acceleration2", timestamps, acc_world2[:, 1], "Acceleration Y", "green")
    plotter.plot_data("acceleration2", timestamps, acc_world2[:, 2], "Acceleration Z", "blue")
    plotter.plot_data("velocity2", timestamps, velocity2[:, 0], "Velocity X", "red")
    plotter.plot_data("velocity2", timestamps, velocity2[:, 1], "Velocity Y", "green")
    plotter.plot_data("velocity2", timestamps, velocity2[:, 2], "Velocity Z", "blue")
    plotter.plot_data("position2", timestamps, position2[:, 0], "Position X", "red")
    plotter.plot_data("position2", timestamps, position2[:, 1], "Position Y", "green")
    plotter.plot_data("position2", timestamps, position2[:, 2], "Position Z", "blue")
    plotter.show()
    
if __name__ == "__main__":
    main()