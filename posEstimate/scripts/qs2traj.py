#!/usr/bin/env python3
"""
Convert solved_qs.csv to test.trj format with interpolation from ~1 Hz to ~1000 Hz
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from pathlib import Path
import matplotlib.pyplot as plt

# Easier naming (edit one value)
DATA_NAME = "p6-a2-g"

def csv_to_trj(input_csv, output_trj, original_hz=200, target_hz=1000, cutoff_hz=50, filter_order=4):
    """
    Convert CSV trajectory data to .trj format with interpolation and low-pass filtering
    
    Args:
        input_csv: Path to input CSV file
        output_trj: Path to output .trj file
        original_hz: Original sampling frequency in Hz (default: 200)
        target_hz: Target sampling frequency in Hz (default: 1000)
        cutoff_hz: Cutoff frequency for low-pass filter in Hz (default: 50)
        filter_order: Order of the Butterworth filter (default: 4)
    """
    # Read the CSV file
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Extract joint data and timestamps
    joint_cols = [col for col in df.columns if col.startswith('joint')]
    joint_data = df[joint_cols].values

    if 't' in df.columns:
        timestamps = df['t'].values - df['t'].values[0]   # zero-base
        original_duration = float(timestamps[-1])
        original_samples = len(joint_data)
        print(f"Original data: {original_samples} samples over {original_duration:.3f} seconds (from 't' column)")
    else:
        original_samples = len(joint_data)
        original_duration = (original_samples - 1) / original_hz
        timestamps = np.linspace(0, original_duration, original_samples)
        print(f"Original data: {original_samples} samples over {original_duration:.3f} seconds")
        print(f"Using specified original frequency: {original_hz} Hz")
    
    # Create interpolation functions for each joint
    print(f"Interpolating to {target_hz} Hz...")
    interpolators = []
    for joint_idx in range(joint_data.shape[1]):
        # Use cubic spline interpolation for smooth trajectories
        interpolator = interp1d(timestamps, joint_data[:, joint_idx], 
                               kind='cubic', bounds_error=False, 
                               fill_value='extrapolate')
        interpolators.append(interpolator)
    
    # Generate new timestamps at target frequency
    num_samples = int(original_duration * target_hz) + 1
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], num_samples)
    
    # Interpolate all joints
    interpolated_data = np.zeros((num_samples, joint_data.shape[1]))
    for joint_idx, interpolator in enumerate(interpolators):
        interpolated_data[:, joint_idx] = interpolator(new_timestamps)
    
    # Apply low-pass filter to smooth the trajectory
    print(f"Applying low-pass filter (cutoff: {cutoff_hz} Hz, order: {filter_order})...")
    nyquist_freq = target_hz / 2
    normalized_cutoff = cutoff_hz / nyquist_freq

    # Design Butterworth low-pass filter
    b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
    
    # Apply filter to each joint using filtfilt (zero-phase filtering)
    filtered_data = np.zeros_like(interpolated_data)
    for joint_idx in range(interpolated_data.shape[1]):
        filtered_data[:, joint_idx] = filtfilt(b, a, interpolated_data[:, joint_idx])
    
    # Use filtered data for output
    interpolated_data = filtered_data
    
    # Calculate actual frequency
    actual_duration = new_timestamps[-1] - new_timestamps[0]
    actual_hz = (num_samples - 1) / actual_duration if actual_duration > 0 else 0
    
    print(f"Interpolated data: {num_samples} samples")
    print(f"Actual frequency: {actual_hz:.6f} Hz")
    
    # Write to .trj format
    print(f"Writing to {output_trj}...")
    with open(output_trj, 'w') as f:
        # Write header
        f.write(f"{num_samples} [size]\n")
        f.write(f"{actual_duration:.6f} [sec]\n")
        f.write(f"{actual_hz:.6f} [Hz]\n")
        f.write("1 [type]\n")

        # Write data rows
        for row in interpolated_data:
            # Format: 7 joint values separated by spaces, ending with space+comma
            line = ' '.join([f"{val:.6f}" for val in row]) + ' ,'
            f.write(line + '\n')
    
    print(f"Conversion complete!")
    print(f"Output saved to: {output_trj}")
    return {
        "timestamps": new_timestamps,
        "joint_data": interpolated_data,
        "output_trj": Path(output_trj),
    }


def plot_joint_traj(timestamps, joint_data, title="Final Trajectory (.trj source data)"):
    """Plot final interpolated+filtered joint trajectory used to write the .trj file."""
    if len(timestamps) == 0 or joint_data.size == 0:
        print("No trajectory data to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timestamps, joint_data, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint position (rad)")
    ax.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).resolve().parents[1] / "data" / DATA_NAME
    input_csv = base_dir / f"{DATA_NAME}_solved_qs.csv"
    output_trj = base_dir / f"{DATA_NAME}.trj"
    
    # Convert with specified frequencies (adjust these values as needed)
    # cutoff_hz: Lower values = more smoothing, higher values = less smoothing
    result = csv_to_trj(
        input_csv,
        output_trj,
        original_hz=100,
        target_hz=200,
        cutoff_hz=1,
        filter_order=6,
    )
    plot_joint_traj(result["timestamps"], result["joint_data"], title=f"{DATA_NAME}.trj trajectory")
