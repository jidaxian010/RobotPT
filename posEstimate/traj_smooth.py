"""
traj_smooth.py — Aggressively smooth a gripper CoM trajectory CSV.

Pipeline per channel
--------------------
  Position   : median filter (remove impulse spikes) → Gaussian (σ=SIGMA)
  Orientation: Euler → quaternion → ensure sign continuity →
               median filter → Gaussian → re-normalise → Euler XYZ

Input : <stem>_com_trajectory.csv
Output: <stem>_com_trajectory_smooth.csv  (same columns, same units)

Also shows a before/after comparison plot.
"""

import csv
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Tuning
# ------------------------------------------------------------------
SIGMA      = 20     # Gaussian σ in frames — increase for more smoothing
MED_KERNEL = 15     # Median filter kernel (must be odd) — removes spikes


# ------------------------------------------------------------------
# Core
# ------------------------------------------------------------------

def _read_csv(path):
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    t      = np.array([float(r["t"])       for r in rows])
    frames = np.array([int(float(r["frame"])) for r in rows])
    pos    = np.array([[float(r["pos_x"]),
                        float(r["pos_y"]),
                        float(r["pos_z"])]   for r in rows])
    euler  = np.array([[float(r["orient_x"]),
                        float(r["orient_y"]),
                        float(r["orient_z"])] for r in rows])
    return t, frames, pos, euler


def _smooth_pos(pos, sigma, med_kernel):
    """Median then Gaussian per axis."""
    med = np.stack([medfilt(pos[:, i], kernel_size=med_kernel)
                    for i in range(3)], axis=1)
    return np.stack([gaussian_filter1d(med[:, i], sigma=sigma)
                     for i in range(3)], axis=1)


def _smooth_orient(euler, sigma, med_kernel):
    """
    Smooth orientations in quaternion space to avoid Euler-angle wrap issues.
    1. Euler → quat
    2. Enforce sign continuity (flip quat if dot < 0 with previous)
    3. Median filter each component
    4. Gaussian filter each component
    5. Re-normalise
    6. Quat → Euler XYZ
    """
    quats = Rotation.from_euler("xyz", euler).as_quat()  # (N, 4) xyzw

    # Continuity: ensure shortest-path between consecutive quats
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    med = np.stack([medfilt(quats[:, i], kernel_size=med_kernel)
                    for i in range(4)], axis=1)
    smooth = np.stack([gaussian_filter1d(med[:, i], sigma=sigma)
                       for i in range(4)], axis=1)

    # Re-normalise
    smooth /= np.linalg.norm(smooth, axis=1, keepdims=True)

    return Rotation.from_quat(smooth).as_euler("xyz")


def smooth_trajectory(csv_path, sigma=SIGMA, med_kernel=MED_KERNEL):
    """
    Read a CoM trajectory CSV and return both raw and smoothed arrays.

    Returns
    -------
    t, frames, pos_raw, euler_raw, pos_smooth, euler_smooth
    """
    t, frames, pos_raw, euler_raw = _read_csv(csv_path)
    pos_smooth   = _smooth_pos(pos_raw, sigma, med_kernel)
    euler_smooth = _smooth_orient(euler_raw, sigma, med_kernel)
    return t, frames, pos_raw, euler_raw, pos_smooth, euler_smooth


def save_smooth_csv(csv_path, t, frames, pos_smooth, euler_smooth):
    """Write smoothed trajectory to <stem>_com_trajectory_smooth.csv."""
    csv_path = Path(csv_path)
    out_path = csv_path.with_name(
        csv_path.stem.replace("_com_trajectory", "") + "_com_trajectory_smooth.csv"
    )
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "frame",
                         "pos_x", "pos_y", "pos_z",
                         "orient_x", "orient_y", "orient_z"])
        for i in range(len(t)):
            writer.writerow([
                round(float(t[i]),             6), int(frames[i]),
                round(float(pos_smooth[i, 0]), 3),
                round(float(pos_smooth[i, 1]), 3),
                round(float(pos_smooth[i, 2]), 3),
                round(float(euler_smooth[i, 0]), 6),
                round(float(euler_smooth[i, 1]), 6),
                round(float(euler_smooth[i, 2]), 6),
            ])
    print(f"Saved smoothed trajectory: {out_path}")
    return out_path


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def plot_comparison(t, pos_raw, euler_raw, pos_smooth, euler_smooth):
    """Two rows: position (mm) and orientation (rad), raw vs smoothed."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    fig.suptitle(f"CoM Trajectory Smoothing  (σ={SIGMA}, median={MED_KERNEL})",
                 fontsize=12)

    pos_labels = ["pos_x (mm)", "pos_y (mm)", "pos_z (mm)"]
    ori_labels = ["orient_x (rad)", "orient_y (rad)", "orient_z (rad)"]

    for i in range(3):
        for row, raw, smooth, labels in [
            (0, pos_raw,   pos_smooth,   pos_labels),
            (1, euler_raw, euler_smooth, ori_labels),
        ]:
            ax = axes[row, i]
            ax.plot(t, raw[:, i],    color="tab:blue",  alpha=0.35,
                    linewidth=1,   label="raw")
            ax.plot(t, smooth[:, i], color="tab:orange", linewidth=1.8,
                    label="smooth")
            ax.set_title(labels[i], fontsize=9)
            ax.set_xlabel("t (s)", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, linewidth=0.4)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    csv_path = "posEstimate/data/gripper2_com_trajectory.csv"

    print(f"Smoothing '{csv_path}'  σ={SIGMA}  median_kernel={MED_KERNEL}")
    t, frames, pos_raw, euler_raw, pos_smooth, euler_smooth = smooth_trajectory(csv_path)
    save_smooth_csv(csv_path, t, frames, pos_smooth, euler_smooth)
    plot_comparison(t, pos_raw, euler_raw, pos_smooth, euler_smooth)


if __name__ == "__main__":
    main()
