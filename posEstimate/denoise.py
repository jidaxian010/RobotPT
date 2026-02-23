"""
denoise.py — Aggressive smoothing of raw pose-list trajectories.

Works directly on the List[dict | None] format used throughout GripperOdometry:
  denoise_pose_list        — single marker or fused: List[dict | None]
  denoise_all_marker_poses — all markers:            Dict[int, List[dict | None]]

Smoothing pipeline per channel:
  1. Median filter  — removes impulse spikes
  2. Gaussian filter — smooth remaining signal

Both parameters are intentionally large for aggressive smoothing.
Gaps (None frames) are handled: only valid frames are smoothed; None
entries are preserved as-is.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from scipy.spatial.transform import Rotation


# ------------------------------------------------------------------
# Default tuning  (override per call if needed)
# ------------------------------------------------------------------
MED_KERNEL = 31   # must be odd;  larger → more spike rejection
SIGMA      = 15   # Gaussian σ in frames;  larger → smoother


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _smooth_1d(arr, med_kernel, sigma):
    med = medfilt(arr.astype(float), kernel_size=med_kernel)
    return gaussian_filter1d(med, sigma=sigma)


def _smooth_pos(pos, med_kernel, sigma):
    """(N, 3) position → smoothed (N, 3)."""
    return np.stack([_smooth_1d(pos[:, i], med_kernel, sigma)
                     for i in range(3)], axis=1)


def _smooth_quats(quats, med_kernel, sigma):
    """
    (N, 4) quaternion array → smoothed, re-normalised (N, 4).
    Enforces sign continuity before filtering to avoid flips.
    """
    q = quats.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    smoothed = np.stack([_smooth_1d(q[:, j], med_kernel, sigma)
                         for j in range(4)], axis=1)
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    return smoothed / np.maximum(norms, 1e-8)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def denoise_pose_list(poses, med_kernel=MED_KERNEL, sigma=SIGMA):
    """
    Smooth a List[dict | None] pose sequence.

    Operates on the valid frames only; None entries are unchanged.

    Args:
        poses:      List[dict | None] — from compute_marker_poses_raw()
                    or compute_gripper_poses_fused() or compute_gripper_poses().
        med_kernel: int (must be odd)
        sigma:      float

    Returns:
        New list with smoothed "position" (np.ndarray (3,)) and
        "rotation" (np.ndarray (3, 3)) in each valid entry.
    """
    valid_idx = [i for i, p in enumerate(poses) if p is not None]
    if len(valid_idx) < 2:
        return list(poses)

    pos   = np.array([poses[i]["position"] for i in valid_idx])        # (N, 3)
    quats = np.array([
        Rotation.from_matrix(poses[i]["rotation"]).as_quat()
        for i in valid_idx
    ])                                                                   # (N, 4)

    pos_s   = _smooth_pos(pos,   med_kernel, sigma)
    quats_s = _smooth_quats(quats, med_kernel, sigma)

    result = list(poses)
    for k, i in enumerate(valid_idx):
        p = dict(poses[i])
        p["position"] = pos_s[k]
        p["rotation"] = Rotation.from_quat(quats_s[k]).as_matrix()
        result[i] = p
    return result


def denoise_all_marker_poses(all_raw_poses, med_kernel=MED_KERNEL, sigma=SIGMA):
    """
    Smooth every marker's pose list.

    Args:
        all_raw_poses: Dict[int, List[dict | None]]

    Returns:
        New dict with the same keys, each value smoothed.
    """
    return {mid: denoise_pose_list(poses, med_kernel, sigma)
            for mid, poses in all_raw_poses.items()}
