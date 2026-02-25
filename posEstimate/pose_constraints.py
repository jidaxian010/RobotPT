"""
pose_constraints.py — Relative anatomical constraints for MediaPipe pose landmarks.

All constraints operate on a (33, 3) positions array where:
    positions[i] = [x, y, z]

Coordinate convention (image / camera frame, Y increases downward):
    "lower" body part  →  larger Y value
    "higher" body part →  smaller Y value

Usage
-----
    from pose_constraints import check_all, correct_vertical_order

    violations = check_all(positions)
    if violations:
        positions = correct_vertical_order(positions)
"""

import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe landmark index map
# ---------------------------------------------------------------------------

LM = {
    "nose":             0,
    "left_eye_inner":   1,  "left_eye":        2,  "left_eye_outer":  3,
    "right_eye_inner":  4,  "right_eye":       5,  "right_eye_outer": 6,
    "left_ear":         7,  "right_ear":       8,
    "mouth_left":       9,  "mouth_right":    10,
    "left_shoulder":   11,  "right_shoulder": 12,
    "left_elbow":      13,  "right_elbow":    14,
    "left_wrist":      15,  "right_wrist":    16,
    "left_pinky":      17,  "right_pinky":    18,
    "left_index":      19,  "right_index":    20,
    "left_thumb":      21,  "right_thumb":    22,
    "left_hip":        23,  "right_hip":      24,
    "left_knee":       25,  "right_knee":     26,
    "left_ankle":      27,  "right_ankle":    28,
    "left_heel":       29,  "right_heel":     30,
    "left_foot_index": 31,  "right_foot_index": 32,
}

# ---------------------------------------------------------------------------
# Individual constraint functions
# Each returns (satisfied: bool, description: str)
# ---------------------------------------------------------------------------

def shoulder_above_hip(pos):
    """Shoulders should be higher (smaller Y) than hips."""
    ls, rs = pos[LM["left_shoulder"]], pos[LM["right_shoulder"]]
    lh, rh = pos[LM["left_hip"]],      pos[LM["right_hip"]]
    ok = (ls[1] < lh[1]) and (rs[1] < rh[1])
    return ok, "shoulder_above_hip"


def hip_above_knee(pos):
    """Hips should be higher than knees."""
    lh, rh = pos[LM["left_hip"]],   pos[LM["right_hip"]]
    lk, rk = pos[LM["left_knee"]],  pos[LM["right_knee"]]
    ok = (lh[1] < lk[1]) and (rh[1] < rk[1])
    return ok, "hip_above_knee"


def knee_above_ankle(pos):
    """Knees should be higher than ankles."""
    lk, rk = pos[LM["left_knee"]],  pos[LM["right_knee"]]
    la, ra = pos[LM["left_ankle"]], pos[LM["right_ankle"]]
    ok = (lk[1] < la[1]) and (rk[1] < ra[1])
    return ok, "knee_above_ankle"


def ankle_above_foot(pos):
    """Ankles should be higher than foot index landmarks."""
    la, ra = pos[LM["left_ankle"]],      pos[LM["right_ankle"]]
    lf, rf = pos[LM["left_foot_index"]], pos[LM["right_foot_index"]]
    ok = (la[1] <= lf[1]) and (ra[1] <= rf[1])
    return ok, "ankle_above_foot"


def nose_above_shoulders(pos):
    """Nose should be higher than shoulders (head is on top)."""
    n  = pos[LM["nose"]]
    ls = pos[LM["left_shoulder"]]
    rs = pos[LM["right_shoulder"]]
    ok = n[1] < min(ls[1], rs[1])
    return ok, "nose_above_shoulders"


def hips_level(pos, tilt_threshold_px=60):
    """
    Left and right hips should be roughly at the same height.
    Large vertical difference suggests a detection error.
    """
    lh = pos[LM["left_hip"]]
    rh = pos[LM["right_hip"]]
    ok = abs(lh[1] - rh[1]) < tilt_threshold_px
    return ok, "hips_level"


def shoulders_level(pos, tilt_threshold_px=60):
    """Left and right shoulders should be roughly at the same height."""
    ls = pos[LM["left_shoulder"]]
    rs = pos[LM["right_shoulder"]]
    ok = abs(ls[1] - rs[1]) < tilt_threshold_px
    return ok, "shoulders_level"


# ---------------------------------------------------------------------------
# Run all constraints
# ---------------------------------------------------------------------------

_ALL_CONSTRAINTS = [
    nose_above_shoulders,
    shoulder_above_hip,
    hip_above_knee,
    knee_above_ankle,
    ankle_above_foot,
    hips_level,
    shoulders_level,
]


def check_all(positions: np.ndarray) -> list[str]:
    """
    Run all constraints on a (33, 3) positions array.

    Returns a list of violated constraint names (empty = all satisfied).
    """
    violations = []
    for fn in _ALL_CONSTRAINTS:
        ok, name = fn(positions)
        if not ok:
            violations.append(name)
    return violations


# ---------------------------------------------------------------------------
# Auto-correction: vertical ordering
# ---------------------------------------------------------------------------

# Ordered chains: each tuple is (upper_joint, lower_joint, margin_px)
_VERTICAL_CHAINS = [
    ("nose",          "left_shoulder",   5),
    ("nose",          "right_shoulder",  5),
    ("left_shoulder", "left_hip",       10),
    ("right_shoulder","right_hip",      10),
    ("left_hip",      "left_knee",      10),
    ("right_hip",     "right_knee",     10),
    ("left_knee",     "left_ankle",     10),
    ("right_knee",    "right_ankle",    10),
    ("left_ankle",    "left_foot_index", 5),
    ("right_ankle",   "right_foot_index",5),
]


def correct_vertical_order(positions: np.ndarray,
                            inplace: bool = False) -> np.ndarray:
    """
    Snap joints back into anatomically correct vertical order.

    For each (upper, lower) pair: if upper.y >= lower.y, set
        upper.y = lower.y - margin
    so the upper joint is forced above the lower one.

    Parameters
    ----------
    positions : (33, 3) array  — will be copied unless inplace=True
    inplace   : bool           — modify in place (faster, no copy)

    Returns corrected (33, 3) array.
    """
    pos = positions if inplace else positions.copy()
    for upper_name, lower_name, margin in _VERTICAL_CHAINS:
        u = LM[upper_name]
        l = LM[lower_name]
        if pos[u, 1] >= pos[l, 1]:
            pos[u, 1] = pos[l, 1] - margin
    return pos
