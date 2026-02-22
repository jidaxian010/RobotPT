"""
Marker-to-gripper geometric configurations.

Each MarkerConfig defines the fixed SE(3) transform from a marker's frame to
the gripper's Centre-of-Mass (CoM) frame.  Replace the placeholder values with
physically measured numbers once the gripper is available for calibration.

Coordinate conventions
----------------------
Gripper body frame (assumed):
    +X  right  (toward the right face — where marker 6 lives)
    +Y  forward / along approach direction
    +Z  up

ArUco marker frame (standard OpenCV):
    +X  right along marker face   (when viewed from the front)
    +Y  up along marker face      (when viewed from the front)
    +Z  pointing OUT of the marker face (toward the camera)

SE(3) math used when computing gripper pose from a detected marker
------------------------------------------------------------------
solvePnP gives T_cam←marker = { R_cm, t_cm } such that
    p_cam = R_cm @ p_marker + t_cm

MarkerConfig gives T_marker←com = { R_mc, t_mc } such that
    p_marker = R_mc @ p_com + t_mc

Therefore:
    R_cam←com = R_cm @ R_mc
    t_cam←com = R_cm @ t_mc + t_cm
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class MarkerConfig:
    marker_id: int
    marker_size_mm: float

    # Translation: CoM origin expressed in the MARKER frame (mm).
    # i.e. "where is the gripper CoM relative to the marker centre?"
    t_marker_to_com: np.ndarray  # shape (3,)

    # Rotation: maps a vector from marker frame to gripper CoM frame.
    # R_marker_to_com @ v_marker = v_gripper
    R_marker_to_com: np.ndarray  # shape (3, 3)


# ---------------------------------------------------------------------------
# Placeholder values — calibrate before use
# ---------------------------------------------------------------------------

_MARKER_SIZE_MM   = 40.0   # physical side length of the printed marker
_HALF_WIDTH_MM    = 50.0   # half the gripper width in X  (placeholder)

# Marker 6 — attached to the EXACT right surface of the gripper.
#
# Geometry of the right-face attachment:
#   • Marker face normal (+Z_marker) points in gripper +X  (outward)
#   • Marker +Y_marker  aligns with gripper +Z             (up)
#   • Marker +X_marker  aligns with gripper −Y             (forward→backward)
#
# The CoM sits ~_HALF_WIDTH_MM inward along the marker's −Z axis.
#
#  R_marker_to_com  columns = marker basis vectors expressed in gripper frame:
#    col-0  marker +X  →  gripper [ 0, −1,  0]
#    col-1  marker +Y  →  gripper [ 0,  0,  1]
#    col-2  marker +Z  →  gripper [ 1,  0,  0]
_R_MARKER6_TO_COM = np.array(
    [[ 0,  0,  1],
     [-1,  0,  0],
     [ 0,  1,  0]],
    dtype=float,
)

MARKER_CONFIGS: dict[int, "MarkerConfig"] = {
    6: MarkerConfig(
        marker_id=6,
        marker_size_mm=_MARKER_SIZE_MM,
        t_marker_to_com=np.array([0.0, 0.0, -_HALF_WIDTH_MM]),
        R_marker_to_com=_R_MARKER6_TO_COM,
    ),
}
