"""
Marker-to-gripper geometric configurations.

Each MarkerConfig defines the fixed SE(3) transform from a marker's frame to
the gripper's frame (f0).

Coordinate conventions
----------------------
Gripper body frame f0:
    origin at the gripper CoM / reference point
    axes as defined by the CAD model

ArUco marker frame (standard OpenCV), shared with each face frame f_i:
    +X  right along marker face   (when viewed from the front)
    +Y  up along marker face      (when viewed from the front)
    +Z  pointing OUT of the marker face (toward the camera)

T_i0  (face frame f_i → gripper frame f0)
-----------------------------------------
Stored as module-level 4×4 homogeneous matrices. Derived from CAD.

    p_f0 = T_i0 @ p_fi   (homogeneous)

SE(3) math used by the two pose-recovery paths
-----------------------------------------------
Path A — single-marker solvePnP (backward compat, gripper_odemetry.compute_gripper_poses):
    solvePnP gives T_cam←marker = { R_cm, t_cm }
        p_cam = R_cm @ p_marker + t_cm
    MarkerConfig gives T_marker←f0 = { R_mc, t_mc }
        p_marker = R_mc @ p_f0 + t_mc
    Therefore:
        R_cam←f0 = R_cm @ R_mc
        t_cam←f0 = R_cm @ t_mc + t_cm

Path B — multi-marker solvePnP (gripper_odemetry.compute_gripper_poses_fused):
    3D corners of each marker are pre-computed in f0 using T_i0.
    A single solvePnP call over all visible corners yields T_cam←f0 directly.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class MarkerConfig:
    marker_id: int
    marker_size_mm: float

    # --- Path A fields (single-marker solvePnP) ---

    # Translation: f0 origin expressed in the MARKER frame (mm).
    # t_marker_to_com = -R_i0.T @ t_i0
    t_marker_to_com: np.ndarray  # shape (3,)

    # Rotation: maps a vector from marker frame to gripper f0 frame.
    # R_marker_to_com @ v_marker = v_f0
    # R_marker_to_com = T_i0[:3, :3]
    R_marker_to_com: np.ndarray  # shape (3, 3)

    # --- Path B field (multi-marker solvePnP) ---

    # Position of the marker origin in the gripper frame f0 (mm).
    # t_gripper_origin_mm = T_i0[:3, 3]
    # Used to compute 3D corner positions in f0 for the pooled solvePnP.
    t_gripper_origin_mm: np.ndarray  # shape (3,)


# ---------------------------------------------------------------------------
# Face-to-gripper transforms  (CAD-derived, exact)
# ---------------------------------------------------------------------------
#
# Naming: T<i>0  =  T_{face_i → gripper_f0}
#   marker 7  on face 1  →  T10
#   marker 10 on face 2  →  T20
#   marker 11 on face 3  →  T30
#   marker 8  on face 4  →  T40
#   marker 9  on face 5  →  T50

_T10 = np.array([
    [0.0, 0.0, -1.0, -214.2],
    [0.0, 1.0,  0.0,  -22.5],
    [1.0, 0.0,  0.0, -392.7],
    [0.0, 0.0,  0.0,    1.0],
], dtype=float)

_T20 = np.array([
    [ 0.7071, 0.0, -0.7071, -187.3],
    [ 0.0,    1.0,  0.0,     -22.5],
    [ 0.7071, 0.0,  0.7071, -327.7],
    [ 0.0,    0.0,  0.0,       1.0],
], dtype=float)

_T30 = np.array([
    [0.0, -0.6654, -0.7465, -188.9],
    [0.0,  0.7465, -0.6654,  -89.04],
    [1.0,  0.0,     0.0,    -392.7],
    [0.0,  0.0,     0.0,       1.0],
], dtype=float)

_T40 = np.array([
    [-0.7071, 0.0, -0.7071, -187.3],
    [ 0.0,    1.0,  0.0,     -22.5],
    [ 0.7071, 0.0, -0.7071, -457.8],
    [ 0.0,    0.0,  0.0,       1.0],
], dtype=float)

_T50 = np.array([
    [0.0,  0.7465, -0.6654, -188.9],
    [0.0,  0.6654,  0.7465,  -44.0],
    [1.0,  0.0,     0.0,    -392.7],
    [0.0,  0.0,     0.0,       1.0],
], dtype=float)


_MARKER_SIZE_MM = 40.0

# Marker size used by the 5-marker gripper 3D print (matches rs.py)
TAG_MARKER_SIZE_MM = 72.5


def _config_from_T(marker_id: int, T: np.ndarray,
                   marker_size_mm: float = _MARKER_SIZE_MM) -> MarkerConfig:
    """Build a MarkerConfig from a 4×4 face-to-gripper transform T_i0."""
    R = T[:3, :3]
    t = T[:3, 3]
    return MarkerConfig(
        marker_id=marker_id,
        marker_size_mm=marker_size_mm,
        R_marker_to_com=R,
        t_marker_to_com=-R.T @ t,   # f0 origin in marker frame
        t_gripper_origin_mm=t,      # marker origin in f0
    )


def _inv_T(T: np.ndarray) -> np.ndarray:
    """Invert a 4×4 SE(3) matrix without np.linalg.inv."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3]  = -R.T @ t
    return T_inv


# ---------------------------------------------------------------------------
# Public config dict  (legacy single-marker paths)
# ---------------------------------------------------------------------------

MARKER_CONFIGS: dict[int, MarkerConfig] = {
    0:  _config_from_T(0,  _T10),
    1: _config_from_T(1, _T20),
    2: _config_from_T(2, _T30),
    3:  _config_from_T(3,  _T40),
    4:  _config_from_T(4,  _T50),
}

# ---------------------------------------------------------------------------
# TAG_TRANSFORMS  (rs.py-style multi-marker fusion)
# ---------------------------------------------------------------------------
#
# TAG_TRANSFORMS[tag_id] = T_jk  =  T_i0^{-1}
#
# Convention:  T_ij (camera←tag, from solvePnP) @ T_jk = T_ik (camera←gripper-k)
#   p_tag = T_jk @ p_k    →    p_cam = T_ij @ T_jk @ p_k
#
# Translation in mm.  Keys 1–5 match the ArUco IDs printed on the 5-face gripper.

TAG_TRANSFORMS: dict[int, np.ndarray] = {
    1: _inv_T(_T10),  # face 1
    2: _inv_T(_T20),  # face 2
    3: _inv_T(_T30),  # face 3
    4: _inv_T(_T40),  # face 4
    5: _inv_T(_T50),  # face 5
}
