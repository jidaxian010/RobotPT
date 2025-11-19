"""
IMU processing package for pose estimation.

This package contains modules for:
- Calibration: IMU sensor calibration (bias and covariance)
- Use calibration: Apply calibration to IMU data with spike removal
- Object frame transformation: Convert IMU data to object reference frame
- Orientation estimation: Madgwick and PyPose filters
- Position estimation: Double integration with bias correction
"""

from .calibration import Calibration, calibrate_data, save_to_json
from .use_calibration import UseCalibration
from .object_frame import ObjectFrame
from .find_orientation import MadgwickOrientation, PyPoseOrientation
from .find_position import FindPosition

__all__ = [
    'Calibration',
    'calibrate_data',
    'save_to_json',
    'UseCalibration',
    'ObjectFrame',
    'MadgwickOrientation',
    'PyPoseOrientation',
    'FindPosition',
]

