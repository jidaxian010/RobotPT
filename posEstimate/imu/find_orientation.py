from ahrs.filters import Madgwick
import numpy as np
from scipy.spatial.transform import Rotation
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from util import quaternion_to_rotation_matrix, quaternion_to_euler
"""
Input: gyro_object
Output: orientation
"""

class MadgwickOrientation:
    def __init__(self, acc_object, gyro_object, dt, data_format, method="madgwick"):
        self.gyro_object = gyro_object
        self.acc_object = acc_object
        self.data_format = data_format
        self.method = method
        self.dt = dt

    def use_madgwick(self):
        """
        Find orientation using Madgwick filter
        """
        madgwick = Madgwick(gyr=self.gyro_object, acc=self.acc_object, Dt=self.dt)
        quaternion = madgwick.Q

        return quaternion


    def find_orientation(self):
        """
        Find orientation using Madgwick filter
        """
        quaternion = self.use_madgwick()
        if self.data_format == "quaternion":
            return quaternion
        elif self.data_format == "rotation_matrix":
            return quaternion_to_rotation_matrix(quaternion)
        elif self.data_format == "euler":
            return quaternion_to_euler(quaternion)
        else:
            raise ValueError(f"Invalid data format: {self.data_format}")


class PyPoseOrientation:
    def __init__(self, acc_object, gyro_object, dt, data_format):
        self.gyro_object = gyro_object
        self.acc_object = acc_object
        self.data_format = data_format
        self.dt = dt  # Sampling period

    def use_pypose(self):
        """
        Find orientation using PyPose IMU integration
        """
        import torch
        import pypose as pp
        
        # Convert gyro, accel to torch
        gyro_arm_torch = torch.from_numpy(self.gyro_object).to(dtype=torch.double)
        accel_world_torch = torch.from_numpy(self.acc_object).to(dtype=torch.double)
        T = gyro_arm_torch.shape[0]

        # Build dt with matching time axis (and batch axis):
        dt_torch = torch.full((1, T, 1), float(self.dt), dtype=torch.double)   # [1,T,1]
        gyro_b = gyro_arm_torch.unsqueeze(0)                                    # [1,T,3]
        acc_b = accel_world_torch.unsqueeze(0)                                  # [1,T,3]

        # Initial conditions
        p0 = torch.zeros(1, 1, 3, dtype=torch.double)
        v0 = torch.zeros(1, 1, 3, dtype=torch.double)
        r0 = pp.so3(torch.zeros(1, 1, 3, dtype=torch.double)).Exp()  # identity SO3

        # Create IMU integrator
        integrator = pp.module.IMUPreintegrator(p0, r0, v0, gravity=9.81, reset=False).to(torch.double)
        states = integrator(dt=dt_torch, gyro=gyro_b, acc=acc_b)

        # Extract rotation (orientation) from states
        # PyPose returns quaternions in states['rot']
        quaternions_torch = states['rot']  # Shape: [1, T, 4]
        
        # Remove batch dimension and convert to numpy
        quaternions = quaternions_torch.squeeze(0).detach().numpy()  # Shape: [T, 4]
        
        # PyPose quaternions are in [x, y, z, w] format, convert to [w, x, y, z]
        quaternions_wxyz = quaternions[:, [3, 0, 1, 2]]  # Reorder to [w, x, y, z]
        
        return quaternions_wxyz

    def find_orientation(self):
        """
        Find orientation using PyPose and return in requested format
        """
        # Get quaternions from PyPose (same as Madgwick)
        quaternions = self.use_pypose()
        
        if self.data_format == "quaternion":
            return quaternions
        elif self.data_format == "rotation_matrix":
            return quaternion_to_rotation_matrix(quaternions)
        elif self.data_format == "euler":
            return quaternion_to_euler(quaternions)
        else:
            raise ValueError(f"Invalid data format: {self.data_format}")