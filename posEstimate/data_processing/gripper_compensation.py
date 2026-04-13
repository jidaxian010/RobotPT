"""
Gripper gravity/inertia compensation.

Fits the gripper mass m and center-of-mass r_CoM (in sensor frame) from a
calibration bag that has *only* the gripper attached to the force sensor.
The resulting model can then subtract the gripper's contribution from any
future wrench measurement where both the gripper and an object are present.

Model (all quantities in sensor frame):
    F_gripper(t) = m * a_sensor(t)
    T_gripper(t) = r_CoM × (m * a_sensor(t))

Written as a linear system with unknowns p = [m, m·rx, m·ry, m·rz]:

    A(t) @ p = w_net(t)

where w_net = wrench after bias subtraction, and A is built from a_sensor:

    A = [ ax   0    0    0  ]
        [ ay   0    0    0  ]
        [ az   0    0    0  ]
        [  0   0    az  -ay ]   (Tx = (m·ry)·az − (m·rz)·ay)
        [  0  -az   0    ax ]   (Ty = (m·rz)·ax − (m·rx)·az)
        [  0   ay  -ax   0  ]   (Tz = (m·rx)·ay − (m·ry)·ax)

Usage
-----
    # --- calibration ---
    comp = GripperCompensator(R_sensor2gripper)
    comp.fit(imu_data_sensor_frame, force_net)   # force_net already bias-subtracted
    comp.save("gripper_comp.npz")

    # --- compensation in future bags ---
    comp = GripperCompensator.load("gripper_comp.npz")
    force_object_only = comp.compensate(force_measured_net, a_sensor)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GripperCompensator:
    """
    Fits and applies gripper gravity/inertia compensation.

    Parameters
    ----------
    R_sensor2gripper : (3, 3) ndarray
        Rotation matrix from sensor frame to gripper frame.
        Used to rotate IMU acceleration (in gripper frame) back to sensor frame.
    """

    def __init__(self, R_sensor2gripper: np.ndarray):
        self.R_sensor2gripper = np.asarray(R_sensor2gripper, dtype=np.float64)
        # gripper→sensor is the transpose (inverse) of sensor→gripper
        self.R_gripper2sensor: np.ndarray = self.R_sensor2gripper.T

        # Results populated by fit()
        self.mass: float | None = None          # kg
        self.r_com: np.ndarray | None = None    # (3,) in sensor frame, metres
        self.residual_rms: float | None = None  # RMS fit residual (N or Nm)
        self._p: np.ndarray | None = None       # raw [m, m*rx, m*ry, m*rz]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_regressor(a: np.ndarray) -> np.ndarray:
        """
        Build (6, 4) regressor matrix for a single acceleration vector a=[ax,ay,az].
        Columns correspond to unknowns [m, m*rx, m*ry, m*rz].
        """
        ax, ay, az = float(a[0]), float(a[1]), float(a[2])
        return np.array([
            [ax,   0.,   0.,   0. ],  # Fx = m*ax
            [ay,   0.,   0.,   0. ],  # Fy = m*ay
            [az,   0.,   0.,   0. ],  # Fz = m*az
            [ 0.,  0.,   az,  -ay ],  # Tx = (m*ry)*az - (m*rz)*ay
            [ 0., -az,   0.,   ax ],  # Ty = (m*rz)*ax - (m*rx)*az
            [ 0.,  ay,  -ax,   0. ],  # Tz = (m*rx)*ay - (m*ry)*ax
        ], dtype=np.float64)

    def accel_to_sensor_frame(self, a_gripper: np.ndarray) -> np.ndarray:
        """
        Rotate acceleration vectors from gripper frame to sensor frame.

        Parameters
        ----------
        a_gripper : (N, 3) or (3,)
        Returns
        -------
        (N, 3) or (3,) in sensor frame
        """
        a = np.asarray(a_gripper, dtype=np.float64)
        if a.ndim == 1:
            return self.R_gripper2sensor @ a
        return (self.R_gripper2sensor @ a.T).T

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(
        self,
        a_sensor: np.ndarray,
        wrench_net: np.ndarray,
        force_weight: float = 1.0,
        torque_weight: float = 1.0,
    ) -> "GripperCompensator":
        """
        Solve for gripper mass and CoM via weighted least squares.

        Parameters
        ----------
        a_sensor : (N, 3)
            IMU acceleration in **sensor frame** (includes gravity).
        wrench_net : (N, 6)
            Bias-subtracted wrench [Fx,Fy,Fz,Tx,Ty,Tz] in sensor frame.
        force_weight : float
            Weight applied to the force rows (N) relative to torque rows (Nm).
        torque_weight : float
            Weight applied to the torque rows.

        Returns
        -------
        self
        """
        a_sensor = np.asarray(a_sensor, dtype=np.float64)
        wrench_net = np.asarray(wrench_net, dtype=np.float64)

        if a_sensor.shape[0] != wrench_net.shape[0]:
            raise ValueError(
                f"Length mismatch: a_sensor {a_sensor.shape[0]} vs "
                f"wrench_net {wrench_net.shape[0]}"
            )

        N = a_sensor.shape[0]

        # Stacked system: (6N, 4) @ (4,) = (6N,)
        A_rows = []
        b_rows = []
        w_diag = np.array([
            force_weight, force_weight, force_weight,
            torque_weight, torque_weight, torque_weight,
        ], dtype=np.float64)

        for i in range(N):
            Ai = self._build_regressor(a_sensor[i])
            A_rows.append(Ai * w_diag[:, None])
            b_rows.append(wrench_net[i] * w_diag)

        A_full = np.vstack(A_rows)   # (6N, 4)
        b_full = np.concatenate(b_rows)  # (6N,)

        p, residuals, rank, sv = np.linalg.lstsq(A_full, b_full, rcond=None)
        self._p = p

        self.mass = float(p[0])
        if abs(self.mass) < 1e-6:
            raise RuntimeError(
                f"Fitted mass is near-zero ({self.mass:.4f} kg). "
                "Check that IMU acceleration is in sensor frame and wrench bias was subtracted."
            )
        self.r_com = p[1:4] / self.mass  # metres, sensor frame

        # RMS residual
        b_pred = A_full @ p
        self.residual_rms = float(np.sqrt(np.mean((b_full - b_pred) ** 2)))

        print(f"[GripperCompensator] Fit results:")
        print(f"  mass        = {self.mass:.4f} kg")
        print(f"  r_CoM       = [{self.r_com[0]:.4f}, {self.r_com[1]:.4f}, {self.r_com[2]:.4f}] m (sensor frame)")
        print(f"  residual RMS= {self.residual_rms:.4f}  (weighted force/torque units)")
        print(f"  matrix rank = {rank}")

        return self

    # ------------------------------------------------------------------
    # Apply compensation
    # ------------------------------------------------------------------

    def gripper_wrench(self, a_sensor: np.ndarray) -> np.ndarray:
        """
        Predict the gripper's contribution to the wrench at given accelerations.

        Parameters
        ----------
        a_sensor : (N, 3) or (3,)
            IMU acceleration in sensor frame.

        Returns
        -------
        (N, 6) or (6,) predicted gripper wrench [Fx,Fy,Fz,Tx,Ty,Tz].
        """
        if self._p is None:
            raise RuntimeError("Call fit() or load() first.")

        a = np.asarray(a_sensor, dtype=np.float64)
        scalar = a.ndim == 1
        if scalar:
            a = a[None, :]

        result = np.einsum("nij,j->ni", np.stack([self._build_regressor(a[k]) for k in range(len(a))]), self._p)

        return result[0] if scalar else result

    def compensate(self, wrench_net: np.ndarray, a_sensor: np.ndarray) -> np.ndarray:
        """
        Remove the gripper's contribution from a bias-subtracted wrench.

        Parameters
        ----------
        wrench_net : (N, 6) or (6,)
            Wrench already corrected for sensor bias, in sensor frame.
        a_sensor : (N, 3) or (3,)
            IMU acceleration in sensor frame (same length as wrench_net).

        Returns
        -------
        (N, 6) or (6,)  wrench due to the object only.
        """
        return wrench_net - self.gripper_wrench(a_sensor)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save calibration parameters to a .npz file."""
        if self._p is None:
            raise RuntimeError("Nothing to save — call fit() first.")
        path = Path(path)
        np.savez(
            path,
            R_sensor2gripper=self.R_sensor2gripper,
            p=self._p,
            mass=np.array([self.mass]),
            r_com=self.r_com,
            residual_rms=np.array([self.residual_rms]),
        )
        print(f"[GripperCompensator] Saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GripperCompensator":
        """Load calibration parameters from a .npz file."""
        path = Path(path)
        data = np.load(path)
        obj = cls(R_sensor2gripper=data["R_sensor2gripper"])
        obj._p = data["p"]
        obj.mass = float(data["mass"][0])
        obj.r_com = data["r_com"]
        obj.residual_rms = float(data["residual_rms"][0])
        print(
            f"[GripperCompensator] Loaded from {path}  "
            f"(m={obj.mass:.4f} kg, r_CoM={np.round(obj.r_com, 4)})"
        )
        return obj

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the calibration."""
        if self._p is None:
            return "GripperCompensator: not fitted."
        lines = [
            "GripperCompensator summary:",
            f"  mass          = {self.mass:.4f} kg",
            f"  r_CoM (sensor)= {np.round(self.r_com, 5).tolist()} m",
            f"  residual RMS  = {self.residual_rms:.5f}",
        ]
        return "\n".join(lines)
