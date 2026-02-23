"""
joint_tracker.py — Per-joint Kalman filter (constant-velocity) for MediaPipe pose landmarks.

State per joint : [px, py, pz, vx, vy, vz]   (6-D)
Measurement     : [px, py, pz]                (3-D)

Observation noise R scales with visibility:
    R = (R_base / max(visibility, 0.05))^2 * I_3
    → low visibility  → large R → measurement trusted less
    → high visibility → small R → measurement fully trusted

Outlier rejection (Mahalanobis distance):
    If the innovation is too far from the predicted state, the measurement is
    rejected and only the predict step runs — covariance grows, uncertainty
    circle expands in the visualisation.

Units
-----
The tracker is unit-agnostic.  Typical configurations:
    • Pixel-space (from MediaPipe normalised coords × image size):
          sigma_a = 500  px/s²,  R_base = 8   px
    • Metric camera frame (after depth fusion):
          sigma_a = 300  mm/s²,  R_base = 20  mm
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# chi-squared threshold: 3 DOF, 95 % confidence → reject measurement if exceeded
_CHI2_3_95 = 7.815

# Visibility below this → predict-only (no measurement update)
VIS_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class JointState:
    """Filtered state of one body joint."""
    position: np.ndarray   # (3,)  camera frame (input units)
    velocity: np.ndarray   # (3,)  input-units / s
    sigma:    np.ndarray   # (3,)  1-sigma position uncertainty (input units)


# ---------------------------------------------------------------------------
# Single-joint filter
# ---------------------------------------------------------------------------

class JointKalmanFilter:
    """
    Constant-velocity Kalman filter for one 3-D body joint.

    Parameters
    ----------
    sigma_a : float
        Acceleration standard deviation (process noise).
    R_base : float
        Measurement noise at visibility = 1.0.
    """

    def __init__(self, sigma_a: float = 300.0, R_base: float = 20.0):
        self.sigma_a = sigma_a
        self.R_base  = R_base
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])  # (3, 6)
        self.x: Optional[np.ndarray] = None  # (6,)
        self.P: Optional[np.ndarray] = None  # (6, 6)

    # ------------------------------------------------------------------

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        """Discrete constant-acceleration process noise."""
        q = self.sigma_a ** 2
        block = q * np.array([[dt**4 / 4, dt**3 / 2],
                               [dt**3 / 2, dt**2      ]])
        Q = np.zeros((6, 6))
        for i in range(3):
            idx = [i, i + 3]
            Q[np.ix_(idx, idx)] = block
        return Q

    def _R(self, visibility: float) -> np.ndarray:
        vis = max(visibility, 0.05)
        return np.eye(3) * (self.R_base / vis) ** 2

    # ------------------------------------------------------------------

    def initialize(self, z: np.ndarray) -> None:
        self.x       = np.zeros(6)
        self.x[:3]   = z
        init_p_var   = (self.R_base * 2) ** 2
        init_v_var   = (self.sigma_a * 0.5) ** 2
        self.P       = np.diag([init_p_var] * 3 + [init_v_var] * 3)

    def predict(self, dt: float) -> None:
        F      = self._F(dt)
        Q      = self._Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, visibility: float) -> bool:
        """
        Update with measurement z.

        Returns True if measurement accepted, False if rejected as outlier.
        """
        R   = self._R(visibility)
        inn = z - self.H @ self.x             # innovation  (3,)
        S   = self.H @ self.P @ self.H.T + R  # innovation covariance (3, 3)

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        # Mahalanobis distance
        d2 = float(inn @ S_inv @ inn)
        if d2 > _CHI2_3_95:
            return False  # outlier — predict only

        K      = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ inn
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return True

    def state(self) -> JointState:
        return JointState(
            position=self.x[:3].copy(),
            velocity=self.x[3:].copy(),
            sigma=np.sqrt(np.diag(self.P)[:3]),
        )


# ---------------------------------------------------------------------------
# Full-body tracker (33 joints)
# ---------------------------------------------------------------------------

class PoseTracker:
    """
    Independent Kalman filters for all 33 MediaPipe pose landmarks.

    Usage
    -----
        tracker = PoseTracker()

        dts = [0.0] + list(np.diff(timestamps))   # per-frame dt
        for positions, visibilities, dt in zip(..., ..., dts):
            states = tracker.update(positions, visibilities, dt)
            # states[i] : JointState | None (None until first observation)

    Parameters
    ----------
    sigma_a : float
        Acceleration std.  Pixel-space: ~500.  Metric (mm): ~300.
    R_base : float
        Measurement noise at vis=1.  Pixel-space: ~8.  Metric (mm): ~20.
    """

    N_JOINTS = 33

    def __init__(self, sigma_a: float = 300.0, R_base: float = 20.0):
        self.filters: list[JointKalmanFilter] = [
            JointKalmanFilter(sigma_a=sigma_a, R_base=R_base)
            for _ in range(self.N_JOINTS)
        ]

    def update(
        self,
        positions:    np.ndarray,   # (33, 3)  in input units
        visibilities: np.ndarray,   # (33,)    0.0 – 1.0
        dt:           float,        # seconds since last call; 0 on first frame
    ) -> list[Optional[JointState]]:
        """
        Predict + update all 33 joint filters.

        - Joints with visibility < VIS_THRESHOLD: predict-only (σ grows).
        - Uninitialized joints: initialized on first visible observation.
        - Outliers (large Mahalanobis distance): treated as predict-only.

        Returns list[JointState | None] of length 33.
        """
        states: list[Optional[JointState]] = []

        for i, kf in enumerate(self.filters):
            z   = positions[i]
            vis = float(visibilities[i])

            if kf.x is None:
                # First time this joint is visible
                if vis >= VIS_THRESHOLD:
                    kf.initialize(z)
                states.append(kf.state() if kf.x is not None else None)
                continue

            if dt > 0:
                kf.predict(dt)

            if vis >= VIS_THRESHOLD:
                kf.update(z, vis)
            # else: covariance grows — uncertainty circle expands

            states.append(kf.state())

        return states

    def reset(self) -> None:
        """Reset all filters (e.g. after a scene cut or long occlusion)."""
        for kf in self.filters:
            kf.x = None
            kf.P = None
