"""
joint_tracker.py — Per-joint EMA (Exponential Moving Average) smoother for
                   MediaPipe pose landmarks.

Smoothing rule
--------------
    pos_smooth = alpha * pos_raw + (1 - alpha) * pos_prev

Dead-band
---------
    If ||pos_raw - pos_smooth|| < dead_band, skip the update (jitter suppressed).
    Once motion exceeds the dead_band, EMA kicks in at rate alpha.

Parameters
----------
alpha : float (0, 1]
    Blend weight toward the new raw measurement.
    Lower  → smoother / more lag on direction changes.
    Higher → faster response / more raw jitter.
    Typical: 0.3 – 0.5

dead_band : float  (same units as input positions)
    Minimum displacement to accept a measurement update.
    Larger → more jitter suppressed, but tiny real motions are also ignored.
    Pixel-space typical: 2 – 5 px.

Visibility gating
-----------------
Joints with visibility < VIS_THRESHOLD are skipped; the smoothed position
stays frozen at its last accepted value.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Visibility below this → skip update (keep last position)
VIS_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Data container  (same public interface as the old KF version)
# ---------------------------------------------------------------------------

@dataclass
class JointState:
    """Smoothed state of one body joint."""
    position: np.ndarray   # (3,)  camera frame (input units)


# ---------------------------------------------------------------------------
# Single-joint EMA filter
# ---------------------------------------------------------------------------

class JointEMAFilter:
    """
    EMA smoother with dead-band for one 3-D body joint.

    Parameters
    ----------
    alpha : float
        Blend weight toward new measurement. Range (0, 1].
    dead_band : float
        Minimum displacement (in input units) to trigger an update.
    """

    def __init__(self, alpha: float = 0.35, dead_band: float = 3.0):
        self.alpha     = alpha
        self.dead_band = dead_band
        self.pos: Optional[np.ndarray] = None  # (3,)  smoothed position

    def update(self, z: np.ndarray, visibility: float) -> Optional[np.ndarray]:
        """
        Apply EMA update.

        Returns the current smoothed position, or None if never initialised.
        """
        if visibility < VIS_THRESHOLD:
            return self.pos  # frozen — no update

        if self.pos is None:
            self.pos = z.copy()
            return self.pos.copy()

        delta = np.linalg.norm(z - self.pos)
        if delta > self.dead_band:
            self.pos = self.alpha * z + (1.0 - self.alpha) * self.pos

        return self.pos.copy()


# ---------------------------------------------------------------------------
# Full-body tracker (33 joints)
# ---------------------------------------------------------------------------

class PoseTracker:
    """
    Independent EMA filters for all 33 MediaPipe pose landmarks.

    Usage
    -----
        tracker = PoseTracker(alpha=0.35, dead_band=3.0)

        for positions, visibilities in zip(..., ...):
            states = tracker.update(positions, visibilities)
            # states[i] : JointState | None  (None until first observation)

    Parameters
    ----------
    alpha : float
        EMA blend weight (0, 1].  Lower = smoother.
    dead_band : float
        Minimum pixel displacement to accept an update.
    """

    N_JOINTS = 33

    def __init__(self, alpha: float = 0.35, dead_band: float = 3.0):
        self.filters: list[JointEMAFilter] = [
            JointEMAFilter(alpha=alpha, dead_band=dead_band)
            for _ in range(self.N_JOINTS)
        ]

    def update(
        self,
        positions:    np.ndarray,          # (33, 3) in input units
        visibilities: np.ndarray,          # (33,)   0.0 – 1.0
        dt:           float = 0.0,         # unused; kept for API compat
    ) -> list[Optional[JointState]]:
        """
        Update all 33 joint filters.

        Returns list[JointState | None] of length 33.
        None for joints that have never had a visible observation.
        """
        states: list[Optional[JointState]] = []

        for i, ema in enumerate(self.filters):
            z   = positions[i]
            vis = float(visibilities[i])
            pos = ema.update(z, vis)
            states.append(JointState(position=pos) if pos is not None else None)

        return states

    def reset(self) -> None:
        """Reset all filters (e.g. after a scene cut or long occlusion)."""
        for ema in self.filters:
            ema.pos = None
