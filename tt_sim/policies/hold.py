"""HoldPolicy: the safe do-nothing baseline — holds whatever joint position it
last observed. Used as the eval CLI's default/fallback policy and as a
physics sanity check (a HoldPolicy should produce ~zero robot motion).
"""
from __future__ import annotations

import numpy as np

from tt_sim.policies.base import Policy


class HoldPolicy(Policy):
    """Returns the robot's current joint position every step (no motion)."""

    def reset(self, episode: dict) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:
        return np.asarray(obs["robot.joint_position"], dtype=np.float64).copy()
