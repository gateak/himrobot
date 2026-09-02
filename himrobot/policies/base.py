"""Policy interface consumed by himrobot.eval.evaluate.

A Policy is reset once per episode with the episode's metadata (whatever
himrobot.eval.evaluate chooses to pass — e.g. the serve dict and params) and is then
called once per control step with the current observation dict (the same
shape TableTennisEnv.step/reset returns) to produce the next 29-dim joint
target in radians, ordered per .him/contracts/runtime.json
extensions.robot.joint_order.
"""
from __future__ import annotations

import numpy as np


class Policy:
    """Base class. Subclasses override reset/act; both are required."""

    def reset(self, episode: dict) -> None:
        """Called once at the start of each episode.

        ``episode`` is caller-defined metadata about the upcoming trial
        (e.g. {"serve": <sample_serve() dict>, "params": <env params>}).
        The base implementation does nothing.
        """
        return None

    def act(self, obs: dict) -> np.ndarray:
        """Return the next joint-position target, shape (29,), radians."""
        raise NotImplementedError
