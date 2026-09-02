"""Sensor noise and latency on the observation dict a policy sees.

Shared by himrobot.eval.evaluate (scoring under noise) and himrobot.rl.param_residual
(training under noise). Scoring and physics always use the true state.
"""
from __future__ import annotations

import numpy as np


class ObsPerturber:
    """Sensor noise and latency on the observation dict the *policy* sees.
    Scoring and the trajectory log always use the true state.

    Mirrors himrobot.rl.env.VecTT's obs_noise/joint_noise/obs_delay_steps:
    position noise std = the knob (metres / radians), velocity noise std =
    10x the knob (a tracker differencing two noisy positions), and a FIFO
    of ``delay_steps + 1`` observations primed with the reset observation so
    the policy sees an observation ``delay_steps`` control steps old. All
    off by default, in which case no random draw is made and the policy
    receives the env's own dict object, untouched."""

    def __init__(self, obs_noise: float = 0.0, joint_noise: float = 0.0, delay_steps: int = 0,
                 seed: int = 0):
        assert obs_noise >= 0.0 and joint_noise >= 0.0 and delay_steps >= 0
        self.obs_noise = float(obs_noise)
        self.joint_noise = float(joint_noise)
        self.delay_steps = int(delay_steps)
        self.rng = np.random.default_rng(seed + 4_242_017)
        self._fifo: list[dict] = []

    @property
    def active(self) -> bool:
        return self.obs_noise > 0.0 or self.joint_noise > 0.0 or self.delay_steps > 0

    def _noisy(self, obs: dict) -> dict:
        if self.obs_noise == 0.0 and self.joint_noise == 0.0:
            return obs
        out = dict(obs)
        if self.obs_noise > 0.0:
            out["ball.position"] = np.asarray(obs["ball.position"], dtype=np.float64) + self.rng.normal(0.0, self.obs_noise, 3)
            out["ball.velocity"] = np.asarray(obs["ball.velocity"], dtype=np.float64) + self.rng.normal(0.0, self.obs_noise * 10.0, 3)
        if self.joint_noise > 0.0:
            n = len(obs["robot.joint_position"])
            out["robot.joint_position"] = np.asarray(obs["robot.joint_position"], dtype=np.float64) + self.rng.normal(0.0, self.joint_noise, n)
            out["robot.joint_velocity"] = np.asarray(obs["robot.joint_velocity"], dtype=np.float64) + self.rng.normal(0.0, self.joint_noise * 10.0, n)
        return out

    def reset(self, obs: dict) -> dict:
        if not self.active:
            return obs
        first = self._noisy(obs)
        self._fifo = [first] * (self.delay_steps + 1)
        return self._fifo[0]

    def step(self, obs: dict) -> dict:
        if not self.active:
            return obs
        self._fifo.append(self._noisy(obs))
        del self._fifo[0]
        return self._fifo[0]
