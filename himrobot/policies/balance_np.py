"""Torch-free runtime for a himrobot/rl/balance_train.py "balance v1" policy.

Mirrors himrobot.policies.param_residual_np.NumpyParamResidualPolicy's own
role (HIM's mujoco worker image guarantees MuJoCo and numpy, not PyTorch):
NumpyBalancePolicy wraps the SAME shipped v3 hitting policy
(NumpyParamResidualPolicy, waist+arm, unchanged) plus a small numpy MLP
(the legs) trained by himrobot.rl.balance_train, exported to an .npz by that
module's export_npz. The obs construction and joint-target combination
here are the exact numpy translation of himrobot.rl.balance_env's
flatten_balance_obs / combined_joint_target -- tests/test_balance_env.py
pins action-level parity against the trainer's own mean action.

Deployment: himrobot.eval.evaluate --policy himrobot.policies.balance_np:NumpyBalancePolicy
--free-base (works out of the box off the module's default weights path;
override with a constructor arg or TT_BALANCE_NPZ, the same env-var-default
pattern NumpyParamResidualPolicy uses for TT_PARAM_NPZ).
"""
from __future__ import annotations

import os

import numpy as np

from himrobot.paths import ASSETS_DIR
from himrobot.sim import scene
from himrobot.policies.base import Policy
from himrobot.policies.param_residual_np import NumpyParamResidualPolicy
from himrobot.rl.balance_env import BALANCE_SCALE, combined_action, flatten_balance_obs, joint_target
from himrobot.rl.demos import target_to_action
from himrobot.rl.env import ACT_DIM, flatten_obs

DEFAULT_BALANCE_NPZ = str(ASSETS_DIR / "policies" / "balance_v1.npz")


class NumpyBalancePolicy(Policy):
    """Per control step: NumpyParamResidualPolicy (waist+arm, unchanged) +
    a 111->128->128->12 tanh MLP (legs) -> combined 29-dim joint target.
    See himrobot.rl.balance_env's module docstring for the obs layout and the
    combination formula this reproduces exactly."""

    def __init__(self, weights_path: str | None = None):
        path = weights_path or os.environ.get("TT_BALANCE_NPZ") or DEFAULT_BALANCE_NPZ
        if not os.path.isfile(path):
            raise FileNotFoundError(f"NumpyBalancePolicy weights not found: {path}")
        z = np.load(path, allow_pickle=False)
        self.layers = [(z[f"W{i}"].astype(np.float64), z[f"b{i}"].astype(np.float64)) for i in range(3)]
        self.rms_mean = z["rms_mean"].astype(np.float64)
        self.rms_var = z["rms_var"].astype(np.float64)
        self.balance_scale = float(z["balance_scale"])
        if self.balance_scale != BALANCE_SCALE:
            raise ValueError(f"{path}: balance_scale {self.balance_scale} != code's {BALANCE_SCALE}")
        self.obs_version = int(z["obs_version"])
        self.jnt_range = z["jnt_range"].astype(np.float64)
        expert_npz = str(z["expert_npz"])
        self.expert = NumpyParamResidualPolicy(expert_npz)

        self._target = self._target_from_params(None)
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float64)

    @staticmethod
    def _target_from_params(params: dict | None) -> tuple[float, float]:
        p = {**scene.DEFAULT_PARAMS, **(params or {})}
        return scene.target_point(p["target_zone"], p["target_depth"])

    def reset(self, episode: dict) -> None:
        self.expert.reset(episode)
        self._target = self._target_from_params((episode or {}).get("params"))
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float64)

    def _forward(self, flat_obs: np.ndarray) -> np.ndarray:
        # Matches ppo.RunningMeanStd.normalize (eps 1e-8, clip +-10) then
        # himrobot.rl.nets.Actor's forward (tanh after every Linear, including
        # the last -- its output is already the deterministic action, no
        # separate squashing needed).
        x = np.clip((flat_obs.astype(np.float64) - self.rms_mean) / np.sqrt(self.rms_var + 1e-8), -10.0, 10.0)
        for w, b in self.layers:
            x = np.tanh(x @ w.T + b)
        return x

    def act(self, obs: dict) -> np.ndarray:
        expert_target = self.expert.act(obs)
        expert_action = target_to_action(expert_target)

        base = flatten_obs(obs, self._prev_action, self._target, obs["time_s"])
        flat = flatten_balance_obs(base, obs)
        a_rl = np.clip(self._forward(flat), -1.0, 1.0)

        combined = combined_action(a_rl, expert_action)
        self._prev_action = combined
        return joint_target(combined, self.jnt_range)
