"""Torch-free runtime for a tt_sim/rl/param_train.py policy.

HIM's mujoco worker image guarantees MuJoCo and numpy, not PyTorch, so the
package's evaluate entrypoint runs this copy of ParamResidualPolicy instead of
tt_sim.rl.policy's. Weights come from an .npz written by
scripts/export_param_policy.py; the forward pass, feature construction, tanh
squash and parameter decoding mirror tt_sim.rl.param_residual exactly
(tt_sim/tests/test_param_residual_np.py pins action-level parity).
"""
from __future__ import annotations

import os

import numpy as np

from tt_sim import scene
from tt_sim.policies.base import Policy
from tt_sim.policies.ik_hitter import IKHitterPolicy
from tt_sim.rl.intercept import predict_intercept

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# v1 (10 knobs): 70.5 % legal on seed 11, 12 % target hits on unseen serves.
S3_WEIGHTS = os.path.join(_REPO, "assets", "policies", "param_residual_s3.npz")
# v2 wa0 (20 knobs, placement reward): same legal rate, 26-31 % target hits, 0.39 m landing error.
V2_WA0_WEIGHTS = os.path.join(_REPO, "assets", "policies", "param_residual_v2_wa0.npz")
DEFAULT_WEIGHTS = V2_WA0_WEIGHTS  # what the package's evaluate entrypoint runs


def serve_features(obs: dict, target: tuple) -> np.ndarray:
    ball_pos = np.asarray(obs["ball.position"], dtype=np.float64)
    ball_vel = np.asarray(obs["ball.velocity"], dtype=np.float64)
    p_int, t_int, valid = predict_intercept(ball_pos, ball_vel)
    return np.concatenate([ball_pos, ball_vel, p_int, [t_int], [1.0 if valid else 0.0],
                           np.asarray(target, dtype=np.float64)]).astype(np.float32)


def unsquash_to_params(u: np.ndarray, names, kinds, bounds) -> dict:
    out = {}
    for value, name, kind, bound in zip(np.asarray(u, dtype=np.float64).reshape(-1), names, kinds, bounds):
        out[str(name)] = float(value * bound) if kind == "add" else float(np.exp(value * np.log(bound)))
    return out


class NumpyParamResidualPolicy(Policy):
    """Once per serve: features -> 2x64 tanh MLP -> tanh -> serve params on an IKHitterPolicy."""

    def __init__(self, weights_path: str | None = None):
        path = weights_path or os.environ.get("TT_PARAM_NPZ") or DEFAULT_WEIGHTS
        if not os.path.isfile(path):
            raise FileNotFoundError(f"NumpyParamResidualPolicy weights not found: {path}")
        z = np.load(path, allow_pickle=False)
        self.layers = [(z[f"W{i}"].astype(np.float64), z[f"b{i}"].astype(np.float64)) for i in range(3)]
        self.rms_mean = z["rms_mean"].astype(np.float64)
        self.rms_var = z["rms_var"].astype(np.float64)
        self.names = [str(n) for n in z["param_names"]]
        self.kinds = [str(k) for k in z["param_kinds"]]
        self.bounds = z["param_bounds"].astype(np.float64)
        # Metadata only -- names/kinds/bounds above are what actually decodes,
        # so an .npz exported before param_space existed still loads.
        self.param_space = str(z["param_space"]) if "param_space" in z.files else "v1"
        self.hitter = IKHitterPolicy()
        self._target = self._target_from_params(None)
        self._pending = True
        self._prev_time_s = None
        self.last_params: dict | None = None

    @staticmethod
    def _target_from_params(params: dict | None) -> tuple[float, float]:
        p = {**scene.DEFAULT_PARAMS, **(params or {})}
        return scene.target_point(p["target_zone"], p["target_depth"])

    def reset(self, episode: dict) -> None:
        self.hitter.reset(episode)
        self._target = self._target_from_params((episode or {}).get("params"))
        self._pending = True
        self._prev_time_s = None

    def _forward(self, feats: np.ndarray) -> np.ndarray:
        # Matches ppo.RunningMeanStd.normalize (eps 1e-8, clip +-10) then ParamActorCritic.actor.
        x = np.clip((feats.astype(np.float64) - self.rms_mean) / np.sqrt(self.rms_var + 1e-8), -10.0, 10.0)
        x = x.astype(np.float32)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T.astype(np.float32) + b.astype(np.float32)
            if i < 2:
                x = np.tanh(x)
        return x

    def _choose(self, obs: dict) -> None:
        mean = self._forward(serve_features(obs, self._target))
        u = np.tanh(mean.astype(np.float64))
        self.last_params = unsquash_to_params(u, self.names, self.kinds, self.bounds)
        self.hitter.set_serve_params(self.last_params)

    def act(self, obs: dict) -> np.ndarray:
        t_now = float(obs["time_s"])
        if self._prev_time_s is not None and t_now <= self._prev_time_s:
            self._pending = True  # the ball was reset under us: a new serve
        self._prev_time_s = t_now
        if self._pending:
            self._choose(obs)
            self._pending = False
        return self.hitter.act(obs)
