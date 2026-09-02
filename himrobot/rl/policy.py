"""himrobot.policies.base.Policy adapters for himrobot/rl's checkpoints, loadable
by himrobot.eval.evaluate via --policy himrobot.rl.policy:BCPolicy /
--policy himrobot.rl.policy:PPOPolicy / --policy himrobot.rl.policy:ParamResidualPolicy (evaluate.py's _load_policy constructs a
policy with a bare `cls()`, so every Policy here must be zero-arg
constructible -- PPOPolicy's checkpoint source is an env var precisely so
`cls()` still works: see TT_RL_CKPT below).

BCPolicy wraps a himrobot/rl/bc.py checkpoint (a plain himrobot.rl.nets.Actor
state_dict). PPOPolicy wraps a himrobot/rl/ppo.py checkpoint (an ActorCritic
state_dict + running obs-normalization stats) and runs it deterministically
(the trained Gaussian's mean, not a sampled action) -- see himrobot/rl/ppo.py
for the training side of both formats.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np
import torch

from himrobot.sim import scene
from himrobot.policies.base import Policy
from himrobot.policies.ik_hitter import IKHitterPolicy
from himrobot.rl import param_residual
from himrobot.rl import ppo as ppo_mod
from himrobot.rl.demos import target_to_action
from himrobot.rl.env import ACT_DIM, flatten_obs, map_action
from himrobot.rl.nets import Actor
from himrobot.rl.residual import DEFAULT_RESIDUAL_SCALE, RESIDUAL_OBS_DIM, flatten_residual_obs

DEFAULT_BC_CHECKPOINT = os.path.join(scene.REPO, "work", "tt", "rl", "bc_seed0.pt")


def _compute_jnt_range() -> np.ndarray:
    """Per-joint (lo, hi) in himrobot.sim.scene.JOINT_ORDER, for himrobot.rl.env.map_action's
    clamp. Duplicated (not imported) since env.py's helper is module-private
    (_joint_range) -- both read the same himrobot.sim.scene.build_model(), so
    they can't drift apart in practice.
    """
    model = scene.build_model(fixed_base=True, paddle=True)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in scene.JOINT_ORDER]
    return np.array([model.jnt_range[j] for j in ids], dtype=np.float64)


def _target_from_params(params: dict | None) -> tuple[float, float]:
    p = {**scene.DEFAULT_PARAMS, **(params or {})}
    return scene.target_point(p["target_zone"], p["target_depth"])


class BCPolicy(Policy):
    """Runs a himrobot/rl/bc.py-trained himrobot.rl.nets.Actor checkpoint.

    act() rebuilds the same 99-dim flat observation himrobot/rl/env.py and
    himrobot/rl/demos.py use (himrobot.rl.env.flatten_obs), forwards it through
    the network to get an action in [-1, 1]^29, and maps that to an
    absolute joint-position target with himrobot.rl.env.map_action (Policy.act
    must return absolute radians -- see himrobot.policies.base.Policy's
    docstring; himrobot.eval.evaluate feeds that straight to TableTennisEnv.step).
    """

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu"):
        path = checkpoint_path or os.environ.get("BC_POLICY_CHECKPOINT") or DEFAULT_BC_CHECKPOINT
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BCPolicy checkpoint not found at {path!r}. Run "
                "`.venv/bin/python -m himrobot.rl.bc` first (it writes "
                f"{DEFAULT_BC_CHECKPOINT}), or pass checkpoint_path=/set "
                "BC_POLICY_CHECKPOINT to point at another one."
            )
        # weights_only=False: our own checkpoint carries plain-Python
        # metadata (epoch counts, loss history) alongside the state_dict --
        # only ever load a checkpoint himrobot/rl/bc.py itself produced.
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.device = torch.device(device)
        self.actor = Actor(
            obs_dim=ckpt.get("obs_dim", 99),
            act_dim=ckpt.get("act_dim", ACT_DIM),
            hidden=ckpt.get("hidden", 256),
        )
        self.actor.load_state_dict(ckpt["model_state_dict"])
        self.actor.to(self.device)
        self.actor.eval()

        self._jnt_range = _compute_jnt_range()
        self._target = _target_from_params(None)
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float64)

    def reset(self, episode: dict) -> None:
        self._target = _target_from_params((episode or {}).get("params"))
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float64)

    def act(self, obs: dict) -> np.ndarray:
        flat = flatten_obs(obs, self._prev_action, self._target, obs["time_s"])
        x = torch.from_numpy(flat).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(x).squeeze(0).cpu().numpy().astype(np.float64)
        action = np.clip(action, -1.0, 1.0)
        self._prev_action = action
        return map_action(action, self._jnt_range)


class PPOPolicy(Policy):
    """Runs a himrobot/rl/ppo.py-trained ActorCritic checkpoint deterministically
    (the Gaussian policy's mean, not a sampled action) through the same
    flat-obs / map_action pipeline as BCPolicy.

    Zero-arg constructible for himrobot.eval.evaluate's `--policy
    himrobot.rl.policy:PPOPolicy` (its _load_policy calls `cls()` with no
    args): pass ckpt_path explicitly, or set the TT_RL_CKPT env var to a
    himrobot/rl/ppo.py checkpoint path (e.g. work/tt/rl/run/final.pt or
    .../best.pt) before running the evaluator CLI.
    """

    def __init__(self, ckpt_path: str | None = None):
        path = ckpt_path or os.environ.get("TT_RL_CKPT")
        if not path:
            raise ValueError(
                "PPOPolicy needs a checkpoint: pass ckpt_path=... or set the "
                "TT_RL_CKPT env var to a himrobot/rl/ppo.py checkpoint (e.g. "
                "work/tt/rl/run/final.pt)."
            )
        self.device = ppo_mod.pick_device()
        self.model, self.obs_rms = ppo_mod.load_checkpoint(path, self.device)
        self.model.eval()

        self._jnt_range = _compute_jnt_range()
        self._target = _target_from_params(None)
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float64)

    def reset(self, episode: dict) -> None:
        self._target = _target_from_params((episode or {}).get("params"))
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float64)

    def act(self, obs: dict) -> np.ndarray:
        flat = flatten_obs(obs, self._prev_action, self._target, obs["time_s"])
        norm = self.obs_rms.normalize(flat[None, :].astype(np.float64))
        obs_t = torch.as_tensor(norm, dtype=torch.float32, device=self.device)
        mean, _logp, _value = self.model.act(obs_t, deterministic=True)
        action = np.clip(mean.cpu().numpy()[0].astype(np.float64), -1.0, 1.0)
        self._prev_action = action
        return map_action(action, self._jnt_range)


class ResidualPPOPolicy(Policy):
    """Runs a himrobot/rl/ppo.py `--residual` checkpoint: internally drives
    its own himrobot.policies.ik_hitter.IKHitterPolicy instance (the same
    "expert" himrobot.rl.residual.VecTTResidual wraps during training) to get
    a_exp for the current obs, forwards the residual ActorCritic's mean
    action a_rl on the same [flatten_obs(obs) ++ a_exp] 128-dim
    (RESIDUAL_OBS_DIM) input himrobot.rl.residual.flatten_residual_obs builds,
    and combines them exactly as VecTTResidual.step does:
    ``a = clip(a_exp + residual_scale * a_rl, -1, 1)`` -- before mapping to
    an absolute joint-position target with himrobot.rl.env.map_action, same
    as PPOPolicy.

    Zero-arg constructible for himrobot.eval.evaluate's `--policy
    himrobot.rl.policy:ResidualPPOPolicy` (see PPOPolicy's docstring on
    TT_RL_CKPT). residual_scale defaults to whatever himrobot/rl/ppo.py's
    --residual training run recorded in the checkpoint
    (ckpt["residual_scale"]; falls back to
    himrobot.rl.residual.DEFAULT_RESIDUAL_SCALE for a checkpoint that
    doesn't carry one), or pass residual_scale=... to override.
    """

    def __init__(self, ckpt_path: str | None = None, residual_scale: float | None = None):
        path = ckpt_path or os.environ.get("TT_RL_CKPT")
        if not path:
            raise ValueError(
                "ResidualPPOPolicy needs a checkpoint: pass ckpt_path=... or set the "
                "TT_RL_CKPT env var to a himrobot/rl/ppo.py --residual checkpoint (e.g. "
                "work/tt/rl/run/final.pt)."
            )
        self.device = ppo_mod.pick_device()
        self.model, self.obs_rms = ppo_mod.load_checkpoint(path, self.device)
        self.model.eval()
        assert self.model.actor.obs_dim == RESIDUAL_OBS_DIM, (
            f"{path!r} is a {self.model.actor.obs_dim}-dim checkpoint, not a --residual "
            f"({RESIDUAL_OBS_DIM}-dim) one."
        )
        # weights_only=False: see ppo_mod.load_checkpoint's own comment; read a second
        # time only to recover the plain-Python residual_scale metadata it doesn't return.
        raw_ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.residual_scale = float(
            residual_scale if residual_scale is not None
            else raw_ckpt.get("residual_scale", DEFAULT_RESIDUAL_SCALE)
        )

        self.hitter = IKHitterPolicy()
        self._jnt_range = _compute_jnt_range()
        self._target = _target_from_params(None)
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float64)

    def reset(self, episode: dict) -> None:
        self.hitter.reset(episode)
        self._target = _target_from_params((episode or {}).get("params"))
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float64)

    def act(self, obs: dict) -> np.ndarray:
        expert_target = self.hitter.act(obs)
        expert_action = target_to_action(expert_target)

        base = flatten_obs(obs, self._prev_action, self._target, obs["time_s"])
        flat = flatten_residual_obs(base, expert_action)
        norm = self.obs_rms.normalize(flat[None, :].astype(np.float64))
        obs_t = torch.as_tensor(norm, dtype=torch.float32, device=self.device)
        mean, _logp, _value = self.model.act(obs_t, deterministic=True)
        a_rl = mean.cpu().numpy()[0].astype(np.float64)

        action = np.clip(expert_action + self.residual_scale * a_rl, -1.0, 1.0)
        self._prev_action = action
        return map_action(action, self._jnt_range)


class ParamResidualPolicy(Policy):
    """Runs a himrobot/rl/param_train.py checkpoint: once per serve it reads
    himrobot.rl.param_residual.serve_features from the episode's first
    observation, forwards them through the trained
    himrobot.rl.param_residual.ParamActorCritic deterministically (the Gaussian's
    mean, not a sample), squashes to u = tanh(mean) and retunes its own
    himrobot.policies.ik_hitter.IKHitterPolicy with the resulting serve params for
    the whole episode. Every control step after that is the plain IK hitter --
    there is no per-step residual.

    Episode starts are detected from the observation dict itself, not only from
    reset(): a new episode is any obs whose ``time_s`` is not greater than the
    previous one (TableTennisEnv restarts its clock at 0 on reset). reset()
    also arms the same one-shot, so the parameters are chosen on the first
    act() of a trial either way -- which is the only point where the serve is
    actually visible, since reset() runs before any observation exists.

    Zero-arg constructible for himrobot.eval.evaluate's `--policy
    himrobot.rl.policy:ParamResidualPolicy` (see PPOPolicy's docstring on
    TT_RL_CKPT). The parameter ranges come from the checkpoint, not from
    param_residual's current PARAM_SPECS.
    """

    def __init__(self, ckpt_path: str | None = None):
        path = ckpt_path or os.environ.get("TT_RL_CKPT")
        if not path:
            raise ValueError(
                "ParamResidualPolicy needs a checkpoint: pass ckpt_path=... or set the "
                "TT_RL_CKPT env var to a himrobot/rl/param_train.py checkpoint (e.g. "
                "work/tt/rl/param_run/best.pt)."
            )
        self.device = ppo_mod.pick_device()
        self.model, self.feat_rms, self.param_specs = param_residual.load_checkpoint(path, self.device)
        self.param_space = getattr(self.model, "param_space", "v1")
        self.model.eval()

        self.hitter = IKHitterPolicy()
        self._target = _target_from_params(None)
        self._pending = True
        self._prev_time_s = None
        self.last_params: dict | None = None

    def reset(self, episode: dict) -> None:
        self.hitter.reset(episode)
        self._target = _target_from_params((episode or {}).get("params"))
        self._pending = True
        self._prev_time_s = None

    def _choose(self, obs: dict) -> None:
        feats = param_residual.serve_features(obs, self._target)
        norm = self.feat_rms.normalize(feats[None, :].astype(np.float64))
        feats_t = torch.as_tensor(norm, dtype=torch.float32, device=self.device)
        mean, _logp, _value = self.model.act(feats_t, deterministic=True)
        u = np.tanh(mean.cpu().numpy()[0].astype(np.float64))
        self.last_params = param_residual.unsquash_to_params(u, self.param_specs)
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
