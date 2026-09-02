"""Per-serve *parameter* residual over himrobot.policies.ik_hitter.IKHitterPolicy.

One decision per episode instead of one per control step: at ball launch a
small MLP reads 13 serve features and emits K bounded scalars that retune
the IK hitter for that whole serve (K = 10 in the default v1 space, 20 in v2;
see PARAM_SPECS / PARAM_SPECS_V2). Why not per-step PPO:
per-step Gaussian noise perturbs all 150 control steps of a swing the hitter
already gets right ~47% of the time, so exploration itself collapses it to
~10% legal and most of the training budget is spent off-distribution. A
per-serve parameter vector keeps every rollout a *valid* swing, so the whole
budget measures real returns -- and one decision per episode makes it a plain
contextual bandit, which needs no GAE, no bootstrapping and no credit
assignment across time.

Pieces here: PARAM_SPECS/PARAM_SPECS_V2 (the bounded action spaces,
u = tanh(z)), serve_features (the 13-dim context), ParamActorCritic (zero-init
head, so an untrained net emits exactly the nominal = plain-hitter
parameters), and VecTTParam (himrobot.rl.env.VecTT stepped by a per-slot
hitter, one parameter draw per episode). Training lives in himrobot/rl/param_train.py.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from himrobot.sim import scene
from himrobot.sim.obs_perturb import ObsPerturber
from himrobot.policies.ik_hitter import (
    SERVE_PARAM_NOMINAL,
    SERVE_PARAM_NOMINAL_V2,
    IKHitterPolicy,
)
from himrobot.rl.demos import target_to_action
from himrobot.rl.env import MAX_EPISODE_STEPS, VecTT, map_action
from himrobot.rl.intercept import predict_intercept
from himrobot.rl.ppo import RunningMeanStd

# ---------------------------------------------------------------------------
# Action space. Each entry is (hitter attribute, kind, bound); the network's
# squashed output u = tanh(z) in [-1, 1] becomes
#     kind "add":  value = u * bound            (nominal 0.0   at u = 0)
#     kind "mul":  value = exp(u * ln(bound))   (nominal 1.0   at u = 0)
# so u = 0 is always the identity and every value is bounded by construction.
# "mul" is geometric on purpose: halving and doubling a scale are equally far
# from 1.0. Bounds are deliberately modest -- these retune a hitter that
# already works, they do not replace its plan:
#   x/y/z_hit_offset  +-8/10/10 cm around a hitting plane whose forward reach
#                     limit is only ~5 cm away (see ik_hitter's X_HIT comment)
#   t_hit_offset      +-40 ms, ~1.5x the measured PREDICTION_DELAY_S bias
#   swing_speed_scale 0.5x-2x the 1.0 m/s commanded-target speed cap
#   pitch/yaw_offset  +-0.35 rad (~20 deg) of paddle face tilt
#   aim_x/y_offset    +-35 cm on the opponent half (the evaluator's
#                     target_hit radius is 30 cm)
#   flight_time_scale 0.7x-1.43x the 0.5 s return flight time (return loft)
# ---------------------------------------------------------------------------
PARAM_SPECS: tuple[tuple[str, str, float], ...] = (
    ("x_hit_offset", "add", 0.08),
    ("y_hit_offset", "add", 0.10),
    ("z_hit_offset", "add", 0.10),
    ("t_hit_offset", "add", 0.04),
    ("swing_speed_scale", "mul", 2.0),
    ("pitch_offset", "add", 0.35),
    ("yaw_offset", "add", 0.35),
    ("aim_x_offset", "add", 0.35),
    ("aim_y_offset", "add", 0.35),
    ("flight_time_scale", "mul", 1.43),
)
PARAM_NAMES = tuple(name for name, _kind, _bound in PARAM_SPECS)
K_PARAMS = len(PARAM_SPECS)
assert set(PARAM_NAMES) <= set(SERVE_PARAM_NOMINAL), "PARAM_SPECS names must be hitter serve params"

# ---------------------------------------------------------------------------
# "hitter v2" action space. The ten v1 entries come first, identical in name,
# kind and bound, so a v1 checkpoint's stored specs still decode exactly as
# they do today and --resume-from a v1 run into a v2 run is meaningful
# coordinate for coordinate. The ten additions drive the swing library and the
# reach expansion in himrobot.policies.ik_hitter (see SERVE_PARAM_NOMINAL_V2):
#   gate_*              blend weight toward one non-drive swing template; only
#                       u > 0 has any effect (the blend rectifies), so u = 0 is
#                       exactly the drive template = the v1 hitter
#   x_hit_coarse        +-30 cm of hitting-plane depth; x_hit_offset keeps its
#                       exact v1 meaning as the +-8 cm fine trim on top of it
#   reach_expand_y/z    0..1 lerp of the Cartesian reach clamp toward the
#                       measured collision-free envelope; negatives clip to 0
#   normal_weight_scale trades blade-face accuracy against position accuracy in
#                       the DLS objective (down = position priority = more
#                       usable lateral reach near the workspace edge)
#   swing_window_scale  0.5x-2x on t_pre/speed_buffer/t_follow together (dwell)
#   swing_dir_pitch     +-0.6 rad on the swing direction, independent of face
# ---------------------------------------------------------------------------
PARAM_SPECS_V2: tuple[tuple[str, str, float], ...] = PARAM_SPECS + (
    ("gate_block", "add", 1.0),
    ("gate_lift", "add", 1.0),
    ("gate_punch", "add", 1.0),
    ("gate_crossbody", "add", 1.0),
    ("x_hit_coarse", "add", 0.30),
    ("reach_expand_y", "add", 1.0),
    ("reach_expand_z", "add", 1.0),
    ("normal_weight_scale", "mul", 16.0),
    ("swing_window_scale", "mul", 2.0),
    ("swing_dir_pitch", "add", 0.60),
)
PARAM_NAMES_V2 = tuple(name for name, _kind, _bound in PARAM_SPECS_V2)
K_PARAMS_V2 = len(PARAM_SPECS_V2)
assert set(PARAM_NAMES_V2) <= set(SERVE_PARAM_NOMINAL_V2), "PARAM_SPECS_V2 names must be hitter serve params"

# Selectable action spaces. DEFAULT_PARAM_SPACE stays "v1" so every existing
# recipe reproduces without a flag.
PARAM_SPACES: dict[str, tuple[tuple[str, str, float], ...]] = {
    "v1": PARAM_SPECS,
    "v2": PARAM_SPECS_V2,
}
DEFAULT_PARAM_SPACE = "v1"


def space_name_for(specs) -> str:
    """The registry key whose specs match ``specs``, or "custom"."""
    target = tuple(tuple(s) for s in specs)
    for name, known in PARAM_SPACES.items():
        if tuple(tuple(s) for s in known) == target:
            return name
    return "custom"


# Serve features: ball.position(3), ball.velocity(3), predicted intercept
# p_int(3), t_int(1), valid(1), aim target x/y(2). Read once, at the first
# observation of the episode.
FEATURE_DIM = 13
HIDDEN = 64


def unsquash_to_params(u: np.ndarray,
                       specs: tuple[tuple[str, str, float], ...] = PARAM_SPECS) -> dict[str, float]:
    """u in [-1, 1]^K -> the serve-param dict IKHitterPolicy.set_serve_params
    takes. u = 0 gives exactly the nominal values (SERVE_PARAM_NOMINAL, or
    SERVE_PARAM_NOMINAL_V2 under the v2 specs), i.e. the plain hitter.
    ``specs`` is overridable so a checkpoint is decoded with the ranges it was
    trained under, not whatever this module currently declares."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    assert u.shape == (len(specs),), u.shape
    out = {}
    for value, (name, kind, bound) in zip(u, specs):
        out[name] = float(value * bound) if kind == "add" else float(np.exp(value * np.log(bound)))
    return out


def serve_features(obs: dict, target: tuple[float, float]) -> np.ndarray:
    """The FEATURE_DIM-dim context the parameter net conditions on, built
    from the same obs dict himrobot.eval.evaluate and TableTennisEnv hand a Policy
    (so training and eval read identical features)."""
    ball_pos = np.asarray(obs["ball.position"], dtype=np.float64)
    ball_vel = np.asarray(obs["ball.velocity"], dtype=np.float64)
    p_int, t_int, valid = predict_intercept(ball_pos, ball_vel)
    feats = np.concatenate([
        ball_pos, ball_vel, p_int, [t_int], [1.0 if valid else 0.0],
        np.asarray(target, dtype=np.float64),
    ])
    assert feats.shape == (FEATURE_DIM,), feats.shape
    return feats.astype(np.float32)


class ParamActorCritic(nn.Module):
    """features -> (K pre-squash means, state value), plus a free
    state-independent log_std. The actor's last Linear is zero-initialized, so
    an untrained net's mean is exactly 0 -> u = tanh(0) = 0 -> nominal serve
    params -> the plain IK hitter, before any gradient step."""

    def __init__(self, feature_dim: int = FEATURE_DIM, k_params: int = K_PARAMS,
                 hidden: int = HIDDEN, log_std_init: float = -1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.k_params = k_params
        self.hidden = hidden
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, k_params),
        )
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((k_params,), log_std_init))
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(feats), self.critic(feats).squeeze(-1)

    @torch.no_grad()
    def act(self, feats: torch.Tensor,
            deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z, logp, value). z is the PRE-squash Gaussian sample --
        the trainer's action variable, so its log-prob needs no tanh Jacobian
        correction; u = tanh(z) is only applied when building the params."""
        mean, value = self.forward(feats)
        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=mean.device), value
        dist = Normal(mean, self.log_std.exp())
        z = dist.sample()
        return z, dist.log_prob(z).sum(-1), value

    def evaluate_actions(self, feats: torch.Tensor,
                         z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(feats)
        dist = Normal(mean, self.log_std.exp())
        return dist.log_prob(z).sum(-1), dist.entropy().sum(-1), value


class VecTTParam(VecTT):
    """VecTT whose 29-dim joint action comes from a per-slot IKHitterPolicy,
    retuned once per episode by ``decide_fn``.

    ``decide_fn(features) -> dict`` is called at every episode reset with that
    slot's serve_features and must return at least {"u": (K,) in [-1,1]}; the
    whole dict is echoed back on that episode's ``info["episode"]`` so a
    trainer can carry its own log-prob/value alongside. decide_fn=None means
    nominal params, i.e. the plain hitter.

    Reward and termination are VecTT's, verbatim -- _reward_and_done is given
    the hitter's own [-1,1]^29 action, matching VecTTResidual's convention
    that the smoothness/velocity terms see what was actually sent to the
    joints. step() takes no action argument: there is nothing per-step to
    decide.
    """

    def __init__(self, n_envs: int, seed: int = 0, params: dict | None = None,
                 decide_fn=None, randomize: bool = True,
                 vel_penalty: float = 0.0005, legal_bonus: float = 5.0,
                 param_specs=PARAM_SPECS, obs_noise: float = 0.0, joint_noise: float = 0.0,
                 obs_delay_steps: int = 0, **env_kwargs):
        assert not env_kwargs.get("intercept_features") and not env_kwargs.get("intercept_reward"), (
            "VecTTParam does not implement the intercept block: step() builds the flat obs without it "
            "and calls _reward_and_done without an intercept argument")
        self.decide_fn = decide_fn
        self.param_specs = tuple(tuple(spec) for spec in param_specs)
        self._hitters = [IKHitterPolicy() for _ in range(n_envs)]
        self._decisions: list[dict] = [{} for _ in range(n_envs)]
        self._features = np.zeros((n_envs, FEATURE_DIM), dtype=np.float32)
        # Sensor noise / latency on the obs dict the hitter and the feature
        # extractor see -- the same knobs himrobot.eval.evaluate applies at scoring
        # time. All off by default: no draw, the env's own dict is used.
        self._perturbers = [ObsPerturber(obs_noise, joint_noise, obs_delay_steps, seed=seed + i)
                            for i in range(n_envs)]
        super().__init__(n_envs=n_envs, seed=seed, params=params, randomize=randomize,
                         vel_penalty=vel_penalty, legal_bonus=legal_bonus,
                         obs_noise=obs_noise, joint_noise=joint_noise,
                         obs_delay_steps=obs_delay_steps, **env_kwargs)

    def _reset_one(self, i: int) -> np.ndarray:
        flat = super()._reset_one(i)
        env = self.envs[i]
        # Pure read of the state env.reset() just wrote, through this slot's perturber.
        obs = self._perturbers[i].reset(env._get_obs())
        self._hitters[i].reset({"serve": None, "params": env.params})
        feats = serve_features(obs, self._targets[i])
        self._features[i] = feats
        decision = dict(self.decide_fn(feats)) if self.decide_fn is not None else {}
        u = decision.get("u", np.zeros(len(self.param_specs)))
        self._hitters[i].set_serve_params(unsquash_to_params(u, self.param_specs))
        self._decisions[i] = decision
        return flat

    def step(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        obs_batch = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)
        reward_batch = np.zeros(self.n_envs, dtype=np.float64)
        done_batch = np.zeros(self.n_envs, dtype=bool)
        infos: list[dict] = [None] * self.n_envs

        for i in range(self.n_envs):
            raw = self._perturbers[i].step(self.envs[i]._get_obs())
            action = target_to_action(self._hitters[i].act(raw))
            obs, events = self.envs[i].step(map_action(action, self._jnt_range))

            reward, done = self._reward_and_done(i, obs, events, action)
            self._ep_len[i] += 1
            self._ep_return[i] += reward
            if self._ep_len[i] >= MAX_EPISODE_STEPS:
                done = True

            info = {"events": events}
            if done:
                info["episode"] = {
                    "return": float(self._ep_return[i]),
                    "length": int(self._ep_len[i]),
                    "contact": bool(self._contact_latched[i]),
                    "legal_return": bool(self._legal_latched[i]),
                    "features": self._features[i].copy(),
                    "decision": self._decisions[i],
                }
                delayed = self._reset_one(i)  # auto-reset: new serve, new parameter draw
            else:
                self._prev_action[i] = action
                flat = self._flat_obs(obs, action, self._targets[i], obs["time_s"])  # == flatten_obs at zero noise
                self._obs_fifo[i].append(flat)
                delayed = self._obs_fifo[i][0]

            obs_batch[i] = delayed
            reward_batch[i] = reward
            done_batch[i] = done
            infos[i] = info

        self._last_obs = obs_batch
        return obs_batch, reward_batch, done_batch, infos


def hitter_from_u(u: np.ndarray, specs: tuple[tuple[str, str, float], ...] = PARAM_SPECS,
                  **hitter_kwargs) -> IKHitterPolicy:
    """An IKHitterPolicy retuned by a squashed parameter vector -- the same
    mapping VecTTParam and ParamResidualPolicy use."""
    hitter = IKHitterPolicy(**hitter_kwargs)
    hitter.set_serve_params(unsquash_to_params(u, specs))
    return hitter


def save_checkpoint(path: str, model: ParamActorCritic, feat_rms: RunningMeanStd,
                    episodes: int, extra: dict | None = None,
                    param_specs=None) -> None:
    """Everything ParamResidualPolicy needs: weights, the feature-normalizer
    stats, and the parameter ranges this run was trained under. ``param_specs``
    must be the space the run actually used -- writing the module's current
    PARAM_SPECS unconditionally would silently mislabel every non-v1 run."""
    if param_specs is None:
        param_specs = PARAM_SPACES.get(DEFAULT_PARAM_SPACE, PARAM_SPECS)
    param_specs = tuple(tuple(spec) for spec in param_specs)
    assert len(param_specs) == model.k_params, (len(param_specs), model.k_params)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "feature_dim": model.feature_dim,
        "k_params": model.k_params,
        "hidden": model.hidden,
        "param_specs": [list(spec) for spec in param_specs],
        "param_names": [str(spec[0]) for spec in param_specs],
        "param_space": space_name_for(param_specs),
        "obs_rms_mean": feat_rms.mean,
        "obs_rms_var": feat_rms.var,
        "obs_rms_count": feat_rms.count,
        "episodes": int(episodes),
    }
    ckpt.update(extra or {})
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: torch.device) -> tuple[
        ParamActorCritic, RunningMeanStd, tuple[tuple[str, str, float], ...]]:
    """(model, feat_rms, specs) from a save_checkpoint() file."""
    # weights_only=False: our own checkpoints carry numpy arrays and plain
    # Python metadata; only ever load checkpoints this package produced.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ParamActorCritic(ckpt["feature_dim"], ckpt["k_params"],
                             hidden=ckpt.get("hidden", HIDDEN)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    feat_rms = RunningMeanStd((ckpt["feature_dim"],))
    feat_rms.mean = np.asarray(ckpt["obs_rms_mean"], dtype=np.float64)
    feat_rms.var = np.asarray(ckpt["obs_rms_var"], dtype=np.float64)
    feat_rms.count = ckpt["obs_rms_count"]
    specs = tuple((name, kind, float(bound))
                  for name, kind, bound in ckpt.get("param_specs", PARAM_SPECS))
    # param_space is metadata only -- the specs above are what actually decodes.
    model.param_space = str(ckpt.get("param_space", space_name_for(specs)))
    return model, feat_rms, specs


def default_target(params: dict | None = None) -> tuple[float, float]:
    p = {**scene.DEFAULT_PARAMS, **(params or {})}
    return scene.target_point(p["target_zone"], p["target_depth"])


__all__ = [
    "PARAM_SPECS", "PARAM_NAMES", "K_PARAMS", "FEATURE_DIM",
    "PARAM_SPECS_V2", "PARAM_NAMES_V2", "K_PARAMS_V2", "PARAM_SPACES",
    "DEFAULT_PARAM_SPACE", "space_name_for",
    "ParamActorCritic", "VecTTParam", "serve_features", "unsquash_to_params",
    "hitter_from_u", "default_target", "save_checkpoint", "load_checkpoint",
]
