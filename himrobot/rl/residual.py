"""VecTTResidual: himrobot.rl.env.VecTT with a per-env IKHitterPolicy
"expert" riding along, whose action the RL policy learns a small residual
correction on top of, instead of learning table tennis from scratch.

Each of the n_envs slots owns its own himrobot.policies.ik_hitter.IKHitterPolicy
instance (reset alongside that slot's TableTennisEnv on every episode
reset -- auto-reset included -- with the same {"serve", "params"} shape
himrobot/rl/demos.py and himrobot.eval.evaluate pass to Policy.reset). At the
observation for control step t, the expert is run on the raw (undelayed)
TableTennisEnv obs to get its absolute joint-position target, which
himrobot.rl.demos.target_to_action (the documented inverse of
himrobot.rl.env.map_action) turns into an expert action a_exp in the same
[-1,1]^29 space step() expects from the RL policy. This a_exp is:

  1. appended to the flat 99-dim env.flatten_obs row to build this env's
     observation, RESIDUAL_OBS_DIM=128 wide (obs_version=2 -- see
     flatten_residual_obs). The RL policy sees the expert's intended
     action as part of its input.
  2. combined with the RL policy's own action a_rl (also in [-1,1]^29,
     step()'s argument) as the actual action sent to
     himrobot.rl.env.map_action: a = clip(a_exp + residual_scale * a_rl,
     -1, 1) -- so a_rl=0 (e.g. a freshly-constructed, zero-initialized
     actor -- see himrobot/rl/ppo.py's --residual init) reproduces the
     expert's own action exactly, and training only ever has to learn a
     *correction*, not the whole swing.

reward/termination is untouched: _reward_and_done is inherited from VecTT
verbatim (called with the *combined* action a, matching VecTT's own
convention that the smoothness/velocity penalty terms see the action
actually sent to the joints).

Implementation note (why __init__/_reset_one are written this way, not as
a straight override): VecTT.__init__ preallocates self._last_obs at
env.py's own (unwidened) obs_dim before its per-env reset loop, which
can't hold our 128-dim rows, and duplicating VecTT.__init__'s full body
here would silently drift from it as env.py evolves (extra latched flags,
reward options, etc.). Instead, __init__ runs the REAL VecTT.__init__
(via super()) with _reset_one momentarily shadowed by VecTT's own
(unwidened) version so its preallocation doesn't crash, then widens every
env's already-reset observation via TableTennisEnv._get_obs() -- a pure
read of the state env.reset() already wrote, so no second reset / rng
draw is spent and parity with a direct IKHitterPolicy rollout holds
bit-for-bit. _reset_one's permanent (class-level) override reuses this
same widen step after calling the real VecTT._reset_one for a later
auto-reset mid-training, so both paths -- initial construction and every
subsequent episode boundary -- share one reset+widen code path.
"""
from __future__ import annotations

import numpy as np

from himrobot.policies.ik_hitter import IKHitterPolicy
from himrobot.rl.demos import target_to_action
from himrobot.rl.env import ACT_DIM, MAX_EPISODE_STEPS, OBS_DIM, VecTT, flatten_obs, map_action

OBS_VERSION = 2
RESIDUAL_OBS_DIM = OBS_DIM + ACT_DIM  # 128: env.flatten_obs's 99 + the expert's own [-1,1]^29 action
DEFAULT_RESIDUAL_SCALE = 0.3


def flatten_residual_obs(base_obs: np.ndarray, expert_action: np.ndarray) -> np.ndarray:
    """obs_version=2 layout: env.flatten_obs's 99-dim row, then the
    expert's [-1,1]^29 action for that SAME underlying obs, appended
    last. Fixed, versioned order -- see module docstring."""
    return np.concatenate([
        np.asarray(base_obs, dtype=np.float32),
        np.asarray(expert_action, dtype=np.float32),
    ]).astype(np.float32)


class VecTTResidual(VecTT):
    """See module docstring. obs_dim is RESIDUAL_OBS_DIM (128); act_dim
    (the RL-facing a_rl) is unchanged at ACT_DIM (29). Only plain
    (intercept_features=False, intercept_reward=0.0) VecTT behaviour is
    supported -- this wrapper always constructs the base VecTT with its
    defaults, so obs/reward stay exactly himrobot.rl.env's un-augmented
    99-dim/reward-unchanged shape before the expert action is appended."""

    obs_version = OBS_VERSION

    def __init__(self, n_envs: int, seed: int = 0, params: dict | None = None,
                 obs_delay_steps: int = 0, randomize: bool = True,
                 residual_scale: float = DEFAULT_RESIDUAL_SCALE, vel_penalty: float = 0.0005,
                 legal_bonus: float = 5.0, obs_noise: float = 0.0, joint_noise: float = 0.0):
        # obs_noise/joint_noise: constructor passthrough only (stored on self
        # via super().__init__ below, so ppo.py can build both VecTT and
        # VecTTResidual with the same kwargs and log what was requested to
        # the checkpoint). NOT applied: _widen_reset/step below build obs via
        # flatten_obs directly, not self._flat_obs, so the actual noise
        # injection in VecTT._add_obs_noise never runs on this path. ppo.py's
        # train() rejects --residual combined with either != 0 accordingly.
        self.residual_scale = float(residual_scale)
        # One IKHitterPolicy per env slot, reset alongside that slot's
        # TableTennisEnv (see _reset_one); _expert_action[i] always holds
        # a_exp for env i's CURRENT (not-yet-acted-on) observation.
        self._hitters = [IKHitterPolicy() for _ in range(n_envs)]
        self._expert_action = np.zeros((n_envs, ACT_DIM), dtype=np.float64)

        # Shadow _reset_one with VecTT's own (unwidened, 99-dim) version
        # for just this call -- see module docstring's implementation note.
        self._reset_one = lambda i: VecTT._reset_one(self, i)
        try:
            super().__init__(n_envs=n_envs, seed=seed, params=params,
                              obs_delay_steps=obs_delay_steps, randomize=randomize,
                              vel_penalty=vel_penalty, legal_bonus=legal_bonus,
                              obs_noise=obs_noise, joint_noise=joint_noise)
        finally:
            del self._reset_one  # restore the class-level (widening) override below

        self.obs_dim = RESIDUAL_OBS_DIM
        widened = np.zeros((n_envs, RESIDUAL_OBS_DIM), dtype=np.float32)
        for i in range(n_envs):
            widened[i] = self._widen_reset(i)
        self._last_obs = widened

    # -- reset -----------------------------------------------------------
    def _widen_reset(self, i: int) -> np.ndarray:
        """Reset env i's hitter and append its expert action to env i's
        already-reset observation, without a second env.reset() (see
        module docstring). Used by both __init__ and _reset_one below."""
        env = self.envs[i]
        obs = env._get_obs()  # pure read of the mujoco state env.reset() just wrote
        self._hitters[i].reset({"serve": None, "params": env.params})
        expert_target = self._hitters[i].act(obs)
        self._expert_action[i] = target_to_action(expert_target)

        base = flatten_obs(obs, self._prev_action[i], self._targets[i], obs["time_s"])
        flat = flatten_residual_obs(base, self._expert_action[i])
        self._obs_fifo[i].clear()
        for _ in range(self.obs_delay_steps + 1):
            self._obs_fifo[i].append(flat)
        return self._obs_fifo[i][0]

    def _reset_one(self, i: int) -> np.ndarray:
        VecTT._reset_one(self, i)  # the real env.reset() + VecTT's own per-env bookkeeping
        return self._widen_reset(i)

    # -- step --------------------------------------------------------------
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """``actions`` is a_rl, the RL policy's own [-1,1]^29 action --
        combined with each env's cached expert action (see class
        docstring) before being sent to himrobot.rl.env.map_action."""
        actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        assert actions.shape == (self.n_envs, ACT_DIM), actions.shape

        combined = np.clip(self._expert_action + self.residual_scale * actions, -1.0, 1.0)

        obs_batch = np.zeros((self.n_envs, RESIDUAL_OBS_DIM), dtype=np.float32)
        reward_batch = np.zeros(self.n_envs, dtype=np.float64)
        done_batch = np.zeros(self.n_envs, dtype=bool)
        infos: list[dict] = [None] * self.n_envs

        for i in range(self.n_envs):
            joint_target = map_action(combined[i], self._jnt_range)
            obs, events = self.envs[i].step(joint_target)

            reward, done = self._reward_and_done(i, obs, events, combined[i])
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
                }
                delayed = self._reset_one(i)  # auto-reset: also resets hitter[i] + recomputes expert_action[i]
            else:
                expert_target = self._hitters[i].act(obs)
                self._expert_action[i] = target_to_action(expert_target)
                base = flatten_obs(obs, combined[i], self._targets[i], obs["time_s"])
                flat = flatten_residual_obs(base, self._expert_action[i])
                self._obs_fifo[i].append(flat)
                delayed = self._obs_fifo[i][0]

            obs_batch[i] = delayed
            reward_batch[i] = reward
            done_batch[i] = done
            infos[i] = info
            self._prev_action[i] = combined[i]

        self._last_obs = obs_batch
        return obs_batch, reward_batch, done_batch, infos
