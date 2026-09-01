"""VecTTBalance: tt_sim.rl.env.VecTT on the FREE-BASE scene
(TableTennisEnv(..., fixed_base=False)) with a shipped v3 hitting policy
(tt_sim.policies.param_residual_np.NumpyParamResidualPolicy) riding along
on waist+arm, exactly like tt_sim.rl.residual.VecTTResidual rides an
IKHitterPolicy -- see that module's docstring for the __init__/_reset_one
shadowing trick this class reuses verbatim.

Unlike VecTTResidual (whose RL action corrects the expert's own 29-dim
action), here the RL policy owns a DIFFERENT slice of the joints: the 12
leg joints, which the expert always drives to tt_sim.scene.STAND_POSE (its
action's leg slots are inert -- ACTION_SCALE zeroes them, see env.py). So
the combined 29-dim joint target is built directly (not through
map_action, whose leg scale is 0.0) as

    STAND_POSE + concat(a_rl * BALANCE_SCALE, a_exp[12:] * 3.0)

clipped to each joint's mujoco range. a_rl=0 reproduces
STAND_POSE + concat(0, a_exp[12:] * 3.0) -- bit-identical to running the
expert alone on the free base with legs pinned at STAND_POSE, since that
is exactly what happens today (the welded scene has no legs to move, and
the free-base probe never moved them either). This is the parity anchor a
freshly-constructed (zero-initialized final actor layer) balance policy
starts at -- see tt_sim/tests/test_balance_env.py's parity test.

Observation (BALANCE_OBS_DIM=111, obs_version=1): env.flatten_obs's 99,
then a 12-dim base block -- gravity unit vector in the pelvis frame (3),
base.position - scene.BASE_NOMINAL_POS (3), base.linear_velocity (3),
base.angular_velocity (3). The gravity vector is world [0,0,-1] rotated by
the INVERSE of base.orientation (mujoco wxyz): mju_negQuat then
mju_rotVecQuat, the same "rotate a world vector into the pelvis frame"
convention tt_sim.policies.ik_hitter.to_shadow_frame uses (there via
R_base^T; here via the conjugate quaternion -- both are the inverse
rotation for a unit quaternion).

Reward: VecTT._reward_and_done, called with the COMBINED 29-dim action
(matching VecTT's own convention that its smoothness/velocity penalty
terms see the action actually sent to the joints -- same choice
VecTTResidual makes), plus per step:
  +alive_bonus                         (default 0.05)
  -tilt_penalty * tilt_rad**2          (default 0.5)
and, the step events["fell"] FIRST latches True: -fall_penalty (default
10.0) and done=True. tilt is the angle between the body-frame gravity
vector and [0,0,-1] -- 0 when upright -- computed as
arccos(clip(-gravity_body[2], -1, 1)).
"""
from __future__ import annotations

import os

import numpy as np

import mujoco

from tt_sim import scene
from tt_sim.policies.param_residual_np import NumpyParamResidualPolicy
from tt_sim.rl.demos import target_to_action
from tt_sim.rl.env import ACT_DIM, MAX_EPISODE_STEPS, OBS_DIM, VecTT, _joint_range, flatten_obs

OBS_VERSION = 1
BALANCE_ACT_DIM = 12  # legs only, JOINT_ORDER slots 0-11
BASE_BLOCK_DIM = 12  # gravity(3) + base.position-nominal(3) + base.linear_velocity(3) + base.angular_velocity(3)
BALANCE_OBS_DIM = OBS_DIM + BASE_BLOCK_DIM  # 111

BALANCE_SCALE = 0.15  # rad, leg action scale
_ARM_WAIST_SCALE = 3.0  # rad -- identical to env.map_action's ACTION_SCALE on non-leg dims

DEFAULT_TILT_PENALTY = 0.5
DEFAULT_ALIVE_BONUS = 0.05
DEFAULT_FALL_PENALTY = 10.0

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_EXPERT_NPZ = os.path.join(_REPO, "assets", "policies", "param_residual_v3_s0final.npz")


def body_frame_gravity(base_orientation: np.ndarray) -> np.ndarray:
    """World [0, 0, -1] rotated into the pelvis frame: rotate by the
    INVERSE of base_orientation (mujoco wxyz). Unit norm by construction
    (rotating a unit vector by a rotation)."""
    quat = np.asarray(base_orientation, dtype=np.float64)
    inv_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_negQuat(inv_quat, quat)
    world_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gravity_body = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(gravity_body, world_down, inv_quat)
    return gravity_body


def tilt_angle(gravity_body: np.ndarray) -> float:
    """Angle (rad) between the body-frame gravity vector and [0, 0, -1] --
    0 when upright, growing as the pelvis tips over. arccos of the -z
    component (clipped to the valid domain against float roundoff)."""
    return float(np.arccos(np.clip(-float(gravity_body[2]), -1.0, 1.0)))


def flatten_balance_obs(base_obs: np.ndarray, obs: dict) -> np.ndarray:
    """obs_version=1 layout: env.flatten_obs's 99-dim row, then the 12-dim
    base block described in the module docstring. ``obs`` is the raw
    TableTennisEnv(fixed_base=False) obs dict (carries the base.* keys)."""
    gravity_body = body_frame_gravity(obs["base.orientation"])
    base_pos_delta = np.asarray(obs["base.position"], dtype=np.float64) - scene.BASE_NOMINAL_POS
    lin_vel = np.asarray(obs["base.linear_velocity"], dtype=np.float64)
    ang_vel = np.asarray(obs["base.angular_velocity"], dtype=np.float64)
    block = np.concatenate([gravity_body, base_pos_delta, lin_vel, ang_vel]).astype(np.float32)
    return np.concatenate([np.asarray(base_obs, dtype=np.float32), block]).astype(np.float32)


def combined_action(a_rl: np.ndarray, a_exp: np.ndarray) -> np.ndarray:
    """The normalized [-1,1]^29 action actually sent to the joints:
    a_rl on the 12 leg slots (units of BALANCE_SCALE), the expert's own
    action on waist+arm (units of 3.0 rad, as in env.ACTION_SCALE). This
    -- not the radian target -- is what the prev_action obs slot, the
    smoothness penalty and _prev_action bookkeeping see, matching VecTT's
    and VecTTResidual's convention."""
    return np.concatenate([np.asarray(a_rl, dtype=np.float64),
                           np.asarray(a_exp, dtype=np.float64)[12:]])


def joint_target(a_comb: np.ndarray, jnt_range: np.ndarray) -> np.ndarray:
    """STAND_POSE + concat(a_comb[:12] * BALANCE_SCALE, a_comb[12:] * 3.0),
    clipped to each joint's mujoco range. The arm/waist arithmetic is
    env.map_action's own STAND_POSE + action * ACTION_SCALE, bit for bit."""
    a_comb = np.asarray(a_comb, dtype=np.float64)
    delta = np.concatenate([a_comb[:BALANCE_ACT_DIM] * BALANCE_SCALE,
                            a_comb[BALANCE_ACT_DIM:] * _ARM_WAIST_SCALE])
    target = scene.STAND_POSE + delta
    return np.clip(target, jnt_range[:, 0], jnt_range[:, 1])


def combined_joint_target(a_rl: np.ndarray, a_exp: np.ndarray, jnt_range: np.ndarray) -> np.ndarray:
    """joint_target(combined_action(a_rl, a_exp)) -- see module docstring.
    a_rl=0 reproduces STAND_POSE + a_exp[12:] * 3.0 on the arm/waist dims
    exactly (the parity anchor)."""
    return joint_target(combined_action(a_rl, a_exp), jnt_range)


class VecTTBalance(VecTT):
    """See module docstring. fixed_base is NOT a constructor kwarg -- every
    env this class builds is TableTennisEnv(..., fixed_base=False). obs_dim
    is BALANCE_OBS_DIM (111); act_dim (the RL-facing a_rl) is
    BALANCE_ACT_DIM (12, legs only)."""

    obs_version = OBS_VERSION

    def __init__(self, n_envs: int, seed: int = 0, params: dict | None = None,
                 obs_delay_steps: int = 0, randomize: bool = True,
                 expert_npz: str = DEFAULT_EXPERT_NPZ, vel_penalty: float = 0.0005,
                 legal_bonus: float = 5.0, obs_noise: float = 0.0, joint_noise: float = 0.0,
                 tilt_penalty: float = DEFAULT_TILT_PENALTY, alive_bonus: float = DEFAULT_ALIVE_BONUS,
                 fall_penalty: float = DEFAULT_FALL_PENALTY):
        self.expert_npz = expert_npz
        self.tilt_penalty = float(tilt_penalty)
        self.alive_bonus = float(alive_bonus)
        self.fall_penalty = float(fall_penalty)
        # One expert per env slot, reset alongside that slot's TableTennisEnv
        # (see _reset_one) -- same pattern as VecTTResidual._hitters.
        self._experts = [NumpyParamResidualPolicy(expert_npz) for _ in range(n_envs)]
        self._expert_action = np.zeros((n_envs, ACT_DIM), dtype=np.float64)
        # Latches the FIRST step a fall is observed this episode, so the
        # -fall_penalty/done fires exactly once per fall (mirrors
        # VecTT._contact_latched/_legal_latched's own latch convention).
        self._fell_latched = np.zeros(n_envs, dtype=bool)

        # Shadow _reset_one with VecTT's own (unwidened, 99-dim, fixed_base
        # default) version for just this call, exactly like VecTTResidual's
        # __init__ -- see that module's docstring for why.
        self._reset_one = lambda i: VecTT._reset_one(self, i)
        try:
            super().__init__(n_envs=n_envs, seed=seed, params=params,
                              obs_delay_steps=obs_delay_steps, randomize=randomize,
                              vel_penalty=vel_penalty, legal_bonus=legal_bonus,
                              obs_noise=obs_noise, joint_noise=joint_noise)
        finally:
            del self._reset_one  # restore the class-level (widening) override below

        # VecTT.__init__ built fixed_base=True envs (see VecTT._reset_one's
        # own super().__init__ call inside the shadow above); replace them
        # with free-base envs of the same seed/params, then re-run every
        # env's reset+widen through the real (free-base-aware) _reset_one.
        self.envs = [scene.TableTennisEnv(seed=seed + i, params=self.params, fixed_base=False)
                     for i in range(n_envs)]
        self._jnt_range = _joint_range(self.envs[0].model)  # same JOINT_ORDER joints; recomputed defensively

        self.obs_dim = BALANCE_OBS_DIM
        self.act_dim = BALANCE_ACT_DIM
        widened = np.zeros((n_envs, BALANCE_OBS_DIM), dtype=np.float32)
        for i in range(n_envs):
            widened[i] = self._reset_one(i)
        self._last_obs = widened

    # -- reset -----------------------------------------------------------
    def _widen_reset(self, i: int) -> np.ndarray:
        """Reset env i's expert (and fall latch) and build env i's 111-dim
        observation from its already-reset state, without a second
        env.reset() -- see module docstring / VecTTResidual's own
        _widen_reset for why this is safe (a pure read of state reset()
        already wrote)."""
        env = self.envs[i]
        obs = env._get_obs()
        self._experts[i].reset({"serve": None, "params": env.params})
        expert_target = self._experts[i].act(obs)
        self._expert_action[i] = target_to_action(expert_target)
        self._fell_latched[i] = False

        base = flatten_obs(obs, self._prev_action[i], self._targets[i], obs["time_s"])
        flat = flatten_balance_obs(base, obs)
        self._obs_fifo[i].clear()
        for _ in range(self.obs_delay_steps + 1):
            self._obs_fifo[i].append(flat)
        return self._obs_fifo[i][0]

    def _reset_one(self, i: int) -> np.ndarray:
        VecTT._reset_one(self, i)  # the real env.reset() (free-base) + VecTT's own per-env bookkeeping
        return self._widen_reset(i)

    # -- step --------------------------------------------------------------
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """``actions`` is a_rl, the RL policy's [-1,1]^12 leg action --
        combined with each env's cached expert action (see class
        docstring) before being sent to the joints."""
        actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        assert actions.shape == (self.n_envs, BALANCE_ACT_DIM), actions.shape

        obs_batch = np.zeros((self.n_envs, BALANCE_OBS_DIM), dtype=np.float32)
        reward_batch = np.zeros(self.n_envs, dtype=np.float64)
        done_batch = np.zeros(self.n_envs, dtype=bool)
        infos: list[dict] = [None] * self.n_envs

        for i in range(self.n_envs):
            combined = combined_action(actions[i], self._expert_action[i])
            obs, events = self.envs[i].step(joint_target(combined, self._jnt_range))

            reward, done = self._reward_and_done(i, obs, events, combined)

            gravity_body = body_frame_gravity(obs["base.orientation"])
            tilt = tilt_angle(gravity_body)
            reward += self.alive_bonus
            reward -= self.tilt_penalty * tilt ** 2

            if events.get("fell") and not self._fell_latched[i]:
                reward -= self.fall_penalty
                done = True
                self._fell_latched[i] = True

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
                    "fell": bool(events.get("fell", False)),
                }
                delayed = self._reset_one(i)  # auto-reset: also resets expert[i]/fall latch + recomputes expert_action[i]
            else:
                expert_target = self._experts[i].act(obs)
                self._expert_action[i] = target_to_action(expert_target)
                base = flatten_obs(obs, combined, self._targets[i], obs["time_s"])
                flat = flatten_balance_obs(base, obs)
                self._obs_fifo[i].append(flat)
                delayed = self._obs_fifo[i][0]

            obs_batch[i] = delayed
            reward_batch[i] = reward
            done_batch[i] = done
            infos[i] = info
            self._prev_action[i] = combined

        self._last_obs = obs_batch
        return obs_batch, reward_batch, done_batch, infos
