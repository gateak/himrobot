"""VecTT: a vectorized Gymnasium-style wrapper over N independent
tt_sim.scene.TableTennisEnv instances, plus the flat-obs / action-mapping /
reward/termination logic shared by tt_sim/rl/ppo.py (training) and
tt_sim/rl/policy.py (the deterministic eval-time PPOPolicy).

Vectorization strategy: MuJoCo's CPU stepping isn't batchable across
instances without mjx (out of scope here), so this runs N independent
TableTennisEnv objects in a single process and loops mj_step per env, but
batches every RL-facing array op (obs stacking, reward, GAE, network
forward/backward) with numpy/torch across the N envs -- "single process is
fine; vectorize with numpy" per the task spec. See ppo.py's reported
steps/s for the resulting throughput.

Episode = one serve. An episode ends (the ball is "dead") on: a second
bounce on the robot's own side, an opponent-side bounce (the return has
already landed -- nothing more to reward), a net hit, an "out", or
MAX_EPISODE_STEPS (150 control steps @ 50 Hz = 3 s, matching
tt_sim.evaluate's EPISODE_S). TableTennisEnv.step()'s ``events`` dict is a
flat *per-episode* accumulator (fields latch True and stay True, they are
not per-step deltas -- see tt_sim/scene.py's TableTennisEnv docstring), so
this wrapper does its own edge-detection on top of it for both reward
triggers (contact/legal-return should fire once) and own-side-bounce
counting (events only exposes a bool, not a count; a second own-side bounce
is detected by watching events['landing_point'] change to a new x<0 point).
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from tt_sim import scene
from tt_sim.rl.intercept import predict_intercept
from tt_sim.scene import TableTennisEnv

import mujoco

# ---------------------------------------------------------------------------
# Action mapping: 29-dim action in [-1, 1] -> joint targets = STAND_POSE +
# action * scale. Per the task spec: arms/waist scale 3.0 rad, legs scale
# 0.0 (frozen -- the base is fixed, per scene.py's fixed-pelvis design, so
# there is nothing for the legs to do; freezing them also shrinks the
# effective action space PPO has to explore). JOINT_ORDER is
# [left leg(6), right leg(6), waist(3), left arm(7), right arm(7)].
# ---------------------------------------------------------------------------
_N_LEG = 12
_N_WAIST = 3
_N_ARM = 14
assert _N_LEG + _N_WAIST + _N_ARM == scene.NUM_JOINTS
ACTION_SCALE = np.array([0.0] * _N_LEG + [3.0] * _N_WAIST + [3.0] * _N_ARM, dtype=np.float64)

# Flat obs layout (99-dim): joint_pos-default[29], joint_vel*0.1[29],
# ball.position[3], ball.velocity*0.2[3], paddle.position[3], target[2],
# phase[1], prev_action[29].
OBS_DIM = scene.NUM_JOINTS * 2 + 3 + 3 + 3 + 2 + 1 + scene.NUM_JOINTS  # 99
ACT_DIM = scene.NUM_JOINTS  # 29

# Slices into the base 99-dim block for obs_noise/joint_noise (see VecTT.__init__)
# -- fixed regardless of which additive feature blocks get appended after it.
_JOINT_POS_SLICE = slice(0, scene.NUM_JOINTS)
_JOINT_VEL_SLICE = slice(scene.NUM_JOINTS, scene.NUM_JOINTS * 2)
_BALL_POS_SLICE = slice(scene.NUM_JOINTS * 2, scene.NUM_JOINTS * 2 + 3)
_BALL_VEL_SLICE = slice(scene.NUM_JOINTS * 2 + 3, scene.NUM_JOINTS * 2 + 6)

# Optional additive obs blocks, appended in this fixed order after the base
# 99: intercept features (this module, intercept_features=True) then expert
# features (tt_sim.rl.ppo's BC-expert mode, appended in its own additive
# change -- not implemented here, this constant only reserves the slot so
# obs_dim() stays correct once it lands).
#
# Intercept block (8 dims), from tt_sim.rl.intercept.predict_intercept(ball
# .position, ball.velocity) each step: [p_int(3), t_int(1),
# p_int-paddle.position(3), valid(1)]. p_int/t_int are zeroed when
# valid==0 (no predicted crossing within the horizon).
INTERCEPT_FEATURES_DIM = 8
EXPERT_FEATURES_DIM = 29  # reserved for Task R's expert-action block

MAX_EPISODE_STEPS = 150  # 3s @ 50Hz, matches tt_sim.evaluate.EPISODE_CONTROL_STEPS
INTERCEPT_BONUS_RADIUS_M = 0.05  # "paddle within 5cm of p_int" bonus threshold
# Hand copy of tt_sim.evaluate.TARGET_HIT_RADIUS_M, kept here so the RL
# package does not import the evaluator CLI; test_rl_env pins the two equal.
# The optional target_hit_bonus (and param_train.deterministic_eval) use it.
TARGET_HIT_RADIUS_M = 0.30


def obs_dim(intercept_features: bool = False, expert_features: bool = False) -> int:
    """Total flat-obs width for a given additive-feature combination, so
    ppo.py (or any other consumer) can size its network without importing
    VecTT itself. Order matches what VecTT actually concatenates: base 99,
    then intercept 8 (if intercept_features), then expert 29 (if
    expert_features)."""
    dim = OBS_DIM
    if intercept_features:
        dim += INTERCEPT_FEATURES_DIM
    if expert_features:
        dim += EXPERT_FEATURES_DIM
    return dim


def _joint_range(model: mujoco.MjModel) -> np.ndarray:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in scene.JOINT_ORDER]
    return np.array([model.jnt_range[j] for j in ids], dtype=np.float64)


def map_action(action: np.ndarray, jnt_range: np.ndarray) -> np.ndarray:
    """action in [-1,1]^29 -> absolute joint-position targets (radians),
    clamped to each joint's own range (STAND_POSE + action*scale can
    otherwise walk an arm joint's target past its physical limit)."""
    target = scene.STAND_POSE + action * ACTION_SCALE
    return np.clip(target, jnt_range[:, 0], jnt_range[:, 1])


def flatten_obs(obs: dict, prev_action: np.ndarray, target: tuple[float, float],
                time_s: float) -> np.ndarray:
    """Build the flat 99-dim policy-facing observation from a
    TableTennisEnv-shaped obs dict (this is exactly the dict shape both
    TableTennisEnv.step/reset and tt_sim.evaluate hand to a Policy)."""
    jp = np.asarray(obs["robot.joint_position"], dtype=np.float64) - scene.STAND_POSE
    jv = np.asarray(obs["robot.joint_velocity"], dtype=np.float64) * 0.1
    bp = np.asarray(obs["ball.position"], dtype=np.float64)
    bv = np.asarray(obs["ball.velocity"], dtype=np.float64) * 0.2
    pp = np.asarray(obs["paddle.position"], dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    phase = np.array([time_s], dtype=np.float64)
    pa = np.asarray(prev_action, dtype=np.float64)
    return np.concatenate([jp, jv, bp, bv, pp, tgt, phase, pa]).astype(np.float32)


class VecTT:
    """N independent TableTennisEnv instances, stepped together.

    step(actions) auto-resets any env whose episode just ended (standard
    vec-env convention): the returned obs row for a just-finished env is
    already the *next* episode's first observation, and per-episode
    summary stats (return, length, contact, legal_return) for the episode
    that just ended are attached at info[i]["episode"].
    """

    def __init__(self, n_envs: int, seed: int = 0, params: dict | None = None,
                 obs_delay_steps: int = 0, randomize: bool = True,
                 intercept_features: bool = False, intercept_reward: float = 0.0,
                 vel_penalty: float = 0.0005, legal_bonus: float = 5.0,
                 obs_noise: float = 0.0, joint_noise: float = 0.0,
                 aim_bonus: float = 3.0, aim_sigma: float = 0.5,
                 target_hit_bonus: float = 0.0, depth_bonus: float = 0.0):
        assert n_envs >= 1
        assert obs_delay_steps >= 0
        assert obs_noise >= 0.0 and joint_noise >= 0.0
        self.n_envs = n_envs
        self.params = dict(params or {})
        self.obs_delay_steps = obs_delay_steps
        self.randomize = randomize
        # obs_noise/joint_noise default to 0.0, which skips the RNG draw
        # entirely in _add_obs_noise -- a seeded rollout is then bit-for-bit
        # identical to before these knobs existed. Sensor noise only, applied
        # to the flat obs the policy reads (see _flat_obs/_add_obs_noise);
        # _reward_and_done always sees the raw, un-noised TableTennisEnv obs.
        self.obs_noise = float(obs_noise)
        self.joint_noise = float(joint_noise)
        # Separate rng (offset well clear of _fixed_serve_rng's seed+999_983
        # below) so enabling noise never perturbs the fixed-serve draw stream.
        self._noise_rng = np.random.default_rng(seed + 4_242_017)
        # intercept_features/intercept_reward default to False/0.0, which
        # keeps obs_dim==OBS_DIM and _reward_and_done's added term a no-op --
        # existing (pre-Task-F) behaviour is exactly preserved unless a
        # caller opts in. See obs_dim()'s docstring for the appended-block
        # layout and intercept.predict_intercept for what p_int/t_int mean.
        self.intercept_features = intercept_features
        self.intercept_reward = float(intercept_reward)
        # Reward knobs. The default vel_penalty dominates the return on a fast swing (~-2.5/episode),
        # so PPO learns to swing slower; the v2 box runs use 1e-5 with legal_bonus 10.
        self.vel_penalty = float(vel_penalty)
        self.legal_bonus = float(legal_bonus)
        # Aim knobs, defaulting to the coefficients _reward_and_done used to
        # hard-code: aim_bonus 3.0 / aim_sigma 0.5 reproduce the old term
        # exactly, and target_hit_bonus 0.0 adds nothing.
        self.aim_bonus = float(aim_bonus)
        self.aim_sigma = float(aim_sigma)
        self.target_hit_bonus = float(target_hit_bonus)
        # Depth-shortfall penalty, default 0.0 = today's reward bit-for-bit.
        # Measured: every legal return lands ~0.28 m short of the target, a bias
        # 1.2-1.5x its own scatter that the aim knobs move by 3% of the requested
        # amount because the swing-speed cap binds on 400/400 trials. This term's
        # gradient points at swing_speed_scale, the one knob with measured
        # authority over the bias (work/tt/design/hitter_v3_reach.md 2.3).
        self.depth_bonus = float(depth_bonus)
        self.obs_dim = obs_dim(intercept_features=self.intercept_features)
        self.act_dim = ACT_DIM

        self.envs = [TableTennisEnv(seed=seed + i, params=self.params) for i in range(n_envs)]
        # Fixed-serve rngs used only when randomize=False, so a "no
        # randomization" env still resets to a reproducible (but non-random
        # across calls) serve each episode -- useful for debugging/tests.
        self._fixed_serve_rng = [np.random.default_rng(seed + 999_983 + i) for i in range(n_envs)]

        self._jnt_range = _joint_range(self.envs[0].model)
        self._targets: list[tuple[float, float] | None] = [None] * n_envs
        self._prev_action = np.zeros((n_envs, ACT_DIM), dtype=np.float64)
        self._ep_len = np.zeros(n_envs, dtype=np.int64)
        self._ep_return = np.zeros(n_envs, dtype=np.float64)
        self._contact_latched = np.zeros(n_envs, dtype=bool)
        self._legal_latched = np.zeros(n_envs, dtype=bool)
        self._fail_latched = np.zeros(n_envs, dtype=bool)
        self._own_bounce_count = np.zeros(n_envs, dtype=np.int64)
        self._prev_landing: list[tuple | None] = [None] * n_envs
        self._intercept_bonus_latched = np.zeros(n_envs, dtype=bool)
        self._obs_fifo = [deque(maxlen=obs_delay_steps + 1) for _ in range(n_envs)]

        self._last_obs = np.zeros((n_envs, self.obs_dim), dtype=np.float32)
        for i in range(n_envs):
            self._last_obs[i] = self._reset_one(i)

    # -- reset -----------------------------------------------------------
    def _reset_one(self, i: int) -> np.ndarray:
        env = self.envs[i]
        serve = None if self.randomize else scene.sample_serve(self._fixed_serve_rng[i], env.params)
        obs = env.reset(serve=serve)  # serve=None -> TableTennisEnv samples one via its own rng

        target_zone = env.params.get("target_zone", scene.DEFAULT_PARAMS["target_zone"])
        target_depth = env.params.get("target_depth", scene.DEFAULT_PARAMS["target_depth"])
        self._targets[i] = scene.target_point(target_zone, target_depth)

        self._prev_action[i] = 0.0
        self._ep_len[i] = 0
        self._ep_return[i] = 0.0
        self._contact_latched[i] = False
        self._legal_latched[i] = False
        self._fail_latched[i] = False
        self._own_bounce_count[i] = 0
        self._prev_landing[i] = None
        self._intercept_bonus_latched[i] = False

        flat = self._flat_obs(obs, self._prev_action[i], self._targets[i], obs["time_s"])
        self._obs_fifo[i].clear()
        for _ in range(self.obs_delay_steps + 1):
            self._obs_fifo[i].append(flat)
        return self._obs_fifo[i][0]

    def reset(self) -> np.ndarray:
        for i in range(self.n_envs):
            self._last_obs[i] = self._reset_one(i)
        return self._last_obs.copy()

    # -- step --------------------------------------------------------------
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        assert actions.shape == (self.n_envs, ACT_DIM), actions.shape

        obs_batch = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)
        reward_batch = np.zeros(self.n_envs, dtype=np.float64)
        done_batch = np.zeros(self.n_envs, dtype=bool)
        infos: list[dict] = [None] * self.n_envs

        for i in range(self.n_envs):
            joint_target = map_action(actions[i], self._jnt_range)
            obs, events = self.envs[i].step(joint_target)

            # Computed once per env per step and reused for both the obs
            # block and the reward term below, rather than calling
            # predict_intercept twice.
            intercept = (self._predict_intercept_for(obs)
                         if (self.intercept_features or self.intercept_reward > 0.0) else None)

            reward, done = self._reward_and_done(i, obs, events, actions[i], intercept)
            self._ep_len[i] += 1
            self._ep_return[i] += reward
            if self._ep_len[i] >= MAX_EPISODE_STEPS:
                done = True

            flat = self._flat_obs(obs, actions[i], self._targets[i], obs["time_s"], intercept)
            self._obs_fifo[i].append(flat)
            delayed = self._obs_fifo[i][0]

            info = {"events": events}
            if done:
                info["episode"] = {
                    "return": float(self._ep_return[i]),
                    "length": int(self._ep_len[i]),
                    "contact": bool(self._contact_latched[i]),
                    "legal_return": bool(self._legal_latched[i]),
                }
                delayed = self._reset_one(i)  # auto-reset; obs returned is the NEW episode's first obs

            obs_batch[i] = delayed
            reward_batch[i] = reward
            done_batch[i] = done
            infos[i] = info
            self._prev_action[i] = actions[i]

        self._last_obs = obs_batch
        return obs_batch, reward_batch, done_batch, infos

    # -- intercept features -------------------------------------------------
    def _predict_intercept_for(self, obs: dict) -> tuple[np.ndarray, float, bool]:
        ball_pos = np.asarray(obs["ball.position"], dtype=np.float64)
        ball_vel = np.asarray(obs["ball.velocity"], dtype=np.float64)
        return predict_intercept(ball_pos, ball_vel)

    def _add_obs_noise(self, flat: np.ndarray) -> np.ndarray:
        """Additive Gaussian sensor noise on the flat obs the policy reads
        -- ball position/velocity always (obs_noise), the robot's own joint
        readings only if joint_noise is separately set (proprioception is
        normally far cleaner than a tracked external object, hence the
        separate default-off knob). Position noise std = the knob value, in
        the same metres/radians the obs slot is already in; velocity noise
        std = 10x the knob value, because obs velocity is what a vision
        tracker would estimate by differencing two noisy position samples
        over one control-step dt -- that differencing amplifies noise well
        past the position measurement it's derived from, and 10x is a coarse
        stand-in for that amplification (exact factor depends on dt/tracker,
        not modelled here). Velocity noise is drawn in the same raw m/s (or
        rad/s) units as the position noise, then scaled by the same 0.1/0.2
        factor flatten_obs already applies to that slot, so both land in obs
        units consistently."""
        out = flat.copy()
        if self.obs_noise > 0.0:
            out[_BALL_POS_SLICE] += self._noise_rng.normal(0.0, self.obs_noise, size=3)
            out[_BALL_VEL_SLICE] += self._noise_rng.normal(0.0, self.obs_noise * 10.0, size=3) * 0.2
        if self.joint_noise > 0.0:
            n = scene.NUM_JOINTS
            out[_JOINT_POS_SLICE] += self._noise_rng.normal(0.0, self.joint_noise, size=n)
            out[_JOINT_VEL_SLICE] += self._noise_rng.normal(0.0, self.joint_noise * 10.0, size=n) * 0.1
        return out.astype(np.float32)

    def _flat_obs(self, obs: dict, prev_action: np.ndarray, target: tuple[float, float], time_s: float,
                  intercept: tuple[np.ndarray, float, bool] | None = None) -> np.ndarray:
        base = flatten_obs(obs, prev_action, target, time_s)
        if self.obs_noise > 0.0 or self.joint_noise > 0.0:
            base = self._add_obs_noise(base)
        if not self.intercept_features:
            return base
        p_int, t_int, valid = intercept if intercept is not None else self._predict_intercept_for(obs)
        paddle_pos = np.asarray(obs["paddle.position"], dtype=np.float64)
        extra = np.concatenate([
            p_int,
            [t_int],
            p_int - paddle_pos,
            [1.0 if valid else 0.0],
        ]).astype(np.float32)
        return np.concatenate([base, extra])

    # -- reward shaping ------------------------------------------------------
    def _reward_and_done(self, i: int, obs: dict, events: dict, action: np.ndarray,
                          intercept: tuple[np.ndarray, float, bool] | None = None) -> tuple[float, bool]:
        reward = 0.0
        done = False

        ball_pos = np.asarray(obs["ball.position"], dtype=np.float64)
        paddle_pos = np.asarray(obs["paddle.position"], dtype=np.float64)
        dist2 = float(np.sum((paddle_pos - ball_pos) ** 2))

        # Approach shaping only while the ball is still incoming, i.e.
        # before this episode's paddle contact has happened.
        if not self._contact_latched[i]:
            reward += 0.02 * float(np.exp(-dist2 / 0.15 ** 2))

        # Dense intercept shaping (opt-in via intercept_reward > 0): pull the
        # paddle toward the predicted intercept point p_int while the ball is
        # still incoming, plus a one-time +1 for getting within
        # INTERCEPT_BONUS_RADIUS_M of p_int before the predicted contact time
        # t_int elapses. Zero (no-op) once contact has latched, matching the
        # approach-shaping term above -- both read _contact_latched before
        # this step's contact_now update below, so a step that both closes
        # to intercept range *and* makes contact still earns this term for
        # that one step, consistent with the un-latched-yet state it reads.
        # getattr, not self.intercept_reward: VecTT subclasses that build
        # their own __init__ without calling super().__init__() (e.g.
        # tt_sim.rl.residual.VecTTResidual, which duplicates VecTT's
        # bookkeeping rather than reusing it, per its own docstring) never
        # set this attribute; inheriting _reward_and_done must stay a
        # behaviour-preserving no-op for them, not an AttributeError.
        if getattr(self, "intercept_reward", 0.0) > 0.0 and intercept is not None and not self._contact_latched[i]:
            p_int, t_int, valid = intercept
            if valid:
                dist = float(np.linalg.norm(paddle_pos - p_int))
                reward -= self.intercept_reward * dist
                if (not self._intercept_bonus_latched[i]
                        and dist <= INTERCEPT_BONUS_RADIUS_M and t_int > 0.0):
                    reward += 1.0
                    self._intercept_bonus_latched[i] = True

        contact_now = bool(events.get("contact_paddle"))
        if contact_now and not self._contact_latched[i]:
            reward += 2.0
            self._contact_latched[i] = True

        legal_now = scene.legal_return(events)
        if legal_now and not self._legal_latched[i]:
            reward += getattr(self, "legal_bonus", 5.0)
            err = scene.landing_error_m(events, self._targets[i])
            if not np.isnan(err):
                # getattr with the v1 constants: see intercept_reward's comment
                # -- VecTT subclasses that skip super().__init__() inherit this.
                sigma = getattr(self, "aim_sigma", 0.5)
                reward += getattr(self, "aim_bonus", 3.0) * float(np.exp(-err ** 2 / sigma ** 2))
                step_bonus = getattr(self, "target_hit_bonus", 0.0)
                if step_bonus > 0.0 and err <= TARGET_HIT_RADIUS_M:
                    reward += step_bonus
            # Depth shortfall, opt-in (see __init__). Only the shortfall is
            # priced -- landing long is already the aim kernel's business, and
            # the measured bias is one-sided. getattr with 0.0, like the knobs
            # above: VecTT subclasses that skip super().__init__() inherit this
            # method and must keep seeing the pre-v3 reward.
            depth_w = getattr(self, "depth_bonus", 0.0)
            landing = events.get("landing_point")
            if depth_w > 0.0 and landing is not None:
                dx = float(landing[0]) - float(self._targets[i][0])
                if not np.isnan(dx):
                    reward -= depth_w * max(0.0, -dx)
            self._legal_latched[i] = True

        da = action - self._prev_action[i]
        reward -= 0.001 * float(np.sum(da ** 2))

        jv = np.asarray(obs["robot.joint_velocity"], dtype=np.float64)
        reward -= getattr(self, "vel_penalty", 0.0005) * float(np.sum(jv ** 2))

        failed_now = bool(events.get("net_hit")) or bool(events.get("out"))
        if failed_now and self._contact_latched[i] and not self._fail_latched[i]:
            reward -= 1.0
            self._fail_latched[i] = True
        if failed_now:
            done = True

        # Own-side-bounce counting / opponent-side-bounce termination:
        # events['landing_point'] is overwritten on every new table bounce
        # (own or opponent side) even though the accompanying boolean flags
        # only ever latch True once -- a *new* landing_point tuple is this
        # wrapper's own edge-detector for "another bounce just happened".
        landing = events.get("landing_point")
        if landing is not None and landing != self._prev_landing[i]:
            self._prev_landing[i] = landing
            if landing[0] < 0.0:
                self._own_bounce_count[i] += 1
                if self._own_bounce_count[i] >= 2:
                    done = True
            else:
                done = True  # landed on the opponent's side: return is over

        return reward, done


def benchmark_steps_per_second(n_envs: int = 32, n_steps: int = 100, seed: int = 0) -> float:
    """Random-action throughput benchmark used by the task report."""
    vec = VecTT(n_envs=n_envs, seed=seed)
    rng = np.random.default_rng(seed)
    vec.reset()
    start = time.time()
    for _ in range(n_steps):
        actions = rng.uniform(-1.0, 1.0, size=(n_envs, ACT_DIM))
        vec.step(actions)
    elapsed = time.time() - start
    total_env_steps = n_envs * n_steps
    return total_env_steps / elapsed


if __name__ == "__main__":
    sps = benchmark_steps_per_second()
    print(f"{sps:.1f} steps/s")
