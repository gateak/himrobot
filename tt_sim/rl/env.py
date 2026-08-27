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
from tt_sim.scene import TableTennisEnv

import mujoco

# ---------------------------------------------------------------------------
# Action mapping: 29-dim action in [-1, 1] -> joint targets = STAND_POSE +
# action * scale. Per the task spec: arms/waist scale 1.0 rad, legs scale
# 0.0 (frozen -- the base is fixed, per scene.py's fixed-pelvis design, so
# there is nothing for the legs to do; freezing them also shrinks the
# effective action space PPO has to explore). JOINT_ORDER is
# [left leg(6), right leg(6), waist(3), left arm(7), right arm(7)].
# ---------------------------------------------------------------------------
_N_LEG = 12
_N_WAIST = 3
_N_ARM = 14
assert _N_LEG + _N_WAIST + _N_ARM == scene.NUM_JOINTS
ACTION_SCALE = np.array([0.0] * _N_LEG + [1.0] * _N_WAIST + [1.0] * _N_ARM, dtype=np.float64)

# Flat obs layout (99-dim): joint_pos-default[29], joint_vel*0.1[29],
# ball.position[3], ball.velocity*0.2[3], paddle.position[3], target[2],
# phase[1], prev_action[29].
OBS_DIM = scene.NUM_JOINTS * 2 + 3 + 3 + 3 + 2 + 1 + scene.NUM_JOINTS  # 99
ACT_DIM = scene.NUM_JOINTS  # 29

MAX_EPISODE_STEPS = 150  # 3s @ 50Hz, matches tt_sim.evaluate.EPISODE_CONTROL_STEPS
REACTION_MOVE_M = 0.02  # unused here (scene.py owns reaction_latency_ms); kept for reference


def _joint_range(model: "mujoco.MjModel") -> np.ndarray:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in scene.JOINT_ORDER]
    return np.array([model.jnt_range[j] for j in ids], dtype=np.float64)


def map_action(action: np.ndarray, jnt_range: np.ndarray) -> np.ndarray:
    """action in [-1,1]^29 -> absolute joint-position targets (radians),
    clamped to each joint's own range (STAND_POSE + action*scale can
    otherwise walk an arm joint's target past its physical limit)."""
    target = scene.STAND_POSE + action * ACTION_SCALE
    return np.clip(target, jnt_range[:, 0], jnt_range[:, 1])


def flatten_obs(obs: dict, prev_action: np.ndarray, target: tuple, time_s: float) -> np.ndarray:
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
                 obs_delay_steps: int = 0, randomize: bool = True):
        assert n_envs >= 1
        assert obs_delay_steps >= 0
        self.n_envs = n_envs
        self.params = dict(params or {})
        self.obs_delay_steps = obs_delay_steps
        self.randomize = randomize
        self.obs_dim = OBS_DIM
        self.act_dim = ACT_DIM

        self.envs = [TableTennisEnv(seed=seed + i, params=self.params) for i in range(n_envs)]
        # Fixed-serve rngs used only when randomize=False, so a "no
        # randomization" env still resets to a reproducible (but non-random
        # across calls) serve each episode -- useful for debugging/tests.
        self._fixed_serve_rng = [np.random.default_rng(seed + 999_983 + i) for i in range(n_envs)]

        self._jnt_range = _joint_range(self.envs[0].model)
        self._targets = [None] * n_envs
        self._prev_action = np.zeros((n_envs, ACT_DIM), dtype=np.float64)
        self._ep_len = np.zeros(n_envs, dtype=np.int64)
        self._ep_return = np.zeros(n_envs, dtype=np.float64)
        self._contact_latched = np.zeros(n_envs, dtype=bool)
        self._legal_latched = np.zeros(n_envs, dtype=bool)
        self._fail_latched = np.zeros(n_envs, dtype=bool)
        self._own_bounce_count = np.zeros(n_envs, dtype=np.int64)
        self._prev_landing = [None] * n_envs
        self._obs_fifo = [deque(maxlen=obs_delay_steps + 1) for _ in range(n_envs)]

        self._last_obs = np.zeros((n_envs, OBS_DIM), dtype=np.float32)
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

        flat = flatten_obs(obs, self._prev_action[i], self._targets[i], obs["time_s"])
        self._obs_fifo[i].clear()
        for _ in range(self.obs_delay_steps + 1):
            self._obs_fifo[i].append(flat)
        return self._obs_fifo[i][0]

    def reset(self) -> np.ndarray:
        for i in range(self.n_envs):
            self._last_obs[i] = self._reset_one(i)
        return self._last_obs.copy()

    # -- step --------------------------------------------------------------
    def step(self, actions: np.ndarray):
        actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        assert actions.shape == (self.n_envs, ACT_DIM), actions.shape

        obs_batch = np.zeros((self.n_envs, OBS_DIM), dtype=np.float32)
        reward_batch = np.zeros(self.n_envs, dtype=np.float64)
        done_batch = np.zeros(self.n_envs, dtype=bool)
        infos: list[dict] = [None] * self.n_envs

        for i in range(self.n_envs):
            joint_target = map_action(actions[i], self._jnt_range)
            obs, events = self.envs[i].step(joint_target)

            reward, done = self._reward_and_done(i, obs, events, actions[i])
            self._ep_len[i] += 1
            self._ep_return[i] += reward
            if self._ep_len[i] >= MAX_EPISODE_STEPS:
                done = True

            flat = flatten_obs(obs, actions[i], self._targets[i], obs["time_s"])
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

    # -- reward shaping ------------------------------------------------------
    def _reward_and_done(self, i: int, obs: dict, events: dict, action: np.ndarray) -> tuple[float, bool]:
        reward = 0.0
        done = False

        ball_pos = np.asarray(obs["ball.position"], dtype=np.float64)
        paddle_pos = np.asarray(obs["paddle.position"], dtype=np.float64)
        dist2 = float(np.sum((paddle_pos - ball_pos) ** 2))

        # Approach shaping only while the ball is still incoming, i.e.
        # before this episode's paddle contact has happened.
        if not self._contact_latched[i]:
            reward += 0.02 * float(np.exp(-dist2 / 0.15 ** 2))

        contact_now = bool(events.get("contact_paddle"))
        if contact_now and not self._contact_latched[i]:
            reward += 2.0
            self._contact_latched[i] = True

        legal_now = scene.legal_return(events)
        if legal_now and not self._legal_latched[i]:
            reward += 5.0
            err = scene.landing_error_m(events, self._targets[i])
            if not np.isnan(err):
                reward += 3.0 * float(np.exp(-err ** 2 / 0.5 ** 2))
            self._legal_latched[i] = True

        da = action - self._prev_action[i]
        reward -= 0.001 * float(np.sum(da ** 2))

        jv = np.asarray(obs["robot.joint_velocity"], dtype=np.float64)
        reward -= 0.0005 * float(np.sum(jv ** 2))

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
