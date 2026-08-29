"""Tests for tt_sim.rl.env.VecTT. Run with:
    .venv/bin/python -m pytest tt_sim/tests/test_rl_env.py -v
"""
from __future__ import annotations

import numpy as np

from tt_sim.rl.env import ACT_DIM, OBS_DIM, VecTT
from tt_sim.policies.ik_hitter import IKHitterPolicy
from tt_sim.scene import DEFAULT_PARAMS


def test_reset_and_step_shapes_finite():
    n = 3
    vec = VecTT(n_envs=n, seed=0)
    obs = vec.reset()
    assert obs.shape == (n, OBS_DIM)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()

    rng = np.random.default_rng(1)
    for _ in range(50):
        actions = rng.uniform(-1.0, 1.0, size=(n, ACT_DIM))
        obs, reward, done, infos = vec.step(actions)
        assert obs.shape == (n, OBS_DIM)
        assert np.isfinite(obs).all()
        assert reward.shape == (n,)
        assert np.isfinite(reward).all()
        assert done.shape == (n,)
        assert done.dtype == bool
        assert len(infos) == n
        for info in infos:
            assert "events" in info


def test_action_out_of_range_is_clipped_not_rejected():
    vec = VecTT(n_envs=2, seed=0)
    vec.reset()
    actions = np.array([[5.0] * ACT_DIM, [-5.0] * ACT_DIM])
    obs, reward, done, infos = vec.step(actions)
    assert np.isfinite(obs).all()
    assert np.isfinite(reward).all()


def test_legs_frozen_regardless_of_action():
    """Legs (JOINT_ORDER[0:12]) have action scale 0.0 -- any commanded
    action there must map to STAND_POSE, never move the leg targets."""
    from tt_sim.rl.env import map_action, ACTION_SCALE
    from tt_sim.scene import STAND_POSE

    assert np.all(ACTION_SCALE[:12] == 0.0)
    jnt_range = np.array([[-100.0, 100.0]] * ACT_DIM)  # wide enough not to clip
    action = np.ones(ACT_DIM)
    target = map_action(action, jnt_range)
    assert np.allclose(target[:12], STAND_POSE[:12])


def test_auto_reset_on_done_returns_next_episode_obs():
    """Driving with zero action for the full 150-step episode budget must
    trigger at least one auto-reset (3s / MAX_EPISODE_STEPS cap) and carry
    a well-formed info['episode'] summary."""
    vec = VecTT(n_envs=2, seed=0)
    vec.reset()
    zero = np.zeros((2, ACT_DIM))
    saw_done = False
    for _ in range(160):
        obs, reward, done, infos = vec.step(zero)
        for i, info in enumerate(infos):
            if done[i]:
                saw_done = True
                ep = info["episode"]
                assert set(ep) == {"return", "length", "contact", "legal_return"}
                assert ep["length"] >= 1
    assert saw_done


def test_obs_delay_fifo_lags_by_requested_steps():
    delay = 3
    vec = VecTT(n_envs=1, seed=0, obs_delay_steps=delay)
    vec.reset()
    rng = np.random.default_rng(0)
    # Track the raw (undelayed) flattened obs by peeking at the fifo's most
    # recent entry each step, and confirm the returned obs matches what was
    # current `delay` steps earlier.
    history = []
    for t in range(10):
        actions = rng.uniform(-1.0, 1.0, size=(1, ACT_DIM))
        obs, _, done, _ = vec.step(actions)
        history.append(np.array(vec._obs_fifo[0][-1]))  # most recent raw obs this step
        if done[0]:
            break  # auto-reset would desync the simple index check below
        if t >= delay:
            expected = history[t - delay]
            assert np.allclose(obs[0], expected)


def test_obs_delay_two_steps_lags_by_two():
    """Same check as test_obs_delay_fifo_lags_by_requested_steps above, at
    delay=2 specifically."""
    delay = 2
    vec = VecTT(n_envs=1, seed=0, obs_delay_steps=delay)
    vec.reset()
    rng = np.random.default_rng(0)
    history = []
    for t in range(10):
        actions = rng.uniform(-1.0, 1.0, size=(1, ACT_DIM))
        obs, _, done, _ = vec.step(actions)
        history.append(np.array(vec._obs_fifo[0][-1]))
        if done[0]:
            break
        if t >= delay:
            assert np.allclose(obs[0], history[t - delay])


def test_obs_noise_zero_matches_pre_noise_behaviour():
    """obs_noise/joint_noise default to 0.0 -- a seeded rollout must be
    bit-identical whether or not 0.0 is passed explicitly (the RNG draw is
    skipped entirely, see VecTT._flat_obs)."""
    n = 2
    vec_default = VecTT(n_envs=n, seed=7)
    vec_explicit_zero = VecTT(n_envs=n, seed=7, obs_noise=0.0, joint_noise=0.0)
    assert np.array_equal(vec_default.reset(), vec_explicit_zero.reset())

    rng = np.random.default_rng(3)
    for _ in range(30):
        actions = rng.uniform(-1.0, 1.0, size=(n, ACT_DIM))
        obs_a, _, _, _ = vec_default.step(actions.copy())
        obs_b, _, _, _ = vec_explicit_zero.step(actions.copy())
        assert np.array_equal(obs_a, obs_b)


def test_obs_noise_only_perturbs_ball_indices_std_within_3x_nominal():
    """obs_noise=0.01: two envs seeded identically and driven by the same
    action sequence see identical physics, so any obs difference is exactly
    the injected noise -- confirms it lands only on the ball pos/vel slices,
    and that its std is in the right ballpark (nominal std = obs_noise for
    position, obs_noise*10*0.2 for velocity, see env.py's _add_obs_noise)."""
    from tt_sim.rl.env import _BALL_POS_SLICE, _BALL_VEL_SLICE

    obs_noise = 0.01
    vec_clean = VecTT(n_envs=1, seed=11)
    vec_noisy = VecTT(n_envs=1, seed=11, obs_noise=obs_noise)
    vec_clean.reset()
    vec_noisy.reset()

    rng = np.random.default_rng(5)
    pos_diffs, vel_diffs = [], []
    other_mask = np.ones(OBS_DIM, dtype=bool)
    other_mask[_BALL_POS_SLICE] = False
    other_mask[_BALL_VEL_SLICE] = False

    for _ in range(500):
        actions = rng.uniform(-1.0, 1.0, size=(1, ACT_DIM))
        obs_clean, _, _, _ = vec_clean.step(actions.copy())
        obs_noisy, _, _, _ = vec_noisy.step(actions.copy())

        diff = obs_noisy - obs_clean
        assert np.allclose(diff[:, other_mask], 0.0, atol=1e-6)
        pos_diffs.append(diff[:, _BALL_POS_SLICE].copy())
        vel_diffs.append(diff[:, _BALL_VEL_SLICE].copy())

    pos_std = float(np.std(np.concatenate(pos_diffs, axis=0)))
    vel_std = float(np.std(np.concatenate(vel_diffs, axis=0)))
    nominal_pos_std = obs_noise
    nominal_vel_std = obs_noise * 10.0 * 0.2  # obs velocity is pre-scaled by 0.2, see flatten_obs
    assert nominal_pos_std / 3.0 <= pos_std <= nominal_pos_std * 3.0
    assert nominal_vel_std / 3.0 <= vel_std <= nominal_vel_std * 3.0


def test_ik_hitter_rollout_yields_at_least_one_contact_in_20_serves():
    """Drives VecTT's own underlying TableTennisEnv instances with
    IKHitterPolicy (bypassing the [-1,1] action-scale mapping, since
    IKHitterPolicy already returns absolute joint-position radians) to
    prove the wrapper's env construction preserves the scene's physics and
    events -- same acceptance shape as tt_sim/tests/test_ik_hitter.py, just
    exercised through tt_sim.rl.env's env objects.
    """
    from tt_sim.scene import sample_serve

    n_serves = 20
    steps_per_trial = 150  # 3s @ 50Hz, matches evaluate.py's EPISODE_S budget
    vec = VecTT(n_envs=1, seed=0)
    env = vec.envs[0]
    rng = np.random.default_rng(0)
    policy = IKHitterPolicy()

    contacts = []
    for _ in range(n_serves):
        serve = sample_serve(rng, DEFAULT_PARAMS)
        obs = env.reset(serve=serve)
        policy.reset({"serve": serve, "params": DEFAULT_PARAMS})
        info = {}
        for _ in range(steps_per_trial):
            action = policy.act(obs)
            obs, info = env.step(action)
        contacts.append(bool(info["contact_paddle"]))

    assert sum(contacts) >= 1, f"expected >=1 contact in {n_serves} serves, got {contacts}"


def test_target_hit_radius_matches_the_evaluator():
    """env.TARGET_HIT_RADIUS_M (target_hit_bonus, param_train.deterministic_eval)
    is a hand copy of the evaluator's radius; they must never drift apart."""
    from tt_sim import evaluate
    from tt_sim.rl import env as rl_env

    assert rl_env.TARGET_HIT_RADIUS_M == evaluate.TARGET_HIT_RADIUS_M == 0.30
