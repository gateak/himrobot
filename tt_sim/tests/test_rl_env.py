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
