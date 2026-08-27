"""Tests for tt_sim.rl.residual.VecTTResidual and
tt_sim.rl.policy.ResidualPPOPolicy. Run with:
    .venv/bin/python -m pytest tt_sim/tests/test_rl_residual.py -v

Note on "exact reproduction" (test 1 below): tt_sim.rl.demos.target_to_action
clips a waist/right-arm joint's delta-from-STAND_POSE to +/-1 rad before
tt_sim.rl.env.map_action reconstructs an absolute joint target from it, so
whenever the IK hitter's own swing genuinely needs more than 1 rad on some
dim (measured at ~2% of kept waist/arm action components in
work/tt/rl/demos_seed0_summary.json's action_saturation_fraction), that
round trip is lossy -- a physically real property of the [-1,1]^29 action
space, not a bug in this wrapper. So this file checks exact reproduction
two ways: (1) a per-step wiring check that VecTTResidual's *executed*
action, at every control step of a real 20-episode rollout, is bit-exact
to independently recomputing target_to_action(IKHitterPolicy.act(obs))
from that exact same live obs (always true by construction -- it would
catch e.g. a wrong obs, a missed per-episode hitter reset, or a
scale/append mixup); (2) a looser per-episode outcome comparison against a
*separately* rolled IKHitterPolicy trajectory (same seeds), which can only
be expected to match in most, not strictly all, of the 20 episodes, for
the saturation reason above.
"""
from __future__ import annotations

import pytest
torch = pytest.importorskip('torch')  # HIM's venv has no torch; RL-only test
import numpy as np

from tt_sim.policies.ik_hitter import IKHitterPolicy
from tt_sim.rl.demos import target_to_action
from tt_sim.rl.env import ACT_DIM, MAX_EPISODE_STEPS, OBS_DIM
from tt_sim.rl.residual import OBS_VERSION, RESIDUAL_OBS_DIM, VecTTResidual, flatten_residual_obs
from tt_sim.scene import TableTennisEnv, legal_return

N_SERVES = 20
SEED = 0


def test_obs_dim_and_layout():
    assert OBS_VERSION == 2
    assert RESIDUAL_OBS_DIM == OBS_DIM + ACT_DIM == 128

    vec = VecTTResidual(n_envs=3, seed=SEED)
    assert vec.obs_dim == RESIDUAL_OBS_DIM
    assert vec._last_obs.shape == (3, RESIDUAL_OBS_DIM)
    assert vec._last_obs.dtype == np.float32
    assert np.isfinite(vec._last_obs).all()

    obs = vec.reset()
    assert obs.shape == (3, RESIDUAL_OBS_DIM)

    # flatten_residual_obs appends the expert action LAST, after the plain
    # 99-dim env.flatten_obs block -- and that appended block is in
    # [-1,1] (it's an action, not a raw observable) while the "raw" 99-dim
    # obs block is not action-scaled at all -- a cheap layout sanity check.
    base = np.zeros(OBS_DIM, dtype=np.float32)
    act = np.linspace(-1.0, 1.0, ACT_DIM, dtype=np.float32)
    row = flatten_residual_obs(base, act)
    assert row.shape == (RESIDUAL_OBS_DIM,)
    assert np.allclose(row[:OBS_DIM], base)
    assert np.allclose(row[OBS_DIM:], act)

    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.uniform(-1.0, 1.0, size=(3, ACT_DIM))
        obs, reward, done, infos = vec.step(a)
        assert obs.shape == (3, RESIDUAL_OBS_DIM)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward).all()


def test_zero_action_wiring_matches_recomputed_expert_at_every_step():
    """residual_scale=0 -> combined == a_exp exactly (clip is a no-op:
    target_to_action already clips to [-1,1]). At every one of
    N_SERVES episodes' control steps, independently recompute
    target_to_action(IKHitterPolicy.act(obs)) from VecTTResidual's own
    live obs (env._get_obs(), a pure read -- no extra reset/rng draw) via
    a *separate* IKHitterPolicy kept reset in lockstep, and check it's
    bit-identical to what VecTTResidual cached/executed as a_exp. See
    module docstring for why this -- not a separately-simulated rollout
    -- is the exact (never-flaky) check."""
    vec = VecTTResidual(n_envs=1, seed=SEED, residual_scale=0.0)  # __init__ already reset episode 0
    shadow = IKHitterPolicy()
    shadow.reset({"serve": None, "params": vec.envs[0].params})

    def check_current_expert_action():
        obs = vec.envs[0]._get_obs()
        expected = target_to_action(shadow.act(obs))
        assert np.array_equal(vec._expert_action[0], expected), "cached a_exp diverged from a lockstep hitter"

    check_current_expert_action()  # the __init__-time reset's a_exp
    zero = np.zeros((1, ACT_DIM))
    n_episodes_seen = 0
    while n_episodes_seen < N_SERVES:
        obs, reward, done, infos = vec.step(zero)
        if done[0]:
            n_episodes_seen += 1
            shadow.reset({"serve": None, "params": vec.envs[0].params})
        check_current_expert_action()


def test_zero_action_reproduces_most_direct_ikhitter_episode_outcomes():
    """Looser per-episode check against a genuinely independent rollout
    (TableTennisEnv + IKHitterPolicy driven directly, same seed/serve
    sequence) -- see module docstring on why a handful of the N_SERVES
    episodes, not strictly all, is the honest bar."""
    vec = VecTTResidual(n_envs=1, seed=SEED, residual_scale=0.0)
    zero = np.zeros((1, ACT_DIM))
    vec_events = []
    for _ in range(N_SERVES):
        info = None
        for _ in range(MAX_EPISODE_STEPS):
            obs, reward, done, infos = vec.step(zero)
            if done[0]:
                info = infos[0]
                break
        assert info is not None, "episode did not terminate within MAX_EPISODE_STEPS"
        vec_events.append((info["episode"]["contact"], info["episode"]["legal_return"]))

    env = TableTennisEnv(seed=SEED)
    hitter = IKHitterPolicy()
    direct_events = []
    for _ in range(N_SERVES):
        obs = env.reset(serve=None)
        hitter.reset({"serve": None, "params": env.params})
        info = {}
        for _ in range(MAX_EPISODE_STEPS):
            obs, info = env.step(hitter.act(obs))
        direct_events.append((bool(info.get("contact_paddle")), bool(legal_return(info))))

    contact_matches = sum(v[0] == d[0] for v, d in zip(vec_events, direct_events))
    assert contact_matches >= 0.8 * N_SERVES, (vec_events, direct_events)


def test_hitter_reset_between_episodes_uses_current_params():
    """Each env's IKHitterPolicy is reset (not just the TableTennisEnv)
    on every episode boundary, per-env, with that env's own params."""
    params = {"target_zone": "line", "target_depth": "short", "risk_preference": 0.6}
    vec = VecTTResidual(n_envs=1, seed=SEED, params=params, residual_scale=0.0)
    assert vec._hitters[0]._target_zone == "line"
    assert vec._hitters[0]._target_depth == "short"
    assert vec._hitters[0]._risk == 0.6


def test_ppo_residual_train_and_load(tmp_path):
    from tt_sim.rl import ppo as ppo_mod
    from tt_sim.rl.policy import ResidualPPOPolicy

    out = str(tmp_path / "residual_run")
    result = ppo_mod.train(
        n_envs=4, total_steps=1, seed=0, out=out,
        rollout_steps=8, residual=True, residual_scale=0.3,
    )
    assert result["n_updates"] >= 1
    ckpt_path = f"{out}/final.pt"

    policy = ResidualPPOPolicy(ckpt_path=ckpt_path)
    env = TableTennisEnv(seed=1)
    obs = env.reset(serve=None)
    policy.reset({"serve": None, "params": env.params})
    action = policy.act(obs)
    assert action.shape == (ACT_DIM,)
    assert np.isfinite(action).all()
