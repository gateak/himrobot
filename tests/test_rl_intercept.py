"""Tests for himrobot.rl.intercept and VecTT's intercept_features/
intercept_reward options. Run with:
    .venv/bin/python -m pytest tests/test_rl_intercept.py -v
"""
from __future__ import annotations

import numpy as np

from himrobot.policies.ik_hitter import IKHitterPolicy, Y_REACH, Z_REACH
from himrobot.rl.env import (
    ACT_DIM,
    INTERCEPT_FEATURES_DIM,
    OBS_DIM,
    VecTT,
    obs_dim,
)
from himrobot.rl.intercept import predict_intercept
from himrobot.sim.scene import DEFAULT_PARAMS, TableTennisEnv, sample_serve

N_SERVES = 10


def _first_plan(seed: int):
    """Serve one ball (sample_serve, same convention as
    test_ik_hitter.py's _run_trials) and drive a fresh IKHitterPolicy for
    one control step -- enough to trigger its first _maybe_replan call and
    populate self._plan. Returns (ball_pos, ball_vel, t_now, plan)."""
    env = TableTennisEnv(seed=seed, params={})
    rng = np.random.default_rng(seed)
    serve = sample_serve(rng, DEFAULT_PARAMS)
    obs = env.reset(serve=serve)

    policy = IKHitterPolicy()
    policy.reset({"serve": serve, "params": DEFAULT_PARAMS})

    ball_pos = np.asarray(obs["ball.position"], dtype=np.float64)
    ball_vel = np.asarray(obs["ball.velocity"], dtype=np.float64)
    t_now = float(obs["time_s"])
    policy.act(obs)  # first call: self._plan was None, so this always replans
    return ball_pos, ball_vel, t_now, policy._plan


def test_predict_intercept_agrees_with_hitter_plan_on_ten_serves():
    n_checked = 0
    for seed in range(N_SERVES):
        ball_pos, ball_vel, t_now, plan = _first_plan(seed)
        if plan is None:
            continue  # this serve's ball flight never reaches x_hit -- nothing to compare
        n_checked += 1

        p_int, t_int, valid = predict_intercept(ball_pos, ball_vel)
        assert valid, f"seed {seed}: hitter found a plan but predict_intercept did not"

        # plan["p_hit"] is the hitter's raw predicted point clamped into the
        # arm's Y/Z reach window (an IK-workspace accommodation --
        # predict_intercept deliberately skips it, see its docstring).
        # Clamp our result the same way here so the comparison is apples
        # to apples.
        p_int_clamped = p_int.copy()
        p_int_clamped[1] = np.clip(p_int_clamped[1], *Y_REACH)
        p_int_clamped[2] = np.clip(p_int_clamped[2], *Z_REACH)

        pos_err = float(np.linalg.norm(p_int_clamped - plan["p_hit"]))
        assert pos_err <= 0.02, f"seed {seed}: intercept position off by {pos_err:.4f} m"

        t_rel_expected = plan["t_contact"] - t_now
        t_err = abs(t_int - t_rel_expected)
        assert t_err <= 0.02, f"seed {seed}: intercept time off by {t_err * 1000:.1f} ms"

    assert n_checked >= 8, f"only {n_checked}/{N_SERVES} serves produced a comparable hitter plan"


def test_predict_intercept_invalid_when_ball_moving_away():
    # Ball at the robot's hitting plane already, moving further away in +x
    # (toward the opponent) -- should never reach x_hit again.
    ball_pos = np.array([-1.20, 0.0, 1.0])
    ball_vel = np.array([5.0, 0.0, 0.0])
    p_int, t_int, valid = predict_intercept(ball_pos, ball_vel)
    assert not valid
    assert np.allclose(p_int, 0.0)
    assert t_int == 0.0


def test_obs_dim_composability():
    assert obs_dim() == OBS_DIM == 99
    assert obs_dim(intercept_features=True) == OBS_DIM + INTERCEPT_FEATURES_DIM == 107
    assert obs_dim(expert_features=True) == OBS_DIM + 29
    assert obs_dim(intercept_features=True, expert_features=True) == OBS_DIM + INTERCEPT_FEATURES_DIM + 29


def test_vectt_default_behaviour_unchanged():
    """intercept_features/intercept_reward default off -- obs width and
    shapes must be identical to a plain VecTT()."""
    vec = VecTT(n_envs=2, seed=0)
    assert vec.obs_dim == OBS_DIM
    obs = vec.reset()
    assert obs.shape == (2, OBS_DIM)


def test_vectt_intercept_features_widen_obs_and_are_finite():
    n = 3
    vec = VecTT(n_envs=n, seed=0, intercept_features=True)
    assert vec.obs_dim == OBS_DIM + INTERCEPT_FEATURES_DIM
    obs = vec.reset()
    assert obs.shape == (n, OBS_DIM + INTERCEPT_FEATURES_DIM)
    assert np.isfinite(obs).all()

    rng = np.random.default_rng(0)
    for _ in range(30):
        actions = rng.uniform(-1.0, 1.0, size=(n, ACT_DIM))
        obs, reward, done, infos = vec.step(actions)
        assert obs.shape == (n, OBS_DIM + INTERCEPT_FEATURES_DIM)
        assert np.isfinite(obs).all()
        # valid flag (last of the 8 appended dims) must be exactly 0 or 1.
        valid_col = obs[:, -1]
        assert np.all((valid_col == 0.0) | (valid_col == 1.0))


def test_intercept_reward_negative_then_zero_across_scripted_paddle_path():
    """Directly exercises VecTT._reward_and_done's intercept term (isolated
    from contact/legal/action/velocity terms by zeroing those inputs) across
    a scripted paddle path: far from p_int (negative), within the bonus
    radius (negative-but-small, +1 once), then post-contact (exactly zero).
    """
    vec = VecTT(n_envs=1, seed=0, intercept_reward=2.0)
    zero_action = np.zeros(ACT_DIM)
    obs = {
        "ball.position": np.array([0.0, 0.0, 1.0]),
        "ball.velocity": np.array([0.0, 0.0, 0.0]),
        "robot.joint_velocity": np.zeros(29),
        "paddle.position": np.array([1.0, 1.0, 1.0]),  # placeholder, overwritten per step below
    }
    events = {"contact_paddle": False, "net_hit": False, "out": False, "landing_point": None}
    p_int = np.array([0.2, -0.3, 0.9])
    intercept_valid = (p_int, 0.20, True)  # t_int > 0: still "before" the predicted contact

    # Step 1: paddle far away -- reward must be negative.
    obs["paddle.position"] = p_int + np.array([1.0, 0.0, 0.0])
    reward1, done1 = vec._reward_and_done(0, obs, events, zero_action, intercept_valid)
    assert reward1 < 0.0
    assert not done1

    # Step 2: paddle within the 5cm bonus radius -- the one-time +1 bonus
    # should make this step's reward clearly larger than step 1's (which
    # was at 1m), even though the same -intercept_reward*dist term is
    # still subtracted.
    obs["paddle.position"] = p_int + np.array([0.03, 0.0, 0.0])
    reward2, _ = vec._reward_and_done(0, obs, events, zero_action, intercept_valid)
    assert reward2 > reward1
    assert vec._intercept_bonus_latched[0]

    # Bonus only fires once: a second step back inside the radius doesn't
    # re-add it.
    reward2b, _ = vec._reward_and_done(0, obs, events, zero_action, intercept_valid)
    assert reward2b < reward2

    # Step 3: contact has now happened (simulate the latch directly, as a
    # real contact_paddle event would set it) -- with action/velocity terms
    # zeroed and paddle far from p_int again, the intercept term (and the
    # matching approach-shaping term) must contribute exactly zero.
    vec._contact_latched[0] = True
    obs["paddle.position"] = p_int + np.array([1.0, 0.0, 0.0])
    reward3, _ = vec._reward_and_done(0, obs, events, zero_action, intercept_valid)
    assert reward3 == 0.0


def test_existing_reward_path_unaffected_when_intercept_reward_zero():
    """intercept_reward=0.0 (the default) must never touch reward, even
    when an `intercept` tuple is passed in (e.g. because intercept_features
    alone is on) -- confirms the two knobs are independent."""
    vec = VecTT(n_envs=1, seed=0, intercept_reward=0.0)
    zero_action = np.zeros(ACT_DIM)
    obs = {
        "ball.position": np.array([0.0, 0.0, 1.0]),
        "ball.velocity": np.array([0.0, 0.0, 0.0]),
        "robot.joint_velocity": np.zeros(29),
        "paddle.position": np.array([5.0, 5.0, 5.0]),  # far from p_int and from ball_pos
    }
    events = {"contact_paddle": False, "net_hit": False, "out": False, "landing_point": None}
    intercept_valid = (np.array([0.2, -0.3, 0.9]), 0.20, True)

    reward, _ = vec._reward_and_done(0, obs, events, zero_action, intercept_valid)
    # Only the (tiny) approach-shaping term can be nonzero here (paddle is
    # far from ball_pos, so it's ~0 too) -- no intercept contribution.
    assert reward >= 0.0
