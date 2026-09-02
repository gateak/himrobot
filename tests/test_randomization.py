"""Serve-range overrides/serve_scale and physics_jitter. Run with:
    .venv/bin/python -m pytest tests/test_randomization.py -v
"""
from __future__ import annotations

import numpy as np

from himrobot.rl.env import VecTT
from himrobot.sim.scene import (
    SERVE_FLIGHT_TIME_RANGE,
    SERVE_TARGET_X_RANGE,
    SERVE_TARGET_Y_RANGE,
    SERVE_X0_RANGE,
    SERVE_Y0_RANGE,
    SERVE_Z0_HEIGHT_RANGE,
    STAND_POSE,
    TABLE_HALF_WIDTH,
    TABLE_TOP_Z,
    TableTennisEnv,
    sample_serve,
    serve_ranges,
)

N = 200


def test_default_params_reproduce_the_constants():
    # params={} must draw from exactly the same ranges as an explicit
    # params dict naming the current constants -- proves the new override
    # keys don't change default behaviour (no pre-edit snapshot needed).
    explicit = {
        "serve_x0_range": list(SERVE_X0_RANGE),
        "serve_y0_range": list(SERVE_Y0_RANGE),
        "serve_z0_height_range": list(SERVE_Z0_HEIGHT_RANGE),
        "serve_flight_time_range": list(SERVE_FLIGHT_TIME_RANGE),
        "serve_target_x_range": list(SERVE_TARGET_X_RANGE),
        "serve_target_y_range": list(SERVE_TARGET_Y_RANGE),
        "serve_scale": 1.0,
    }
    assert serve_ranges({}) == serve_ranges(explicit)

    for seed in (0, 7, 123):
        a = sample_serve(np.random.default_rng(seed), {})
        b = sample_serve(np.random.default_rng(seed), explicit)
        assert np.array_equal(a["pos"], b["pos"])
        assert np.array_equal(a["vel"], b["vel"])


def test_serve_range_overrides_are_honoured():
    params = {
        "serve_x0_range": [2.0, 2.1],
        "serve_y0_range": [0.3, 0.31],
        "serve_z0_height_range": [0.5, 0.51],
        "serve_flight_time_range": [1.0, 1.01],
        "serve_target_x_range": [-0.5, -0.49],
        "serve_target_y_range": [-0.2, -0.19],
    }
    ranges = serve_ranges(params)
    assert ranges["target_x"] == (-0.5, -0.49)
    assert ranges["target_y"] == (-0.2, -0.19)
    assert ranges["flight_time"] == (1.0, 1.01)

    rng = np.random.default_rng(1)
    for _ in range(N):
        serve = sample_serve(rng, params)
        x0, y0, z0 = serve["pos"]
        assert 2.0 <= x0 < 2.1
        assert 0.3 <= y0 < 0.31
        assert 0.5 <= z0 - TABLE_TOP_Z < 0.51


def test_serve_scale_widens_symmetrically_and_clips_target_y():
    base = serve_ranges({})["x0"]
    centre = (base[0] + base[1]) / 2.0
    scaled = serve_ranges({"serve_scale": 2.0})["x0"]
    assert scaled[1] - scaled[0] == 2 * (base[1] - base[0])
    assert (scaled[0] + scaled[1]) / 2.0 == centre  # widened about the same centre

    # default target_y (all <= 0) widened hugely: clipped to the table
    # half-width and stays on the forehand side (y<=0), never crosses over.
    ty = serve_ranges({"serve_scale": 50.0})["target_y"]
    assert ty[0] == -TABLE_HALF_WIDTH
    assert ty[1] == 0.0

    # an override that isn't entirely on the forehand side doesn't get the
    # y<=0 clamp, only the half-width clamp.
    ty2 = serve_ranges({"serve_target_y_range": [-0.1, 0.1], "serve_scale": 50.0})["target_y"]
    assert ty2[0] == -TABLE_HALF_WIDTH
    assert ty2[1] == TABLE_HALF_WIDTH

    rng = np.random.default_rng(2)
    for _ in range(N):
        serve = sample_serve(rng, {"serve_scale": 2.0})
        assert scaled[0] <= serve["pos"][0] < scaled[1]


def _model_snapshot(env: TableTennisEnv):
    return (
        env.model.pair_solref[env._jitter_pair_ids].copy(),
        env.model.geom_friction[env._jitter_geom_ids].copy(),
        float(env.model.body_mass[env._ball_body_id]),
    )


def test_physics_jitter_zero_leaves_model_untouched():
    env = TableTennisEnv(seed=0, params={})
    nominal = _model_snapshot(env)
    for _ in range(5):
        env.reset()
        assert env._events["physics"] is None
        solref, friction, mass = _model_snapshot(env)
        assert np.array_equal(solref, nominal[0])
        assert np.array_equal(friction, nominal[1])
        assert mass == nominal[2]


def test_physics_jitter_changes_fields_within_bounds_and_sim_runs():
    j = 0.2
    env = TableTennisEnv(seed=0, params={"physics_jitter": j})
    nominal_solref, nominal_friction, nominal_mass = env._nominal_pair_solref, env._nominal_geom_friction, env._nominal_ball_mass

    for _ in range(10):
        env.reset()
        info = env._events["physics"]
        assert info is not None
        for mult in info.values():
            assert 1.0 - j <= mult <= 1.0 + j

        solref, friction, mass = _model_snapshot(env)
        assert np.all(solref >= nominal_solref * (1.0 - j) - 1e-12)
        assert np.all(solref <= nominal_solref * (1.0 + j) + 1e-12)
        assert np.all(friction >= nominal_friction * (1.0 - j) - 1e-12)
        assert np.all(friction <= nominal_friction * (1.0 + j) + 1e-12)
        assert nominal_mass * (1.0 - j) - 1e-12 <= mass <= nominal_mass * (1.0 + j) + 1e-12
        # actually moved from nominal (uniform(0.8,1.2) landing exactly on
        # 1.0 is a measure-zero event, so this also catches an accidental no-op)
        assert not np.array_equal(solref, nominal_solref)

    env.reset(serve={"pos": np.array([1.1, 0.0, 0.95]), "vel": np.array([-3.0, 0.0, 1.0]), "spin": None})
    for _ in range(200):
        obs, _ = env.step(STAND_POSE)
    assert np.all(np.isfinite(obs["ball.position"]))


def test_vectt_reset_applies_jitter_per_env():
    vec = VecTT(n_envs=4, seed=0, params={"physics_jitter": 0.2})
    assert all(e._events["physics"] is not None for e in vec.envs)

    vec_off = VecTT(n_envs=4, seed=0, params={})
    assert all(e._events["physics"] is None for e in vec_off.envs)

    # auto-reset mid-rollout also re-jitters (not just the constructor reset);
    # 150 steps = MAX_EPISODE_STEPS, guaranteeing at least one auto-reset.
    rng = np.random.default_rng(0)
    for _ in range(150):
        vec.step(rng.uniform(-1.0, 1.0, size=(4, vec.act_dim)))
    assert all(e._events["physics"] is not None for e in vec.envs)


def test_default_vectt_rollout_unchanged_by_new_params():
    explicit_defaults = {
        "serve_scale": 1.0,
        "physics_jitter": 0.0,
        "serve_x0_range": list(SERVE_X0_RANGE),
        "serve_y0_range": list(SERVE_Y0_RANGE),
        "serve_z0_height_range": list(SERVE_Z0_HEIGHT_RANGE),
        "serve_flight_time_range": list(SERVE_FLIGHT_TIME_RANGE),
        "serve_target_x_range": list(SERVE_TARGET_X_RANGE),
        "serve_target_y_range": list(SERVE_TARGET_Y_RANGE),
    }
    vec_a = VecTT(n_envs=3, seed=5, params={})
    vec_b = VecTT(n_envs=3, seed=5, params=explicit_defaults)

    assert np.array_equal(vec_a._last_obs, vec_b._last_obs)

    rng = np.random.default_rng(9)
    for _ in range(30):
        actions = rng.uniform(-1.0, 1.0, size=(3, vec_a.act_dim))
        obs_a, rew_a, done_a, _ = vec_a.step(actions)
        obs_b, rew_b, done_b, _ = vec_b.step(actions)
        assert np.array_equal(obs_a, obs_b)
        assert np.array_equal(rew_a, rew_b)
        assert np.array_equal(done_a, done_b)
