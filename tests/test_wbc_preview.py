"""Tests for IKHitterPolicy.plan()/preview() (WB1, docs/design/whole_body_v1.md
sec 5.2 -- read-only accessors a whole-body controller uses to see the
hitter's swing plan and its upcoming joint commands without driving
anything). Run with:
    .venv/bin/python -m pytest tests/test_wbc_preview.py -v

The load-bearing test is (a): both are pure additions, so any sequence of
plan()/preview() calls interleaved with act() must reproduce exactly what
act() alone would have done.
"""
from __future__ import annotations

import numpy as np

from himrobot.policies.ik_hitter import PREVIEW_DT, IKHitterPolicy
from himrobot.sim.scene import (
    BASE_NOMINAL_POS,
    BASE_NOMINAL_QUAT,
    DEFAULT_PARAMS,
    STAND_POSE,
    TableTennisEnv,
    sample_serve,
)

N_SERVES = 3
STEPS = 60
HORIZON = 10


def _rollout(seed: int, call_plan_and_preview: bool) -> list[np.ndarray]:
    env = TableTennisEnv(seed=seed, params={})
    rng = np.random.default_rng(seed)
    policy = IKHitterPolicy()
    actions = []
    for _ in range(N_SERVES):
        serve = sample_serve(rng, DEFAULT_PARAMS)
        obs = env.reset(serve=serve)
        policy.reset({"serve": serve, "params": DEFAULT_PARAMS})
        for _ in range(STEPS):
            if call_plan_and_preview:
                policy.plan(obs)
                policy.preview(obs, HORIZON)
            action = policy.act(obs)
            actions.append(action.copy())
            obs, _events = env.step(action)
    return actions


# ---------------------------------------------------------------------------
# (a) plan()/preview() never change what act() does.
# ---------------------------------------------------------------------------
def test_plan_and_preview_do_not_change_act():
    plain = _rollout(seed=0, call_plan_and_preview=False)
    interleaved = _rollout(seed=0, call_plan_and_preview=True)
    assert len(plain) == len(interleaved) == N_SERVES * STEPS
    for a, b in zip(plain, interleaved):
        assert np.array_equal(a, b)


def test_golden_param_actions_fixture_still_reproduces():
    """The stronger, project-wide version of (a): NumpyParamResidualPolicy's
    golden rollout (tools/make_golden_param_actions.py --check) must still
    reproduce tests/data/param_s3_actions.npz bit for bit -- plan()/preview()
    add nothing to act()'s own code path."""
    from tools.make_golden_param_actions import FIXTURE, rollout

    existing = np.load(FIXTURE, allow_pickle=False)
    fresh = rollout(list(existing["seeds"]))
    for key in ("seeds", "actions", "params", "names"):
        assert np.array_equal(fresh[key], existing[key]), key


# ---------------------------------------------------------------------------
# (b) No plan yet -> preview is all STAND_POSE.
# ---------------------------------------------------------------------------
def test_preview_before_any_plan_is_all_stand_pose():
    env = TableTennisEnv(seed=0, params={})
    policy = IKHitterPolicy()
    serve = sample_serve(np.random.default_rng(0), DEFAULT_PARAMS)
    obs = env.reset(serve=serve)
    policy.reset({"serve": serve, "params": DEFAULT_PARAMS})

    pre = policy.preview(obs, HORIZON)
    assert pre.shape == (HORIZON, 29)
    for row in pre:
        assert np.array_equal(row, STAND_POSE)


# ---------------------------------------------------------------------------
# (c)/(d) preview(obs, 1)'s one-step-ahead forecast matches what act() itself
# computes when handed the same plan and joint/base state one control step
# later, with nothing else changed. This isolates exactly what preview()'s
# open-loop assumption claims: given no new information, the forecast IS the
# real computation, not an approximation of it.
# ---------------------------------------------------------------------------
def _assert_preview_matches_synthetic_act(env: TableTennisEnv, seed: int,
                                          base_aware: bool) -> int:
    rng = np.random.default_rng(seed)
    serve = sample_serve(rng, env.params)
    obs = env.reset(serve=serve)

    live = IKHitterPolicy(base_aware=base_aware)
    live.reset({"serve": serve, "params": env.params})
    twin = IKHitterPolicy(base_aware=base_aware)  # never reset: only self._plan is used below

    checked = 0
    for _ in range(STEPS):
        action = live.act(obs)
        plan = live._plan
        if plan is not None:
            t_now = float(obs["time_s"])
            window_start = plan["t_contact"] - plan["speed_buffer"] - plan["t_pre"]
            window_end = plan["t_contact"] + plan["t_follow"]
            if window_start <= t_now + PREVIEW_DT <= window_end:
                pre = live.preview(obs, 1)
                twin._plan = plan  # the exact plan live just kept/produced
                obs_ahead = dict(obs)
                obs_ahead["time_s"] = t_now + PREVIEW_DT
                twin_action = twin.act(obs_ahead)
                assert np.allclose(pre[0], twin_action, atol=1e-9, rtol=0)
                checked += 1
        obs, _events = env.step(action)
    return checked


def test_preview_one_step_ahead_matches_a_synthetic_act_call_welded():
    env = TableTennisEnv(seed=0, params={})
    checked = sum(_assert_preview_matches_synthetic_act(env, seed, base_aware=False)
                 for seed in range(10))
    assert checked > 0, "no serve in this seed range entered a swing window"


def test_preview_one_step_ahead_matches_a_synthetic_act_call_free_base_aware():
    """Same claim on the free base with base_aware=True: preview() must run
    the target through to_shadow_frame exactly as act() does, using the
    obs's own base.position/orientation for every forecast step."""
    env = TableTennisEnv(seed=0, params={}, fixed_base=False)
    checked = sum(_assert_preview_matches_synthetic_act(env, seed, base_aware=True)
                 for seed in range(10))
    assert checked > 0, "no serve in this seed range entered a swing window"


def test_importing_preview_error_as_a_library_does_not_set_the_stance():
    """himrobot.sim.scene reads TT_ROBOT_Y once at import, so a module that
    exports the env var at import time would silently move the stance of
    every scene built afterwards in that process. himrobot.wbc.preview_error
    may default the shipped -0.10 only under its own __main__; imported as a
    library it must leave both the env var and scene.ROBOT_Y alone."""
    import os
    import subprocess
    import sys

    from himrobot.paths import REPO_ROOT

    env = {k: v for k, v in os.environ.items() if k != "TT_ROBOT_Y"}
    env["PYTHONPATH"] = str(REPO_ROOT)
    out = subprocess.run(
        [sys.executable, "-c",
         "import os, himrobot.wbc.preview_error; from himrobot.sim import scene; "
         "print(os.environ.get('TT_ROBOT_Y'), scene.ROBOT_Y)"],
        env=env, capture_output=True, text=True, check=True, cwd=str(REPO_ROOT))
    assert out.stdout.split() == ["None", "0.0"], out.stdout


def test_transform_used_by_preview_is_the_identity_at_the_nominal_pose():
    """Sanity check for the transform preview() calls: at the pose the free
    base resets to, to_shadow_frame is the identity (test_free_base.py pins
    the same claim for act())."""
    hitter = IKHitterPolicy()
    p_des = np.array([-1.20, -0.30, 1.05])
    n_des = np.array([0.6, -0.8, 0.0])
    p_out, n_out = hitter.to_shadow_frame(p_des, n_des, BASE_NOMINAL_POS, BASE_NOMINAL_QUAT)
    assert np.allclose(p_out, p_des, atol=1e-12, rtol=0)
    assert np.allclose(n_out, n_des, atol=1e-12, rtol=0)
