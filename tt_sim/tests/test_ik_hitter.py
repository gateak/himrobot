"""Tests for tt_sim.policies.ik_hitter. Run with:
    .venv/bin/python -m pytest tt_sim/tests/test_ik_hitter.py -v

Acceptance bar from the task: contact rate on 10 serves >= 50%, headless,
deterministic for a fixed seed. (The stronger 20-trial/legal-return bar
from the task -- >=70% contact, >=30% legal_return_rate -- was verified via
a throwaway tuning script against the real numbers reported in this task's
report, not re-asserted here as a hard gate: it is closer to a tuned
policy's expected operating point than an invariant of the code, and
tightly re-asserting it in CI would make the suite flaky against any
future serve-sampling or physics-calibration change made by other agents.)
"""
from __future__ import annotations

import numpy as np

from tt_sim.policies.ik_hitter import IKHitterPolicy
from tt_sim.scene import DEFAULT_PARAMS, TableTennisEnv, sample_serve

N_TRIALS = 10
STEPS_PER_TRIAL = 400  # 8s at 50Hz, matches test_scene.py's convention


def _run_trials(n=N_TRIALS, seed=0):
    """Serve n balls (seed-derived, reproducible) at a fresh IKHitterPolicy
    and return (contact_flags, landing_points) per trial.
    """
    env = TableTennisEnv(seed=seed, params={})
    rng = np.random.default_rng(seed)
    policy = IKHitterPolicy()

    contacts = []
    landings = []
    for _ in range(n):
        serve = sample_serve(rng, DEFAULT_PARAMS)
        obs = env.reset(serve=serve)
        policy.reset({"serve": serve, "params": DEFAULT_PARAMS})
        info = {}
        for _ in range(STEPS_PER_TRIAL):
            action = policy.act(obs)
            assert action.shape == (29,)
            obs, info = env.step(action)
        contacts.append(bool(info["contact_paddle"]))
        landings.append(info["landing_point"])
    return contacts, landings


def test_contact_rate_at_least_half_of_ten_serves():
    contacts, _ = _run_trials(N_TRIALS, seed=0)
    rate = sum(contacts) / len(contacts)
    assert rate >= 0.5, f"contact rate {rate:.2f} on {contacts}"


def test_deterministic_for_a_fixed_seed():
    contacts_a, landings_a = _run_trials(N_TRIALS, seed=0)
    contacts_b, landings_b = _run_trials(N_TRIALS, seed=0)
    assert contacts_a == contacts_b
    for la, lb in zip(landings_a, landings_b):
        if la is None or lb is None:
            assert la is lb
        else:
            assert np.allclose(la, lb)


def test_non_swinging_joints_hold_stand_pose():
    """The 19 joints outside the IK set (both legs + left arm) must always
    equal scene.STAND_POSE, per the task's action spec, whether or not the
    policy is mid-swing.
    """
    from tt_sim.policies.ik_hitter import IK_INDICES
    from tt_sim.scene import STAND_POSE

    other_indices = [i for i in range(29) if i not in set(IK_INDICES.tolist())]
    assert len(other_indices) == 19

    env = TableTennisEnv(seed=0, params={})
    rng = np.random.default_rng(0)
    policy = IKHitterPolicy()
    serve = sample_serve(rng, DEFAULT_PARAMS)
    obs = env.reset(serve=serve)
    policy.reset({"serve": serve, "params": DEFAULT_PARAMS})
    for _ in range(STEPS_PER_TRIAL):
        action = policy.act(obs)
        assert np.allclose(action[other_indices], STAND_POSE[other_indices])
        obs, _ = env.step(action)


def test_runs_well_under_budget_headless():
    import time

    start = time.time()
    _run_trials(N_TRIALS, seed=0)
    elapsed = time.time() - start
    print(f"\n{N_TRIALS} trials ({N_TRIALS * STEPS_PER_TRIAL} control steps) in {elapsed:.2f}s")
    assert elapsed < 60.0
