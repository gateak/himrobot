"""Tests for the free-base (unwelded pelvis) option. Run with:
    .venv/bin/python -m pytest tt_sim/tests/test_free_base.py -v

Torch-free on purpose: HIM's mujoco worker has no torch, and every guard
here is about the scene and the hitter, not about a learned policy.

The load-bearing test is the parity one. The free base is *additive*: the
welded scene stays the project's scored configuration and must reproduce
its old evaluator.json per_trial byte for byte, with the new code present
and importable. Everything else here pins the new behaviour: the robot
stands, the four base.* observation keys appear only when the pelvis is
free, and the base-aware IK transform is the identity at the nominal pose.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tt_sim import scene
from tt_sim.evaluate import main as evaluate_main
from tt_sim.policies.ik_hitter import IKHitterPolicy
from tt_sim.scene import (
    BASE_NOMINAL_POS,
    BASE_NOMINAL_QUAT,
    FALL_PELVIS_Z,
    STAND_POSE,
    TableTennisEnv,
)

# Frozen from the pre-free-base main tree (commit 68d26c9); see its "note".
GOLDEN = os.path.join(os.path.dirname(__file__), "data", "free_base_welded_parity.json")

BASE_KEYS = ("base.position", "base.orientation", "base.linear_velocity", "base.angular_velocity")


# ---------------------------------------------------------------------------
# (a) Welded parity: the scored scene is untouched.
# ---------------------------------------------------------------------------
def test_welded_evaluator_per_trial_is_byte_identical_to_the_pre_free_base_tree(tmp_path):
    """The exact command the golden was recorded from must still reproduce it.

    per_trial is the bit-exact surface: it carries every serve, every event
    and every landing point. metrics gains the additive "falls" key and
    nothing else.
    """
    with open(GOLDEN) as fh:
        golden = json.load(fh)

    evaluate_main(["--policy", "ik_hitter", "--trials", "3", "--seed", "5",
                   "--out", str(tmp_path), "--no-trajectory"])
    with open(tmp_path / "evaluation" / "evaluator.json") as fh:
        doc = json.load(fh)

    assert json.dumps(doc["per_trial"], sort_keys=True) == json.dumps(golden["per_trial"], sort_keys=True)
    for key, value in golden["metrics"].items():
        assert doc["metrics"][key] == value, key
    assert set(doc["metrics"]) - set(golden["metrics"]) == {"falls"}
    assert doc["metrics"]["falls"] == 0
    assert doc["free_base"] is False


def test_welded_events_have_no_fell_key():
    """per_trial embeds events verbatim, so an extra key there would break parity."""
    env = TableTennisEnv(seed=0)
    env.reset()
    _obs, events = env.step(STAND_POSE)
    assert "fell" not in events


def test_welded_hitter_never_touches_the_base_aware_path():
    """With no base.* signals the transform is not entered at all, so the
    welded action is bit-identical to a hitter built with base_aware=False."""
    aware, unaware = IKHitterPolicy(base_aware=True), IKHitterPolicy(base_aware=False)
    env = TableTennisEnv(seed=3)
    serve = scene.sample_serve(np.random.default_rng(3), env.params)

    for policy in (aware, unaware):
        policy.reset({"params": env.params})
    actions = {}
    for name, policy in (("aware", aware), ("unaware", unaware)):
        obs = env.reset(serve=serve)
        rolled = []
        for _ in range(150):
            action = policy.act(obs)
            rolled.append(action.copy())
            obs, _ = env.step(action)
        actions[name] = np.array(rolled)

    assert np.array_equal(actions["aware"], actions["unaware"])


# ---------------------------------------------------------------------------
# (b) The free-base robot stands.
# ---------------------------------------------------------------------------
def test_free_base_holds_stand_pose_for_three_seconds():
    env = TableTennisEnv(seed=0, fixed_base=False)
    obs = env.reset()
    assert np.allclose(obs["base.position"], BASE_NOMINAL_POS)
    assert np.allclose(obs["base.orientation"], BASE_NOMINAL_QUAT)
    assert np.allclose(obs["base.linear_velocity"], 0.0)
    assert np.allclose(obs["base.angular_velocity"], 0.0)

    events = {}
    for _ in range(150):  # 3 s at 50 Hz
        obs, events = env.step(STAND_POSE)

    assert obs["base.position"][2] > 0.70, obs["base.position"]
    assert events["fell"] is False


def test_free_base_reset_restores_the_nominal_pose_after_a_disturbance():
    """Every reset re-plants the pelvis, so trial k+1 never inherits trial k's fall."""
    env = TableTennisEnv(seed=0, fixed_base=False)
    env.reset()
    for _ in range(60):  # shove it over with a hard one-sided arm command
        env.step(STAND_POSE + 1.2)

    obs = env.reset()
    assert np.allclose(obs["base.position"], BASE_NOMINAL_POS)
    assert np.allclose(obs["base.orientation"], BASE_NOMINAL_QUAT)
    assert np.allclose(obs["base.linear_velocity"], 0.0)
    assert np.allclose(obs["base.angular_velocity"], 0.0)


def test_fell_latches_once_the_pelvis_drops_and_the_episode_keeps_running():
    """The fall latch is a diagnostic: it never shortens the episode or scores."""
    env = TableTennisEnv(seed=0, fixed_base=False)
    env.reset()
    # Drive the knees out from under it rather than asserting on a specific
    # physical collapse: what is pinned is latch-and-stay, not the fall itself.
    env._events["fell"] = False
    env.data.qpos[env._base_qpos_adr + 2] = FALL_PELVIS_Z - 0.05
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    env._update_events()
    assert env._events["fell"] is True

    # Latched: it stays True after the pelvis comes back up.
    env.data.qpos[env._base_qpos_adr + 2] = scene.ROBOT_Z
    mujoco.mj_forward(env.model, env.data)
    env._update_events()
    assert env._events["fell"] is True

    # Scoring is untouched by the fall.
    assert scene.legal_return({"contact_paddle": True, "crossed_net": True,
                               "bounce_opponent_side": True, "fell": True}) is True


# ---------------------------------------------------------------------------
# (c) The observation contract.
# ---------------------------------------------------------------------------
def test_only_the_free_base_obs_carries_the_base_keys():
    welded = TableTennisEnv(seed=0).reset()
    free = TableTennisEnv(seed=0, fixed_base=False).reset()

    assert not any(k.startswith("base.") for k in welded)
    for key in BASE_KEYS:
        assert key in free, key
    # Additive: the free obs is the welded obs plus exactly the base keys.
    assert set(free) - set(welded) == set(BASE_KEYS)
    assert set(welded) - set(free) == set()

    assert free["base.position"].shape == (3,)
    assert free["base.orientation"].shape == (4,)
    assert free["base.linear_velocity"].shape == (3,)
    assert free["base.angular_velocity"].shape == (3,)


def test_fixed_base_defaults_to_true():
    assert TableTennisEnv(seed=0).fixed_base is True
    assert TableTennisEnv(seed=0, fixed_base=False).fixed_base is False


# ---------------------------------------------------------------------------
# (d) The base-aware transform.
# ---------------------------------------------------------------------------
def test_transform_is_the_identity_at_the_nominal_pose():
    hitter = IKHitterPolicy()
    p_des = np.array([-1.20, -0.30, 1.05])
    n_des = np.array([0.6, -0.8, 0.0])

    p_out, n_out = hitter.to_shadow_frame(p_des, n_des, BASE_NOMINAL_POS, BASE_NOMINAL_QUAT)

    assert np.allclose(p_out, p_des, atol=1e-12, rtol=0)
    assert np.allclose(n_out, n_des, atol=1e-12, rtol=0)


def test_a_shifted_base_shifts_the_target_by_the_negative():
    """The target is pelvis-relative, so moving the pelvis +d moves the
    shadow-frame target -d: the arm is asked for the same reach it always was."""
    hitter = IKHitterPolicy()
    p_des = np.array([-1.20, -0.30, 1.05])
    n_des = np.array([0.6, -0.8, 0.0])
    shift = np.array([0.05, -0.02, -0.01])

    p_out, n_out = hitter.to_shadow_frame(p_des, n_des, BASE_NOMINAL_POS + shift, BASE_NOMINAL_QUAT)

    assert np.allclose(p_out - p_des, -shift, atol=1e-12, rtol=0)
    assert np.allclose(n_out, n_des, atol=1e-12, rtol=0)  # a pure translation cannot turn the blade


def test_a_rotated_base_preserves_pelvis_relative_geometry():
    hitter = IKHitterPolicy()
    p_des = np.array([-1.20, -0.30, 1.05])
    n_des = np.array([0.6, -0.8, 0.0])
    yaw = 0.3
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])  # wxyz, about +z

    p_out, n_out = hitter.to_shadow_frame(p_des, n_des, BASE_NOMINAL_POS, quat)

    # A rotation about the pelvis changes direction, never distance or length.
    assert np.isclose(np.linalg.norm(p_out - BASE_NOMINAL_POS),
                      np.linalg.norm(p_des - BASE_NOMINAL_POS), atol=1e-12, rtol=0)
    assert np.isclose(np.linalg.norm(n_out), np.linalg.norm(n_des), atol=1e-12, rtol=0)
    assert not np.allclose(p_out, p_des)


def test_tt_base_aware_env_var_disables_the_transform(monkeypatch):
    monkeypatch.delenv("TT_BASE_AWARE", raising=False)
    assert IKHitterPolicy().base_aware is False  # off unless asked: measured worse without a balance controller
    monkeypatch.setenv("TT_BASE_AWARE", "0")
    assert IKHitterPolicy().base_aware is False
    monkeypatch.setenv("TT_BASE_AWARE", "1")
    assert IKHitterPolicy().base_aware is True
    assert IKHitterPolicy(base_aware=False).base_aware is False  # an explicit argument beats the env var
    monkeypatch.delenv("TT_BASE_AWARE", raising=False)
    assert IKHitterPolicy(base_aware=True).base_aware is True


def test_base_aware_is_read_once_at_construction(monkeypatch):
    """A/B runs set the env var before launching; flipping it mid-episode
    must not change a policy that is already running."""
    monkeypatch.setenv("TT_BASE_AWARE", "0")
    hitter = IKHitterPolicy()
    monkeypatch.setenv("TT_BASE_AWARE", "1")
    assert hitter.base_aware is False


# ---------------------------------------------------------------------------
# (e) The evaluator CLI.
# ---------------------------------------------------------------------------
def test_evaluate_free_base_records_falls_and_the_flag(tmp_path):
    metrics = evaluate_main(["--policy", "ik_hitter", "--trials", "3", "--seed", "2000",
                             "--free-base", "--out", str(tmp_path), "--no-trajectory"])
    with open(tmp_path / "evaluation" / "evaluator.json") as fh:
        doc = json.load(fh)

    assert doc["free_base"] is True
    assert "falls" in doc["metrics"]
    assert 0 <= doc["metrics"]["falls"] <= 3
    assert metrics == doc["metrics"]
    for trial in doc["per_trial"]:
        assert "fell" in trial["events"]


# ---------------------------------------------------------------------------
# The shipped runtime policy forwards the base signals to its inner hitter.
# ---------------------------------------------------------------------------
def test_numpy_param_residual_policy_passes_base_signals_through():
    """NumpyParamResidualPolicy hands the whole obs dict to its IKHitterPolicy,
    so base awareness reaches the shipped policy with no change of its own.
    Pinned by behaviour: the same serve must produce different actions with the
    transform on and off, which is only possible if base.* got through."""
    from tt_sim.policies.param_residual_np import NumpyParamResidualPolicy

    env = TableTennisEnv(seed=0, fixed_base=False)
    serve = scene.sample_serve(np.random.default_rng(2000), env.params)

    rolled = {}
    for name, aware in (("on", True), ("off", False)):
        policy = NumpyParamResidualPolicy()
        policy.hitter.base_aware = aware
        obs = env.reset(serve=serve)
        policy.reset({"params": env.params})
        actions = []
        for _ in range(150):
            action = policy.act(obs)
            actions.append(action.copy())
            obs, _ = env.step(action)
        rolled[name] = np.array(actions)

    # serve_features reads ball.position/velocity in world coordinates, so the
    # per-serve knobs are identical either way; only the IK target differs.
    assert policy.last_params is not None
    assert not np.array_equal(rolled["on"], rolled["off"])


@pytest.mark.parametrize("key", ["legal_return_rate", "target_hit_rate", "landing_error_m",
                                 "reaction_latency_ms", "n_trials", "contacts", "falls"])
def test_metric_keys_are_present_in_both_scenes(tmp_path, key):
    evaluate_main(["--policy", "hold", "--trials", "1", "--seed", "0",
                   "--out", str(tmp_path / "welded"), "--no-trajectory"])
    evaluate_main(["--policy", "hold", "--trials", "1", "--seed", "0",
                   "--out", str(tmp_path / "free"), "--free-base", "--no-trajectory"])
    for which in ("welded", "free"):
        with open(tmp_path / which / "evaluation" / "evaluator.json") as fh:
            assert key in json.load(fh)["metrics"], (which, key)
