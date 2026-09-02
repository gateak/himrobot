"""Tests for himrobot/rl/demos.py + himrobot/rl/bc.py + himrobot/rl/policy.py:BCPolicy.

Kept small (20 serves, 2 epochs) for CI speed -- himrobot/rl/README/task
report has the real 2000-trial/20-epoch numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')  # HIM's venv has no torch; these tests are RL-only
from himrobot.rl import bc as bc_mod
from himrobot.rl import demos as demos_mod
from himrobot.rl.env import ACT_DIM, OBS_DIM
from himrobot.rl.policy import BCPolicy
from himrobot.sim.scene import STAND_POSE


@pytest.fixture(scope="module")
def small_demos():
    return demos_mod.roll_demos(n_trials=20, seed=0)


def test_demos_shapes_20_serves(small_demos):
    data = small_demos
    assert data["obs"].shape == (data["obs"].shape[0], OBS_DIM)
    assert data["act"].shape == (data["act"].shape[0], ACT_DIM)
    assert data["obs"].shape[0] == data["act"].shape[0]

    n_kept = len(data["episode_starts"])
    assert len(data["legal"]) == n_kept
    assert len(data["trial_contact"]) == 20
    assert len(data["trial_legal"]) == 20
    # IKHitterPolicy is a real model-based hitter, not degenerate -- expect
    # at least a few contacts out of 20 seeded serves.
    assert data["trial_contact"].sum() > 0

    if n_kept > 0:
        assert data["episode_starts"][0] == 0
        assert data["episode_starts"][-1] < data["obs"].shape[0]
        if n_kept > 1:
            assert np.all(np.diff(data["episode_starts"]) > 0)

    assert np.isfinite(data["obs"]).all()
    assert np.isfinite(data["act"]).all()
    assert np.all(np.abs(data["act"]) <= 1.0 + 1e-6)


def test_bc_trains_two_epochs_and_saved_policy_returns_finite_action(small_demos, tmp_path):
    data = small_demos
    if data["obs"].shape[0] == 0:
        pytest.skip("no contact trials in this small demo run -- nothing to train on")

    actor, history = bc_mod.train(data, epochs=2, seed=0, device="cpu")
    assert len(history["train_loss"]) == 2
    assert all(np.isfinite(v) for v in history["train_loss"])

    ckpt_path = tmp_path / "bc_test.pt"
    bc_mod.save_checkpoint(actor, str(ckpt_path), history=history, seed=0, epochs=2, val_frac=bc_mod.DEFAULT_VAL_FRAC)
    assert ckpt_path.exists()

    policy = BCPolicy(checkpoint_path=str(ckpt_path), device="cpu")
    policy.reset({"params": {}, "serve": None, "trial": 0, "seed": 0})

    dummy_obs = {
        "robot.joint_position": STAND_POSE.copy(),
        "robot.joint_velocity": np.zeros(29),
        "ball.position": np.array([0.5, 0.0, 1.0]),
        "ball.velocity": np.array([-1.0, 0.0, 0.0]),
        "paddle.position": np.array([-1.0, 0.0, 1.0]),
        "paddle.quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        "time_s": 0.1,
    }
    action = np.asarray(policy.act(dummy_obs))
    assert action.shape == (29,)
    assert np.isfinite(action).all()

    # A second call exercises the prev_action feedback path (reset() ->
    # first act() -> second act() with a nonzero prev_action).
    action2 = np.asarray(policy.act(dummy_obs))
    assert action2.shape == (29,)
    assert np.isfinite(action2).all()
