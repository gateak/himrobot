"""Tests for himrobot.rl.ppo + himrobot.rl.policy. Run with:
    .venv/bin/python -m pytest tests/test_rl_ppo.py -v
"""
from __future__ import annotations

import os
import shutil
import time

import csv

import numpy as np
import pytest

torch = pytest.importorskip('torch')  # HIM's venv has no torch; these tests are RL-only
from himrobot.rl.ppo import train
from himrobot.rl.policy import PPOPolicy
from himrobot.sim.scene import DEFAULT_PARAMS, TableTennisEnv

OUT_DIR = os.path.join("work", "tt", "rl", "test_ppo_smoke")


def _clean_out():
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(OUT_DIR, exist_ok=True)


def test_two_updates_on_four_envs_under_60s_and_writes_checkpoint():
    _clean_out()
    rollout_steps = 64
    n_envs = 4
    start = time.time()
    result = train(
        n_envs=n_envs, total_steps=rollout_steps * n_envs * 2, seed=0, out=OUT_DIR,
        rollout_steps=rollout_steps, eval_every_steps=1,
    )
    elapsed = time.time() - start
    print(f"\n2 PPO updates on {n_envs} envs in {elapsed:.2f}s")
    assert elapsed < 60.0
    assert result["n_updates"] == 2
    assert os.path.exists(os.path.join(OUT_DIR, "final.pt"))
    assert os.path.exists(os.path.join(OUT_DIR, "train_log.csv"))
    with open(os.path.join(OUT_DIR, "train_log.csv")) as fh:
        rows = fh.read().strip().splitlines()
    assert len(rows) == 3  # header + 2 update rows (eval_every_steps=1)


def test_ppo_policy_loads_checkpoint_and_returns_finite_29_action():
    ckpt = os.path.join(OUT_DIR, "final.pt")
    assert os.path.exists(ckpt), "run the training test first (same module) to produce a checkpoint"
    policy = PPOPolicy(ckpt_path=ckpt)
    env = TableTennisEnv(seed=0, params={})
    obs = env.reset()
    policy.reset({"params": DEFAULT_PARAMS})
    for _ in range(5):
        action = policy.act(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (29,)
        assert np.isfinite(action).all()
        obs, _ = env.step(action)


def test_ppo_policy_requires_a_checkpoint_source():
    os.environ.pop("TT_RL_CKPT", None)
    with pytest.raises(ValueError):
        PPOPolicy()


def test_det_eval_selection_and_log_std_anneal(tmp_path):
    from himrobot.rl import ppo
    out = tmp_path / "run"
    ppo.train(n_envs=4, total_steps=256, seed=3, out=str(out), eval_every_steps=64, rollout_steps=16,
              residual=True, residual_scale=0.2, det_eval_episodes=3, log_std_init=-1.0, log_std_final=-2.0)
    rows = list(csv.DictReader(open(out / "train_log.csv")))
    assert rows and "det_legal_rate" in rows[0]
    det = [float(r["det_legal_rate"]) for r in rows]
    assert all(0.0 <= d <= 100.0 for d in det)
    assert (out / "best.pt").is_file()
    ck = torch.load(out / "final.pt", map_location="cpu", weights_only=False)
    assert float(ck["model_state"]["log_std"].max()) <= -2.0 + 1e-6
