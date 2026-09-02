"""Tests for the per-serve parameter-residual action space. Run with:
    .venv/bin/python -m pytest tests/test_param_residual.py -v

The load-bearing one is step-zero parity: a hitter at nominal serve params,
and an untrained (zero-init-head) parameter policy, must both reproduce the
plain IKHitterPolicy exactly -- otherwise "residual" is a misnomer and every
training run starts from an unmeasured baseline.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from himrobot.policies.ik_hitter import SERVE_PARAM_NOMINAL, IKHitterPolicy
from himrobot.sim.scene import DEFAULT_PARAMS, TableTennisEnv, sample_serve

torch = pytest.importorskip("torch")

from himrobot.rl import param_residual as pr  # noqa: E402  (after importorskip)
from himrobot.rl import param_train  # noqa: E402

STEPS = 150


def _roll(policy, seed=0, steps=STEPS):
    """One serve through a policy; returns the stacked per-step actions."""
    env = TableTennisEnv(seed=seed, params={})
    serve = sample_serve(np.random.default_rng(seed), DEFAULT_PARAMS)
    obs = env.reset(serve=serve)
    policy.reset({"serve": serve, "params": DEFAULT_PARAMS})
    actions = []
    for _ in range(steps):
        action = policy.act(obs)
        actions.append(np.asarray(action, dtype=np.float64).copy())
        obs, _events = env.step(action)
    return np.stack(actions)


# -- parity ---------------------------------------------------------------
def test_nominal_serve_params_match_plain_hitter():
    """Explicitly setting every serve param to its nominal value must be a
    bit-for-bit no-op against the untouched hitter."""
    plain = _roll(IKHitterPolicy(), seed=0)

    tuned = IKHitterPolicy()
    tuned.set_serve_params(SERVE_PARAM_NOMINAL)
    assert np.array_equal(plain, _roll(tuned, seed=0))

    # ...and the same via the u -> params mapping the network actually uses.
    from_u = pr.hitter_from_u(np.zeros(pr.K_PARAMS))
    assert np.array_equal(plain, _roll(from_u, seed=0))


def test_nominal_is_the_only_identity_construction_path():
    """The constructor kwarg path and the default path agree at nominal."""
    plain = _roll(IKHitterPolicy(), seed=1)
    ctor = _roll(IKHitterPolicy(serve_params=dict(SERVE_PARAM_NOMINAL)), seed=1)
    assert np.array_equal(plain, ctor)


def test_untrained_param_policy_matches_plain_hitter(tmp_path):
    """Step zero of training == the plain hitter: a fresh ParamActorCritic has
    a zero-init head, so its deterministic mean is 0 -> u = 0 -> nominal."""
    from himrobot.rl.policy import ParamResidualPolicy

    model = pr.ParamActorCritic()
    feat_rms = pr.RunningMeanStd((pr.FEATURE_DIM,))
    ckpt = str(tmp_path / "zero.pt")
    pr.save_checkpoint(ckpt, model, feat_rms, episodes=0)

    policy = ParamResidualPolicy(ckpt_path=ckpt)
    assert np.array_equal(_roll(IKHitterPolicy(), seed=2), _roll(policy, seed=2))
    assert policy.last_params == pytest.approx(SERVE_PARAM_NOMINAL)


def test_param_policy_reads_checkpoint_from_env_var(tmp_path, monkeypatch):
    """himrobot.eval.evaluate constructs policies with a bare cls()."""
    from himrobot.rl.policy import ParamResidualPolicy

    ckpt = str(tmp_path / "zero.pt")
    pr.save_checkpoint(ckpt, pr.ParamActorCritic(), pr.RunningMeanStd((pr.FEATURE_DIM,)), 0)
    monkeypatch.setenv("TT_RL_CKPT", ckpt)
    assert ParamResidualPolicy().model.k_params == pr.K_PARAMS


def test_param_policy_detects_a_ball_reset_from_the_obs_alone(tmp_path):
    """No reset() call at all: a time_s that stops increasing is an episode
    boundary, and the policy must re-decide there."""
    from himrobot.rl.policy import ParamResidualPolicy

    ckpt = str(tmp_path / "zero.pt")
    pr.save_checkpoint(ckpt, pr.ParamActorCritic(), pr.RunningMeanStd((pr.FEATURE_DIM,)), 0)
    policy = ParamResidualPolicy(ckpt_path=ckpt)

    env = TableTennisEnv(seed=3, params={})
    obs = env.reset(serve=sample_serve(np.random.default_rng(3), DEFAULT_PARAMS))
    for _ in range(10):
        obs, _ = env.step(policy.act(obs))
    assert policy._pending is False

    obs = env.reset(serve=sample_serve(np.random.default_rng(4), DEFAULT_PARAMS))
    assert obs["time_s"] == 0.0
    policy.act(obs)  # sees time_s go backwards -> re-decides
    assert policy._prev_time_s == 0.0


# -- action space ---------------------------------------------------------
def test_params_are_bounded_and_nominal_at_zero():
    assert pr.unsquash_to_params(np.zeros(pr.K_PARAMS)) == pytest.approx(SERVE_PARAM_NOMINAL)
    for extreme in (1.0, -1.0):
        values = pr.unsquash_to_params(np.full(pr.K_PARAMS, extreme))
        for name, kind, bound in pr.PARAM_SPECS:
            if kind == "add":
                assert abs(values[name]) <= bound + 1e-9
            else:
                assert 1.0 / bound - 1e-9 <= values[name] <= bound + 1e-9


def test_every_param_actually_changes_the_swing():
    """A knob that no longer reaches the hitter would silently shrink the
    action space to fewer than K dimensions."""
    baseline = _roll(IKHitterPolicy(), seed=0, steps=60)
    for i, (name, _kind, _bound) in enumerate(pr.PARAM_SPECS):
        u = np.zeros(pr.K_PARAMS)
        u[i] = 0.9
        assert not np.array_equal(baseline, _roll(pr.hitter_from_u(u), seed=0, steps=60)), name


def test_set_serve_params_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown serve params"):
        IKHitterPolicy().set_serve_params(no_such_knob=1.0)


def test_serve_features_shape_and_determinism():
    env = TableTennisEnv(seed=5, params={})
    obs = env.reset(serve=sample_serve(np.random.default_rng(5), DEFAULT_PARAMS))
    target = pr.default_target()
    feats = pr.serve_features(obs, target)
    assert feats.shape == (pr.FEATURE_DIM,) and feats.dtype == np.float32
    assert np.array_equal(feats, pr.serve_features(obs, target))


# -- env + trainer --------------------------------------------------------
def test_vec_param_env_runs_and_reports_episodes():
    drawn = []

    def decide(features):
        assert features.shape == (pr.FEATURE_DIM,)
        u = np.full(pr.K_PARAMS, 0.1)
        drawn.append(u)
        return {"u": u, "tag": len(drawn)}

    vec = pr.VecTTParam(n_envs=2, seed=0, decide_fn=decide)
    assert len(drawn) == 2  # one draw per env at construction
    finished = 0
    for _ in range(pr.MAX_EPISODE_STEPS + 1):
        _obs, rew, done, infos = vec.step()
        assert rew.shape == (2,) and done.shape == (2,)
        for info in infos:
            if "episode" in info:
                finished += 1
                assert info["episode"]["decision"]["tag"] >= 1
                assert info["episode"]["features"].shape == (pr.FEATURE_DIM,)
    assert finished >= 2


def test_vec_param_env_without_decide_fn_is_the_plain_hitter():
    vec = pr.VecTTParam(n_envs=1, seed=0, decide_fn=None)
    for _ in range(5):
        vec.step()
    hitter = vec._hitters[0]
    assert {name: getattr(hitter, name) for name in SERVE_PARAM_NOMINAL} == pytest.approx(SERVE_PARAM_NOMINAL)


def test_train_smoke_writes_log_and_checkpoints(tmp_path):
    out = str(tmp_path / "run")
    result = param_train.train(n_envs=2, episodes=4, seed=0, out=out, eval_every=1,
                               batch_episodes=2, minutes=1.0)
    assert result["episodes"] >= 1
    assert os.path.exists(os.path.join(out, "final.pt"))
    with open(os.path.join(out, "train_log.csv")) as fh:
        header = fh.readline().strip().split(",")
    assert header[:5] == ["episode", "wall_s", "mean_return", "legal_return_rate", "contact_rate"]
    assert header[5:9] == ["det_legal_rate", "det_contact_rate", "det_target_rate", "det_landing_error_m"]
    assert header[9:] == [f"u_{name}" for name in pr.PARAM_NAMES]  # v1 has no gates -> no blend_* columns

    model, feat_rms, specs = pr.load_checkpoint(os.path.join(out, "final.pt"),
                                                torch.device("cpu"))
    assert model.k_params == pr.K_PARAMS
    assert specs == pr.PARAM_SPECS
    assert feat_rms.mean.shape == (pr.FEATURE_DIM,)
