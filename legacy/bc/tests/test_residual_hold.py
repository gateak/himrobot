"""Zero-init residual heads must exactly reproduce "hold current joints" once
denormalized -- the invariant legacy/bc/model.py's docstring claims and
him_adapter.BCPolicy.infer relies on. The identity holds only if obs/action
norm stats agree on the shared joint columns (bc.dataset.compute_norm_stats).
Tests deliberately use different (but joint-angle-realistic) obs/act scales, so
reverting to independently-fit obs/act stats would blow past 1e-6; offsets stay
modest since huge ones would trip float32 rounding on their own, regardless of
any real bug.
"""

from __future__ import annotations

import numpy as np
import torch

from ..dataset import compute_norm_stats
from ..model import MLPPolicy, TinyTransformerPolicy


def _denormalize_like_deploy(y: torch.Tensor, act_mean, act_std) -> np.ndarray:
    """Mirrors him_adapter.policy.BCPolicy.infer's post-model denormalization."""
    return (y.detach().numpy() * act_std) + act_mean


def test_zero_init_mlp_residual_equals_raw_hold_to_1e6():
    rng = np.random.default_rng(0)
    n, obs_dim, act_dim = 300, 12, 7
    # Joint-angle-scale (radians) values, but with obs and act on deliberately
    # different scale/offset, so the two normalizations would disagree if
    # fit independently (as the pre-fix code did).
    X = (0.4 + 0.15 * rng.standard_normal((n, 1, obs_dim))).astype(np.float32)
    Y = (-0.2 + 0.08 * rng.standard_normal((n, act_dim))).astype(np.float32)

    obs_mean, obs_std, act_mean, act_std = compute_norm_stats(X, Y, act_dim)
    # The action stats and the raw obs scale genuinely differ -- confirms this
    # test would catch a regression to independent obs/act normalization.
    assert np.mean(np.abs(act_std)) < np.mean(np.abs(obs_std[act_dim:]))

    Xn = (X - obs_mean) / obs_std
    Xt = torch.from_numpy(Xn.astype(np.float32)).squeeze(1)  # [N, obs_dim], as train.py feeds the MLP

    model = MLPPolicy(obs_dim, act_dim, hidden=(16,), residual=True)  # zero-init head (default)
    with torch.no_grad():
        y = model(Xt)
    y_denorm = _denormalize_like_deploy(y, act_mean, act_std)

    raw_hold = X[:, -1, :act_dim]  # current joints, exactly what "hold" would return
    np.testing.assert_allclose(y_denorm, raw_hold, atol=1e-6)


def test_zero_init_transformer_residual_equals_raw_hold_to_1e6():
    rng = np.random.default_rng(1)
    n, context, obs_dim, act_dim = 200, 4, 10, 6
    X = (0.05 + 0.3 * rng.standard_normal((n, context, obs_dim))).astype(np.float32)
    Y = (1.0 + 0.5 * rng.standard_normal((n, act_dim))).astype(np.float32)

    obs_mean, obs_std, act_mean, act_std = compute_norm_stats(X, Y, act_dim)
    Xn = ((X - obs_mean) / obs_std).astype(np.float32)
    Xt = torch.from_numpy(Xn)  # [N, T, obs_dim], as train.py feeds the transformer

    model = TinyTransformerPolicy(obs_dim, act_dim, context=context, d_model=16, nhead=2, layers=1,
                                   residual=True)
    with torch.no_grad():
        y = model(Xt)
    y_denorm = _denormalize_like_deploy(y, act_mean, act_std)

    raw_hold = X[:, -1, :act_dim]  # current joints at the last (most recent) context step
    np.testing.assert_allclose(y_denorm, raw_hold, atol=1e-6)


def test_compute_norm_stats_does_not_mutate_inputs():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50, 1, 8)).astype(np.float32)
    Y = rng.standard_normal((50, 5)).astype(np.float32)
    X_before, Y_before = X.copy(), Y.copy()
    compute_norm_stats(X, Y, act_dim=5)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(Y, Y_before)
