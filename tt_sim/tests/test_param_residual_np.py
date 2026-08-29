"""The numpy runtime must reproduce the torch ParamResidualPolicy action for action."""
import os

import numpy as np
import pytest

from tt_sim import scene

torch = pytest.importorskip("torch")


def _rollout(policy, seeds):
    from tt_sim.scene import TableTennisEnv

    env = TableTennisEnv(seed=0, params={})
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        obs = env.reset(serve=scene.sample_serve(rng, env.params))
        policy.reset({"params": env.params})
        for _ in range(40):
            a = policy.act(obs)
            out.append(np.asarray(a, dtype=np.float64).copy())
            obs, _events = env.step(a)
        out.append(np.asarray(list(policy.last_params.values()), dtype=np.float64))
    return out


def test_numpy_policy_matches_torch_policy(tmp_path):
    from scripts.export_param_policy import main as export
    from tt_sim.policies.param_residual_np import NumpyParamResidualPolicy
    from tt_sim.rl import param_residual
    from tt_sim.rl.policy import ParamResidualPolicy
    from tt_sim.rl.ppo import RunningMeanStd

    # A random (untrained) checkpoint exercises every code path without needing a training run.
    torch.manual_seed(0)
    model = param_residual.ParamActorCritic()
    with torch.no_grad():
        for p in model.actor.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    rms = RunningMeanStd((param_residual.FEATURE_DIM,))
    rms.mean = np.random.default_rng(1).normal(size=param_residual.FEATURE_DIM)
    rms.var = np.random.default_rng(2).uniform(0.5, 2.0, size=param_residual.FEATURE_DIM)
    ckpt = tmp_path / "ck.pt"
    param_residual.save_checkpoint(str(ckpt), model, rms, 0)
    npz = tmp_path / "ck.npz"
    export(str(ckpt), str(npz))

    a_torch = _rollout(ParamResidualPolicy(str(ckpt)), seeds=(3, 4, 5))
    a_np = _rollout(NumpyParamResidualPolicy(str(npz)), seeds=(3, 4, 5))
    assert len(a_torch) == len(a_np)
    for x, y in zip(a_torch, a_np):
        np.testing.assert_allclose(x, y, rtol=0, atol=2e-3)  # float32 matmul order differs numpy vs torch; IK amplifies to ~1e-4 rad


def test_shipped_weights_load_when_present():
    from tt_sim.policies.param_residual_np import DEFAULT_WEIGHTS, NumpyParamResidualPolicy

    if not os.path.isfile(DEFAULT_WEIGHTS):
        pytest.skip("no shipped weights")
    pol = NumpyParamResidualPolicy()
    assert len(pol.names) == 10
