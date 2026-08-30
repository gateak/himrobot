"""(Re)generate tt_sim/tests/data/param_s3_actions.npz, the golden rollout of the shipped v1 policy.

    PYTHONPATH=. .venv/bin/python scripts/make_golden_param_actions.py [--check]

The fixture pins NumpyParamResidualPolicy (assets/policies/param_residual_s3.npz) bit for bit: for each serve
seed, 60 control steps of joint targets plus the decoded per-serve parameters, produced exactly the way
tt_sim/tests/test_hitter_v2.py::test_v1_checkpoint_reproduces_its_golden_actions replays them. It was first
generated before hitter v2 existed; regenerating it with later code is only legitimate when that code is
provably nominal-identical (the v2 parity tests) -- so use --check first, which fails if the current code
does not reproduce the committed file.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tt_sim import scene  # noqa: E402
from tt_sim.policies.param_residual_np import S3_WEIGHTS, NumpyParamResidualPolicy  # noqa: E402

FIXTURE = os.path.join(REPO, "tt_sim", "tests", "data", "param_s3_actions.npz")
STEPS = 60


def rollout(seeds) -> dict[str, np.ndarray]:
    policy = NumpyParamResidualPolicy(S3_WEIGHTS)
    env = scene.TableTennisEnv(seed=0, params={})
    actions, params = [], []
    for seed in seeds:
        obs = env.reset(serve=scene.sample_serve(np.random.default_rng(int(seed)), env.params))
        policy.reset({"params": env.params})
        for _ in range(STEPS):
            a = policy.act(obs)
            actions.append(np.asarray(a, dtype=np.float64).copy())
            obs, _events = env.step(a)
        params.append(np.array([policy.last_params[n] for n in policy.names]))
    return {"seeds": np.asarray(seeds, dtype=np.int64), "actions": np.stack(actions),
            "params": np.stack(params), "names": np.array(policy.names)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="serve seeds (default: the committed fixture's, else 0 3 7)")
    ap.add_argument("--check", action="store_true", help="compare with the committed fixture instead of writing")
    ap.add_argument("--out", default=FIXTURE)
    args = ap.parse_args(argv)

    existing = np.load(FIXTURE, allow_pickle=False) if os.path.isfile(FIXTURE) else None
    seeds = args.seeds or (list(existing["seeds"]) if existing is not None else [0, 3, 7])
    fresh = rollout(seeds)
    if args.check:
        if existing is None:
            print("no committed fixture to check against", file=sys.stderr)
            return 2
        same = all(np.array_equal(fresh[k], existing[k]) for k in ("seeds", "actions", "params", "names"))
        print("fixture reproduced bit for bit" if same else "MISMATCH: current code does not reproduce the fixture")
        return 0 if same else 1
    np.savez(args.out, **fresh)
    print(f"wrote {args.out}: seeds {seeds}, actions {fresh['actions'].shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
