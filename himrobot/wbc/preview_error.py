"""Measures IKHitterPolicy.preview()'s open-loop forecast error against the
realised swing (docs/design/whole_body_v1.md sec 5.3, WB1).

Runs the shipped v3 policy (assets/policies/param_residual_v3_s0final.npz)
on N_TRIALS welded serves at the shipped stance (TT_ROBOT_Y=-0.10) for
STEPS_PER_TRIAL control steps each, recording act()'s realised action and
preview(obs, HORIZON)'s forecast at every step. For each recorded step i,
preview_i[k-1] is a k-step-ahead forecast made at step i; the realised
action at step i+k (actions[i+k], recorded when that step was actually
reached) is what really happened. This compares the two, over the 10 IK
joints (waist yaw/roll/pitch + right arm), restricted to (i, k) pairs where
BOTH rows are inside a swing window (neither equals scene.STAND_POSE) --
"inside swing windows only": the forecast is trivially exact outside one.

    PYTHONPATH=. .venv/bin/python -m himrobot.wbc.preview_error

Prints and writes work/wbc/preview_error.json: joint-space RMS and max error
(radians) at k=5 (0.1 s ahead) and k=10 (0.2 s ahead), against THRESHOLD_RAD.
preview() is evaluated with hitter.dls_iters = --dls-iters, defaulting to
himrobot.policies.wbc_np.PREVIEW_DLS_ITERS -- the throttled setting the
deploy runtime actually feeds the WBC (its act() lowers dls_iters around the
preview() call only; see that module's docstring) -- so the number here is
the gap the policy will really see; pass --dls-iters 20 for the hitter's
own default. The JSON records ``dls_iters``.
Decision rule (docs/design/whole_body_v1.md sec 5.3, verbatim): the runtime
WBC sees preview(t) in its observation, so if this error exceeds the
threshold the swing libraries must be re-recorded storing preview(t) instead
of the true closed-loop future -- train-time input must match what the
policy actually gets at test time. Below the threshold either is fine and
record_library.py's true future stays. It also counts "mixed" pairs (one
side STAND_POSE, the other inside a window): window-timing mismatches the
in-window RMS cannot see, reported for WB2 and not part of the verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# himrobot.sim.scene reads TT_ROBOT_Y once at import, so the stance has to be
# chosen before that import. Only the command-line entry point defaults it to
# the shipped stance (docs/design/whole_body_v1.md sec 4.1); importing this
# module as a library must not change the stance of the importing process.
if __name__ == "__main__":
    os.environ.setdefault("TT_ROBOT_Y", "-0.10")

import numpy as np  # noqa: E402

from himrobot.paths import ASSETS_DIR, REPO_ROOT  # noqa: E402
from himrobot.sim import scene  # noqa: E402
from himrobot.policies.ik_hitter import IK_INDICES  # noqa: E402
from himrobot.policies.param_residual_np import NumpyParamResidualPolicy  # noqa: E402
from himrobot.policies.wbc_np import PREVIEW_DLS_ITERS  # noqa: E402

V3_WEIGHTS = str(ASSETS_DIR / "policies" / "param_residual_v3_s0final.npz")
N_TRIALS = 200
SEED_BASE = 5000  # disjoint from the scoreboard's 11/2000/4000; may overlap the >=5000 library bank (diagnostic only, nothing trains on this)
STEPS_PER_TRIAL = 150  # 3 s at 50 Hz, matches evaluate.EPISODE_CONTROL_STEPS
HORIZON = 10
K_CHECK = (5, 10)
THRESHOLD_RAD = 0.05
DEFAULT_OUT = str(REPO_ROOT / "work" / "wbc" / "preview_error.json")


def _in_window(row: np.ndarray) -> bool:
    """A row is "inside a swing window" iff it differs from the STAND_POSE
    fallback -- both act() and preview() return an exact STAND_POSE.copy()
    whenever there is no plan or the requested time falls outside its
    window (see IKHitterPolicy.act/preview), so equality is an exact test."""
    return not np.array_equal(row, scene.STAND_POSE)


def run(weights: str, n_trials: int, seed_base: int, steps_per_trial: int,
       horizon: int, k_check: tuple[int, ...], dls_iters: int = PREVIEW_DLS_ITERS) -> dict:
    env = scene.TableTennisEnv(seed=0, params={})
    policy = NumpyParamResidualPolicy(weights)

    # One flat list of (pred - real) per k, over the 10 IK joints, collected
    # across every trial -- pooled, not averaged per trial, so a few bad
    # trials cannot hide behind many easy ones.
    diffs: dict[int, list[np.ndarray]] = {k: [] for k in k_check}
    n_pairs: dict[int, int] = {k: 0 for k in k_check}
    n_mixed: dict[int, int] = {k: 0 for k in k_check}  # exactly one side inside a window

    for trial in range(n_trials):
        seed = seed_base + trial
        rng = np.random.default_rng(seed)
        serve = scene.sample_serve(rng, env.params)
        obs = env.reset(serve=serve)
        policy.reset({"params": env.params})

        actions = np.empty((steps_per_trial, scene.NUM_JOINTS), dtype=np.float64)
        previews = np.empty((steps_per_trial, horizon, scene.NUM_JOINTS), dtype=np.float64)
        for step in range(steps_per_trial):
            action = policy.act(obs)  # at the hitter's real dls_iters, exactly as NumpyWBCPolicy.act
            actions[step] = action
            saved_iters = policy.hitter.dls_iters
            policy.hitter.dls_iters = dls_iters
            try:
                previews[step] = policy.hitter.preview(obs, horizon)
            finally:
                policy.hitter.dls_iters = saved_iters
            obs, _events = env.step(action)

        for i in range(steps_per_trial):
            for k in k_check:
                j = i + k
                if j >= steps_per_trial:
                    continue
                pred, real = previews[i, k - 1], actions[j]
                in_pred, in_real = _in_window(pred), _in_window(real)
                if in_pred != in_real:
                    n_mixed[k] += 1
                if not (in_pred and in_real):
                    continue
                diffs[k].append(pred[IK_INDICES] - real[IK_INDICES])
                n_pairs[k] += 1

    # Repo-relative when the weights live under the repo, so the JSON is not
    # tied to one checkout path.
    try:
        weights_rec = str(os.path.relpath(weights, REPO_ROOT))
        if weights_rec.startswith(".."):
            weights_rec = weights
    except ValueError:  # different drive on Windows
        weights_rec = weights
    result: dict = {
        "weights": weights_rec, "n_trials": n_trials, "seed_base": seed_base,
        "steps_per_trial": steps_per_trial, "horizon": horizon, "dls_iters": int(dls_iters),
        "robot_y": float(scene.ROBOT_Y), "threshold_rad": THRESHOLD_RAD, "k": {},
    }
    for k in k_check:
        if diffs[k]:
            arr = np.stack(diffs[k])  # (n_pairs, 10)
            rms = float(np.sqrt(np.mean(arr ** 2)))
            mx = float(np.max(np.abs(arr)))
        else:
            rms = mx = float("nan")
        result["k"][str(k)] = {
            "n_pairs": n_pairs[k], "n_pairs_mixed": n_mixed[k], "rms_rad": rms, "max_rad": mx,
            "verdict": "pass" if rms <= THRESHOLD_RAD else "fail",
        }
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=V3_WEIGHTS)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--steps-per-trial", type=int, default=STEPS_PER_TRIAL)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--dls-iters", type=int, default=PREVIEW_DLS_ITERS,
                    help="hitter.dls_iters for preview(); default = the deploy runtime's throttle")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    start = time.time()
    result = run(args.weights, args.n_trials, args.seed_base, args.steps_per_trial,
                args.horizon, K_CHECK, dls_iters=args.dls_iters)
    result["elapsed_s"] = time.time() - start

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print(f"preview_error: {args.n_trials} serves x {args.steps_per_trial} steps "
          f"(ROBOT_Y={scene.ROBOT_Y:g}, preview dls_iters={args.dls_iters}), {result['elapsed_s']:.1f}s")
    for k in K_CHECK:
        row = result["k"][str(k)]
        print(f"  k={k:<2d} ({k * scene.SIM_DT * scene.DECIMATION * 1000:.0f} ms ahead): "
              f"n={row['n_pairs']} rms={row['rms_rad']:.4f} rad max={row['max_rad']:.4f} rad "
              f"[{row['verdict']}, threshold {THRESHOLD_RAD} rad]; mixed pairs={row['n_pairs_mixed']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
