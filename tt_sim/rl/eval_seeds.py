"""Scoreboard: score policies on the fixed serve sets (seeds 11 / 2000 / 4000, 200 serves each) in one command.

Serve k of a run is drawn from ``default_rng(seed + k)`` (tt_sim.evaluate), so two seeds closer together than
``--trials`` share most of their serves: the old 11/7/3 convention shared 192-196 of 200. Seeds must differ by at
least ``--trials`` to be independent; this tool refuses overlapping seeds unless ``--allow-overlap`` is given.

    python -m tt_sim.rl.eval_seeds --ckpt work/tt/rl/param_v2/w0/best.pt --ckpt work/tt/rl/param_v2/w1/best.pt \\
        --npz assets/policies/param_residual_s3.npz --hitter --out work/tt/rl/param_v2/scoreboard --jobs 6

Every number comes from ``python -m tt_sim.evaluate`` (mean action, the project's only accepted scorer), one
subprocess per (policy, seed), run in parallel. Writes <out>/<label>/seed<S>/evaluation/evaluator.json per
run plus <out>/scoreboard.md and <out>/scoreboard.json: legal-return %, target-hit %, landing error and
contacts per seed and averaged over seeds. Labels default to the checkpoint's parent directory + file stem.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

DEFAULT_SEEDS = (11, 2000, 4000)
METRIC_KEYS = ("legal_return_rate", "target_hit_rate", "landing_error_m", "contacts")


def _label_for(path: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{parent}/{stem}"


def _run_one(spec: dict, seed: int, trials: int, out: str, params: str, python: str,
             extra_args: list[str]) -> dict:
    run_dir = os.path.join(out, spec["label"].replace("/", "_"), f"seed{seed}")
    cmd = [python, "-m", "tt_sim.evaluate", "--policy", spec["policy"], "--trials", str(trials),
           "--seed", str(seed), "--out", run_dir, "--params", params, "--no-trajectory"] + extra_args
    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "."), "OMP_NUM_THREADS": "1", **spec["env"]}
    log_path = os.path.join(run_dir, "evaluate.log")
    os.makedirs(run_dir, exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    doc_path = os.path.join(run_dir, "evaluation", "evaluator.json")
    if proc.returncode != 0 or not os.path.isfile(doc_path):
        raise RuntimeError(f"evaluate failed for {spec['label']} seed {seed}: see {log_path}")
    with open(doc_path) as fh:
        metrics = json.load(fh)["metrics"]
    return {"label": spec["label"], "seed": seed, **{k: metrics.get(k) for k in METRIC_KEYS}}


def _fmt(value, spec: str) -> str:
    return "–" if value is None else format(value, spec)


def render_table(rows: list[dict], seeds: tuple[int, ...], trials: int) -> str:
    by_label: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], {})[r["seed"]] = r
    head = ["policy"] + [f"seed {s} legal / target / err" for s in seeds] + ["mean legal", "mean target", "mean err", "contacts"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, per_seed in by_label.items():
        cells = [label]
        legal, target, errs, contacts = [], [], [], []
        for s in seeds:
            r = per_seed.get(s)
            if r is None:
                cells.append("–")
                continue
            cells.append(f"{_fmt(r['legal_return_rate'], '.1f')} % / {_fmt(r['target_hit_rate'], '.1f')} % / "
                         f"{_fmt(r['landing_error_m'], '.2f')} m")
            legal.append(r["legal_return_rate"])
            target.append(r["target_hit_rate"])
            if r["landing_error_m"] is not None:
                errs.append(r["landing_error_m"])
            contacts.append(r["contacts"])
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
        cells += [f"**{_fmt(mean(legal), '.1f')} %**", f"{_fmt(mean(target), '.1f')} %",
                  f"{_fmt(mean(errs), '.2f')} m", "/".join(str(c) for c in contacts)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"{trials} serves per seed; mean action (`tt_sim.evaluate`); contacts listed per seed.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", action="append", default=[], help="torch checkpoint (ParamResidualPolicy); repeatable")
    parser.add_argument("--npz", action="append", default=[], help="numpy weights (NumpyParamResidualPolicy); repeatable")
    parser.add_argument("--hitter", action="store_true", help="also score the plain IK hitter")
    parser.add_argument("--policy", action="append", default=[], help="label=module.path:Class; repeatable")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--params", type=str, default="{}", help="JSON dict merged over tt_sim.scene.DEFAULT_PARAMS")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--jobs", type=int, default=4, help="parallel evaluate processes")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--obs-noise", type=float, default=0.0, help="forwarded to tt_sim.evaluate")
    parser.add_argument("--joint-noise", type=float, default=0.0, help="forwarded to tt_sim.evaluate")
    parser.add_argument("--obs-delay-steps", type=int, default=0, help="forwarded to tt_sim.evaluate")
    parser.add_argument("--allow-overlap", action="store_true", help="permit seeds closer together than --trials")
    args = parser.parse_args(argv)
    seeds = sorted(args.seeds)
    overlap = [(a, b) for a, b in zip(seeds, seeds[1:]) if b - a < args.trials]
    if overlap and not args.allow_overlap:
        parser.error(f"seeds {overlap} are closer than --trials={args.trials} and share serves "
                     f"(serve k = default_rng(seed + k)); pick seeds >= {args.trials} apart or pass --allow-overlap")
    extra_args = ["--obs-noise", str(args.obs_noise), "--joint-noise", str(args.joint_noise),
                  "--obs-delay-steps", str(args.obs_delay_steps)]

    specs: list[dict] = []
    for path in args.ckpt:
        specs.append({"label": _label_for(path), "policy": "tt_sim.rl.policy:ParamResidualPolicy",
                      "env": {"TT_RL_CKPT": os.path.abspath(path)}})
    for path in args.npz:
        specs.append({"label": _label_for(path), "policy": "tt_sim.policies.param_residual_np:NumpyParamResidualPolicy",
                      "env": {"TT_PARAM_NPZ": os.path.abspath(path)}})
    for item in args.policy:
        label, _, policy = item.partition("=")
        specs.append({"label": label or policy, "policy": policy, "env": {}})
    if args.hitter:
        specs.append({"label": "ik_hitter", "policy": "ik_hitter", "env": {}})
    if not specs:
        parser.error("nothing to score: pass --ckpt/--npz/--policy/--hitter")

    os.makedirs(args.out, exist_ok=True)
    jobs = [(spec, seed) for spec in specs for seed in args.seeds]
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = list(pool.map(lambda js: _run_one(js[0], js[1], args.trials, args.out, args.params,
                                                 args.python, extra_args), jobs))

    table = render_table(rows, tuple(args.seeds), args.trials)
    with open(os.path.join(args.out, "scoreboard.md"), "w") as fh:
        fh.write(table + "\n")
    with open(os.path.join(args.out, "scoreboard.json"), "w") as fh:
        json.dump({"trials": args.trials, "seeds": args.seeds, "params": json.loads(args.params),
                   "policy_obs": {"obs_noise": args.obs_noise, "joint_noise": args.joint_noise,
                                  "delay_steps": args.obs_delay_steps}, "rows": rows}, fh, indent=1)
    print(table)
    return {"rows": rows, "out": args.out}


if __name__ == "__main__":
    main()
