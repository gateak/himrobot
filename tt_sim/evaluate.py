"""tt_sim evaluation CLI.

    python -m tt_sim.evaluate --policy {hold,ik_hitter,module.path:ClassName}
        [--trials N] [--seed S] [--params JSON] [--out DIR] [--video] [--no-trajectory]
        [--obs-noise M] [--joint-noise RAD] [--obs-delay-steps K] [--free-base] [--base-aware]

Runs ``--trials`` served-ball episodes against the named policy in
tt_sim.scene.TableTennisEnv and writes:

    <candidate_dir>/trajectory.jsonl   one JSON object per control step,
                                        per trial: {trial, step, time_s,
                                        obs, action, events}
    <evaluation_dir>/evaluator.json    {authoritative: true, evaluator,
                                        params, metrics, per_trial}
    <media_dir>/trial_<k>.mp4          per-trial video (only with --video)
    <media_dir>/summary.mp4            all trials concatenated (--video)

``--free-base`` unwelds the pelvis (tt_sim.scene's free-base contract): the
robot must stand on its own, ``metrics["falls"]`` counts the trials whose
pelvis went down, and ``evaluator.json`` records ``"free_base": true``.
Default off — the scored scene is the welded one, and its numbers are
unchanged by this flag existing. ``metrics["falls"]`` is always present and
reads 0 on the welded scene.

Where each of candidate_dir/evaluation_dir/media_dir is the matching
HIM_CANDIDATE_OUTPUT / HIM_EVALUATION_OUTPUT / HIM_MEDIA_OUTPUT env var
when set (the hosted HIM runtime provides these — see docs/him/llms-
full.txt), else that subdirectory of --out (default work/tt/eval). See
tt_sim/README.md for the full metric definitions and the standing
disclaimer that this is a **local proxy** of HIM's real (hidden)
evaluator, not a reimplementation of it.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time

import numpy as np

from tt_sim.obs_perturb import ObsPerturber
from tt_sim.policies.base import Policy
from tt_sim.scene import (
    CONTROL_HZ,
    DEFAULT_PARAMS,
    TableTennisEnv,
    landing_error_m,
    legal_return,
    sample_serve,
    target_point,
)

EVALUATOR_ID = "himrobot-local-proxy@0.1.0"

# Episode length: long enough to comfortably cover one serve (flight time
# 0.45-0.75s, see tt_sim.scene.SERVE_FLIGHT_TIME_RANGE) plus a reaction,
# swing, and return flight to the opponent half, with margin; short enough
# that a legal return's table bounce is very unlikely to be overwritten by
# a later, off-table-adjacent second bounce before the episode ends (see
# tt_sim.scene's TableTennisEnv docstring note: events['landing_point'] is
# the *last* recorded bounce, not necessarily the return's). Not a cited
# fact -- a chosen trial-horizon assumption, documented here and in the
# README.
EPISODE_S = 3.0
EPISODE_CONTROL_STEPS = round(EPISODE_S * CONTROL_HZ)  # 150 at the env's 50 Hz control_hz

# target_hit_rate threshold: a legal return "hits the target" if its
# landing point is within this radius (metres) of the requested
# target_zone/target_depth point. No ITTF/HIM fact gives this number --
# it is this proxy's own assumption, documented in the README.
TARGET_HIT_RADIUS_M = 0.30

POLICY_SHORTCUTS = {
    "hold": ("tt_sim.policies.hold", "HoldPolicy"),
    "ik_hitter": ("tt_sim.policies.ik_hitter", "IKHitterPolicy"),
}


def _load_policy(spec: str) -> Policy:
    """Resolve --policy: a shortcut name, or an explicit 'module.path:ClassName'."""
    if spec in POLICY_SHORTCUTS:
        module_name, class_name = POLICY_SHORTCUTS[spec]
    elif ":" in spec:
        module_name, class_name = spec.split(":", 1)
    else:
        raise ValueError(
            f"--policy {spec!r}: expected one of {sorted(POLICY_SHORTCUTS)} "
            "or an explicit 'module.path:ClassName'."
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"--policy {spec!r} resolves to module {module_name!r}, which failed to "
            f"import ({exc}). Built-in shortcuts: {sorted(POLICY_SHORTCUTS)}; otherwise "
            "pass 'module.path:ClassName' explicitly."
        ) from exc
    cls = getattr(module, class_name)
    policy = cls()
    assert isinstance(policy, Policy), f"{module_name}.{class_name} must subclass tt_sim.policies.base.Policy"
    return policy


def _resolve_output_dirs(out: str | None) -> tuple[str, str, str]:
    """(candidate_dir, evaluation_dir, media_dir): each is the matching
    HIM_*_OUTPUT env var when set, else a subdirectory of --out/work/tt/eval."""
    base = out or os.path.join("work", "tt", "eval")
    candidate_dir = os.environ.get("HIM_CANDIDATE_OUTPUT") or os.path.join(base, "candidate")
    evaluation_dir = os.environ.get("HIM_EVALUATION_OUTPUT") or os.path.join(base, "evaluation")
    media_dir = os.environ.get("HIM_MEDIA_OUTPUT") or os.path.join(base, "media")
    return candidate_dir, evaluation_dir, media_dir


def _json_safe(value):
    """Recursively convert numpy/NaN values to plain, strict-JSON-safe Python
    (json.dump's default allow_nan=True would otherwise emit the bare,
    non-standard token `NaN`)."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _running_legal_rate_text(legal_flags: list[bool]) -> str:
    n = len(legal_flags)
    if n == 0:
        return "legal_return_rate so far: n/a (0 trials completed)"
    pct = 100.0 * sum(legal_flags) / n
    return f"legal_return_rate so far: {pct:.0f}% ({sum(legal_flags)}/{n})"


def run_trial(
    env: TableTennisEnv,
    policy: Policy,
    trial_idx: int,
    n_trials: int,
    trial_seed: int,
    target: tuple[float, float],
    traj_fh,
    recorder=None,
    legal_flags: list[bool] | None = None,
    perturber: ObsPerturber | None = None,
) -> dict:
    """Run one episode, stream its steps to traj_fh (JSON Lines), and
    return this trial's per_trial summary dict."""
    serve_rng = np.random.default_rng(trial_seed)
    serve = sample_serve(serve_rng, env.params)
    obs = env.reset(serve=serve)
    policy.reset({"serve": _json_safe(serve), "params": env.params, "trial": trial_idx, "seed": trial_seed})
    if perturber is None:
        perturber = ObsPerturber()
    policy_obs = perturber.reset(obs)

    info: dict = {}
    # The HUD's running rate is fixed for this trial: the caller appends this
    # trial's own flag to legal_flags only after run_trial returns.
    metric_text = _running_legal_rate_text(legal_flags or []) if recorder is not None else ""
    for step_i in range(EPISODE_CONTROL_STEPS):
        action = policy.act(policy_obs)
        obs, info = env.step(action)
        policy_obs = perturber.step(obs)

        traj_fh.write(json.dumps({
            "trial": trial_idx,
            "step": step_i,
            "time_s": obs["time_s"],
            "obs": _json_safe(obs),
            "action": _json_safe(np.asarray(action, dtype=np.float64)),
            "events": _json_safe(info),
        }) + "\n")

        if recorder is not None:
            recorder.maybe_capture(env.data, obs["time_s"], trial_idx, n_trials, info, metric_text)

    legal = legal_return(info)
    err = landing_error_m(info, target)
    err_is_finite = not math.isnan(err)
    hit = bool(legal and err_is_finite and err <= TARGET_HIT_RADIUS_M)

    return {
        "trial": trial_idx,
        "seed": trial_seed,
        "serve": _json_safe(serve),
        "legal_return": legal,
        "target_hit": hit,
        "landing_error_m": err if err_is_finite else None,
        "reaction_latency_ms": info.get("reaction_latency_ms"),
        "events": _json_safe(info),
    }


def aggregate_metrics(per_trial: list[dict], n_trials: int) -> dict:
    contacts = sum(1 for t in per_trial if t["events"].get("contact_paddle"))
    # Additive, always present: trials whose pelvis fell (free base only —
    # the welded scene never emits events["fell"], so this reads 0 there).
    # Diagnostic; no other metric depends on it. See tt_sim.scene's free-base
    # contract.
    falls = sum(1 for t in per_trial if t["events"].get("fell"))
    legal = [t["legal_return"] for t in per_trial]
    hits = [t["target_hit"] for t in per_trial]
    errors = [t["landing_error_m"] for t in per_trial if t["legal_return"] and t["landing_error_m"] is not None]
    latencies = [t["reaction_latency_ms"] for t in per_trial if t["reaction_latency_ms"] is not None]

    return {
        "legal_return_rate": 100.0 * sum(legal) / n_trials if n_trials else 0.0,
        "target_hit_rate": 100.0 * sum(hits) / n_trials if n_trials else 0.0,
        "landing_error_m": (sum(errors) / len(errors)) if errors else None,
        "reaction_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "n_trials": n_trials,
        "contacts": contacts,
        "falls": falls,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="hold", help="hold | ik_hitter | module.path:ClassName")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params", type=str, default="{}", help="JSON dict, merged over tt_sim.scene.DEFAULT_PARAMS")
    parser.add_argument("--out", type=str, default=None, help="base output dir (default work/tt/eval); see HIM_* env vars")
    parser.add_argument("--video", action="store_true", help="render trial_<k>.mp4 + summary.mp4")
    parser.add_argument("--no-trajectory", action="store_true",
                        help="skip candidate/trajectory.jsonl (~30 MB per 200 trials); evaluator.json is unchanged")
    parser.add_argument("--obs-noise", type=float, default=0.0,
                        help="std (m) of Gaussian noise on the ball position the policy sees; velocity gets 10x")
    parser.add_argument("--joint-noise", type=float, default=0.0,
                        help="std (rad) of Gaussian noise on the joint readings the policy sees; velocity gets 10x")
    parser.add_argument("--obs-delay-steps", type=int, default=0,
                        help="the policy sees observations this many control steps (20 ms each) old")
    parser.add_argument("--base-aware", action="store_true",
                        help="free base only: the hitter re-expresses its IK target through the measured pelvis pose "
                             "(sets TT_BASE_AWARE=1 for the policy; default off -- measured worse without a balance "
                             "controller). Recorded in evaluator.json as base_aware.")
    parser.add_argument("--free-base", action="store_true",
                        help="unweld the pelvis: the robot must stand on its own (default off — the "
                             "scored scene is the welded one). See tt_sim.scene's free-base contract")
    args = parser.parse_args(argv)

    params = {**DEFAULT_PARAMS, **json.loads(args.params)}
    if args.base_aware:
        os.environ["TT_BASE_AWARE"] = "1"  # before the policy builds its IKHitterPolicy
    policy = _load_policy(args.policy)
    candidate_dir, evaluation_dir, media_dir = _resolve_output_dirs(args.out)
    os.makedirs(candidate_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    if args.video:
        os.makedirs(media_dir, exist_ok=True)

    env = TableTennisEnv(seed=args.seed, params=params, fixed_base=not args.free_base)
    target = target_point(params.get("target_zone", DEFAULT_PARAMS["target_zone"]),
                           params.get("target_depth", DEFAULT_PARAMS["target_depth"]))

    per_trial: list[dict] = []
    legal_flags: list[bool] = []
    trial_video_paths: list[str] = []
    start = time.time()

    trajectory_path = os.devnull if args.no_trajectory else os.path.join(candidate_dir, "trajectory.jsonl")
    with open(trajectory_path, "w") as traj_fh:
        for k in range(args.trials):
            recorder = None
            if args.video:
                try:
                    from tt_sim.media import TrialRecorder
                    recorder = TrialRecorder(env.model)
                except Exception as exc:  # no GL on this host: keep scoring, skip video
                    print(f"[evaluate] video disabled: {exc!r}", file=sys.stderr)

            perturber = ObsPerturber(args.obs_noise, args.joint_noise, args.obs_delay_steps,
                                     seed=args.seed + k)
            result = run_trial(
                env, policy, k, args.trials, args.seed + k, target, traj_fh,
                recorder=recorder, legal_flags=legal_flags, perturber=perturber,
            )
            per_trial.append(result)
            legal_flags.append(result["legal_return"])

            if recorder is not None:
                trial_path = os.path.join(media_dir, f"trial_{k}.mp4")
                try:
                    recorder.write(trial_path)
                    recorder.close()
                except Exception as exc:
                    print(f"[evaluate] trial video failed: {exc!r}", file=sys.stderr)
                trial_video_paths.append(trial_path)

    metrics = aggregate_metrics(per_trial, args.trials)

    evaluator_doc = {
        "authoritative": True,
        "evaluator": EVALUATOR_ID,
        "note": (
            "This is himrobot's own local proxy evaluator (tt_sim/evaluate.py), a "
            "faithful reimplementation of the published challenge scoring for local "
            "iteration only. It is NOT HIM's real hidden evaluator; HIM's own scored "
            "run uses a separate, server-owned evaluator and holdout seeds. See "
            "tt_sim/README.md."
        ),
        "params": params,
        "policy_obs": {"obs_noise": args.obs_noise, "joint_noise": args.joint_noise,
                       "delay_steps": args.obs_delay_steps},
        "free_base": bool(args.free_base),
        "base_aware": bool(args.base_aware),
        "metrics": metrics,
        "per_trial": per_trial,
        "wall_time_s": time.time() - start,
    }
    with open(os.path.join(evaluation_dir, "evaluator.json"), "w") as fh:
        json.dump(_json_safe(evaluator_doc), fh, indent=2)

    if args.video and trial_video_paths:
        try:
            from tt_sim.media import concat_videos
            concat_videos(trial_video_paths, os.path.join(media_dir, "summary.mp4"))
        except Exception as exc:
            print(f"[evaluate] summary video failed: {exc!r}", file=sys.stderr)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
