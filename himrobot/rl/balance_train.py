"""PPO trainer for himrobot.rl.balance_env.VecTTBalance ("balance v1" -- see
work/tt/design/balance_v1.md): a 12-dim leg policy learns to keep the G1
standing on the free-base scene while a shipped v3 hitting policy
(himrobot.policies.param_residual_np.NumpyParamResidualPolicy) drives
waist+arm, unchanged.

    python -m himrobot.rl.balance_train --n-envs 16 --minutes 60 --seed 0 \\
        --out work/tt/rl/balance_v1/run1 --det-eval-episodes 40 --det-eval-every 20

Sets TT_ROBOT_Y=-0.10 and TT_PARAM_NPZ (the v3 checkpoint) BEFORE importing
himrobot.sim.scene -- exactly himrobot/eval/evaluate_stance.py's mechanism, since
scene.ROBOT_Y is read once at import (a plain flag cannot change it after
himrobot.sim.scene has already been imported). Either variable already set in the
environment is left alone (os.environ.setdefault).

Rollout collection / GAE / clipped-ratio update are himrobot.rl.ppo's own
(imported, not reimplemented -- they are plain numpy/torch utilities, not
specific to VecTT's 99-dim obs or 29-dim action, so they work unmodified
against VecTTBalance's 111-dim obs / 12-dim action). The actor-critic is
himrobot.rl.ppo.ActorCritic built at (BALANCE_OBS_DIM=111, BALANCE_ACT_DIM=12,
hidden=128) -- nets.py's MLP-block conventions (Linear/Tanh stacks), just
sized down from PPO's default 256 the way himrobot.rl.nets.Actor's own
``hidden`` parameter already supports. The actor's final Linear layer is
zero-initialized (weight+bias), so a_rl=0 at step zero: the parity anchor
himrobot.rl.balance_env's module docstring describes.

log_std is annealed the way --residual training is in himrobot.rl.ppo: an
upper-bound clamp that shrinks linearly from --log-std-init to
--log-std-final over the run (by wall-clock progress against --minutes,
since this trainer -- like himrobot.rl.param_train -- is driven by a
--minutes wall-clock cap rather than a step target in normal use; the
argparse --total-steps default is only a very large fallback stop so a
call without --minutes still terminates eventually).

Deterministic evaluation (deterministic_eval) mirrors
himrobot.rl.param_train.deterministic_eval's approach as closely as a
per-step (not one-shot bandit) policy allows: for each of --det-eval-
episodes fixed serves (seed + k, himrobot.eval.evaluate's own recipe), build a
FRESH TableTennisEnv(fixed_base=False)/NumpyParamResidualPolicy pair (not
VecTTBalance -- no auto-reset, no train-time RNG interaction), run the mean
(deterministic) action every control step for the full MAX_EPISODE_STEPS,
and score legal/contact/target-hit/landing-error/stand (no "fell" latched)
exactly as himrobot.eval.evaluate would. obs_rms is read, never updated.

train_log.csv columns: episode, wall_s, mean_return, legal_return_rate,
contact_rate, stand_rate (all window means over the last 200 completed
train-time episodes), then det_legal_rate, det_contact_rate,
det_target_rate, det_landing_error_m, det_stand_rate (blank unless a
deterministic evaluation ran that row). best.pt is the checkpoint with the
highest det_legal_rate (ties: higher det_stand_rate); both checkpoints
carry the obs normalizer stats, BALANCE_SCALE, expert_npz path and joint
ranges needed to reconstruct a runnable policy (see export_npz).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque

from himrobot.paths import ASSETS_DIR  # safe: himrobot/__init__.py never imports himrobot.sim.scene

# TT_ROBOT_Y/TT_PARAM_NPZ must precede any import that reaches himrobot.sim.scene
# (scene.ROBOT_Y is read once, at import) -- see module docstring and
# himrobot/eval/evaluate_stance.py, whose mechanism this copies.
os.environ.setdefault("TT_ROBOT_Y", "-0.10")
os.environ.setdefault("TT_PARAM_NPZ", str(ASSETS_DIR / "policies" / "param_residual_v3_s0final.npz"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from himrobot.sim import scene  # noqa: E402
from himrobot.policies.param_residual_np import NumpyParamResidualPolicy  # noqa: E402
from himrobot.rl.balance_env import (  # noqa: E402
    BALANCE_ACT_DIM,
    BALANCE_OBS_DIM,
    BALANCE_SCALE,
    DEFAULT_ALIVE_BONUS,
    DEFAULT_EXPERT_NPZ,
    DEFAULT_FALL_PENALTY,
    DEFAULT_TILT_PENALTY,
    OBS_VERSION,
    VecTTBalance,
    combined_action,
    joint_target,
    flatten_balance_obs,
)
from himrobot.rl.demos import target_to_action  # noqa: E402
from himrobot.rl.env import ACT_DIM, MAX_EPISODE_STEPS, TARGET_HIT_RADIUS_M, flatten_obs  # noqa: E402
from himrobot.rl.ppo import (  # noqa: E402
    ActorCritic,
    RunningMeanStd,
    collect_rollout,
    compute_gae,
    pick_device,
    ppo_update,
)

HIDDEN = 128
LR = 3e-4
ROLLOUT_STEPS = 128
EPISODE_WINDOW = 200
LOG_STD_INIT = -1.6
LOG_STD_FINAL = -2.5


def deterministic_eval(model: ActorCritic, obs_rms: RunningMeanStd, jnt_range: np.ndarray,
                       device: torch.device, n_serves: int, seed: int, expert_npz: str,
                       params: dict | None = None) -> dict[str, float]:
    """Score the MEAN (deterministic) action on a fixed serve set, the way
    himrobot.eval.evaluate would -- see module docstring. Fresh env/expert per
    call; obs_rms is read, never updated."""
    eval_params = {**scene.DEFAULT_PARAMS, **(params or {}), "physics_jitter": 0.0}
    env = scene.TableTennisEnv(seed=seed, params=eval_params, fixed_base=False)
    expert = NumpyParamResidualPolicy(expert_npz)
    target = scene.target_point(eval_params["target_zone"], eval_params["target_depth"])
    was_training = model.training
    model.eval()

    legal = contact = hits = stands = 0
    errors: list[float] = []
    for k in range(n_serves):
        serve = scene.sample_serve(np.random.default_rng(seed + k), env.params)
        obs = env.reset(serve=serve)
        expert.reset({"serve": None, "params": env.params})
        prev_action = np.zeros(ACT_DIM, dtype=np.float64)
        expert_action = target_to_action(expert.act(obs))
        info: dict = {}
        for _ in range(MAX_EPISODE_STEPS):
            base = flatten_obs(obs, prev_action, target, obs["time_s"])
            flat = flatten_balance_obs(base, obs)
            norm = obs_rms.normalize(flat.astype(np.float64)[None, :])
            obs_t = torch.as_tensor(norm, dtype=torch.float32, device=device)
            mean, _logp, _value = model.act(obs_t, deterministic=True)
            a_rl = np.clip(mean.cpu().numpy()[0].astype(np.float64), -1.0, 1.0)
            combined = combined_action(a_rl, expert_action)
            obs, info = env.step(joint_target(combined, jnt_range))
            prev_action = combined
            expert_action = target_to_action(expert.act(obs))
        is_legal = bool(scene.legal_return(info))
        legal += is_legal
        contact += bool(info.get("contact_paddle"))
        stands += not bool(info.get("fell", False))
        err = scene.landing_error_m(info, target)
        if is_legal and not np.isnan(err):
            errors.append(float(err))
            hits += bool(err <= TARGET_HIT_RADIUS_M)

    if was_training:
        model.train()
    n = max(n_serves, 1)
    return {
        "legal_rate": 100.0 * legal / n,
        "contact_rate": 100.0 * contact / n,
        "target_rate": 100.0 * hits / n,
        "landing_error_m": float(np.mean(errors)) if errors else float("nan"),
        "stand_rate": 100.0 * stands / n,
    }


SELECT_MODES = ("legal", "legal_x_stand")


def _det_key(det: dict[str, float], select: str = "legal") -> tuple[float, float]:
    """best.pt ranking. "legal" (v1 default): highest det_legal_rate, ties by
    det_stand_rate. "legal_x_stand": highest legal_rate * stand_rate -- v1
    showed that ranking by legal rate alone buys returns with falls (s2:
    62.7 % legal, 25.5 % falls vs the no-balance 21.8 %)."""
    if select == "legal_x_stand":
        return (det["legal_rate"] * det["stand_rate"] / 100.0, det["legal_rate"])
    return (det["legal_rate"], det["stand_rate"])


def save_checkpoint(path: str, model: ActorCritic, obs_rms: RunningMeanStd, episode: int,
                    expert_npz: str, jnt_range: np.ndarray, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "obs_dim": BALANCE_OBS_DIM,
        "act_dim": BALANCE_ACT_DIM,
        "hidden": model.actor.hidden,  # the model's ACTUAL hidden size, not this module's default
        "obs_version": OBS_VERSION,
        "balance_scale": BALANCE_SCALE,
        "expert_npz": expert_npz,
        "jnt_range": jnt_range,
        "obs_rms_mean": obs_rms.mean,
        "obs_rms_var": obs_rms.var,
        "obs_rms_count": obs_rms.count,
        "episode": episode,
    }
    ckpt.update(extra or {})
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: torch.device) -> tuple[ActorCritic, RunningMeanStd, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"], hidden=ckpt.get("hidden", HIDDEN)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    obs_rms = RunningMeanStd((ckpt["obs_dim"],))
    obs_rms.mean = np.asarray(ckpt["obs_rms_mean"], dtype=np.float64)
    obs_rms.var = np.asarray(ckpt["obs_rms_var"], dtype=np.float64)
    obs_rms.count = ckpt["obs_rms_count"]
    return model, obs_rms, ckpt


def export_npz(checkpoint_path: str, out_npz_path: str, device: torch.device | None = None) -> str:
    """checkpoint (.pt, torch) -> a torch-free .npz for
    himrobot.policies.balance_np.NumpyBalancePolicy: the actor MLP's Linear
    weights (net.0/2/4 -- Tanh has no parameters), the obs normalizer stats,
    and enough metadata (balance_scale, expert_npz, jnt_range, obs_version)
    to reconstruct the deployable policy with no PyTorch at runtime."""
    device = device or torch.device("cpu")
    model, obs_rms, ckpt = load_checkpoint(checkpoint_path, device)
    state = model.actor.net.state_dict()  # keys: "0.weight","0.bias","2.weight","2.bias","4.weight","4.bias"
    os.makedirs(os.path.dirname(out_npz_path) or ".", exist_ok=True)
    np.savez(
        out_npz_path,
        W0=state["0.weight"].detach().cpu().numpy().astype(np.float64),
        b0=state["0.bias"].detach().cpu().numpy().astype(np.float64),
        W1=state["2.weight"].detach().cpu().numpy().astype(np.float64),
        b1=state["2.bias"].detach().cpu().numpy().astype(np.float64),
        W2=state["4.weight"].detach().cpu().numpy().astype(np.float64),
        b2=state["4.bias"].detach().cpu().numpy().astype(np.float64),
        rms_mean=obs_rms.mean.astype(np.float64),
        rms_var=obs_rms.var.astype(np.float64),
        balance_scale=np.float64(ckpt.get("balance_scale", BALANCE_SCALE)),
        obs_version=np.int64(ckpt.get("obs_version", OBS_VERSION)),
        expert_npz=str(ckpt.get("expert_npz", DEFAULT_EXPERT_NPZ)),
        jnt_range=np.asarray(ckpt["jnt_range"], dtype=np.float64),
    )
    return out_npz_path


def train(n_envs: int = 16, seed: int = 0, out: str = os.path.join("work", "tt", "rl", "balance_v1", "run"),
          minutes: float | None = None, total_steps: int = 1_000_000, lr: float = LR,
          vel_penalty: float = 0.0005, legal_bonus: float = 5.0,
          tilt_penalty: float = DEFAULT_TILT_PENALTY, alive_bonus: float = DEFAULT_ALIVE_BONUS,
          fall_penalty: float = DEFAULT_FALL_PENALTY, expert_npz: str = DEFAULT_EXPERT_NPZ,
          log_std_init: float = LOG_STD_INIT, log_std_final: float = LOG_STD_FINAL,
          rollout_steps: int = ROLLOUT_STEPS, params: dict | None = None,
          det_eval_episodes: int = 0, det_eval_every: int = 20, det_eval_seed: int = 1001,
          select: str = "legal", resume_from: str | None = None) -> dict:
    assert select in SELECT_MODES, select
    os.makedirs(out, exist_ok=True)
    device = pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    vec = VecTTBalance(n_envs=n_envs, seed=seed, params=params or {}, expert_npz=expert_npz,
                      vel_penalty=vel_penalty, legal_bonus=legal_bonus,
                      tilt_penalty=tilt_penalty, alive_bonus=alive_bonus, fall_penalty=fall_penalty)
    jnt_range = vec._jnt_range.copy()

    model = ActorCritic(BALANCE_OBS_DIM, BALANCE_ACT_DIM, hidden=HIDDEN, log_std_init=log_std_init).to(device)
    # Zero-init the actor's final Linear layer: a_rl=0 at step zero, before
    # any gradient update -- the parity anchor (see balance_env's docstring).
    last_linear = model.actor.net[-2]  # net = [Linear, Tanh, Linear, Tanh, Linear, Tanh]
    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)
    obs_rms = RunningMeanStd((BALANCE_OBS_DIM,))
    if resume_from:
        # Warm start: actor/critic weights and the obs normalizer from a v1
        # checkpoint; the optimizer, log-std schedule and episode count start
        # fresh (the schedule overwrites the loaded log_std on the first update).
        loaded, obs_rms, _ckpt = load_checkpoint(resume_from, device)
        model.load_state_dict(loaded.state_dict())
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def det_eval() -> dict:
        return deterministic_eval(model, obs_rms, jnt_range, device, det_eval_episodes, det_eval_seed,
                                  expert_npz, params)

    csv_path = os.path.join(out, "train_log.csv")
    csv_fh = open(csv_path, "w", newline="")
    writer = csv.writer(csv_fh)
    det_cols = ["det_legal_rate", "det_contact_rate", "det_target_rate", "det_landing_error_m", "det_stand_rate"]
    writer.writerow(["episode", "wall_s", "mean_return", "legal_return_rate", "contact_rate", "stand_rate"]
                    + det_cols)
    csv_fh.flush()

    def det_cells(det: dict[str, float] | None) -> list[str]:
        if det is None:
            return [""] * len(det_cols)
        return [f"{det['legal_rate']:.2f}", f"{det['contact_rate']:.2f}", f"{det['target_rate']:.2f}",
                f"{det['landing_error_m']:.4f}", f"{det['stand_rate']:.2f}"]

    episode_window: deque[dict] = deque(maxlen=EPISODE_WINDOW)
    start = time.time()
    n_episodes = 0
    total = 0
    n_updates = 0
    last_log = 0
    best_det: dict | None = None
    best_det_episode = 0

    def maybe_save_best_det(det: dict, episode: int) -> None:
        nonlocal best_det, best_det_episode
        if best_det is None or _det_key(det, select) > _det_key(best_det, select):
            best_det, best_det_episode = det, episode
            save_checkpoint(os.path.join(out, "best.pt"), model, obs_rms, episode, expert_npz, jnt_range,
                            extra={"seed": seed, "det_eval": {**det, "n_serves": det_eval_episodes,
                                                              "seed": det_eval_seed}})

    if det_eval_episodes > 0:
        # Row 0: the starting point's true (mean-action) score, before any
        # update -- with the zero-init actor this is the "no balance"
        # a_rl=0 baseline (param_train.py's own row-0 convention).
        det0 = det_eval()
        writer.writerow([0, f"{time.time() - start:.2f}", "", "", "", ""] + det_cells(det0))
        csv_fh.flush()
        print(f"[balance] episode=0 det_legal_rate={det0['legal_rate']:.1f}% "
              f"det_stand_rate={det0['stand_rate']:.1f}% det_target_rate={det0['target_rate']:.1f}% "
              f"(n={det_eval_episodes}, seed={det_eval_seed})", flush=True)
        maybe_save_best_det(det0, 0)

    try:
        while total < total_steps:
            if minutes is not None and (time.time() - start) >= minutes * 60.0:
                print(f"[balance] --minutes {minutes} wall cap reached at episode {n_episodes}")
                break

            obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf, last_value, episode_stats = \
                collect_rollout(vec, model, obs_rms, rollout_steps, device, obs_dim=BALANCE_OBS_DIM)
            episode_window.extend(episode_stats)
            n_episodes += len(episode_stats)
            adv_buf, ret_buf = compute_gae(rew_buf, val_buf, done_buf, last_value)
            ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, adv_buf, ret_buf, obs_rms, device)

            total += rollout_steps * n_envs
            n_updates += 1

            # Upper-bound clamp on exploration noise, shrinking linearly by
            # wall-clock progress against --minutes (falling back to step
            # progress against --total-steps when no --minutes is given) --
            # same mechanism as himrobot.rl.ppo's --residual log-std anneal.
            if minutes is not None:
                progress = (time.time() - start) / (minutes * 60.0)
            else:
                progress = total / float(total_steps)
            cap = log_std_init + (log_std_final - log_std_init) * min(max(progress, 0.0), 1.0)
            with torch.no_grad():
                model.log_std.clamp_(max=cap)

            if n_episodes - last_log >= det_eval_every or total >= total_steps:
                last_log = n_episodes
                if episode_window:
                    mean_return = float(np.mean([e["return"] for e in episode_window]))
                    legal_rate = 100.0 * float(np.mean([e["legal_return"] for e in episode_window]))
                    contact_rate = 100.0 * float(np.mean([e["contact"] for e in episode_window]))
                    stand_rate = 100.0 * float(np.mean([not e.get("fell", False) for e in episode_window]))
                else:
                    mean_return = legal_rate = contact_rate = stand_rate = float("nan")
                det = det_eval() if det_eval_episodes > 0 else None
                wall_s = time.time() - start
                writer.writerow([n_episodes, f"{wall_s:.2f}", mean_return, legal_rate, contact_rate, stand_rate]
                                + det_cells(det))
                csv_fh.flush()
                det_text = ("" if det is None else
                            f" det_legal_rate={det['legal_rate']:.1f}% det_stand_rate={det['stand_rate']:.1f}%")
                print(f"[balance] episode={n_episodes} update={n_updates} wall_s={wall_s:.1f} "
                      f"mean_return={mean_return:.3f} legal_return_rate={legal_rate:.1f}% "
                      f"contact_rate={contact_rate:.1f}% stand_rate={stand_rate:.1f}%{det_text}", flush=True)
                if det is not None:
                    maybe_save_best_det(det, n_episodes)
    finally:
        # Weights first: nothing below may cost final.pt (a second Ctrl-C or
        # an eval failure). Then score the final weights so the last
        # stretch of training can still win best.pt.
        save_checkpoint(os.path.join(out, "final.pt"), model, obs_rms, n_episodes, expert_npz, jnt_range,
                        extra={"seed": seed})
        final_det = None
        if det_eval_episodes > 0:
            try:
                final_det = det_eval()
                maybe_save_best_det(final_det, n_episodes)
                save_checkpoint(os.path.join(out, "final.pt"), model, obs_rms, n_episodes, expert_npz, jnt_range,
                                extra={"seed": seed, "det_eval": final_det})
            except BaseException as exc:  # noqa: BLE001 -- interrupt or eval failure: final.pt is already on disk
                print(f"[balance] closing det_eval skipped: {exc!r}", flush=True)
        csv_fh.close()

    return {"episodes": n_episodes, "total_steps": total, "n_updates": n_updates, "out": out, "csv": csv_path,
            "best_det": best_det, "best_det_episode": best_det_episode, "final_det": final_det,
            "wall_s": time.time() - start}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=os.path.join("work", "tt", "rl", "balance_v1", "run"))
    parser.add_argument("--minutes", type=float, default=None, help="hard wall-clock cap, minutes")
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="fallback stop if --minutes is not given")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS, help="per-env steps collected per PPO update")
    parser.add_argument("--vel-penalty", type=float, default=0.0005, help="per-step weight on sum(joint_velocity^2)")
    parser.add_argument("--legal-bonus", type=float, default=5.0, help="one-off reward for a legal return")
    parser.add_argument("--tilt-penalty", type=float, default=DEFAULT_TILT_PENALTY, help="per-step weight on tilt_rad**2")
    parser.add_argument("--alive-bonus", type=float, default=DEFAULT_ALIVE_BONUS, help="per-step reward for not having fallen yet")
    parser.add_argument("--fall-penalty", type=float, default=DEFAULT_FALL_PENALTY, help="one-off penalty when events['fell'] first latches")
    parser.add_argument("--select", choices=SELECT_MODES, default="legal", help="best.pt ranking (see _det_key)")
    parser.add_argument("--resume-from", type=str, default=None, help="warm start from a balance checkpoint (.pt)")
    parser.add_argument("--expert-npz", type=str, default=DEFAULT_EXPERT_NPZ, help="waist+arm expert weights (NumpyParamResidualPolicy)")
    parser.add_argument("--log-std-init", type=float, default=LOG_STD_INIT)
    parser.add_argument("--log-std-final", type=float, default=LOG_STD_FINAL, help="linear anneal target for the log-std upper bound")
    parser.add_argument("--params", type=str, default="{}", help="JSON dict merged over himrobot.sim.scene.DEFAULT_PARAMS")
    parser.add_argument("--det-eval-episodes", type=int, default=40,
                        help="select best.pt by a mean-action score on this many fixed serves (0 = never)")
    parser.add_argument("--det-eval-every", type=int, default=20, help="episodes between deterministic evaluations / train_log.csv rows")
    parser.add_argument("--det-eval-seed", type=int, default=1001, help="serve-set seed for the deterministic evaluation")
    parser.add_argument("--export", type=str, default=None, help="path to also export a NumpyBalancePolicy .npz from the run's best.pt")
    args = parser.parse_args(argv)

    result = train(
        n_envs=args.n_envs, seed=args.seed, out=args.out, minutes=args.minutes, total_steps=args.total_steps,
        lr=args.lr, vel_penalty=args.vel_penalty, legal_bonus=args.legal_bonus,
        tilt_penalty=args.tilt_penalty, alive_bonus=args.alive_bonus, fall_penalty=args.fall_penalty,
        expert_npz=args.expert_npz, log_std_init=args.log_std_init, log_std_final=args.log_std_final,
        rollout_steps=args.rollout_steps, params=json.loads(args.params),
        det_eval_episodes=args.det_eval_episodes, det_eval_every=args.det_eval_every,
        det_eval_seed=args.det_eval_seed, select=args.select, resume_from=args.resume_from,
    )
    if args.export:
        best_path = os.path.join(args.out, "best.pt")
        src = best_path if os.path.isfile(best_path) else os.path.join(args.out, "final.pt")
        export_npz(src, args.export)
        print(f"[balance] exported {src} -> {args.export}")
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
