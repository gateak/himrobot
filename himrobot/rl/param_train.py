"""Contextual-bandit trainer for himrobot.rl.param_residual's per-serve params.

    python -m himrobot.rl.param_train --n-envs 32 --episodes 20000 --minutes 8 \\
        --seed 0 --out work/tt/rl/param_run --eval-every 200

Each episode is a single decision (one parameter vector per serve), so there
is no temporal credit assignment: advantage = episode_return - V(features),
learned by a clipped-ratio (PPO-style) update on the pre-squash Gaussian, with
a value head as baseline. Clipping rather than plain REINFORCE only because it
makes the step size forgiving when a batch happens to be mostly failures; the
bandit structure is what removes GAE, bootstrapping and per-step noise.

Writes train_log.csv (episode, wall_s, mean_return, legal_return_rate,
contact_rate -- each a mean over the last 200 completed episodes -- then the
deterministic-eval columns, then the window mean of every sampled parameter
u_<name> and of every gate's blend share blend_<gate>), best.pt at every new
best, and final.pt at exit. Both checkpoints carry the feature-normalizer
stats and the parameter ranges, and are loadable by himrobot.eval.evaluate via
``--policy himrobot.rl.policy:ParamResidualPolicy`` with ``TT_RL_CKPT=<path>``.

best.pt selection: with ``--det-eval-episodes N`` (recommended), every
``--det-eval-every`` episodes the *mean* action is scored on the same N fixed
serves exactly the way himrobot.eval.evaluate scores a trial (deterministic_eval),
and best.pt is the checkpoint with the highest deterministic legal rate
(ties: target-hit rate, then landing error). Without it, best.pt falls back to
the noisy windowed train-time legal rate, which is a diagnostic, not a score.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from himrobot.sim import scene
from himrobot.policies.ik_hitter import GATE_GAIN, IKHitterPolicy
from himrobot.rl.env import MAX_EPISODE_STEPS, TARGET_HIT_RADIUS_M
from himrobot.rl.param_residual import (
    DEFAULT_PARAM_SPACE,
    FEATURE_DIM,
    PARAM_SPACES,
    ParamActorCritic,
    VecTTParam,
    load_checkpoint,
    save_checkpoint,
    serve_features,
    unsquash_to_params,
)
from himrobot.rl.ppo import RunningMeanStd, pick_device
from himrobot.sim.scene import TableTennisEnv

CLIP_EPS = 0.2
EPOCHS = 4
LR = 3e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.002
MAX_GRAD_NORM = 0.5
EPISODE_WINDOW = 200
BATCH_EPISODES = 128
MIN_WINDOW_FOR_BEST = 20


def bandit_update(model: ParamActorCritic, optimizer: torch.optim.Optimizer,
                  feats: np.ndarray, z: np.ndarray, old_logp: np.ndarray,
                  returns: np.ndarray, device: torch.device,
                  entropy_coef: float = ENTROPY_COEF) -> dict[str, float]:
    """One clipped-ratio update over a batch of completed episodes.
    ``feats`` is already normalized; ``z`` is the pre-squash sample."""
    feats_t = torch.as_tensor(feats, dtype=torch.float32, device=device)
    z_t = torch.as_tensor(z, dtype=torch.float32, device=device)
    old_logp_t = torch.as_tensor(old_logp, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    stats = {}
    for _ in range(EPOCHS):
        logp, entropy, value = model.evaluate_actions(feats_t, z_t)
        # Advantage against the *current* baseline, renormalized each epoch:
        # with one decision per episode there is nothing else to subtract.
        adv = (ret_t - value).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = torch.exp(logp - old_logp_t)
        policy_loss = -torch.min(ratio * adv,
                                 torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv).mean()
        value_loss = 0.5 * (ret_t - value).pow(2).mean()
        entropy_mean = entropy.mean()
        loss = policy_loss + VALUE_COEF * value_loss - entropy_coef * entropy_mean

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        stats = {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                 "entropy": entropy_mean.item()}
    return stats


def deterministic_eval(model: ParamActorCritic, feat_rms: RunningMeanStd,
                       specs: tuple[tuple[str, str, float], ...],
                       device: torch.device, n_serves: int, seed: int,
                       params: dict | None = None) -> dict[str, float]:
    """Score the MEAN action on a fixed serve set, the way himrobot.eval.evaluate does.

    Serve k is ``scene.sample_serve(np.random.default_rng(seed + k), params)``
    -- the evaluator's own recipe, so ``seed=11, n_serves=200`` reproduces the
    scoreboard set exactly. The hitter's absolute joint targets go straight to
    TableTennisEnv.step (as in evaluate.py), not through the training
    action pipeline, and every trial runs the full MAX_EPISODE_STEPS so the
    latched events match. physics_jitter is forced off. feat_rms is read, never
    updated. Returns legal_rate / contact_rate / target_rate (%) and the mean
    landing error over legal returns (nan if none)."""
    eval_params = {**scene.DEFAULT_PARAMS, **(params or {}), "physics_jitter": 0.0}
    env = TableTennisEnv(seed=seed, params=eval_params,
                         fixed_base=not bool(eval_params.get("free_base", False)))
    hitter = IKHitterPolicy()
    target = scene.target_point(eval_params["target_zone"], eval_params["target_depth"])
    was_training = model.training
    model.eval()
    legal = contact = hits = 0
    errors: list[float] = []
    for k in range(n_serves):
        serve = scene.sample_serve(np.random.default_rng(seed + k), env.params)
        obs = env.reset(serve=serve)
        hitter.reset({"params": env.params})
        feats = serve_features(obs, target)
        norm = feat_rms.normalize(feats.astype(np.float64)[None, :])
        mean, _logp, _value = model.act(torch.as_tensor(norm, dtype=torch.float32, device=device),
                                        deterministic=True)
        u = np.tanh(mean.cpu().numpy()[0].astype(np.float64))
        hitter.set_serve_params(unsquash_to_params(u, specs))
        info: dict = {}
        for _ in range(MAX_EPISODE_STEPS):
            obs, info = env.step(hitter.act(obs))
        is_legal = bool(scene.legal_return(info))
        legal += is_legal
        contact += bool(info.get("contact_paddle"))
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
    }


def _det_key(det: dict[str, float], select: str = "legal") -> tuple[float, float, float]:
    """Checkpoint ranking. ``select="legal"``: legal rate, then target-hit
    rate, then landing error (phase one). ``select="target"``: target-hit rate
    first, then legal rate (phase two, when placement is being bought at the
    price of a few legal returns -- see hitter_v2.md R6)."""
    err = det["landing_error_m"]
    err_key = -(err if not np.isnan(err) else float("inf"))
    if select == "target":
        return (det["target_rate"], det["legal_rate"], err_key)
    return (det["legal_rate"], det["target_rate"], err_key)


def blend_fractions(u: np.ndarray, names) -> np.ndarray:
    """What the hitter actually blends for a sampled u: each gate's share of
    the swing shape, GATE_GAIN*relu(u_gate) / (1 + sum). Zero-length if the
    space has no gates. The raw u mean cannot show a blend collapse; this can."""
    idx = [i for i, n in enumerate(names) if n.startswith("gate_")]
    if not idx:
        return np.zeros(0)
    w = GATE_GAIN * np.maximum(np.asarray(u, dtype=np.float64)[idx], 0.0)
    return w / (1.0 + w.sum())


def copy_actor_prefix(src: ParamActorCritic, dst: ParamActorCritic,
                      log_std_init: float = -1.0) -> None:
    """Load ``src`` into ``dst`` when ``dst`` has the wider head: every tensor
    whose shape matches is copied outright, and the actor's output Linear (plus
    log_std) has its first src.k_params rows copied, leaving the new
    coordinates at their zero/log_std_init init. So resuming a v1 checkpoint
    into a v2 run starts from exactly the v1 policy's behaviour -- the new
    coordinates decode to nominal, which by construction is the v1 hitter."""
    if src.k_params == dst.k_params:
        dst.load_state_dict(src.state_dict())
        return
    assert src.k_params < dst.k_params, (src.k_params, dst.k_params)
    src_state = src.state_dict()
    with torch.no_grad():
        for name, param in dst.state_dict().items():
            if name not in src_state:
                continue
            loaded = src_state[name]
            if loaded.shape == param.shape:
                param.copy_(loaded)
            elif (loaded.shape[0] == src.k_params and param.shape[0] == dst.k_params
                    and param.shape[1:] == loaded.shape[1:]):
                param[:src.k_params].copy_(loaded)
                if name == "log_std":
                    param[src.k_params:] = log_std_init


def train(n_envs: int = 32, episodes: int = 20_000, seed: int = 0,
          out: str = os.path.join("work", "tt", "rl", "param_run"),
          minutes: float | None = None, eval_every: int = 200,
          batch_episodes: int = BATCH_EPISODES, lr: float = LR,
          log_std_init: float = -1.0, entropy_coef: float = ENTROPY_COEF,
          params: dict | None = None, vel_penalty: float = 0.0005,
          legal_bonus: float = 5.0, resume_from: str | None = None,
          param_space: str = DEFAULT_PARAM_SPACE, aim_bonus: float = 3.0,
          aim_sigma: float = 0.5, target_hit_bonus: float = 0.0,
          depth_bonus: float = 0.0, fall_penalty: float = 0.0,
          det_eval_episodes: int = 0, det_eval_every: int = 1000,
          det_eval_seed: int = 1001, log_std_init_new: float | None = None,
          obs_noise: float = 0.0, joint_noise: float = 0.0, obs_delay_steps: int = 0,
          det_select: str = "legal") -> dict:
    os.makedirs(out, exist_ok=True)
    device = pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    specs = PARAM_SPACES[param_space]
    names = [name for name, _kind, _bound in specs]
    k_params = len(specs)
    model = ParamActorCritic(FEATURE_DIM, k_params, log_std_init=log_std_init).to(device)
    feat_rms = RunningMeanStd((FEATURE_DIM,))
    if resume_from:
        loaded, feat_rms, _specs = load_checkpoint(resume_from, device)
        # Coordinates the checkpoint does not have start at log_std_init_new
        # (default: log_std_init); the ones it has keep their learned noise.
        new_init = log_std_init if log_std_init_new is None else log_std_init_new
        if loaded.k_params < k_params and new_init > -2.0:
            print(f"[param] WARNING: {k_params - loaded.k_params} new coordinates start at log_std {new_init:.2f} "
                  f"(std {np.exp(new_init):.2f}); measured on hitter v2, std >= 0.1 on the new knobs collapses a "
                  f"warm-started policy during training -- pass --log-std-init-new -2.5 or lower", flush=True)
        copy_actor_prefix(loaded, model, log_std_init=new_init)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def det_eval() -> dict:
        return deterministic_eval(model, feat_rms, specs, device, det_eval_episodes,
                                  det_eval_seed, params)

    def decide(features: np.ndarray) -> dict:
        """One parameter draw for one serve. Called by VecTTParam at every
        episode reset; the returned dict rides along to the episode's info."""
        raw = features.astype(np.float64)[None, :]
        feat_rms.update(raw)
        norm = feat_rms.normalize(raw)
        feats_t = torch.as_tensor(norm, dtype=torch.float32, device=device)
        z_t, logp_t, value_t = model.act(feats_t)
        z = z_t.cpu().numpy()[0].astype(np.float64)
        return {
            "u": np.tanh(z),
            "z": z,
            "logp": float(logp_t.cpu().numpy()[0]),
            "value": float(value_t.cpu().numpy()[0]),
            "feat_norm": norm[0].astype(np.float32),
        }

    vec = VecTTParam(n_envs=n_envs, seed=seed, params=params or {}, decide_fn=decide,
                     vel_penalty=vel_penalty, legal_bonus=legal_bonus, param_specs=specs,
                     aim_bonus=aim_bonus, aim_sigma=aim_sigma,
                     target_hit_bonus=target_hit_bonus, depth_bonus=depth_bonus,
                     fall_penalty=fall_penalty, obs_noise=obs_noise,
                     joint_noise=joint_noise, obs_delay_steps=obs_delay_steps)

    csv_path = os.path.join(out, "train_log.csv")
    csv_fh = open(csv_path, "w", newline="")
    writer = csv.writer(csv_fh)
    det_cols = ["det_legal_rate", "det_contact_rate", "det_target_rate", "det_landing_error_m"]
    gate_names = [name[len("gate_"):] for name in names if name.startswith("gate_")]
    writer.writerow(["episode", "wall_s", "mean_return", "legal_return_rate", "contact_rate"]
                    + det_cols + [f"u_{name}" for name in names] + [f"blend_{g}" for g in gate_names])
    csv_fh.flush()

    def det_cells(det: dict[str, float] | None) -> list[str]:
        if det is None:
            return [""] * len(det_cols)
        return [f"{det['legal_rate']:.2f}", f"{det['contact_rate']:.2f}",
                f"{det['target_rate']:.2f}", f"{det['landing_error_m']:.4f}"]

    window: deque[dict] = deque(maxlen=EPISODE_WINDOW)
    batch: list[dict] = []
    start = time.time()
    n_episodes = 0
    n_updates = 0
    last_log = 0
    last_det = 0
    best_legal_rate = -1.0
    best_det: dict | None = None
    best_det_episode = 0

    def maybe_save_best_det(det: dict, episode: int) -> None:
        nonlocal best_det, best_det_episode
        if best_det is None or _det_key(det, det_select) > _det_key(best_det, det_select):
            best_det, best_det_episode = det, episode
            save_checkpoint(os.path.join(out, "best.pt"), model, feat_rms, episode,
                            extra={"seed": seed, "det_eval": {**det, "n_serves": det_eval_episodes,
                                                              "seed": det_eval_seed}},
                            param_specs=specs)

    if det_eval_episodes > 0:
        # Row 0: the starting point's true (mean-action) score -- for a warm
        # start this is the checkpoint we resumed from, before any update.
        det0 = det_eval()
        writer.writerow([0, f"{time.time() - start:.2f}", "", "", ""] + det_cells(det0)
                        + [""] * (k_params + len(gate_names)))
        csv_fh.flush()
        print(f"[param] episode=0 det_legal_rate={det0['legal_rate']:.1f}% "
              f"det_target_rate={det0['target_rate']:.1f}% det_landing_error={det0['landing_error_m']:.3f}m "
              f"(n={det_eval_episodes}, seed={det_eval_seed})")
        maybe_save_best_det(det0, 0)

    try:
        while n_episodes < episodes:
            if minutes is not None and (time.time() - start) >= minutes * 60.0:
                print(f"[param] --minutes {minutes} wall cap reached at episode {n_episodes}")
                break

            _obs, _rew, _done, infos = vec.step()
            for info in infos:
                ep = info.get("episode")
                if ep is None:
                    continue
                window.append(ep)
                batch.append(ep)
                n_episodes += 1

            if len(batch) >= batch_episodes:
                bandit_update(
                    model, optimizer,
                    feats=np.stack([e["decision"]["feat_norm"] for e in batch]),
                    z=np.stack([e["decision"]["z"] for e in batch]),
                    old_logp=np.array([e["decision"]["logp"] for e in batch]),
                    returns=np.array([e["return"] for e in batch]),
                    device=device, entropy_coef=entropy_coef,
                )
                batch.clear()
                n_updates += 1

            if n_episodes - last_log >= eval_every:
                last_log = n_episodes
                mean_return = float(np.mean([e["return"] for e in window]))
                legal_rate = 100.0 * float(np.mean([e["legal_return"] for e in window]))
                contact_rate = 100.0 * float(np.mean([e["contact"] for e in window]))
                u_mean = np.mean([e["decision"]["u"] for e in window], axis=0)
                blend_mean = np.mean([blend_fractions(e["decision"]["u"], names) for e in window], axis=0)
                det = None
                if det_eval_episodes > 0 and n_episodes - last_det >= det_eval_every:
                    last_det = n_episodes
                    det = det_eval()
                wall_s = time.time() - start
                writer.writerow([n_episodes, f"{wall_s:.2f}", mean_return, legal_rate, contact_rate]
                                + det_cells(det) + [f"{v:.4f}" for v in u_mean]
                                + [f"{v:.4f}" for v in blend_mean])
                csv_fh.flush()
                det_text = ("" if det is None else
                            f" det_legal_rate={det['legal_rate']:.1f}% det_target_rate={det['target_rate']:.1f}%")
                print(f"[param] episode={n_episodes} update={n_updates} wall_s={wall_s:.1f} "
                      f"mean_return={mean_return:.3f} legal_return_rate={legal_rate:.1f}% "
                      f"contact_rate={contact_rate:.1f}%{det_text}", flush=True)

                if det is not None:
                    maybe_save_best_det(det, n_episodes)
                elif (det_eval_episodes == 0 and len(window) >= MIN_WINDOW_FOR_BEST
                        and legal_rate > best_legal_rate):
                    best_legal_rate = legal_rate
                    save_checkpoint(os.path.join(out, "best.pt"), model, feat_rms, n_episodes,
                                    extra={"legal_return_rate": legal_rate, "seed": seed},
                                    param_specs=specs)
    finally:
        # Weights first: nothing below may cost final.pt (a second Ctrl-C or an
        # eval failure). Then score the final weights so the last stretch of
        # training can still win best.pt, and re-save with the score attached.
        save_checkpoint(os.path.join(out, "final.pt"), model, feat_rms, n_episodes,
                        extra={"seed": seed}, param_specs=specs)
        final_det = None
        if det_eval_episodes > 0 and n_episodes > last_det:
            try:
                final_det = det_eval()
                maybe_save_best_det(final_det, n_episodes)
                save_checkpoint(os.path.join(out, "final.pt"), model, feat_rms, n_episodes,
                                extra={"seed": seed, "det_eval": final_det}, param_specs=specs)
            except BaseException as exc:  # noqa: BLE001 -- interrupt or eval failure: final.pt is already on disk
                print(f"[param] closing det_eval skipped: {exc!r}", flush=True)
        csv_fh.close()

    return {"episodes": n_episodes, "n_updates": n_updates, "out": out, "csv": csv_path,
            "param_space": param_space, "k_params": k_params,
            "best_legal_rate": best_legal_rate, "best_det": best_det,
            "best_det_episode": best_det_episode, "final_det": final_det,
            "wall_s": time.time() - start}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=20_000, help="stop after this many completed episodes")
    parser.add_argument("--minutes", type=float, default=None, help="hard wall-clock cap, minutes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=os.path.join("work", "tt", "rl", "param_run"))
    parser.add_argument("--eval-every", type=int, default=200, help="episodes between train_log.csv rows")
    parser.add_argument("--batch-episodes", type=int, default=BATCH_EPISODES,
                        help="completed episodes per policy update")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--log-std-init", type=float, default=-1.0,
                        help="initial exploration std (pre-squash), per parameter")
    parser.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF)
    parser.add_argument("--vel-penalty", type=float, default=0.0005, help="per-step weight on sum(joint_velocity^2)")
    parser.add_argument("--legal-bonus", type=float, default=5.0, help="one-off reward for a legal return")
    parser.add_argument("--param-space", type=str, default=DEFAULT_PARAM_SPACE,
                        choices=sorted(PARAM_SPACES), help="per-serve action space (v2 adds the swing library)")
    parser.add_argument("--aim-bonus", type=float, default=3.0,
                        help="coefficient on the legal-return aim term (3.0 = pre-v2 behaviour)")
    parser.add_argument("--aim-sigma", type=float, default=0.5,
                        help="aim-term kernel width in metres (0.5 = pre-v2 behaviour)")
    parser.add_argument("--target-hit-bonus", type=float, default=0.0,
                        help="extra reward when the return lands inside the evaluator's target radius")
    parser.add_argument("--depth-bonus", type=float, default=0.0,
                        help="penalty per metre the landing point falls SHORT of the target x (0 = off)")
    parser.add_argument("--fall-penalty", type=float, default=0.0,
                        help="free base only (--params '{\"free_base\": true}'): one-off penalty when the robot falls (0 = off)")
    parser.add_argument("--params", type=str, default="{}", help="JSON dict merged over himrobot.sim.scene.DEFAULT_PARAMS")
    parser.add_argument("--resume-from", type=str, default=None, help="continue from a param checkpoint")
    parser.add_argument("--log-std-init-new", type=float, default=None,
                        help="with --resume-from into a wider space: initial exploration std (pre-squash) "
                             "for the coordinates the checkpoint lacks (default: --log-std-init)")
    parser.add_argument("--det-eval-episodes", type=int, default=0,
                        help="select best.pt by a mean-action score on this many fixed serves (0 = noisy window)")
    parser.add_argument("--det-eval-every", type=int, default=1000,
                        help="episodes between deterministic evaluations")
    parser.add_argument("--det-eval-seed", type=int, default=1001,
                        help="serve-set seed for the deterministic evaluation (keep it off 11/7/3, the scoreboard)")
    parser.add_argument("--det-select", type=str, default="legal", choices=("legal", "target"),
                        help="rank best.pt by legal rate first (default) or by target-hit rate first (phase two)")
    parser.add_argument("--obs-noise", type=float, default=0.0,
                        help="train-time std (m) of noise on the ball position the hitter sees; velocity gets 10x")
    parser.add_argument("--joint-noise", type=float, default=0.0,
                        help="train-time std (rad) of noise on the joint readings the hitter sees")
    parser.add_argument("--obs-delay-steps", type=int, default=0,
                        help="train-time latency: the hitter sees observations this many control steps old")
    args = parser.parse_args(argv)

    result = train(
        n_envs=args.n_envs, episodes=args.episodes, seed=args.seed, out=args.out,
        minutes=args.minutes, eval_every=args.eval_every, batch_episodes=args.batch_episodes,
        lr=args.lr, log_std_init=args.log_std_init, entropy_coef=args.entropy_coef,
        params=json.loads(args.params), vel_penalty=args.vel_penalty,
        legal_bonus=args.legal_bonus, resume_from=args.resume_from,
        param_space=args.param_space, aim_bonus=args.aim_bonus,
        aim_sigma=args.aim_sigma, target_hit_bonus=args.target_hit_bonus,
        depth_bonus=args.depth_bonus, fall_penalty=args.fall_penalty,
        det_eval_episodes=args.det_eval_episodes, det_eval_every=args.det_eval_every,
        det_eval_seed=args.det_eval_seed, log_std_init_new=args.log_std_init_new,
        obs_noise=args.obs_noise, joint_noise=args.joint_noise, obs_delay_steps=args.obs_delay_steps,
        det_select=args.det_select,
    )
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
