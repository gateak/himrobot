"""Contextual-bandit trainer for tt_sim.rl.param_residual's per-serve params.

    python -m tt_sim.rl.param_train --n-envs 32 --episodes 20000 --minutes 8 \\
        --seed 0 --out work/tt/rl/param_run --eval-every 200

Each episode is a single decision (one parameter vector per serve), so there
is no temporal credit assignment: advantage = episode_return - V(features),
learned by a clipped-ratio (PPO-style) update on the pre-squash Gaussian, with
a value head as baseline. Clipping rather than plain REINFORCE only because it
makes the step size forgiving when a batch happens to be mostly failures; the
bandit structure is what removes GAE, bootstrapping and per-step noise.

Writes train_log.csv (episode, wall_s, mean_return, legal_return_rate,
contact_rate -- each a mean over the last 200 completed episodes), best.pt at
every new best windowed legal_return_rate, and final.pt at exit. Both
checkpoints carry the feature-normalizer stats and the parameter ranges, and
are loadable by tt_sim.evaluate via
``--policy tt_sim.rl.policy:ParamResidualPolicy`` with ``TT_RL_CKPT=<path>``.
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

from tt_sim.rl.param_residual import (
    FEATURE_DIM,
    K_PARAMS,
    ParamActorCritic,
    VecTTParam,
    save_checkpoint,
)
from tt_sim.rl.ppo import RunningMeanStd, pick_device

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
                  entropy_coef: float = ENTROPY_COEF) -> dict:
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


def train(n_envs: int = 32, episodes: int = 20_000, seed: int = 0,
          out: str = os.path.join("work", "tt", "rl", "param_run"),
          minutes: float | None = None, eval_every: int = 200,
          batch_episodes: int = BATCH_EPISODES, lr: float = LR,
          log_std_init: float = -1.0, entropy_coef: float = ENTROPY_COEF,
          params: dict | None = None, vel_penalty: float = 0.0005,
          legal_bonus: float = 5.0, resume_from: str | None = None) -> dict:
    os.makedirs(out, exist_ok=True)
    device = pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ParamActorCritic(FEATURE_DIM, K_PARAMS, log_std_init=log_std_init).to(device)
    feat_rms = RunningMeanStd((FEATURE_DIM,))
    if resume_from:
        from tt_sim.rl.param_residual import load_checkpoint
        loaded, feat_rms, _specs = load_checkpoint(resume_from, device)
        model.load_state_dict(loaded.state_dict())
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
                     vel_penalty=vel_penalty, legal_bonus=legal_bonus)

    csv_path = os.path.join(out, "train_log.csv")
    csv_fh = open(csv_path, "w", newline="")
    writer = csv.writer(csv_fh)
    writer.writerow(["episode", "wall_s", "mean_return", "legal_return_rate", "contact_rate"])
    csv_fh.flush()

    window: deque = deque(maxlen=EPISODE_WINDOW)
    batch: list[dict] = []
    start = time.time()
    n_episodes = 0
    n_updates = 0
    last_log = 0
    best_legal_rate = -1.0

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
                wall_s = time.time() - start
                writer.writerow([n_episodes, f"{wall_s:.2f}", mean_return, legal_rate, contact_rate])
                csv_fh.flush()
                print(f"[param] episode={n_episodes} update={n_updates} wall_s={wall_s:.1f} "
                      f"mean_return={mean_return:.3f} legal_return_rate={legal_rate:.1f}% "
                      f"contact_rate={contact_rate:.1f}%")

                if len(window) >= MIN_WINDOW_FOR_BEST and legal_rate > best_legal_rate:
                    best_legal_rate = legal_rate
                    save_checkpoint(os.path.join(out, "best.pt"), model, feat_rms, n_episodes,
                                    extra={"legal_return_rate": legal_rate, "seed": seed})
    finally:
        save_checkpoint(os.path.join(out, "final.pt"), model, feat_rms, n_episodes,
                        extra={"seed": seed})
        csv_fh.close()

    return {"episodes": n_episodes, "n_updates": n_updates, "out": out, "csv": csv_path,
            "best_legal_rate": best_legal_rate, "wall_s": time.time() - start}


def main(argv=None) -> dict:
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
    parser.add_argument("--params", type=str, default="{}", help="JSON dict merged over tt_sim.scene.DEFAULT_PARAMS")
    parser.add_argument("--resume-from", type=str, default=None, help="continue from a param checkpoint")
    args = parser.parse_args(argv)

    result = train(
        n_envs=args.n_envs, episodes=args.episodes, seed=args.seed, out=args.out,
        minutes=args.minutes, eval_every=args.eval_every, batch_episodes=args.batch_episodes,
        lr=args.lr, log_std_init=args.log_std_init, entropy_coef=args.entropy_coef,
        params=json.loads(args.params), vel_penalty=args.vel_penalty,
        legal_bonus=args.legal_bonus, resume_from=args.resume_from,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
