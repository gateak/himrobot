"""Behavior-cloning trainer: fits tt_sim.rl.nets.Actor to tt_sim/rl/demos.py's
recorded (obs, action) pairs with plain per-step MSE.

    .venv/bin/python -m tt_sim.rl.bc --demos work/tt/rl/demos_seed0.npz \\
        --out work/tt/rl/bc_seed0.pt --epochs 20 --seed 0

Checkpoint format (torch.save'd dict, loaded with weights_only=False since
it carries plain-Python metadata alongside the state_dict):

    {
      "model_state_dict": Actor(...).state_dict(),
      "obs_dim": int, "act_dim": int, "hidden": int,
      "epochs": int, "seed": int, "val_frac": float,
      "train_loss": [float, ...],       # one per epoch
      "val_loss": [float or None, ...], # one per epoch, None if val_frac
                                         # left no held-out episodes
      "n_train_episodes": int, "n_val_episodes": int,
    }

so a future tt_sim/rl/ppo.py's `--init-from` can do:

    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor = Actor(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"])
    actor.load_state_dict(ckpt["model_state_dict"])

tt_sim/rl/policy.py's BCPolicy loads the same file for
`--policy tt_sim.rl.policy:BCPolicy` in tt_sim.evaluate.

The 10% held-out split is done at the EPISODE level (whole contact-episode
trajectories, not shuffled steps) since adjacent steps within one episode
are highly correlated -- splitting steps would leak a swing's neighbouring
frames into both train and val.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from tt_sim.rl.nets import Actor

DEFAULT_EPOCHS = 20
DEFAULT_VAL_FRAC = 0.10
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3


def load_demos(npz_path: str) -> dict:
    with np.load(npz_path) as z:
        return {k: z[k] for k in z.files}


def _episode_bounds(episode_starts: np.ndarray, n_steps: int) -> list[tuple[int, int]]:
    starts = list(episode_starts) + [n_steps]
    return [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]


def split_episodes(n_episodes: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Episode indices [0, n_episodes) split into (train_idx, val_idx).
    Keeps at least 1 train episode; val is empty if val_frac*n_episodes
    rounds to 0 or fewer than 2 episodes exist (nothing sensible to hold out).
    """
    idx = np.arange(n_episodes)
    if n_episodes < 2:
        return idx, np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(0, min(n_episodes - 1, round(n_episodes * val_frac)))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def _steps_for_episodes(bounds: list[tuple[int, int]], ep_idx: np.ndarray) -> np.ndarray:
    if len(ep_idx) == 0:
        return np.array([], dtype=np.int64)
    return np.concatenate([np.arange(bounds[e][0], bounds[e][1]) for e in ep_idx])


def _pick_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    data: dict,
    epochs: int = DEFAULT_EPOCHS,
    val_frac: float = DEFAULT_VAL_FRAC,
    seed: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    device: str | None = None,
    hidden: int = 256,
) -> tuple[Actor, dict]:
    """Train an Actor on data (a roll_demos()-shaped dict, or an
    np.load'd demos_*.npz) with per-step MSE. Returns (actor, history)
    where history has train_loss/val_loss lists (one entry per epoch) plus
    n_train_episodes/n_val_episodes/device.
    """
    obs = np.asarray(data["obs"], dtype=np.float32)
    act = np.asarray(data["act"], dtype=np.float32)
    episode_starts = np.asarray(data["episode_starts"], dtype=np.int64)
    n_steps = obs.shape[0]
    obs_dim = obs.shape[1]
    act_dim = act.shape[1]

    dev = _pick_device(device)
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden).to(dev)

    history = {
        "train_loss": [], "val_loss": [],
        "n_train_episodes": 0, "n_val_episodes": 0, "device": str(dev),
    }

    n_episodes = len(episode_starts)
    if n_episodes == 0 or n_steps == 0:
        # Nothing to train on (e.g. a tiny/unlucky demo run with zero
        # contact trials) -- return an untrained-but-valid actor rather
        # than crashing, so callers can still save/evaluate it.
        history["train_loss"] = [float("nan")] * epochs
        history["val_loss"] = [None] * epochs
        return actor, history

    bounds = _episode_bounds(episode_starts, n_steps)
    train_ep, val_ep = split_episodes(n_episodes, val_frac, seed)
    train_idx = _steps_for_episodes(bounds, train_ep)
    val_idx = _steps_for_episodes(bounds, val_ep)
    history["n_train_episodes"] = int(len(train_ep))
    history["n_val_episodes"] = int(len(val_ep))

    obs_t = torch.from_numpy(obs).to(dev)
    act_t = torch.from_numpy(act).to(dev)
    train_idx_t = torch.from_numpy(train_idx).long()
    val_obs = obs_t[torch.from_numpy(val_idx).long()] if len(val_idx) else None
    val_act = act_t[torch.from_numpy(val_idx).long()] if len(val_idx) else None

    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)

    n_train = len(train_idx_t)
    for _epoch in range(epochs):
        perm = train_idx_t[torch.from_numpy(rng.permutation(n_train))]
        actor.train()
        running = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            batch_idx = perm[start:start + batch_size]
            pred = actor(obs_t[batch_idx])
            loss = loss_fn(pred, act_t[batch_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        history["train_loss"].append(running / max(1, n_batches))

        if val_obs is not None and len(val_obs) > 0:
            actor.eval()
            with torch.no_grad():
                vloss = float(loss_fn(actor(val_obs), val_act).item())
            history["val_loss"].append(vloss)
        else:
            history["val_loss"].append(None)

    actor.eval()
    return actor, history


def save_checkpoint(actor: Actor, path: str, history: dict, seed: int, epochs: int, val_frac: float) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": actor.to("cpu").state_dict(),
        "obs_dim": actor.obs_dim, "act_dim": actor.act_dim, "hidden": actor.hidden,
        "epochs": epochs, "seed": seed, "val_frac": val_frac,
        "train_loss": history["train_loss"], "val_loss": history["val_loss"],
        "n_train_episodes": history["n_train_episodes"], "n_val_episodes": history["n_val_episodes"],
    }, path)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=str, default=os.path.join("work", "tt", "rl", "demos_seed0.npz"))
    parser.add_argument("--out", type=str, default=os.path.join("work", "tt", "rl", "bc_seed0.pt"))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", type=str, default=None, help="cpu | mps (default: mps if available else cpu)")
    args = parser.parse_args(argv)

    data = load_demos(args.demos)
    start = time.time()
    actor, history = train(
        data, epochs=args.epochs, val_frac=args.val_frac, seed=args.seed,
        batch_size=args.batch_size, lr=args.lr, device=args.device,
    )
    wall_time_s = time.time() - start
    save_checkpoint(actor, args.out, history, seed=args.seed, epochs=args.epochs, val_frac=args.val_frac)

    summary = {
        "demos_path": args.demos, "out_path": args.out, "epochs": args.epochs,
        "seed": args.seed, "val_frac": args.val_frac, "device": history["device"],
        "n_train_episodes": history["n_train_episodes"], "n_val_episodes": history["n_val_episodes"],
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "train_loss": history["train_loss"], "val_loss": history["val_loss"],
        "wall_time_s": wall_time_s,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
