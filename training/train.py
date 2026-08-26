"""Behavior-cloning training loop.

Run (from repo root, inside the torch venv):
    ./.venv/bin/python -m training.train                 # MLP on synthetic sample
    ./.venv/bin/python -m training.train --model transformer --epochs 30
    ./.venv/bin/python -m training.train --root data --ball   # real parquet + ball_state

Writes a self-contained checkpoint (default models/bc_policy.pt) that
him_adapter.policy loads at deploy time.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .dataset import build_arrays, make_windows
from .model import build_model, save_checkpoint


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_args(cfg: Config) -> Config:
    p = argparse.ArgumentParser(description="Behavior-cloning trainer for the HIM policy")
    p.add_argument("--model", choices=["mlp", "transformer"], default=cfg.model.kind)
    p.add_argument("--epochs", type=int, default=cfg.train.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.train.batch_size)
    p.add_argument("--lr", type=float, default=cfg.train.lr)
    p.add_argument("--root", default=cfg.data.root, help="dir of LeRobot-v3 parquet episodes")
    p.add_argument("--joints", type=int, default=cfg.data.joints)
    p.add_argument("--ball", action="store_true", help="concat ball_state to observation")
    p.add_argument("--context", type=int, default=cfg.model.context, help="transformer history window")
    p.add_argument("--seed", type=int, default=cfg.train.seed)
    p.add_argument("--device", default=cfg.train.device)
    p.add_argument("--out", default=cfg.train.out)
    a = p.parse_args()

    cfg.model.kind = a.model
    cfg.model.context = a.context
    cfg.train.epochs = a.epochs
    cfg.train.batch_size = a.batch_size
    cfg.train.lr = a.lr
    cfg.train.seed = a.seed
    cfg.train.device = a.device
    cfg.train.out = a.out
    cfg.data.root = a.root
    cfg.data.joints = a.joints
    if a.ball:
        cfg.data.ball_key = "observation.state.ball"
    return cfg


def main() -> int:
    cfg = _parse_args(Config())
    torch.manual_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    obs, act, episodes, source = build_arrays(cfg)
    print(f"data source     : {source}")
    print(f"samples/episodes: {len(obs)} steps across {len(episodes)} episodes")
    print(f"obs_dim/act_dim : {obs.shape[1]} -> {act.shape[1]}  (device={device.type})")
    if obs.shape[1] != cfg.obs_dim or act.shape[1] != cfg.act_dim:
        print(f"[warn] data dims ({obs.shape[1]}->{act.shape[1]}) differ from config "
              f"({cfg.obs_dim}->{cfg.act_dim}); using data dims. Update config for real data.")
        cfg.data.joints = act.shape[1]  # keep act_dim authoritative

    X, Y = make_windows(obs, act, episodes, cfg.context)

    # deterministic train/val split
    n = len(X)
    rng = np.random.default_rng(cfg.train.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * cfg.train.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # normalization from TRAIN only (last obs frame drives obs stats for windowed input)
    obs_flat_train = X[train_idx].reshape(-1, X.shape[-1])
    obs_mean = obs_flat_train.mean(0)
    obs_std = obs_flat_train.std(0) + 1e-6
    act_mean = Y[train_idx].mean(0)
    act_std = Y[train_idx].std(0) + 1e-6

    Xn = (X - obs_mean) / obs_std
    Yn = (Y - act_mean) / act_std

    Xt = torch.from_numpy(Xn.astype(np.float32))
    Yt = torch.from_numpy(Yn.astype(np.float32))
    if cfg.context <= 1:
        Xt = Xt.squeeze(1)  # MLP wants [B, obs_dim]

    train_ds = TensorDataset(Xt[train_idx], Yt[train_idx])
    val_ds = TensorDataset(Xt[val_idx], Yt[val_idx])
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    model = build_model(cfg.model, cfg.obs_dim, cfg.act_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = torch.nn.MSELoss()

    # "hold" baseline in normalized action space: predict the current joint state.
    # (only comparable directly when obs joints and act share dim/units, which holds here)
    def hold_baseline_val() -> float:
        with torch.no_grad():
            last_obs = X[val_idx][:, -1, : cfg.act_dim]         # current joints
            pred = (last_obs - act_mean) / act_std
            tgt = Yn[val_idx]
            return float(np.mean((pred - tgt) ** 2))

    best_val = float("inf")
    provenance = (
        f"source={source}; shapes from SDK example manifests (example_only tier), "
        "UNCONFIRMED against real data"
    )
    os.makedirs(os.path.dirname(cfg.train.out) or ".", exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            seen += len(xb)
        train_loss = tot / max(seen, 1)

        model.eval()
        vtot, vseen = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vtot += loss_fn(model(xb), yb).item() * len(xb)
                vseen += len(xb)
        val_loss = vtot / max(vseen, 1)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                cfg.train.out,
                model=model.to("cpu"),
                cfg=cfg,
                norm={
                    "obs_mean": obs_mean.tolist(), "obs_std": obs_std.tolist(),
                    "act_mean": act_mean.tolist(), "act_std": act_std.tolist(),
                },
                val_loss=val_loss,
                provenance=provenance,
            )
            model.to(device)
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.train.epochs:
            print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}")

    print(f"\nbest val MSE (normalized) : {best_val:.5f}")
    print(f"hold baseline val MSE      : {hold_baseline_val():.5f}  (lower model MSE => BC beats hold)")
    print(f"saved checkpoint           : {cfg.train.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
