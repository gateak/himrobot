"""Behavior-cloning training loop.

Run from repo root, inside the torch venv:
    ./.venv/bin/python -m legacy.bc.train --root data/g1 --epochs 40 --val-split episode
        # next-pose BC on retargeted G1 CSVs
    ./.venv/bin/python -m legacy.bc.train --root /nonexistent  # MLP on the synthetic sample
    ./.venv/bin/python -m legacy.bc.train --model transformer --context 8 --epochs 30
    ./.venv/bin/python -m legacy.bc.train --root data/parquet --ball  # LeRobot parquet + ball_state

Writes a self-contained checkpoint (default models/bc_policy.pt) that him_adapter.policy loads at deploy time.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .dataset import build_arrays, compute_norm_stats, make_windows, split_indices
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
    p.add_argument("--root", default=cfg.data.root,
                   help="dir of G1 motion CSVs or LeRobot-v3 parquet episodes (else synthetic)")
    p.add_argument("--joints", type=int, default=cfg.data.joints)
    p.add_argument("--ball", action="store_true", help="concat ball_state to observation (parquet/synthetic)")
    p.add_argument("--csv-root-obs", action="store_true",
                   help="CSV source: append root height + quat xyzw to obs (not deployable via him_adapter)")
    p.add_argument("--context", type=int, default=cfg.model.context,
                   help="transformer history window (ignored for --model mlp, which is always 1); "
                        "1 works but degenerates to single-step attention, no history")
    p.add_argument("--no-residual", action="store_true", help="plain head instead of hold + delta")
    p.add_argument("--val-split", choices=["random", "episode"], default=cfg.train.val_split,
                   help="random = per-step split; episode = hold out whole clips")
    p.add_argument("--val-frac", type=float, default=cfg.train.val_frac)
    p.add_argument("--seed", type=int, default=cfg.train.seed)
    p.add_argument("--device", default=cfg.train.device)
    p.add_argument("--out", default=cfg.train.out)
    a = p.parse_args()

    cfg.model.kind = a.model
    cfg.model.context = a.context
    cfg.model.residual = not a.no_residual
    cfg.train.epochs = a.epochs
    cfg.train.batch_size = a.batch_size
    cfg.train.lr = a.lr
    cfg.train.val_split = a.val_split
    cfg.train.val_frac = a.val_frac
    cfg.train.seed = a.seed
    cfg.train.device = a.device
    cfg.train.out = a.out
    cfg.data.root = a.root
    cfg.data.joints = a.joints
    cfg.data.csv_root_in_obs = a.csv_root_obs
    if a.ball:
        cfg.data.ball_key = "observation.state.ball"
    return cfg


def _provenance(cfg: Config, source: str) -> str:
    common = (
        f"joints={cfg.data.joints} in .him/contracts/runtime.json extensions.robot.joint_order "
        f"(Unitree G1, Menagerie/SDK order); deploy signals "
        f"{cfg.data.deploy_obs_signal}->{cfg.data.deploy_act_signal} UNCONFIRMED against HIM's evaluator"
    )
    if source.startswith("g1-motion-csv"):
        return (
            f"source={source}; data = GMR-retargeted Rokoko human mocap (kinematic poses @ "
            f"{cfg.data.csv_fps:g} fps), NOT robot obs/action data; model = next-pose predictor "
            f"(joints[t] -> joints[t+1]), a motion prior, not a table-tennis controller; {common}"
        )
    if source.startswith("lerobot-parquet"):
        return f"source={source}; column names/units per the parquet files; {common}"
    return f"source={source}; synthetic expert, no real data; {common}"


def main() -> int:
    t0 = time.time()
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
        cfg.data.ball_key = None
        cfg.data.csv_root_in_obs = False
        assert cfg.act_dim == act.shape[1]

    if len(cfg.data.joint_order) != cfg.data.joints:
        print(f"[warn] joint_order has {len(cfg.data.joint_order)} names (from "
              f".him/contracts/runtime.json) but act_dim is {cfg.data.joints} (--joints, or the "
              f"data's own width); the checkpoint's joint_order metadata will not match its "
              f"{cfg.data.joints} outputs and is informational only -- do not use it to label them.")

    X, Y = make_windows(obs, act, episodes, cfg.context)
    n = len(X)
    train_idx, val_idx = split_indices(n, episodes, cfg.train.val_frac, cfg.train.seed, cfg.train.val_split)
    print(f"split           : {cfg.train.val_split}, train {len(train_idx)} / val {len(val_idx)}")

    # normalization from TRAIN only. The first act_dim obs columns (the current
    # joints) are forced onto the action's own stats -- see compute_norm_stats --
    # so a zero-init residual head exactly reproduces "hold" (legacy/bc/model.py).
    obs_mean, obs_std, act_mean, act_std = compute_norm_stats(X[train_idx], Y[train_idx], cfg.act_dim)

    Xn = (X - obs_mean) / obs_std
    Yn = (Y - act_mean) / act_std

    Xt = torch.from_numpy(Xn.astype(np.float32))
    Yt = torch.from_numpy(Yn.astype(np.float32))
    if cfg.model.kind == "mlp":
        Xt = Xt.squeeze(1)  # MLP wants [B, obs_dim]; transformer always wants [B, T, obs_dim],
                             # even at context 1 -- squeezing that too crashes its 3-D forward.

    train_ds = TensorDataset(Xt[train_idx], Yt[train_idx])
    val_ds = TensorDataset(Xt[val_idx], Yt[val_idx])
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    obs_dim = X.shape[-1]
    model = build_model(cfg.model, obs_dim, cfg.act_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = torch.nn.MSELoss()

    # "hold" baseline in normalized action space: predict the current joint state.
    # Valid because the first act_dim obs columns are the current joints in the same
    # order/units as the action (all three sources put the joints first).
    def hold_baseline_val() -> float:
        with torch.no_grad():
            last_obs = X[val_idx][:, -1, : cfg.act_dim]         # current joints
            pred = (last_obs - act_mean) / act_std
            tgt = Yn[val_idx]
            return float(np.mean((pred - tgt) ** 2))

    best_val = float("inf")
    provenance = _provenance(cfg, source)
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
                extra={"source": source, "val_split": cfg.train.val_split,
                       "hold_baseline_val_mse": hold_baseline_val(), "epoch": epoch,
                       "obs_dim": obs_dim},
            )
            model.to(device)
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.train.epochs:
            print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}")

    print(f"\nbest val MSE (normalized) : {best_val:.5f}")
    print(f"hold baseline val MSE      : {hold_baseline_val():.5f}  (lower model MSE => BC beats hold)")
    print(f"device / wall time         : {device.type} / {time.time() - t0:.1f} s")
    print(f"saved checkpoint           : {cfg.train.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
