"""Dataset loading for behavior cloning.

Two sources, chosen automatically:

  * LeRobot-v3 parquet episodes under ``cfg.data.root`` (what the real HIM
    table-tennis release exports to). Requires ``pyarrow``.
  * A synthetic sample, used when no parquet is present, so the training loop
    is runnable end-to-end today. The synthetic "expert" is a fixed, mildly
    non-trivial joint-state -> joint-target map, so BC has something learnable
    (and a trained model should clearly beat the hold baseline on it).

Both paths return per-timestep arrays plus episode boundaries, so a windowed
(sequence) model can be trained without leaking across episode edges.
"""

from __future__ import annotations

import glob
import os

import numpy as np

# Aliases tried when resolving a dataset column, most-specific first.
_OBS_ALIASES = ("observation.state.joints", "observation.state", "observation.joints", "joints")
_ACT_ALIASES = ("action.joint_targets", "action", "joint_targets")


def _resolve_column(columns, primary: str, aliases) -> str | None:
    for name in (primary, *aliases):
        if name in columns:
            return name
    return None


def _load_parquet(cfg):
    files = sorted(glob.glob(os.path.join(cfg.data.root, "**", "*.parquet"), recursive=True))
    if not files:
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # data present but reader missing -> be loud, don't silently synthesize
        raise RuntimeError(
            f"Found {len(files)} parquet file(s) under {cfg.data.root!r} but pyarrow is not "
            "installed. Install it (`pip install pyarrow pandas`) to train on real data."
        ) from exc

    obs_rows, act_rows, episodes = [], [], []
    cursor = 0
    for path in files:
        table = pq.read_table(path)
        cols = table.column_names
        okey = _resolve_column(cols, cfg.data.obs_joint_key, _OBS_ALIASES)
        akey = _resolve_column(cols, cfg.data.action_key, _ACT_ALIASES)
        if okey is None or akey is None:
            raise RuntimeError(
                f"{path}: could not resolve obs/action columns. "
                f"Looked for obs~{cfg.data.obs_joint_key!r}, act~{cfg.data.action_key!r}; "
                f"available columns: {cols}"
            )
        obs = np.asarray(table.column(okey).to_pylist(), dtype=np.float32)
        act = np.asarray(table.column(akey).to_pylist(), dtype=np.float32)
        if cfg.data.ball_key and cfg.data.ball_key in cols:
            ball = np.asarray(table.column(cfg.data.ball_key).to_pylist(), dtype=np.float32)
            obs = np.concatenate([obs, ball], axis=1)
        n = min(len(obs), len(act))
        obs_rows.append(obs[:n])
        act_rows.append(act[:n])
        episodes.append((cursor, cursor + n))
        cursor += n

    return np.concatenate(obs_rows), np.concatenate(act_rows), episodes


def _synthetic(cfg):
    rng = np.random.default_rng(cfg.train.seed)
    J = cfg.data.joints
    # Fixed "expert": joint_targets = A @ q + b, plus a small ball influence when enabled.
    A = 0.9 * np.eye(J) + (0.05 / np.sqrt(J)) * rng.standard_normal((J, J))
    b = 0.02 * rng.standard_normal(J)
    Wb = (0.1 / np.sqrt(cfg.data.ball_dim)) * rng.standard_normal((J, cfg.data.ball_dim))

    obs_rows, act_rows, episodes = [], [], []
    cursor = 0
    for _ in range(cfg.data.synth_episodes):
        q = 0.3 * rng.standard_normal(J)               # start pose
        ep_obs, ep_act = [], []
        for _ in range(cfg.data.synth_steps):
            q = 0.98 * q + 0.02 * rng.standard_normal(J)  # smooth joint random walk
            target = A @ q + b
            if cfg.data.ball_key:
                ball = 0.5 * rng.standard_normal(cfg.data.ball_dim)
                target = target + Wb @ ball
                ep_obs.append(np.concatenate([q, ball]).astype(np.float32))
            else:
                ep_obs.append(q.astype(np.float32))
            target = target + 0.01 * rng.standard_normal(J)  # observation noise
            ep_act.append(target.astype(np.float32))
        n = len(ep_obs)
        obs_rows.extend(ep_obs)
        act_rows.extend(ep_act)
        episodes.append((cursor, cursor + n))
        cursor += n

    return np.asarray(obs_rows, np.float32), np.asarray(act_rows, np.float32), episodes


def build_arrays(cfg):
    """Return (obs[N,obs_dim], act[N,act_dim], episodes, source_str)."""
    loaded = _load_parquet(cfg)
    if loaded is not None:
        obs, act, episodes = loaded
        return obs, act, episodes, f"lerobot-parquet:{cfg.data.root}"
    obs, act, episodes = _synthetic(cfg)
    return obs, act, episodes, "synthetic"


def make_windows(obs, act, episodes, context: int):
    """Build (X[N,context,obs_dim], Y[N,act_dim]) windows that never cross episode edges.

    Left-pads short windows by repeating the first available frame so early
    timesteps still produce a training target.
    """
    if context <= 1:
        return obs[:, None, :], act  # [N,1,obs_dim] keeps a uniform rank

    xs, ys = [], []
    for start, end in episodes:
        for t in range(start, end):
            lo = max(start, t - context + 1)
            window = obs[lo : t + 1]
            if len(window) < context:
                pad = np.repeat(window[:1], context - len(window), axis=0)
                window = np.concatenate([pad, window])
            xs.append(window)
            ys.append(act[t])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)
