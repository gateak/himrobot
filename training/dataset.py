"""Dataset loading for behavior cloning.

Three sources, chosen automatically from what is under ``cfg.data.root``:

  1. LeRobot-v3 parquet episodes (what a real HIM robot obs/action release
     would export to). Requires ``pyarrow``. Takes priority when present.
  2. Retargeted G1 motion CSVs (``sim/gmr_bvh_to_g1_headless.py`` output):
     one ``*.csv`` per clip, no header, each row =
     ``root xyz (3, m) | root quat xyzw (4) | 29 joint angles (rad)`` at 30 fps,
     joints in the order of ``gmr_g1_joint_order.txt`` alongside (checked against
     the runtime contract's ``extensions.robot.joint_order``). Selected when the
     root holds CSVs and no parquet. Becomes a next-pose behavior-cloning set:
     ``obs = joints[t]``, ``act = joints[t+1]``; one episode per file.
     By default the root pose is EXCLUDED from obs (see ``_load_csv`` for why);
     ``cfg.data.csv_root_in_obs`` appends root height + quat.
  3. A synthetic sample, used when neither is present, so the training loop
     is runnable end-to-end anywhere. The synthetic "expert" is a fixed, mildly
     non-trivial joint-state -> joint-target map, so BC has something learnable.

All paths return per-timestep arrays plus episode boundaries, so a windowed
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


def check_csv_joint_order(root: str, cfg) -> str:
    """Compare ``<root>/<csv_joint_order_file>`` with the contract joint order.

    Returns a short note for provenance. Raises if the file exists and disagrees:
    a mis-ordered checkpoint would emit actions in the wrong joint order at deploy
    time, and that must not pass silently.
    """
    path = os.path.join(root, cfg.data.csv_joint_order_file)
    if not os.path.isfile(path):
        return "joint order file absent (order assumed = contract)"
    with open(path) as fh:
        names = [ln.strip() for ln in fh if ln.strip()]
    expected = list(cfg.data.joint_order)
    if names != expected:
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(names, expected)) if a != b]
        raise RuntimeError(
            f"{path}: joint order differs from the runtime contract "
            f"(extensions.robot.joint_order): {len(names)} vs {len(expected)} names; "
            f"first mismatches {diff[:3]}"
        )
    return f"joint order verified against contract ({len(names)} names)"


def _load_csv(cfg):
    """Retargeted G1 motion CSVs -> next-pose BC arrays.

    Observation design: obs[t] = joint angles at t (29). The root position and
    orientation are dropped by default because (a) the deployment contract only
    provides ``robot.joint_position`` -- there is no root/base signal to feed
    at inference time, and (b) root xy is world-frame mocap placement, which is
    meaningless to a proprioceptive policy. With ``csv_root_in_obs=True`` the
    root height z and quaternion xyzw are appended (5 extra dims, joints still
    first) for offline experiments; such a checkpoint cannot be served by
    ``him_adapter`` as-is.
    Action: act[t] = joint angles at t+1 (the pose 1/30 s later). The last
    frame of each clip has no successor and is dropped.
    """
    files = sorted(glob.glob(os.path.join(cfg.data.root, "*.csv")))
    if not files:
        return None
    J, R = cfg.data.joints, cfg.data.csv_root_cols
    want = R + J
    order_note = check_csv_joint_order(cfg.data.root, cfg)

    obs_rows, act_rows, episodes = [], [], []
    cursor = 0
    for path in files:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
        if arr.shape[1] != want:
            raise RuntimeError(
                f"{path}: expected {want} columns (root {R} + joints {J}), got {arr.shape[1]}"
            )
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{path}: non-finite values")
        if len(arr) < 2:
            continue  # no (t, t+1) pair
        joints = arr[:, R:].astype(np.float32)
        obs = joints[:-1]
        if cfg.data.csv_root_in_obs:
            root_feat = arr[:-1, [2, 3, 4, 5, 6]].astype(np.float32)  # z, qx, qy, qz, qw
            obs = np.concatenate([obs, root_feat], axis=1)
        act = joints[1:]
        n = len(obs)
        obs_rows.append(obs)
        act_rows.append(act)
        episodes.append((cursor, cursor + n))
        cursor += n
    if not obs_rows:
        raise RuntimeError(f"{cfg.data.root}: CSV files present but none has >= 2 rows")
    return np.concatenate(obs_rows), np.concatenate(act_rows), episodes, order_note


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
    """Return (obs[N,obs_dim], act[N,act_dim], episodes, source_str).

    Priority: parquet > csv > synthetic.
    """
    loaded = _load_parquet(cfg)
    if loaded is not None:
        obs, act, episodes = loaded
        return obs, act, episodes, f"lerobot-parquet:{cfg.data.root}"
    loaded = _load_csv(cfg)
    if loaded is not None:
        obs, act, episodes, note = loaded
        root_note = "obs=joints+root(z,quat)" if cfg.data.csv_root_in_obs else "obs=joints only"
        return obs, act, episodes, (
            f"g1-motion-csv:{cfg.data.root} ({len(episodes)} clips @ {cfg.data.csv_fps:g} fps, "
            f"next-pose BC, {root_note}; {note})"
        )
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


def compute_norm_stats(X: np.ndarray, Y: np.ndarray, act_dim: int):
    """Per-feature (mean, std) for obs windows ``X`` [N,T,obs_dim] and actions
    ``Y`` [N,act_dim], with the first ``act_dim`` obs columns forced to use the
    *action's* statistics.

    Those columns are the current joints, the same physical quantity (and, for
    every source in this module, the same units) as the action -- see the
    module docstring and ``build_arrays``. Fitting them independently (obs
    stats over ``X``, action stats over ``Y``) makes the two normalizations
    drift apart whenever the obs and action distributions differ even
    slightly, which silently breaks the residual heads' "zero-init == hold"
    identity (``training/model.py``): denormalizing a zero-init prediction
    with the action stats no longer reproduces the raw current-joint value.
    Aligning the stats here keeps that identity exact regardless of how obs
    and act happen to be distributed.
    """
    obs_mean = X.reshape(-1, X.shape[-1]).mean(0)
    obs_std = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    act_mean = Y.mean(0)
    act_std = Y.std(0) + 1e-6
    obs_mean[:act_dim] = act_mean
    obs_std[:act_dim] = act_std
    return obs_mean, obs_std, act_mean, act_std


def split_indices(n: int, episodes, val_frac: float, seed: int, mode: str = "random"):
    """Deterministic (train_idx, val_idx) over the N windowed samples.

    ``random``  : per-step permutation (adjacent frames can land on both sides --
                  optimistic for next-pose data).
    ``episode`` : hold out whole episodes (the last ceil(val_frac * n_episodes)
                  in a seeded shuffle), so validation frames come from clips the
                  model never saw. Requires >= 2 episodes.
    """
    rng = np.random.default_rng(seed)
    if mode == "random":
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_frac))
        return perm[n_val:], perm[:n_val]
    if mode == "episode":
        if len(episodes) < 2:
            raise ValueError("episode split needs at least 2 episodes")
        ep_perm = rng.permutation(len(episodes))
        n_val_ep = max(1, int(np.ceil(len(episodes) * val_frac)))
        n_val_ep = min(n_val_ep, len(episodes) - 1)
        val_eps = set(ep_perm[:n_val_ep].tolist())
        is_val = np.zeros(n, dtype=bool)
        for i, (s, e) in enumerate(episodes):
            if i in val_eps:
                is_val[s:e] = True
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)
    raise ValueError(f"unknown val_split: {mode!r}")
