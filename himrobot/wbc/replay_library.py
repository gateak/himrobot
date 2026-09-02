"""The MuJoCo half of Gate 0 (docs/design/whole_body_v1.md sec 7.2) and the
per-checkpoint sim-to-sim check: replays a recorded swing library on the
honest free-base scene (``robot_table_collision=False``, so the body cannot
lean on the table -- see himrobot.sim.scene's module docstring) and reports
whether the G1 stands.

    PYTHONPATH=. TT_ROBOT_Y=-0.10 .venv/bin/python -m himrobot.wbc.replay_library \\
        --lib work/wbc/swing_lib_eval.npz --mode with-ball

``TT_ROBOT_Y`` is read once at himrobot.sim.scene's *import* time (see that
module's own comment on the same variable) -- set it in the environment
before this process starts if the library was recorded at the shipped
-0.10 m stance; this file does not set it itself.

Swing-library npz schema (himrobot.wbc.record_library.py's output; this file
is written against the schema documented in docs/design/whole_body_v1.md sec
4.1, since record_library.py has not landed yet at the time this module was
written -- one row per served swing):

    u                float32 [N, T, 17]  JOINT_ORDER[12:29] targets (waist
                                          yaw/roll/pitch + left arm x7 +
                                          right arm x7), one per control step
    swing_start_step int32   [N]         first step where u differs from
                                          scene.STAND_POSE[12:]
    contact_step     int32   [N]         control step nearest predicted contact
    t_contact        float32 [N]         seconds (env clock) of the swing plan
    t_pre            float32 [N]         seconds, backswing/ease-in width
    t_follow         float32 [N]         seconds, follow-through width
    serve_pos        float32 [N, 3]      the row's serve (scene.sample_serve)
    serve_vel        float32 [N, 3]
    welded_legal     bool    [N]         legal_return() when this row was
    welded_contact   bool    [N]         recorded on the WELDED scene

Two replay modes (``--mode``):
    with-ball    each row's own recorded serve (serve_pos/serve_vel) --
                 measures standing AND return quality together.
    ball-parked  a fixed serve that never reaches the robot
                 (PARKED_SERVE) -- isolates standing from return quality,
                 matching the design doc's "ball-free Isaac task" (no ball
                 body is simulated there at all).

Two action sources:
    a=0 (default, no --wbc-npz)   open loop: joint target =
                                  concat(leg_default, u[t]) every step --
                                  today's free-base hitter, exactly (the
                                  parity anchor docs/design/whole_body_v1.md
                                  sec 3.2 names: a WBC with an all-zero final
                                  actor layer reproduces this bit for bit).
    --wbc-npz PATH                closed loop: a trained WBC checkpoint in
                                  the exact .npz schema himrobot.wbc.
                                  export_npz writes and himrobot.policies.
                                  wbc_np.NumpyWBCPolicy loads (4-layer ELU
                                  MLP + rms_mean/var + leg_default +
                                  jnt_range) -- so a checkpoint destined for
                                  evaluate.py can be replayed against a
                                  library here first (the per-checkpoint
                                  sim-to-sim check this module exists for).
                                  The row's u(t) still drives waist+arms as
                                  the "hitter command" the WBC observes and
                                  partially corrects, via
                                  himrobot.wbc.obs.flatten /
                                  himrobot.wbc.action.joint_target_wbc --
                                  shared with the live deploy policy, not
                                  reimplemented here (only the LIVE plan()/
                                  preview() inputs those functions normally
                                  take are missing offline; this module
                                  builds the equivalent ``plan`` dict and
                                  padded 29-dim ``u`` vectors from the
                                  library's own stored per-row fields
                                  instead -- see ``_pad29``/``_fake_plan``).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from himrobot.paths import ASSETS_DIR
from himrobot.rl.balance_env import body_frame_gravity, tilt_angle
from himrobot.rl.env import _joint_range
from himrobot.sim import scene
from himrobot.wbc import obs as wbc_obs
from himrobot.wbc.action import joint_target_wbc

# himrobot.wbc.export_npz (which owns numpy_forward, the shared MLP forward)
# imports torch lazily itself; it is still imported inside WBCCheckpoint only
# so the a=0 open-loop path -- the MuJoCo half of Gate 0 -- stays torch-free
# like the rest of the deploy runtime (himrobot.policies.wbc_np).

MOCAP_LEG_PRIOR_NPZ = str(ASSETS_DIR / "wbc" / "mocap_leg_prior.npz")

# ---------------------------------------------------------------------------
# Library schema (see module docstring) and modes.
# ---------------------------------------------------------------------------
LIBRARY_KEYS = ("u", "swing_start_step", "contact_step", "t_contact", "t_pre", "t_follow",
                "serve_pos", "serve_vel", "welded_legal", "welded_contact")
N_CMD = wbc_obs.N_CMD  # 17 = len(scene.JOINT_ORDER[12:29]) -- waist(3) + left arm(7) + right arm(7)

MODES = ("with-ball", "ball-parked")
# Fixed parked serve: far enough opponent-side, at rest, that it never
# reaches the robot's forehand zone within an episode -- matches the design
# doc's "the ball never reaches the robot" (sec "SPEC" for this item).
PARKED_SERVE = {"pos": np.array([1.2, 0.0, 0.9]), "vel": np.array([0.0, 0.0, 0.0]), "spin": None}

LEG_DEFAULT_CHOICES = ("stand", "mocap")

CONTACT_STEP_ERROR_WINDOW = 3  # steps either side of a row's contact_step averaged for the pelvis-xy metric


def leg_default_pose(name: str) -> np.ndarray:
    """"stand" (parity, STAND_POSE[:12]) or "mocap" (tools/mocap/
    mocap_leg_prior.py's global_mean, the play-clip mean lower body --
    docs/design/whole_body_v1.md sec 3.2's two named run flags)."""
    if name == "stand":
        return scene.STAND_POSE[:12].copy()
    if name == "mocap":
        if not os.path.isfile(MOCAP_LEG_PRIOR_NPZ):
            raise FileNotFoundError(
                f"leg_default='mocap' needs {MOCAP_LEG_PRIOR_NPZ}; "
                "run tools/mocap/mocap_leg_prior.py first")
        z = np.load(MOCAP_LEG_PRIOR_NPZ, allow_pickle=False)
        mean = z["global_mean"].astype(np.float64)
        assert mean.shape == (12,), f"{MOCAP_LEG_PRIOR_NPZ}: global_mean must be (12,), got {mean.shape}"
        return mean
    raise ValueError(f"unknown leg_default {name!r}; choices: {LEG_DEFAULT_CHOICES}")


def load_library(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    missing = [k for k in LIBRARY_KEYS if k not in z.files]
    if missing:
        raise ValueError(f"{path}: missing library keys {missing} -- see this module's "
                         "docstring for the expected schema")
    lib = {k: z[k] for k in LIBRARY_KEYS}
    n = lib["u"].shape[0]
    if lib["u"].ndim != 3 or lib["u"].shape[2] != N_CMD:
        raise ValueError(f"{path}: 'u' must be [N, T, {N_CMD}], got {lib['u'].shape}")
    for k in LIBRARY_KEYS[1:]:
        if lib[k].shape[0] != n:
            raise ValueError(f"{path}: '{k}' has {lib[k].shape[0]} rows, 'u' has {n}")
    return lib


def _row(lib: dict, i: int) -> dict:
    return {
        "u": np.asarray(lib["u"][i], dtype=np.float64),
        "swing_start_step": int(lib["swing_start_step"][i]),
        "contact_step": int(lib["contact_step"][i]),
        "t_contact": float(lib["t_contact"][i]),
        "t_pre": float(lib["t_pre"][i]),
        "t_follow": float(lib["t_follow"][i]),
        "serve_pos": np.asarray(lib["serve_pos"][i], dtype=np.float64),
        "serve_vel": np.asarray(lib["serve_vel"][i], dtype=np.float64),
        "welded_legal": bool(lib["welded_legal"][i]),
        "welded_contact": bool(lib["welded_contact"][i]),
    }


# ---------------------------------------------------------------------------
# Closed-loop WBC: reuses himrobot.wbc.obs / himrobot.wbc.action (numpy,
# already the shared formula both this module and himrobot.policies.wbc_np
# consume) plus himrobot.wbc.export_npz.numpy_forward (the exact rsl_rl-
# shaped ELU MLP forward). Only the LIVE inputs those normally take from
# IKHitterPolicy.plan()/preview() are missing offline; _pad29/_fake_plan
# rebuild the equivalent shapes from a library row's own stored fields.
# ---------------------------------------------------------------------------
WBC_OBS_DIM = wbc_obs.OBS_DIM
WBC_ACT_DIM = wbc_obs.ACT_DIM


def _pad29(u17: np.ndarray) -> np.ndarray:
    """u17 (JOINT_ORDER[12:29]) -> a 29-vector with zeros on the leg slots.
    Both wbc_obs.flatten and action.joint_target_wbc only ever read
    JOINT_ORDER[12:29] out of a 29-dim ``u``/``u_now`` argument (legs come
    from leg_default instead -- see their own docstrings), so the padding
    values on [0:12] are never read; this only exists to match their shape."""
    return np.concatenate([np.zeros(12, dtype=np.float64), np.asarray(u17, dtype=np.float64)])


def _fake_plan(t_contact: float, t_pre: float, t_follow: float) -> dict:
    """The subset of IKHitterPolicy's real ``self._plan`` that
    wbc_obs.swing_clock reads. ``speed_buffer=0.0`` because the library
    schema (this module's docstring, sec 4.1 of the design doc) does not
    carry it -- act()'s true window subtracts it too (see
    IKHitterPolicy.act), so this is a documented approximation of the live
    window, not the exact one."""
    return {"t_contact": t_contact, "speed_buffer": 0.0, "t_pre": t_pre, "t_follow": t_follow}


class WBCCheckpoint:
    """Torch-free runtime for a himrobot.wbc.export_npz-produced checkpoint
    -- the exact schema himrobot.policies.wbc_np.NumpyWBCPolicy loads (4-
    layer ELU MLP, rms_mean/var, leg_default, jnt_range; leg_scale/
    waist_scale/expert_npz/obs_version are also present but unused here --
    action.joint_target_wbc already hardcodes wbc_obs.LEG_SCALE/WAIST_SCALE
    and this module has no live expert to select). ``leg_default``/
    ``jnt_range`` come from the checkpoint itself, matching what
    NumpyWBCPolicy would actually run with -- NOT this module's own
    --leg-default flag, which only governs the a=0 (no checkpoint) path."""

    def __init__(self, path: str):
        from himrobot.wbc.export_npz import NORM_EPS, numpy_forward  # see top of module

        self._forward = numpy_forward
        self._norm_eps = NORM_EPS  # rsl_rl EmpiricalNormalization eps; overridden from the npz below
        z = np.load(path, allow_pickle=False)
        self.layers = [(z[f"W{i}"].astype(np.float64), z[f"b{i}"].astype(np.float64)) for i in range(4)]
        assert self.layers[0][0].shape[1] == WBC_OBS_DIM, \
            f"{path}: W0 expects {WBC_OBS_DIM}-dim input, got {self.layers[0][0].shape[1]}"
        assert self.layers[-1][0].shape[0] == WBC_ACT_DIM, \
            f"{path}: final layer must output {WBC_ACT_DIM}, got {self.layers[-1][0].shape[0]}"
        # Metadata guards (present in every export_npz output; optional here
        # so hand-built parity checkpoints still load): a checkpoint trained
        # against another observation layout or action scaling must not be
        # silently replayed with this one's formulas.
        if "obs_version" in z.files:
            assert int(z["obs_version"]) == wbc_obs.OBS_VERSION, \
                f"{path}: obs_version {int(z['obs_version'])} != {wbc_obs.OBS_VERSION}"
        for key, expected in (("leg_scale", wbc_obs.LEG_SCALE), ("waist_scale", wbc_obs.WAIST_SCALE)):
            if key in z.files:
                assert float(z[key]) == expected, f"{path}: {key} {float(z[key])} != {expected}"
        self.rms_mean = z["rms_mean"].astype(np.float64) if "rms_mean" in z.files else np.zeros(WBC_OBS_DIM)
        self.rms_var = z["rms_var"].astype(np.float64) if "rms_var" in z.files else np.ones(WBC_OBS_DIM)
        if "norm_eps" in z.files:
            self._norm_eps = float(z["norm_eps"])  # 0.0 for an identity-normalized export
        self.leg_default = z["leg_default"].astype(np.float64) if "leg_default" in z.files \
            else scene.STAND_POSE[:12].copy()
        self.jnt_range = z["jnt_range"].astype(np.float64) if "jnt_range" in z.files else None

    def act(self, obs141: np.ndarray) -> np.ndarray:
        """The RAW actor output, as NumpyWBCPolicy.act keeps it for the
        prev-action slot (the Isaac side observes the unclipped
        action_manager.action); action.joint_target_wbc clips to [-1, 1]
        itself before scaling."""
        return self._forward(self.layers, self.rms_mean, self.rms_var, obs141, self._norm_eps)


# ---------------------------------------------------------------------------
# Replay.
# ---------------------------------------------------------------------------
def replay_row(env: scene.TableTennisEnv, jnt_range: np.ndarray, leg_default: np.ndarray,
              row: dict, mode: str, wbc: WBCCheckpoint | None) -> dict:
    """``jnt_range``/``leg_default`` govern the a=0 open-loop path; a real
    ``wbc`` checkpoint carries its own (matching what NumpyWBCPolicy would
    actually use) and those override both for the closed-loop path -- see
    WBCCheckpoint's docstring."""
    serve = PARKED_SERVE if mode == "ball-parked" else {"pos": row["serve_pos"], "vel": row["serve_vel"], "spin": None}
    obs = env.reset(serve=serve)

    n_steps = row["u"].shape[0]
    active_leg_default = wbc.leg_default if wbc is not None else leg_default
    active_jnt_range = wbc.jnt_range if (wbc is not None and wbc.jnt_range is not None) else jnt_range
    prev_wbc_action = np.zeros(WBC_ACT_DIM, dtype=np.float64)
    fell_step: int | None = None
    tilt_peak = 0.0
    pelvis_xy_at_contact: float | None = None
    pelvis_xy_err: list[tuple[int, float]] = []
    aborted_at: int | None = None

    for t in range(n_steps):
        u_t = row["u"][t]
        if wbc is None:
            target = np.concatenate([leg_default, u_t])
        else:
            u29_now = _pad29(u_t)
            u29_p5 = _pad29(row["u"][min(t + 5, n_steps - 1)])
            u29_p10 = _pad29(row["u"][min(t + 10, n_steps - 1)])
            plan = _fake_plan(row["t_contact"], row["t_pre"], row["t_follow"])
            obs141 = wbc_obs.flatten(obs, prev_wbc_action, u29_now, u29_p5, u29_p10, plan, active_leg_default)
            a = wbc.act(obs141)
            target = joint_target_wbc(a, u29_now, active_jnt_range, active_leg_default)
            prev_wbc_action = a

        obs, events = env.step(target)

        if not np.all(np.isfinite(obs["base.position"])):
            fell_step = fell_step if fell_step is not None else t
            aborted_at = t
            break  # the physics diverged; nothing past this step is meaningful

        if events.get("fell") and fell_step is None:
            fell_step = t

        tilt_peak = max(tilt_peak, tilt_angle(body_frame_gravity(obs["base.orientation"])))
        pelvis_xy_err.append((t, float(np.linalg.norm(
            np.asarray(obs["base.position"][:2]) - scene.BASE_NOMINAL_POS[:2]))))

    contact_step = row["contact_step"]
    window = [e for t, e in pelvis_xy_err if abs(t - contact_step) <= CONTACT_STEP_ERROR_WINDOW]
    if window:
        pelvis_xy_at_contact = float(np.mean(window))

    return {
        "fell": fell_step is not None,
        "fell_step": fell_step,
        "fell_time_s": (fell_step / scene.CONTROL_HZ) if fell_step is not None else None,
        "aborted_at": aborted_at,
        "tilt_peak_rad": tilt_peak,
        "pelvis_xy_err_at_contact_m": pelvis_xy_at_contact,
        "welded_legal": row["welded_legal"],
        "welded_contact": row["welded_contact"],
    }


FALL_TIME_BIN_EDGES = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 3.0]


def _histogram(values: list[float], edges: list[float]) -> dict:
    counts, _ = np.histogram(values, bins=edges)
    return {"edges": edges, "counts": [int(c) for c in counts]}


def run(lib_path: str, mode: str, leg_default_name: str, wbc_npz: str | None,
       max_rows: int | None, seed: int) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    lib = load_library(lib_path)
    n_rows = lib["u"].shape[0] if max_rows is None else min(max_rows, lib["u"].shape[0])
    leg_default = leg_default_pose(leg_default_name)
    wbc = WBCCheckpoint(wbc_npz) if wbc_npz else None

    env = scene.TableTennisEnv(seed=seed, params={"robot_table_collision": False}, fixed_base=False)
    jnt_range = _joint_range(env.model)

    rows = [replay_row(env, jnt_range, leg_default, _row(lib, i), mode, wbc) for i in range(n_rows)]

    n_fell = sum(1 for r in rows if r["fell"])
    contact_errs = [r["pelvis_xy_err_at_contact_m"] for r in rows if r["pelvis_xy_err_at_contact_m"] is not None]
    tilts = [r["tilt_peak_rad"] for r in rows]
    fall_times_rel_contact = [
        r["fell_time_s"] - lib["t_contact"][i] for i, r in enumerate(rows) if r["fell_time_s"] is not None
    ]

    return {
        "lib": lib_path, "mode": mode, "leg_default": leg_default_name,
        "wbc_npz": wbc_npz, "n_rows": n_rows,
        "falls_pct": 100.0 * n_fell / n_rows if n_rows else float("nan"),
        "pelvis_xy_err_at_contact_p90_m": float(np.percentile(contact_errs, 90)) if contact_errs else None,
        "tilt_p90_rad": float(np.percentile(tilts, 90)) if tilts else None,
        "fall_time_rel_contact_histogram": _histogram(fall_times_rel_contact, FALL_TIME_BIN_EDGES),
        "per_row": rows,
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True, help="swing-library .npz path (see module docstring for the schema)")
    ap.add_argument("--mode", choices=MODES, default="with-ball")
    ap.add_argument("--leg-default", choices=LEG_DEFAULT_CHOICES, default="stand")
    ap.add_argument("--wbc-npz", default=None, help="closed-loop WBC checkpoint; default a=0 open loop")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default work/wbc/replay_<lib>_<mode>.json")
    args = ap.parse_args(argv)

    start = time.time()
    result = run(args.lib, args.mode, args.leg_default, args.wbc_npz, args.max_rows, args.seed)
    result["elapsed_s"] = time.time() - start
    result["tt_robot_y"] = os.environ.get("TT_ROBOT_Y")
    result["robot_y"] = float(scene.ROBOT_Y)  # the stance actually in effect (0.0 when the env var is unset)

    out_path = args.out
    if out_path is None:
        lib_stem = os.path.splitext(os.path.basename(args.lib))[0]
        out_path = os.path.join("work", "wbc", f"replay_{lib_stem}_{args.mode}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"{args.lib} [{args.mode}]: {result['n_rows']} rows, {result['elapsed_s']:.1f}s, "
          f"falls={result['falls_pct']:.1f}%, tilt_p90={result['tilt_p90_rad']}, "
          f"pelvis_xy_err_p90={result['pelvis_xy_err_at_contact_p90_m']}")
    print(f"wrote {out_path}")
    return result


if __name__ == "__main__":
    main()
