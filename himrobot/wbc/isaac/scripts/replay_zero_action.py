#!/usr/bin/env python3
"""Gate 0 (Isaac half), docs/design/whole_body_v1.md sec 7.2: play the
held-out ``swing_lib_eval.npz`` rows (WB1, 600 rows) plus a batch of idle
rows (u == STAND_POSE throughout) open-loop with ``a = 0`` in Isaac, and
report the same falls/pelvis/tilt schema as the MuJoCo half
(``himrobot/wbc/replay_library.py --mode ball-parked``: no ball reaches the
robot there either, so the two halves simulate the same thing) so
``tools/box/wbc_box_run.sh`` can diff the two. a = 0 with
``leg_default = STAND_POSE[:12]`` is the parity anchor
(docs/design/whole_body_v1.md sec 3.2): STAND_POSE rows must stand the full
episode with 0 falls, and the library rows' fall rate must land within the
runbook's +-15-point tolerance of the MuJoCo number.

Output JSON schema (read by tools/box/wbc_box_run.sh's Gate 0 comparison):
    {"n_rows": int, "steps_per_episode": int,
     "falls": {"count": int, "rate": float},
     "pelvis_z_p90": float, "pelvis_z_min": float,
     "tilt_p90_rad": float, "tilt_max_rad": float,
     "idle": {"n_rows": int, "falls": {"count": int, "rate": float}}}

    /workspace/venv/bin/python -m himrobot.wbc.isaac.scripts.replay_zero_action \\
        --swing_lib work/wbc/swing_lib_eval.npz --num_envs 600 --headless \\
        --out work/wbc/gate0_isaac.json
"""
from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swing_lib", required=True, help="the held-out swing_lib_eval.npz (WB1, 600 rows)")
    ap.add_argument("--num_envs", type=int, default=600)
    ap.add_argument("--n_idle", type=int, default=32, help="idle (STAND_POSE-throughout) rows, run alongside")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import torch

    from himrobot.wbc.isaac import env_cfg
    from himrobot.wbc.isaac.mdp import actions, terminations

    cfg = env_cfg.HimrobotWBCG1EnvCfg()
    cfg.scene.num_envs = args.num_envs + args.n_idle
    cfg.commands.swing.library_paths = {"eval": args.swing_lib}
    # Exact nominal resets, no pushes/material/mass/gain jitter: the MuJoCo
    # half (himrobot.wbc.replay_library) resets to the exact straight-leg pose.
    env_cfg.disable_randomization(cfg)
    # MuJoCo's replay has no tilt cut-off -- it counts a fall only once the
    # pelvis drops below scene.FALL_PELVIS_Z. With the 0.7 rad "tilted"
    # termination in place Isaac would reset a toppling robot before that
    # happens and undercount; drop it for the replay so both halves count
    # the same physical event.
    cfg.terminations.tilted = None

    env = env_cfg.ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    swing = env.command_manager.get_term("swing")
    n_lib_rows = swing.bank.n_rows
    if args.num_envs > n_lib_rows:
        raise SystemExit(f"--num_envs {args.num_envs} > {n_lib_rows} rows in {args.swing_lib}")
    # env.reset() drew random rows per the curriculum; Gate 0 wants row i in
    # env i, from step 0, and the trailing n_idle envs flagged idle (STAND_POSE
    # on every step -- SwingLibraryCommand's idle mask).
    idle_start = args.num_envs
    swing.row_idx[:idle_start] = torch.arange(idle_start, device=env.device)
    swing.row_idx[idle_start:] = 0
    swing.time_idx[:] = 0
    swing.idle[:] = False
    swing.idle[idle_start:] = True

    zero_action = torch.zeros(cfg.scene.num_envs, actions.ACTION_DIM, device=env.device)
    pelvis_z_hist, tilt_hist = [], []
    fell_any = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=env.device)

    n_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    for _ in range(n_steps):
        env.step(zero_action)
        # A fallen env is auto-reset INSIDE step() (back to the nominal pose,
        # a fresh random row), so the fall must be read off the termination
        # manager's per-term flag, not the post-step pelvis height -- which
        # is already the reset value. Same threshold as MuJoCo's own
        # ``fell`` latch (scene.FALL_PELVIS_Z), so the two halves count the
        # same event.
        fell_any |= env.termination_manager.get_term("fell")
        robot = env.scene["robot"]
        z = robot.data.root_pos_w[:, 2]
        gravity_b = env_cfg.observations.body_frame_gravity(robot.data.root_quat_w)
        tilt = torch.arccos(torch.clamp(-gravity_b[:, 2], -1.0, 1.0))
        pelvis_z_hist.append(z.clone())
        tilt_hist.append(tilt.clone())
        fell_any |= terminations.fell(z)

    pelvis_z = torch.stack(pelvis_z_hist)  # (T, N)
    tilt = torch.stack(tilt_hist)

    def _p90(x: torch.Tensor) -> float:
        return float(torch.quantile(x.flatten().float(), 0.90).item())

    lib_fell = fell_any[:args.num_envs].cpu()
    idle_fell = fell_any[args.num_envs:].cpu()
    result = {
        "n_rows": args.num_envs,
        "steps_per_episode": n_steps,
        "falls": {"count": int(lib_fell.sum().item()), "rate": float(lib_fell.float().mean().item())},
        "pelvis_z_p90": _p90(pelvis_z),
        "pelvis_z_min": float(pelvis_z.min().item()),
        "tilt_p90_rad": _p90(tilt),
        "tilt_max_rad": float(tilt.max().item()),
        "idle": {
            "n_rows": args.n_idle,
            "falls": {"count": int(idle_fell.sum().item()), "rate": float(idle_fell.float().mean().item())},
        },
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(f"wrote {args.out}: lib falls {result['falls']['rate']:.1%}, idle falls {result['idle']['falls']['rate']:.1%}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
