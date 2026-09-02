#!/usr/bin/env python3
"""Gate 0 (Isaac half), docs/design/whole_body_v1.md sec 7.2: play the
held-out ``swing_lib_eval.npz`` rows (WB1, 600 rows) plus a batch of idle
rows (u == STAND_POSE throughout) open-loop with ``a = 0`` in Isaac, and
report the same falls/pelvis/tilt schema as the MuJoCo half
(``himrobot/wbc/replay_library.py``, WB4, not yet landed) so
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

    env = env_cfg.ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    swing = env.command_manager.get_term("swing")
    # First num_envs rows play the eval library; the last n_idle envs are
    # pinned to the row's own final (post-swing/idle) frame throughout, the
    # env-cfg convention for an "idle" row (see SwingLibraryCommand._resample_command).
    idle_start = args.num_envs
    swing.time_idx[idle_start:] = swing.bank.combined.n_steps - 1

    zero_action = torch.zeros(cfg.scene.num_envs, actions.ACTION_DIM, device=env.device)
    pelvis_z_hist, tilt_hist, fell_any = [], [], torch.zeros(cfg.scene.num_envs, dtype=torch.bool)

    n_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
    for _ in range(n_steps):
        env.step(zero_action)
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

    lib_fell = fell_any[:args.num_envs]
    idle_fell = fell_any[args.num_envs:]
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
