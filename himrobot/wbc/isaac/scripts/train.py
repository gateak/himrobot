#!/usr/bin/env python3
"""Train ``Himrobot-WBC-G1-v0`` (docs/design/whole_body_v1.md sec 7.2-7.3).
Isaac Sim/Isaac Lab only -- run with the box's own interpreter
(``/workspace/venv/bin/python``), never bare ``python``. Never run in this
session (no GPU here); see tools/box/wbc_box_run.sh for the full box
sequence this script is one step of.

    /workspace/venv/bin/python -m himrobot.wbc.isaac.scripts.train \\
        --task Himrobot-WBC-G1-v0 --num_envs 4096 --max_iterations 200 \\
        --headless --logger tensorboard --run_name smoke \\
        --swing_lib work/wbc/nominal.npz --leg_default stand
"""
from __future__ import annotations

import argparse
import os
import time


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Himrobot-WBC-G1-v0")
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--max_iterations", type=int, default=None,
                    help="default: himrobot.wbc.isaac.ppo_cfg.MAX_ITERATIONS")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--logger", choices=["tensorboard", "wandb"], default="tensorboard",
                    help="tensorboard -- no WandB account needed on the box")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_dir", default="logs/rsl_rl/wbc_v1")
    ap.add_argument("--swing_lib", action="append", default=[], metavar="LABEL=PATH",
                    help="e.g. nominal=/workspace/motions/wbc/nominal.npz; repeatable")
    ap.add_argument("--leg_default", choices=["stand", "mocap"], default="stand",
                    help="stand = STAND_POSE[:12], the parity anchor; mocap = the mocap-prior mean (WB5)")
    ap.add_argument("--curriculum", choices=["default", "off"], default="default")
    ap.add_argument("--max_minutes", type=float, default=None, help="wall-clock cap; SIGINT-safe (see box_run.sh)")
    ap.add_argument("--resume", default=None, help="a previous run's checkpoint (model_<it>.pt)")
    ap.add_argument("--sim_dt", type=float, default=None, help="override sim.dt (paired with --decimation)")
    ap.add_argument("--decimation", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # isaaclab's own convention: launch the simulation app BEFORE importing
    # anything that touches isaacsim/omni, so headless mode is honoured.
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import torch
    from rsl_rl.runners import OnPolicyRunner

    from himrobot.wbc.isaac import env_cfg, ppo_cfg
    from himrobot.wbc.isaac.mdp import actions

    cfg = env_cfg.HimrobotWBCG1EnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.sim_dt is not None and args.decimation is not None:
        env_cfg.apply_sim_dt_override(cfg, args.sim_dt, args.decimation)

    library_paths = dict(env_cfg.DEFAULT_LIBRARY_PATHS)
    for item in args.swing_lib:
        label, _, path = item.partition("=")
        library_paths[label] = path
    cfg.commands.swing.library_paths = library_paths
    cfg.commands.swing.leg_default = (
        actions.STAND_LEG_DEFAULT if args.leg_default == "stand"
        else tuple(_load_mocap_leg_default())
    )

    runner_cfg = ppo_cfg.HimrobotWBCPPORunnerCfg()
    if args.max_iterations is not None:
        runner_cfg.max_iterations = args.max_iterations
    runner_cfg.experiment_name = args.run_name

    env = env_cfg.ManagerBasedRLEnv(cfg=cfg)
    log_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, vars(runner_cfg), log_dir=log_dir, device="cuda:0")

    if runner_cfg.zero_init_final_actor_layer:
        zero_init_final_actor_layer(runner)
    if args.resume:
        runner.load(args.resume)

    start = time.time()
    deadline = start + args.max_minutes * 60.0 if args.max_minutes else None
    try:
        if deadline is None:
            runner.learn(num_learning_iterations=runner_cfg.max_iterations)
        else:
            # rsl_rl's own learn() has no wall-clock cap; tools/box/
            # wbc_box_run.sh instead SIGINTs this process near the deadline
            # (see its header) and this loop just makes the cap explicit
            # for anyone running the script directly.
            it = 0
            while time.time() < deadline and it < runner_cfg.max_iterations:
                runner.learn(num_learning_iterations=1)
                it += 1
    except KeyboardInterrupt:
        pass
    finally:
        runner.save(os.path.join(log_dir, "final.pt"))
        simulation_app.close()
    return 0


def zero_init_final_actor_layer(runner) -> None:  # noqa: ANN001 -- runner is an rsl_rl OnPolicyRunner
    """docs/design/whole_body_v1.md sec 3.2's parity anchor: iteration 0's
    mean action is exactly a = 0, so a freshly-constructed policy reproduces
    today's free-base hitter's own 61.5%/75.7%-falls row bit-for-bit. Zeroes
    the last Linear layer's weight and bias in the actor MLP (the standard
    "zero-init the policy head" trick); the critic is left alone."""
    import torch.nn as nn
    actor = runner.alg.actor_critic.actor
    last_linear = next(m for m in reversed(list(actor.modules())) if isinstance(m, nn.Linear))
    nn.init.zeros_(last_linear.weight)
    nn.init.zeros_(last_linear.bias)


def _load_mocap_leg_default() -> "list[float]":
    import numpy as np

    from himrobot.paths import ASSETS_DIR
    z = np.load(str(ASSETS_DIR / "wbc" / "mocap_leg_prior.npz"), allow_pickle=False)
    return [float(v) for v in z["global_mean"]]  # WB5's assets/wbc/mocap_leg_prior.npz, not yet landed


if __name__ == "__main__":
    raise SystemExit(main())
