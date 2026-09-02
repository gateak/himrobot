#!/usr/bin/env python3
"""Gate 1b (docs/design/whole_body_v1.md sec 7.2, "0:10-0:30"): the converted
USD must put the robot where the scored MuJoCo model puts it. Builds the real
``Himrobot-WBC-G1-v0`` env with ONE env and no randomisation, forces the
articulation to STAND_POSE at the nominal pelvis pose, and compares against
``assets/wbc/actuators_mujoco.json``'s ``fk_truth`` block
(himrobot.wbc.mujoco_mirror, pelvis-frame FK of the compiled scored model):

  1. the 29 USD joints cover scene.JOINT_ORDER one-for-one (by name);
  2. paddle blade centre within POS_TOL (5 mm) and blade normal dot > DOT_MIN
     (0.999) of the MuJoCo truth -- the blade is a geom in the MJCF, so its
     pose is rebuilt from the ``paddle`` body pose plus the geom's own
     pos/zaxis parsed out of the mirror XML;
  3. both feet within POS_TOL;
  4. 10 zero-action env steps: finite observations, no termination.

Writes ``--out`` (usd_parity.json) and exits 0 only when every check passes;
2 on a measured mismatch, 3 on an exception inside the Isaac-side code (an
API path this sandbox could never run -- a loud stop, never a silent pass:
docs/design/whole_body_v1.md sec 11 names this conversion the first real
risk of any box session).

    /workspace/venv/bin/python -m himrobot.wbc.isaac.scripts.verify_usd_parity \\
        --swing_lib work/wbc/swing_lib_eval.npz --out logs/rsl_rl/wbc_v1/usd_parity.json --headless
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
import xml.etree.ElementTree as ET

import numpy as np

POS_TOL_M = 0.005
DOT_MIN = 0.999
N_ZERO_ACTION_STEPS = 10


def blade_geom_in_paddle_body(xml_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(pos, zaxis) of the ``paddle_blade`` geom inside the ``paddle`` body,
    from the mirror MJCF itself. MuJoCo's geom_xmat[:, 2] (the blade normal
    mujoco_mirror stores) is the geom frame's z axis == ``zaxis`` expressed
    in the body frame."""
    root = ET.parse(xml_path).getroot()
    for body in root.iter("body"):
        if body.get("name") == "paddle":
            for geom in body.iter("geom"):
                if geom.get("name") == "paddle_blade":
                    pos = np.array([float(v) for v in geom.get("pos", "0 0 0").split()])
                    zaxis = np.array([float(v) for v in geom.get("zaxis", "0 0 1").split()])
                    return pos, zaxis / np.linalg.norm(zaxis)
    raise RuntimeError(f"{xml_path}: no paddle body with a paddle_blade geom")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swing_lib", required=True, help="any library npz (the command term needs one to load)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args(argv)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    report: dict = {"checks": {}, "status": "error"}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: dict) -> None:
        report["checks"][name] = {"ok": bool(ok), **detail}
        if not ok:
            failures.append(name)

    try:
        import torch
        import isaaclab.utils.math as math_utils

        from himrobot.sim import scene
        from himrobot.wbc import mujoco_mirror
        from himrobot.wbc.isaac import env_cfg
        from himrobot.wbc.isaac.mdp import actions

        truth = json.load(open(str(mujoco_mirror.JSON_PATH)))["fk_truth"]
        blade_pos_b, blade_axis_b = blade_geom_in_paddle_body(str(mujoco_mirror.XML_PATH))

        cfg = env_cfg.HimrobotWBCG1EnvCfg()
        cfg.scene.num_envs = 1
        cfg.commands.swing.library_paths = {"eval": args.swing_lib}
        env_cfg.disable_randomization(cfg)
        env = env_cfg.ManagerBasedRLEnv(cfg=cfg)
        env.reset()
        robot = env.scene["robot"]
        swing = env.command_manager.get_term("swing")
        report["usd_joint_names"] = list(robot.joint_names)
        report["usd_body_names"] = list(robot.body_names)
        check("joint_names_cover_JOINT_ORDER", len(swing.joint_ids) == scene.NUM_JOINTS,
              {"n_usd_joints": len(robot.joint_names)})

        # Force STAND_POSE at the nominal pelvis pose (fk_truth's frame: pelvis
        # at BASE_NOMINAL_POS, identity orientation), zero velocity, one
        # physics step so PhysX publishes the link transforms (sub-mm drift at
        # the default PD targets, against a 5 mm tolerance).
        device = env.device
        origin = env.scene.env_origins[0]
        root = torch.zeros(1, 7, device=device)
        root[0, 0:3] = origin + torch.as_tensor(scene.BASE_NOMINAL_POS, dtype=torch.float32, device=device)
        root[0, 3:7] = torch.as_tensor(scene.BASE_NOMINAL_QUAT, dtype=torch.float32, device=device)
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        jp = robot.data.joint_pos.clone()
        jp[0, swing.joint_ids] = torch.as_tensor(scene.STAND_POSE, dtype=torch.float32, device=device)
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        robot.set_joint_position_target(jp)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.sim.get_physics_dt())

        nominal = root[0, 0:3]
        paddle_ids, _ = robot.find_bodies("paddle")
        check("paddle_body_present", len(paddle_ids) == 1, {"found": len(paddle_ids)})
        if paddle_ids:
            pid = paddle_ids[0]
            p_body = robot.data.body_pos_w[0, pid, :]
            q_body = robot.data.body_quat_w[0:1, pid, :]
            q_body = q_body / q_body.norm(dim=1, keepdim=True)
            off = torch.as_tensor(blade_pos_b, dtype=torch.float32, device=device).unsqueeze(0)
            axis = torch.as_tensor(blade_axis_b, dtype=torch.float32, device=device).unsqueeze(0)
            blade_pos = (p_body + math_utils.quat_apply(q_body, off)[0] - nominal).cpu().numpy()
            blade_normal = math_utils.quat_apply(q_body, axis)[0].cpu().numpy()
            body_rel = (p_body - nominal).cpu().numpy()
            t_pos = np.asarray(truth["paddle_blade_pos"])
            t_n = np.asarray(truth["paddle_blade_normal"])
            t_n = t_n / np.linalg.norm(t_n)
            pos_err = float(np.linalg.norm(blade_pos - t_pos))
            dot = float(abs(np.dot(blade_normal, t_n)))  # a cylinder's normal is sign-free
            check("paddle_blade_pos", pos_err <= POS_TOL_M,
                  {"err_m": pos_err, "tol_m": POS_TOL_M, "isaac": blade_pos.tolist(), "truth": t_pos.tolist()})
            check("paddle_blade_normal", dot >= DOT_MIN,
                  {"abs_dot": dot, "min": DOT_MIN, "isaac": blade_normal.tolist(), "truth": t_n.tolist()})
            check("paddle_body_pos", float(np.linalg.norm(body_rel - np.asarray(truth["paddle_body_pos"]))) <= POS_TOL_M,
                  {"isaac": body_rel.tolist(), "truth": truth["paddle_body_pos"]})

        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            ids, _ = robot.find_bodies(name)
            if len(ids) != 1:
                check(f"{name}_present", False, {"found": len(ids)})
                continue
            rel = (robot.data.body_pos_w[0, ids[0], :] - nominal).cpu().numpy()
            t = np.asarray(truth[f"{name}_pos"])
            err = float(np.linalg.norm(rel - t))
            check(f"{name}_pos", err <= POS_TOL_M, {"err_m": err, "isaac": rel.tolist(), "truth": t.tolist()})

        # 10 zero-action steps: finite observations, no termination.
        zero = torch.zeros(1, actions.ACTION_DIM, device=device)
        finite, terminated_any = True, False
        for _ in range(N_ZERO_ACTION_STEPS):
            obs, _, terminated, _, _ = env.step(zero)
            finite &= bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(obs["critic"]).all())
            terminated_any |= bool(terminated.any())
        check("zero_action_steps", finite and not terminated_any,
              {"finite": finite, "terminated": terminated_any, "steps": N_ZERO_ACTION_STEPS,
               "policy_obs_dim": int(obs["policy"].shape[-1]), "critic_obs_dim": int(obs["critic"].shape[-1])})
        check("obs_dims", int(obs["policy"].shape[-1]) == 141 and int(obs["critic"].shape[-1]) == 176, {})

        report["status"] = "fail" if failures else "pass"
        rc = 2 if failures else 0
    except Exception:  # noqa: BLE001 -- report and stop, never a silent pass
        report["status"] = "error"
        report["traceback"] = traceback.format_exc()
        rc = 3
    finally:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(json.dumps({k: v for k, v in report.items() if k != "usd_body_names"}, indent=2))
        print(f"USD parity: {report['status'].upper()} -> {args.out}")
        simulation_app.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
