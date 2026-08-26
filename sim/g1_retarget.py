"""Retarget a real human table-tennis swing (Rokoko mocap) onto the Unitree G1.

Method (position-based, robust — no fragile euler-axis matching): read the
human's per-frame shoulder/elbow/wrist world positions (from fbx_extract.py),
express the wrist relative to the shoulder in a torso frame, scale to the
robot's arm length, and solve the G1's 7-DOF right arm so its wrist reaches
that point. A null-space bias keeps the rest of the arm near the natural stand
pose. This makes the robot "reach where the human reached" — the HITTER-style
use of human motion as a reference.

    ./.venv/bin/python sim/g1_retarget.py            # -> sim/g1_retarget.gif
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_model  # noqa: E402

import mujoco  # noqa: E402


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def torso_frame(hips, spine_top, shoulder):
    """Right-handed [right, up, forward] frame from the human skeleton."""
    up = unit(spine_top - hips)
    tmp = shoulder - spine_top          # roughly toward the right shoulder
    right = unit(tmp - np.dot(tmp, up) * up)
    fwd = unit(np.cross(up, right))
    return np.stack([right, up, fwd], axis=1)   # columns = axes


def human_to_robot(v):
    """[right, up, forward] (human torso) -> [x_fwd, y_left, z_up] (robot base)."""
    return np.array([v[2], -v[0], v[1]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=os.path.join(g1_model.REPO, "data", "mocap_clip4.npz"))
    ap.add_argument("--out", default=os.path.join(g1_model.REPO, "sim", "g1_retarget.gif"))
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--damping", type=float, default=0.1)
    args = ap.parse_args()

    import imageio

    d = np.load(args.npz)
    lm = list(d["landmarks"])
    pos = d["pos"].astype(float)                       # (F, L, 3)
    idx = {n: lm.index(n) for n in lm}
    hips = pos[:, idx["Hips.head"]]
    spine = pos[:, idx["Spine4.head"]]
    sh_h = pos[:, idx["RightArm.head"]]                # human shoulder
    el_h = pos[:, idx["RightForeArm.head"]]            # human elbow
    wr_h = pos[:, idx["RightHand.head"]]               # human wrist
    n_all = len(pos)
    stride = max(1, n_all // args.frames)
    sl = slice(None, None, stride)
    hips, spine, sh_h, el_h, wr_h = hips[sl], spine[sl], sh_h[sl], el_h[sl], wr_h[sl]

    # Human arm length (upper + fore), averaged over the clip.
    L_h = float(np.mean(np.linalg.norm(el_h - sh_h, axis=1) + np.linalg.norm(wr_h - el_h, axis=1)))

    model, data = g1_model.load(paddle=True)
    bid = {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, v) for k, v in
           dict(sh=g1_model.SHOULDER_BODY, el=g1_model.ELBOW_BODY, wr=g1_model.HAND_BODY).items()}
    qadr = np.array([int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                     for n in g1_model.RIGHT_ARM_JOINTS])
    dadr = np.array([int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                     for n in g1_model.RIGHT_ARM_JOINTS])
    jrng = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in g1_model.RIGHT_ARM_JOINTS])
    q_home = data.qpos[qadr].copy()

    S_r = data.xpos[bid["sh"]].copy()
    L_r = (np.linalg.norm(data.xpos[bid["el"]] - data.xpos[bid["sh"]]) +
           np.linalg.norm(data.xpos[bid["wr"]] - data.xpos[bid["el"]]))
    scale = L_r / L_h
    print(f"human arm {L_h:.3f} m, robot arm {L_r:.3f} m, scale {scale:.3f}")

    cam = mujoco.MjvCamera(); cam.azimuth = 140; cam.elevation = -8; cam.distance = 2.3
    cam.lookat[:] = [0.15, -0.15, 0.95]
    renderer = mujoco.Renderer(model, height=420, width=560)
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))

    frames, errs = [], []
    for k in range(len(wr_h)):
        R_h = torso_frame(hips[k], spine[k], sh_h[k])
        v = R_h.T @ (wr_h[k] - sh_h[k])                # wrist rel shoulder, torso coords
        target = S_r + scale * human_to_robot(v)
        reach = np.linalg.norm(target - S_r)           # keep inside the robot's reach
        if reach > 0.97 * L_r:
            target = S_r + (target - S_r) * (0.97 * L_r / reach)
        for _ in range(args.iters):
            mujoco.mj_forward(model, data)
            err = target - data.xpos[bid["wr"]]
            mujoco.mj_jacBody(model, data, jacp, jacr, bid["wr"])
            J = jacp[:, dadr]                                          # 3x7
            Jpinv = J.T @ np.linalg.inv(J @ J.T + (args.damping ** 2) * np.eye(3))
            qcur = data.qpos[qadr]
            dq = Jpinv @ err + (np.eye(7) - Jpinv @ J) @ (0.3 * (q_home - qcur))
            nrm = np.linalg.norm(dq)
            if nrm > 0.25:
                dq *= 0.25 / nrm
            data.qpos[qadr] = np.clip(qcur + dq, jrng[:, 0], jrng[:, 1])
        errs.append(np.linalg.norm(err) * 100)         # cm
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps)
    errs = np.array(errs)
    print(f"[g1_retarget] wrote {args.out} ({len(frames)} frames @ {args.fps}fps)")
    print(f"[g1_retarget] wrist position error: mean {errs.mean():.1f} cm, "
          f"median {np.median(errs):.1f} cm, max {errs.max():.1f} cm")


if __name__ == "__main__":
    raise SystemExit(main())
