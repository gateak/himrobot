"""Retarget a real human table-tennis swing (Rokoko mocap) onto the Unitree G1.

Position-based, robust (no fragile euler-axis matching): read the human's
per-frame shoulder/elbow/wrist world positions (from fbx_extract.py), express
the wrist relative to the shoulder in a torso frame, scale to the robot's arm
length, and solve the G1's 7-DOF right arm so its wrist reaches that point,
with a null-space bias keeping the rest of the arm near the natural stand
pose — the robot "reaches where the human reached" (HITTER-style use of
human motion as reference).

    ./.venv/bin/python tools/mocap/g1_retarget.py    # -> tools/mocap/g1_retarget.gif
"""
from __future__ import annotations

import argparse
import math
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
    """Orthonormal [right, up, forward] frame from the human skeleton."""
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
    ap.add_argument("--out", default=os.path.join(g1_model.REPO, "tools", "mocap", "g1_retarget.gif"))
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--damping", type=float, default=0.1)
    ap.add_argument("--base-yaw", type=float, default=35.0,
                    help="rotate the robot body (deg about vertical) so the swing faces the board")
    ap.add_argument("--no-table", action="store_true", help="hide the fake table board")
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

    model, data = g1_model.load(paddle=True, table=not args.no_table)
    # Angle the body about vertical so the swing plane faces the board.
    yaw = math.radians(args.base_yaw)
    data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    cz, sz = math.cos(yaw), math.sin(yaw)
    R_base = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
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

    cam = mujoco.MjvCamera(); cam.azimuth = 150; cam.elevation = -10; cam.distance = 3.0
    cam.lookat[:] = [0.7, 0.0, 0.85]
    renderer = mujoco.Renderer(model, height=420, width=560)
    jacp = np.zeros((3, model.nv)); jacp2 = np.zeros((3, model.nv))
    W_EL = 0.6   # elbow weight (posture); the wrist stays the primary target

    def to_robot(human_pt, k):
        """Human landmark -> robot-frame target, scaled and body-angled."""
        return S_r + R_base @ (scale * human_to_robot(R_h_all[k].T @ (human_pt - sh_h[k])))

    R_h_all = [torso_frame(hips[k], spine[k], sh_h[k]) for k in range(len(wr_h))]
    wt_all, et_all = [], []
    for k in range(len(wr_h)):
        wt = to_robot(wr_h[k], k)                       # wrist target
        rr = np.linalg.norm(wt - S_r)                   # keep the wrist inside reach
        if rr > 0.97 * L_r:
            wt = S_r + (wt - S_r) * (0.97 * L_r / rr)
        wt_all.append(wt)
        et_all.append(to_robot(el_h[k], k))             # elbow target (fixes arm posture)
    wt_all, et_all = np.array(wt_all), np.array(et_all)

    # Drop the fake board to just under the swing's lowest point so the paddle
    # sweeps above it (kinematic preview has no collision -> would else clip through).
    if not args.no_table:
        delta = (float(wt_all[:, 2].min()) - 0.12) - 0.76
        for gname in ("tt_table", "tt_centerline", "tt_net"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                model.geom_pos[gid][2] += delta

    frames, errs = [], []
    for k in range(len(wr_h)):
        wt, et = wt_all[k], et_all[k]
        for _ in range(args.iters):
            mujoco.mj_forward(model, data)
            mujoco.mj_jacBody(model, data, jacp, None, bid["wr"])
            mujoco.mj_jacBody(model, data, jacp2, None, bid["el"])
            ew = wt - data.xpos[bid["wr"]]
            ee = et - data.xpos[bid["el"]]
            J = np.vstack([jacp[:, dadr], W_EL * jacp2[:, dadr]])       # 6x7 (wrist + elbow)
            err = np.concatenate([ew, W_EL * ee])
            Jpinv = J.T @ np.linalg.inv(J @ J.T + (args.damping ** 2) * np.eye(6))
            qcur = data.qpos[qadr]
            dq = Jpinv @ err + (np.eye(7) - Jpinv @ J) @ (0.2 * (q_home - qcur))
            nrm = np.linalg.norm(dq)
            if nrm > 0.25:
                dq *= 0.25 / nrm
            data.qpos[qadr] = np.clip(qcur + dq, jrng[:, 0], jrng[:, 1])
        errs.append(np.linalg.norm(ew) * 100)           # wrist error, cm
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps)
    errs = np.array(errs)
    print(f"[g1_retarget] wrote {args.out} ({len(frames)} frames @ {args.fps}fps)")
    print(f"[g1_retarget] wrist position error: mean {errs.mean():.1f} cm, "
          f"median {np.median(errs):.1f} cm, max {errs.max():.1f} cm")


if __name__ == "__main__":
    raise SystemExit(main())
