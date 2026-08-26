"""First movement-retargeting pass: make the K1's right arm reproduce a REAL
recorded paddle motion, via damped-least-squares inverse kinematics.

Pipeline: read one human table-tennis episode's paddle IMU orientation ->
build a target orientation trajectory -> solve the 4 right-arm joints each
frame so the robot's paddle tracks it -> render the K1 swinging to a GIF.

HONESTY / scope:
  * Uses the paddle IMU euler angles — the only directly-readable real stream
    (the body mocap is FBX, which needs Blender/FBX-SDK we don't have yet).
  * Tracks the *relative* orientation change (rebased to the first frame),
    because the IMU's absolute frame is not calibrated to any world/robot
    frame. So this reproduces "how the paddle turned during the swing," not an
    absolute pose. It's the real mechanism on real data — a scaffold for the
    movement track, not a calibrated, deployable retarget.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k1_model  # noqa: E402  (local module)

import mujoco  # noqa: E402

# Ready/"on guard" pose for the right arm (rad), within URDF limits.
READY = {
    "ARight_Shoulder_Pitch": -0.6,
    "Right_Shoulder_Roll": -0.2,
    "Right_Elbow_Pitch": 1.0,
    "Right_Elbow_Yaw": 0.8,
}


def euler_to_R(rx, ry, rz):
    """ZYX composition R = Rz @ Ry @ Rx (radians). Documented convention."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_episode(csv_path, episode=None):
    """Return (episode_id, Nx3 euler radians) for the chosen or largest episode."""
    rows = {}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["episode_id"], []).append(r)
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")
    ep = episode or max(rows, key=lambda k: len(rows[k]))
    e = np.array(
        [[float(r["euler_x_deg"]), float(r["euler_y_deg"]), float(r["euler_z_deg"])]
         for r in rows[ep]],
        dtype=float,
    )
    return ep, np.deg2rad(e), len(rows[ep])


def rotvec(R):
    """World-frame orientation error as an axis*angle vector."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(9))
    v = q[1:]
    n = np.linalg.norm(v)
    return v / n * (2 * math.atan2(n, q[0])) if n > 1e-9 else np.zeros(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=os.path.join(k1_model.REPO, "data", "paddle-imu-samples.csv"))
    ap.add_argument("--episode", default=None)
    ap.add_argument("--out", default=os.path.join(k1_model.REPO, "sim", "k1_retarget.gif"))
    ap.add_argument("--frames", type=int, default=150, help="max rendered frames (episode is strided to fit)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--damping", type=float, default=0.15)
    args = ap.parse_args()

    import imageio

    ep, eul, n_all = load_episode(args.csv, args.episode)
    stride = max(1, len(eul) // args.frames)
    eul = eul[::stride]
    print(f"episode {ep}: {n_all} samples -> {len(eul)} frames (stride {stride})")

    model, data = k1_model.load(paddle=True)
    pbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle")
    qadr = {n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in k1_model.RIGHT_ARM_JOINTS}
    dadr = np.array([int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                     for n in k1_model.RIGHT_ARM_JOINTS])
    jrange = {n: model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
              for n in k1_model.RIGHT_ARM_JOINTS}

    # Ready pose -> record the paddle's home orientation.
    mujoco.mj_resetData(model, data)
    for n, v in READY.items():
        data.qpos[qadr[n]] = v
    mujoco.mj_forward(model, data)
    R_home = data.xmat[pbid].reshape(3, 3).copy()
    R0 = euler_to_R(*eul[0])  # first IMU frame = reference

    cam = mujoco.MjvCamera(); cam.azimuth = 118; cam.elevation = -8; cam.distance = 1.7
    cam.lookat[:] = [0.0, -0.15, 1.0]
    renderer = mujoco.Renderer(model, height=384, width=512)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    frames, errs = [], []
    for k in range(len(eul)):
        R_rel = R0.T @ euler_to_R(*eul[k])           # rotation in the initial paddle frame
        R_target = R_home @ R_rel                    # applied to the robot paddle's home pose
        for _ in range(args.iters):
            mujoco.mj_forward(model, data)
            err = rotvec(R_target @ data.xmat[pbid].reshape(3, 3).T)
            mujoco.mj_jacBody(model, data, jacp, jacr, pbid)
            J = jacr[:, dadr]                        # 3x4
            dq = J.T @ np.linalg.solve(J @ J.T + (args.damping ** 2) * np.eye(3), err)
            for i, n in enumerate(k1_model.RIGHT_ARM_JOINTS):
                lo, hi = jrange[n]
                data.qpos[qadr[n]] = float(np.clip(data.qpos[qadr[n]] + dq[i], lo, hi))
        errs.append(np.degrees(np.linalg.norm(err)))
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps)
    errs = np.array(errs)
    print(f"[retarget] wrote {args.out} ({len(frames)} frames @ {args.fps}fps)")
    print(f"[retarget] orientation tracking error: mean {errs.mean():.1f} deg, "
          f"median {np.median(errs):.1f} deg, max {errs.max():.1f} deg")


if __name__ == "__main__":
    raise SystemExit(main())
