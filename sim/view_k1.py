#!/usr/bin/env python
"""Watch the Booster K1 (22-DOF) humanoid move — locally, no HIM servers.

Model: third_party/booster_assets/robots/K1/K1_22dof.xml
       (c) 2025 Booster Robotics, BSD-3-Clause. This is the real, public K1
       whose 22-joint layout matches HIM's "Adam" table-tennis fixture.

Two modes:
  * default  -> render a short kinematic arm-wave to a GIF you can open.
  * --live   -> open the interactive MuJoCo viewer (drag to orbit).

This is a KINEMATIC preview: we set joint angles and call mj_forward. We do
NOT step physics — a free-base humanoid with no balance controller would just
fall over. The point here is to *see* the real robot and its joints move.
"""
from __future__ import annotations

import argparse
import math
import os

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(REPO, "third_party/booster_assets/robots/K1/K1_22dof.xml")

# Gentle wave on a few joints: name -> (amplitude rad, phase). All within the
# URDF limits so the preview looks natural.
WAVE = {
    "AAHead_yaw": (0.4, 0.5),
    "ALeft_Shoulder_Pitch": (0.6, 0.0),
    "Left_Elbow_Pitch": (0.7, 1.2),
    "ARight_Shoulder_Pitch": (0.6, math.pi),
    "Right_Elbow_Pitch": (0.7, 1.2 + math.pi),
}
FREQ_HZ = 0.5  # slow, one full swing every 2 s


def _build():
    model = mujoco.MjModel.from_xml_path(XML)
    return model, mujoco.MjData(model)


def _qadr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[jid]) if jid >= 0 else None


def _apply_pose(model, data, t):
    mujoco.mj_resetData(model, data)
    for name, (amp, phase) in WAVE.items():
        adr = _qadr(model, name)
        if adr is not None:
            data.qpos[adr] = amp * math.sin(2 * math.pi * FREQ_HZ * t + phase)
    mujoco.mj_forward(model, data)


def _camera():
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 135, -12, 2.4
    cam.lookat[:] = [0.0, 0.0, 0.6]
    return cam


def render_gif(out, seconds, fps, width, height):
    import imageio

    model, data = _build()
    cam = _camera()
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:  # no GL context available (headless CI, etc.)
        print(f"[view_k1] offscreen render unavailable ({exc!r}).")
        print("[view_k1] run with --live to open the interactive viewer instead.")
        return 1
    frames = []
    for i in range(int(seconds * fps)):
        _apply_pose(model, data, i / fps)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, frames, fps=fps)
    print(f"[view_k1] wrote {out} ({len(frames)} frames, {width}x{height})")
    return 0


def live(seconds):
    import time

    import mujoco.viewer

    model, data = _build()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and (time.time() - start) < seconds:
            _apply_pose(model, data, time.time() - start)
            viewer.sync()
            time.sleep(1 / 60)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="open interactive viewer")
    ap.add_argument("--out", default=os.path.join(REPO, "sim", "k1_wave.gif"))
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=384)
    args = ap.parse_args()
    if args.live:
        return live(args.seconds)
    return render_gif(args.out, args.seconds, args.fps, args.width, args.height)


if __name__ == "__main__":
    raise SystemExit(main())
