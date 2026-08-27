#!/usr/bin/env python3
"""Plays a GMR/Unitree-style motion CSV (root xyz, quat xyzw, 29 joint
angles per row) on the Menagerie Unitree G1 in MuJoCo: live passive viewer
by default, offscreen video render with --video. Joint order comes from
--joint-names (text file, one name per line, written by
gmr_bvh_to_g1_headless.py) and is matched to Menagerie joints by name.
"""
import argparse, os, time
import numpy as np, mujoco

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATES = [os.path.join(REPO, "third_party", "mujoco_menagerie", "unitree_g1", "scene.xml"),
               os.path.expanduser("~/him-tools/mujoco_menagerie/unitree_g1/scene.xml")]
SCENE = os.environ.get("G1_SCENE") or next((c for c in _CANDIDATES if os.path.isfile(c)), _CANDIDATES[0])

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--joint-names", default=None, help="default: gmr_g1_joint_order.txt next to the CSV")
ap.add_argument("--fps", type=float, default=30)
ap.add_argument("--video", default=None)
ap.add_argument("--scene", default=SCENE)
a = ap.parse_args()
if a.joint_names is None:
    a.joint_names = os.path.join(os.path.dirname(os.path.abspath(a.csv)), "gmr_g1_joint_order.txt")

m = mujoco.MjModel.from_xml_path(a.scene)
d = mujoco.MjData(m)
data = np.loadtxt(a.csv, delimiter=",", ndmin=2)
names = [l.strip() for l in open(a.joint_names) if l.strip()]
assert data.shape[1] == 7 + len(names), (data.shape, len(names))
qadr = []
for n in names:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    assert jid >= 0, f"joint {n} not in Menagerie model"
    qadr.append(m.jnt_qposadr[jid])
qadr = np.array(qadr)

def set_frame(row):
    d.qpos[:] = 0
    d.qpos[0:3] = row[0:3]
    x, y, z, w = row[3:7]
    d.qpos[3:7] = [w, x, y, z]          # MuJoCo wants wxyz
    d.qpos[qadr] = row[7:]
    mujoco.mj_forward(m, d)

if a.video:
    import imageio
    r = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 3.0, -15, 135
    w = imageio.get_writer(a.video, fps=a.fps)
    for row in data:
        set_frame(row)
        cam.lookat[:] = d.qpos[0:3]
        r.update_scene(d, camera=cam); w.append_data(r.render())
    w.close(); print("wrote", a.video, len(data), "frames")
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as v:
        i = 0
        while v.is_running():
            set_frame(data[i]); v.sync(); i = (i + 1) % len(data)
            time.sleep(1.0 / a.fps)
