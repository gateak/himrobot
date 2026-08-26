"""Load the Unitree G1 (mujoco_menagerie) and rigidly attach a table-tennis
paddle to the right hand — without editing the vendored asset.

The G1 has a full 7-DOF right arm (shoulder pitch/roll/yaw, elbow, wrist
roll/pitch/yaw), so retargeted human motion looks natural (unlike the
wrist-less K1). Model source: github.com/google-deepmind/mujoco_menagerie.
"""
from __future__ import annotations

import os

import mujoco

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G1_DIR = os.path.join(REPO, "third_party/mujoco_menagerie/unitree_g1")
G1_XML = os.path.join(G1_DIR, "g1.xml")
MESHDIR = os.path.join(G1_DIR, "assets")

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
HAND_BODY = "right_wrist_yaw_link"
SHOULDER_BODY = "right_shoulder_pitch_link"
ELBOW_BODY = "right_elbow_link"

# Paddle welded to the wrist, extending along +x (the hand's pointing dir).
# Blade disc lies in the x-z plane (flat faces point +/- y) like a real paddle.
PADDLE_XML = """
        <body name="paddle" pos="0.085 0 0">
          <inertial pos="0.06 0 0" mass="0.17" diaginertia="0.0004 0.0004 0.0004"/>
          <geom name="paddle_handle" type="capsule" fromto="0 0 0 0.06 0 0" size="0.012" rgba="0.35 0.22 0.12 1" contype="0" conaffinity="0"/>
          <geom name="paddle_blade" type="cylinder" pos="0.12 0 0" zaxis="0 1 0" size="0.085 0.006" rgba="0.85 0.12 0.12 1" contype="0" conaffinity="0"/>
        </body>
"""
_MARKER = 'mesh="right_rubber_hand"/>'  # last geom inside right_wrist_yaw_link


def _xml_with_paddle() -> str:
    with open(G1_XML) as fh:
        text = fh.read()
    text = text.replace('meshdir="assets"', f'meshdir="{MESHDIR}"')
    if _MARKER not in text:
        raise RuntimeError("paddle-injection marker not found; vendored G1 XML changed?")
    return text.replace(_MARKER, _MARKER + "\n" + PADDLE_XML, 1)


def load(paddle: bool = True):
    if paddle:
        model = mujoco.MjModel.from_xml_string(_xml_with_paddle())
    else:
        model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)
    # Stand at the built-in "stand" keyframe if present.
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)
    return model, data
