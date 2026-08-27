"""Load the Booster K1 and rigidly attach a table-tennis paddle to the right
hand, without modifying the vendored asset: read the vendored K1 MuJoCo XML
as text, point meshdir at the vendored meshes (absolute, so it loads from
anywhere), inject a paddle body into `right_hand_link`, and compile from the
string. The vendored file remains the single source of truth; nothing under
third_party/ is edited.
"""
from __future__ import annotations

import os

import mujoco

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K1_DIR = os.path.join(REPO, "third_party/booster_assets/robots/K1")
K1_XML = os.path.join(K1_DIR, "K1_22dof.xml")
MESHDIR = os.path.join(K1_DIR, "meshes")

# The 4 actuated right-arm joints, shoulder -> hand.
RIGHT_ARM_JOINTS = [
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]

# Paddle welded (no joint) to the right hand. Blade faces +/- local x; handle
# and blade extend along -y, matching how the forearm hangs from the elbow.
PADDLE_XML = """
      <body name="paddle" pos="0 -0.13 0">
        <inertial pos="0 -0.09 0" mass="0.17" diaginertia="0.0004 0.0004 0.0004"/>
        <geom name="paddle_handle" type="capsule" fromto="0 0 0 0 -0.05 0" size="0.013" rgba="0.35 0.22 0.12 1" contype="0" conaffinity="0"/>
        <geom name="paddle_blade" type="cylinder" pos="0 -0.115 0" zaxis="1 0 0" size="0.085 0.006" rgba="0.85 0.12 0.12 1" contype="0" conaffinity="0"/>
      </body>
"""

# Last child geom of right_hand_link in the vendored file; paddle goes after it.
_ANCHOR = '              <geom type="mesh" rgba="0.75294 0.75294 0.75294 1" mesh="Right_Arm_4"/>'


def _xml_with_paddle() -> str:
    with open(K1_XML) as fh:
        text = fh.read()
    text = text.replace('meshdir="meshes/"', f'meshdir="{MESHDIR}"')
    if _ANCHOR not in text:
        raise RuntimeError("paddle-injection anchor not found; vendored XML changed?")
    return text.replace(_ANCHOR, _ANCHOR + "\n" + PADDLE_XML, 1)


def load(paddle: bool = True):
    if paddle:
        model = mujoco.MjModel.from_xml_string(_xml_with_paddle())
    else:
        model = mujoco.MjModel.from_xml_path(K1_XML)
    return model, mujoco.MjData(model)
