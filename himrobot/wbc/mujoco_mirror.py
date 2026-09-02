"""Writes the MuJoCo-side artefacts a WBC Isaac Lab conversion (and the
MuJoCo replay gate, himrobot.wbc.replay_library) both read:

    assets/wbc/g1_29dof_paddle_free.xml   -- vendored G1 + paddle, free
                                              pelvis, no table/net/floor/ball
    assets/wbc/actuators_mujoco.json      -- per-joint PD gains/limits, per-
                                              body mass/inertia, FK truth

docs/design/whole_body_v1.md sec 2 ("Actuators (the mirror)") and sec 3.2
("Actuator mirror"): Isaac's ImplicitActuatorCfg per joint must come from the
COMPILED scored model, never typed by hand and never copied from
BeyondMimic -- this module is that single source. Both outputs are meant to
be committed (regenerate with ``python -m himrobot.wbc.mujoco_mirror`` after
any change to assets/unitree_g1/g1.xml or himrobot.sim.scene's paddle/pose
constants; tests/test_wbc_mirror.py pins the regenerated JSON against the
committed one so a stale commit is caught in CI).

The XML mirrors himrobot.sim.scene._build_xml's own edit sequence (meshdir
rewrite, <option> timestep, paddle injection via the same scene.PADDLE_XML /
scene._PADDLE_MARKER constants, keyframe drop) rather than calling that
function directly: _build_xml always appends scene's table/net/floor/ball
and contact-pair XML, which Isaac Lab's ``scripts/tools/convert_mjcf.py``
neither wants nor can use (no ball, no table, in the WBC lane -- the
disturbance is the hitter's own recorded swing, not a served ball, see the
design doc's thesis). Reusing scene.py's own edit calls -- not scene.py
itself -- keeps the string surgery in exactly one place; scene.py is not in
this work item's file list and is not touched.

Collision mask of the paddle blade: the honest scored scene
(``robot_table_collision=False``) gives ``paddle_blade`` contype/conaffinity
4/4 so it meets the ball and nothing else -- never the robot's own body,
never the floor (scene._build_xml's own comment: "no propping the body up
on the paddle"). The mirror has no ball, so the equivalent statement is
"collides with nothing", written as contype="0" conaffinity="0" -- the form
MJCF importers read as non-colliding. Without this edit the vendored 1/1
default makes the blade a collider that can strike the hips mid-swing
(measured in a coarse shoulder/waist sweep), which the scored scene never
does.

FK truth is expressed in the PELVIS frame (pelvis body at STAND_POSE), not
the world frame, so the file does not depend on TT_ROBOT_Y (scene.ROBOT_Y is
read at import; the WBC lane runs at -0.10 while the default is 0.0). World
pose at a given stance = pelvis-frame value + scene.BASE_NOMINAL_POS
(BASE_NOMINAL_QUAT is the identity).
"""
from __future__ import annotations

import json

import mujoco
import numpy as np

from himrobot.paths import ASSETS_DIR
from himrobot.sim import scene

OUT_DIR = ASSETS_DIR / "wbc"
XML_PATH = OUT_DIR / "g1_29dof_paddle_free.xml"
JSON_PATH = OUT_DIR / "actuators_mujoco.json"

# assets/wbc/g1_29dof_paddle_free.xml -> assets/unitree_g1/assets, relative
# (Isaac's convert_mjcf.py resolves meshdir relative to the MJCF's own path,
# not the caller's cwd -- an absolute path baked in at generation time would
# only work on this machine).
_MESHDIR_REL = "../unitree_g1/assets"


def build_mirror_xml() -> str:
    """The free-base, no-scene MJCF text (see module docstring)."""
    with open(scene.G1_XML) as fh:
        text = fh.read()

    text = text.replace('meshdir="assets"', f'meshdir="{_MESHDIR_REL}"')
    text = text.replace(
        '<option integrator="implicitfast"/>',
        '<option integrator="implicitfast" timestep="0.002" gravity="0 0 -9.81"/>',
        1,
    )

    if scene._PADDLE_MARKER not in text:
        raise RuntimeError("paddle-injection marker not found; vendored G1 XML changed?")
    text = text.replace(scene._PADDLE_MARKER, scene._PADDLE_MARKER + "\n" + scene.PADDLE_XML, 1)
    # Honest-scene blade mask (see module docstring): meets nothing in a
    # ball-free model. Same edit point scene._build_xml uses for the 4/4 mask.
    blade_tag = '<geom name="paddle_blade" type="cylinder"'
    if blade_tag not in text:
        raise RuntimeError("paddle_blade geom not found after injection; scene.PADDLE_XML changed?")
    text = text.replace(blade_tag, blade_tag + ' contype="0" conaffinity="0"', 1)

    # Drop the vendor <keyframe> exactly as scene._build_xml does (its qpos
    # length no longer matches nq once a ball/table are never added but the
    # paddle body is -- see STAND_POSE's docstring in scene.py for the same
    # reasoning applied there); nothing is added in its place here.
    kf_start = text.find("<keyframe>")
    kf_end = text.find("</keyframe>")
    if kf_start < 0 or kf_end < 0:
        raise RuntimeError("vendored G1 keyframe not found; XML changed?")
    text = text[:kf_start] + text[kf_end + len("</keyframe>"):]
    return text


def _joints_table(m: mujoco.MjModel) -> dict:
    """Per-joint PD gains/limits, keyed by scene.JOINT_ORDER name. Every
    number is read off the COMPILED model (mujoco resolves a <position
    class="g1" dampratio="1"> actuator's implied damping into an explicit
    kv at compile time -- see actuator_biasprm[:, 2] -- so this is not a
    re-derivation of MuJoCo's own dampratio formula, just a read)."""
    out: dict[str, dict] = {}
    for name in scene.JOINT_ORDER:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert jid >= 0 and aid >= 0, name
        assert bool(m.jnt_actfrclimited[jid]), f"{name}: expected jnt_actfrclimited=True"
        dof = int(m.jnt_dofadr[jid])
        out[name] = {
            "kp": float(m.actuator_gainprm[aid, 0]),
            "kv": float(-m.actuator_biasprm[aid, 2]),
            "armature": float(m.dof_armature[dof]),
            "frictionloss": float(m.dof_frictionloss[dof]),
            "effort_limit": [float(v) for v in m.jnt_actfrcrange[jid]],
            "jnt_range": [float(v) for v in m.jnt_range[jid]],
        }
    return out


def _bodies_table(m: mujoco.MjModel, skip: tuple[str, ...] = ("ball",)) -> dict:
    """Per-body mass + diagonal inertia (the body's own inertial frame,
    matching the XML's <inertial diaginertia=.../> convention), for every
    named body in the model except ``skip`` (the ball: present in the scored
    model this reads, absent from the mirror this JSON describes)."""
    out: dict[str, dict] = {}
    for bid in range(m.nbody):
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not bname or bname in skip:
            continue
        out[bname] = {
            "mass": float(m.body_mass[bid]),
            "diaginertia": [float(v) for v in m.body_inertia[bid]],
        }
    return out


def _min_foot_sphere_z(m: mujoco.MjModel, d: mujoco.MjData, body_id: int) -> float:
    """Lowest world z, minus its own radius, over every collision sphere
    geom hung off ``body_id`` (the four foot-contact spheres per ankle_roll
    link -- see assets/unitree_g1/g1.xml's ``class="foot"`` geoms)."""
    z_min = None
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] != body_id or m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_SPHERE:
            continue
        z = float(d.geom_xpos[gid][2] - m.geom_size[gid][0])
        if z_min is None or z < z_min:
            z_min = z
    assert z_min is not None, "no foot-sphere geom found"
    return z_min


def _fk_truth(m: mujoco.MjModel, d: mujoco.MjData) -> dict:
    """Pelvis-frame facts at STAND_POSE on the honest free-base scene
    (robot_table_collision=False, the model the Isaac conversion's paddle-
    pose parity check compares against -- docs/design/whole_body_v1.md sec
    7.2, "Gate 1"). Positions/normal are rotated into and measured from the
    pelvis body, so the block is identical whatever TT_ROBOT_Y (or any other
    pelvis placement) is in effect; ``min_foot_sphere_z`` is the one world-z
    number, quoted with the pelvis at the nominal ``pelvis_height_m``
    (scene.ROBOT_Z, a constant). The FK is evaluated with the pelvis AT THE
    ORIGIN with the identity orientation, so the pelvis frame coincides with
    the world frame and the stored numbers involve no subtraction -- they are
    bit-identical across stances, which the committed-file test relies on."""
    qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in scene.JOINT_ORDER])
    base_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    base_qadr = int(m.jnt_qposadr[base_jid])

    mujoco.mj_resetData(m, d)
    d.qpos[base_qadr:base_qadr + 3] = 0.0
    d.qpos[base_qadr + 3:base_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[qadr] = scene.STAND_POSE
    mujoco.mj_forward(m, d)

    pelvis_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    assert pelvis_bid >= 0
    p0 = d.xpos[pelvis_bid].copy()
    r0 = d.xmat[pelvis_bid].reshape(3, 3).copy()
    assert np.all(p0 == 0.0) and np.array_equal(r0, np.eye(3)), "pelvis is not the free-joint body?"

    def rel_pos(p_world: np.ndarray) -> list[float]:
        return [float(v) for v in r0.T @ (np.asarray(p_world) - p0)]

    def rel_dir(v_world: np.ndarray) -> list[float]:
        return [float(v) for v in r0.T @ np.asarray(v_world)]

    paddle_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "paddle")
    blade_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "paddle_blade")
    assert paddle_bid >= 0 and blade_gid >= 0
    # Column 2 (not 1) is the blade's true flat-face normal -- see
    # ik_hitter._solve_ik_stock's own comment on this same geom for why.
    normal = d.geom_xmat[blade_gid].reshape(3, 3)[:, 2].copy()

    left_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    assert left_bid >= 0 and right_bid >= 0
    min_foot_z_world = min(_min_foot_sphere_z(m, d, left_bid), _min_foot_sphere_z(m, d, right_bid))
    # Re-quote at the nominal stand height so the number does not depend on
    # where this process happened to place the pelvis.
    min_foot_z = float(min_foot_z_world - p0[2] + scene.ROBOT_Z)

    return {
        "frame": "pelvis body frame at STAND_POSE; world = value + scene.BASE_NOMINAL_POS "
                 "(BASE_NOMINAL_QUAT is the identity); independent of TT_ROBOT_Y",
        "stand_pose": [float(v) for v in scene.STAND_POSE],
        "pelvis_height_m": float(scene.ROBOT_Z),
        "base_nominal_quat": [float(v) for v in scene.BASE_NOMINAL_QUAT],
        "paddle_body_pos": rel_pos(d.xpos[paddle_bid]),
        "paddle_blade_pos": rel_pos(d.geom_xpos[blade_gid]),
        "paddle_blade_normal": rel_dir(normal),
        "left_ankle_roll_link_pos": rel_pos(d.xpos[left_bid]),
        "right_ankle_roll_link_pos": rel_pos(d.xpos[right_bid]),
        "min_foot_sphere_z": min_foot_z,
    }


def build_actuators_json() -> dict:
    """The full assets/wbc/actuators_mujoco.json payload, computed fresh
    from scene.build_model(fixed_base=False, paddle=True,
    robot_table_collision=False) -- see module docstring."""
    m = scene.build_model(fixed_base=False, paddle=True, robot_table_collision=False)
    d = mujoco.MjData(m)

    ball_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
    assert ball_bid >= 0
    total_mass = float(sum(m.body_mass) - m.body_mass[ball_bid])  # excl. ball, incl. paddle
    bodies = _bodies_table(m)  # the ball is skipped: it is not in the mirror

    return {
        "joint_order": list(scene.JOINT_ORDER),
        "joints": _joints_table(m),
        "body_names": list(bodies),
        "bodies": bodies,
        "total_mass_kg": total_mass,
        "fk_truth": _fk_truth(m, d),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_text = build_mirror_xml()
    XML_PATH.write_text(xml_text)
    # Loads on its own, no scene.py involved -- the same check
    # tests/test_wbc_mirror.py runs, kept here so a manual run catches a
    # broken XML immediately instead of only under pytest.
    mujoco.MjModel.from_xml_path(str(XML_PATH))
    print(f"wrote {XML_PATH}")

    data = build_actuators_json()
    with open(JSON_PATH, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    print(f"wrote {JSON_PATH} (total_mass_kg={data['total_mass_kg']:.5f})")


if __name__ == "__main__":
    main()
