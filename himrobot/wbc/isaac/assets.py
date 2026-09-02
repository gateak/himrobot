"""``G1_WBC_CFG`` -- the Isaac Lab ArticulationCfg for the 29-DoF paddle-free
G1, docs/design/whole_body_v1.md sec 2 ("Robot asset", "Actuators (the
mirror)") and sec 3.2. Requires isaaclab (real, on the box, or stubbed --
tests/test_wbc_isaac_stub.py) to import; the JSON-loading and validation
functions below have no isaaclab dependency and are exercised directly by
tests without any stub.

Asset: ``himrobot.wbc.mujoco_mirror.XML_PATH`` (``assets/wbc/
g1_29dof_paddle_free.xml`` -- WB4, landed: vendored G1 MJCF + the scored
scene's own paddle, freejoint kept, no table/net/floor/ball) converted on
the box with Isaac Lab's ``scripts/tools/convert_mjcf.py`` to a USD at
``G1_WBC_USD_PATH`` (env var; default is the box-side convention,
``/workspace/assets/g1_29dof_paddle.usd``) -- this conversion has never been
run (see docs/design/whole_body_v1.md sec 11).

Actuators: read from ``himrobot.wbc.mujoco_mirror.JSON_PATH`` (``assets/wbc/
actuators_mujoco.json``, WB4, landed) -- **never typed by hand**, so the
Isaac gains can never silently drift from the compiled scored-scene model.
Schema (``mujoco_mirror.build_actuators_json``): ``{"joint_order": [...],
"joints": {name: {"kp", "kv", "armature", "frictionloss",
"effort_limit": [lo, hi], "jnt_range": [lo, hi]}}, "body_names": [...],
"bodies": {...}, "total_mass_kg": ..., "fk_truth": {...}}``. One
``ImplicitActuatorCfg`` per joint (29 entries): stiffness=kp, damping=kv,
effort_limit=``max(abs(effort_limit))`` (the scored model's torque clamp is
symmetric in practice; the JSON keeps both signs, this module takes the
magnitude isaaclab's own ``effort_limit`` scalar field wants), armature,
friction=frictionloss.
"""
from __future__ import annotations

import json
import os

from himrobot.sim import scene
from himrobot.wbc import mujoco_mirror

try:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg
    HAS_ISAACLAB = True
except ImportError:  # pragma: no cover -- exercised only on a real/stubbed isaaclab
    sim_utils = None
    ImplicitActuatorCfg = None
    ArticulationCfg = None
    HAS_ISAACLAB = False

ACTUATORS_JSON_ENV = "G1_WBC_ACTUATORS_JSON"
DEFAULT_ACTUATORS_JSON = str(mujoco_mirror.JSON_PATH)  # == ASSETS_DIR / "wbc" / "actuators_mujoco.json"
MIRROR_XML_PATH = str(mujoco_mirror.XML_PATH)  # convert_mjcf.py's input on the box
JOINT_FIELDS = ("kp", "kv", "armature", "frictionloss", "effort_limit", "jnt_range")

G1_WBC_USD_ENV = "G1_WBC_USD_PATH"
DEFAULT_USD_PATH = "/workspace/assets/g1_29dof_paddle.usd"  # convert_mjcf.py's box-side output path

# The crouch-pose joint targets convert_mjcf.py's USD must spawn at: STAND_POSE,
# same as the scored scene (docs/design/whole_body_v1.md sec 2/3.2).
INIT_JOINT_POS: dict[str, float] = dict(zip(scene.JOINT_ORDER, (float(v) for v in scene.STAND_POSE)))


def actuators_json_path() -> str:
    """Resolution order: G1_WBC_ACTUATORS_JSON env var, then the checked-in
    assets/wbc/actuators_mujoco.json (himrobot.wbc.mujoco_mirror's output)."""
    return os.environ.get(ACTUATORS_JSON_ENV, DEFAULT_ACTUATORS_JSON)


def load_actuator_table(path: str | None = None) -> dict:
    """Load + validate the actuator-mirror JSON (himrobot.wbc.mujoco_mirror's
    schema -- see module docstring): every scene.JOINT_ORDER name present
    under "joints", each with all of JOINT_FIELDS."""
    p = path or actuators_json_path()
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"actuator mirror not found at {p!r}; generate it with "
            f"`python -m himrobot.wbc.mujoco_mirror`, or point {ACTUATORS_JSON_ENV} at one."
        )
    with open(p) as fh:
        table = json.load(fh)
    joints = table.get("joints")
    if not isinstance(joints, dict):
        raise ValueError(f"{p}: missing or malformed 'joints' field")
    missing = [j for j in scene.JOINT_ORDER if j not in joints]
    if missing:
        raise ValueError(f"{p}: 'joints' missing entries for {missing}")
    for name, entry in joints.items():
        absent = [f for f in JOINT_FIELDS if f not in entry]
        if absent:
            raise ValueError(f"{p}: joints[{name!r}] missing fields {absent}")
    return table


def build_actuators(table: dict | None = None) -> dict:
    """One ImplicitActuatorCfg per joint (29 entries, keyed by joint name --
    an exact single-joint expr each, not a regex group, so the mirror is
    never ambiguous about which joint a gain belongs to)."""
    if not HAS_ISAACLAB:
        raise ImportError("build_actuators needs isaaclab (or the stub gate)")
    table = table or load_actuator_table()
    joints = table["joints"]
    return {
        name: ImplicitActuatorCfg(
            joint_names_expr=[name],
            stiffness=float(joints[name]["kp"]),
            damping=float(joints[name]["kv"]),
            effort_limit=float(max(abs(v) for v in joints[name]["effort_limit"])),
            armature=float(joints[name]["armature"]),
            friction=float(joints[name]["frictionloss"]),
        )
        for name in scene.JOINT_ORDER
    }


def build_g1_wbc_cfg(table: dict | None = None, usd_path: str | None = None):
    """The ArticulationCfg env_cfg.py's SceneCfg attaches as ``robot``."""
    if not HAS_ISAACLAB:
        raise ImportError("build_g1_wbc_cfg needs isaaclab (or the stub gate)")
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path or os.environ.get(G1_WBC_USD_ENV, DEFAULT_USD_PATH),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # ON, unlike BeyondMimic's own G1 config -- docs/design/whole_body_v1.md
                # sec 11 ("Self-collision") keeps self-collisions enabled for the G1.
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(scene.ROBOT_Z)),
            joint_pos=dict(INIT_JOINT_POS),
            joint_vel={".*": 0.0},
        ),
        # 1.0, not BeyondMimic's 0.9: the hitter already commands the model's
        # own limits (waist roll/pitch pinned at 0.52 rad in most serves,
        # results/waist_lock/RESULT.md), so softening them would clip a
        # target the scored scene itself allows.
        soft_joint_pos_limit_factor=1.0,
        actuators=build_actuators(table),
    )
