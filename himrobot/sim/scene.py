"""G1-vs-table-tennis-ball MuJoCo scene: model construction + gym-ish env.

Builds the scene by editing the vendored Menagerie G1 XML **in memory**
(string surgery, same technique as sim/g1_model.py's paddle injection —
duplicated here rather than imported so this package has no runtime
dependency on sim/, which other agents are concurrently editing):

  1. the pelvis `<freejoint>` is dropped and the pelvis body is repositioned,
     turning the free-floating G1 into a fixed-base manipulator (legs still
     have their own hinge joints and actuators; they just can't fall over —
     the robot doesn't need to balance to play table tennis from a fixed
     stance) so it can't fall;
  2. a paddle is welded onto the right wrist (right-handed);
  3. an ITTF table + net + floor + a free-joint ball are added, with
     contact solver params tuned for ~0.9 ball/table restitution.

Table geometry (ITTF, cited in the constants below): 2.74 m x 1.525 m,
playing surface at 0.76 m, net 15.25 cm tall spanning the width. The table
is centred at the world origin with x running along the 2.74 m length and
y along the 1.525 m width; the robot stands at the x<0 end, facing +x.

Free-base contract (``TableTennisEnv(..., fixed_base=False)``)
-------------------------------------------------------------
``fixed_base=True`` is the default and stays the project's scored scene:
the pelvis is welded, the robot cannot fall, and every observation, event
and evaluator number is exactly what it has always been. Passing
``fixed_base=False`` builds the model with the vendor pelvis ``<freejoint>``
kept (``build_model(fixed_base=False)``) so the G1 must hold itself up, and
changes three things and nothing else:

  * **Reset.** Every reset places the pelvis free joint at
    ``(ROBOT_X, 0.0, ROBOT_Z)`` with identity orientation ``(1, 0, 0, 0)``
    (MuJoCo wxyz) and zero linear/angular velocity, so the robot starts in
    exactly the pose the welded scene bolts it to. Joints, paddle, table
    and ball are set up identically to the welded path.
  * **Observation.** The obs dict gains four keys — ``base.position`` (3),
    ``base.orientation`` (4, wxyz), ``base.linear_velocity`` (3) and
    ``base.angular_velocity`` (3). These are emitted **only** in the
    free-base scene; the welded obs dict is unchanged key for key, which is
    what lets a base-aware policy detect the free base from the observation
    alone (see himrobot.policies.ik_hitter's base-aware IK).
  * **Events.** ``events`` gains ``"fell"``, latched True once the pelvis
    drops below FALL_PELVIS_Z. It is a diagnostic only: the episode keeps
    running to its normal length and ``legal_return()`` is untouched, so a
    fall costs a return solely through the physics, never through scoring.
    ``"fell"`` is likewise absent from the welded scene's events.
"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence

import mujoco
import numpy as np

from himrobot.paths import ASSETS_DIR, REPO_ROOT

REPO = str(REPO_ROOT)
# Vendored copy (assets/unitree_g1, BSD-3-Clause, google-deepmind/mujoco_menagerie) ships with the HIM package;
# the sparse checkout under third_party/ is the developer fallback.
_G1_DIR = next((d for d in (os.path.join(str(ASSETS_DIR), "unitree_g1"), os.path.join(REPO, "third_party", "mujoco_menagerie", "unitree_g1")) if os.path.isfile(os.path.join(d, "g1.xml"))), os.path.join(str(ASSETS_DIR), "unitree_g1"))
G1_XML = os.path.join(_G1_DIR, "g1.xml")
MESHDIR = os.path.join(_G1_DIR, "assets")

# ---------------------------------------------------------------------------
# Table-tennis facts (ITTF regulation dimensions, as given to this task).
# ---------------------------------------------------------------------------
TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
TABLE_HALF_LENGTH = TABLE_LENGTH / 2.0  # 1.37, table spans x in [-1.37, 1.37]
TABLE_HALF_WIDTH = TABLE_WIDTH / 2.0  # 0.7625, table spans y in [-0.7625, 0.7625]
TABLE_TOP_Z = 0.76  # playing-surface height above the floor
TABLE_THICKNESS = 0.02  # assumption: slab visual/collision thickness (not an ITTF fact)
NET_HEIGHT = 0.1525  # above the playing surface, spans the full width
NET_TOP_Z = TABLE_TOP_Z + NET_HEIGHT

BALL_DIAMETER = 0.040
BALL_RADIUS = BALL_DIAMETER / 2.0
BALL_MASS = 0.0027

# Assumption: ball/table coefficient of restitution ~= 0.9 (ITTF fact given).
# MuJoCo has no literal restitution parameter; we approximate it by tuning
# the normal-contact solver reference (solref = [timeconst, dampratio],
# underdamped so the contact spring rebounds) against an explicit drop-test
# calibration: dropping the ball from 0.3 m onto table_top, impact speed is
# 2.335 m/s (free fall, independent of solref) and with these values rebound
# speed is 2.102 m/s -> e = 0.90 (scratch calibration script, not shipped —
# numbers in the task report). A stiffer/quicker pair (timeconst=0.008 s,
# dampratio=0.16) also hits e=0.90 in that same vertical drop and was the
# first local optimum found, but it turned out to be on a knife-edge: still
# numerically stable, but the *slightest* further increase in bounciness
# (e.g. dampratio 0.158) diverges. timeconst=0.05 s trades a slightly
# softer/slower contact for a wide, well-behaved stable region, confirmed
# under oblique/spinning impacts (6 m/s lateral + 40 rad/s spin: bounded
# ~8 m/s peak speed, no blow-up, energy decays over repeated bounces).
# solimp is a near-rigid impedance so the ball doesn't sink into the table.
BOUNCE_SOLREF = "0.05 0.048"
BOUNCE_SOLIMP = "0.9 0.95 0.001 0.5 2"
CONTACT_FRICTION = "0.4 0.005 0.0001"  # sliding/torsional/rolling — rubber-ish default

# Robot standing pose: pelvis ~0.4 m behind the robot's own end line
# (x = -TABLE_HALF_LENGTH), facing +x (identity pelvis orientation already
# points local +x = the G1's forward/toe direction — confirmed from the
# vendor XML's foot-collision-sphere offsets — at world +x, so no rotation
# is needed to face the table).
ROBOT_STANDBACK_M = 0.4
ROBOT_X = -(TABLE_HALF_LENGTH + ROBOT_STANDBACK_M)  # -1.77
# Lateral stance offset (m). The serves DEFAULT_PARAMS draws land forehand-only
# (target_y in [-0.65, -0.10]) and the misses cluster at far-forehand arrival,
# so sliding the welded pelvis toward -y turns the far forehand into the middle
# of the arm's workspace: measured -0.10 m takes the shipped, unretrained policy
# from 441/600 to 467/600 legal returns (work/tt/design/hitter_v3_reach.md 2.1).
# Read once at import from TT_ROBOT_Y so a run opts in without editing source;
# the default 0.0 renders as "0" in _build_xml's {ROBOT_Y:g}, which keeps the
# scene XML byte-identical to the pre-stance tree. It is tuned to *this* serve
# distribution -- re-sweep it before any serve-distribution change ships.
ROBOT_Y = float(os.environ.get("TT_ROBOT_Y", "0.0"))
ROBOT_Z = 0.793  # unchanged from the vendor pelvis pos / "stand" keyframe height

# Nominal pelvis pose the welded scene bolts the robot to, and the pose the
# free-base scene resets to. Kept as a (position, wxyz-quaternion) pair so
# base-aware code (himrobot.policies.ik_hitter) can name one origin instead of
# rebuilding it from the three scalars above.
BASE_NOMINAL_POS = np.array([ROBOT_X, ROBOT_Y, ROBOT_Z], dtype=np.float64)
BASE_NOMINAL_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Free base only: pelvis height (m) below which the episode is latched as a
# fall. 0.45 m is well under the 0.793 m stand height and under any crouch the
# stand-pose legs can produce, so it fires on a real collapse, not on a wobble.
# Diagnostic only — it never changes scoring (see the module docstring).
FALL_PELVIS_Z = 0.45

# Joint order = .him/contracts/runtime.json extensions.robot.joint_order,
# which is also the vendored G1 XML's actuator declaration order.
JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
NUM_JOINTS = len(JOINT_ORDER)  # 29

# Default "ready" pose: the vendor G1 XML's own "stand" keyframe `ctrl`
# array (29 values, already in JOINT_ORDER) — arms relaxed, elbows bent.
# We can't keep the XML <keyframe> itself (its `qpos` was 7 free-joint
# values + 29 joint values; with the free joint removed the lengths
# coincidentally still add up to 36 [29 robot + 7 ball], which would
# silently misassign values), so it's dropped from the XML and this array
# is used directly instead.
STAND_POSE = np.array([
    0, 0, 0, 0, 0, 0,        # left leg
    0, 0, 0, 0, 0, 0,        # right leg
    0, 0, 0,                 # waist
    0.2, 0.2, 0, 1.28, 0, 0, 0,   # left arm
    0.2, -0.2, 0, 1.28, 0, 0, 0,  # right arm
], dtype=np.float64)
# Optional leg-stance override, read once at import like TT_ROBOT_Y: twelve
# comma-separated radians for JOINT_ORDER slots 0-11 (left leg hip
# pitch/roll/yaw, knee, ankle pitch/roll; then the right leg). Absent (the
# default) leaves STAND_POSE exactly as above -- straight legs, the vendor
# "stand" keyframe -- so every parity fixture is untouched. Used by the
# free-base stance sweep (results/stance_sweep/RESULT.md).
_LEG_POSE_ENV = os.environ.get("TT_LEG_POSE", "")
if _LEG_POSE_ENV:
    _leg = np.array([float(v) for v in _LEG_POSE_ENV.split(",")], dtype=np.float64)
    if _leg.shape != (12,):
        raise ValueError(f"TT_LEG_POSE needs 12 comma-separated values, got {_leg.shape}")
    STAND_POSE[:12] = _leg

# Paddle geometry: identical to sim/g1_model.py's PADDLE_XML (same welded
# 0.17 kg disc on the right wrist), except paddle_blade keeps real collision
# (contype/conaffinity default = 1) instead of being visual-only, since it
# has to actually hit the ball.
PADDLE_XML = """
        <body name="paddle" pos="0.085 0 0">
          <inertial pos="0.06 0 0" mass="0.17" diaginertia="0.0004 0.0004 0.0004"/>
          <geom name="paddle_handle" type="capsule" fromto="0 0 0 0.06 0 0" size="0.012" rgba="0.35 0.22 0.12 1" contype="0" conaffinity="0"/>
          <geom name="paddle_blade" type="cylinder" pos="0.12 0 0" zaxis="0 1 0" size="0.085 0.006" rgba="0.85 0.12 0.12 1" friction="{friction}"/>
        </body>
""".format(friction=CONTACT_FRICTION)
_PADDLE_MARKER = 'mesh="right_rubber_hand"/>'  # last geom inside right_wrist_yaw_link

_SCENE_XML = """
    <geom name="table_top" type="box" pos="0 0 {table_z}" size="{half_len} {half_wid} {half_thick}"
          rgba="0.05 0.30 0.55 1" friction="{friction}"/>
    <geom name="tt_centerline" type="box" pos="0 0 {line_z}" size="{half_len} 0.01 0.0005"
          rgba="1 1 1 1" contype="0" conaffinity="0"/>
    <geom name="net" type="box" pos="0 0 {net_z}" size="0.01 {half_wid} {net_half_h}"
          rgba="0.9 0.9 0.9 0.5" friction="0.2 0.005 0.0001"/>
    <geom name="floor" type="plane" pos="0 0 0" size="0 0 0.05" rgba="0.25 0.30 0.35 1" friction="0.6 0.005 0.0001"/>
    <body name="ball" pos="0.6 0 1.2">
      <freejoint name="ball_joint"/>
      <geom name="ball" type="sphere" size="{ball_r}" mass="{ball_mass}" rgba="1 0.55 0.05 1"
            friction="{friction}"/>
    </body>
""".format(
    table_z=TABLE_TOP_Z - TABLE_THICKNESS / 2.0,
    half_len=TABLE_HALF_LENGTH, half_wid=TABLE_HALF_WIDTH, half_thick=TABLE_THICKNESS / 2.0,
    line_z=TABLE_TOP_Z + 0.0006,
    net_z=TABLE_TOP_Z + NET_HEIGHT / 2.0, net_half_h=NET_HEIGHT / 2.0,
    ball_r=BALL_RADIUS, ball_mass=BALL_MASS,
    friction=CONTACT_FRICTION,
)

def _contact_xml(paddle: bool) -> str:
    # Named (unlike a bare geom1/geom2 pair) so TableTennisEnv can look up
    # pair_solref by mj_name2id for physics_jitter instead of assuming index order.
    pairs = f'    <pair name="ball_table_pair" geom1="ball" geom2="table_top" solref="{BOUNCE_SOLREF}" solimp="{BOUNCE_SOLIMP}"/>\n'
    if paddle:
        pairs += f'    <pair name="ball_paddle_pair" geom1="ball" geom2="paddle_blade" solref="{BOUNCE_SOLREF}" solimp="{BOUNCE_SOLIMP}"/>\n'
    return f"\n  <contact>\n{pairs}  </contact>\n"


def _build_xml(fixed_base: bool, paddle: bool, robot_table_collision: bool = True) -> str:
    with open(G1_XML) as fh:
        text = fh.read()
    text = text.replace('meshdir="assets"', f'meshdir="{MESHDIR}"')
    text = text.replace(
        '<option integrator="implicitfast"/>',
        '<option integrator="implicitfast" timestep="0.002" gravity="0 0 -9.81"/>',
        1,
    )

    if fixed_base:
        if '<freejoint name="floating_base_joint"/>' not in text:
            raise RuntimeError("pelvis freejoint not found; vendored G1 XML changed?")
        text = text.replace('<freejoint name="floating_base_joint"/>', "", 1)
        text = text.replace(
            '<body name="pelvis" pos="0 0 0.793" childclass="g1">',
            f'<body name="pelvis" pos="{ROBOT_X} {ROBOT_Y:g} {ROBOT_Z}" childclass="g1">',
            1,
        )

    if paddle:
        if _PADDLE_MARKER not in text:
            raise RuntimeError("paddle-injection marker not found; vendored G1 XML changed?")
        text = text.replace(_PADDLE_MARKER, _PADDLE_MARKER + "\n" + PADDLE_XML, 1)

    # Drop the vendor <keyframe> (see STAND_POSE docstring for why it can't
    # be reused post-edit — its qpos length no longer matches nq once the
    # pelvis freejoint is gone and the ball's freejoint is added) and add
    # our table/net/floor/ball + contact pairs in its place, before </mujoco>.
    kf_start = text.find("<keyframe>")
    kf_end = text.find("</keyframe>")
    if kf_start < 0 or kf_end < 0:
        raise RuntimeError("vendored G1 keyframe not found; XML changed?")
    text = text[:kf_start] + text[kf_end + len("</keyframe>"):]
    scene_xml = _SCENE_XML
    if not robot_table_collision:
        # MuJoCo collides two geoms iff (contype_a & conaffinity_b) or
        # (contype_b & conaffinity_a). Table and net move to bit 2 only; ball
        # and paddle_blade carry bits 1|2 so they still meet the table, the
        # floor and the robot; robot geoms (bit 1) no longer meet the table.
        scene_xml = scene_xml.replace('<geom name="table_top" type="box"',
                                      '<geom name="table_top" type="box" contype="2" conaffinity="2"', 1)
        scene_xml = scene_xml.replace('<geom name="net" type="box"',
                                      '<geom name="net" type="box" contype="2" conaffinity="2"', 1)
        # Ball: bits 1|2|4 -> meets robot (1), table/net (2), paddle (4).
        # Paddle blade: bit 4 only -> meets the ball and nothing else (no
        # propping the body up on the paddle against table or floor).
        scene_xml = scene_xml.replace('<geom name="ball" type="sphere"',
                                      '<geom name="ball" type="sphere" contype="7" conaffinity="7"', 1)
        text = text.replace('<geom name="paddle_blade" type="cylinder"',
                            '<geom name="paddle_blade" type="cylinder" contype="4" conaffinity="4"', 1)
    text = text.replace("</worldbody>", scene_xml + "  </worldbody>", 1)
    text = text.replace("</mujoco>", _contact_xml(paddle) + "</mujoco>", 1)
    return text


def build_model(fixed_base: bool = True, paddle: bool = True,
                robot_table_collision: bool = True) -> mujoco.MjModel:
    """Build the G1 + paddle + table + ball MuJoCo model.

    fixed_base=True (default) welds the pelvis in the standing pose above
    (required for TableTennisEnv — see module docstring); fixed_base=False
    keeps the vendor free-floating pelvis, for callers that want the
    unmodified G1 dynamics with the table-tennis scene grafted on.
    """
    return mujoco.MjModel.from_xml_string(_build_xml(fixed_base, paddle, robot_table_collision))


# ---------------------------------------------------------------------------
# Table targets: (zone, depth) -> explicit (x, y) point on the opponent's
# half (x > 0). depth is distance from the net (x), zone is lateral
# position (y). The robot hits with its right hand from the y<0 side, so
# "line" (down-the-line, same side) is y<0 and "crosscourt" (the long
# diagonal, the default/easiest shot) is y>0.
# ---------------------------------------------------------------------------
_TARGET_DEPTH_X = {"short": 0.35, "mid": 0.75, "deep": 1.15}
_TARGET_ZONE_Y = {"line": -0.55, "center": 0.0, "crosscourt": 0.55}

DEFAULT_PARAMS = {
    "target_zone": "crosscourt",
    "target_depth": "deep",
    "risk_preference": 0.45,
    "physics_jitter": 0.0,  # 0 = off; see TableTennisEnv._apply_physics_jitter
    # True (default): the robot's body collides with the table and net as always.
    # False: only the ball and paddle do -- the body passes through them, so a
    # free-standing robot cannot lean on the table and count as upright
    # (found 2026-09-01: balance v1 learned exactly that). Welded scene: no-op
    # in practice, kept default for bit-exact parity.
    "robot_table_collision": True,
}


def target_point(target_zone: str = "crosscourt", target_depth: str = "deep") -> tuple[float, float]:
    """(x, y) landing point on the opponent's half for a target_zone/target_depth pair."""
    return (_TARGET_DEPTH_X[target_depth], _TARGET_ZONE_Y[target_zone])


def landing_error_m(events: dict, target: tuple[str, str] | tuple[float, float]) -> float:
    """Euclidean distance (m) from events['landing_point'] to a target.

    ``target`` is either a (target_zone, target_depth) string pair (see
    target_point) or an already-resolved (x, y) point. Returns nan if
    events has no recorded landing_point.
    """
    landing = events.get("landing_point")
    if landing is None:
        return float("nan")
    if isinstance(target[0], str):
        tx, ty = target_point(*target)
    else:
        tx, ty = target
    lx, ly = landing
    return math.hypot(lx - tx, ly - ty)


def legal_return(events: dict) -> bool:
    """True iff the robot's paddle hit the ball, it then crossed the net
    plane above net height heading toward the opponent, and (the first
    such crossing after the hit) landed on the opponent's half.

    ``events`` is the flat per-episode dict TableTennisEnv accumulates
    (see TableTennisEnv docstring); crossed_net is only ever set True by
    the env for an outgoing (robot-side -> opponent-side) crossing that
    happens after contact_paddle is already True, so this reduces to the
    3-flag check below. See TableTennisEnv._update_events for exactly how
    each flag is set.
    """
    return bool(events.get("contact_paddle")) and bool(events.get("crossed_net")) and bool(
        events.get("bounce_opponent_side")
    )


# ---------------------------------------------------------------------------
# Serve sampling
# ---------------------------------------------------------------------------
# Ranges (documented for the eval report): launched from the opponent's
# half toward the robot's forehand (right = -y, see module docstring) zone.
SERVE_X0_RANGE = (1.00, 1.30)      # opponent-side launch x
SERVE_Y0_RANGE = (-0.20, 0.20)     # opponent-side launch y (near centre)
SERVE_Z0_HEIGHT_RANGE = (0.05, 0.25)  # launch height above the table (toss height)
SERVE_FLIGHT_TIME_RANGE = (0.45, 0.75)  # s, controls effective speed
SERVE_TARGET_X_RANGE = (-1.05, -0.25)  # robot-side arrival x (depth)
SERVE_TARGET_Y_RANGE = (-0.65, -0.10)  # robot-side arrival y (forehand/right, lateral)

# Overridable via params keys of the same name (serve_x0_range, ...) plus
# serve_scale (widens every range symmetrically about its centre) — see
# serve_ranges()/sample_serve(). Mirroring the constants above here means
# params={} stays behaviour-identical to before these keys existed.
DEFAULT_PARAMS.update({
    "serve_x0_range": list(SERVE_X0_RANGE),
    "serve_y0_range": list(SERVE_Y0_RANGE),
    "serve_z0_height_range": list(SERVE_Z0_HEIGHT_RANGE),
    "serve_flight_time_range": list(SERVE_FLIGHT_TIME_RANGE),
    "serve_target_x_range": list(SERVE_TARGET_X_RANGE),
    "serve_target_y_range": list(SERVE_TARGET_Y_RANGE),
    "serve_scale": 1.0,
})

_SERVE_RANGE_KEYS = (
    ("serve_x0_range", "x0", SERVE_X0_RANGE),
    ("serve_y0_range", "y0", SERVE_Y0_RANGE),
    ("serve_z0_height_range", "z0_height", SERVE_Z0_HEIGHT_RANGE),
    ("serve_flight_time_range", "flight_time", SERVE_FLIGHT_TIME_RANGE),
    ("serve_target_x_range", "target_x", SERVE_TARGET_X_RANGE),
    ("serve_target_y_range", "target_y", SERVE_TARGET_Y_RANGE),
)


def _scale_range(lo_hi: Sequence[float], scale: float) -> tuple[float, float]:
    """Widen (lo, hi) symmetrically about its centre. scale=1.0 short-circuits
    to the input unchanged (not recomputed via the centre/half arithmetic) so
    the no-override, no-scale path stays bit-exact with the old fixed ranges."""
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    if scale == 1.0:
        return (lo, hi)
    centre = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 * scale
    return (centre - half, centre + half)


def serve_ranges(params: dict | None = None) -> dict[str, tuple[float, float]]:
    """Effective (lo, hi) serve ranges sample_serve draws from, after param
    overrides and serve_scale widening — exposed so callers/tests can check
    bounds directly instead of reverse-engineering them from sampled pos/vel."""
    p = params or {}
    scale = float(p.get("serve_scale", 1.0))
    out: dict[str, tuple[float, float]] = {}
    for param_key, short, default in _SERVE_RANGE_KEYS:
        out[short] = _scale_range(p.get(param_key, default), scale)

    # target_y (lateral arrival point) must stay on the robot's forehand
    # side: widening can't push it across the centreline or off the table.
    ty_lo, ty_hi = out["target_y"]
    ty_lo = max(ty_lo, -TABLE_HALF_WIDTH)
    ty_hi = min(ty_hi, TABLE_HALF_WIDTH)
    ty_base_hi = float(p.get("serve_target_y_range", SERVE_TARGET_Y_RANGE)[1])
    if ty_base_hi <= 0.0:  # base range already forehand-side (y<=0)
        ty_hi = min(ty_hi, 0.0)
    out["target_y"] = (ty_lo, ty_hi)
    return out


def sample_serve(rng: np.random.Generator, params: dict | None = None) -> dict:
    """Sample a serve: a straight ballistic (no aerodynamic drag/Magnus)
    trajectory solved to land at a randomized point in the robot's forehand
    zone at a randomized flight time, launched from a randomized point/height
    on the opponent's half. Returns {"pos": (3,), "vel": (3,), "spin": None}.

    Ranges come from serve_ranges(params) — params={} (or None) reproduces
    the SERVE_*_RANGE constants above unchanged; see serve_ranges() for the
    override/serve_scale keys.

    Spin is not modelled aerodynamically in this pass (no Magnus force in
    plain MuJoCo without a custom force callback), so it is always None —
    documented simplification, not a missing feature of the ball body
    itself (which does have full 6-dof free-joint dynamics and would spin
    correctly on contact regardless).
    """
    ranges = serve_ranges(params)
    g = 9.81
    x0 = rng.uniform(*ranges["x0"])
    y0 = rng.uniform(*ranges["y0"])
    z0 = TABLE_TOP_Z + rng.uniform(*ranges["z0_height"])
    t = rng.uniform(*ranges["flight_time"])
    xt = rng.uniform(*ranges["target_x"])
    yt = rng.uniform(*ranges["target_y"])
    zt = TABLE_TOP_Z + BALL_RADIUS  # arrives just touching the table (the serve bounce)

    vx = (xt - x0) / t
    vy = (yt - y0) / t
    vz = (zt - z0 + 0.5 * g * t * t) / t

    return {"pos": np.array([x0, y0, z0]), "vel": np.array([vx, vy, vz]), "spin": None}


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
CONTROL_HZ = 50
SIM_DT = 0.002
DECIMATION = 10  # 0.002 * 10 = 0.02 s = 1 / 50 Hz


def _empty_events(free_base: bool = False) -> dict:
    """Fresh per-episode event dict.

    free_base=True adds the single extra key ``"fell"``; the default is the
    welded scene's dict, unchanged key for key (himrobot.eval.evaluate writes it
    straight into evaluator.json's per_trial, which is pinned bit-for-bit).
    """
    return {
        "contact_paddle": False,
        "bounce_own_side": False,
        "bounce_opponent_side": False,
        "net_hit": False,
        "out": False,
        "crossed_net": False,
        "landing_point": None,
        # Extra (not in the minimum contract, additive): ms from serve to
        # the paddle first moving > 2 cm from its reset pose, or None.
        "reaction_latency_ms": None,
        # Extra (additive): sampled physics_jitter multipliers for this
        # episode (dict, see TableTennisEnv._apply_physics_jitter), or None
        # when physics_jitter is off (default).
        "physics": None,
        # Free base only (absent entirely on the welded scene): latched True
        # once the pelvis fell below FALL_PELVIS_Z. Diagnostic; scoring is
        # untouched. See the module docstring's free-base contract.
        **({"fell": False} if free_base else {}),
    }


class TableTennisEnv:
    """G1 + paddle vs. a served ball. See module docstring for the
    table/robot layout and himrobot.policies for the Policy interface this is
    meant to be driven by.

    fixed_base=True (the default, and the project's scored configuration)
    welds the pelvis: the robot cannot fall and every output is exactly what
    it has always been. fixed_base=False keeps the vendor pelvis free joint
    so the robot must stand on its own; see the module docstring's free-base
    contract for the reset pose, the four extra ``base.*`` observation keys
    and the ``"fell"`` event that mode adds.
    """

    def __init__(self, seed: int, params: dict | None = None, fixed_base: bool = True):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self._rng = np.random.default_rng(seed)
        self.fixed_base = bool(fixed_base)
        self.model = build_model(fixed_base=self.fixed_base, paddle=True,
                                 robot_table_collision=bool(self.params["robot_table_collision"]))
        self.data = mujoco.MjData(self.model)
        self.control_hz = CONTROL_HZ
        self.decimation = DECIMATION
        assert abs(self.model.opt.timestep - SIM_DT) < 1e-9

        self._joint_qpos_adr = np.array(
            [self.model.jnt_qposadr[self._jid(n)] for n in JOINT_ORDER]
        )
        self._joint_dof_adr = np.array(
            [self.model.jnt_dofadr[self._jid(n)] for n in JOINT_ORDER]
        )
        self._actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_ORDER]
        )
        assert np.all(self._actuator_ids >= 0)

        ball_jid = self._jid("ball_joint")
        self._ball_qpos_adr = self.model.jnt_qposadr[ball_jid]  # start of 7 (pos3+quat4)
        self._ball_dof_adr = self.model.jnt_dofadr[ball_jid]  # start of 6 (lin3+ang3)

        # Free base only: the pelvis free joint's qpos (3 pos + 4 wxyz quat)
        # and qvel (3 linear + 3 angular) slices, plus the pelvis body whose
        # world height the fall latch watches. Left None on the welded scene,
        # where the joint does not exist at all.
        self._base_qpos_adr: int | None = None
        self._base_dof_adr: int | None = None
        self._pelvis_body_id: int | None = None
        if not self.fixed_base:
            base_jid = self._jid("floating_base_joint")
            self._base_qpos_adr = int(self.model.jnt_qposadr[base_jid])
            self._base_dof_adr = int(self.model.jnt_dofadr[base_jid])
            self._pelvis_body_id = self._body("pelvis")

        self._paddle_body_id = self._body("paddle")
        self._ball_body_id = self._body("ball")
        self._ball_geom_id = self._geom("ball")
        self._table_geom_id = self._geom("table_top")
        self._net_geom_id = self._geom("net")
        self._paddle_geom_id = self._geom("paddle_blade")
        self._floor_geom_id = self._geom("floor")

        # Nominal (unjittered) contact/friction/mass values, snapshotted once
        # so physics_jitter scales from a fixed baseline every reset instead
        # of compounding episode over episode. pair_solref is the ball-table
        # / ball-paddle restitution-tuned params (BOUNCE_SOLREF, see module
        # header); geom_friction is the ball/paddle/table CONTACT_FRICTION rows.
        bt_pid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_PAIR, "ball_table_pair")
        bp_pid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_PAIR, "ball_paddle_pair")
        assert bt_pid >= 0 and bp_pid >= 0
        self._jitter_pair_ids = np.array([bt_pid, bp_pid])
        self._jitter_geom_ids = np.array([self._ball_geom_id, self._paddle_geom_id, self._table_geom_id])
        self._nominal_pair_solref = self.model.pair_solref[self._jitter_pair_ids].copy()
        self._nominal_geom_friction = self.model.geom_friction[self._jitter_geom_ids].copy()
        self._nominal_ball_mass = float(self.model.body_mass[self._ball_body_id])

        self._events: dict = _empty_events(free_base=not self.fixed_base)
        self._paddle_reset_pos: np.ndarray | None = None
        self._prev_ball_x: float | None = None
        self._prev_touching_table = False
        self._elapsed_s = 0.0

    def _jid(self, name: str) -> int:
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert j >= 0, name
        return j

    def _body(self, name: str) -> int:
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert b >= 0, name
        return b

    def _geom(self, name: str) -> int:
        g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert g >= 0, name
        return g

    def _apply_physics_jitter(self) -> dict | None:
        """Domain-randomize ball/paddle/table contact params + ball mass by
        +-physics_jitter (opt-in via params['physics_jitter'], default 0.0 =
        off, model left untouched). Scales from the snapshotted nominal
        values (not the current ones) so repeated resets don't compound."""
        j = float(self.params.get("physics_jitter", 0.0))
        if j <= 0.0:
            return None
        contact_mult = float(self._rng.uniform(1.0 - j, 1.0 + j))
        friction_mult = float(self._rng.uniform(1.0 - j, 1.0 + j))
        mass_mult = float(self._rng.uniform(1.0 - j, 1.0 + j))
        self.model.pair_solref[self._jitter_pair_ids] = self._nominal_pair_solref * contact_mult
        self.model.geom_friction[self._jitter_geom_ids] = self._nominal_geom_friction * friction_mult
        self.model.body_mass[self._ball_body_id] = self._nominal_ball_mass * mass_mult
        return {"contact_mult": contact_mult, "friction_mult": friction_mult, "mass_mult": mass_mult}

    def reset(self, serve: dict | None = None) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._joint_qpos_adr] = STAND_POSE
        self.data.ctrl[self._actuator_ids] = STAND_POSE

        if not self.fixed_base:
            # Stand the robot exactly where the welded scene bolts it.
            a = self._base_qpos_adr
            self.data.qpos[a:a + 3] = BASE_NOMINAL_POS
            self.data.qpos[a + 3:a + 7] = BASE_NOMINAL_QUAT
            v = self._base_dof_adr
            self.data.qvel[v:v + 6] = 0.0

        self._events = _empty_events(free_base=not self.fixed_base)
        self._events["physics"] = self._apply_physics_jitter()

        if serve is None:
            serve = sample_serve(self._rng, self.params)
        pos = np.asarray(serve["pos"], dtype=np.float64)
        vel = np.asarray(serve["vel"], dtype=np.float64)
        spin = serve.get("spin")

        self.data.qpos[self._ball_qpos_adr:self._ball_qpos_adr + 3] = pos
        self.data.qpos[self._ball_qpos_adr + 3:self._ball_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._ball_dof_adr:self._ball_dof_adr + 3] = vel
        if spin is not None:
            self.data.qvel[self._ball_dof_adr + 3:self._ball_dof_adr + 6] = spin

        mujoco.mj_forward(self.model, self.data)

        self._paddle_reset_pos = self.data.xpos[self._paddle_body_id].copy()
        self._prev_ball_x = float(self.data.qpos[self._ball_qpos_adr])
        self._prev_touching_table = False
        self._elapsed_s = 0.0

        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, dict]:
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (NUM_JOINTS,), action.shape
        self.data.ctrl[self._actuator_ids] = action

        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
            self._elapsed_s += SIM_DT
            self._update_events()

        return self._get_obs(), dict(self._events)

    def _get_obs(self) -> dict:
        d = self.data
        obs = {
            "robot.joint_position": d.qpos[self._joint_qpos_adr].copy(),
            "robot.joint_velocity": d.qvel[self._joint_dof_adr].copy(),
            "ball.position": d.qpos[self._ball_qpos_adr:self._ball_qpos_adr + 3].copy(),
            "ball.velocity": d.qvel[self._ball_dof_adr:self._ball_dof_adr + 3].copy(),
            "paddle.position": d.xpos[self._paddle_body_id].copy(),
            "paddle.quaternion": d.xquat[self._paddle_body_id].copy(),
            "time_s": self._elapsed_s,
        }
        if not self.fixed_base:
            # Free base only: the welded scene must stay unchanged key for
            # key, so a policy can treat "these keys exist" as "the pelvis
            # can move" (see himrobot.policies.ik_hitter's base-aware IK).
            a, v = self._base_qpos_adr, self._base_dof_adr
            obs["base.position"] = d.qpos[a:a + 3].copy()
            obs["base.orientation"] = d.qpos[a + 3:a + 7].copy()  # wxyz
            obs["base.linear_velocity"] = d.qvel[v:v + 3].copy()
            obs["base.angular_velocity"] = d.qvel[v + 3:v + 6].copy()
        return obs

    def _update_events(self) -> None:
        d = self.data
        ev = self._events

        # Free base only: latch a fall the first time the pelvis drops below
        # FALL_PELVIS_Z. The episode deliberately keeps running (see the
        # module docstring) — this records what happened, it does not score it.
        if not self.fixed_base and not ev["fell"] and float(d.xpos[self._pelvis_body_id][2]) < FALL_PELVIS_Z:
            ev["fell"] = True

        # Reaction latency: first time the paddle has moved > 2 cm from its
        # reset pose.
        if ev["reaction_latency_ms"] is None:
            moved = np.linalg.norm(d.xpos[self._paddle_body_id] - self._paddle_reset_pos)
            if moved > 0.02:
                ev["reaction_latency_ms"] = self._elapsed_s * 1000.0

        # Contact classification for this substep.
        touching_table = False
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self._ball_geom_id:
                other = g2
            elif g2 == self._ball_geom_id:
                other = g1
            else:
                continue
            if other == self._paddle_geom_id:
                ev["contact_paddle"] = True
            elif other == self._net_geom_id:
                ev["net_hit"] = True
            elif other == self._floor_geom_id:
                ev["out"] = True
            elif other == self._table_geom_id:
                touching_table = True

        ball_x = float(d.qpos[self._ball_qpos_adr])
        ball_y = float(d.qpos[self._ball_qpos_adr + 1])

        # Table bounce: rising edge of "touching table_top".
        if touching_table and not self._prev_touching_table:
            if ball_x < 0:
                ev["bounce_own_side"] = True
            else:
                ev["bounce_opponent_side"] = True
            ev["landing_point"] = (ball_x, ball_y)
        self._prev_touching_table = touching_table

        # crossed_net: outgoing (robot -> opponent) crossing of x=0 while
        # above net height, only counted once a paddle hit has occurred
        # (see legal_return docstring).
        if ev["contact_paddle"] and self._prev_ball_x is not None and self._prev_ball_x < 0.0 <= ball_x:
            ball_z = float(d.qpos[self._ball_qpos_adr + 2])
            if ball_z - BALL_RADIUS > NET_TOP_Z:
                ev["crossed_net"] = True
        self._prev_ball_x = ball_x
