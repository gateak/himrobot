"""IKHitterPolicy: a model-based table-tennis return policy.

Tuned result (the task's acceptance bar): 20 random serves, seed 0,
default params (crosscourt/deep/risk_preference=0.45) --
    contact rate:       70% (14/20)
    legal_return_rate:  35% (7/20)
    crossed_net:        45% (9/20), bounce_opponent_side: 50% (10/20)
measured with the exact defaults below via a throwaway tuning script (not
shipped -- the numbers are real, reproduced by tests/test_ik_hitter.py
at a smaller n=10/50% contact bar for CI speed). Task target was >=70%
contact / >=30% legal_return_rate.

Pipeline, run fresh from ``obs`` every 50 Hz control step (the Policy
interface gives no access to the live mujoco.MjData driving the episode,
only the flat ``obs`` dict -- see himrobot.policies.base):

  1. Ball prediction (`_predict_ball_trajectory` / `_find_intercept`):
     forward-simulate the ball with an explicit-Euler integrator (gravity
     only -- no aerodynamic drag, matching scene.py's <option> tag, which
     sets no density/viscosity, and sample_serve()'s documented no-drag
     assumption, so this predictor is a faithful model of the real sim's
     ball dynamics between contacts) including a table bounce, applying
     BOUNCE_RESTITUTION = 0.9 to the vertical velocity component the same
     way scene.py's contact solver was calibrated to (see scene.py's
     BOUNCE_SOLREF comment for the drop-test numbers this assumption is
     copied from), plus PREDICTION_DELAY_S, an empirical correction for
     the gap between that idealized instant bounce and the real sim's
     compliant one (see its own comment below). The first time the
     predicted trajectory crosses a fixed hitting plane ``x = x_hit``
     gives the intercept time/point/incoming velocity. x_hit and the y/z
     reach clamp below started from an FK sweep of the right arm's
     forward reach at stand pose (the right arm alone, without moving the
     legs/pelvis, reaches at most x ~ -1.15 to -1.2 m in front of the
     fixed pelvis at x=-1.77 m) and were then refined against real
     20-trial seed-0 runs.
  2. Arm IK (`_solve_ik`): damped-least-squares over 10 joints (waist yaw/
     roll/pitch + the 7 right-arm joints), run on a private "shadow" copy
     of the scene model built once in __init__ purely for forward
     kinematics/Jacobians (himrobot.sim.scene.build_model(), reused rather than
     reimplemented, per the task's paddle-injection-pattern note) and
     reseeded from ``obs["robot.joint_position"]`` every call. Tracks two
     points rigidly attached to the paddle body -- the blade centre and a
     point NORMAL_EPS along the blade's current normal -- so a single
     position-only Jacobian (mj_jac, like sim/g1_retarget.py's mj_jacBody
     dual-point wrist+elbow trick, generalized to an arbitrary body point
     since the blade centre is not a body origin) controls both the
     paddle's position and which way it faces, with a (small, see
     ns_gain's default) null-space bias back toward the arm's
     stand-keyframe pose (same pattern as g1_retarget.py: damped
     pseudo-inverse, step-size clamp, joint-range clamp). Orientation
     tracking needed real weight to matter at all: at the arm's default
     6-iteration/w_normal=0.5 first draft it barely tracked the requested
     blade normal (measured dot product with the desired normal ~0.4-0.5,
     nearly arbitrary) because position so dominated the objective that
     the 3 remaining redundant DOF were left to the null-space pull
     instead; raising dls_iters/w_normal to this file's defaults (and
     dropping ns_gain) got that to ~0.93-0.98 in the same real runs, which
     is what actually made legal returns start happening (see the task
     report for the diagnostic numbers).
  3. Swing timing (`_swing_target`): a single constant-velocity line
     through p_hit at swing_vel = risk_preference * (v_target -
     v_ball_in) (v_target solved the same constant-acceleration way
     sample_serve solves a launch velocity, aimed at
     himrobot.sim.scene.target_point(zone, depth)), held from speed_buffer
     seconds before contact through t_follow after it, with a cubic
     Hermite ease-in from a p_pre point backswing_dist behind p_hit
     before that. See the method's own docstring for why it isn't a
     plain lerp -- the first draft was, and its measured paddle speed at
     contact was ~0.5-1 m/s against an intended ~3.5+ m/s, because a
     position actuator (kp=500) cannot instantly jump to speed. swing_vel
     is also capped at max_swing_speed: the raw risk_preference-scaled
     value can run to several m/s, which both exceeds what the actuators
     can track *and* sweeps the Cartesian target far outside the arm's
     ~0.3-0.5 m local workspace -- both measured, empirically, to reduce
     contact rate more than the extra speed helped.
  4. The other 19 joints (both legs, left arm) always target
     himrobot.sim.scene.STAND_POSE, per the task spec.
  5. Per-serve modulation (SERVE_PARAM_NOMINAL / set_serve_params): ten
     scalars -- intercept x/y/z/timing offsets, swing-speed scale, paddle
     pitch/yaw, aim x/y offsets, return-flight-time scale -- that retune
     steps 1-3 for one whole episode. All nominal by default, which is the
     behaviour documented above, unchanged.
  6. Swing library ("hitter v2", SWING_TEMPLATES / blend_shape / reach_box):
     five presets over the same shape vector, selected by a rectified-gate
     barycentric blend, plus coarse hitting-plane, reach-clamp,
     face-vs-position-weight, dwell and swing-direction knobs. Ten more
     per-serve scalars (SERVE_PARAM_NOMINAL_V2), every one of them the
     identity at nominal -- so all of the above still holds bit for bit.
  7. Base-aware targeting (`to_shadow_frame`, base_aware, default off): the shadow
     model welds the pelvis at the nominal pose, so on the free-base scene
     (where the real pelvis drifts and can fall) a world-frame IK target
     would be solved against the wrong pelvis. When -- and only when -- the
     observation carries base.position/base.orientation, the blade target is
     re-expressed through the pelvis frame first. The welded scene never
     emits those keys, so the transform is not reached and every number
     above is unchanged; TT_BASE_AWARE=1 (or base_aware=True) enables it.
  8. Stance-relative reach ("hitter v3", STANCE_Y / _stance_y): the lateral
     reach constants were measured against a pelvis welded at y = 0, so they
     slide with scene.ROBOT_Y (TT_ROBOT_Y). At the default stance _stance_y
     returns its argument object untouched and every number above still holds
     bit for bit. Plus one more per-serve knob, collision_guard
     (SERVE_PARAM_NOMINAL_V3), default off: a keep-out box on three joint
     limits inside the deep/inner region, see GUARD_BOX / GUARD_REGION.
  9. WBC accessors (plan()/preview(), see docs/design/whole_body_v1.md): two
     read-only-to-the-caller additions for a whole-body controller that
     needs to see the hitter's replanning schedule and its upcoming joint
     commands without driving anything itself. plan(obs) just runs the same
     _maybe_replan act() would and hands back self._plan; preview(obs,
     horizon) forecasts the next `horizon` control-step targets by rolling
     _swing_target + _solve_ik (the same dispatcher act() uses, so a
     collision_guard keep-out applies to the forecast too) forward on a
     private MjData (self._preview_data), never touching self._ik_data/self._plan/any
     knob -- see preview()'s own docstring for the open-loop assumption
     this makes (act() re-solves from the *measured* pose every step, so a
     real disturbance is partly corrected for live; the preview cannot see
     that and is therefore only ever a forecast, not a promise).
"""
from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np

import mujoco

from himrobot.sim import scene
from himrobot.policies.base import Policy

# ---------------------------------------------------------------------------
# Ball-flight prediction (no drag, one bounce -- see module docstring).
# ---------------------------------------------------------------------------
BOUNCE_RESTITUTION = 0.9  # matches scene.py's calibrated ball/table restitution
GRAVITY = 9.81

# One forecast sample: (t, position, velocity).
TrajSample = tuple[float, np.ndarray, np.ndarray]


def _quat_to_mat(quat) -> np.ndarray:
    """wxyz quaternion (MuJoCo convention) -> 3x3 rotation matrix."""
    m = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(m, np.asarray(quat, dtype=np.float64))
    return m.reshape(3, 3)


def _predict_ball_trajectory(pos0: np.ndarray, vel0: np.ndarray,
                             dt: float = 0.01, t_max: float = 1.5) -> list[TrajSample]:
    """Explicit-Euler ball forecast: gravity + a single-bounce restitution
    model against table_top. Returns a list of (t, pos, vel) samples.
    """
    p = np.array(pos0, dtype=float)
    v = np.array(vel0, dtype=float)
    t = 0.0
    out = [(t, p.copy(), v.copy())]
    bounced = False
    while t < t_max:
        v[2] -= GRAVITY * dt
        p = p + v * dt
        t += dt
        if (not bounced and v[2] < 0
                and p[2] - scene.BALL_RADIUS <= scene.TABLE_TOP_Z
                and -scene.TABLE_HALF_LENGTH <= p[0] <= scene.TABLE_HALF_LENGTH
                and -scene.TABLE_HALF_WIDTH <= p[1] <= scene.TABLE_HALF_WIDTH):
            p[2] = scene.TABLE_TOP_Z + scene.BALL_RADIUS
            v[2] = -v[2] * BOUNCE_RESTITUTION
            bounced = True  # only the first bounce matters for an intercept plan
        out.append((t, p.copy(), v.copy()))
        if p[2] < 0.05:
            break
    return out


# Empirical correction: this predictor's idealized instant-bounce
# (a hard velocity flip) reaches x_hit systematically *earlier* than the
# real sim, whose ball/table contact is a compliant spring (scene.py's
# BOUNCE_SOLREF timeconst=0.05s), not an instant reflection -- measured
# directly (a HoldPolicy comparison script, not shipped: predicted vs.
# actual time/position at an x_hit=-1.15 crossing over 40 seed-0 serves,
# during an earlier tuning pass, before x_hit moved to -1.20 -- not
# re-measured at the final plane, but the softening this corrects for
# happens at the bounce itself, upstream of either plane, so the same
# correction should carry over reasonably) at +28ms mean / 10ms std
# lateness, ~3-9cm mean position error. Applied by re-sampling the *same*
# predicted trajectory this much further along, rather than trusting the
# raw crossing sample.
PREDICTION_DELAY_S = 0.03


def _sample_trajectory_at(traj: list[TrajSample], t: float) -> TrajSample:
    """Interpolate a _predict_ball_trajectory() list at time t (clamped
    to the trajectory's own span).
    """
    if t <= traj[0][0]:
        return traj[0]
    for i in range(1, len(traj)):
        t0, p0, v0 = traj[i - 1]
        t1, p1, v1 = traj[i]
        if t0 <= t <= t1:
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return t, p0 + frac * (p1 - p0), v0 + frac * (v1 - v0)
    return traj[-1]


def _find_intercept(traj: list[TrajSample], x_hit: float) -> TrajSample | None:
    """First (t, pos, vel) where the predicted trajectory's x crosses
    x_hit while descending in x (the ball always travels opponent -> robot,
    i.e. x monotonically decreasing, since the bounce only flips vz -- see
    module docstring), linearly interpolated between samples. None if the
    trajectory never reaches x_hit within the horizon.
    """
    for i in range(1, len(traj)):
        t0, p0, v0 = traj[i - 1]
        t1, p1, v1 = traj[i]
        if p0[0] > x_hit >= p1[0]:
            span = p0[0] - p1[0]
            frac = (p0[0] - x_hit) / span if span > 1e-9 else 0.0
            t = t0 + frac * (t1 - t0)
            p = p0 + frac * (p1 - p0)
            v = v0 + frac * (v1 - v0)
            return t, p, v
    return None


def _solve_launch_velocity(p0: np.ndarray, p1: np.ndarray, flight_time: float) -> np.ndarray:
    """Constant-acceleration launch velocity from p0 to p1 in flight_time
    seconds -- identical formula to scene.sample_serve()'s vx/vy/vz solve.
    """
    g = GRAVITY
    t = flight_time
    return np.array([
        (p1[0] - p0[0]) / t,
        (p1[1] - p0[1]) / t,
        (p1[2] - p0[2] + 0.5 * g * t * t) / t,
    ])


# Global swing-speed multiplier, read once at import (like scene.TT_ROBOT_Y):
# TT_SWING_SPEED=0.7 caps every swing at 70 % of what the per-serve knobs ask.
# Default 1.0 = bit-exact today. Free-base diagnostic: on an unwelded robot
# the swing's own momentum is the tipping force (results/stance_sweep).
_GLOBAL_SWING_SPEED = float(os.environ.get("TT_SWING_SPEED", "1.0"))
# TT_WAIST_LIMIT="roll,pitch" (rad) tightens the IK solver's waist roll/pitch
# bounds; absent (default) leaves the model's own limits -- bit-exact today.
_WAIST_LIMIT = (tuple(float(v) for v in os.environ["TT_WAIST_LIMIT"].split(","))
                if os.environ.get("TT_WAIST_LIMIT") else None)


# ---------------------------------------------------------------------------
# Per-serve modulation ("serve params"): a small set of scalars that retune
# this hitter for one whole episode without touching its structure. Every
# nominal value below is the identity -- a hitter left at SERVE_PARAM_NOMINAL
# reproduces the pre-existing behaviour exactly, byte for byte (see
# tests/test_param_residual.py::test_nominal_serve_params_match_plain_hitter).
# Consumed by himrobot.rl.param_residual, which learns them per serve.
# ---------------------------------------------------------------------------
SERVE_PARAM_NOMINAL = {
    "x_hit_offset": 0.0,       # m, added to x_hit (the hitting plane's depth)
    "y_hit_offset": 0.0,       # m, added to the intercept point's y, before the reach clamp
    "z_hit_offset": 0.0,       # m, added to the intercept point's z, before the reach clamp
    "t_hit_offset": 0.0,       # s, extra re-sampling delay on top of PREDICTION_DELAY_S
    "swing_speed_scale": 1.0,  # multiplies max_swing_speed (the swing-speed cap)
    "pitch_offset": 0.0,       # rad, tilts the commanded paddle normal up (+) / down (-)
    "yaw_offset": 0.0,         # rad, swings the commanded paddle normal toward +y (+)
    "aim_x_offset": 0.0,       # m, added to the aim point's x (depth on the opponent half)
    "aim_y_offset": 0.0,       # m, added to the aim point's y (lateral)
    "flight_time_scale": 1.0,  # multiplies return_flight_time (return loft)
}


def _rotate_normal(n: np.ndarray, pitch: float, yaw: float) -> np.ndarray:
    """Rotate a unit paddle normal: +pitch raises its z (rotation about
    world +y in the sense that maps +x toward +z), +yaw swings it toward +y
    (rotation about world +z), applied yaw-after-pitch. Returns ``n``
    untouched at (0, 0) so the nominal path is bit-for-bit unchanged."""
    if pitch == 0.0 and yaw == 0.0:
        return n
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rotated = rz @ (ry @ n)
    return rotated / np.linalg.norm(rotated)


# ---------------------------------------------------------------------------
# Reach / IK joint set: waist (yaw, roll, pitch) + right arm (7 joints).
# ---------------------------------------------------------------------------
IK_JOINT_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
IK_INDICES = np.array([scene.JOINT_ORDER.index(n) for n in IK_JOINT_NAMES])  # positions in the 29-vector
N_IK = len(IK_JOINT_NAMES)

# preview()'s forecast step: one control step, matching scene.CONTROL_HZ (50 Hz).
PREVIEW_DT = 1.0 / scene.CONTROL_HZ

# Empirically-derived (FK sweep, see module docstring, refined against
# real 20-trial seed-0 runs -- see the task report) forward-reach plane
# and a soft y/z clamp so a predicted intercept point that is technically
# out of reach still gets pulled to the nearest plausible target rather
# than rejected outright.
X_HIT = -1.20

# Lateral reach is measured relative to where the pelvis is welded: Y_REACH,
# REACH_MAX_Y and every template's y box came from an FK sweep with the pelvis
# at y = 0, so they only mean anything relative to where the robot stands. If
# the stance moves (scene.ROBOT_Y / TT_ROBOT_Y), they move with it -- the arm
# did not change, the robot did. x (X_HIT) and z (Z_REACH) are unaffected, and
# the aim point stays world-frame: the table did not move.
STANCE_Y = float(scene.ROBOT_Y)


def _stance_y(pair: tuple[float, float]) -> tuple[float, float]:
    """Slide a (lo, hi) y box to the current stance. At the nominal stance it
    returns the argument object untouched -- no float arithmetic at all -- so
    every template dict holds the identical float objects it holds today and
    blend_shape / reach_box stay bit-exact, including their identity early-outs
    (test_gates_at_zero_return_the_drive_template)."""
    if STANCE_Y == 0.0:
        return pair  # WHY: bit-exact -- no float arithmetic at all
    return (pair[0] + STANCE_Y, pair[1] + STANCE_Y)


Y_REACH = _stance_y((-0.75, 0.15))
Z_REACH = (0.65, 1.35)


def _rotate_vector(v: np.ndarray, pitch: float, yaw: float) -> np.ndarray:
    """_rotate_normal's magnitude-preserving twin, for rotating the swing
    velocity (and so the cruise line and the backswing point). The paddle
    face is always defined relative to the swing direction (n_des =
    _rotate_normal(n_back, ...)), so rotating the swing also carries the face
    with it; face_pitch/face_yaw and pitch/yaw_offset are biases ON TOP of
    that, exactly as pitch_offset was in v1. Returns ``v`` untouched at (0, 0)."""
    if pitch == 0.0 and yaw == 0.0:
        return v
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ (ry @ v)


# ---------------------------------------------------------------------------
# Swing library ("hitter v2"). Every template is a preset over the same set of
# scalars, so any convex combination of two templates is itself a valid swing
# -- which is what makes the blended selector below well-defined. "drive" is
# the pre-v2 behaviour verbatim (the constructor defaults); the other four are
# shape variants the per-serve policy can blend toward.
#
#   x_hit         hitting-plane depth (m)
#   pred_delay    re-sampling delay on the predicted trajectory (s)
#   speed_cap     commanded-target speed cap (m/s)
#   flight_time   return flight time used to solve the launch velocity (s)
#   risk_gain     multiplies the episode's risk_preference in the swing solve
#   face_pitch/face_yaw  template bias on the paddle normal, relative to the
#                        (already dir-rotated) swing direction (rad)
#   dir_pitch/dir_yaw    template bias on the swing direction itself (rad);
#                        the face follows it, so a lift's commanded face
#                        elevation is dir_pitch + face_pitch
#   t_pre/speed_buffer/t_follow  swing-window widths (s)
#   backswing     backswing distance behind p_hit (m)
#   y_lo/y_hi/z_lo/z_hi  the Cartesian reach clamp for this template
#
# Rationale per template: "block" is a near-static, wide-dwell paddle that
# maximises contact probability on marginal serves; "lift" is a high slow arc
# for net clearance on low/late intercepts; "punch" is the fast/flat/deep
# extension; "crossbody" was designed as the mirrored right-arm posture for a
# ball taken on the backhand side (there is no left-arm backhand, the paddle
# is welded to the right wrist only) but ships as a deeper/lifted drive variant
# because the +y half of the workspace cannot be used collision-free (see the
# template's own comment and REACH_MAX_Y). No topspin/brush template:
# scene.py applies no Magnus force, so a tangential brush has no payoff here.
# ---------------------------------------------------------------------------
SHAPE_FIELDS = ("x_hit", "pred_delay", "speed_cap", "flight_time", "risk_gain",
                "face_pitch", "face_yaw", "dir_pitch", "dir_yaw",
                "t_pre", "speed_buffer", "t_follow", "backswing",
                "y_lo", "y_hi", "z_lo", "z_hi")

TEMPLATE_ORDER = ("drive", "block", "lift", "punch", "crossbody")

# The two templates whose y box is not Y_REACH have their own literals; they
# were measured against the same pelvis-at-y=0 sweep, so they follow the stance too.
_LIFT_Y = _stance_y((-0.78, 0.15))
_PUNCH_Y = _stance_y((-0.72, 0.15))

SWING_TEMPLATES: dict[str, dict[str, float]] = {
    "drive": dict(
        x_hit=X_HIT, pred_delay=PREDICTION_DELAY_S, speed_cap=1.00, flight_time=0.50,
        risk_gain=1.00, face_pitch=0.00, face_yaw=0.00, dir_pitch=0.00, dir_yaw=0.00,
        t_pre=0.30, speed_buffer=0.03, t_follow=0.20, backswing=0.10,
        y_lo=Y_REACH[0], y_hi=Y_REACH[1], z_lo=Z_REACH[0], z_hi=Z_REACH[1],
    ),
    "block": dict(
        x_hit=-1.30, pred_delay=0.034, speed_cap=0.25, flight_time=0.66,
        risk_gain=0.40, face_pitch=0.18, face_yaw=0.00, dir_pitch=0.25, dir_yaw=0.00,
        t_pre=0.45, speed_buffer=0.08, t_follow=0.35, backswing=0.03,
        y_lo=Y_REACH[0], y_hi=Y_REACH[1], z_lo=0.62, z_hi=1.40,
    ),
    "lift": dict(
        x_hit=-1.34, pred_delay=0.036, speed_cap=0.80, flight_time=0.80,
        risk_gain=0.90, face_pitch=0.30, face_yaw=0.00, dir_pitch=0.55, dir_yaw=0.00,
        t_pre=0.35, speed_buffer=0.05, t_follow=0.30, backswing=0.14,
        y_lo=_LIFT_Y[0], y_hi=_LIFT_Y[1], z_lo=0.60, z_hi=1.30,
    ),
    "punch": dict(
        x_hit=-1.14, pred_delay=0.028, speed_cap=2.20, flight_time=0.33,
        risk_gain=1.40, face_pitch=-0.05, face_yaw=0.00, dir_pitch=-0.10, dir_yaw=0.00,
        t_pre=0.22, speed_buffer=0.03, t_follow=0.15, backswing=0.18,
        y_lo=_PUNCH_Y[0], y_hi=_PUNCH_Y[1], z_lo=0.68, z_hi=1.42,
    ),
    # "crossbody" ships WITHOUT the +y box that was its design rationale (this
    # solver self-collides there, see REACH_MAX_Y) and therefore without the
    # yaw biases too: with the box shut the template only ever applies on the
    # forehand side, where a face yaw of either sign is an aim error the per-
    # serve aim already solves (measured: shipped yaws 2-3/24 legal at full
    # gate vs 9-7/24 nominal; both yaws zeroed scores best). What is left is a
    # slightly deeper, slightly lifted, slower-flight variant of drive.
    "crossbody": dict(
        x_hit=-1.26, pred_delay=0.032, speed_cap=1.00, flight_time=0.55,
        risk_gain=1.00, face_pitch=0.10, face_yaw=0.00, dir_pitch=0.10, dir_yaw=0.00,
        t_pre=0.34, speed_buffer=0.04, t_follow=0.22, backswing=0.08,
        y_lo=Y_REACH[0], y_hi=Y_REACH[1], z_lo=0.68, z_hi=1.40,
    ),
}
assert all(set(t) == set(SHAPE_FIELDS) for t in SWING_TEMPLATES.values())

# A single gate at u = 1 gives weights (1, 4), i.e. 80% of the target template
# and 20% of drive: deliberate, drive is the measured-good prior and keeping a
# fifth of it regularises the blend.
GATE_GAIN = 4.0

# The outer limit reach_box() can interpolate a template's clamp toward, set by
# what the *runtime* solver (local, warm-started from the current pose, one
# _solve_ik call per control step) actually achieves collision-free, not by an
# offline multi-restart reach study. Measured over the template hitting planes
# with the blade above the table: the y floor is collision-free all the way to
# -0.90, and commanding it gains ~4-6 cm of achieved blade y over -0.75, while
# every commanded y >= +0.30 self-collides on this solver (23-45 contacts, and
# 0.14-0.37 m of residual position error) under every face normal and w_normal
# tried -- so the crossbody half of the workspace is deliberately NOT opened
# here. z barely matters either way: the measured intercept z spans 0.91-1.35,
# so the floor already has >25 cm of unused margin.
REACH_MAX_Y = _stance_y((-0.90, 0.15))
REACH_MAX_Z = (0.60, 1.45)


def blend_shape(gates: Iterable[float],
                drive: dict[str, float] | None = None) -> dict[str, float]:
    """Rectified-gate barycentric blend over SWING_TEMPLATES. ``gates`` are the
    four non-drive gate values; only positive ones count, drive always carries
    weight 1. ``drive`` overrides the drive template (an IKHitterPolicy passes
    its own constructor-kwarg-derived one)."""
    base = SWING_TEMPLATES["drive"] if drive is None else drive
    w = [GATE_GAIN * g if g > 0.0 else 0.0 for g in gates]
    if not any(w):
        return base  # WHY: bit-exact v1 path -- returns the drive dict with no arithmetic
    shapes = [base] + [SWING_TEMPLATES[n] for n in TEMPLATE_ORDER[1:]]
    weights = [1.0] + w
    total = sum(weights)
    return {f: sum(wk * s[f] for wk, s in zip(weights, shapes)) / total for f in SHAPE_FIELDS}


def reach_box(shape: dict[str, float], expand_y: float,
              expand_z: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """((y_lo, y_hi), (z_lo, z_hi)) for a blended shape, each interpolated
    ``expand_*`` of the way from the template's own clamp to REACH_MAX_*.
    Joint limits stay enforced by _solve_ik's per-iteration jnt_range clip, so
    a wider box can only stretch the pose, never violate a limit."""
    ey = min(max(expand_y, 0.0), 1.0)
    ez = min(max(expand_z, 0.0), 1.0)
    if ey == 0.0 and ez == 0.0:
        return (shape["y_lo"], shape["y_hi"]), (shape["z_lo"], shape["z_hi"])  # WHY: bit-exact v1 clamp
    return ((shape["y_lo"] + ey * (REACH_MAX_Y[0] - shape["y_lo"]),
             shape["y_hi"] + ey * (REACH_MAX_Y[1] - shape["y_hi"])),
            (shape["z_lo"] + ez * (REACH_MAX_Z[0] - shape["z_lo"]),
             shape["z_hi"] + ez * (REACH_MAX_Z[1] - shape["z_hi"])))


# v2 superset of SERVE_PARAM_NOMINAL: the ten v1 knobs plus the swing-library
# selector and the reach/shape knobs. Every added nominal is again the
# identity, so a hitter at SERVE_PARAM_NOMINAL_V2 is the plain hitter.
SERVE_PARAM_NOMINAL_V2 = {
    **SERVE_PARAM_NOMINAL,
    "gate_block": 0.0,          # blend weight GATE_GAIN*relu(u) toward the block template
    "gate_lift": 0.0,           # ...toward lift
    "gate_punch": 0.0,          # ...toward punch
    "gate_crossbody": 0.0,      # ...toward crossbody
    "x_hit_coarse": 0.0,        # m, coarse hitting-plane depth, stacks on x_hit_offset
    "reach_expand_y": 0.0,      # 0..1 lerp of the y clamp toward REACH_MAX_Y
    "reach_expand_z": 0.0,      # 0..1 lerp of the z clamp toward REACH_MAX_Z
    "normal_weight_scale": 1.0, # multiplies w_normal (face-accuracy vs. position priority)
    "swing_window_scale": 1.0,  # multiplies t_pre/speed_buffer/t_follow together (dwell)
    "swing_dir_pitch": 0.0,     # rad, rotates the swing direction up (+) / down (-)
}

# Clip on the effective w_normal (8 * normal_weight_scale): the useful
# direction is DOWN (position priority near the workspace edge), and the low
# end 8/16 = 0.5 is exact; the top clips at 16 from u = 0.25 on, on purpose --
# w_normal = 128 makes the 6x10 DLS ill-conditioned and buys nothing.
W_NORMAL_RANGE = (0.5, 16.0)

# v3 superset of SERVE_PARAM_NOMINAL_V2: one more knob, again the identity at
# nominal, so a hitter at SERVE_PARAM_NOMINAL_V3 is still the plain hitter.
# Deliberately NOT in any PARAM_SPACES entry (K_PARAMS_V2 stays 20): the guard
# is a switch, not a lerp, and its sign is known from the stance rather than
# from the serve -- there is nothing per-serve for PPO to learn. It lives here
# so a future space can expose it without another parity migration.
SERVE_PARAM_NOMINAL_V3 = {
    **SERVE_PARAM_NOMINAL_V2,
    "collision_guard": 0.0,     # >0.5 = keep the arm off the crossbody branch in the deep/inner region
}

# The collision guard (work/tt/design/hitter_v3_reach.md 2.4). Measured: the
# branch that drives the upper arm into the chest is one posture -- waist yaw
# strongly negative, waist roll on its -0.520 stop, shoulder roll adducted
# across the chest. Three per-iteration clip bounds remove it; no single bound
# and no pair does (0 vs 9-113 of 1450), and a half-strength box is worse than
# none (222 vs 216), so this is a SWITCH, not a lerp. It buys 0 legal returns
# on today's forehand-only serve set at either stance and costs ~8 points of
# target hits when forced on, so it ships OFF: its value is the latent defect
# it removes (306 of 950 poses at x = -1.50) for a widened serve set or a +y
# crossbody box.
GUARD_BOX = (-0.20, -0.30, -0.15)          # right_shoulder_roll <=, waist_yaw >=, waist_roll >=
# Region, pelvis-relative in x and y (it is arm geometry) and world in z (it is
# the table): x <= ROBOT_X + 0.67, z >= 0.85, y >= ROBOT_Y - 0.45.
GUARD_REGION = (0.67, 0.85, -0.45)


class IKHitterPolicy(Policy):
    """Model-based hitter: predict, IK, swing. See module docstring."""

    def __init__(
        self,
        t_pre: float = 0.30,
        t_follow: float = 0.20,
        return_flight_time: float = 0.50,
        # Task suggested 20-30cm; empirically (20-trial seed-0 sweeps, see
        # module docstring) a shorter 10cm backswing measurably raised
        # both contact rate and legal_return_rate here, because it keeps
        # the whole swing path inside the arm's local reachable workspace
        # (see max_swing_speed's comment below on the same effect) --
        # deviating from the suggestion is a considered result, not an
        # oversight.
        backswing_dist: float = 0.10,
        dls_damping: float = 0.10,
        dls_iters: int = 20,
        ns_gain: float = 0.0,
        step_clamp: float = 0.35,
        normal_eps: float = 0.15,
        w_normal: float = 8.0,
        x_hit: float = X_HIT,
        speed_buffer: float = 0.03,
        max_swing_speed: float = 1.0,
        serve_params: dict[str, float] | None = None,
        base_aware: bool | None = None,
    ):
        self.t_pre = t_pre
        self.t_follow = t_follow
        self.return_flight_time = return_flight_time
        self.backswing_dist = backswing_dist
        self.speed_buffer = speed_buffer
        self.max_swing_speed = max_swing_speed
        self.dls_damping = dls_damping
        self.dls_iters = dls_iters
        self.ns_gain = ns_gain
        self.step_clamp = step_clamp
        self.normal_eps = normal_eps
        self.w_normal = w_normal
        self.x_hit = x_hit
        # Base-aware IK: OFF by default. Measured on the free base (600
        # serves, work/tt/rl/free_base/RESULT.md) it fixes the paddle-belief
        # error but doubles the falls -- the arm chasing the drifting pelvis
        # feeds reaction torque into a robot nothing is balancing -- so it
        # costs legal returns until a balance controller exists. It only ever
        # engages when the observation carries base.position/base.orientation
        # (free-base scene only); on the welded scene the transform below is
        # never reached, so the hitter stays bit-for-bit what it was.
        # base_aware=None reads TT_BASE_AWARE ("1" = on) once, at construction.
        if base_aware is None:
            base_aware = os.environ.get("TT_BASE_AWARE", "0") == "1"
        self.base_aware = bool(base_aware)
        self._base_nominal_pos = scene.BASE_NOMINAL_POS.copy()
        self._base_nominal_mat = _quat_to_mat(scene.BASE_NOMINAL_QUAT)
        # Nominal until told otherwise: see SERVE_PARAM_NOMINAL_V3 / set_serve_params.
        for _name, _nominal in SERVE_PARAM_NOMINAL_V3.items():
            setattr(self, _name, float(_nominal))
        # The blend's drive template *is* this instance's constructor kwargs, so
        # a hitter built with custom widths still blends around its own defaults.
        self._drive_shape = {
            **SWING_TEMPLATES["drive"],
            "x_hit": self.x_hit, "speed_cap": self.max_swing_speed,
            "flight_time": self.return_flight_time, "t_pre": self.t_pre,
            "speed_buffer": self.speed_buffer, "t_follow": self.t_follow,
            "backswing": self.backswing_dist,
        }
        if serve_params:
            self.set_serve_params(serve_params)

        # Private shadow model: FK/IK only, never stepped, never seen by
        # the env. Built via himrobot.sim.scene.build_model() (the sanctioned
        # shared builder) rather than duplicating scene.py's XML surgery.
        self._ik_model = scene.build_model(fixed_base=True, paddle=True)
        self._ik_data = mujoco.MjData(self._ik_model)

        self._all_qpos_adr = np.array(
            [self._ik_model.jnt_qposadr[self._jid(n)] for n in scene.JOINT_ORDER]
        )
        self._ik_qpos_adr = self._all_qpos_adr[IK_INDICES]
        self._ik_dof_adr = np.array(
            [self._ik_model.jnt_dofadr[self._jid(n)] for n in IK_JOINT_NAMES]
        )
        self._ik_jnt_range = np.array(
            [self._ik_model.jnt_range[self._jid(n)] for n in IK_JOINT_NAMES]
        )
        if _WAIST_LIMIT is not None:
            # Free-base diagnostic (results/waist_limit): the bolted-pelvis
            # solver leans the torso 30 deg (the waist roll/pitch limit) mid
            # swing, which topples an unwelded robot ~1 s later. Tighten the
            # solver's waist roll/pitch bounds so the arm absorbs the reach.
            for name, lim in (("waist_roll_joint", _WAIST_LIMIT[0]), ("waist_pitch_joint", _WAIST_LIMIT[1])):
                i = IK_JOINT_NAMES.index(name)
                self._ik_jnt_range[i] = [max(self._ik_jnt_range[i][0], -lim), min(self._ik_jnt_range[i][1], lim)]
        self._paddle_body_id = mujoco.mj_name2id(self._ik_model, mujoco.mjtObj.mjOBJ_BODY, "paddle")
        self._blade_gid = mujoco.mj_name2id(self._ik_model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_blade")
        assert self._paddle_body_id >= 0 and self._blade_gid >= 0

        self._ready_ik_pose = scene.STAND_POSE[IK_INDICES].copy()
        self._jacp_a = np.zeros((3, self._ik_model.nv))
        self._jacp_b = np.zeros((3, self._ik_model.nv))

        self._plan: dict | None = None
        # preview()'s private MjData, built lazily (most rollouts never call
        # preview() at all, e.g. every existing test and eval run) so it costs
        # nothing until a WBC caller actually asks for a forecast.
        self._preview_data: mujoco.MjData | None = None
        self._target_zone = scene.DEFAULT_PARAMS["target_zone"]
        self._target_depth = scene.DEFAULT_PARAMS["target_depth"]
        self._risk = scene.DEFAULT_PARAMS["risk_preference"]

    def _jid(self, name: str) -> int:
        j = mujoco.mj_name2id(self._ik_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert j >= 0, name
        return j

    def set_serve_params(self, params: dict[str, float] | None = None,
                         **kwargs: float) -> None:
        """Set the whole per-serve modulation vector at once. Anything not
        named is reset to its nominal (identity) value, so one call fully
        describes the hitter's retuning for the upcoming serve; the current
        plan is dropped so the next act() replans under the new values.

        reset() deliberately leaves these alone -- the per-serve parameters
        outlive an episode reset so a caller can order reset()/
        set_serve_params() either way round.
        """
        merged = {**SERVE_PARAM_NOMINAL_V3, **(params or {}), **kwargs}
        unknown = sorted(set(merged) - set(SERVE_PARAM_NOMINAL_V3))
        if unknown:
            raise ValueError(
                f"unknown serve params {unknown}; known: {sorted(SERVE_PARAM_NOMINAL_V3)}")
        for name, value in merged.items():
            setattr(self, name, float(value))
        self._plan = None

    # -- Policy interface -----------------------------------------------
    def reset(self, episode: dict) -> None:
        params = {**scene.DEFAULT_PARAMS, **((episode or {}).get("params") or {})}
        self._target_zone = params["target_zone"]
        self._target_depth = params["target_depth"]
        self._risk = float(params["risk_preference"])
        self._plan = None

    def act(self, obs: dict) -> np.ndarray:
        t_now = float(obs["time_s"])
        ball_pos = np.asarray(obs["ball.position"], dtype=float)
        ball_vel = np.asarray(obs["ball.velocity"], dtype=float)
        q29 = np.asarray(obs["robot.joint_position"], dtype=float)

        self._maybe_replan(t_now, ball_pos, ball_vel)

        action = scene.STAND_POSE.copy()
        plan = self._plan
        if plan is None:
            return action
        window_start = plan["t_contact"] - plan["speed_buffer"] - plan["t_pre"]
        if not window_start <= t_now <= plan["t_contact"] + plan["t_follow"]:
            return action

        p_des, n_des = self._swing_target(t_now)
        # Short-circuits on the welded scene (no base.* keys) before any
        # arithmetic happens: see to_shadow_frame and self.base_aware.
        if self.base_aware and "base.position" in obs and "base.orientation" in obs:
            p_des, n_des = self.to_shadow_frame(
                p_des, n_des, obs["base.position"], obs["base.orientation"])
        action[IK_INDICES] = self._solve_ik(q29, p_des, n_des, plan["w_normal"])
        return action

    # -- WBC accessors (see docs/design/whole_body_v1.md sec 5.2) ---------
    def plan(self, obs: dict) -> dict | None:
        """The swing plan act() would use, computed exactly as act() computes
        it: same _maybe_replan call, same t_now/ball state, so this mutates
        self._plan exactly as much as act() would and no more (mid-swing
        that is nothing at all -- see _maybe_replan). A whole-body
        controller reads t_contact/p_hit/v_in/swing_vel/n_des/t_pre/
        speed_buffer/t_follow/... from the result; it never touches IK.
        None before a plan exists or once a swing window has elapsed.
        """
        t_now = float(obs["time_s"])
        ball_pos = np.asarray(obs["ball.position"], dtype=float)
        ball_vel = np.asarray(obs["ball.velocity"], dtype=float)
        self._maybe_replan(t_now, ball_pos, ball_vel)
        return self._plan

    def preview(self, obs: dict, horizon: int) -> np.ndarray:
        """Open-loop forecast of the next `horizon` act()-shaped targets,
        shape (horizon, 29): row k-1 is the predicted action at
        obs["time_s"] + k * PREVIEW_DT, k = 1..horizon.

        Rolls _swing_target(t_now + k*PREVIEW_DT) + _solve_ik (same
        dispatcher act() calls, collision_guard included) forward on a
        private MjData (self._preview_data, built lazily on first use),
        seeding step 1 from the measured q29 (obs["robot.joint_position"])
        and each later step from the previous step's IK solution -- an
        open-loop chain, since nothing here can know what the real robot's
        pose will actually be that far ahead. Rows that fall outside the
        current plan's swing window (or there is no plan at all) are
        scene.STAND_POSE copies, exactly act()'s own fallback, evaluated at
        the forecast time rather than at t_now. The current plan is used
        as of this call and held fixed for the whole horizon -- this never
        calls _maybe_replan, so it cannot see a future replan -- and it
        never writes self._ik_data, self._plan, or any serve-param knob, so
        calling this between act() calls changes nothing about them
        (tests/test_wbc_preview.py, tools/make_golden_param_actions.py --check).

        base_aware applies to_shadow_frame exactly as act() does, using the
        CURRENT base.position/orientation from `obs` for every forecast step
        (there is no forecast of where the pelvis itself will be).
        """
        rows = np.tile(scene.STAND_POSE, (horizon, 1))
        plan = self._plan
        if plan is None:
            return rows

        if self._preview_data is None:
            self._preview_data = mujoco.MjData(self._ik_model)
        have_base = self.base_aware and "base.position" in obs and "base.orientation" in obs
        base_pos = obs.get("base.position")
        base_quat = obs.get("base.orientation")
        q_seed = np.asarray(obs["robot.joint_position"], dtype=float).copy()

        t_now = float(obs["time_s"])
        window_start = plan["t_contact"] - plan["speed_buffer"] - plan["t_pre"]
        window_end = plan["t_contact"] + plan["t_follow"]
        for k in range(1, horizon + 1):
            t = t_now + k * PREVIEW_DT
            if not window_start <= t <= window_end:
                continue  # STAND_POSE fallback row already in place
            p_des, n_des = self._swing_target(t)
            if have_base:
                p_des, n_des = self.to_shadow_frame(p_des, n_des, base_pos, base_quat)
            q_ik = self._solve_ik(q_seed, p_des, n_des, plan["w_normal"], data=self._preview_data)
            rows[k - 1, IK_INDICES] = q_ik
            q_seed = q_seed.copy()
            q_seed[IK_INDICES] = q_ik  # chain: the next step's seed is this step's solution
        return rows

    # -- Base-aware targeting ---------------------------------------------
    def to_shadow_frame(self, p_des: np.ndarray, n_des: np.ndarray,
                        base_pos, base_quat) -> tuple[np.ndarray, np.ndarray]:
        """Re-express a world-frame blade target for the shadow model.

        The IK runs on a private shadow copy of the *welded* scene, whose
        pelvis sits at the nominal pose (BASE_NOMINAL_POS/QUAT) no matter
        where the real pelvis has drifted to. So a target computed in world
        coordinates — from the ball, which really is in world coordinates —
        is the wrong thing to hand it: the arm would be solved as if the
        pelvis had never moved, and the blade would miss by the drift. (The
        free-pelvis probe measured exactly that: an 11 cm median paddle-
        position error at contact, 40 cm when falling.)

        The fix is to ask for the target *relative to the pelvis* instead:
        take the world target into the real pelvis frame, then put it back
        down in the shadow model's nominal-pelvis world —

            p_shadow = P0 + R0 @ R_base^T @ (p_des - p_base)
            n_shadow =      R0 @ R_base^T @ n_des

        with P0/R0 the nominal pelvis pose. R0 is the identity today; it is
        written out anyway so a scene that stands the robot somewhere else
        needs no change here. A base at exactly the nominal pose makes this
        the identity, which tests/test_free_base.py pins.

        Only the IK target moves. The ball prediction and the aim point stay
        in world coordinates, where the ball actually is.
        """
        p_base = np.asarray(base_pos, dtype=np.float64)
        r_base = _quat_to_mat(base_quat)
        rel_pos = r_base.T @ (np.asarray(p_des, dtype=np.float64) - p_base)
        rel_normal = r_base.T @ np.asarray(n_des, dtype=np.float64)
        return (self._base_nominal_pos + self._base_nominal_mat @ rel_pos,
                self._base_nominal_mat @ rel_normal)

    # -- Planning ----------------------------------------------------------
    def _maybe_replan(self, t_now: float, ball_pos: np.ndarray, ball_vel: np.ndarray) -> None:
        plan = self._plan
        if plan is not None:
            # Nested on purpose: the flat elif form diverges from this one when
            # t_now is NaN (stale plan kept instead of cleared). Unreachable
            # from the sim clock, but this file's contract is exactness.
            if t_now < plan["t_contact"] + plan["t_follow"]:
                if t_now >= plan["t_contact"] - plan["speed_buffer"] - plan["t_pre"]:
                    return  # mid-swing: stay committed, don't retarget
            else:
                self._plan = None  # this swing window has elapsed

        # One swing shape per replan: the four gates blend the templates, the
        # ten v1 offsets then ride on top of the blended shape exactly as they
        # rode on the constructor defaults before v2.
        shape = blend_shape((self.gate_block, self.gate_lift,
                             self.gate_punch, self.gate_crossbody), self._drive_shape)

        x_hit_eff = shape["x_hit"] + self.x_hit_offset
        if self.x_hit_coarse != 0.0:
            x_hit_eff = x_hit_eff + self.x_hit_coarse
        traj = _predict_ball_trajectory(ball_pos, ball_vel)
        hit = _find_intercept(traj, x_hit_eff)
        if hit is None:
            return
        # t_hit_offset rides on PREDICTION_DELAY_S: both re-sample the SAME
        # predicted trajectory further along, so timing and position stay
        # mutually consistent (see PREDICTION_DELAY_S's comment).
        t_rel, p_hit, v_in = _sample_trajectory_at(
            traj, hit[0] + shape["pred_delay"] + self.t_hit_offset)
        p_hit = p_hit.copy()
        # Offsets go in *before* the reach clamp, so a requested offset can
        # never push the target outside the arm's workspace.
        y_box, z_box = reach_box(shape, self.reach_expand_y, self.reach_expand_z)
        p_hit[1] = float(np.clip(p_hit[1] + self.y_hit_offset, *y_box))
        p_hit[2] = float(np.clip(p_hit[2] + self.z_hit_offset, *z_box))

        tx, ty = scene.target_point(self._target_zone, self._target_depth)
        tx += self.aim_x_offset
        ty += self.aim_y_offset
        target3 = np.array([tx, ty, scene.TABLE_TOP_Z + scene.BALL_RADIUS])
        v_target = _solve_launch_velocity(
            p_hit, target3, shape["flight_time"] * self.flight_time_scale)
        risk = self._risk if shape["risk_gain"] == 1.0 else self._risk * shape["risk_gain"]
        swing_vel = risk * (v_target - v_in)
        # Swing direction: rotating swing_vel moves the cruise line and the
        # backswing point with it -- and, because n_des below is built from
        # n_back, the paddle face too (face biases are relative to the swing).
        swing_vel = _rotate_vector(swing_vel, shape["dir_pitch"] + self.swing_dir_pitch,
                                   shape["dir_yaw"])
        speed = np.linalg.norm(swing_vel)
        if speed < 1e-6:
            swing_vel = np.array([2.0, 0.0, 0.5])  # degenerate fallback: a modest forward-up push
            speed = np.linalg.norm(swing_vel)
        speed_cap = shape["speed_cap"] * self.swing_speed_scale * _GLOBAL_SWING_SPEED
        if speed > speed_cap:
            # Cap the *commanded Cartesian target's* speed: the raw
            # risk_preference * (v_target - v_ball_in) magnitude can run to
            # several m/s, which (a) the kp=500 position actuators cannot
            # actually track (see _swing_target's docstring) and (b) would
            # sweep the target across a Cartesian distance far outside the
            # ~0.3-0.5 m the arm can physically reach around p_hit -- both
            # of which were observed, empirically, to *hurt* contact rate
            # (a faster but unreachable/untrackable target is a worse
            # target). Capping keeps the swing achievable; direction (and
            # so aim) is preserved exactly, only magnitude is reduced.
            swing_vel = swing_vel * (speed_cap / speed)
            speed = speed_cap
        # n_back is the swing direction (used for the backswing point);
        # n_des is the commanded paddle face, which pitch/yaw_offset tilt
        # away from it. Identical at nominal.
        n_back = swing_vel / speed
        n_des = _rotate_normal(n_back, shape["face_pitch"] + self.pitch_offset,
                               shape["face_yaw"] + self.yaw_offset)

        w = self.swing_window_scale
        if w == 1.0:
            t_pre, speed_buffer, t_follow = shape["t_pre"], shape["speed_buffer"], shape["t_follow"]
        else:
            t_pre, speed_buffer, t_follow = (shape["t_pre"] * w, shape["speed_buffer"] * w,
                                             shape["t_follow"] * w)
        if self.normal_weight_scale == 1.0:
            w_normal = self.w_normal
        else:
            w_normal = float(np.clip(self.w_normal * self.normal_weight_scale, *W_NORMAL_RANGE))

        self._plan = dict(t_contact=t_now + t_rel, p_hit=p_hit, v_in=v_in,
                           swing_vel=swing_vel, n_des=n_des, n_back=n_back,
                           shape=shape, t_pre=t_pre, speed_buffer=speed_buffer,
                           t_follow=t_follow, backswing=shape["backswing"],
                           w_normal=w_normal)

    def _swing_target(self, t_now: float) -> tuple[np.ndarray, np.ndarray]:
        """Paddle-centre target trajectory.

        The core of the swing is a single constant-velocity line, target
        position ramping at swing_vel through p_hit at t_contact, held from
        speed_buffer seconds *before* contact through t_follow seconds
        *after* it -- not just switched on exactly at contact. A plain
        target that only starts moving at contact time left the paddle
        badly under-speed exactly when it needed to be fastest: a position
        actuator (kp=500, see scene.py) cannot instantly jump to a several
        m/s paddle speed (empirically, a step-response script, not
        shipped, measured ~150-200ms to settle a comparable joint change),
        and our own ball-flight/contact-time prediction has tens of ms of
        error against the real solver-driven bounce, so a target that only
        reaches speed for a single instant misses that instant more often
        than not. Cruising at swing_vel through a window around t_contact
        gives both the actuators time to spin up and the (imperfect)
        contact-time estimate some slack. Before that window, a cubic
        Hermite spline eases the target from p_pre (at rest) up to the
        cruise line's speed, so there is no discontinuous velocity jump at
        the window's start either. Tuned against 20-trial seed-0 runs (see
        the task report) after diagnosing exactly this failure mode.
        """
        plan = self._plan
        t_c = plan["t_contact"]
        p_hit, swing_vel, n_des = plan["p_hit"], plan["swing_vel"], plan["n_des"]
        n_back = plan["n_back"]
        t_speed = t_c - plan["speed_buffer"]  # target already at full swing_vel by here
        if t_now >= t_speed:
            p_des = p_hit + swing_vel * (t_now - t_c)
        else:
            p_atspeed = p_hit - swing_vel * plan["speed_buffer"]
            p_pre = p_atspeed - plan["backswing"] * n_back
            T = plan["t_pre"]
            s = np.clip((t_now - (t_speed - T)) / T, 0.0, 1.0)
            h00 = 2 * s**3 - 3 * s**2 + 1
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            p_des = h00 * p_pre + h01 * p_atspeed + h11 * T * swing_vel
        return p_des, n_des

    # -- IK ------------------------------------------------------------
    def _solve_ik(self, q29_current: np.ndarray, p_des: np.ndarray, n_des: np.ndarray,
                  w_normal: float | None = None, data=None) -> np.ndarray:
        """Dispatcher. At collision_guard <= 0.5 (the nominal) this returns
        before any arithmetic and before the region test, so the off path is
        _solve_ik_stock itself -- the same code, not a re-derivation.

        ``data`` is the MjData to solve on, forwarded to _solve_ik_stock
        unchanged (default None -> self._ik_data, act()'s path); preview()
        is the only caller that ever passes its own (self._preview_data), so
        a forecast rollout can share this exact dispatch -- collision_guard
        included -- without disturbing the real self._ik_data/self._plan.
        """
        if self.collision_guard <= 0.5:
            return self._solve_ik_stock(q29_current, p_des, n_des, w_normal, data=data)
        dx, z_min, dy = GUARD_REGION
        if not (p_des[0] <= scene.ROBOT_X + dx and p_des[2] >= z_min
                and p_des[1] >= scene.ROBOT_Y + dy):
            return self._solve_ik_stock(q29_current, p_des, n_des, w_normal, data=data)
        # Inside the region: tighten three joint limits for the duration of this
        # one solve, so the per-iteration jnt_range clip already in the loop does
        # the keep-out. Restored in `finally` -- the box is per-call, never state.
        saved = self._ik_jnt_range
        rng = saved.copy()
        rng[IK_JOINT_NAMES.index("right_shoulder_roll_joint"), 1] = GUARD_BOX[0]
        rng[IK_JOINT_NAMES.index("waist_yaw_joint"), 0] = GUARD_BOX[1]
        rng[IK_JOINT_NAMES.index("waist_roll_joint"), 0] = GUARD_BOX[2]
        self._ik_jnt_range = rng
        try:
            return self._solve_ik_stock(q29_current, p_des, n_des, w_normal, data=data)
        finally:
            self._ik_jnt_range = saved

    def _solve_ik_stock(self, q29_current: np.ndarray, p_des: np.ndarray, n_des: np.ndarray,
                        w_normal: float | None = None, data=None) -> np.ndarray:
        w_normal = self.w_normal if w_normal is None else w_normal
        d = self._ik_data if data is None else data
        d.qpos[self._all_qpos_adr] = q29_current
        eye10 = np.eye(N_IK)
        eye6 = np.eye(6)
        for _ in range(self.dls_iters):
            mujoco.mj_forward(self._ik_model, d)
            blade_pos = d.geom_xpos[self._blade_gid].copy()
            # NOTE: column index 2 (not 1) is the blade's true flat-face
            # normal -- the paddle_blade geom is a cylinder with
            # size="0.085 0.006" (radius, half-height) along its LOCAL Z
            # axis (MuJoCo's cylinder convention), and PADDLE_XML's
            # zaxis="0 1 0" maps that local Z to the paddle body's
            # local Y -- verified numerically (geom_xmat[:,2] ==
            # body xmat[:,1] at stand pose): scene.py/test_scene.py's
            # own "paddle_normal" (column 1) is actually an in-plane/
            # radial direction, fine for a coarse "aim near the disc
            # center" test but wrong for face-on contact here.
            n_cur = d.geom_xmat[self._blade_gid].reshape(3, 3)[:, 2].copy()
            pa_cur, pb_cur = blade_pos, blade_pos + self.normal_eps * n_cur
            pa_des, pb_des = p_des, p_des + self.normal_eps * n_des

            mujoco.mj_jac(self._ik_model, d, self._jacp_a, None, pa_cur, self._paddle_body_id)
            mujoco.mj_jac(self._ik_model, d, self._jacp_b, None, pb_cur, self._paddle_body_id)
            J = np.vstack([self._jacp_a[:, self._ik_dof_adr],
                            w_normal * self._jacp_b[:, self._ik_dof_adr]])
            err = np.concatenate([pa_des - pa_cur, w_normal * (pb_des - pb_cur)])

            JJt_damped = J @ J.T + (self.dls_damping ** 2) * eye6
            Jpinv = J.T @ np.linalg.inv(JJt_damped)
            qcur = d.qpos[self._ik_qpos_adr]
            dq = Jpinv @ err + (eye10 - Jpinv @ J) @ (self.ns_gain * (self._ready_ik_pose - qcur))
            nrm = np.linalg.norm(dq)
            if nrm > self.step_clamp:
                dq *= self.step_clamp / nrm
            newq = np.clip(qcur + dq, self._ik_jnt_range[:, 0], self._ik_jnt_range[:, 1])
            d.qpos[self._ik_qpos_adr] = newq
        return d.qpos[self._ik_qpos_adr].copy()
