"""himrobot/wbc/obs.py: observation layout ``wbc_obs_v1`` (numpy runtime
side) -- docs/design/whole_body_v1.md sec 3.1. Numpy twin of
himrobot.wbc.isaac.mdp.observations (torch, Isaac-side): read that module's
docstring for the full slot table -- this file follows it block for block
(same names, same order, same math), so a future numpy-vs-torch parity test
can pin both to 1e-6 on random poses (docs/design/whole_body_v1.md sec 9,
item 3).

Consumed by himrobot.policies.wbc_np.NumpyWBCPolicy (the torch-free deploy
runtime) and, at train time, himrobot.wbc.record_library.py.
"""
from __future__ import annotations

import mujoco
import numpy as np

from himrobot.rl.balance_env import body_frame_gravity
from himrobot.sim import scene

OBS_VERSION = 1
OBS_DIM = 141
ACT_DIM = 15  # the WBC's own action: 12 legs + 3 waist residual
LEG_SCALE = 0.5    # rad, leg action authority (himrobot.wbc.action.joint_target_wbc)
WAIST_SCALE = 0.2  # rad, bounded waist residual on top of the hitter's own command

N_LEG = 12
N_WAIST = 3
N_ARM = 14
N_CMD = N_WAIST + N_ARM  # 17, JOINT_ORDER[12:29] (waist + both arms)
assert N_LEG + N_CMD == scene.NUM_JOINTS

# Named slots, in the exact block order of
# himrobot.wbc.isaac.mdp.observations.ACTOR_OBS_BLOCKS.
GRAVITY_SLICE = slice(0, 3)
BASE_ANG_VEL_SLICE = slice(3, 6)
BASE_LIN_VEL_SLICE = slice(6, 9)
BASE_POS_DELTA_SLICE = slice(9, 12)
HEADING_SLICE = slice(12, 14)
JOINT_POS_SLICE = slice(14, 14 + scene.NUM_JOINTS)                                     # 14:43
JOINT_VEL_SLICE = slice(JOINT_POS_SLICE.stop, JOINT_POS_SLICE.stop + scene.NUM_JOINTS)  # 43:72
PREV_ACTION_SLICE = slice(JOINT_VEL_SLICE.stop, JOINT_VEL_SLICE.stop + ACT_DIM)         # 72:87
U_NOW_SLICE = slice(PREV_ACTION_SLICE.stop, PREV_ACTION_SLICE.stop + N_CMD)             # 87:104
U_P5_SLICE = slice(U_NOW_SLICE.stop, U_NOW_SLICE.stop + N_CMD)                          # 104:121
U_P10_SLICE = slice(U_P5_SLICE.stop, U_P5_SLICE.stop + N_CMD)                           # 121:138
SWING_CLOCK_SLICE = slice(U_P10_SLICE.stop, U_P10_SLICE.stop + 3)                       # 138:141
assert SWING_CLOCK_SLICE.stop == OBS_DIM

_CMD_SLICE_29 = slice(N_LEG, scene.NUM_JOINTS)          # JOINT_ORDER[12:29] inside a 29-vector
_CMD_DEFAULT = scene.STAND_POSE[_CMD_SLICE_29].copy()   # (17,) -- waist+arms never leave STAND_POSE by default

# Training noise table (uniform +-value, TRAIN TIME ONLY -- flatten() below
# is deterministic and never adds noise itself; see
# himrobot.wbc.isaac.mdp.observations.add_uniform_noise for the torch-side
# trainer that does). joint_velocity's figure is pre-scale (rad/s, before
# flatten()'s own *0.1) per the design table.
NOISE = {
    "gravity": 0.05,
    "base_angular_velocity": 0.2,
    "base_linear_velocity": 0.5,
    "base_position_delta": 0.02,
    "heading": 0.02,
    "joint_position": 0.01,
    "joint_velocity": 0.5,  # pre-scale
}


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    out = np.zeros(4, dtype=np.float64)
    mujoco.mju_negQuat(out, np.asarray(quat, dtype=np.float64))
    return out


def body_frame_vector(base_quat_wxyz: np.ndarray, world_vec: np.ndarray) -> np.ndarray:
    """Rotate a WORLD-frame vector into the pelvis frame: rotate by the
    quaternion's conjugate (the inverse rotation for a unit quaternion) --
    himrobot.rl.balance_env.body_frame_gravity's operation, generalised to
    any vector, not just world-down (himrobot.wbc.isaac.mdp.observations's
    torch twin of the same name)."""
    out = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(out, np.asarray(world_vec, dtype=np.float64), _quat_conjugate(base_quat_wxyz))
    return out


def heading_cos_sin(base_quat_wxyz: np.ndarray) -> np.ndarray:
    """(cos psi, sin psi) of the pelvis +x axis in world xy: standard
    yaw-from-quaternion extraction (z-up world; identity orientation faces
    +x, scene.py's own convention). Independent of pitch/roll -- unlike
    reading off the rotated local x-axis directly, this cannot shrink when
    the pelvis tips, which is why it (not a raw rotation-matrix column) is
    what himrobot.wbc.isaac.mdp.observations.heading_cos_sin computes too."""
    w, x, y, z = (float(v) for v in base_quat_wxyz)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)


def default_pose(leg_default: np.ndarray) -> np.ndarray:
    """q_default = concat(leg_default, STAND_POSE[12:]) -- the 29-dim pose
    both the joint-position slot and himrobot.wbc.action.joint_target_wbc's
    leg term are measured against."""
    return np.concatenate([np.asarray(leg_default, dtype=np.float64), _CMD_DEFAULT])


def swing_clock(t_now: float, plan: dict | None) -> np.ndarray:
    """[in_window, t_now - window_start, t_contact - t_now], each clipped to
    +-1 s; zeros with no plan. window_start/window_end mirror
    IKHitterPolicy.act's own swing-window check, so "in_window" here means
    exactly what act() would use to decide whether to swing."""
    if plan is None:
        return np.zeros(3, dtype=np.float64)
    window_start = plan["t_contact"] - plan["speed_buffer"] - plan["t_pre"]
    window_end = plan["t_contact"] + plan["t_follow"]
    in_window = 1.0 if window_start <= t_now <= window_end else 0.0
    return np.array([
        in_window,
        float(np.clip(t_now - window_start, -1.0, 1.0)),
        float(np.clip(plan["t_contact"] - t_now, -1.0, 1.0)),
    ], dtype=np.float64)


def flatten(obs: dict, prev_action: np.ndarray, u_now: np.ndarray, u_p5: np.ndarray,
           u_p10: np.ndarray, plan: dict | None, leg_default: np.ndarray) -> np.ndarray:
    """Build the flat 141-dim WBC actor observation from a
    TableTennisEnv(fixed_base=False)-shaped obs dict (must carry the four
    base.* keys -- see himrobot.sim.scene's free-base contract).

    ``u_now``/``u_p5``/``u_p10`` are full 29-dim joint targets (JOINT_ORDER)
    -- the hitter's command now (IKHitterPolicy.act's return) and its
    IKHitterPolicy.preview(obs, 10) rows at k=5/k=10 (0.1 s / 0.2 s ahead);
    only JOINT_ORDER[12:29] (waist + both arms) of each is used, see module
    docstring. ``leg_default`` is the 12-dim run flag (STAND_POSE[:12], or
    the mocap crouch). Deterministic and noise-free -- see NOISE for the
    train-time table this never applies itself.
    """
    q_default = default_pose(leg_default)
    quat = obs["base.orientation"]
    out = np.zeros(OBS_DIM, dtype=np.float64)
    out[GRAVITY_SLICE] = body_frame_gravity(quat)
    out[BASE_ANG_VEL_SLICE] = np.asarray(obs["base.angular_velocity"], dtype=np.float64)
    out[BASE_LIN_VEL_SLICE] = body_frame_vector(quat, obs["base.linear_velocity"])
    out[BASE_POS_DELTA_SLICE] = np.asarray(obs["base.position"], dtype=np.float64) - scene.BASE_NOMINAL_POS
    out[HEADING_SLICE] = heading_cos_sin(quat)
    q = np.asarray(obs["robot.joint_position"], dtype=np.float64)
    out[JOINT_POS_SLICE] = q - q_default
    out[JOINT_VEL_SLICE] = np.asarray(obs["robot.joint_velocity"], dtype=np.float64) * 0.1
    out[PREV_ACTION_SLICE] = np.asarray(prev_action, dtype=np.float64)
    out[U_NOW_SLICE] = np.asarray(u_now, dtype=np.float64)[_CMD_SLICE_29] - _CMD_DEFAULT
    out[U_P5_SLICE] = np.asarray(u_p5, dtype=np.float64)[_CMD_SLICE_29] - _CMD_DEFAULT
    out[U_P10_SLICE] = np.asarray(u_p10, dtype=np.float64)[_CMD_SLICE_29] - _CMD_DEFAULT
    out[SWING_CLOCK_SLICE] = swing_clock(float(obs["time_s"]), plan)
    return out
