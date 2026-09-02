"""Observation layout ``wbc_obs_v1`` (141 actor / 176 critic) --
docs/design/whole_body_v1.md sec 3.1. Torch twin of ``himrobot/wbc/obs.py``
(WB3, landed): same names, same slot order, same math -- every function here
that has a numpy counterpart is checked against it to 1e-6 on 1000 random
poses in tests/test_wbc_mdp_torch.py. See the ``mdp`` package docstring for
the three-tier split this file follows.

Slot order (actor, 141 floats) -- one ``ObsBlock`` per himrobot.wbc.obs slice:

    0:3   gravity unit vector, pelvis frame        (train noise +-0.05)
    3:6   base angular velocity, pelvis frame       (+-0.2)
    6:9   base linear velocity, pelvis frame        (+-0.5)
    9:12  pelvis position - BASE_NOMINAL_POS         (+-0.02)
    12:14 heading (cos psi, sin psi)                (+-0.02)
    14:43 joint position - default pose (29)         (+-0.01)
    43:72 joint velocity * 0.1 (29)                  (+-0.5, pre-scale)
    72:87 previous WBC action (15)                    (no noise)
    87:104  u(t)    - STAND_POSE[12:29] (17)          (no noise)
    104:121 u(t+5)  - STAND_POSE[12:29] (17)          (no noise)
    121:138 u(t+10) - STAND_POSE[12:29] (17)          (no noise)
    138:141 swing clock: in-window flag, s since window start,
            s to contact clipped +-1                  (no noise)

Critic adds, without noise: u(t+15), u(t+25) (17 each) and the row's
serve_scale (1) = 176 total. Isaac-training-only -- himrobot.wbc.obs (the
deploy runtime) never needs a critic observation at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from himrobot.sim import scene

# ---------------------------------------------------------------------------
# tier 1: plain-data layout -- no isaaclab, no torch dependency at all.
# Mirrors himrobot.wbc.obs's own constants (N_LEG/N_WAIST/N_ARM/N_CMD, same
# values) so both files' slice arithmetic agrees by construction.
# ---------------------------------------------------------------------------
N_LEG = 12
N_WAIST = 3
N_ARM = 14
N_CMD = N_WAIST + N_ARM  # 17, JOINT_ORDER[12:29]
assert N_LEG + N_CMD == scene.NUM_JOINTS

ACT_DIM = N_LEG + N_WAIST  # 15, himrobot.wbc.obs.ACT_DIM
_CMD_SLICE_29 = slice(N_LEG, scene.NUM_JOINTS)  # JOINT_ORDER[12:29] inside a 29-vector


@dataclass(frozen=True)
class ObsBlock:
    name: str
    dim: int
    noise: float | None  # uniform +-noise added at train time; None = never noised


ACTOR_OBS_BLOCKS: list[ObsBlock] = [
    ObsBlock("gravity_b", 3, 0.05),
    ObsBlock("base_ang_vel_b", 3, 0.2),
    ObsBlock("base_lin_vel_b", 3, 0.5),
    ObsBlock("base_pos_delta", 3, 0.02),
    ObsBlock("heading_cos_sin", 2, 0.02),
    ObsBlock("joint_pos_delta", scene.NUM_JOINTS, 0.01),
    ObsBlock("joint_vel_scaled", scene.NUM_JOINTS, 0.5),  # noise added pre-scale, see docstring
    ObsBlock("prev_action", ACT_DIM, None),
    ObsBlock("u_now", N_CMD, None),
    ObsBlock("u_p5", N_CMD, None),
    ObsBlock("u_p10", N_CMD, None),
    ObsBlock("swing_clock", 3, None),
]
ACTOR_OBS_DIM = sum(b.dim for b in ACTOR_OBS_BLOCKS)
assert ACTOR_OBS_DIM == 141, ACTOR_OBS_DIM

CRITIC_EXTRA_BLOCKS: list[ObsBlock] = [
    ObsBlock("u_p15", N_CMD, None),
    ObsBlock("u_p25", N_CMD, None),
    ObsBlock("serve_scale", 1, None),
]
CRITIC_EXTRA_DIM = sum(b.dim for b in CRITIC_EXTRA_BLOCKS)
CRITIC_OBS_DIM = ACTOR_OBS_DIM + CRITIC_EXTRA_DIM
assert CRITIC_OBS_DIM == 176, CRITIC_OBS_DIM

# Doubles as the tail of the 29-dim default joint pose (waist+arms never
# leave STAND_POSE by default) -- himrobot.wbc.obs._CMD_DEFAULT, same slice.
_CMD_DEFAULT = torch.as_tensor(scene.STAND_POSE[_CMD_SLICE_29], dtype=torch.float32)  # (17,)
_BASE_NOMINAL_POS = torch.as_tensor(scene.BASE_NOMINAL_POS, dtype=torch.float32)  # (3,)


# ---------------------------------------------------------------------------
# tier 2: pure torch math -- no isaaclab, exercised directly by
# tests/test_wbc_mdp_torch.py against himrobot.wbc.obs's numpy twins.
# ---------------------------------------------------------------------------
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz -> its conjugate (== inverse for a unit quaternion)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vector ``v`` (..., 3) by quaternion ``q`` (..., 4, wxyz):
    v' = v + 2*w*(q_vec x v) + 2*q_vec x (q_vec x v) -- the standard
    optimized quaternion-rotation formula; agrees with MuJoCo's own
    mju_rotVecQuat (himrobot.wbc.obs.body_frame_vector's own primitive) to
    float precision -- both are the same rotation, computed differently."""
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def body_frame_vector(base_quat_wxyz: torch.Tensor, world_vec: torch.Tensor) -> torch.Tensor:
    """Rotate a WORLD-frame vector into the pelvis frame -- torch twin of
    himrobot.wbc.obs.body_frame_vector (mju_rotVecQuat(v, conjugate(q)))."""
    return quat_rotate(quat_conjugate(base_quat_wxyz), world_vec)


def body_frame_gravity(base_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """World [0, 0, -1] rotated into the pelvis frame; unit norm by
    construction. himrobot.rl.balance_env.body_frame_gravity's torch twin."""
    world_down = torch.zeros_like(base_quat_wxyz[..., :3])
    world_down[..., 2] = -1.0
    return body_frame_vector(base_quat_wxyz, world_down)


def heading_cos_sin(base_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """(cos psi, sin psi) of the pelvis +x axis in world xy -- torch twin of
    himrobot.wbc.obs.heading_cos_sin (same yaw-from-quaternion formula)."""
    w, x, y, z = base_quat_wxyz.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def default_pose(leg_default: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """concat(leg_default, STAND_POSE[12:]) -- torch twin of
    himrobot.wbc.obs.default_pose. ``like`` supplies the batch shape to
    broadcast a bare (12,)/(17,) leg_default/cmd_default against."""
    cmd_default = _CMD_DEFAULT.expand_as(like[..., N_LEG:])
    return torch.cat([leg_default.expand_as(like[..., :N_LEG]), cmd_default], dim=-1)


def swing_clock(t_now: torch.Tensor, t_contact: torch.Tensor, speed_buffer: torch.Tensor,
                t_pre: torch.Tensor, t_follow: torch.Tensor) -> torch.Tensor:
    """[in_window, t_now - window_start, t_contact - t_now], each clipped to
    +-1s -- torch twin of himrobot.wbc.obs.swing_clock(t_now, plan), with the
    ``plan`` dict's four fields passed as their own (batched) tensors instead
    (Isaac has no live IKHitterPolicy.plan(); a library row's own t_contact/
    t_pre/t_follow stand in, speed_buffer=0 -- himrobot.wbc.replay_library.
    _fake_plan's same documented approximation, see mdp.commands)."""
    window_start = t_contact - speed_buffer - t_pre
    window_end = t_contact + t_follow
    in_window = ((t_now >= window_start) & (t_now <= window_end)).to(t_now.dtype)
    since_start = torch.clamp(t_now - window_start, -1.0, 1.0)
    to_contact = torch.clamp(t_contact - t_now, -1.0, 1.0)
    return torch.stack([in_window, since_start, to_contact], dim=-1)


def compute_actor_obs(
    base_quat: torch.Tensor,       # (..., 4) wxyz
    base_ang_vel: torch.Tensor,    # (..., 3) already pelvis-frame (mujoco qvel[3:6] convention)
    base_lin_vel_w: torch.Tensor,  # (..., 3) WORLD frame (mujoco qvel[0:3] convention)
    base_pos: torch.Tensor,        # (..., 3) world
    joint_pos: torch.Tensor,       # (..., 29)
    joint_vel: torch.Tensor,       # (..., 29) rad/s, raw (unscaled)
    leg_default: torch.Tensor,     # (..., 12) or (12,)
    prev_action: torch.Tensor,     # (..., 15)
    u_now: torch.Tensor,           # (..., 29) FULL joint target -- only [12:29] is read, matching wbc_obs.flatten
    u_p5: torch.Tensor,            # (..., 29)
    u_p10: torch.Tensor,           # (..., 29)
    t_now: torch.Tensor,           # (...,) seconds, env clock
    t_contact: torch.Tensor,       # (...,) seconds -- the swing plan's fields (see swing_clock)
    speed_buffer: torch.Tensor,    # (...,)
    t_pre: torch.Tensor,           # (...,)
    t_follow: torch.Tensor,        # (...,)
) -> torch.Tensor:
    """The 141-dim actor observation, noise-free (noise is a caller concern
    -- see add_uniform_noise -- so this function is the deterministic core
    both the parity test and a real training-time noised wrapper share).
    Torch twin of himrobot.wbc.obs.flatten."""
    q_default = default_pose(leg_default, joint_pos)
    cmd_default = _CMD_DEFAULT.expand_as(u_now[..., _CMD_SLICE_29])
    clock = swing_clock(t_now, t_contact, speed_buffer, t_pre, t_follow)
    return torch.cat([
        body_frame_gravity(base_quat),
        base_ang_vel,
        body_frame_vector(base_quat, base_lin_vel_w),
        base_pos - _BASE_NOMINAL_POS,
        heading_cos_sin(base_quat),
        joint_pos - q_default,
        joint_vel * 0.1,
        prev_action,
        u_now[..., _CMD_SLICE_29] - cmd_default,
        u_p5[..., _CMD_SLICE_29] - cmd_default,
        u_p10[..., _CMD_SLICE_29] - cmd_default,
        clock,
    ], dim=-1)


def compute_critic_extra(u_p15: torch.Tensor, u_p25: torch.Tensor, serve_scale: torch.Tensor) -> torch.Tensor:
    """The 35-dim critic-only tail: u(t+15), u(t+25) (FULL 29-dim targets,
    only [12:29] read, each minus the JOINT_ORDER[12:29] default) and the
    row's serve_scale. Isaac-training-only, no numpy counterpart (see module
    docstring)."""
    cmd_default = _CMD_DEFAULT.expand_as(u_p15[..., _CMD_SLICE_29])
    if serve_scale.dim() < u_p15.dim() - 1:
        serve_scale = serve_scale.unsqueeze(-1)
    elif serve_scale.dim() == u_p15.dim() - 1:
        serve_scale = serve_scale.unsqueeze(-1)
    return torch.cat([u_p15[..., _CMD_SLICE_29] - cmd_default, u_p25[..., _CMD_SLICE_29] - cmd_default, serve_scale],
                     dim=-1)


def add_uniform_noise(obs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Add the per-slot uniform train-time noise from ACTOR_OBS_BLOCKS (a
    no-op slot keeps its exact value). joint_vel_scaled's noise is declared
    "pre-scale" in the design table -- +-0.5 rad/s before the *0.1 the block
    already applied -- so its slot gets +-0.05 here (0.5 * 0.1), matching
    what adding the noise before scaling would produce."""
    out = obs.clone()
    offset = 0
    for block in ACTOR_OBS_BLOCKS:
        if block.noise:
            noise_scale = block.noise * 0.1 if block.name == "joint_vel_scaled" else block.noise
            low, high = -noise_scale, noise_scale
            noise = torch.empty(obs.shape[:-1] + (block.dim,), dtype=obs.dtype, device=obs.device)
            noise.uniform_(low, high, generator=generator)
            out[..., offset:offset + block.dim] += noise
        offset += block.dim
    assert offset == ACTOR_OBS_DIM
    return out


# The isaaclab ObservationsCfg (policy/critic ObservationGroupCfg, wired to
# actor_obs_func/critic_extra_func below via ObsTerm) is built in env_cfg.py,
# the only file in this package that imports isaaclab unconditionally -- see
# the ``mdp`` package docstring's three-tier split. This module stays pure
# torch (tier 1 + 2) so tests/test_wbc_mdp_torch.py can import it with no
# isaaclab anywhere, real or stubbed.
def actor_obs_func(env, asset_cfg=None):  # noqa: ANN001 -- env is an isaaclab ManagerBasedRLEnv
    """isaaclab ObsTerm entry point (registered by env_cfg.py): pulls the
    per-env state this task's scene/command manager expose and calls
    compute_actor_obs. The exact attribute names below (env.scene["robot"],
    env.command_manager) are isaaclab's real manager API and are unverified
    against a live box (no Isaac session has ever run this file) -- see
    docs/design/whole_body_v1.md sec 11 and PORT.md-style caveats elsewhere
    in this repo for the same honesty convention."""
    robot = env.scene["robot"]
    swing = env.command_manager.get_term("swing")
    return compute_actor_obs(
        base_quat=robot.data.root_quat_w,
        base_ang_vel=robot.data.root_ang_vel_b,
        base_lin_vel_w=robot.data.root_lin_vel_w,
        base_pos=robot.data.root_pos_w,
        joint_pos=robot.data.joint_pos,
        joint_vel=robot.data.joint_vel,
        leg_default=swing.leg_default,
        prev_action=env.action_manager.action,
        u_now=swing.u_now,
        u_p5=swing.u_p5,
        u_p10=swing.u_p10,
        t_now=swing.t_now,
        t_contact=swing.t_contact,
        speed_buffer=swing.speed_buffer,
        t_pre=swing.t_pre,
        t_follow=swing.t_follow,
    )


def critic_extra_func(env, asset_cfg=None):  # noqa: ANN001
    swing = env.command_manager.get_term("swing")
    return compute_critic_extra(swing.u_p15, swing.u_p25, swing.serve_scale)
