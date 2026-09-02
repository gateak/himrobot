"""Reward kernels for ``Himrobot-WBC-G1-v0``, docs/design/whole_body_v1.md
sec 3.3. Pure torch (tier 1 + 2); the isaaclab ``RewardsCfg`` wiring these
into real ``RewardTermCfg`` objects lives in ``env_cfg.py`` -- see the
``mdp`` package docstring for the three-tier split.

``REWARD_TERMS`` is the single source of truth for the name/weight table;
the kernel functions below are the actual per-env math each term uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

STATION_STD = 0.05
HEIGHT_STD = 0.05
NOMINAL_HEIGHT = 0.793  # scene.ROBOT_Z
UPRIGHT_STD = 0.15
HEADING_STD = 0.2
BASE_LIN_VEL_STD = 0.5
BASE_ANG_VEL_STD = 1.0
MOCAP_LEG_STD = 0.4
STATION_BOX_X = 0.15
STATION_BOX_Y = 0.30
FEET_CONTACT_FORCE_N = 1.0
UNDESIRED_CONTACT_FORCE_N = 1.0


@dataclass(frozen=True)
class RewardSpec:
    name: str
    weight: float
    note: str = ""


# docs/design/whole_body_v1.md sec 3.3's table, verbatim (weight x kernel column).
REWARD_TERMS: list[RewardSpec] = [
    RewardSpec("station", 1.0, "exp(-|p_xy-p_nom_xy|^2/0.05^2); full weight until t_contact+t_follow"),
    RewardSpec("station_post_swing", 0.3, "same kernel, active only after the swing window"),
    RewardSpec("box_excursion", -1.0, "post-swing only: box +-0.15 x +-0.30 m around the nominal pelvis xy"),
    RewardSpec("height", 0.5, "exp(-(z-0.793)^2/0.05^2)"),
    RewardSpec("upright", 1.0, "exp(-tilt^2/0.15^2)"),
    RewardSpec("heading", 0.5, "exp(-psi_err^2/0.2^2)"),
    RewardSpec("base_lin_vel", 0.5, "exp(-|v_lin|^2/0.5^2)"),
    RewardSpec("base_ang_vel", 0.3, "exp(-|omega|^2/1.0^2)"),
    RewardSpec("feet_contact", 0.3, "+0.3 if both ankle_roll_link bodies are in contact (>1 N)"),
    RewardSpec("foot_slip", -0.5, "xy speed of a foot that is in contact"),
    RewardSpec("mocap_leg_prior", 0.3, "exp(-|q_legs\\ankles - q_prior(phase)|^2/0.4^2)"),
    RewardSpec("action_rate", -0.1, "action-rate^2"),
    RewardSpec("leg_joint_vel", -1e-4, "leg joint-velocity^2"),
    RewardSpec("leg_torque", -2e-5, "leg torque^2 (torques are clamped, so this term is meaningful)"),
    RewardSpec("joint_limit", -10.0, "joint-limit violation"),
    RewardSpec("waist_residual", -0.5, "|waist residual|^2"),
    RewardSpec("undesired_contact", -0.1, "any body but the two feet touching the ground (>1 N)"),
    RewardSpec("termination", -10.0, "charged once on the step a non-time-out termination fires (sec 3.3)"),
]
REWARD_TERM_NAMES = [t.name for t in REWARD_TERMS]
REWARD_WEIGHTS = {t.name: t.weight for t in REWARD_TERMS}


# ---------------------------------------------------------------------------
# pure torch kernels
# ---------------------------------------------------------------------------
def exp_kernel(err_sq: torch.Tensor, std: float) -> torch.Tensor:
    """exp(-err_sq / std^2), the shared DeepMimic/BeyondMimic exp-kernel
    form every non-linear term in the table uses."""
    return torch.exp(-err_sq / (std * std))


def station_kernel(base_pos_xy: torch.Tensor, target_xy: torch.Tensor, std: float = STATION_STD) -> torch.Tensor:
    err_sq = ((base_pos_xy - target_xy) ** 2).sum(dim=-1)
    return exp_kernel(err_sq, std)


def box_excursion_sq(base_pos_xy: torch.Tensor, nominal_xy: torch.Tensor,
                     box_x: float = STATION_BOX_X, box_y: float = STATION_BOX_Y) -> torch.Tensor:
    """Squared distance outside the +-box_x/+-box_y zone around
    ``nominal_xy``; 0 anywhere inside it (himrobot.rl.balance_env's
    box_excursion_sq, generalised to torch batches)."""
    d = base_pos_xy - nominal_xy
    ex = torch.clamp(d[..., 0].abs() - box_x, min=0.0)
    ey = torch.clamp(d[..., 1].abs() - box_y, min=0.0)
    return ex * ex + ey * ey


def height_kernel(base_z: torch.Tensor, nominal_z: float = NOMINAL_HEIGHT) -> torch.Tensor:
    return exp_kernel((base_z - nominal_z) ** 2, HEIGHT_STD)


def upright_kernel(tilt_rad: torch.Tensor) -> torch.Tensor:
    return exp_kernel(tilt_rad ** 2, UPRIGHT_STD)


def heading_error_rad(cur_cos_sin: torch.Tensor, target_cos_sin: torch.Tensor) -> torch.Tensor:
    """Signed angle (rad) between two (cos, sin) unit vectors, via
    atan2(cross, dot) -- exact for any pair of angles, no wraparound."""
    cross = cur_cos_sin[..., 0] * target_cos_sin[..., 1] - cur_cos_sin[..., 1] * target_cos_sin[..., 0]
    dot = (cur_cos_sin * target_cos_sin).sum(-1)
    return torch.atan2(cross, dot)


def heading_kernel(psi_err_rad: torch.Tensor) -> torch.Tensor:
    return exp_kernel(psi_err_rad ** 2, HEADING_STD)


def base_lin_vel_kernel(v_lin: torch.Tensor) -> torch.Tensor:
    return exp_kernel((v_lin ** 2).sum(-1), BASE_LIN_VEL_STD)


def base_ang_vel_kernel(omega: torch.Tensor) -> torch.Tensor:
    return exp_kernel((omega ** 2).sum(-1), BASE_ANG_VEL_STD)


def mocap_leg_prior_kernel(q_legs_no_ankle: torch.Tensor, q_prior: torch.Tensor) -> torch.Tensor:
    return exp_kernel(((q_legs_no_ankle - q_prior) ** 2).sum(-1), MOCAP_LEG_STD)


def feet_contact_bonus(left_force_n: torch.Tensor, right_force_n: torch.Tensor,
                       threshold: float = FEET_CONTACT_FORCE_N) -> torch.Tensor:
    """1.0 where BOTH ankle_roll_link contact forces exceed ``threshold``, else 0."""
    return ((left_force_n > threshold) & (right_force_n > threshold)).to(torch.float32)


def foot_slip_penalty(foot_xy_vel: torch.Tensor, in_contact: torch.Tensor) -> torch.Tensor:
    """xy speed of a foot, zeroed where that foot is not in contact (a
    lifted, moving foot does not "slip"). foot_xy_vel: (..., feet, 2);
    in_contact: (..., feet) bool. Returns the summed per-foot speed."""
    speed = torch.linalg.norm(foot_xy_vel, dim=-1)
    return (speed * in_contact.to(speed.dtype)).sum(dim=-1)


def action_rate_penalty(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    return ((action - prev_action) ** 2).sum(dim=-1)


def joint_vel_penalty(joint_vel: torch.Tensor) -> torch.Tensor:
    return (joint_vel ** 2).sum(dim=-1)


def torque_penalty(torque: torch.Tensor) -> torch.Tensor:
    return (torque ** 2).sum(dim=-1)


def joint_limit_violation(q: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Summed out-of-range magnitude, 0 when every joint is within [low, high]."""
    below = torch.clamp(low - q, min=0.0)
    above = torch.clamp(q - high, min=0.0)
    return (below + above).sum(dim=-1)


def waist_residual_penalty(waist_residual: torch.Tensor) -> torch.Tensor:
    return (waist_residual ** 2).sum(dim=-1)


def undesired_contact_penalty(contact_force_n: torch.Tensor, threshold: float = UNDESIRED_CONTACT_FORCE_N
                              ) -> torch.Tensor:
    """Count of non-foot bodies whose contact force exceeds ``threshold``.
    contact_force_n: (..., n_bodies) -- caller passes only the "undesired"
    bodies (every body except the two ankle_roll_link feet)."""
    return (contact_force_n > threshold).to(torch.float32).sum(dim=-1)
