"""``WBCJointAction`` -- torch twin of ``himrobot/wbc/action.py`` (WB3,
landed): action dim 15 (12 legs + 3 waist), docs/design/whole_body_v1.md sec
3.2. Checked against the numpy version to 1e-6 in tests/test_wbc_mdp_torch.py.
See the ``mdp`` package docstring for the three-tier split.

    legs (a[0:12]):  q_des = leg_default + 0.5 rad * a,  clipped to jnt_range
    waist (a[12:15]): q_des = u[12:15] + 0.2 rad * a,     clipped to jnt_range
    arms:             q_des = u[15:29]                     -- passthrough

then the WHOLE 29-dim target is clipped to jnt_range (himrobot.wbc.action's
own final step, kept here too -- the arms slice is already a legal IK
solution so this is a no-op in practice, but it means both sides do the
identical min/max rather than "clip some slices, passthrough others").

Parity anchor: ``a = 0`` with ``leg_default = STAND_POSE[:12]`` reproduces
``u`` exactly, bit for bit (today's free-base hitter, unchanged) -- the
state a freshly-constructed, zero-initialised final actor layer starts
training from.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from himrobot.sim import scene

N_LEG = 12
N_WAIST = 3
N_ARM = 14
ACTION_DIM = N_LEG + N_WAIST  # 15
assert N_LEG + N_WAIST + N_ARM == scene.NUM_JOINTS

LEG_SCALE = 0.5
WAIST_SCALE = 0.2

_WAIST_SLICE = slice(N_LEG, N_LEG + N_WAIST)           # JOINT_ORDER[12:15], himrobot.wbc.action's own slice
_ARM_SLICE = slice(N_LEG + N_WAIST, scene.NUM_JOINTS)  # JOINT_ORDER[15:29]


@dataclass(frozen=True)
class ActionBlock:
    name: str
    dim: int
    scale: float | None  # rad; None = passthrough (no scale, no clip beyond jnt_range)


ACTION_BLOCKS: list[ActionBlock] = [
    ActionBlock("legs", N_LEG, LEG_SCALE),
    ActionBlock("waist_residual", N_WAIST, WAIST_SCALE),
    ActionBlock("arms_passthrough", N_ARM, None),
]
assert sum(b.dim for b in ACTION_BLOCKS) == scene.NUM_JOINTS


# ---------------------------------------------------------------------------
# tier 2: pure torch -- exercised directly by tests/test_wbc_mdp_torch.py
# against himrobot.wbc.action.joint_target_wbc.
# ---------------------------------------------------------------------------
def joint_target_wbc(
    a: torch.Tensor,             # (..., 15) raw MLP output, clipped to [-1,1] again here
    u: torch.Tensor,             # (..., 29) FULL hitter joint target (JOINT_ORDER)
    leg_default: torch.Tensor,   # (..., 12) or (12,)
    jnt_range: torch.Tensor,     # (29, 2) [low, high] per joint -- himrobot.rl.env._joint_range's own shape
) -> torch.Tensor:
    """The 29-dim joint-position target actually sent to the actuators --
    torch twin of himrobot.wbc.action.joint_target_wbc, arithmetic for
    arithmetic (same final whole-vector clip, see module docstring)."""
    a = torch.clamp(a, -1.0, 1.0)
    legs = leg_default + LEG_SCALE * a[..., :N_LEG]
    waist = u[..., _WAIST_SLICE] + WAIST_SCALE * a[..., N_LEG:N_LEG + N_WAIST]
    arms = u[..., _ARM_SLICE]
    target = torch.cat([legs, waist, arms], dim=-1)
    return torch.minimum(torch.maximum(target, jnt_range[..., 0]), jnt_range[..., 1])


# Default leg_default for the parity-anchored "stand" run (env_cfg.py's
# --leg_default stand): STAND_POSE[:12], the pose a = 0 must reproduce
# exactly (see module docstring). The "mocap" run wires in the mocap-prior
# mean instead (docs/design/whole_body_v1.md sec 4.4, tools/mocap/
# mocap_leg_prior.py's global_mean) -- a run-time choice, not something this
# module decides.
STAND_LEG_DEFAULT = tuple(float(v) for v in scene.STAND_POSE[:N_LEG])

# The isaaclab ActionTerm/ActionTermCfg subclass that wires joint_target_wbc
# into a real Isaac Lab ActionManager lives in env_cfg.py, which is the only
# file in this package that imports isaaclab unconditionally -- see the
# ``mdp`` package docstring's three-tier split. This module stays pure torch
# (tier 1 + 2) so tests/test_wbc_mdp_torch.py can import it with no isaaclab
# anywhere, real or stubbed.
