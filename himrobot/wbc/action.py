"""himrobot/wbc/action.py: the WBC's 15-dim action -> 29-dim joint target
-- docs/design/whole_body_v1.md sec 3.2. Numpy twin of
himrobot.wbc.isaac.mdp.actions.joint_target_wbc (torch, Isaac-side); this
file matches it arithmetic-for-arithmetic, including clipping the FULL
29-dim target (legs+waist+arms) to jnt_range at the end, not just legs and
waist -- the arms slice is already a legal IK solution, so that clip is a
no-op in practice, but keeping it means both sides do the identical
min/max, not "clip some slices, passthrough others".

Consumed by himrobot.policies.wbc_np.NumpyWBCPolicy (the torch-free deploy
runtime).
"""
from __future__ import annotations

import numpy as np

from himrobot.sim import scene
from himrobot.wbc.obs import LEG_SCALE, N_LEG, N_WAIST, WAIST_SCALE

_WAIST_SLICE = slice(N_LEG, N_LEG + N_WAIST)           # JOINT_ORDER[12:15]
_ARM_SLICE = slice(N_LEG + N_WAIST, scene.NUM_JOINTS)  # JOINT_ORDER[15:29]


def joint_target_wbc(a15: np.ndarray, u29: np.ndarray, jnt_range: np.ndarray,
                     leg_default: np.ndarray) -> np.ndarray:
    """``a15`` in [-1, 1]^15 (defensively clipped again here) -> a 29-dim
    absolute joint-position target:

        legs  = leg_default + LEG_SCALE * a[0:12]
        waist = u29[12:15] + WAIST_SCALE * a[12:15]
        arms  = u29[15:29]  -- the hitter's own command, otherwise untouched

    then the whole 29-vector is clipped to jnt_range (himrobot.wbc.isaac.mdp.
    actions.joint_target_wbc's own final step).

    ``a15 = 0`` with ``leg_default == scene.STAND_POSE[:12]`` reproduces
    ``u29`` exactly, bit for bit (today's free-base hitter, unchanged) --
    the parity anchor a freshly zero-initialized WBC actor starts at
    (docs/design/whole_body_v1.md sec 3.2, tests/test_wbc_runtime.py).
    """
    a = np.clip(np.asarray(a15, dtype=np.float64), -1.0, 1.0)
    u = np.asarray(u29, dtype=np.float64)
    leg_default = np.asarray(leg_default, dtype=np.float64)

    legs = leg_default + LEG_SCALE * a[:N_LEG]
    waist = u[_WAIST_SLICE] + WAIST_SCALE * a[N_LEG:N_LEG + N_WAIST]
    arms = u[_ARM_SLICE]

    target = np.concatenate([legs, waist, arms])
    return np.clip(target, jnt_range[:, 0], jnt_range[:, 1])
