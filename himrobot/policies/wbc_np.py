"""Torch-free runtime for a himrobot/wbc/ (Isaac Lab, rsl_rl PPO) whole-body
controller (WBC) checkpoint, exported to an .npz by himrobot.wbc.export_npz.

Mirrors himrobot.policies.balance_np.NumpyBalancePolicy's own role in this
package (HIM's mujoco worker image guarantees MuJoCo and numpy, not PyTorch
or Isaac Lab): wraps an UNCHANGED
himrobot.policies.param_residual_np.NumpyParamResidualPolicy (the shipped v3
hitter, waist+arm) exactly as balance_np does, but -- unlike balance_np's
12-dim legs-only action -- also lets a small, bounded residual reach the
waist (himrobot.wbc.action.joint_target_wbc). See
docs/design/whole_body_v1.md secs 3 and 5 for the full design this
implements.

Per control step: the expert's own act() gives u(t), the hitter's full
29-dim joint target; IKHitterPolicy.plan(obs) / preview(obs, 10) (public
accessors landed on IKHitterPolicy by the WB2 work item) give the swing
clock and the 5-/10-step (0.1 s / 0.2 s) open-loop previews.
himrobot.wbc.obs.flatten builds the 141-dim actor observation from all of
that; a 141->512->256->128->15 ELU MLP (raw rsl_rl actor weights, loaded
straight from the .npz -- no torch, no Isaac Lab) produces the WBC's own
15-dim action, clipped to [-1, 1]; himrobot.wbc.action.joint_target_wbc
combines it with u(t) into the 29-dim target TableTennisEnv.step consumes.

Preview cost (measured, this work item): IKHitterPolicy.preview's default
dls_iters=20, chained over horizon=10, costs ~4.3-4.5 ms/control-step on
average across seeds 3/7/11/2000/4000 (most control steps are free -- no
swing plan yet, no cost at all; a mid-swing call runs up to 10 damped-
least-squares solves). This policy only ever reads two of preview()'s ten
rows (k=5, k=10 -- see himrobot.wbc.obs's u_p5/u_p10 slots) and does not
need them fully converged: PREVIEW_DLS_ITERS below temporarily lowers
hitter.dls_iters around the preview() call ONLY, restored in a finally
block regardless of outcome. act() itself, called first every step, keeps
the hitter's real dls_iters throughout -- today's shipped swing is
untouched bit for bit. Measured at PREVIEW_DLS_ITERS=6: mean act() time
1.4-1.7 ms across those same seeds, comfortably under evaluate's own
budget, vs 4.3-4.5 ms unthrottled. The preview values feed only an
observation slot a policy learns to read approximately -- WB2's own
preview_error.py already treats a full-fidelity preview's 0.05 rad RMS
gap against the realised swing as an acceptable approximation for exactly
this reason -- so a lower-fidelity forecast here is a considered
performance choice, not a shortcut that changes what gets swung: u(t)
(the only thing that actually reaches the actuators unmodified, see
himrobot.wbc.action) never passes through this code path.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np

from himrobot.paths import ASSETS_DIR
from himrobot.policies.base import Policy
from himrobot.policies.param_residual_np import NumpyParamResidualPolicy
from himrobot.rl.balance_env import DEFAULT_EXPERT_NPZ
from himrobot.wbc import obs as wbc_obs
from himrobot.wbc.action import joint_target_wbc

DEFAULT_WBC_NPZ = str(ASSETS_DIR / "policies" / "wbc_v1.npz")

PREVIEW_HORIZON = 10  # IKHitterPolicy.preview(obs, PREVIEW_HORIZON) -- rows[4]/[9] are u_p5/u_p10
PREVIEW_DLS_ITERS = 6  # see module docstring's performance note


@contextmanager
def _reduced_dls_iters(hitter, iters: int):
    """Temporarily lower ``hitter.dls_iters`` (a plain instance attribute)
    for the duration of one preview() call, restored unconditionally. Never
    touches ik_hitter.py itself -- see module docstring."""
    saved = hitter.dls_iters
    hitter.dls_iters = iters
    try:
        yield
    finally:
        hitter.dls_iters = saved


class NumpyWBCPolicy(Policy):
    """Per control step: NumpyParamResidualPolicy (waist+arm, unchanged) +
    a 141->512->256->128->15 ELU MLP (legs + a bounded waist residual) ->
    combined 29-dim joint target. See module docstring / himrobot.wbc.obs /
    himrobot.wbc.action for the exact layout and combination formula.

    Deploy: himrobot.eval.evaluate --policy himrobot.policies.wbc_np:NumpyWBCPolicy
    --free-base (TT_WBC_NPZ overrides the default weights path, the same
    env-var-default pattern NumpyParamResidualPolicy/NumpyBalancePolicy use).
    """

    def __init__(self, weights_path: str | None = None):
        path = weights_path or os.environ.get("TT_WBC_NPZ") or DEFAULT_WBC_NPZ
        if not os.path.isfile(path):
            raise FileNotFoundError(f"NumpyWBCPolicy weights not found: {path}")
        z = np.load(path, allow_pickle=False)
        self.layers = [(z[f"W{i}"].astype(np.float64), z[f"b{i}"].astype(np.float64)) for i in range(4)]
        self.rms_mean = z["rms_mean"].astype(np.float64)
        self.rms_var = z["rms_var"].astype(np.float64)
        self.obs_version = int(z["obs_version"])
        self.leg_default = z["leg_default"].astype(np.float64)
        self.leg_scale = float(z["leg_scale"])
        self.waist_scale = float(z["waist_scale"])
        self.jnt_range = z["jnt_range"].astype(np.float64)
        # Metadata only -- the WBC action always combines with the LIVE
        # hitter's own act() output (u_now), so expert_npz just selects
        # which hitter checkpoint this WBC was trained/is evaluated against.
        expert_npz = str(z["expert_npz"]) if "expert_npz" in z.files else DEFAULT_EXPERT_NPZ
        self.expert = NumpyParamResidualPolicy(expert_npz)

        self._prev_action = np.zeros(wbc_obs.ACT_DIM, dtype=np.float64)

    def reset(self, episode: dict) -> None:
        self.expert.reset(episode)
        self._prev_action = np.zeros(wbc_obs.ACT_DIM, dtype=np.float64)

    def _forward(self, flat_obs: np.ndarray) -> np.ndarray:
        # rsl_rl EmpiricalNormalization (eps 1e-8, clip +-10), then the
        # ActorCritic actor: Linear->ELU x3, final Linear with no activation
        # (rsl_rl's actor head outputs the action-distribution mean directly).
        x = np.clip((flat_obs.astype(np.float64) - self.rms_mean) / np.sqrt(self.rms_var + 1e-8), -10.0, 10.0)
        n_layers = len(self.layers)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < n_layers - 1:
                x = np.where(x > 0.0, x, np.exp(x) - 1.0)  # ELU, alpha=1
        return x

    def act(self, obs: dict) -> np.ndarray:
        u_now = self.expert.act(obs)  # the hitter's real swing, full fidelity, untouched
        hitter = self.expert.hitter
        plan = hitter.plan(obs)  # idempotent here -- expert.act() already replanned this step
        with _reduced_dls_iters(hitter, PREVIEW_DLS_ITERS):
            preview = hitter.preview(obs, PREVIEW_HORIZON)
        u_p5, u_p10 = preview[4], preview[9]

        flat = wbc_obs.flatten(obs, self._prev_action, u_now, u_p5, u_p10, plan, self.leg_default)
        a = np.clip(self._forward(flat), -1.0, 1.0)
        self._prev_action = a
        return joint_target_wbc(a, u_now, self.jnt_range, self.leg_default)
