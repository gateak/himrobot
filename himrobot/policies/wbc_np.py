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

CONTRACT for the training side (measured, verifier of this work item, 40
welded serves at TT_ROBOT_Y=-0.10 with preview_error.py's own in-window /
10-IK-joint methodology): preview at PREVIEW_DLS_ITERS=6 differs from the
hitter's default dls_iters=20 preview by 0.095 rad RMS at k=5 and 0.097 rad
at k=10 -- about twice the 0.05 rad train/test threshold in design sec 5.3
on its own. Any swing library that stores preview(t) instead of the true
future (the design's fallback once preview_error.py fails its threshold --
it does: work/wbc/preview_error.json, 0.120 / 0.180 rad) MUST record it
with hitter.dls_iters = PREVIEW_DLS_ITERS, exactly as act() below does;
himrobot.wbc.preview_error measures at this setting by default for the
same reason.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np

from himrobot.paths import ASSETS_DIR
from himrobot.policies.base import Policy
from himrobot.policies.param_residual_np import NumpyParamResidualPolicy
from himrobot.rl.balance_env import DEFAULT_EXPERT_NPZ
from himrobot.sim import scene
from himrobot.wbc import obs as wbc_obs
from himrobot.wbc.action import joint_target_wbc
from himrobot.wbc.export_npz import NORM_EPS, numpy_forward  # torch-free: export_npz imports torch lazily

DEFAULT_WBC_NPZ = str(ASSETS_DIR / "policies" / "wbc_v1.npz")

PREVIEW_HORIZON = 10  # IKHitterPolicy.preview(obs, PREVIEW_HORIZON) -- rows[4]/[9] are u_p5/u_p10
PREVIEW_DLS_ITERS = 6  # see module docstring: performance note AND the recorder contract


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
        # rsl_rl EmpiricalNormalization's eps (see export_npz); an .npz
        # written before the key existed can only have come from a
        # normalized run, so rsl_rl's default is the right fallback.
        self.norm_eps = float(z["norm_eps"]) if "norm_eps" in z.files else NORM_EPS
        self.obs_version = int(z["obs_version"])
        self.leg_default = z["leg_default"].astype(np.float64)
        self.leg_scale = float(z["leg_scale"])
        self.waist_scale = float(z["waist_scale"])
        self.jnt_range = z["jnt_range"].astype(np.float64)
        # A checkpoint trained against another layout or action scaling must
        # not be silently run through this one's formulas (wbc_obs.flatten and
        # joint_target_wbc hardcode the layout and LEG_SCALE/WAIST_SCALE).
        if self.obs_version != wbc_obs.OBS_VERSION:
            raise ValueError(f"{path}: obs_version {self.obs_version} != {wbc_obs.OBS_VERSION}")
        if self.leg_scale != wbc_obs.LEG_SCALE or self.waist_scale != wbc_obs.WAIST_SCALE:
            raise ValueError(f"{path}: leg/waist scale {self.leg_scale}/{self.waist_scale} != "
                             f"{wbc_obs.LEG_SCALE}/{wbc_obs.WAIST_SCALE}")
        if self.layers[0][0].shape[1] != wbc_obs.OBS_DIM or self.layers[-1][0].shape[0] != wbc_obs.ACT_DIM:
            raise ValueError(f"{path}: actor shape {self.layers[0][0].shape} .. {self.layers[-1][0].shape}")
        if self.leg_default.shape != (wbc_obs.N_LEG,) or self.jnt_range.shape != (scene.NUM_JOINTS, 2):
            raise ValueError(f"{path}: leg_default {self.leg_default.shape} / jnt_range {self.jnt_range.shape}")
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
        # rsl_rl EmpiricalNormalization ((x - mean) / (std + eps), no clip),
        # then the ActorCritic actor: Linear->ELU x3, final Linear with no
        # activation (act_inference returns the mean directly) -- the one
        # shared implementation, see himrobot.wbc.export_npz.numpy_forward.
        return numpy_forward(self.layers, self.rms_mean, self.rms_var, flat_obs, self.norm_eps)

    def act(self, obs: dict) -> np.ndarray:
        if "base.orientation" not in obs:
            raise KeyError("NumpyWBCPolicy needs the free-base obs (base.* keys): run evaluate with --free-base")
        u_now = self.expert.act(obs)  # the hitter's real swing, full fidelity, untouched
        hitter = self.expert.hitter
        plan = hitter.plan(obs)  # idempotent here -- expert.act() already replanned this step
        with _reduced_dls_iters(hitter, PREVIEW_DLS_ITERS):
            preview = hitter.preview(obs, PREVIEW_HORIZON)
        u_p5, u_p10 = preview[4], preview[9]

        flat = wbc_obs.flatten(obs, self._prev_action, u_now, u_p5, u_p10, plan, self.leg_default)
        a_raw = self._forward(flat)
        # The prev-action slot (wbc_obs.PREV_ACTION_SLICE) carries the RAW
        # actor output, not the clipped one: the Isaac twin feeds
        # env.action_manager.action -- WBCJointAction's _raw_actions, the
        # unclipped tensor rsl_rl handed to env.step -- into the same slot
        # (himrobot/wbc/isaac/env_cfg.py process_actions,
        # mdp/observations.py actor_obs_func). joint_target_wbc clips to
        # [-1, 1] itself before scaling, on both sides.
        self._prev_action = a_raw
        return joint_target_wbc(a_raw, u_now, self.jnt_range, self.leg_default)
