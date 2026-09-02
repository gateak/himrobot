"""himrobot/wbc/export_npz.py: an rsl_rl PPO checkpoint (Isaac Lab box,
model_<it>.pt) -> the torch-free .npz himrobot.policies.wbc_np.NumpyWBCPolicy
loads. See docs/design/whole_body_v1.md secs 3 and 7 for the training side;
the conversion itself only ever runs OFFLINE, on the Mac or the box, on a
finished checkpoint. torch is imported lazily inside the functions that read
a checkpoint, so importing this module (for numpy_forward, the shared MLP
forward) never pulls torch into the deploy runtime.

Checkpoint format (rsl_rl 2.3.1 -- the rsl-rl-lib pin in
work/pace/pace_pins.json; verified against that source, rsl_rl/runners/
on_policy_runner.py::save and rsl_rl/modules/normalizer.py):

    checkpoint = torch.load(path, map_location="cpu")
    checkpoint["model_state_dict"]  -- ActorCritic.state_dict(); the actor is
        nn.Sequential(Linear(141,512), ELU(), Linear(512,256), ELU(),
        Linear(256,128), ELU(), Linear(128,15)), so its Linear layers sit at
        "actor.0", "actor.2", "actor.4", "actor.6" (ELU has no parameters).
    checkpoint["obs_norm_state_dict"]  -- present iff empirical_normalization
        was on: EmpiricalNormalization's buffers "_mean", "_var", "_std"
        (each (1, 141)) and "count". Its forward is

            (x - _mean) / (_std + eps),   eps = 1e-2, NO clipping

        (normalizer.py, EmpiricalNormalization.forward) -- NOT the
        (x - mean) / sqrt(var + 1e-8) clipped to +-10 that himrobot's own
        ppo.RunningMeanStd / balance_np use. The .npz keeps balance_np's
        key names (rms_mean, rms_var) plus ``norm_eps`` so NumpyWBCPolicy
        reproduces rsl_rl's formula exactly: norm_eps = 1e-2 when the
        checkpoint carries a normalizer, 0.0 (identity: mean 0, var 1) when
        it does not.

    rsl_rl 3.x (Isaac Lab 2.2+ may ship it; unverified) moves the normalizer
    INTO the policy as "actor_obs_normalizer._mean"/"_std" inside
    model_state_dict; obs_norm_from_checkpoint handles that layout too and
    raises on any normalizer it does not recognise rather than silently
    exporting identity normalization.

Only tests/test_wbc_runtime.py's synthetic checkpoint (built with exactly
these key names and shapes) has been through this module -- no GPU box has
been rented for this work item (docs/design/whole_body_v1.md sec 11).
"""
from __future__ import annotations

import argparse

import numpy as np

from himrobot.paths import ASSETS_DIR
from himrobot.rl.balance_env import DEFAULT_EXPERT_NPZ
from himrobot.sim import scene
from himrobot.wbc import obs as wbc_obs

HIDDEN = (512, 256, 128)  # rsl_rl's config/g1/agents/rsl_rl_ppo_cfg.py-style hidden sizes
ACTOR_PREFIX = "actor."
NORM_EPS = 1e-2  # rsl_rl EmpiricalNormalization's default eps (normalizer.py)
MOCAP_LEG_PRIOR_NPZ = str(ASSETS_DIR / "wbc" / "mocap_leg_prior.npz")


def _linear_indices(n_hidden_layers: int) -> list[int]:
    """nn.Sequential slot indices of the Linear layers when an activation
    with no parameters (ELU here) sits at every odd slot: 0, 2, 4, ...,
    2 * n_hidden_layers."""
    return [2 * i for i in range(n_hidden_layers + 1)]


def _to_np(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


def state_dict_to_layers(model_state_dict: dict, prefix: str = ACTOR_PREFIX) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(W0, b0), (W1, b1), (W2, b2), (W3, b3)], each W shaped (out, in) --
    the same convention himrobot.policies.wbc_np.NumpyWBCPolicy._forward
    consumes (``x @ W.T + b``)."""
    layers = []
    for idx in _linear_indices(len(HIDDEN)):
        w = _to_np(model_state_dict[f"{prefix}{idx}.weight"])
        b = _to_np(model_state_dict[f"{prefix}{idx}.bias"])
        layers.append((w, b))
    return layers


def obs_norm_from_checkpoint(checkpoint: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """(rms_mean, rms_var, norm_eps) for the actor observation normalizer.

    rsl_rl 2.x: checkpoint["obs_norm_state_dict"] with "_mean"/"_var"/"_std"
    buffers, shape (1, OBS_DIM). rsl_rl 3.x: the same buffers under
    "actor_obs_normalizer." inside model_state_dict. Neither present ->
    identity (mean 0, var 1, eps 0) -- rsl_rl's torch.nn.Identity when
    empirical_normalization is off. rms_var is the "_var" buffer itself
    (= _std ** 2); the runtime rebuilds _std as sqrt(rms_var).
    """
    dim = wbc_obs.OBS_DIM
    norm_state = checkpoint.get("obs_norm_state_dict")
    model_state = checkpoint.get("model_state_dict", {})
    prefix = ""
    if not norm_state:
        keys = [k for k in model_state if k.startswith("actor_obs_normalizer.")]
        if keys:
            prefix = "actor_obs_normalizer."
            norm_state = {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}
    if not norm_state:
        stray = [k for k in model_state if "normalizer" in k]
        if stray:
            raise KeyError(f"unrecognised observation normalizer keys in model_state_dict: {stray[:4]} ...")
        return np.zeros(dim, dtype=np.float64), np.ones(dim, dtype=np.float64), 0.0
    missing = [k for k in ("_mean", "_var") if k not in norm_state]
    if missing:
        raise KeyError(f"observation normalizer is missing buffers {missing}; has {sorted(norm_state)}")
    mean = _to_np(norm_state["_mean"]).reshape(-1)
    var = _to_np(norm_state["_var"]).reshape(-1)
    if mean.shape != (dim,) or var.shape != (dim,):
        raise ValueError(f"normalizer shape {mean.shape}/{var.shape} != ({dim},)")
    return mean, var, NORM_EPS


def _default_leg_default() -> np.ndarray:
    return scene.STAND_POSE[:wbc_obs.N_LEG].copy()


def mocap_leg_default(path: str = MOCAP_LEG_PRIOR_NPZ) -> np.ndarray:
    """The ``mocap`` run flag's leg_default: the play-clip global mean of the
    12 leg joints from tools/mocap/mocap_leg_prior.py's output
    (docs/design/whole_body_v1.md secs 3.2, 4.4)."""
    z = np.load(path, allow_pickle=False)
    leg = np.asarray(z["global_mean"], dtype=np.float64).reshape(-1)
    if leg.shape != (wbc_obs.N_LEG,):
        raise ValueError(f"{path}: global_mean shape {leg.shape} != ({wbc_obs.N_LEG},)")
    return leg


def _default_jnt_range() -> np.ndarray:
    """himrobot.rl.env._joint_range's own computation, duplicated here (not
    imported) so this module has no runtime dependency on himrobot.rl --
    scene.py's own module docstring documents the same convention for the
    same reason. Identical whether read off the welded or the free-base
    model: only the pelvis freejoint differs between them (see
    himrobot.sim.scene's free-base contract), and JOINT_ORDER never
    includes it."""
    import mujoco

    model = scene.build_model(fixed_base=True, paddle=True)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in scene.JOINT_ORDER]
    return np.array([model.jnt_range[j] for j in ids], dtype=np.float64)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0.0, x, alpha * (np.exp(x) - 1.0))


def numpy_forward(layers: list[tuple[np.ndarray, np.ndarray]], rms_mean: np.ndarray, rms_var: np.ndarray,
                  flat_obs: np.ndarray, norm_eps: float = NORM_EPS) -> np.ndarray:
    """The exact forward himrobot.policies.wbc_np.NumpyWBCPolicy runs (it
    calls this function): rsl_rl's EmpiricalNormalization
    ``(x - mean) / (std + eps)`` with std = sqrt(rms_var) and no clipping,
    then the ActorCritic actor -- Linear->ELU x3, final Linear with no
    activation (act_inference returns the mean directly). ``norm_eps``
    defaults to rsl_rl's 1e-2 (a checkpoint exported with a normalizer);
    an identity-normalized export carries norm_eps = 0.0 in its .npz."""
    x = (np.asarray(flat_obs, dtype=np.float64) - rms_mean) / (np.sqrt(rms_var) + norm_eps)
    n_layers = len(layers)
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < n_layers - 1:
            x = elu(x)
    return x


def torch_forward(model_state_dict: dict, obs_norm_state_dict: dict | None, flat_obs: np.ndarray,
                  prefix: str = ACTOR_PREFIX):
    """Reference torch (float64) forward over the SAME checkpoint dicts
    export_npz reads, written as rsl_rl 2.3.1 computes it (normalizer.py
    ``(x - _mean) / (_std + eps)``, then actor_critic.py's Sequential) --
    for parity tests only. ``obs_norm_state_dict`` is the rsl_rl buffer dict
    ("_mean"/"_std") or None for identity."""
    import torch

    x = torch.as_tensor(np.asarray(flat_obs, dtype=np.float64), dtype=torch.float64)
    if obs_norm_state_dict:
        mean = obs_norm_state_dict["_mean"].double().reshape(-1)
        std = obs_norm_state_dict["_std"].double().reshape(-1)
        x = (x - mean) / (std + NORM_EPS)
    idxs = _linear_indices(len(HIDDEN))
    for j, idx in enumerate(idxs):
        w = model_state_dict[f"{prefix}{idx}.weight"].double()
        b = model_state_dict[f"{prefix}{idx}.bias"].double()
        x = x @ w.T + b
        if j < len(idxs) - 1:
            x = torch.nn.functional.elu(x)
    return x.detach().cpu().numpy()


def load_checkpoint(checkpoint_path: str) -> dict:
    """torch.load on CPU. rsl_rl's own runner loads with weights_only=False
    (its "infos" entry may hold arbitrary Python objects); try the safe
    default first and fall back only for a checkpoint that needs it."""
    import torch

    try:
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception:  # noqa: BLE001 -- torch raises pickle.UnpicklingError subclasses here
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def export_npz(checkpoint_path: str, out_npz_path: str, leg_default: np.ndarray | None = None,
               expert_npz: str = DEFAULT_EXPERT_NPZ, jnt_range: np.ndarray | None = None) -> str:
    """rsl_rl model_<it>.pt -> the .npz himrobot.policies.wbc_np.NumpyWBCPolicy
    loads. ``leg_default``/``jnt_range`` default to STAND_POSE[:12] / the
    model's own joint ranges for JOINT_ORDER when not supplied -- the
    "stand" run flag (docs/design/whole_body_v1.md sec 3.2); pass
    mocap_leg_default() for the "mocap" run (the CLI's --leg-default mocap).
    The leg_default baked here MUST be the one the checkpoint was trained
    with: it enters both the observation (joint_pos - default) and the leg
    action term."""
    checkpoint = load_checkpoint(checkpoint_path)
    model_state = checkpoint["model_state_dict"]
    layers = state_dict_to_layers(model_state)
    rms_mean, rms_var, norm_eps = obs_norm_from_checkpoint(checkpoint)
    if layers[0][0].shape[1] != wbc_obs.OBS_DIM or layers[-1][0].shape[0] != wbc_obs.ACT_DIM:
        raise ValueError(f"actor is {layers[0][0].shape[1]}->{layers[-1][0].shape[0]}, "
                         f"expected {wbc_obs.OBS_DIM}->{wbc_obs.ACT_DIM}")

    leg_default = _default_leg_default() if leg_default is None else np.asarray(leg_default, dtype=np.float64)
    jnt_range = _default_jnt_range() if jnt_range is None else np.asarray(jnt_range, dtype=np.float64)

    np.savez(
        out_npz_path,
        W0=layers[0][0], b0=layers[0][1],
        W1=layers[1][0], b1=layers[1][1],
        W2=layers[2][0], b2=layers[2][1],
        W3=layers[3][0], b3=layers[3][1],
        rms_mean=rms_mean,
        rms_var=rms_var,
        norm_eps=np.float64(norm_eps),
        obs_version=np.int64(wbc_obs.OBS_VERSION),
        leg_default=leg_default,
        leg_scale=np.float64(wbc_obs.LEG_SCALE),
        waist_scale=np.float64(wbc_obs.WAIST_SCALE),
        jnt_range=jnt_range,
        expert_npz=str(expert_npz),
    )
    return out_npz_path


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="rsl_rl model_<it>.pt")
    parser.add_argument("out_npz", help="output .npz path")
    parser.add_argument("--expert-npz", default=DEFAULT_EXPERT_NPZ)
    parser.add_argument("--leg-default", choices=["stand", "mocap"], default="stand",
                        help="the run flag the checkpoint was trained with (design sec 3.2): "
                             "stand = STAND_POSE[:12]; mocap = assets/wbc/mocap_leg_prior.npz global_mean")
    args = parser.parse_args(argv)
    leg_default = mocap_leg_default() if args.leg_default == "mocap" else None
    path = export_npz(args.checkpoint, args.out_npz, leg_default=leg_default, expert_npz=args.expert_npz)
    print(path)
    return path


if __name__ == "__main__":
    main()
