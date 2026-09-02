"""himrobot/wbc/export_npz.py: an rsl_rl PPO checkpoint (Isaac Lab box,
model_<it>.pt) -> the torch-free .npz himrobot.policies.wbc_np.NumpyWBCPolicy
loads. See docs/design/whole_body_v1.md secs 3 and 7 for the training side;
this module only ever runs OFFLINE, on the Mac or the box, to convert a
finished checkpoint -- himrobot.policies.wbc_np never imports torch.

Checkpoint format assumed (rsl_rl 2.x OnPolicyRunner.save; ActorCritic actor
= nn.Sequential(Linear(141,512), ELU(), Linear(512,256), ELU(),
Linear(256,128), ELU(), Linear(128,15)) -- Linear layers land at
nn.Sequential indices 0, 2, 4, 6 (ELU has no parameters, at 1, 3, 5):

    checkpoint = torch.load(path, map_location="cpu")
    checkpoint["model_state_dict"]["actor.0.weight"] / "...0.bias" / "...2.weight" / ... / "...6.bias"
    checkpoint.get("obs_norm_state_dict")  -- present iff empirical_normalization
        was on: rsl_rl's EmpiricalNormalization buffers "mean" / "var".
        Absent -> identity normalization (rms_mean=0, rms_var=1).

UNVERIFIED against a real Isaac Lab box (no GPU has been rented for this
work item -- see docs/design/whole_body_v1.md sec 11/"Unknown image
versions"). tests/test_wbc_runtime.py builds a SYNTHETIC checkpoint in
exactly this shape and checks this module's numpy forward against a torch
reference (torch_forward, below) to 1e-6; that is the parity this module's
own tests can offer today. If a real box's rsl_rl version names the actor
prefix or the normalizer buffers differently, only state_dict_to_layers /
obs_norm_from_checkpoint below need to change -- the .npz schema and
himrobot.policies.wbc_np are unaffected either way.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from himrobot.rl.balance_env import DEFAULT_EXPERT_NPZ
from himrobot.sim import scene
from himrobot.wbc import obs as wbc_obs

HIDDEN = (512, 256, 128)  # rsl_rl's config/g1/agents/rsl_rl_ppo_cfg.py-style hidden sizes
ACTOR_PREFIX = "actor."


def _linear_indices(n_hidden_layers: int) -> list[int]:
    """nn.Sequential slot indices of the Linear layers when an activation
    with no parameters (ELU here) sits at every odd slot: 0, 2, 4, ...,
    2 * n_hidden_layers."""
    return [2 * i for i in range(n_hidden_layers + 1)]


def state_dict_to_layers(model_state_dict: dict, prefix: str = ACTOR_PREFIX) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(W0, b0), (W1, b1), (W2, b2), (W3, b3)], each W shaped (out, in) --
    the same convention himrobot.policies.wbc_np.NumpyWBCPolicy._forward
    consumes (``x @ W.T + b``)."""
    layers = []
    for idx in _linear_indices(len(HIDDEN)):
        w = model_state_dict[f"{prefix}{idx}.weight"].detach().cpu().numpy().astype(np.float64)
        b = model_state_dict[f"{prefix}{idx}.bias"].detach().cpu().numpy().astype(np.float64)
        layers.append((w, b))
    return layers


def obs_norm_from_checkpoint(checkpoint: dict) -> tuple[np.ndarray, np.ndarray]:
    """(rms_mean, rms_var); identity (0, 1) when the checkpoint carries no
    obs_norm_state_dict (empirical_normalization off)."""
    norm_state = checkpoint.get("obs_norm_state_dict")
    if not norm_state:
        return np.zeros(wbc_obs.OBS_DIM, dtype=np.float64), np.ones(wbc_obs.OBS_DIM, dtype=np.float64)
    mean = norm_state["mean"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    var = norm_state["var"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    return mean, var


def _default_leg_default() -> np.ndarray:
    return scene.STAND_POSE[:wbc_obs.N_LEG].copy()


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
                  flat_obs: np.ndarray) -> np.ndarray:
    """The exact forward himrobot.policies.wbc_np.NumpyWBCPolicy._forward
    runs, exposed here so export_npz can be self-checked (and tested)
    without importing wbc_np."""
    x = np.clip((flat_obs.astype(np.float64) - rms_mean) / np.sqrt(rms_var + 1e-8), -10.0, 10.0)
    n_layers = len(layers)
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < n_layers - 1:
            x = elu(x)
    return x


def torch_forward(model_state_dict: dict, obs_norm_state_dict: dict | None, flat_obs: np.ndarray,
                  prefix: str = ACTOR_PREFIX) -> np.ndarray:
    """Reference torch (float64) forward over the SAME checkpoint dicts
    export_npz reads, for parity tests only -- himrobot.policies.wbc_np
    itself never imports torch."""
    x = torch.as_tensor(np.asarray(flat_obs, dtype=np.float64), dtype=torch.float64)
    if obs_norm_state_dict:
        mean = obs_norm_state_dict["mean"].double()
        var = obs_norm_state_dict["var"].double()
        x = torch.clamp((x - mean) / torch.sqrt(var + 1e-8), -10.0, 10.0)
    idxs = _linear_indices(len(HIDDEN))
    for j, idx in enumerate(idxs):
        w = model_state_dict[f"{prefix}{idx}.weight"].double()
        b = model_state_dict[f"{prefix}{idx}.bias"].double()
        x = x @ w.T + b
        if j < len(idxs) - 1:
            x = torch.nn.functional.elu(x)
    return x.detach().cpu().numpy()


def export_npz(checkpoint_path: str, out_npz_path: str, leg_default: np.ndarray | None = None,
              expert_npz: str = DEFAULT_EXPERT_NPZ, jnt_range: np.ndarray | None = None) -> str:
    """rsl_rl model_<it>.pt -> the .npz himrobot.policies.wbc_np.NumpyWBCPolicy
    loads. ``leg_default``/``jnt_range`` default to STAND_POSE[:12] / the
    model's own joint ranges for JOINT_ORDER when not supplied -- the
    "stand" run flag (docs/design/whole_body_v1.md sec 3.2); pass the
    mocap-prior mean for the "mocap" run."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model_state_dict"]
    layers = state_dict_to_layers(model_state)
    rms_mean, rms_var = obs_norm_from_checkpoint(checkpoint)

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
    args = parser.parse_args(argv)
    path = export_npz(args.checkpoint, args.out_npz, expert_npz=args.expert_npz)
    print(path)
    return path


if __name__ == "__main__":
    main()
