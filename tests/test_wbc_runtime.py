"""Tests for himrobot.wbc.obs, himrobot.wbc.action, himrobot.wbc.export_npz
and himrobot.policies.wbc_np.NumpyWBCPolicy (WB3 -- docs/design/whole_body_v1.md
secs 3, 5, 9 item 3). Run with:

    .venv/bin/python -m pytest tests/test_wbc_runtime.py -v

Torch is used only for the export round-trip test (torch is how an rsl_rl
checkpoint arrives) -- NumpyWBCPolicy itself never imports it, matching
himrobot.policies.balance_np/param_residual_np's own torch-free contract for
HIM's mujoco worker image.
"""
from __future__ import annotations

import math
import time

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch")  # only himrobot.wbc.export_npz needs it

from himrobot.eval.evaluate import main as evaluate_main  # noqa: E402
from himrobot.policies.param_residual_np import NumpyParamResidualPolicy  # noqa: E402
from himrobot.policies.wbc_np import NumpyWBCPolicy  # noqa: E402
from himrobot.rl.balance_env import DEFAULT_EXPERT_NPZ, body_frame_gravity  # noqa: E402
from himrobot.sim import scene  # noqa: E402
from himrobot.sim.scene import TableTennisEnv  # noqa: E402
from himrobot.wbc import export_npz, obs as wbc_obs  # noqa: E402
from himrobot.wbc.action import joint_target_wbc  # noqa: E402

FREE_BASE_PARAMS = {"robot_table_collision": False}
N_STEPS = 150  # 3 s @ 50 Hz, matches evaluate.EPISODE_CONTROL_STEPS


def _jnt_range() -> np.ndarray:
    model = scene.build_model(fixed_base=True, paddle=True)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in scene.JOINT_ORDER]
    return np.array([model.jnt_range[j] for j in ids], dtype=np.float64)


def _zero_npz(tmp_path, name: str = "wbc_zero.npz") -> str:
    """All-zero W/b, identity normalization, leg_default = STAND_POSE[:12]
    -- the parity-anchor npz (see test 1)."""
    path = tmp_path / name
    np.savez(
        path,
        W0=np.zeros((512, 141)), b0=np.zeros(512),
        W1=np.zeros((256, 512)), b1=np.zeros(256),
        W2=np.zeros((128, 256)), b2=np.zeros(128),
        W3=np.zeros((15, 128)), b3=np.zeros(15),
        rms_mean=np.zeros(141), rms_var=np.ones(141), norm_eps=np.float64(0.0),
        obs_version=np.int64(1),
        leg_default=scene.STAND_POSE[:12].copy(),
        leg_scale=np.float64(wbc_obs.LEG_SCALE),
        waist_scale=np.float64(wbc_obs.WAIST_SCALE),
        jnt_range=_jnt_range(),
        expert_npz=DEFAULT_EXPERT_NPZ,
    )
    return str(path)


def _make_obs(base_pos=None, base_quat=None, base_linvel=None, base_angvel=None) -> dict:
    n = scene.NUM_JOINTS
    return {
        "robot.joint_position": np.zeros(n),
        "robot.joint_velocity": np.zeros(n),
        "base.position": base_pos if base_pos is not None else scene.BASE_NOMINAL_POS.copy(),
        "base.orientation": base_quat if base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0]),
        "base.linear_velocity": base_linvel if base_linvel is not None else np.zeros(3),
        "base.angular_velocity": base_angvel if base_angvel is not None else np.zeros(3),
        "time_s": 0.0,
    }


# ---------------------------------------------------------------------------
# (a) Parity anchor: a zero-weight WBC reproduces the plain hitter, bit for bit.
# ---------------------------------------------------------------------------
def test_parity_anchor_zero_weights_matches_plain_hitter_on_free_base(tmp_path):
    npz_path = _zero_npz(tmp_path)
    seed = 11

    env_wbc = TableTennisEnv(seed=seed, params=FREE_BASE_PARAMS, fixed_base=False)
    wbc = NumpyWBCPolicy(npz_path)
    obs_wbc = env_wbc.reset()
    wbc.reset({"serve": None, "params": env_wbc.params})

    env_plain = TableTennisEnv(seed=seed, params=FREE_BASE_PARAMS, fixed_base=False)
    expert = NumpyParamResidualPolicy(DEFAULT_EXPERT_NPZ)
    obs_plain = env_plain.reset()
    expert.reset({"serve": None, "params": env_plain.params})

    for _ in range(N_STEPS):
        action_wbc = wbc.act(obs_wbc)
        action_plain = expert.act(obs_plain)
        assert np.array_equal(action_wbc, action_plain)

        obs_wbc, _ = env_wbc.step(action_wbc)
        obs_plain, _ = env_plain.step(action_plain)
        assert np.array_equal(obs_wbc["robot.joint_position"], obs_plain["robot.joint_position"])


# ---------------------------------------------------------------------------
# (b) Slot pins.
# ---------------------------------------------------------------------------
def test_obs_and_act_dims_and_upright_slots():
    assert wbc_obs.OBS_DIM == 141
    assert wbc_obs.ACT_DIM == 15

    obs = _make_obs()
    leg_default = scene.STAND_POSE[:12].copy()
    u = scene.STAND_POSE.copy()
    flat = wbc_obs.flatten(obs, np.zeros(15), u, u, u, None, leg_default)

    assert flat.shape == (141,)
    assert np.allclose(flat[wbc_obs.GRAVITY_SLICE], [0.0, 0.0, -1.0])
    assert np.allclose(flat[wbc_obs.HEADING_SLICE], [1.0, 0.0])
    assert np.allclose(flat[wbc_obs.SWING_CLOCK_SLICE], [0.0, 0.0, 0.0])  # no plan
    assert np.allclose(flat[wbc_obs.U_NOW_SLICE], 0.0)  # u == STAND_POSE everywhere


# ---------------------------------------------------------------------------
# (c) Frame conventions.
# ---------------------------------------------------------------------------
def test_base_velocity_frame_conventions_against_mj_object_velocity():
    half = math.pi / 4.0  # 90 deg yaw about +z
    quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
    obs = _make_obs(base_quat=quat, base_linvel=np.array([1.0, 0.0, 0.0]),
                    base_angvel=np.array([0.0, 0.0, 2.0]))
    leg_default = scene.STAND_POSE[:12].copy()
    u = scene.STAND_POSE.copy()
    flat = wbc_obs.flatten(obs, np.zeros(15), u, u, u, None, leg_default)

    # base.angular_velocity is passed through as-is (already body-frame).
    assert np.allclose(flat[wbc_obs.BASE_ANG_VEL_SLICE], [0.0, 0.0, 2.0], atol=1e-9)
    # base.linear_velocity is WORLD-frame; the obs slot is R_base^T @ it.
    assert np.allclose(flat[wbc_obs.BASE_LIN_VEL_SLICE], [0.0, -1.0, 0.0], atol=1e-9)

    # Cross-check against mujoco.mj_objectVelocity(flg_local=1) on a real
    # free-base model: same pelvis pose/velocity, read back through MuJoCo's
    # own local-frame spatial-velocity routine rather than this module's math.
    model = scene.build_model(fixed_base=False, paddle=True)
    data = mujoco.MjData(model)
    pelvis_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    qpos_adr = int(model.jnt_qposadr[pelvis_jid])
    dof_adr = int(model.jnt_dofadr[pelvis_jid])
    data.qpos[qpos_adr:qpos_adr + 3] = scene.BASE_NOMINAL_POS
    data.qpos[qpos_adr + 3:qpos_adr + 7] = quat
    data.qvel[dof_adr:dof_adr + 3] = [1.0, 0.0, 0.0]
    data.qvel[dof_adr + 3:dof_adr + 6] = [0.0, 0.0, 2.0]
    mujoco.mj_forward(model, data)
    pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis_bid, res, 1)
    assert np.allclose(res[3:6], flat[wbc_obs.BASE_LIN_VEL_SLICE], atol=1e-6)


def test_heading_matches_yaw_extraction_at_a_non_identity_orientation():
    yaw = 0.3
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    heading = wbc_obs.heading_cos_sin(quat)
    assert np.allclose(heading, [np.cos(yaw), np.sin(yaw)], atol=1e-9)


# ---------------------------------------------------------------------------
# (d) Export round trip: numpy forward matches a torch reference to 1e-6.
# ---------------------------------------------------------------------------
def _synthetic_rsl_rl_checkpoint(seed: int = 0, with_norm: bool = True) -> dict:
    torch.manual_seed(seed)
    dims = [wbc_obs.OBS_DIM, 512, 256, 128, wbc_obs.ACT_DIM]
    idxs = export_npz._linear_indices(3)
    state = {}
    for idx, d_in, d_out in zip(idxs, dims[:-1], dims[1:]):
        state[f"actor.{idx}.weight"] = torch.randn(d_out, d_in, dtype=torch.float64) * 0.1
        state[f"actor.{idx}.bias"] = torch.randn(d_out, dtype=torch.float64) * 0.1
        # A critic sibling that export_npz must ignore, same shape convention.
        state[f"critic.{idx}.weight"] = torch.randn(d_out, d_in, dtype=torch.float64) * 0.1
        state[f"critic.{idx}.bias"] = torch.randn(d_out, dtype=torch.float64) * 0.1
    checkpoint = {"model_state_dict": state, "iter": 100, "infos": None}
    if with_norm:
        # rsl_rl 2.3.1 EmpiricalNormalization.state_dict(): buffers _mean /
        # _var / _std, each (1, D), plus count (rsl_rl/modules/normalizer.py).
        # Spread the variances over three decades so a wrong formula
        # (sqrt(var + 1e-8) instead of std + 1e-2, or a +-10 clip) cannot
        # hide inside the 1e-6 tolerance.
        var = (10.0 ** torch.empty(wbc_obs.OBS_DIM, dtype=torch.float64).uniform_(-4.0, 0.5))
        checkpoint["obs_norm_state_dict"] = {
            "_mean": (torch.randn(wbc_obs.OBS_DIM, dtype=torch.float64) * 0.1).unsqueeze(0),
            "_var": var.unsqueeze(0),
            "_std": torch.sqrt(var).unsqueeze(0),
            "count": torch.tensor(123456, dtype=torch.long),
        }
    return checkpoint


@pytest.mark.parametrize("with_norm", [True, False])
def test_export_npz_numpy_forward_matches_torch_reference(tmp_path, with_norm):
    checkpoint = _synthetic_rsl_rl_checkpoint(seed=3, with_norm=with_norm)
    ckpt_path = tmp_path / "model_100.pt"
    torch.save(checkpoint, ckpt_path)
    out_path = tmp_path / "wbc_export.npz"
    export_npz.export_npz(str(ckpt_path), str(out_path))

    z = np.load(out_path)
    assert int(z["obs_version"]) == wbc_obs.OBS_VERSION
    assert z["jnt_range"].shape == (scene.NUM_JOINTS, 2)
    assert z["leg_default"].shape == (12,)
    assert float(z["norm_eps"]) == (export_npz.NORM_EPS if with_norm else 0.0)
    layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(4)]

    rng = np.random.default_rng(7)
    # Wide inputs (up to ~30 sigma on the small-variance slots) so a +-10
    # clip anywhere in the numpy path would show up against the reference.
    obs_batch = rng.normal(size=(100, wbc_obs.OBS_DIM)) * 3.0
    obs_norm = checkpoint.get("obs_norm_state_dict")
    for row in obs_batch:
        np_out = export_npz.numpy_forward(layers, z["rms_mean"], z["rms_var"], row, float(z["norm_eps"]))
        torch_out = export_npz.torch_forward(checkpoint["model_state_dict"], obs_norm, row)
        assert np_out.shape == (wbc_obs.ACT_DIM,)
        assert np.allclose(np_out, torch_out, rtol=0, atol=1e-6)

    # The deploy runtime's own forward is the same function with the same
    # eps read back from the npz (not a second copy of the formula).
    monkey = NumpyWBCPolicy(str(out_path))
    for row in obs_batch[:10]:
        assert np.array_equal(monkey._forward(row),
                              export_npz.numpy_forward(layers, z["rms_mean"], z["rms_var"], row, monkey.norm_eps))


def test_export_normalizer_is_rsl_rl_formula_not_running_mean_std():
    """Pin the normalizer formula to rsl_rl 2.3.1's EmpiricalNormalization
    (rsl_rl/modules/normalizer.py: (x - _mean) / (_std + 1e-2), no clip) by
    hand: one slot with a tiny variance, where sqrt(var + 1e-8) and
    (std + 1e-2) differ by more than 2x, and one input 40 sigma out."""
    dim = wbc_obs.OBS_DIM
    mean = np.zeros(dim)
    var = np.ones(dim)
    var[0] = 1e-4  # std 0.01 -> rsl_rl divides by 0.02, RunningMeanStd by ~0.01
    layers = [(np.eye(512, dim), np.zeros(512)), (np.eye(256, 512), np.zeros(256)),
              (np.eye(128, 256), np.zeros(128)), (np.eye(15, 128), np.zeros(15))]
    x = np.zeros(dim)
    x[0] = 0.4  # 40 sigma: a +-10 clip would cap it
    out = export_npz.numpy_forward(layers, mean, var, x, export_npz.NORM_EPS)
    assert np.isclose(out[0], 0.4 / (0.01 + 1e-2)), out[0]  # 20.0, not 10 (clip) and not ~40
    # Identity export (empirical_normalization off): eps 0 -> x unchanged.
    out_id = export_npz.numpy_forward(layers, mean, np.ones(dim), x, 0.0)
    assert np.isclose(out_id[0], 0.4)


def test_export_rejects_unknown_normalizer_layout_instead_of_silently_exporting_identity(tmp_path):
    checkpoint = _synthetic_rsl_rl_checkpoint(seed=5, with_norm=False)
    checkpoint["model_state_dict"]["actor_obs_normalizer.some_new_buffer"] = torch.zeros(1, wbc_obs.OBS_DIM)
    ckpt_path = tmp_path / "model_weird.pt"
    torch.save(checkpoint, ckpt_path)
    with pytest.raises(KeyError):
        export_npz.export_npz(str(ckpt_path), str(tmp_path / "weird.npz"))


def test_export_cli_leg_default_mocap_bakes_the_prior_mean(tmp_path):
    checkpoint = _synthetic_rsl_rl_checkpoint(seed=4, with_norm=True)
    ckpt_path = tmp_path / "model_200.pt"
    torch.save(checkpoint, ckpt_path)
    out = tmp_path / "mocap.npz"
    export_npz.main([str(ckpt_path), str(out), "--leg-default", "mocap"])
    z = np.load(out)
    prior = np.load(export_npz.MOCAP_LEG_PRIOR_NPZ)["global_mean"]
    assert np.array_equal(z["leg_default"], prior)
    assert not np.array_equal(z["leg_default"], scene.STAND_POSE[:12])
    policy = NumpyWBCPolicy(str(out))
    assert np.array_equal(policy.leg_default, prior)


def test_prev_action_slot_carries_the_raw_actor_output_like_isaac(tmp_path):
    """Isaac feeds env.action_manager.action (the unclipped policy output)
    into the prev-action slot; the runtime must keep the raw output too,
    while the joint target itself is built from the +-1-clipped action."""
    path = _zero_npz(tmp_path, name="wbc_bias.npz")
    z = dict(np.load(path))
    z["b3"] = np.full(15, 5.0)  # every layer zero -> actor output == b3 == 5 everywhere
    np.savez(path, **z)
    policy = NumpyWBCPolicy(path)
    env = TableTennisEnv(seed=3, params=FREE_BASE_PARAMS, fixed_base=False)
    obs = env.reset()
    policy.reset({"serve": None, "params": env.params})
    target = policy.act(obs)
    assert np.allclose(policy._prev_action, 5.0)  # raw, not clipped to 1
    jr = _jnt_range()
    legs_expected = np.clip(scene.STAND_POSE[:12] + wbc_obs.LEG_SCALE * 1.0, jr[:12, 0], jr[:12, 1])
    assert np.allclose(target[:12], legs_expected)  # a clipped to +1 before scaling
    flat_next = wbc_obs.flatten(obs, policy._prev_action, scene.STAND_POSE, scene.STAND_POSE,
                                scene.STAND_POSE, None, policy.leg_default)
    assert np.allclose(flat_next[wbc_obs.PREV_ACTION_SLICE], 5.0)


def test_runtime_rejects_a_checkpoint_with_another_obs_version(tmp_path):
    path = _zero_npz(tmp_path, name="wbc_v99.npz")
    z = dict(np.load(path))
    z["obs_version"] = np.int64(99)
    np.savez(path, **z)
    with pytest.raises(ValueError):
        NumpyWBCPolicy(path)


def test_deploy_runtime_imports_without_torch():
    """HIM's mujoco worker image has no torch: importing the deploy policy
    (and, through it, himrobot.wbc.export_npz for numpy_forward) must not
    load torch. Checked in a fresh interpreter so this session's own torch
    import cannot mask it."""
    import subprocess
    import sys

    code = ("import sys; import himrobot.policies.wbc_np; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout


# ---------------------------------------------------------------------------
# (e) End to end: the evaluator CLI runs and act() stays fast.
# ---------------------------------------------------------------------------
def test_evaluate_cli_runs_with_wbc_policy_free_base(tmp_path, monkeypatch):
    npz_path = _zero_npz(tmp_path)
    monkeypatch.setenv("TT_WBC_NPZ", npz_path)
    metrics = evaluate_main([
        "--policy", "himrobot.policies.wbc_np:NumpyWBCPolicy",
        "--free-base", "--params", '{"robot_table_collision": false}',
        "--trials", "3", "--seed", "0", "--out", str(tmp_path / "eval"), "--no-trajectory",
    ])
    assert "falls" in metrics
    assert "legal_return_rate" in metrics
    assert 0 <= metrics["falls"] <= 3


@pytest.mark.parametrize("seed", [11, 7, 3, 2000, 4000])
def test_act_mean_latency_under_two_milliseconds(tmp_path, seed):
    npz_path = _zero_npz(tmp_path, name=f"wbc_zero_{seed}.npz")
    policy = NumpyWBCPolicy(npz_path)
    env = TableTennisEnv(seed=seed, params=FREE_BASE_PARAMS, fixed_base=False)
    obs = env.reset()
    policy.reset({"serve": None, "params": env.params})

    for _ in range(5):  # warm up (first-touch page faults, etc.)
        action = policy.act(obs)
        obs, _ = env.step(action)

    # Times act() alone -- env.step()'s own MuJoCo physics cost (~1 ms, 10
    # substeps of a 35-dof free-base contact-rich model) is real but is not
    # what this bar is about; see the module docstring's preview-cost note
    # for what act() itself spends its budget on.
    total = 0.0
    for _ in range(N_STEPS):
        start = time.perf_counter()
        action = policy.act(obs)
        total += time.perf_counter() - start
        obs, _ = env.step(action)

    mean_ms = total / N_STEPS * 1000.0
    assert mean_ms < 2.0, mean_ms
