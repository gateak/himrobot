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
        rms_mean=np.zeros(141), rms_var=np.ones(141),
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
    checkpoint = {"model_state_dict": state}
    if with_norm:
        checkpoint["obs_norm_state_dict"] = {
            "mean": torch.randn(wbc_obs.OBS_DIM, dtype=torch.float64) * 0.1,
            "var": torch.rand(wbc_obs.OBS_DIM, dtype=torch.float64) + 0.5,
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
    layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(4)]

    rng = np.random.default_rng(7)
    obs_batch = rng.normal(size=(100, wbc_obs.OBS_DIM))
    obs_norm = checkpoint.get("obs_norm_state_dict")
    for row in obs_batch:
        np_out = export_npz.numpy_forward(layers, z["rms_mean"], z["rms_var"], row)
        torch_out = export_npz.torch_forward(checkpoint["model_state_dict"], obs_norm, row)
        assert np_out.shape == (wbc_obs.ACT_DIM,)
        assert np.allclose(np_out, torch_out, rtol=0, atol=1e-6)


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
