"""Pure-torch parity/unit tests for himrobot.wbc.isaac.mdp -- WB6's tier-1/2
(no isaaclab, no stub gate; plain torch on CPU). See that package's
docstring for the three-tier split.

Note on "parity": docs/design/whole_body_v1.md sec 3 specifies these
functions as torch twins of himrobot/wbc/{obs,action}.py -- WB3, which has
not landed in this branch yet (checked at the top of this file so the
intent is unmistakable once it does). Until then, "parity" here means each
torch function agrees with an INDEPENDENTLY written reference of the same
design-doc formula (a second, differently-coded implementation in this
file), which is exactly what a numpy-vs-torch parity test buys once WB3
exists -- replace REFERENCE_* below with a direct call into
himrobot.wbc.obs/action when it lands.

Run with: .venv/bin/python -m pytest tests/test_wbc_mdp_torch.py -v
"""
from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from himrobot.wbc.isaac.mdp import actions, commands, observations, rewards, terminations  # noqa: E402

WB3_LANDED = importlib.util.find_spec("himrobot.wbc.obs") is not None


# ---------------------------------------------------------------------------
# helpers: an independently-coded reference for every torch function under test.
# ---------------------------------------------------------------------------
def _ref_quat_rotate_matrix(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by q via an explicit 3x3 rotation matrix -- a different
    code path from observations.quat_rotate's cross-product formula."""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R @ v


def _random_unit_quats(n: int, rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _ref_joint_target_wbc(a, u, leg_default, low, high):
    """Independent numpy re-implementation of actions.joint_target_wbc."""
    a = np.clip(a, -1.0, 1.0)
    legs = leg_default + actions.LEG_SCALE * a[..., :actions.N_LEG]
    waist = u[..., :actions.N_WAIST] + actions.WAIST_SCALE * a[..., actions.N_LEG:actions.N_LEG + actions.N_WAIST]
    arms = u[..., actions.N_WAIST:actions.N_WAIST + actions.N_ARM]
    target = np.concatenate([legs, waist, arms], axis=-1)
    return np.clip(target, low, high)


# ---------------------------------------------------------------------------
# 1. quaternion rotation utilities
# ---------------------------------------------------------------------------
def test_quat_rotate_matches_independent_rotation_matrix():
    rng = np.random.default_rng(0)
    quats = _random_unit_quats(1000, rng)
    vecs = rng.normal(size=(1000, 3))

    got = observations.quat_rotate(torch.as_tensor(quats, dtype=torch.float64),
                                   torch.as_tensor(vecs, dtype=torch.float64)).numpy()
    want = np.stack([_ref_quat_rotate_matrix(quats[i], vecs[i]) for i in range(1000)])
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_quat_rotate_is_a_rotation_norm_preserving():
    rng = np.random.default_rng(1)
    quats = torch.as_tensor(_random_unit_quats(200, rng), dtype=torch.float64)
    vecs = torch.as_tensor(rng.normal(size=(200, 3)), dtype=torch.float64)
    rotated = observations.quat_rotate(quats, vecs)
    np.testing.assert_allclose(rotated.norm(dim=-1).numpy(), vecs.norm(dim=-1).numpy(), atol=1e-9)


def test_body_frame_vector_is_the_inverse_rotation_of_quat_rotate():
    """Rotating a world vector into the body frame, then back into world
    (via quat_rotate with the SAME quaternion, undoing the conjugate),
    returns the original vector -- two independent code paths composed
    should be the identity."""
    rng = np.random.default_rng(2)
    q = torch.as_tensor(_random_unit_quats(500, rng), dtype=torch.float64)
    v = torch.as_tensor(rng.normal(size=(500, 3)), dtype=torch.float64)
    body = observations.body_frame_vector(q, v)
    back_to_world = observations.quat_rotate(q, body)
    np.testing.assert_allclose(back_to_world.numpy(), v.numpy(), atol=1e-9)


def test_body_frame_gravity_is_unit_norm_and_matches_manual_formula():
    rng = np.random.default_rng(3)
    q = torch.as_tensor(_random_unit_quats(300, rng), dtype=torch.float64)
    g = observations.body_frame_gravity(q)
    np.testing.assert_allclose(g.norm(dim=-1).numpy(), np.ones(300), atol=1e-9)
    world_down = torch.zeros(300, 3, dtype=torch.float64)
    world_down[:, 2] = -1.0
    manual = observations.body_frame_vector(q, world_down)
    np.testing.assert_allclose(g.numpy(), manual.numpy(), atol=0.0)


def test_heading_cos_sin_identity_quat_faces_plus_x():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    cos_sin = observations.heading_cos_sin(identity)
    np.testing.assert_allclose(cos_sin.numpy(), [[1.0, 0.0]], atol=1e-12)


def test_heading_cos_sin_matches_independent_atan2_of_yaw_only_quat():
    """For a pure-yaw quaternion (w=cos(psi/2), z=sin(psi/2)), heading_cos_sin
    must reproduce (cos psi, sin psi) exactly -- checked against psi drawn
    directly, not recomputed through the same trig path."""
    rng = np.random.default_rng(4)
    psi = rng.uniform(-math.pi, math.pi, size=200)
    q = np.stack([np.cos(psi / 2), np.zeros_like(psi), np.zeros_like(psi), np.sin(psi / 2)], axis=-1)
    got = observations.heading_cos_sin(torch.as_tensor(q, dtype=torch.float64)).numpy()
    want = np.stack([np.cos(psi), np.sin(psi)], axis=-1)
    np.testing.assert_allclose(got, want, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. observation layout
# ---------------------------------------------------------------------------
def test_actor_and_critic_obs_dims():
    assert observations.ACTOR_OBS_DIM == 141
    assert observations.CRITIC_EXTRA_DIM == 35
    assert observations.CRITIC_OBS_DIM == 176


def test_compute_actor_obs_shape_and_zero_state_is_mostly_zero():
    n = 4
    zeros3 = torch.zeros(n, 3, dtype=torch.float32)
    identity_quat = torch.zeros(n, 4, dtype=torch.float32)
    identity_quat[:, 0] = 1.0
    leg_default = torch.zeros(n, observations.N_LEG, dtype=torch.float32)
    joint_pos = torch.zeros(n, 29, dtype=torch.float32)
    joint_pos[:, observations.N_LEG:] = observations._CMD_DEFAULT  # matches the default pose exactly
    prev_action = torch.zeros(n, 15, dtype=torch.float32)
    u = observations._CMD_DEFAULT.unsqueeze(0).expand(n, -1)
    swing_clock = torch.zeros(n, 3, dtype=torch.float32)
    base_pos = observations._BASE_NOMINAL_POS.unsqueeze(0).expand(n, -1)

    obs = observations.compute_actor_obs(
        base_quat=identity_quat, base_ang_vel=zeros3, base_lin_vel_w=zeros3, base_pos=base_pos,
        joint_pos=joint_pos, joint_vel=torch.zeros(n, 29, dtype=torch.float32), leg_default=leg_default,
        prev_action=prev_action, u_now=u, u_p5=u, u_p10=u, swing_clock=swing_clock,
    )
    assert obs.shape == (n, 141)
    # gravity_b at identity orientation is world-down == [0, 0, -1]; heading
    # at identity orientation is (cos 0, sin 0) = (1, 0) (slots 12:14); every
    # other slot in this all-default state is exactly zero.
    expected = torch.zeros(n, 141)
    expected[:, 2] = -1.0
    expected[:, 12] = 1.0
    np.testing.assert_allclose(obs.numpy(), expected.numpy(), atol=1e-9)


def test_compute_critic_extra_shape_and_values():
    n = 3
    u15 = torch.rand(n, 17)
    u25 = torch.rand(n, 17)
    serve_scale = torch.tensor([1.0, 1.3, 0.5])
    extra = observations.compute_critic_extra(u15, u25, serve_scale)
    assert extra.shape == (n, 35)
    np.testing.assert_allclose(extra[:, :17].numpy(), (u15 - observations._CMD_DEFAULT).numpy(), atol=1e-6)
    np.testing.assert_allclose(extra[:, 17:34].numpy(), (u25 - observations._CMD_DEFAULT).numpy(), atol=1e-6)
    np.testing.assert_allclose(extra[:, 34].numpy(), serve_scale.numpy(), atol=1e-9)


# ---------------------------------------------------------------------------
# 3. action mapping
# ---------------------------------------------------------------------------
def test_action_dim_and_scales():
    assert actions.ACTION_DIM == 15
    assert actions.LEG_SCALE == 0.5
    assert actions.WAIST_SCALE == 0.2
    assert [b.name for b in actions.ACTION_BLOCKS] == ["legs", "waist_residual", "arms_passthrough"]
    assert [b.dim for b in actions.ACTION_BLOCKS] == [12, 3, 14]


def test_joint_target_wbc_matches_independent_numpy_reference():
    rng = np.random.default_rng(5)
    n = 1000
    a = rng.uniform(-1.5, 1.5, size=(n, 15))  # deliberately outside [-1,1] to exercise the clamp
    u = rng.uniform(-1.0, 1.0, size=(n, 17))
    leg_default = rng.uniform(-0.3, 0.3, size=(n, 12))
    low = np.full((n, 29), -2.5)
    high = np.full((n, 29), 2.5)

    got = actions.joint_target_wbc(
        torch.as_tensor(a), torch.as_tensor(u), torch.as_tensor(leg_default),
        torch.as_tensor(low), torch.as_tensor(high),
    ).numpy()
    want = _ref_joint_target_wbc(a, u, leg_default, low, high)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_joint_target_wbc_parity_anchor_zero_action_reproduces_hitter_output():
    """a = 0 with leg_default = STAND_POSE[:12] reproduces the hitter's own
    29-dim action exactly: legs -> STAND_POSE[:12], waist+arms -> u
    unchanged (docs/design/whole_body_v1.md sec 3.2)."""
    from himrobot.sim import scene
    n = 10
    a = torch.zeros(n, 15, dtype=torch.float64)
    u = torch.as_tensor(np.tile(scene.STAND_POSE[12:], (n, 1)), dtype=torch.float64)
    leg_default = torch.as_tensor(np.tile(scene.STAND_POSE[:12], (n, 1)), dtype=torch.float64)
    low = torch.full((n, 29), -10.0, dtype=torch.float64)
    high = torch.full((n, 29), 10.0, dtype=torch.float64)

    target = actions.joint_target_wbc(a, u, leg_default, low, high).numpy()
    expected = np.tile(scene.STAND_POSE, (n, 1))
    np.testing.assert_allclose(target, expected, atol=1e-12)


def test_joint_target_wbc_clips_to_range():
    a = torch.zeros(1, 15)
    u = torch.full((1, 17), 100.0)  # way outside any real joint range
    leg_default = torch.full((1, 12), -100.0)
    low = torch.full((1, 29), -1.0)
    high = torch.full((1, 29), 1.0)
    target = actions.joint_target_wbc(a, u, leg_default, low, high)
    assert torch.all(target <= 1.0) and torch.all(target >= -1.0)


# ---------------------------------------------------------------------------
# 4. reward kernels -- 3 hand-built states with hand-computed values.
# ---------------------------------------------------------------------------
def test_reward_term_table_matches_design_doc():
    names = [t.name for t in rewards.REWARD_TERMS]
    assert names == [
        "station", "station_post_swing", "box_excursion", "height", "upright", "heading",
        "base_lin_vel", "base_ang_vel", "feet_contact", "foot_slip", "mocap_leg_prior",
        "action_rate", "leg_joint_vel", "leg_torque", "joint_limit", "waist_residual", "undesired_contact",
    ]
    assert rewards.REWARD_WEIGHTS["station"] == 1.0
    assert rewards.REWARD_WEIGHTS["station_post_swing"] == 0.3
    assert rewards.REWARD_WEIGHTS["box_excursion"] == -1.0
    assert rewards.REWARD_WEIGHTS["height"] == 0.5
    assert rewards.REWARD_WEIGHTS["upright"] == 1.0
    assert rewards.REWARD_WEIGHTS["heading"] == 0.5
    assert rewards.REWARD_WEIGHTS["base_lin_vel"] == 0.5
    assert rewards.REWARD_WEIGHTS["base_ang_vel"] == 0.3
    assert rewards.REWARD_WEIGHTS["feet_contact"] == 0.3
    assert rewards.REWARD_WEIGHTS["foot_slip"] == -0.5
    assert rewards.REWARD_WEIGHTS["mocap_leg_prior"] == 0.3
    assert rewards.REWARD_WEIGHTS["action_rate"] == pytest.approx(-0.1)
    assert rewards.REWARD_WEIGHTS["leg_joint_vel"] == pytest.approx(-1e-4)
    assert rewards.REWARD_WEIGHTS["leg_torque"] == pytest.approx(-2e-5)
    assert rewards.REWARD_WEIGHTS["joint_limit"] == -10.0
    assert rewards.REWARD_WEIGHTS["waist_residual"] == -0.5
    assert rewards.REWARD_WEIGHTS["undesired_contact"] == pytest.approx(-0.1)


def test_station_kernel_three_hand_built_states():
    # state A: pelvis exactly at the nominal xy -> kernel == 1.0
    pos = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.0, 2.0]])
    np.testing.assert_allclose(rewards.station_kernel(pos, target).numpy(), [1.0], atol=1e-12)

    # state B: pelvis 0.05 m away in x only (== the std) -> exp(-1) = 0.36788
    pos_b = torch.tensor([[1.05, 2.0]])
    got_b = rewards.station_kernel(pos_b, target).item()
    assert got_b == pytest.approx(math.exp(-1.0), abs=1e-6)

    # state C: pelvis 0.03 m in x and 0.04 m in y away (3-4-5 triangle,
    # err^2 = 0.0025) -> exp(-0.0025 / 0.05^2) = exp(-1.0)
    pos_c = torch.tensor([[1.03, 2.04]])
    got_c = rewards.station_kernel(pos_c, target).item()
    assert got_c == pytest.approx(math.exp(-1.0), abs=1e-6)


def test_height_upright_heading_kernels_hand_computed():
    assert rewards.height_kernel(torch.tensor([0.793])).item() == pytest.approx(1.0, abs=1e-12)
    assert rewards.height_kernel(torch.tensor([0.843])).item() == pytest.approx(math.exp(-1.0), abs=1e-6)

    assert rewards.upright_kernel(torch.tensor([0.0])).item() == pytest.approx(1.0, abs=1e-12)
    assert rewards.upright_kernel(torch.tensor([0.15])).item() == pytest.approx(math.exp(-1.0), abs=1e-6)

    cur = torch.tensor([[1.0, 0.0]])
    same = torch.tensor([[1.0, 0.0]])
    opposite = torch.tensor([[-1.0, 0.0]])  # 180 deg away
    assert rewards.heading_kernel(rewards.heading_error_rad(cur, same)).item() == pytest.approx(1.0, abs=1e-12)
    err = rewards.heading_error_rad(cur, opposite).item()
    assert abs(err) == pytest.approx(math.pi, abs=1e-6)


def test_box_excursion_zero_inside_box_positive_outside():
    nominal = torch.tensor([[0.0, 0.0]])
    inside = torch.tensor([[0.10, 0.20]])  # within +-0.15 x +-0.30
    assert rewards.box_excursion_sq(inside, nominal).item() == pytest.approx(0.0, abs=1e-12)
    outside = torch.tensor([[0.25, 0.0]])  # 0.10 m beyond the 0.15 m box edge
    assert rewards.box_excursion_sq(outside, nominal).item() == pytest.approx(0.10 ** 2, abs=1e-6)


def test_feet_contact_bonus_and_foot_slip_and_undesired_contact():
    left = torch.tensor([0.5, 5.0])
    right = torch.tensor([5.0, 5.0])
    np.testing.assert_allclose(rewards.feet_contact_bonus(left, right).numpy(), [0.0, 1.0])

    foot_vel = torch.tensor([[[0.3, 0.4]], [[0.0, 0.0]]])  # (2 envs, 1 foot, 2)
    in_contact = torch.tensor([[True], [True]])
    np.testing.assert_allclose(rewards.foot_slip_penalty(foot_vel, in_contact).numpy(), [0.5, 0.0], atol=1e-6)
    # not in contact -> the slip is not counted even if the foot is moving
    in_contact2 = torch.tensor([[False], [True]])
    np.testing.assert_allclose(rewards.foot_slip_penalty(foot_vel, in_contact2).numpy(), [0.0, 0.0], atol=1e-6)

    forces = torch.tensor([[0.0, 2.0], [0.5, 0.5]])
    np.testing.assert_allclose(rewards.undesired_contact_penalty(forces).numpy(), [1.0, 0.0])


def test_joint_limit_violation_and_action_rate_and_torque_penalties():
    q = torch.tensor([[0.0, -1.0, 2.0]])
    low = torch.tensor([[-0.5, -0.5, -0.5]])
    high = torch.tensor([[0.5, 0.5, 0.5]])
    # joint 0 in range (0 violation), joint 1 below by 0.5, joint 2 above by 1.5
    assert rewards.joint_limit_violation(q, low, high).item() == pytest.approx(2.0, abs=1e-9)

    a = torch.tensor([[1.0, 2.0]])
    prev = torch.tensor([[0.0, 0.0]])
    assert rewards.action_rate_penalty(a, prev).item() == pytest.approx(5.0, abs=1e-9)  # 1^2 + 2^2

    torque = torch.tensor([[3.0, 4.0]])
    assert rewards.torque_penalty(torque).item() == pytest.approx(25.0, abs=1e-9)  # 3^2 + 4^2


# ---------------------------------------------------------------------------
# 5. terminations
# ---------------------------------------------------------------------------
def test_termination_terms_and_thresholds():
    assert terminations.FALL_PELVIS_Z == pytest.approx(0.45)
    assert terminations.TILT_LIMIT_RAD == pytest.approx(0.7)
    assert terminations.EPISODE_LENGTH_STEPS == 150
    assert terminations.TERMINATION_TERM_NAMES == ["fell", "tilted", "time_out"]

    assert bool(terminations.fell(torch.tensor([0.44])).item()) is True
    assert bool(terminations.fell(torch.tensor([0.46])).item()) is False
    assert bool(terminations.tilted(torch.tensor([0.71])).item()) is True
    assert bool(terminations.tilted(torch.tensor([0.69])).item()) is False
    assert bool(terminations.time_out(torch.tensor([150])).item()) is True
    assert bool(terminations.time_out(torch.tensor([149])).item()) is False


# ---------------------------------------------------------------------------
# 6. SwingLibraryCommand's pure-torch gather/clock/library-mix logic, on a
#    fake 3-row library, CPU.
# ---------------------------------------------------------------------------
def _fake_library() -> commands.SwingLibrary:
    # 3 rows, 6 steps, 17 channels; row r's u[t] = r*100 + t so a gathered
    # value pins down exactly which (row, step) was read.
    u = torch.zeros(3, 6, 17)
    for r in range(3):
        for t in range(6):
            u[r, t, :] = r * 100 + t
    return commands.SwingLibrary(
        label="fake",
        u=u,
        contact_step=torch.tensor([3, 4, 2]),
        swing_start_step=torch.tensor([1, 1, 0]),
        serve_scale=torch.tensor([1.0, 1.3, 0.5]),
    )


def test_gather_u_reads_the_right_row_and_step():
    lib = _fake_library()
    row_idx = torch.tensor([0, 1, 2])
    time_idx = torch.tensor([0, 3, 5])
    got = commands.gather_u(lib.u, row_idx, time_idx)
    want = torch.tensor([[0.0] * 17, [103.0] * 17, [205.0] * 17])
    np.testing.assert_allclose(got.numpy(), want.numpy())


def test_gather_u_clamps_past_the_last_recorded_step():
    lib = _fake_library()
    row_idx = torch.tensor([0, 1])
    time_idx = torch.tensor([9, 100])  # both well past T=6
    got = commands.gather_u(lib.u, row_idx, time_idx)
    want = torch.tensor([[5.0] * 17, [105.0] * 17])  # each row's own last frame (t=5)
    np.testing.assert_allclose(got.numpy(), want.numpy())


def test_gather_previews_reads_every_offset():
    lib = _fake_library()
    row_idx = torch.tensor([0])
    time_idx = torch.tensor([0])
    previews = commands.gather_previews(lib.u, row_idx, time_idx, offsets=(0, 5, 10))
    assert set(previews.keys()) == {0, 5, 10}
    np.testing.assert_allclose(previews[0].numpy(), [[0.0] * 17])
    np.testing.assert_allclose(previews[5].numpy(), [[5.0] * 17])   # exact last step
    np.testing.assert_allclose(previews[10].numpy(), [[5.0] * 17])  # clamped past the end


def test_swing_clock_in_window_flag_and_seconds():
    contact = torch.tensor([10, 10])
    start = torch.tensor([2, 2])
    step_dt = 0.02  # 50 Hz
    # before the window
    clock_before = commands.swing_clock(torch.tensor([0, 0]), contact, start, step_dt)
    assert clock_before[0, 0].item() == 0.0  # not in window
    assert clock_before[0, 1].item() == pytest.approx(0.0)  # since_start clamped at 0
    assert clock_before[0, 2].item() == pytest.approx(0.20, abs=1e-6)  # 10 steps * 0.02s

    # inside the window
    clock_in = commands.swing_clock(torch.tensor([5, 5]), contact, start, step_dt)
    assert clock_in[0, 0].item() == 1.0
    assert clock_in[0, 1].item() == pytest.approx(3 * step_dt, abs=1e-6)  # (5-2)*0.02
    assert clock_in[0, 2].item() == pytest.approx(5 * step_dt, abs=1e-6)  # (10-5)*0.02

    # to-contact clipping at +-1s
    clock_far = commands.swing_clock(torch.tensor([0]), torch.tensor([1000]), torch.tensor([0]), step_dt, clip_s=1.0)
    assert clock_far[0, 2].item() == pytest.approx(1.0)


def test_library_bank_pads_shorter_rows_and_offsets_are_contiguous():
    lib_a = _fake_library()  # 3 rows, T=6
    lib_b = commands.SwingLibrary(
        label="short", u=torch.ones(2, 3, 17) * 999.0,
        contact_step=torch.tensor([1, 1]), swing_start_step=torch.tensor([0, 0]),
        serve_scale=torch.tensor([1.0, 1.0]),
    )
    bank = commands.LibraryBank.build({"fake": lib_a, "short": lib_b})
    assert bank.row_offset == {"fake": 0, "short": 3}
    assert bank.combined.n_rows == 5
    assert bank.combined.n_steps == 6  # padded up to the longer library's T
    # short's rows repeat their own last frame (999) for the padded tail, per gather_u's clamp rule
    np.testing.assert_allclose(bank.combined.u[3, 5].numpy(), np.full(17, 999.0))
    np.testing.assert_allclose(bank.combined.u[3, 2].numpy(), np.full(17, 999.0))


def test_library_bank_sample_rows_only_draws_from_the_named_label():
    lib_a = _fake_library()
    lib_b = commands.SwingLibrary(
        label="short", u=torch.ones(2, 3, 17), contact_step=torch.tensor([1, 1]),
        swing_start_step=torch.tensor([0, 0]), serve_scale=torch.tensor([1.0, 1.0]),
    )
    bank = commands.LibraryBank.build({"fake": lib_a, "short": lib_b})
    gen = torch.Generator().manual_seed(0)
    rows = bank.sample_rows("short", 50, generator=gen)
    assert torch.all((rows >= 3) & (rows < 5))


def test_library_mix_schedule_sums_to_one_and_steps_at_thresholds():
    assert commands.library_mix_at(0) == {"slow": 0.70, "nominal": 0.30, "wide": 0.00}
    assert commands.library_mix_at(999) == {"slow": 0.70, "nominal": 0.30, "wide": 0.00}
    assert commands.library_mix_at(1000) == {"slow": 0.20, "nominal": 0.60, "wide": 0.20}
    assert commands.library_mix_at(3000) == {"slow": 0.10, "nominal": 0.50, "wide": 0.40}
    assert commands.library_mix_at(10 ** 9) == {"slow": 0.10, "nominal": 0.50, "wide": 0.40}


def test_swing_library_record_outcomes_ema_and_floor():
    lib = _fake_library()
    assert torch.all(lib.fail_ema == commands.ADAPTIVE_FLOOR)
    row_idx = torch.tensor([0])
    lib.record_outcomes(row_idx, torch.tensor([1.0]))  # one failure
    expected = (1.0 - commands.ADAPTIVE_EMA_ALPHA) * commands.ADAPTIVE_FLOOR + commands.ADAPTIVE_EMA_ALPHA * 1.0
    assert lib.fail_ema[0].item() == pytest.approx(expected, abs=1e-6)
    assert lib.fail_ema[1].item() == pytest.approx(commands.ADAPTIVE_FLOOR, abs=1e-6)  # untouched row


def test_post_swing_ring_push_and_sample():
    ring = commands.PostSwingRing(capacity=3)
    assert ring.filled == 0
    for i in range(5):  # exceeds capacity -> wraps
        ring.push({"tag": i})
    assert ring.filled == 3
    tags = {s["tag"] for s in [ring._states[i] for i in range(3)]}
    assert tags == {2, 3, 4}  # the oldest two (0, 1) were overwritten
    gen = torch.Generator().manual_seed(1)
    sampled = ring.sample(10, generator=gen)
    assert all(s["tag"] in {2, 3, 4} for s in sampled)


def test_sample_reset_sources_respects_min_fill_guard():
    gen = torch.Generator().manual_seed(2)
    labels_empty_ring = set(commands.sample_reset_sources(2000, ring_filled=0, generator=gen))
    assert "post_swing_ring" not in labels_empty_ring
    labels_full_ring = set(commands.sample_reset_sources(2000, ring_filled=commands.RING_MIN_FILL, generator=gen))
    assert "post_swing_ring" in labels_full_ring
