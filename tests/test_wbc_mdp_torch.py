"""Pure-torch parity/unit tests for himrobot.wbc.isaac.mdp -- WB6's tier-1/2
(no isaaclab, no stub gate; plain torch on CPU). See that package's
docstring for the three-tier split.

Genuine numpy-vs-torch parity: himrobot.wbc.obs / himrobot.wbc.action (WB3)
landed in this branch alongside this work item, so the tests below check the
torch functions directly against them (not an independently-coded stand-in)
-- exactly the parity docs/design/whole_body_v1.md sec 9 item 3 asks for.

Run with: .venv/bin/python -m pytest tests/test_wbc_mdp_torch.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from himrobot.sim import scene  # noqa: E402
from himrobot.wbc import action as wbc_action  # noqa: E402
from himrobot.wbc import obs as wbc_obs  # noqa: E402
from himrobot.wbc.isaac.mdp import actions, commands, observations, rewards, terminations  # noqa: E402


def _random_unit_quats(n: int, rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. quaternion rotation utilities vs himrobot.wbc.obs.
# ---------------------------------------------------------------------------
def test_body_frame_vector_matches_numpy_wbc_obs():
    rng = np.random.default_rng(0)
    quats = _random_unit_quats(1000, rng)
    vecs = rng.normal(size=(1000, 3))

    want = np.stack([wbc_obs.body_frame_vector(quats[i], vecs[i]) for i in range(1000)])
    got = observations.body_frame_vector(
        torch.as_tensor(quats, dtype=torch.float64), torch.as_tensor(vecs, dtype=torch.float64)
    ).numpy()
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_body_frame_gravity_matches_numpy_via_balance_env():
    """himrobot.wbc.obs re-exports himrobot.rl.balance_env.body_frame_gravity
    unmodified -- checked directly against that numpy function."""
    from himrobot.rl.balance_env import body_frame_gravity as np_body_frame_gravity

    rng = np.random.default_rng(1)
    quats = _random_unit_quats(300, rng)
    want = np.stack([np_body_frame_gravity(q) for q in quats])
    got = observations.body_frame_gravity(torch.as_tensor(quats, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_heading_cos_sin_matches_numpy_wbc_obs():
    rng = np.random.default_rng(2)
    quats = _random_unit_quats(500, rng)
    want = np.stack([wbc_obs.heading_cos_sin(q) for q in quats])
    got = observations.heading_cos_sin(torch.as_tensor(quats, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_quat_rotate_is_a_rotation_norm_preserving():
    rng = np.random.default_rng(3)
    quats = torch.as_tensor(_random_unit_quats(200, rng), dtype=torch.float64)
    vecs = torch.as_tensor(rng.normal(size=(200, 3)), dtype=torch.float64)
    rotated = observations.quat_rotate(quats, vecs)
    np.testing.assert_allclose(rotated.norm(dim=-1).numpy(), vecs.norm(dim=-1).numpy(), atol=1e-9)


# ---------------------------------------------------------------------------
# 2. default_pose / swing_clock vs himrobot.wbc.obs.
# ---------------------------------------------------------------------------
def test_default_pose_matches_numpy_wbc_obs():
    rng = np.random.default_rng(4)
    leg_defaults = rng.uniform(-0.3, 0.3, size=(50, 12))
    like = torch.zeros(50, 29, dtype=torch.float64)
    got = observations.default_pose(torch.as_tensor(leg_defaults, dtype=torch.float64), like).numpy()
    want = np.stack([wbc_obs.default_pose(leg_defaults[i]) for i in range(50)])
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_swing_clock_matches_numpy_wbc_obs():
    rng = np.random.default_rng(5)
    n = 500
    t_now = rng.uniform(0.0, 2.0, size=n)
    t_contact = rng.uniform(0.4, 1.4, size=n)
    t_pre = rng.uniform(0.2, 0.4, size=n)
    t_follow = rng.uniform(0.1, 0.3, size=n)
    speed_buffer = np.zeros(n)  # himrobot.wbc.replay_library._fake_plan's own convention

    want = np.stack([
        wbc_obs.swing_clock(float(t_now[i]),
                            {"t_contact": float(t_contact[i]), "speed_buffer": float(speed_buffer[i]),
                             "t_pre": float(t_pre[i]), "t_follow": float(t_follow[i])})
        for i in range(n)
    ])
    got = observations.swing_clock(
        torch.as_tensor(t_now), torch.as_tensor(t_contact), torch.as_tensor(speed_buffer),
        torch.as_tensor(t_pre), torch.as_tensor(t_follow),
    ).numpy()
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_swing_clock_no_plan_is_all_zero():
    zero = wbc_obs.swing_clock(0.5, None)
    np.testing.assert_allclose(zero, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# 3. observation layout dims and full flatten() parity.
# ---------------------------------------------------------------------------
def test_actor_and_critic_obs_dims_match_wbc_obs():
    assert observations.ACTOR_OBS_DIM == wbc_obs.OBS_DIM == 141
    assert observations.ACT_DIM == wbc_obs.ACT_DIM == 15
    assert observations.CRITIC_EXTRA_DIM == 35
    assert observations.CRITIC_OBS_DIM == 176


def _random_obs_dict(rng: np.random.Generator) -> dict:
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    return {
        "base.orientation": quat,
        "base.angular_velocity": rng.normal(size=3),
        "base.linear_velocity": rng.normal(size=3),
        "base.position": scene.BASE_NOMINAL_POS + rng.normal(size=3) * 0.1,
        "robot.joint_position": scene.STAND_POSE + rng.normal(size=29) * 0.05,
        "robot.joint_velocity": rng.normal(size=29),
        "time_s": float(rng.uniform(0.0, 2.0)),
    }


def test_compute_actor_obs_matches_himrobot_wbc_obs_flatten_over_1000_random_states():
    rng = np.random.default_rng(6)
    n = 1000
    obs_dicts = [_random_obs_dict(rng) for _ in range(n)]
    prev_actions = rng.uniform(-1.0, 1.0, size=(n, 15))
    leg_defaults = rng.uniform(-0.3, 0.3, size=(n, 12))
    u_now = np.stack([scene.STAND_POSE + rng.normal(size=29) * 0.1 for _ in range(n)])
    u_p5 = np.stack([scene.STAND_POSE + rng.normal(size=29) * 0.1 for _ in range(n)])
    u_p10 = np.stack([scene.STAND_POSE + rng.normal(size=29) * 0.1 for _ in range(n)])
    t_contact = rng.uniform(0.4, 1.4, size=n)
    t_pre = rng.uniform(0.2, 0.4, size=n)
    t_follow = rng.uniform(0.1, 0.3, size=n)

    want = np.stack([
        wbc_obs.flatten(
            obs_dicts[i], prev_actions[i], u_now[i], u_p5[i], u_p10[i],
            {"t_contact": t_contact[i], "speed_buffer": 0.0, "t_pre": t_pre[i], "t_follow": t_follow[i]},
            leg_defaults[i],
        )
        for i in range(n)
    ])

    base_quat = torch.as_tensor(np.stack([d["base.orientation"] for d in obs_dicts]), dtype=torch.float64)
    base_ang_vel = torch.as_tensor(np.stack([d["base.angular_velocity"] for d in obs_dicts]), dtype=torch.float64)
    base_lin_vel = torch.as_tensor(np.stack([d["base.linear_velocity"] for d in obs_dicts]), dtype=torch.float64)
    base_pos = torch.as_tensor(np.stack([d["base.position"] for d in obs_dicts]), dtype=torch.float64)
    joint_pos = torch.as_tensor(np.stack([d["robot.joint_position"] for d in obs_dicts]), dtype=torch.float64)
    joint_vel = torch.as_tensor(np.stack([d["robot.joint_velocity"] for d in obs_dicts]), dtype=torch.float64)
    t_now = torch.as_tensor(np.array([d["time_s"] for d in obs_dicts]), dtype=torch.float64)

    got = observations.compute_actor_obs(
        base_quat=base_quat, base_ang_vel=base_ang_vel, base_lin_vel_w=base_lin_vel, base_pos=base_pos,
        joint_pos=joint_pos, joint_vel=joint_vel, leg_default=torch.as_tensor(leg_defaults, dtype=torch.float64),
        prev_action=torch.as_tensor(prev_actions, dtype=torch.float64),
        u_now=torch.as_tensor(u_now, dtype=torch.float64), u_p5=torch.as_tensor(u_p5, dtype=torch.float64),
        u_p10=torch.as_tensor(u_p10, dtype=torch.float64),
        t_now=t_now, t_contact=torch.as_tensor(t_contact, dtype=torch.float64),
        speed_buffer=torch.zeros(n, dtype=torch.float64),
        t_pre=torch.as_tensor(t_pre, dtype=torch.float64), t_follow=torch.as_tensor(t_follow, dtype=torch.float64),
    ).numpy()

    np.testing.assert_allclose(got, want, atol=1e-6)


def test_compute_critic_extra_shape_and_values():
    n = 3
    u15 = torch.rand(n, 29)
    u25 = torch.rand(n, 29)
    serve_scale = torch.tensor([1.0, 1.3, 0.5])
    extra = observations.compute_critic_extra(u15, u25, serve_scale)
    assert extra.shape == (n, 35)
    cmd_default = observations._CMD_DEFAULT
    np.testing.assert_allclose(extra[:, :17].numpy(),
                               (u15[:, observations._CMD_SLICE_29] - cmd_default).numpy(), atol=1e-6)
    np.testing.assert_allclose(extra[:, 17:34].numpy(),
                               (u25[:, observations._CMD_SLICE_29] - cmd_default).numpy(), atol=1e-6)
    np.testing.assert_allclose(extra[:, 34].numpy(), serve_scale.numpy(), atol=1e-9)


# ---------------------------------------------------------------------------
# 4. action mapping vs himrobot.wbc.action.
# ---------------------------------------------------------------------------
def test_action_dim_and_scales_match_wbc_action():
    assert actions.ACTION_DIM == 15
    assert actions.LEG_SCALE == wbc_action.LEG_SCALE == 0.5
    assert actions.WAIST_SCALE == wbc_action.WAIST_SCALE == 0.2
    assert [b.name for b in actions.ACTION_BLOCKS] == ["legs", "waist_residual", "arms_passthrough"]
    assert [b.dim for b in actions.ACTION_BLOCKS] == [12, 3, 14]


def test_joint_target_wbc_matches_himrobot_wbc_action_over_1000_random_states():
    rng = np.random.default_rng(7)
    n = 1000
    a = rng.uniform(-1.5, 1.5, size=(n, 15))  # deliberately outside [-1,1] to exercise the clamp
    u = np.stack([scene.STAND_POSE + rng.normal(size=29) * 0.2 for _ in range(n)])
    leg_default = rng.uniform(-0.3, 0.3, size=(n, 12))
    jnt_range = np.tile(np.stack([np.full(29, -2.5), np.full(29, 2.5)], axis=-1), (n, 1, 1))

    want = np.stack([wbc_action.joint_target_wbc(a[i], u[i], jnt_range[i], leg_default[i]) for i in range(n)])
    got = actions.joint_target_wbc(
        torch.as_tensor(a), torch.as_tensor(u), torch.as_tensor(leg_default), torch.as_tensor(jnt_range),
    ).numpy()
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_joint_target_wbc_parity_anchor_zero_action_reproduces_hitter_output():
    """a = 0 with leg_default = STAND_POSE[:12] reproduces the hitter's own
    29-dim action exactly (docs/design/whole_body_v1.md sec 3.2)."""
    n = 10
    a = torch.zeros(n, 15, dtype=torch.float64)
    u = torch.as_tensor(np.tile(scene.STAND_POSE, (n, 1)), dtype=torch.float64)
    leg_default = torch.as_tensor(np.tile(scene.STAND_POSE[:12], (n, 1)), dtype=torch.float64)
    jnt_range = torch.as_tensor(np.tile(np.stack([np.full(29, -10.0), np.full(29, 10.0)], axis=-1), (n, 1, 1)),
                                dtype=torch.float64)

    target = actions.joint_target_wbc(a, u, leg_default, jnt_range).numpy()
    expected = np.tile(scene.STAND_POSE, (n, 1))
    np.testing.assert_allclose(target, expected, atol=1e-12)


def test_joint_target_wbc_clips_to_range():
    a = torch.zeros(1, 15)
    u = torch.full((1, 29), 100.0)  # way outside any real joint range
    leg_default = torch.full((1, 12), -100.0)
    jnt_range = torch.tensor([[[-1.0, 1.0]] * 29])
    target = actions.joint_target_wbc(a, u, leg_default, jnt_range)
    assert torch.all(target <= 1.0) and torch.all(target >= -1.0)


# ---------------------------------------------------------------------------
# 5. reward kernels -- 3 hand-built states with hand-computed values.
# ---------------------------------------------------------------------------
def test_reward_term_table_matches_design_doc():
    names = [t.name for t in rewards.REWARD_TERMS]
    assert names == [
        "station", "station_post_swing", "box_excursion", "height", "upright", "heading",
        "base_lin_vel", "base_ang_vel", "feet_contact", "foot_slip", "mocap_leg_prior",
        "action_rate", "leg_joint_vel", "leg_torque", "joint_limit", "waist_residual", "undesired_contact",
        "termination",
    ]
    assert rewards.REWARD_WEIGHTS["termination"] == -10.0  # sec 3.3: "a termination costs -10 once"
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
    pos = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.0, 2.0]])
    np.testing.assert_allclose(rewards.station_kernel(pos, target).numpy(), [1.0], atol=1e-12)

    pos_b = torch.tensor([[1.05, 2.0]])  # 0.05 m away == the std -> exp(-1)
    assert rewards.station_kernel(pos_b, target).item() == pytest.approx(math.exp(-1.0), abs=1e-6)

    pos_c = torch.tensor([[1.03, 2.04]])  # err^2 = 0.0025 == 0.05^2 -> exp(-1)
    assert rewards.station_kernel(pos_c, target).item() == pytest.approx(math.exp(-1.0), abs=1e-6)


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
    in_contact2 = torch.tensor([[False], [True]])
    np.testing.assert_allclose(rewards.foot_slip_penalty(foot_vel, in_contact2).numpy(), [0.0, 0.0], atol=1e-6)

    forces = torch.tensor([[0.0, 2.0], [0.5, 0.5]])
    np.testing.assert_allclose(rewards.undesired_contact_penalty(forces).numpy(), [1.0, 0.0])


def test_joint_limit_violation_and_action_rate_and_torque_penalties():
    q = torch.tensor([[0.0, -1.0, 2.0]])
    low = torch.tensor([[-0.5, -0.5, -0.5]])
    high = torch.tensor([[0.5, 0.5, 0.5]])
    assert rewards.joint_limit_violation(q, low, high).item() == pytest.approx(2.0, abs=1e-9)

    a = torch.tensor([[1.0, 2.0]])
    prev = torch.tensor([[0.0, 0.0]])
    assert rewards.action_rate_penalty(a, prev).item() == pytest.approx(5.0, abs=1e-9)

    torque = torch.tensor([[3.0, 4.0]])
    assert rewards.torque_penalty(torque).item() == pytest.approx(25.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. terminations.
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
# 7. SwingLibraryCommand's pure-torch gather/clock/library-mix logic, on a
#    fake 3-row library, CPU.
# ---------------------------------------------------------------------------
def _fake_library(label: str = "fake", serve_scale: float = 1.0) -> commands.SwingLibrary:
    # 3 rows, 6 steps, 17 channels; row r's u[t] = r*100 + t so a gathered
    # value pins down exactly which (row, step) was read.
    u = torch.zeros(3, 6, 17)
    for r in range(3):
        for t in range(6):
            u[r, t, :] = r * 100 + t
    return commands.SwingLibrary(
        label=label, u=u,
        swing_start_step=torch.tensor([1, 1, 0]),
        contact_step=torch.tensor([3, 4, 2]),
        t_contact=torch.tensor([0.06, 0.08, 0.04]),
        t_pre=torch.tensor([0.02, 0.02, 0.02]),
        t_follow=torch.tensor([0.02, 0.02, 0.02]),
        serve_scale=serve_scale,
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


def test_gather_scalar_reads_the_right_row():
    lib = _fake_library()
    row_idx = torch.tensor([2, 0, 1])
    got = commands.gather_scalar(lib.t_contact, row_idx)
    np.testing.assert_allclose(got.numpy(), [0.04, 0.06, 0.08], atol=1e-6)


def test_library_bank_pads_shorter_rows_and_broadcasts_serve_scale():
    lib_a = _fake_library("fake", serve_scale=1.0)
    lib_b = commands.SwingLibrary(
        label="short", u=torch.ones(2, 3, 17) * 999.0,
        swing_start_step=torch.tensor([0, 0]), contact_step=torch.tensor([1, 1]),
        t_contact=torch.tensor([0.02, 0.02]), t_pre=torch.tensor([0.0, 0.0]), t_follow=torch.tensor([0.0, 0.0]),
        serve_scale=1.3,
    )
    bank = commands.LibraryBank.build({"fake": lib_a, "short": lib_b})
    assert bank.row_offset == {"fake": 0, "short": 3}
    assert bank.n_rows == 5
    assert bank.n_steps == 6  # padded up to the longer library's T
    # short's rows repeat their own last frame (999) for the padded tail, per gather_u's clamp rule
    np.testing.assert_allclose(bank.u[3, 5].numpy(), np.full(17, 999.0))
    np.testing.assert_allclose(bank.u[3, 2].numpy(), np.full(17, 999.0))
    np.testing.assert_allclose(bank.serve_scale[:3].numpy(), np.full(3, 1.0))
    np.testing.assert_allclose(bank.serve_scale[3:].numpy(), np.full(2, 1.3))


def test_library_bank_sample_rows_only_draws_from_the_named_label():
    lib_a = _fake_library()
    lib_b = commands.SwingLibrary(
        label="short", u=torch.ones(2, 3, 17), swing_start_step=torch.tensor([0, 0]),
        contact_step=torch.tensor([1, 1]), t_contact=torch.tensor([0.02, 0.02]),
        t_pre=torch.tensor([0.0, 0.0]), t_follow=torch.tensor([0.0, 0.0]),
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


# ---------------------------------------------------------------------------
# 8. The 17 -> 29 padding, idle rows and mix restriction the command term
#    relies on -- the library stores JOINT_ORDER[12:29] only, while
#    compute_actor_obs / joint_target_wbc read [12:29] of a 29-vector.
# ---------------------------------------------------------------------------
def test_pad29_matches_replay_library_pad29_and_feeds_the_29_wide_consumers():
    from himrobot.wbc.replay_library import _pad29

    rng = np.random.default_rng(8)
    u17 = rng.normal(size=(4, 17))
    got = commands.pad29(torch.as_tensor(u17)).numpy()
    want = np.stack([_pad29(u17[i]) for i in range(4)])
    np.testing.assert_allclose(got, want, atol=1e-12)
    assert got.shape == (4, 29)

    # a padded row goes through the action mapping and lands on the arm/waist
    # slots untouched (a = 0, jnt_range wide open)
    a = torch.zeros(4, 15, dtype=torch.float64)
    leg_default = torch.as_tensor(np.tile(scene.STAND_POSE[:12], (4, 1)))
    jnt_range = torch.as_tensor(np.tile(np.stack([np.full(29, -10.0), np.full(29, 10.0)], -1), (4, 1, 1)))
    target = actions.joint_target_wbc(a, torch.as_tensor(got), leg_default, jnt_range).numpy()
    np.testing.assert_allclose(target[:, 12:], u17, atol=1e-12)
    np.testing.assert_allclose(target[:, :12], np.tile(scene.STAND_POSE[:12], (4, 1)), atol=1e-12)


def test_apply_idle_reads_stand_pose_on_flagged_rows_only():
    lib = _fake_library()
    u = commands.gather_u(lib.u, torch.tensor([0, 1, 2]), torch.tensor([3, 3, 3]))
    idle = torch.tensor([False, True, False])
    out = commands.apply_idle(u, idle)
    np.testing.assert_allclose(out[0].numpy(), np.full(17, 3.0))
    np.testing.assert_allclose(out[1].numpy(), scene.STAND_POSE[12:])
    np.testing.assert_allclose(out[2].numpy(), np.full(17, 203.0))


def test_restrict_mix_renormalises_to_loaded_labels_and_falls_back_to_uniform():
    mix = commands.library_mix_at(0)  # slow 0.7 / nominal 0.3 / wide 0
    got = commands.restrict_mix(mix, ["nominal", "wide"])
    assert got == {"nominal": 1.0}
    got = commands.restrict_mix(mix, ["slow", "nominal", "wide"])
    assert got == pytest.approx({"slow": 0.7, "nominal": 0.3})
    # Gate 0 loads a single "eval" library that no schedule entry names
    assert commands.restrict_mix(mix, ["eval"]) == {"eval": 1.0}
