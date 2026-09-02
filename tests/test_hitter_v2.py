"""Tests for "hitter v2": the swing library, the reach expansion and the v2
parameter space. Run with:
    .venv/bin/python -m pytest tests/test_hitter_v2.py -v

The load-bearing ones are the parity tests: v2 exists only as an *additive*
action space, so a hitter at SERVE_PARAM_NOMINAL_V2, a v2 policy at u = 0 and
any v1 checkpoint must all still be the plain hitter, bit for bit.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np
import pytest

from himrobot.sim import scene
from himrobot.policies.ik_hitter import (
    IK_INDICES,
    Y_REACH,
    Z_REACH,
    REACH_MAX_Y,
    REACH_MAX_Z,
    SERVE_PARAM_NOMINAL,
    SERVE_PARAM_NOMINAL_V2,
    SHAPE_FIELDS,
    SWING_TEMPLATES,
    TEMPLATE_ORDER,
    IKHitterPolicy,
    blend_shape,
    reach_box,
)
from himrobot.sim.scene import DEFAULT_PARAMS, TableTennisEnv, sample_serve
from tests.hitter_helpers import benign_contact, ik_pose_at, roll_actions

from himrobot.rl.demos import target_to_action
from himrobot.rl.env import map_action

try:  # HIM's worker has no torch: every parity / geometry test below must still run there.
    import torch
    from himrobot.rl import param_residual as pr
    from himrobot.rl import param_train
except ImportError:  # pragma: no cover
    torch = pr = param_train = None
needs_torch = pytest.mark.skipif(torch is None, reason="needs torch (himrobot.rl.param_residual)")

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "param_s3_actions.npz")


# -- parity ---------------------------------------------------------------
def test_v2_nominal_matches_plain_hitter():
    """Every v2 addition is the identity at nominal, so all three routes to a
    "nominal v2 hitter" reproduce the plain hitter bit for bit."""
    plain = roll_actions(IKHitterPolicy(), seed=0)

    tuned = IKHitterPolicy()
    tuned.set_serve_params(SERVE_PARAM_NOMINAL_V2)
    assert np.array_equal(plain, roll_actions(tuned, seed=0))

    ctor = IKHitterPolicy(serve_params=dict(SERVE_PARAM_NOMINAL_V2))
    assert np.array_equal(plain, roll_actions(ctor, seed=0))


@needs_torch
def test_v2_zero_vector_matches_plain_hitter():
    """The fourth route: the action space's u = 0 decodes to the plain hitter."""
    from_u = pr.hitter_from_u(np.zeros(pr.K_PARAMS_V2), pr.PARAM_SPECS_V2)
    assert np.array_equal(roll_actions(IKHitterPolicy(), seed=0), roll_actions(from_u, seed=0))


def test_v1_nominal_still_matches_under_the_v2_superset():
    """A v1-shaped dict must still set every v2 knob back to its nominal."""
    hitter = IKHitterPolicy()
    hitter.set_serve_params(gate_lift=0.8)
    hitter.set_serve_params(SERVE_PARAM_NOMINAL)  # v1 dict: resets the v2 keys too
    assert hitter.gate_lift == 0.0
    assert np.array_equal(roll_actions(IKHitterPolicy(), seed=1), roll_actions(hitter, seed=1))


@needs_torch
def test_v2_specs_extend_v1_without_touching_it():
    v1_literal = (
        ("x_hit_offset", "add", 0.08), ("y_hit_offset", "add", 0.10), ("z_hit_offset", "add", 0.10),
        ("t_hit_offset", "add", 0.04), ("swing_speed_scale", "mul", 2.0), ("pitch_offset", "add", 0.35),
        ("yaw_offset", "add", 0.35), ("aim_x_offset", "add", 0.35), ("aim_y_offset", "add", 0.35),
        ("flight_time_scale", "mul", 1.43),
    )
    assert pr.PARAM_SPECS == v1_literal  # a reorder or re-bound of a v1 knob breaks every shipped checkpoint
    assert pr.PARAM_SPECS_V2[:pr.K_PARAMS] == v1_literal
    assert pr.K_PARAMS_V2 == 20
    assert pr.PARAM_SPACES == {"v1": pr.PARAM_SPECS, "v2": pr.PARAM_SPECS_V2}
    assert pr.DEFAULT_PARAM_SPACE == "v1"
    assert set(pr.PARAM_NAMES_V2) <= set(SERVE_PARAM_NOMINAL_V2)


def test_v1_checkpoint_reproduces_its_golden_actions():
    """The real back-compatibility guard: a committed rollout of the shipped v1
    .npz, generated before hitter v2 existed. A round-trip test would only
    prove self-consistency; this proves nothing regressed."""
    from himrobot.policies.param_residual_np import S3_WEIGHTS, NumpyParamResidualPolicy

    assert os.path.isfile(GOLDEN), f"golden fixture missing ({GOLDEN}); scripts/make_golden_param_actions.py"
    assert os.path.isfile(S3_WEIGHTS), f"shipped v1 weights missing ({S3_WEIGHTS})"
    golden = np.load(GOLDEN, allow_pickle=False)
    policy = NumpyParamResidualPolicy(S3_WEIGHTS)
    assert policy.names == [str(n) for n in golden["names"]]

    env = TableTennisEnv(seed=0, params={})
    actions, params = [], []
    for seed in golden["seeds"]:
        obs = env.reset(serve=scene.sample_serve(np.random.default_rng(int(seed)), env.params))
        policy.reset({"params": env.params})
        for _ in range(60):
            a = policy.act(obs)
            actions.append(np.asarray(a, dtype=np.float64).copy())
            obs, _events = env.step(a)
        params.append(np.array([policy.last_params[n] for n in policy.names]))
    assert np.array_equal(np.stack(actions), golden["actions"])
    assert np.array_equal(np.stack(params), golden["params"])


# -- the swing library ----------------------------------------------------
def test_gates_at_zero_return_the_drive_template():
    """Identity, not equality: proves the bit-exact early-out actually fires."""
    hitter = IKHitterPolicy()
    assert blend_shape((0.0, 0.0, 0.0, 0.0), hitter._drive_shape) is hitter._drive_shape
    assert blend_shape((-1.0, -0.5, 0.0, -0.2)) is SWING_TEMPLATES["drive"]
    assert hitter._drive_shape == SWING_TEMPLATES["drive"]


def test_blend_stays_inside_the_template_hull():
    rng = np.random.default_rng(0)
    lo = {f: min(t[f] for t in SWING_TEMPLATES.values()) for f in SHAPE_FIELDS}
    hi = {f: max(t[f] for t in SWING_TEMPLATES.values()) for f in SHAPE_FIELDS}
    for _ in range(50):
        shape = blend_shape(rng.uniform(-1.0, 1.0, size=4))
        for f in SHAPE_FIELDS:
            assert lo[f] - 1e-12 <= shape[f] <= hi[f] + 1e-12, f


def test_a_single_saturated_gate_is_mostly_that_template():
    for i, name in enumerate(TEMPLATE_ORDER[1:]):
        gates = [0.0] * 4
        gates[i] = 1.0
        shape = blend_shape(gates)
        drive, target = SWING_TEMPLATES["drive"], SWING_TEMPLATES[name]
        for f in SHAPE_FIELDS:
            expected = (drive[f] + 4.0 * target[f]) / 5.0
            assert shape[f] == pytest.approx(expected), (name, f)


@pytest.mark.parametrize("i,name", list(enumerate(TEMPLATE_ORDER[1:])))
def test_each_gate_name_selects_its_template(i, name):
    """gate_<name> must reach blend_shape in the slot TEMPLATE_ORDER gives that
    template: the only binding is the 4-tuple in _maybe_replan, and nothing
    else in the suite would notice a permutation."""
    hitter = IKHitterPolicy()
    hitter.set_serve_params(**{f"gate_{name}": 1.0})
    env = TableTennisEnv(seed=0, params={})
    obs = env.reset(serve=sample_serve(np.random.default_rng(0), DEFAULT_PARAMS))
    hitter.reset({"params": DEFAULT_PARAMS})
    hitter.act(obs)
    assert hitter._plan is not None
    expected = blend_shape([1.0 if j == i else 0.0 for j in range(4)], hitter._drive_shape)
    assert hitter._plan["shape"] == expected
    signature = {"block": "speed_cap", "lift": "flight_time", "punch": "backswing", "crossbody": "x_hit"}[name]
    assert hitter._plan["shape"][signature] != SWING_TEMPLATES["drive"][signature]


@pytest.mark.parametrize("name", TEMPLATE_ORDER)
def test_each_template_runs_a_full_episode_cleanly(name):
    """200 control steps on a template's own shape: finite actions, every IK
    joint inside its range, and a real swing (the hitter left the stand pose)."""
    hitter = IKHitterPolicy()
    hitter._drive_shape = dict(SWING_TEMPLATES[name])
    actions = roll_actions(hitter, seed=0, steps=200)
    assert np.all(np.isfinite(actions))
    lo, hi = hitter._ik_jnt_range[:, 0], hitter._ik_jnt_range[:, 1]
    ik = actions[:, IK_INDICES]
    assert np.all(ik >= lo - 1e-9) and np.all(ik <= hi + 1e-9)
    assert np.abs(actions - scene.STAND_POSE).max() > 0.1


@needs_torch
@pytest.mark.parametrize("name", TEMPLATE_ORDER[1:])
def test_each_gate_drives_a_full_episode_cleanly(name):
    """Finite actions through the action space the policy actually has."""
    u = np.zeros(pr.K_PARAMS_V2)
    u[pr.PARAM_NAMES_V2.index(f"gate_{name}")] = 1.0
    actions = roll_actions(pr.hitter_from_u(u, pr.PARAM_SPECS_V2), seed=0, steps=200)
    assert np.all(np.isfinite(actions))


# -- reach expansion ------------------------------------------------------
def test_reach_box_is_the_v1_clamp_at_zero_and_the_envelope_at_one():
    for template in SWING_TEMPLATES.values():
        assert reach_box(template, 0.0, 0.0) == (
            (template["y_lo"], template["y_hi"]), (template["z_lo"], template["z_hi"]))
        assert reach_box(template, -3.0, -3.0) == reach_box(template, 0.0, 0.0)  # negatives clip
        assert reach_box(template, 1.0, 1.0) == (REACH_MAX_Y, REACH_MAX_Z)
        (ylo, yhi), (zlo, zhi) = reach_box(template, 0.5, 0.5)
        assert min(template["y_lo"], REACH_MAX_Y[0]) <= ylo <= max(template["y_lo"], REACH_MAX_Y[0])
        assert zlo <= template["z_lo"] and zhi >= template["z_hi"] - 1e-12
        assert ylo <= yhi and zlo <= zhi


def test_the_newly_opened_reach_is_collision_free():
    """The regions reach_expand_* newly opens must not put the local,
    warm-started solver on a self-colliding IK branch.

    Scope, stated honestly. Collision-freedom is not a property of the v1
    clamp box either: sampling the *existing* box at x=-1.20 on a 0.05 grid
    finds 99 colliding points of 285 -- 46 the paddle/wrist through the table
    (blade centre below TABLE_TOP_Z=0.76 inside the table footprint) and 53
    the pre-existing right-wrist roll/yaw self-contact at blade z 0.90-1.25,
    y in [-0.25, 0.15]. Both predate v2, so the newly-opened low-z sliver is
    excluded on those grounds and the wrist pair is filtered out explicitly,
    as is the idle left hand brushing the left hip when the waist bends
    (hitter_helpers.benign_contact, widened on 2026-08-30 to excuse the whole
    idle-left-arm-against-the-hips family, which lowered the pinned deep-plane
    counts from 27/57/90/277 to the 22/45/66/184 below) -- the right arm, torso
    and table must stay clear.
    The guard then covers the two regions v2 opens -- the lateral band
    y in [-0.90, -0.75) and the high-z band (1.35, 1.45] -- at the two
    shallowest hitting planes (drive -1.20, punch -1.14) and at the face
    normals the templates and pitch/yaw offsets can command there, and PINS
    the deeper planes (block -1.28, lift -1.31, crossbody -1.25, x_hit_coarse
    -1.50) at their measured collision counts: there the warm-started solver
    puts the right arm through the torso on 22-184 of 1450 sampled poses --
    an accepted limitation of the reach expansion that the policy has to
    learn around; this test fails only if it gets worse."""
    hitter = IKHitterPolicy()
    data = mujoco.MjData(hitter._ik_model)
    (ylo, _yhi), (_zlo, zhi) = reach_box(SWING_TEMPLATES["drive"], 1.0, 1.0)
    lateral = [(y, z) for y in np.arange(ylo, Y_REACH[0], 0.02)
               for z in np.arange(0.60, zhi + 1e-9, 0.04)]
    high_z = [(y, z) for y in np.arange(Y_REACH[0], Y_REACH[1] + 1e-9, 0.05)
              for z in np.arange(Z_REACH[1], zhi + 1e-9, 0.02)]
    normals = ((1.0, 0.0, 0.2), (1.0, 0.0, 0.7), (1.0, 0.0, -0.3), (1.0, 0.35, 0.2), (1.0, -0.35, 0.2))
    # Every plane a single gate at u = 1 (or x_hit_coarse at its bound) can
    # command, with the measured number of colliding points out of 1450 as the
    # pinned upper bound. 0 = a real guard; the rest record the accepted
    # limitation so a regression (more collisions) fails and an improvement
    # can lower the number.
    drive_x = SWING_TEMPLATES["drive"]["x_hit"]
    planes = {"drive": (drive_x, 0), "punch": (blend_shape([0, 0, 1, 0])["x_hit"], 0),
              "coarse_shallow": (drive_x + 0.30, 0),
              "crossbody": (blend_shape([0, 0, 0, 1])["x_hit"], 22), "block": (blend_shape([1, 0, 0, 0])["x_hit"], 45),
              "lift": (blend_shape([0, 1, 0, 0])["x_hit"], 66), "coarse_deep": (drive_x - 0.30, 184)}
    lo, hi = hitter._ik_jnt_range[:, 0] - 1e-9, hitter._ik_jnt_range[:, 1] + 1e-9
    for name, (x, allowed) in planes.items():
        colliding = []
        for normal in normals:
            for y, z in lateral + high_z:
                q, pairs = ik_pose_at(hitter, data, x, y, z, normal)
                assert np.all(q[IK_INDICES] >= lo) and np.all(q[IK_INDICES] <= hi), (name, y, z)
                real = [p for p in pairs if not benign_contact(p)]
                if real:
                    colliding.append((normal, round(float(y), 2), round(float(z), 2), sorted(set(real))[:2]))
        assert len(colliding) <= allowed, (name, x, len(colliding), allowed, colliding[:3])


def _per_joint_stand_excursion(specs, k_params, seeds=(0, 1), steps=90):
    """Max |target - STAND_POSE| per joint over every one-hot u = +-1 rollout."""
    worst = np.zeros(scene.NUM_JOINTS)
    for i in range(k_params):
        for extreme in (1.0, -1.0):
            u = np.zeros(k_params)
            u[i] = extreme
            hitter_u = pr.hitter_from_u(u, specs)
            for seed in seeds:
                targets = roll_actions(hitter_u, seed=seed, steps=steps)
                worst = np.maximum(worst, np.abs(targets - scene.STAND_POSE).max(axis=0))
    return worst


@needs_torch
def test_v2_does_not_widen_the_train_eval_pipeline_gap():
    """VecTTParam sends the hitter's target through target_to_action (which
    clips to +-3 rad from STAND_POSE and zeroes the legs) then map_action,
    while himrobot.eval.evaluate feeds the target straight to the env -- so a pose
    more than 3 rad from stand trains differently from how it evaluates.

    Measured: v1 already crosses the band on exactly one joint
    (right_shoulder_pitch: 3.25 rad, swing_speed_scale at its bound; v2 reaches
    3.27 rad there) and that joint's own limit caps the excursion at 3.29 rad.
    So the gate is per joint: every joint v1 keeps inside +-3 rad, v2 must keep
    inside +-3 rad too, and the one joint beyond the band may go no further
    than its physical limit allows."""
    hitter = IKHitterPolicy()
    jnt_range = np.array([
        hitter._ik_model.jnt_range[mujoco.mj_name2id(hitter._ik_model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in scene.JOINT_ORDER], dtype=np.float64)
    physical_cap = np.abs(jnt_range - scene.STAND_POSE[:, None]).max(axis=1)

    v1 = _per_joint_stand_excursion(pr.PARAM_SPECS, pr.K_PARAMS)
    v2 = _per_joint_stand_excursion(pr.PARAM_SPECS_V2, pr.K_PARAMS_V2)
    beyond_v1 = [scene.JOINT_ORDER[j] for j in np.flatnonzero(v1 > 3.0)]
    assert beyond_v1 == ["right_shoulder_pitch_joint"], beyond_v1
    band = np.where(v1 > 3.0, physical_cap, 3.0) + 1e-9
    over = [(scene.JOINT_ORDER[j], float(v1[j]), float(v2[j]), float(band[j]))
            for j in range(scene.NUM_JOINTS) if v2[j] > band[j]]
    assert not over, over

    # ...and inside the +-3 rad band the pipeline round-trip is exact.
    for target in roll_actions(hitter, seed=0, steps=90):
        if np.abs(target - scene.STAND_POSE).max() <= 3.0:
            assert np.allclose(map_action(target_to_action(target), jnt_range), target, atol=1e-9)


# -- action space ---------------------------------------------------------
@needs_torch
def test_v2_params_are_bounded_and_nominal_at_zero():
    zero = pr.unsquash_to_params(np.zeros(pr.K_PARAMS_V2), pr.PARAM_SPECS_V2)
    assert zero == pytest.approx(SERVE_PARAM_NOMINAL_V2)
    for extreme in (1.0, -1.0):
        values = pr.unsquash_to_params(np.full(pr.K_PARAMS_V2, extreme), pr.PARAM_SPECS_V2)
        for name, kind, bound in pr.PARAM_SPECS_V2:
            if kind == "add":
                assert abs(values[name]) <= bound + 1e-9
            else:
                assert 1.0 / bound - 1e-9 <= values[name] <= bound + 1e-9


@needs_torch
def test_every_v2_param_actually_changes_the_swing():
    """A knob that no longer reaches the hitter would silently shrink the action
    space. The two reach knobs only bite when the clamp binds, so they are
    probed alongside a hit offset that pushes the intercept onto the bound."""
    # (probe u applied alongside the knob under test, seeds to search)
    probes = {
        "reach_expand_y": (("y_hit_offset", -1.0), range(12)),
        "reach_expand_z": (("z_hit_offset", 1.0), range(12)),
    }
    for i, (name, _kind, _bound) in enumerate(pr.PARAM_SPECS_V2):
        probe, seeds = probes.get(name, (None, range(2)))
        u0 = np.zeros(pr.K_PARAMS_V2)
        if probe is not None:
            u0[pr.PARAM_NAMES_V2.index(probe[0])] = probe[1]
        u1 = u0.copy()
        u1[i] = 0.9
        changed = any(
            not np.array_equal(roll_actions(pr.hitter_from_u(u0, pr.PARAM_SPECS_V2), s, 60),
                               roll_actions(pr.hitter_from_u(u1, pr.PARAM_SPECS_V2), s, 60))
            for s in seeds)
        assert changed, name


# -- checkpoints ----------------------------------------------------------
@needs_torch
def test_checkpoints_record_the_space_they_were_trained_with(tmp_path):
    for space, specs in pr.PARAM_SPACES.items():
        model = pr.ParamActorCritic(pr.FEATURE_DIM, len(specs))
        path = str(tmp_path / f"{space}.pt")
        pr.save_checkpoint(path, model, pr.RunningMeanStd((pr.FEATURE_DIM,)), 0,
                           param_specs=specs)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["param_space"] == space
        assert tuple(tuple(s) for s in loaded["param_specs"]) == specs

        loaded_model, _rms, decoded = pr.load_checkpoint(path, torch.device("cpu"))
        assert decoded == specs
        assert loaded_model.param_space == space


@needs_torch
def test_a_v2_checkpoint_at_zero_is_still_the_plain_hitter(tmp_path):
    from himrobot.rl.policy import ParamResidualPolicy

    path = str(tmp_path / "v2_zero.pt")
    pr.save_checkpoint(path, pr.ParamActorCritic(pr.FEATURE_DIM, pr.K_PARAMS_V2),
                       pr.RunningMeanStd((pr.FEATURE_DIM,)), 0,
                       param_specs=pr.PARAM_SPECS_V2)
    policy = ParamResidualPolicy(ckpt_path=path)
    assert policy.param_space == "v2"
    assert policy.model.k_params == pr.K_PARAMS_V2
    assert np.array_equal(roll_actions(IKHitterPolicy(), seed=2), roll_actions(policy, seed=2))
    assert policy.last_params == pytest.approx(SERVE_PARAM_NOMINAL_V2)


@needs_torch
def test_resuming_a_v1_checkpoint_into_v2_keeps_the_v1_head(tmp_path):
    """The warm start: the first 10 coordinates are the v1 policy's, the 10 new
    ones stay at their zero init (= nominal = the v1 hitter)."""
    torch.manual_seed(0)
    src = pr.ParamActorCritic(pr.FEATURE_DIM, pr.K_PARAMS)
    with torch.no_grad():
        for p in src.parameters():  # actor AND critic: both must survive the resume
            p.add_(torch.randn_like(p) * 0.3)
    dst = pr.ParamActorCritic(pr.FEATURE_DIM, pr.K_PARAMS_V2, log_std_init=-2.0)
    param_train.copy_actor_prefix(src, dst, log_std_init=-2.0)
    for name, tensor in src.critic.state_dict().items():
        assert torch.equal(dst.critic.state_dict()[name], tensor), name

    feats = torch.randn(4, pr.FEATURE_DIM)
    with torch.no_grad():
        mean_src = src.actor(feats)
        mean_dst = dst.actor(feats)
    assert torch.allclose(mean_dst[:, :pr.K_PARAMS], mean_src)
    assert torch.count_nonzero(mean_dst[:, pr.K_PARAMS:]) == 0
    assert torch.allclose(dst.log_std[:pr.K_PARAMS], src.log_std)
    assert torch.allclose(dst.log_std[pr.K_PARAMS:], torch.full((pr.K_PARAMS,), -2.0))


# -- env + trainer --------------------------------------------------------
def test_reward_knobs_default_to_the_pre_v2_reward():
    """aim_bonus/aim_sigma/target_hit_bonus must reproduce the hard-coded
    +3*exp(-err^2/0.5^2) term exactly, and add nothing else."""
    from himrobot.rl.env import VecTT

    vec = VecTT(n_envs=1, seed=0)
    assert (vec.aim_bonus, vec.aim_sigma, vec.target_hit_bonus) == (3.0, 0.5, 0.0)

    obs = {"ball.position": np.zeros(3), "paddle.position": np.zeros(3),
           "robot.joint_velocity": np.zeros(scene.NUM_JOINTS)}
    events = {"contact_paddle": True, "bounce_own_side": True,
              "bounce_opponent_side": True, "crossed_net": True,
              "net_hit": False, "out": False, "landing_point": (1.0, 0.3)}
    vec._targets[0] = (1.15, 0.55)
    err = scene.landing_error_m(events, vec._targets[0])
    action = np.zeros(scene.NUM_JOINTS)

    vec._contact_latched[0] = True  # isolate the legal-return branch
    reward, _done = vec._reward_and_done(0, obs, events, action)
    assert reward == pytest.approx(vec.legal_bonus + 3.0 * np.exp(-err ** 2 / 0.5 ** 2))

    tuned = VecTT(n_envs=1, seed=0, aim_bonus=8.0, aim_sigma=0.25, target_hit_bonus=4.0)
    tuned._targets[0] = vec._targets[0]
    tuned._contact_latched[0] = True
    assert err <= 0.30  # (1.0, 0.3) vs (1.15, 0.55): inside the evaluator's radius
    reward2, _done = tuned._reward_and_done(0, obs, events, action)
    assert reward2 == pytest.approx(tuned.legal_bonus + 8.0 * np.exp(-err ** 2 / 0.25 ** 2) + 4.0)

    # ...and a legal return that lands on the table but 0.40 m from the target earns the kernel, not the +4.
    far = VecTT(n_envs=1, seed=0, aim_bonus=8.0, aim_sigma=0.25, target_hit_bonus=4.0)
    far._targets[0] = vec._targets[0]
    far._contact_latched[0] = True
    far_events = {**events, "landing_point": (0.75, 0.55)}
    far_err = scene.landing_error_m(far_events, far._targets[0])
    assert 0.30 < far_err < 0.45
    reward3, _done = far._reward_and_done(0, obs, far_events, action)
    assert reward3 == pytest.approx(far.legal_bonus + 8.0 * np.exp(-far_err ** 2 / 0.25 ** 2))


@needs_torch
def test_vec_param_env_runs_the_v2_space():
    def decide(features):
        u = np.zeros(pr.K_PARAMS_V2)
        u[pr.PARAM_NAMES_V2.index("gate_lift")] = 0.5
        return {"u": u}

    vec = pr.VecTTParam(n_envs=2, seed=0, decide_fn=decide, param_specs=pr.PARAM_SPECS_V2)
    assert vec.param_specs == pr.PARAM_SPECS_V2
    for _ in range(5):
        vec.step()
    assert vec._hitters[0].gate_lift == pytest.approx(0.5)


@needs_torch
def test_v2_train_smoke_writes_a_v2_checkpoint(tmp_path):
    out = str(tmp_path / "v2run")
    result = param_train.train(n_envs=2, episodes=4, seed=0, out=out, eval_every=1,
                               batch_episodes=2, minutes=1.0, param_space="v2",
                               aim_bonus=8.0, aim_sigma=0.30, target_hit_bonus=4.0)
    assert result["param_space"] == "v2" and result["k_params"] == pr.K_PARAMS_V2
    model, _rms, specs = pr.load_checkpoint(os.path.join(out, "final.pt"), torch.device("cpu"))
    assert model.k_params == pr.K_PARAMS_V2
    assert specs == pr.PARAM_SPECS_V2


@needs_torch
def test_param_space_flag_defaults_to_v1(tmp_path):
    out = str(tmp_path / "cli")
    result = param_train.main([
        "--n-envs", "2", "--episodes", "2", "--minutes", "1", "--batch-episodes", "2",
        "--eval-every", "1", "--out", out,
    ])
    assert result["param_space"] == "v1" and result["k_params"] == pr.K_PARAMS
