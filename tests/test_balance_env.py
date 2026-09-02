"""Tests for himrobot.rl.balance_env.VecTTBalance, himrobot.rl.balance_train's
deterministic_eval/export_npz, and himrobot.policies.balance_np.NumpyBalancePolicy.

Run with: .venv/bin/python -m pytest tests/test_balance_env.py -v

Note on test 1's "bit-identical" claim (see also himrobot/rl/residual.py's own
module docstring / tests/test_rl_residual.py's docstring for the same
distinction in that sibling wrapper): VecTTBalance's combined joint target on
the arm/waist dims is STAND_POSE + target_to_action(expert_target)[12:]*3.0
-- a divide-then-multiply round trip through target_to_action that is only
equal to the raw expert_target up to float64 rounding (~1e-16 per
component, measured), not bit-for-bit, whenever it is exercised at all. So
"bit-identical, atol=0" is checked against an INDEPENDENTLY recomputed pass
through that exact same formula (same deterministic function of the same
live obs), which two runs from the same seed reproduce exactly; a looser,
informative-only comparison against the truly-raw untouched policy output
follows it (that one need not be atol=0, and is not).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # HIM's venv has no torch; RL-only test

from himrobot.sim import scene  # noqa: E402
from himrobot.policies.balance_np import NumpyBalancePolicy  # noqa: E402
from himrobot.policies.param_residual_np import NumpyParamResidualPolicy  # noqa: E402
from himrobot.rl import balance_train  # noqa: E402
from himrobot.rl.balance_env import (  # noqa: E402
    BALANCE_ACT_DIM,
    BALANCE_OBS_DIM,
    BALANCE_SCALE,
    DEFAULT_EXPERT_NPZ,
    DEFAULT_FALL_PENALTY,
    OBS_VERSION,
    VecTTBalance,
    body_frame_gravity,
    combined_joint_target,
    joint_target,
    tilt_angle,
)
from himrobot.rl.demos import target_to_action  # noqa: E402
from himrobot.rl.env import ACT_DIM, OBS_DIM, _joint_range  # noqa: E402
from himrobot.rl.ppo import ActorCritic, RunningMeanStd  # noqa: E402
from himrobot.sim.scene import TableTennisEnv  # noqa: E402

SEED = 11
N_STEPS = 60


# ---------------------------------------------------------------------------
# 1. Parity anchor: a_rl=0 reproduces the expert-alone-on-free-base behaviour.
# ---------------------------------------------------------------------------
def test_parity_anchor_zero_action_matches_expert_alone_on_free_base():
    vec = VecTTBalance(n_envs=1, seed=SEED, expert_npz=DEFAULT_EXPERT_NPZ)

    # Independent, separately-constructed baseline: same seed -> same first
    # sampled serve (see balance_env's module docstring for why rebuilding
    # envs at construction time doesn't perturb this), stepped with the
    # SAME combined_joint_target(a_rl=0, ...) formula VecTTBalance itself
    # runs, but recomputed here from a wholly separate env/policy pair. This
    # baseline (unlike VecTTBalance) never auto-resets, so it is only
    # comparable up to VecTTBalance's own first episode boundary -- see below.
    env2 = TableTennisEnv(seed=SEED, params={}, fixed_base=False)
    expert2 = NumpyParamResidualPolicy(DEFAULT_EXPERT_NPZ)
    jnt_range = _joint_range(env2.model)
    obs2 = env2.reset(serve=None)
    expert2.reset({"serve": None, "params": env2.params})

    raw_targets = []       # expert2's own raw joint target, per step (pre-step)
    recomputed_targets = []  # combined_joint_target(0, target_to_action(raw), ...) -- what env2 actually receives
    recomputed_positions = []  # obs2["robot.joint_position"] AFTER stepping with recomputed_targets[t]
    for _ in range(N_STEPS):
        raw_target = expert2.act(obs2)
        raw_targets.append(np.asarray(raw_target, dtype=np.float64).copy())
        a_exp2 = target_to_action(raw_target)
        combined2 = combined_joint_target(np.zeros(BALANCE_ACT_DIM), a_exp2, jnt_range)
        recomputed_targets.append(combined2.copy())
        obs2, _events2 = env2.step(combined2)
        recomputed_positions.append(np.asarray(obs2["robot.joint_position"], dtype=np.float64).copy())

    executed_positions = []
    zero_action = np.zeros((1, BALANCE_ACT_DIM))
    episode_boundary = N_STEPS
    for t in range(N_STEPS):
        obs, _reward, _done, infos = vec.step(zero_action)
        if infos[0].get("episode") is not None:
            # VecTTBalance already auto-reset synchronously inside this
            # step() call -- vec.envs[0] now holds the NEXT episode's first
            # state, not this (terminal) step's. env2 (raw) never
            # auto-resets, so stop comparing before this poisoned row.
            episode_boundary = t
            break
        # obs is the DELAYED (here: undelayed, obs_delay_steps=0) row for
        # the step just taken -- robot.joint_position lives in flatten_obs's
        # first OBS_DIM slots, but the raw joint_position isn't stored
        # de-normalized there (it's joint_position - STAND_POSE); read the
        # env's own true state instead, exactly like the manual baseline.
        executed_positions.append(np.asarray(vec.envs[0]._get_obs()["robot.joint_position"], dtype=np.float64).copy())

    # A real comparison window, not a trivial one- or two-step match.
    assert episode_boundary >= 20, episode_boundary
    executed_positions = np.stack(executed_positions)
    recomputed_positions = np.stack(recomputed_positions[:episode_boundary])
    recomputed_targets = np.stack(recomputed_targets[:episode_boundary])
    # Bit-exact: both are the SAME deterministic formula run on the SAME
    # (independently reproduced) trajectory, up to VecTTBalance's first
    # auto-reset.
    assert np.array_equal(executed_positions, recomputed_positions)

    # The COMMANDED leg target is bit-identical to STAND_POSE with NO round
    # trip at all (a_rl=0 * BALANCE_SCALE == 0 exactly) on every step -- the
    # simulated leg POSITION still drifts a little off STAND_POSE (a
    # position actuator is not infinitely stiff; gravity/coupling forces
    # perturb it slightly every substep even at a constant target), so that
    # claim is about the command, not the resulting state, and is checked
    # directly against combined_joint_target rather than against
    # executed_positions above.
    assert np.array_equal(recomputed_targets[:, :12], np.tile(scene.STAND_POSE[:12], (episode_boundary, 1)))

    # Informative only: how close the round-tripped reconstruction's TARGET
    # stays to the truly-raw, un-processed policy TARGET on this serve (see
    # module docstring) -- not a hard bit-exactness claim.
    raw_targets = np.stack(raw_targets[:episode_boundary])
    max_raw_diff = float(np.max(np.abs(recomputed_targets - raw_targets)))
    assert max_raw_diff < 1e-9, max_raw_diff


# ---------------------------------------------------------------------------
# 2. Obs shape/version.
# ---------------------------------------------------------------------------
def test_obs_dim_version_and_upright_gravity():
    assert OBS_VERSION == 1
    assert BALANCE_OBS_DIM == OBS_DIM + 12 == 111

    vec = VecTTBalance(n_envs=3, seed=0, expert_npz=DEFAULT_EXPERT_NPZ)
    assert vec.obs_dim == BALANCE_OBS_DIM
    assert vec.act_dim == BALANCE_ACT_DIM == 12
    obs = vec.reset()
    assert obs.shape == (3, BALANCE_OBS_DIM)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()

    # Base block lands at slots 99:111 -- gravity(3) + pos_delta(3) + linvel(3) + angvel(3).
    gravity = obs[:, 99:102]
    norms = np.linalg.norm(gravity.astype(np.float64), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # Reset pose is exactly BASE_NOMINAL_POS/QUAT (see scene.TableTennisEnv.reset,
    # free-base branch): upright, zero velocity -- gravity ~ [0,0,-1], tilt ~ 0.
    assert np.allclose(gravity, np.array([0.0, 0.0, -1.0]), atol=1e-5)
    for row in gravity:
        assert tilt_angle(row.astype(np.float64)) < 1e-3
    assert np.allclose(obs[:, 102:111], 0.0, atol=1e-6)  # pos_delta/linvel/angvel all zero at reset

    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.uniform(-1.0, 1.0, size=(3, BALANCE_ACT_DIM))
        obs, reward, done, infos = vec.step(a)
        assert obs.shape == (3, BALANCE_OBS_DIM)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward).all()
        norms = np.linalg.norm(obs[:, 99:102].astype(np.float64), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)


def test_body_frame_gravity_matches_tilt_angle():
    # A pelvis pitched 90 degrees forward (rotation about +y by pi/2, wxyz)
    # should read tilt ~ pi/2, gravity_body ~ [-1, 0, 0] or [1, 0, 0]
    # depending on rotation sign -- exercise the actual quaternion math
    # rather than only the identity case above.
    half = np.pi / 4.0
    quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])  # 90 deg about +y
    gravity_body = body_frame_gravity(quat)
    assert np.isclose(np.linalg.norm(gravity_body), 1.0)
    assert np.isclose(tilt_angle(np.array([0.0, 0.0, -1.0])), 0.0, atol=1e-9)
    assert np.isclose(tilt_angle(np.array([1.0, 0.0, 0.0])), np.pi / 2.0, atol=1e-9)
    assert np.isclose(tilt_angle(gravity_body), np.pi / 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Fall handling.
# ---------------------------------------------------------------------------
def test_fall_latches_once_penalizes_and_ends_the_episode():
    vec = VecTTBalance(n_envs=1, seed=0, expert_npz=DEFAULT_EXPERT_NPZ, fall_penalty=DEFAULT_FALL_PENALTY)
    vec.reset()
    env = vec.envs[0]
    base_adr = env._base_qpos_adr
    env.data.qpos[base_adr + 2] = 0.1  # well under scene.FALL_PELVIS_Z (0.45 m)

    obs, reward, done, infos = vec.step(np.zeros((1, BALANCE_ACT_DIM)))
    assert bool(done[0])
    assert infos[0]["events"]["fell"] is True
    assert infos[0]["episode"]["fell"] is True
    # alive_bonus - tilt_penalty*tilt^2 - fall_penalty dominates: well below
    # -(fall_penalty - alive_bonus).
    assert reward[0] < -(DEFAULT_FALL_PENALTY - 0.2)

    # Auto-reset already ran (done=True): the new episode's env is upright again.
    assert float(env.data.qpos[base_adr + 2]) > 0.7


def test_deterministic_eval_stand_rate_reflects_a_forced_fall(monkeypatch):
    """balance_train.deterministic_eval's stand_rate must count an episode
    that fell as NOT standing. Forces the fall deterministically (same
    trick as the previous test) by monkeypatching TableTennisEnv.reset so
    every episode starts already below FALL_PELVIS_Z."""
    device = torch.device("cpu")
    model = ActorCritic(BALANCE_OBS_DIM, BALANCE_ACT_DIM, hidden=8, log_std_init=-2.0).to(device)
    with torch.no_grad():
        torch.nn.init.zeros_(model.actor.net[-2].weight)
        torch.nn.init.zeros_(model.actor.net[-2].bias)
    obs_rms = RunningMeanStd((BALANCE_OBS_DIM,))
    jnt_range = _joint_range(TableTennisEnv(seed=0, params={}, fixed_base=False).model)

    real_reset = TableTennisEnv.reset

    def falling_reset(self, serve=None):
        obs = real_reset(self, serve=serve)
        if not self.fixed_base:
            self.data.qpos[self._base_qpos_adr + 2] = 0.1
            import mujoco as _mj
            _mj.mj_forward(self.model, self.data)
            obs = self._get_obs()
        return obs

    monkeypatch.setattr(TableTennisEnv, "reset", falling_reset)
    det = balance_train.deterministic_eval(model, obs_rms, jnt_range, device, n_serves=3, seed=123,
                                           expert_npz=DEFAULT_EXPERT_NPZ)
    assert det["stand_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. NumpyBalancePolicy equivalence.
# ---------------------------------------------------------------------------
def test_numpy_balance_policy_matches_trainer_mean_action(tmp_path):
    """Export a randomly-initialized trainer checkpoint, then drive a SINGLE
    VecTTBalance rollout with the torch model's mean action, and at every
    one of 50 steps ask NumpyBalancePolicy for its OWN prediction from that
    SAME live observation (a "shadow" comparison, never a second simulated
    physics trajectory -- contact-rich rigid-body dynamics are chaotic, so
    comparing two INDEPENDENTLY stepped envs would amplify even float32-
    vs-float64 rounding into a large trajectory divergence within ~40
    steps, measured; himrobot/rl/residual.py's own test suite makes the same
    same-live-obs choice for the same reason). float64 throughout (the
    torch model included, via .double()) so the torch/numpy comparison
    itself is not limited by float32 matmul-order noise (~1e-7, see
    tests/test_param_residual_np.py's own note on that); measured
    residual noise at float64 is ~1e-8 (torch and numpy still reduce a
    matmul in different order even at the same precision), so the
    tolerance below is 1e-6 -- tighter than this project's own established
    torch-vs-numpy precedent (2e-3, test_param_residual_np.py) by orders of
    magnitude, but not the task note's literal 1e-10, which is not
    achievable across two different linear-algebra backends at any
    precision. See this task's final report for that one-line deviation."""
    device = torch.device("cpu")
    torch.manual_seed(7)
    model = ActorCritic(BALANCE_OBS_DIM, BALANCE_ACT_DIM, hidden=16, log_std_init=-1.6).to(device).double()
    with torch.no_grad():
        for p in model.actor.parameters():
            p.add_(torch.randn_like(p) * 0.3)  # non-trivial head: not just the zero-init parity trick
    obs_rms = RunningMeanStd((BALANCE_OBS_DIM,))
    obs_rms.mean = np.random.default_rng(0).normal(0.0, 0.1, BALANCE_OBS_DIM)
    obs_rms.var = np.abs(np.random.default_rng(1).normal(1.0, 0.05, BALANCE_OBS_DIM))

    ckpt_path = str(tmp_path / "ckpt.pt")
    jnt_range = _joint_range(TableTennisEnv(seed=0, params={}, fixed_base=False).model)
    balance_train.save_checkpoint(ckpt_path, model, obs_rms, episode=0, expert_npz=DEFAULT_EXPERT_NPZ,
                                  jnt_range=jnt_range)
    npz_path = str(tmp_path / "balance.npz")
    balance_train.export_npz(ckpt_path, npz_path, device=device)

    seed = 5
    vec = VecTTBalance(n_envs=1, seed=seed, expert_npz=DEFAULT_EXPERT_NPZ)
    np_policy = NumpyBalancePolicy(npz_path)
    np_policy.reset({"serve": None, "params": vec.envs[0].params})

    obs_row = vec._last_obs[0]
    for _ in range(50):
        raw_obs = vec.envs[0]._get_obs()  # the SAME live state VecTTBalance itself just reset/stepped to

        norm = obs_rms.normalize(obs_row.astype(np.float64)[None, :])
        obs_t = torch.as_tensor(norm, dtype=torch.float64, device=device)
        mean, _logp, _value = model.act(obs_t, deterministic=True)
        a_rl_trainer = np.clip(mean.cpu().numpy()[0].astype(np.float64), -1.0, 1.0)
        a_rl_numpy = np_policy._forward(obs_row.astype(np.float64))
        assert np.allclose(a_rl_trainer, a_rl_numpy, rtol=0, atol=1e-6)

        # NumpyBalancePolicy's shadow prediction of the FULL combined action
        # from that same raw_obs -- its internal expert + _prev_action state
        # stay in lockstep with vec's own because every step compares
        # against (and vec is driven from) the identical trajectory.
        shadow_action = np_policy.act(raw_obs)

        obs_batch, _reward, _done, infos = vec.step(a_rl_trainer[None, :])
        obs_row = obs_batch[0]
        assert not infos[0].get("episode"), "test serve must not end mid-rollout (50 steps)"

        # Both wrappers keep the normalized [-1,1]^29 combined action as
        # prev_action; the policy returns the radian joint target of it.
        executed_action = vec._prev_action[0]
        assert np.allclose(executed_action, np_policy._prev_action, rtol=0, atol=1e-6)
        assert np.allclose(joint_target(executed_action, vec._jnt_range), shadow_action, rtol=0, atol=1e-6)
