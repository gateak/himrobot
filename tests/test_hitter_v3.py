"""Tests for "hitter v3": the stance frame and the collision guard. Run with:
    .venv/bin/python -m pytest tests/test_hitter_v3.py -v

Two changes, both gated so the default tree is bit-identical:

  * the welded pelvis can slide laterally (scene.ROBOT_Y, read once at import
    from TT_ROBOT_Y), and the hitter's lateral reach constants slide with it
    (ik_hitter.STANCE_Y / _stance_y). At ROBOT_Y = 0.0 the scene XML string is
    byte-for-byte what it was and _stance_y returns its argument object, so
    nothing downstream does any float arithmetic it did not do before.
  * collision_guard, a per-serve knob defaulting to 0.0, that tightens three
    joint limits inside the deep/inner region for the duration of one solve.
    At nominal _solve_ik returns before the region test, so the off path is
    literally _solve_ik_stock.

The stance is deliberately an environment-read constant rather than a learned
coordinate (it cannot change mid-episode), and collision_guard is deliberately
NOT in PARAM_SPACES (K_PARAMS_V2 stays 20) -- see
work/tt/design/hitter_v3_reach.md 3.3/3.4.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import mujoco
import numpy as np
import pytest

from himrobot.paths import REPO_ROOT as _REPO_ROOT
from himrobot.sim import scene
from himrobot.policies.ik_hitter import (
    GUARD_BOX,
    IK_INDICES,
    IK_JOINT_NAMES,
    GUARD_REGION,
    REACH_MAX_Y,
    REACH_MAX_Z,
    SERVE_PARAM_NOMINAL,
    SERVE_PARAM_NOMINAL_V2,
    SERVE_PARAM_NOMINAL_V3,
    STANCE_Y,
    SWING_TEMPLATES,
    X_HIT,
    Y_REACH,
    Z_REACH,
    IKHitterPolicy,
    _stance_y,
    blend_shape,
)
from tests.hitter_helpers import (
    NORMALS,
    benign_contact,
    ik_pose_at,
    right_arm_collides,
    roll_actions,
)

try:  # HIM's worker has no torch: every parity / geometry test below must still run there.
    import torch
    from himrobot.rl import param_residual as pr
except ImportError:  # pragma: no cover
    torch = pr = None
needs_torch = pytest.mark.skipif(torch is None, reason="needs torch (himrobot.rl.param_residual)")

REPO_ROOT = str(_REPO_ROOT)
SHIFT = -0.10  # the stance the design measured; see hitter_v3_reach.md 2.1


def _in_stance(source: str, robot_y: float = SHIFT) -> dict:
    """Run `source` in a fresh interpreter with TT_ROBOT_Y set, and parse the
    single JSON object it prints. ROBOT_Y is read once at import, so a shifted
    stance is a new process -- which is also how a real run opts in."""
    env = {**os.environ, "TT_ROBOT_Y": repr(robot_y), "OMP_NUM_THREADS": "1",
           "PYTHONPATH": os.pathsep.join([REPO_ROOT, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    proc = subprocess.run([sys.executable, "-c", source], env=env, cwd=REPO_ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


# -- 1/2. the default stance is the tree we had ---------------------------
def test_nominal_stance_is_bit_identical():
    """Three routes to "nothing moved": the constant, the identity early-out,
    and a full rollout. blend_shape's own early-out is checked by identity, so
    a _stance_y that returned an equal-but-new tuple would fail here."""
    assert scene.ROBOT_Y == 0.0
    assert STANCE_Y == 0.0
    assert tuple(scene.BASE_NOMINAL_POS) == (scene.ROBOT_X, 0.0, scene.ROBOT_Z)

    box = (-0.75, 0.15)
    assert _stance_y(box) is box  # no float arithmetic at all at the nominal stance

    hitter = IKHitterPolicy()
    assert blend_shape((0.0, 0.0, 0.0, 0.0), hitter._drive_shape) is hitter._drive_shape
    assert blend_shape((-1.0, -0.5, 0.0, -0.2)) is SWING_TEMPLATES["drive"]

    nominal_v3 = IKHitterPolicy()
    nominal_v3.set_serve_params(SERVE_PARAM_NOMINAL_V3)  # collision_guard=0.0 included
    assert np.array_equal(roll_actions(IKHitterPolicy(), seed=0),
                          roll_actions(nominal_v3, seed=0))
    assert Y_REACH == (-0.75, 0.15) and REACH_MAX_Y == (-0.90, 0.15)


def test_the_shipped_golden_fixture_still_reproduces():
    """The committed pre-v2 rollout of the shipped v1 .npz, re-run after the
    stance frame landed. It is the only artifact that would catch a change to
    a template dict that the parity rollouts above happened to miss."""
    from himrobot.policies.param_residual_np import S3_WEIGHTS, NumpyParamResidualPolicy

    golden_path = os.path.join(os.path.dirname(__file__), "data", "param_s3_actions.npz")
    golden = np.load(golden_path, allow_pickle=False)
    policy = NumpyParamResidualPolicy(S3_WEIGHTS)
    env = scene.TableTennisEnv(seed=0, params={})
    actions = []
    for seed in golden["seeds"]:
        obs = env.reset(serve=scene.sample_serve(np.random.default_rng(int(seed)), env.params))
        policy.reset({"params": env.params})
        for _ in range(60):
            a = policy.act(obs)
            actions.append(np.asarray(a, dtype=np.float64).copy())
            obs, _events = env.step(a)
    assert np.array_equal(np.stack(actions), golden["actions"])


def test_scene_xml_is_byte_identical_at_the_default_stance():
    """The guard for _build_xml's {ROBOT_Y:g}: f"{0.0:g}" renders "0", so the
    pelvis line -- and therefore the whole XML string, every model hash and
    every golden fixture -- is exactly what it was before ROBOT_Y existed."""
    assert '<body name="pelvis" pos="-1.77 0 0.793" childclass="g1">' in scene._build_xml(True, True)


# -- 3. a non-zero stance moves the lateral frame and nothing else --------
_STANCE_PROBE = """
import json
from himrobot.sim import scene
from himrobot.policies import ik_hitter as ik
fields = ("x_hit", "y_lo", "y_hi", "z_lo", "z_hi",
          "face_pitch", "face_yaw", "dir_pitch", "dir_yaw")
print(json.dumps({
    "robot_y": scene.ROBOT_Y,
    "stance_y": ik.STANCE_Y,
    "base_nominal_pos": list(scene.BASE_NOMINAL_POS),
    "pelvis_line": [l.strip() for l in scene._build_xml(True, True).splitlines()
                    if 'name="pelvis"' in l][0],
    "x_hit": ik.X_HIT,
    "y_reach": list(ik.Y_REACH),
    "z_reach": list(ik.Z_REACH),
    "reach_max_y": list(ik.REACH_MAX_Y),
    "reach_max_z": list(ik.REACH_MAX_Z),
    "templates": {k: {f: t[f] for f in fields} for k, t in ik.SWING_TEMPLATES.items()},
}))
"""


def test_reach_boxes_follow_a_non_zero_stance():
    """Everything lateral slides by exactly the stance; x, z and every face /
    direction bias is untouched. The pelvis follows too -- the arm did not
    change, the robot did."""
    got = _in_stance(_STANCE_PROBE)

    assert got["robot_y"] == SHIFT and got["stance_y"] == SHIFT
    assert got["pelvis_line"] == '<body name="pelvis" pos="-1.77 -0.1 0.793" childclass="g1">'
    assert got["base_nominal_pos"] == [scene.ROBOT_X, SHIFT, scene.ROBOT_Z]

    assert got["y_reach"] == [Y_REACH[0] + SHIFT, Y_REACH[1] + SHIFT]
    assert got["reach_max_y"] == [REACH_MAX_Y[0] + SHIFT, REACH_MAX_Y[1] + SHIFT]

    # x and z are the same numbers they always were.
    assert got["x_hit"] == X_HIT
    assert got["z_reach"] == list(Z_REACH)
    assert got["reach_max_z"] == list(REACH_MAX_Z)

    for name, here in SWING_TEMPLATES.items():
        there = got["templates"][name]
        assert there["y_lo"] == here["y_lo"] + SHIFT, name
        assert there["y_hi"] == here["y_hi"] + SHIFT, name
        for field in ("x_hit", "z_lo", "z_hi", "face_pitch", "face_yaw", "dir_pitch", "dir_yaw"):
            assert there[field] == here[field], (name, field)


# -- 4/5/6. the collision guard -------------------------------------------
def _in_region_targets(n: int = 200, seed: int = 0) -> np.ndarray:
    """Random blade targets strictly inside GUARD_REGION."""
    dx, z_min, dy = GUARD_REGION
    rng = np.random.default_rng(seed)
    return np.stack([
        rng.uniform(scene.ROBOT_X - 0.30, scene.ROBOT_X + dx - 0.05, n),
        rng.uniform(scene.ROBOT_Y + dy + 0.05, scene.ROBOT_Y + 0.05, n),
        rng.uniform(z_min + 0.05, 1.35, n),
    ], axis=1)


def test_collision_guard_off_is_the_stock_solver():
    """At the nominal the dispatcher returns before the region test, so targets
    that are *inside* the region -- where the guard would otherwise fire --
    still go through the stock solver bit for bit."""
    hitter = IKHitterPolicy()
    assert hitter.collision_guard == 0.0
    n_des = np.array([1.0, 0.0, 0.2])
    n_des = n_des / np.linalg.norm(n_des)
    dx, z_min, dy = GUARD_REGION
    for p_des in _in_region_targets():
        assert (p_des[0] <= scene.ROBOT_X + dx and p_des[2] >= z_min
                and p_des[1] >= scene.ROBOT_Y + dy)
        q = scene.STAND_POSE.copy()
        got = hitter._solve_ik(q, p_des, n_des)
        want = hitter._solve_ik_stock(q, p_des, n_des)
        assert np.array_equal(got, want)

    guarded_off = IKHitterPolicy(serve_params={"collision_guard": 0.0})
    assert np.array_equal(roll_actions(IKHitterPolicy(), seed=0),
                          roll_actions(guarded_off, seed=0))


# The band the balls actually occupy (committed intercepts of the 400
# scoreboard trials: y in [-0.75, +0.05], z in [0.91, 1.35]), not the two bands
# hitter v2's reach knobs opened. 190 grid points x 5 face normals = 950 poses
# per plane, ~10-20 s each.
_OP_YS = np.round(np.arange(-0.85, 0.0501, 0.05), 3)
_OP_ZS = np.round(np.arange(0.90, 1.3501, 0.05), 3)
OP_BAND = [(float(y), float(z)) for y in _OP_YS for z in _OP_ZS]

# Measured today (work/tt/analysis/reach/v3_opband.py, v3_guard.py scan) and
# reproduced by this test to the pose. Guard ON: the region covers the whole
# band, so the colliding branch is gone except at the deepest plane. Guard OFF:
# a regression ceiling, not an endorsement -- up to a third of the deep-band
# poses drive the upper arm into the chest today, which is the latent defect
# the guard exists to remove for a widened serve set.
OP_BAND_RIGHT_ARM = {
    (-1.20, 1.0): 0, (-1.31, 1.0): 0, (-1.41, 1.0): 0, (-1.50, 1.0): 5,
    (-1.20, 0.0): 3, (-1.31, 0.0): 86, (-1.41, 0.0): 227, (-1.50, 0.0): 306,
}


@pytest.mark.parametrize("plane,guard", sorted(OP_BAND_RIGHT_ARM))
def test_the_operational_band_is_collision_free_under_the_guard(plane, guard):
    expected = OP_BAND_RIGHT_ARM[(plane, guard)]
    hitter = IKHitterPolicy(serve_params={"collision_guard": guard})
    data = mujoco.MjData(hitter._ik_model)
    # The instance's own limits, which the guard's per-call box must always
    # restore: every sampled pose has to stay inside them either way.
    limits = hitter._ik_jnt_range.copy()
    lo, hi = limits[:, 0] - 1e-9, limits[:, 1] + 1e-9
    colliding = []
    for normal in NORMALS:
        for y, z in OP_BAND:
            q, pairs = ik_pose_at(hitter, data, plane, y, z, normal)
            assert np.all(q[IK_INDICES] >= lo) and np.all(q[IK_INDICES] <= hi), (plane, y, z)
            real = [p for p in pairs if not benign_contact(p)]
            if right_arm_collides(real):
                colliding.append((normal, round(float(y), 2), round(float(z), 2), sorted(set(real))[:2]))
    assert np.array_equal(hitter._ik_jnt_range, limits)
    assert len(colliding) == expected, (plane, guard, len(colliding), expected, colliding[:3])


def test_the_guard_fits_the_50_hz_budget():
    """It is a keep-out, so it produces fewer contacts for mj_forward to solve
    and is genuinely cheaper than the stock solver on the same targets. The
    control step is 20 ms; one solve must stay well inside it."""
    targets = _in_region_targets(n=200, seed=1)
    n_des = np.array([1.0, 0.0, 0.2])
    n_des = n_des / np.linalg.norm(n_des)

    def mean_ms(hitter):
        q = scene.STAND_POSE.copy()
        hitter._solve_ik(q, targets[0], n_des)  # warm the caches
        t0 = time.perf_counter()
        for p_des in targets:
            hitter._solve_ik(q, p_des, n_des)
        return (time.perf_counter() - t0) * 1000.0 / len(targets)

    guard_off = mean_ms(IKHitterPolicy())
    guard_on = mean_ms(IKHitterPolicy(serve_params={"collision_guard": 1.0}))
    assert guard_on < 5.0, guard_on
    assert guard_on <= guard_off, (guard_on, guard_off)


# -- 7. the train/eval pipeline gap, at the new stance with the guard on ---
_PIPELINE_PROBE = """
import json
import numpy as np
import mujoco
from himrobot.sim import scene
from himrobot.policies.ik_hitter import IKHitterPolicy
from himrobot.rl import param_residual as pr
from himrobot.rl.demos import target_to_action
from himrobot.rl.env import map_action
from tests.hitter_helpers import roll_actions

hitter = IKHitterPolicy()
jnt_range = np.array([
    hitter._ik_model.jnt_range[mujoco.mj_name2id(hitter._ik_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in scene.JOINT_ORDER], dtype=np.float64)

worst = np.zeros(scene.NUM_JOINTS)
for i in range(pr.K_PARAMS_V2):
    for extreme in (1.0, -1.0):
        u = np.zeros(pr.K_PARAMS_V2)
        u[i] = extreme
        h = IKHitterPolicy()
        h.set_serve_params(pr.unsquash_to_params(u, pr.PARAM_SPECS_V2), collision_guard=1.0)
        for seed in (0, 1):
            worst = np.maximum(worst, np.abs(roll_actions(h, seed=seed, steps=90)
                                             - scene.STAND_POSE).max(axis=0))

roundtrip = True
for target in roll_actions(IKHitterPolicy(), seed=0, steps=90):
    if np.abs(target - scene.STAND_POSE).max() <= 3.0:
        roundtrip &= bool(np.allclose(map_action(target_to_action(target), jnt_range), target, atol=1e-9))

print(json.dumps({
    "worst": worst.tolist(),
    "physical_cap": np.abs(jnt_range - scene.STAND_POSE[:, None]).max(axis=1).tolist(),
    "roundtrip": roundtrip,
    "guard_is_on": IKHitterPolicy(serve_params={"collision_guard": 1.0}).collision_guard == 1.0,
}))
"""


@needs_torch
def test_v3_does_not_widen_the_train_eval_pipeline_gap():
    """The v2 test's R3 guard, re-run at the shifted stance with the guard on:
    VecTTParam sends the hitter's target through target_to_action (which clips
    to +-3 rad from STAND_POSE) while himrobot.eval.evaluate feeds it straight to the
    env, so a pose more than 3 rad from stand trains differently from how it
    evaluates. The stance moves every commanded pose, so the band has to be
    re-checked; only right_shoulder_pitch_joint may leave it, and only as far
    as its own joint limit allows."""
    got = _in_stance(_PIPELINE_PROBE)
    assert got["guard_is_on"]

    worst = np.array(got["worst"])
    cap = np.array(got["physical_cap"])
    beyond = [scene.JOINT_ORDER[j] for j in np.flatnonzero(worst > 3.0)]
    assert beyond == ["right_shoulder_pitch_joint"], beyond
    band = np.where(worst > 3.0, cap, 3.0) + 1e-9
    over = [(scene.JOINT_ORDER[j], float(worst[j]), float(band[j]))
            for j in range(scene.NUM_JOINTS) if worst[j] > band[j]]
    assert not over, over
    assert got["roundtrip"]


# -- 8. the v3 nominal is a superset; the action space is untouched --------
@needs_torch
def test_v3_nominal_superset_and_v2_dicts_still_reset():
    """collision_guard lives in the hitter's nominal, NOT in a parameter space:
    it is a switch whose sign is known from the stance, so there is nothing
    per-serve for PPO to learn (hitter_v3_reach.md 3.4). K_PARAMS_V2 stays 20,
    every shipped checkpoint keeps decoding the way it always did, and a v1 or
    v2 dict still resets *every* key -- including the new one."""
    assert set(SERVE_PARAM_NOMINAL_V3) == set(SERVE_PARAM_NOMINAL_V2) | {"collision_guard"}
    assert all(SERVE_PARAM_NOMINAL_V3[k] == v for k, v in SERVE_PARAM_NOMINAL_V2.items())
    assert SERVE_PARAM_NOMINAL_V3["collision_guard"] == 0.0

    assert pr.K_PARAMS_V2 == 20
    assert pr.PARAM_SPACES == {"v1": pr.PARAM_SPECS, "v2": pr.PARAM_SPECS_V2}
    assert pr.DEFAULT_PARAM_SPACE == "v1"
    assert "collision_guard" not in pr.PARAM_NAMES_V2
    assert set(pr.PARAM_NAMES_V2) <= set(SERVE_PARAM_NOMINAL_V3)

    hitter = IKHitterPolicy()
    hitter.set_serve_params(collision_guard=1.0)
    assert hitter.collision_guard == 1.0
    hitter.set_serve_params(SERVE_PARAM_NOMINAL_V2)  # a v2 dict resets the v3 key
    assert hitter.collision_guard == 0.0
    hitter.set_serve_params(collision_guard=1.0)
    hitter.set_serve_params(SERVE_PARAM_NOMINAL)     # ...and so does a v1 dict
    assert hitter.collision_guard == 0.0

    with pytest.raises(ValueError):
        hitter.set_serve_params(no_such_knob=1.0)


def test_the_guard_box_and_region_are_pelvis_relative():
    """The region is arm geometry in x and y (so it follows the stance) and the
    table in z (so it does not). GUARD_BOX's three bounds are inside the joints'
    real limits, or the guard would be a no-op."""
    dx, z_min, dy = GUARD_REGION
    assert (dx, z_min, dy) == (0.67, 0.85, -0.45)
    assert scene.ROBOT_X + dx == pytest.approx(-1.10)
    assert scene.ROBOT_Y + dy == pytest.approx(-0.45)  # the default stance

    hitter = IKHitterPolicy()
    rng = hitter._ik_jnt_range
    names = ("right_shoulder_roll_joint", "waist_yaw_joint", "waist_roll_joint")
    for name, bound in zip(names, GUARD_BOX, strict=True):  # strict: GUARD_BOX arity is pinned
        lo, hi = rng[IK_JOINT_NAMES.index(name)]
        assert lo < bound < hi, name


# -- the depth-shortfall reward knob (design 2.3) -------------------------
def test_depth_bonus_defaults_to_the_pre_v3_reward():
    """depth_bonus = 0.0 must add exactly nothing, and at a positive weight it
    must price the shortfall only -- landing long is the aim kernel's job."""
    from himrobot.rl.env import VecTT

    vec = VecTT(n_envs=1, seed=0)
    assert vec.depth_bonus == 0.0

    obs = {"ball.position": np.zeros(3), "paddle.position": np.zeros(3),
           "robot.joint_velocity": np.zeros(scene.NUM_JOINTS)}
    events = {"contact_paddle": True, "bounce_own_side": True,
              "bounce_opponent_side": True, "crossed_net": True,
              "net_hit": False, "out": False, "landing_point": (0.60, -0.30)}
    action = np.zeros(vec.act_dim)
    target = np.asarray(scene.target_point("crosscourt", "deep"), dtype=np.float64)
    short = float(target[0]) - 0.60
    assert short > 0.0

    def reward(depth_bonus, landing_x):
        env = VecTT(n_envs=1, seed=0, depth_bonus=depth_bonus)
        env._targets[0] = target
        ev = {**events, "landing_point": (landing_x, -0.30)}
        return env._reward_and_done(0, obs, ev, action)[0]

    base = reward(0.0, 0.60)
    assert reward(3.0, 0.60) == pytest.approx(base - 3.0 * short)
    # Landing long (past the target x) is not priced at all.
    long_x = float(target[0]) + 0.20
    assert reward(3.0, long_x) == pytest.approx(reward(0.0, long_x))


@needs_torch
def test_depth_bonus_reaches_the_param_env_and_the_cli(monkeypatch, tmp_path):
    """param_train --depth-bonus -> train(depth_bonus=) -> VecTTParam's
    **env_kwargs -> VecTT. A knob that never arrives is worse than no knob."""
    from himrobot.rl import param_train
    from himrobot.rl.param_residual import VecTTParam

    vec = VecTTParam(n_envs=1, seed=0, depth_bonus=2.5)
    assert vec.depth_bonus == 2.5
    assert VecTTParam(n_envs=1, seed=0).depth_bonus == 0.0

    seen = {}
    monkeypatch.setattr(param_train, "train", lambda **kw: seen.update(kw) or {})
    param_train.main(["--out", str(tmp_path)])
    assert seen["depth_bonus"] == 0.0
    param_train.main(["--out", str(tmp_path), "--depth-bonus", "3"])
    assert seen["depth_bonus"] == 3.0
