"""Physics sanity tests for himrobot.sim.scene. Run with:
    .venv/bin/python -m pytest tests/test_scene.py -v
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from himrobot.sim.scene import (
    NUM_JOINTS,
    STAND_POSE,
    TABLE_HALF_LENGTH,
    TableTennisEnv,
    build_model,
    legal_return,
    sample_serve,
)


def test_model_loads_with_expected_dims():
    model = build_model(fixed_base=True, paddle=True)
    # 29 robot hinge joints (1 qpos/qvel each) + 1 ball freejoint (7 qpos, 6 qvel).
    assert model.nq == NUM_JOINTS + 7
    assert model.nv == NUM_JOINTS + 6
    assert model.nu == NUM_JOINTS
    for name in ("table_top", "net", "paddle_blade", "floor", "ball"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0, name


def test_robot_does_not_move_when_holding_current_pose():
    env = TableTennisEnv(seed=0, params={})
    obs = env.reset(serve={"pos": np.array([5.0, 5.0, 5.0]), "vel": np.zeros(3), "spin": None})
    assert np.allclose(obs["robot.joint_position"], STAND_POSE)

    for _ in range(500):  # 10 s of sim time
        obs, _ = env.step(STAND_POSE)

    drift = np.abs(obs["robot.joint_position"] - STAND_POSE).max()
    assert drift < 0.02, f"position actuators let the pose drift {drift:.4f} rad while holding"


def test_served_ball_bounces_once_on_robot_half_and_leaves_untouched():
    env = TableTennisEnv(seed=0, params={})
    rng = np.random.default_rng(123)
    serve = sample_serve(rng, {})
    assert serve["pos"][0] > 0, "serve must launch from the opponent's half (x>0)"

    obs = env.reset(serve=serve)
    info = {}
    for _ in range(400):  # 8 s of sim time
        obs, info = env.step(STAND_POSE)

    assert info["contact_paddle"] is False  # HoldPolicy-style no-op action, never swings
    assert info["bounce_own_side"] is True
    assert info["bounce_opponent_side"] is False
    # "leaves the table": ends up outside the table's x-extent (or has already
    # settled on the floor, both are "left the table").
    left_table = abs(float(obs["ball.position"][0])) > TABLE_HALF_LENGTH or info["out"]
    assert left_table


def test_scripted_paddle_in_ball_path_registers_contact():
    env = TableTennisEnv(seed=0, params={})
    # Park the ball out of the way first so reset()'s default serve doesn't
    # interfere, then place it on a collision course with the paddle by
    # setting qpos/qvel directly.
    env.reset(serve={"pos": np.array([5.0, 5.0, 5.0]), "vel": np.zeros(3), "spin": None})

    paddle_pos = env.data.geom_xpos[env._paddle_geom_id].copy()
    paddle_normal = env.data.geom_xmat[env._paddle_geom_id].reshape(3, 3)[:, 1]
    ball_start = paddle_pos - 0.15 * paddle_normal
    velocity = paddle_normal * 3.0

    env.data.qpos[env._ball_qpos_adr:env._ball_qpos_adr + 3] = ball_start
    env.data.qpos[env._ball_qpos_adr + 3:env._ball_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[env._ball_dof_adr:env._ball_dof_adr + 3] = velocity
    mujoco.mj_forward(env.model, env.data)

    hit = False
    for _ in range(100):
        _, info = env.step(STAND_POSE)
        if info["contact_paddle"]:
            hit = True
            break
    assert hit


def test_legal_return():
    env = TableTennisEnv(seed=0, params={})
    rng = np.random.default_rng(7)
    serve = sample_serve(rng, {})
    obs = env.reset(serve=serve)
    info = {}
    for _ in range(400):
        obs, info = env.step(STAND_POSE)
    assert legal_return(info) is False  # never touched -> not a legal return

    synthetic = {
        "contact_paddle": True,
        "bounce_own_side": False,
        "bounce_opponent_side": True,
        "net_hit": False,
        "out": False,
        "crossed_net": True,
        "landing_point": (0.75, 0.4),
        "reaction_latency_ms": 180.0,
    }
    assert legal_return(synthetic) is True


def test_1000_steps_runs_well_under_10s():
    env = TableTennisEnv(seed=0, params={})
    env.reset(serve={"pos": np.array([1.1, 0.0, 0.95]), "vel": np.array([-3.0, 0.0, 1.0]), "spin": None})

    start = time.time()
    for _ in range(1000):
        env.step(STAND_POSE)
    elapsed = time.time() - start

    print(f"\n1000 control steps in {elapsed:.3f}s ({1000 / elapsed:.0f} steps/s)")
    assert elapsed < 10.0
