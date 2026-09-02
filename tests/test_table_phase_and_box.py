"""The two balance v1.2 knobs, both default-off: robot_table_collision
(scene) and the stay-in-box excursion term (balance_env)."""
from __future__ import annotations

import mujoco
import numpy as np

from himrobot.rl.balance_env import box_excursion_sq
from himrobot.sim import scene


def _gid(model: mujoco.MjModel, ref: str | int) -> int:
    if isinstance(ref, int):
        return ref
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ref)
    assert g >= 0, ref
    return g


def _collides(model: mujoco.MjModel, a: str | int, b: str | int) -> bool:
    ga, gb = _gid(model, a), _gid(model, b)
    ta, aa = int(model.geom_contype[ga]), int(model.geom_conaffinity[ga])
    tb, ab = int(model.geom_contype[gb]), int(model.geom_conaffinity[gb])
    return bool((ta & ab) or (tb & aa))


def test_default_scene_xml_is_unchanged_by_the_knob():
    assert scene.DEFAULT_PARAMS["robot_table_collision"] is True
    assert scene._build_xml(False, True) == scene._build_xml(False, True, robot_table_collision=True)


def test_phase_through_keeps_ball_and_paddle_contacts_and_drops_robot_table():
    m = scene.build_model(fixed_base=False, robot_table_collision=False)
    assert _collides(m, "ball", "table_top")
    assert _collides(m, "ball", "net")
    assert _collides(m, "ball", "floor")
    assert _collides(m, "paddle_blade", "table_top")
    assert _collides(m, "ball", "paddle_blade")
    # a robot collision geom vs the table / net: no; vs the floor: yes
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    robot_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == pelvis and m.geom_contype[i]]
    assert robot_geoms  # the G1's collision geoms are unnamed: use ids
    g = robot_geoms[0]
    assert not _collides(m, g, "table_top")
    assert not _collides(m, g, "net")
    assert _collides(m, g, "floor")
    assert _collides(m, g, "ball")


def test_env_reads_the_param():
    env = scene.TableTennisEnv(seed=0, params={"robot_table_collision": False}, fixed_base=False)
    assert (int(env.model.geom_contype[_gid(env.model, "table_top")]),
            int(env.model.geom_conaffinity[_gid(env.model, "table_top")])) == (2, 2)
    assert _collides(env.model, "ball", "table_top")
    default = scene.TableTennisEnv(seed=0, params={}, fixed_base=False)
    assert (int(default.model.geom_contype[_gid(default.model, "table_top")]),
            int(default.model.geom_conaffinity[_gid(default.model, "table_top")])) == (1, 1)


def test_box_excursion_is_zero_inside_and_quadratic_outside():
    nominal = scene.BASE_NOMINAL_POS
    assert box_excursion_sq(nominal, 0.15, 0.30) == 0.0
    assert box_excursion_sq(nominal + np.array([0.1, -0.25, 0.0]), 0.15, 0.30) == 0.0
    out = box_excursion_sq(nominal + np.array([0.25, 0.0, 0.0]), 0.15, 0.30)
    assert abs(out - 0.1 ** 2) < 1e-12
    out2 = box_excursion_sq(nominal + np.array([0.25, -0.5, 5.0]), 0.15, 0.30)
    assert abs(out2 - (0.1 ** 2 + 0.2 ** 2)) < 1e-12  # z ignored
