"""Tests for the WBC MuJoCo-mirror artefacts (WB4, docs/design/
whole_body_v1.md sec 9 item 4): himrobot.wbc.mujoco_mirror's generated
assets/wbc/g1_29dof_paddle_free.xml + actuators_mujoco.json, and
himrobot.wbc.replay_library's open-loop / closed-loop replay.

    .venv/bin/python -m pytest tests/test_wbc_mirror.py -v

himrobot.wbc.record_library.py (WB2) has not landed at the time this file
was written, so (c) below builds its own small "prototype-style" library
in-process (a few real IKHitterPolicy rollouts on the welded scene, plus one
explicit idle row) rather than importing a module that does not exist yet --
this is a stand-in for the design doc's "40-row prototype ... recorded via
WB2 with n=3", scaled to n=3 real swing rows for the same reason the design
doc gives (kept fast).
"""
from __future__ import annotations

import json
import os
import time

import mujoco
import numpy as np
import pytest

from himrobot.paths import ASSETS_DIR
from himrobot.policies.param_residual_np import NumpyParamResidualPolicy
from himrobot.sim import scene
from himrobot.wbc import mujoco_mirror
from himrobot.wbc import replay_library as rl

XML_PATH = mujoco_mirror.XML_PATH
JSON_PATH = mujoco_mirror.JSON_PATH
V3_WEIGHTS = str(ASSETS_DIR / "policies" / "param_residual_v3_s0final.npz")


# ---------------------------------------------------------------------------
# (a) Regenerating actuators_mujoco.json in-process reproduces the committed
# file exactly, including the design doc's own spot-checked numbers.
# ---------------------------------------------------------------------------
def test_actuators_json_matches_committed_file():
    fresh = json.loads(json.dumps(mujoco_mirror.build_actuators_json()))
    with open(JSON_PATH) as fh:
        committed = json.load(fh)
    assert fresh == committed


def test_actuators_json_numbers():
    data = mujoco_mirror.build_actuators_json()
    joints = data["joints"]
    assert set(joints) == set(scene.JOINT_ORDER)

    for name, row in joints.items():
        assert row["kp"] == 500.0, name
        assert row["armature"] == pytest.approx(0.01), name
        assert row["frictionloss"] == pytest.approx(0.3), name

    spot_checks = {
        "left_hip_pitch_joint": (43.011, [-88.0, 88.0]),
        "left_ankle_roll_joint": (4.558, [-50.0, 50.0]),
        "waist_roll_joint": (37.463, [-50.0, 50.0]),
        "right_wrist_pitch_joint": (6.543, [-5.0, 5.0]),
    }
    for name, (kv, effort) in spot_checks.items():
        assert joints[name]["kv"] == pytest.approx(kv, abs=5e-4), name
        assert joints[name]["effort_limit"] == pytest.approx(effort), name

    assert data["total_mass_kg"] == pytest.approx(33.51114, abs=1e-4)
    assert data["fk_truth"]["min_foot_sphere_z"] == pytest.approx(0.0011, abs=2e-4)


# ---------------------------------------------------------------------------
# (b) The mirror XML loads, has the right joints/body, and matches the full
# free-base scene's mass and paddle FK at STAND_POSE / BASE_NOMINAL_POS.
# ---------------------------------------------------------------------------
def _stand_at_nominal(m: mujoco.MjModel) -> mujoco.MjData:
    d = mujoco.MjData(m)
    qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in scene.JOINT_ORDER])
    base_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    base_qadr = int(m.jnt_qposadr[base_jid])
    d.qpos[base_qadr:base_qadr + 3] = scene.BASE_NOMINAL_POS
    d.qpos[base_qadr + 3:base_qadr + 7] = scene.BASE_NOMINAL_QUAT
    d.qpos[qadr] = scene.STAND_POSE
    mujoco.mj_forward(m, d)
    return d


def test_mirror_xml_loads_with_expected_joints_and_body():
    m = mujoco.MjModel.from_xml_path(str(XML_PATH))

    hinge = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    free = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
           for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert hinge == list(scene.JOINT_ORDER)
    assert free == ["floating_base_joint"]

    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "paddle") >= 0


def test_mirror_xml_mass_matches_full_scene_minus_ball():
    m = mujoco.MjModel.from_xml_path(str(XML_PATH))
    mirror_mass = float(sum(m.body_mass))

    full = scene.build_model(fixed_base=False, paddle=True, robot_table_collision=False)
    ball_bid = mujoco.mj_name2id(full, mujoco.mjtObj.mjOBJ_BODY, "ball")
    full_mass_minus_ball = float(sum(full.body_mass) - full.body_mass[ball_bid])

    assert mirror_mass == pytest.approx(full_mass_minus_ball, abs=1e-9)
    assert mirror_mass == pytest.approx(scene.build_model(
        fixed_base=False, paddle=True, robot_table_collision=False).body_mass.sum() - scene.BALL_MASS, abs=1e-9)


def test_mirror_xml_paddle_fk_matches_full_scene_at_stand_pose():
    m = mujoco.MjModel.from_xml_path(str(XML_PATH))
    d = _stand_at_nominal(m)
    blade_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "paddle_blade")

    full = scene.build_model(fixed_base=False, paddle=True, robot_table_collision=False)
    fd = _stand_at_nominal(full)
    full_blade_gid = mujoco.mj_name2id(full, mujoco.mjtObj.mjOBJ_GEOM, "paddle_blade")

    assert np.allclose(d.geom_xpos[blade_gid], fd.geom_xpos[full_blade_gid], atol=1e-9, rtol=0)
    n_mirror = d.geom_xmat[blade_gid].reshape(3, 3)[:, 2]
    n_full = fd.geom_xmat[full_blade_gid].reshape(3, 3)[:, 2]
    assert np.allclose(n_mirror, n_full, atol=1e-9, rtol=0)


def test_mirror_xml_ten_zero_action_steps_no_nan():
    """Gate 1's own smoke check (docs/design/whole_body_v1.md sec 7.2), run
    here on the Mac side: the bare mirror model (no scene chrome at all)
    must step cleanly holding STAND_POSE."""
    m = mujoco.MjModel.from_xml_path(str(XML_PATH))
    d = _stand_at_nominal(m)
    actuator_ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in scene.JOINT_ORDER])
    d.ctrl[actuator_ids] = scene.STAND_POSE
    for _ in range(10):
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos))
    assert np.all(np.isfinite(d.qvel))


# ---------------------------------------------------------------------------
# (c) replay_library: a small in-process "prototype-style" library (see
# module docstring), --ball-parked mode, a=0 open loop.
# ---------------------------------------------------------------------------
N_STEPS = 150


def _record_swing_row(seed: int) -> dict:
    """One real IKHitterPolicy (v3) rollout on the WELDED scene, in the
    schema documented by himrobot.wbc.replay_library's module docstring."""
    env = scene.TableTennisEnv(seed=seed, params={})
    policy = NumpyParamResidualPolicy(V3_WEIGHTS)
    rng = np.random.default_rng(seed)
    serve = scene.sample_serve(rng, env.params)
    obs = env.reset(serve=serve)
    policy.reset({"serve": serve, "params": env.params})

    u = np.zeros((N_STEPS, 17), dtype=np.float32)
    events: dict = {}
    for t in range(N_STEPS):
        action = policy.act(obs)
        u[t] = action[12:29]
        obs, events = env.step(action)

    default = scene.STAND_POSE[12:]
    moved = np.where(np.any(np.abs(u - default) > 1e-6, axis=1))[0]
    swing_start = int(moved[0]) if len(moved) else 0
    # No ground truth for "contact" without WB1's plan() replayed step by
    # step (out of scope here) -- the step of peak upper-body joint speed is
    # a reasonable proxy, and is all this test needs (a plausible contact
    # step to center the pelvis-xy-error window on).
    speed = np.linalg.norm(np.diff(u, axis=0), axis=1)
    contact_step = int(np.argmax(speed)) if len(speed) else swing_start

    return dict(
        u=u, swing_start_step=swing_start, contact_step=contact_step,
        t_contact=contact_step / scene.CONTROL_HZ, t_pre=0.3, t_follow=0.2,
        serve_pos=serve["pos"].astype(np.float32), serve_vel=serve["vel"].astype(np.float32),
        welded_legal=bool(scene.legal_return(events)), welded_contact=bool(events.get("contact_paddle")),
    )


def _idle_row() -> dict:
    return dict(
        u=np.tile(scene.STAND_POSE[12:], (N_STEPS, 1)).astype(np.float32),
        swing_start_step=0, contact_step=0, t_contact=0.0, t_pre=0.1, t_follow=0.1,
        serve_pos=np.zeros(3, dtype=np.float32), serve_vel=np.zeros(3, dtype=np.float32),
        welded_legal=False, welded_contact=False,
    )


def _write_library(path: str, rows: list[dict]) -> None:
    stacked = {k: np.stack([r[k] for r in rows]) for k in rl.LIBRARY_KEYS}
    np.savez(path, **stacked)


@pytest.fixture(scope="module")
def prototype_library(tmp_path_factory) -> tuple[str, int]:
    """3 real swing rows (seeds 5000-5002, disjoint from the scoreboard's
    11/2000/4000 and record_library.py's own >=5000 bank per preview_error.py)
    + 1 idle row, in himrobot.wbc.replay_library's documented npz schema."""
    rows = [_record_swing_row(5000 + i) for i in range(3)] + [_idle_row()]
    path = str(tmp_path_factory.mktemp("wbc_lib") / "proto_lib.npz")
    _write_library(path, rows)
    return path, len(rows)


def test_replay_idle_row_ball_parked_no_falls_pelvis_holds(prototype_library):
    path, n_rows = prototype_library
    result = rl.run(path, mode="ball-parked", leg_default_name="stand",
                    wbc_npz=None, max_rows=n_rows, seed=0)
    idle_row = result["per_row"][-1]  # the idle row was appended last
    assert idle_row["fell"] is False
    assert idle_row["welded_legal"] is False and idle_row["welded_contact"] is False
    # pelvis-xy error is only recorded from base.position, which is what
    # "drift" means here -- < 1 mm over the full 150-step episode.
    assert idle_row["pelvis_xy_err_at_contact_m"] < 1e-3


def test_replay_swinging_row_ball_parked_runs_and_reports(prototype_library):
    path, n_rows = prototype_library
    result = rl.run(path, mode="ball-parked", leg_default_name="stand",
                    wbc_npz=None, max_rows=n_rows, seed=0)
    assert result["n_rows"] == n_rows
    assert result["tilt_p90_rad"] is not None and result["tilt_p90_rad"] >= 0.0
    swing_rows = result["per_row"][:3]
    assert all(isinstance(r["fell"], bool) for r in swing_rows)
    # At least one swinging row's per-row report carries a real tilt reading
    # and (see docs/design/whole_body_v1.md: waist-free falls 75.7% honest)
    # falling is the expected, not the surprising, outcome here.
    assert any(r["fell"] for r in swing_rows)
    for r in swing_rows:
        assert r["tilt_peak_rad"] >= 0.0


def test_replay_with_ball_mode_runs_without_error(prototype_library):
    path, n_rows = prototype_library
    result = rl.run(path, mode="with-ball", leg_default_name="stand",
                    wbc_npz=None, max_rows=n_rows, seed=0)
    assert result["n_rows"] == n_rows


def test_unknown_leg_default_rejected(prototype_library):
    path, n_rows = prototype_library
    with pytest.raises(ValueError, match="unknown leg_default"):
        rl.run(path, mode="ball-parked", leg_default_name="bogus",
              wbc_npz=None, max_rows=n_rows, seed=0)


def test_mocap_leg_default_loads_the_committed_prior(prototype_library):
    path, n_rows = prototype_library
    result = rl.run(path, mode="ball-parked", leg_default_name="mocap",
                    wbc_npz=None, max_rows=n_rows, seed=0)
    assert result["leg_default"] == "mocap"
    assert result["n_rows"] == n_rows


def test_library_missing_keys_rejected(tmp_path):
    bad = str(tmp_path / "bad.npz")
    np.savez(bad, u=np.zeros((1, N_STEPS, 17), dtype=np.float32))
    with pytest.raises(ValueError, match="missing library keys"):
        rl.load_library(bad)


# ---------------------------------------------------------------------------
# Closed-loop WBC path: a zero-weight checkpoint must reproduce the a=0
# open-loop replay bit for bit (docs/design/whole_body_v1.md sec 3.2's
# parity anchor -- "the final actor layer is zero-initialised so iteration 0
# *is* the ... row").
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def zero_wbc_npz(tmp_path_factory) -> str:
    """All-zero 4-layer checkpoint in himrobot.wbc.export_npz's exact
    schema (W0..W3/b0..b3, rms_mean/var, leg_default, jnt_range) -- an
    all-zero final layer always outputs a=0 regardless of the observation,
    so this is the "iteration 0" parity anchor docs/design/whole_body_v1.md
    sec 3.2 describes."""
    path = str(tmp_path_factory.mktemp("wbc_ckpt") / "zero.npz")
    jnt_range = rl._joint_range(scene.build_model(fixed_base=True, paddle=True))
    np.savez(
        path,
        W0=np.zeros((512, rl.WBC_OBS_DIM)), b0=np.zeros(512),
        W1=np.zeros((256, 512)), b1=np.zeros(256),
        W2=np.zeros((128, 256)), b2=np.zeros(128),
        W3=np.zeros((rl.WBC_ACT_DIM, 128)), b3=np.zeros(rl.WBC_ACT_DIM),
        rms_mean=np.zeros(rl.WBC_OBS_DIM), rms_var=np.ones(rl.WBC_OBS_DIM),
        leg_default=scene.STAND_POSE[:12].copy(), jnt_range=jnt_range,
    )
    return path


def test_zero_weight_wbc_checkpoint_matches_open_loop(prototype_library, zero_wbc_npz):
    path, n_rows = prototype_library
    open_loop = rl.run(path, mode="ball-parked", leg_default_name="stand",
                       wbc_npz=None, max_rows=n_rows, seed=0)
    closed_loop = rl.run(path, mode="ball-parked", leg_default_name="stand",
                         wbc_npz=zero_wbc_npz, max_rows=n_rows, seed=0)
    assert closed_loop["falls_pct"] == open_loop["falls_pct"]
    for a, b in zip(open_loop["per_row"], closed_loop["per_row"]):
        assert a["fell"] == b["fell"]
        assert a["fell_step"] == b["fell_step"]
        if a["pelvis_xy_err_at_contact_m"] is None:
            assert b["pelvis_xy_err_at_contact_m"] is None
        else:
            assert a["pelvis_xy_err_at_contact_m"] == pytest.approx(b["pelvis_xy_err_at_contact_m"], abs=1e-12)


def test_wbc_checkpoint_shape_mismatch_rejected(tmp_path):
    bad = str(tmp_path / "bad_shape.npz")
    np.savez(
        bad,
        W0=np.zeros((512, rl.WBC_OBS_DIM - 1)), b0=np.zeros(512),  # wrong input width
        W1=np.zeros((256, 512)), b1=np.zeros(256),
        W2=np.zeros((128, 256)), b2=np.zeros(128),
        W3=np.zeros((rl.WBC_ACT_DIM, 128)), b3=np.zeros(rl.WBC_ACT_DIM),
    )
    with pytest.raises(AssertionError):
        rl.WBCCheckpoint(bad)


# ---------------------------------------------------------------------------
# (d) Throughput projection to 600 rows in <5 min. work/wbc/swing_lib_eval.npz
# (WB2's held-out library) does not exist yet in this tree, so this measures
# per-row wall time on a same-shape synthetic library and projects it, rather
# than literally reproducing the 600-row command -- see this file's own
# docstring and the task write-up for why.
# ---------------------------------------------------------------------------
def test_replay_throughput_projects_under_five_minutes_for_600_rows(tmp_path):
    n_probe = 20
    rows = [_idle_row() for _ in range(n_probe)]
    # Half the rows swing (a cheap synthetic sinusoid, not a real IK solve --
    # this measures the MuJoCo step cost the real library would also pay,
    # without the expense of n_probe real hitter rollouts).
    t = np.arange(N_STEPS, dtype=np.float32) / scene.CONTROL_HZ
    default = scene.STAND_POSE[12:].astype(np.float32)
    for i in range(0, n_probe, 2):
        wiggle = 0.3 * np.sin(2 * np.pi * 1.5 * t)[:, None] * np.ones((1, 17), dtype=np.float32)
        rows[i] = {**rows[i], "u": (default + wiggle).astype(np.float32)}

    path = str(tmp_path / "probe_lib.npz")
    _write_library(path, rows)

    start = time.time()
    rl.run(path, mode="with-ball", leg_default_name="stand", wbc_npz=None, max_rows=n_probe, seed=0)
    elapsed = time.time() - start

    projected_600 = elapsed / n_probe * 600
    assert projected_600 < 300.0, (
        f"{n_probe} rows took {elapsed:.2f}s ({elapsed / n_probe:.3f}s/row); "
        f"projected 600-row time {projected_600:.1f}s exceeds the 5-minute budget"
    )


# ---------------------------------------------------------------------------
# CLI smoke: main() writes the JSON to --out (never to work/, in this test).
# ---------------------------------------------------------------------------
def test_cli_writes_out_json(prototype_library, tmp_path):
    path, n_rows = prototype_library
    out = str(tmp_path / "replay_out.json")
    rl.main(["--lib", path, "--mode", "ball-parked", "--max-rows", str(n_rows), "--out", out])
    assert os.path.isfile(out)
    with open(out) as fh:
        data = json.load(fh)
    assert data["n_rows"] == n_rows
    assert "falls_pct" in data
