"""Stdio round-trip tests for him_adapter.server.

Runs the real entrypoint (``python -m him_adapter.server``) as a subprocess and
feeds it raw lines on stdin, exactly as HIM's runtime would. Forces the
HoldPolicy branch via HIM_POLICY_CHECKPOINT so these tests are deterministic
and torch-free regardless of whether a trained checkpoint happens to exist
under models/ locally.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "observation.json"


def _run_lines(lines: list[str], timeout: float = 30) -> tuple[list[dict], int]:
    """Feed raw newline-delimited lines to the server and return (responses, exit_code)."""
    env = dict(os.environ)
    env["HIM_POLICY_CHECKPOINT"] = str(REPO_ROOT / "__no_such_checkpoint__.pt")
    proc = subprocess.run(
        [sys.executable, "-m", "him_adapter.server"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=timeout,
    )
    responses = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    return responses, proc.returncode


def _req(**kwargs) -> str:
    return json.dumps(kwargs)


def test_lifecycle_round_trip() -> None:
    fixture = json.loads(FIXTURE.read_text())
    joints = fixture["signals"]["robot.joint_position"]
    lines = [
        _req(type="health"),
        _req(type="metadata"),
        _req(type="reset", episode={"episode_id": "e1"}),
        _req(type="infer", observation=fixture),
        _req(type="close"),
    ]
    responses, code = _run_lines(lines)
    assert code == 0
    assert len(responses) == 5

    assert responses[0] == {"ok": True}

    meta = responses[1]
    assert meta["interface_id"] == "himrobot-g1-v1"
    assert meta["policy_class"] == "HoldPolicy"
    assert meta["joint_count"] == 29
    assert any(o["name"] == "robot.joint_position" and o["shape"] == [29] for o in meta["observations"])
    assert any(a["name"] == "robot.desired_joint_position" and a["shape"] == [29] for a in meta["actions"])

    assert responses[2] == {"ok": True}

    infer = responses[3]
    assert "error" not in infer
    assert infer["signals"]["robot.desired_joint_position"] == joints
    assert infer["valid_for_ms"] > 0

    assert responses[4] == {"ok": True}


def test_process_survives_eof_without_close() -> None:
    responses, code = _run_lines([_req(type="health")])
    assert code == 0
    assert responses == [{"ok": True}]


def test_blank_lines_are_skipped_silently() -> None:
    lines = [_req(type="health"), "", "   ", _req(type="health"), _req(type="close")]
    responses, code = _run_lines(lines)
    assert code == 0
    assert responses == [{"ok": True}, {"ok": True}, {"ok": True}]


def test_garbage_line_returns_invalid_request_and_continues() -> None:
    lines = ["not json {{{", _req(type="health"), _req(type="close")]
    responses, code = _run_lines(lines)
    assert code == 0
    assert responses == [{"error": "invalid_request"}, {"ok": True}, {"ok": True}]


def test_non_object_json_returns_invalid_request() -> None:
    lines = [json.dumps([1, 2, 3]), json.dumps("just a string"), json.dumps(5), _req(type="close")]
    responses, code = _run_lines(lines)
    assert code == 0
    assert responses[:3] == [{"error": "invalid_request"}] * 3
    assert responses[3] == {"ok": True}


def test_missing_type_returns_invalid_request() -> None:
    lines = [json.dumps({"foo": "bar"}), _req(type="close")]
    responses, code = _run_lines(lines)
    assert code == 0
    assert responses == [{"error": "invalid_request"}, {"ok": True}]


def test_unsupported_type_returns_unsupported_request() -> None:
    lines = [_req(type="not_a_real_lifecycle"), _req(type="close")]
    responses, code = _run_lines(lines)
    assert code == 0
    assert responses[0] == {"error": "unsupported_request", "type": "not_a_real_lifecycle"}


def test_infer_missing_observation_key_falls_back_to_hold() -> None:
    lines = [_req(type="infer"), _req(type="close")]
    responses, code = _run_lines(lines)
    assert code == 0
    resp = responses[0]
    assert resp["error"] == "INVALID_OBSERVATION"
    assert resp["signals"]["robot.desired_joint_position"] == [0.0] * 29


def test_infer_wrong_length_returns_invalid_observation_and_zero_hold() -> None:
    obs = {"signals": {"robot.joint_position": [0.0] * 28}}
    lines = [_req(type="infer", observation=obs), _req(type="close")]
    responses, code = _run_lines(lines)
    assert code == 0
    resp = responses[0]
    assert resp["error"] == "INVALID_OBSERVATION"
    assert resp["signals"]["robot.desired_joint_position"] == [0.0] * 29
    assert resp["valid_for_ms"] > 0


def test_infer_nan_returns_invalid_observation() -> None:
    q = [float("nan")] + [0.0] * 28
    obs = {"signals": {"robot.joint_position": q}}
    lines = [_req(type="infer", observation=obs), _req(type="close")]
    responses, code = _run_lines(lines)
    assert code == 0
    resp = responses[0]
    assert resp["error"] == "INVALID_OBSERVATION"
    assert resp["signals"]["robot.desired_joint_position"] == [0.0] * 29


def test_infer_infinity_returns_invalid_observation() -> None:
    q = [0.0] * 28 + [float("inf")]
    obs = {"signals": {"robot.joint_position": q}}
    lines = [_req(type="infer", observation=obs), _req(type="close")]
    responses, code = _run_lines(lines)
    resp = responses[0]
    assert resp["error"] == "INVALID_OBSERVATION"


def test_infer_string_value_returns_invalid_observation() -> None:
    obs = {"signals": {"robot.joint_position": "abcdefghijklmnopqrstuvwxyz123"}}
    lines = [_req(type="infer", observation=obs), _req(type="close")]
    responses, code = _run_lines(lines)
    resp = responses[0]
    assert resp["error"] == "INVALID_OBSERVATION"
    assert resp["signals"]["robot.desired_joint_position"] == [0.0] * 29


def test_infer_falls_back_to_last_valid_hold() -> None:
    fixture = json.loads(FIXTURE.read_text())
    good_joints = fixture["signals"]["robot.joint_position"]
    bad = {"signals": {"robot.joint_position": [0.0] * 28}}
    lines = [
        _req(type="infer", observation=fixture),
        _req(type="infer", observation=bad),
        _req(type="close"),
    ]
    responses, code = _run_lines(lines)
    assert "error" not in responses[0]
    assert responses[1]["error"] == "INVALID_OBSERVATION"
    assert responses[1]["signals"]["robot.desired_joint_position"] == good_joints


def test_process_stays_alive_through_a_run_of_bad_lines() -> None:
    lines = [
        "",
        "not json",
        json.dumps([1, 2, 3]),
        json.dumps({"no_type": True}),
        _req(type="infer", observation={"signals": {"robot.joint_position": [1, 2]}}),
        _req(type="health"),
        _req(type="close"),
    ]
    responses, code = _run_lines(lines)
    assert code == 0
    assert responses[-2] == {"ok": True}
    assert responses[-1] == {"ok": True}
