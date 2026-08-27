"""tt_sim.evaluate CLI tests (headless, no --video: exercised separately by
the task's manual video-confirmation run — see tt_sim/README.md). Run with:
    .venv/bin/python -m pytest tt_sim/tests/test_evaluate.py -v
"""
from __future__ import annotations

import json
import os

from tt_sim.evaluate import main as evaluate_main


def test_hold_policy_two_trials_writes_valid_outputs(tmp_path, monkeypatch):
    # Clear any HIM_* env vars so --out fully controls the output layout.
    for var in ("HIM_CANDIDATE_OUTPUT", "HIM_EVALUATION_OUTPUT", "HIM_MEDIA_OUTPUT"):
        monkeypatch.delenv(var, raising=False)

    out_dir = str(tmp_path / "eval")
    metrics = evaluate_main(["--policy", "hold", "--trials", "2", "--seed", "0", "--out", out_dir])

    trajectory_path = os.path.join(out_dir, "candidate", "trajectory.jsonl")
    evaluator_path = os.path.join(out_dir, "evaluation", "evaluator.json")
    assert os.path.isfile(trajectory_path)
    assert os.path.isfile(evaluator_path)

    # trajectory.jsonl: one JSON object per line, both trials represented.
    with open(trajectory_path) as fh:
        lines = [json.loads(line) for line in fh]
    assert len(lines) > 0
    assert {rec["trial"] for rec in lines} == {0, 1}
    first = lines[0]
    assert set(["trial", "step", "time_s", "obs", "action", "events"]) <= first.keys()
    assert len(first["obs"]["robot.joint_position"]) == 29
    assert len(first["action"]) == 29

    # evaluator.json schema: authoritative true, this-proxy's evaluator id,
    # all 6 required metric keys, and a per_trial entry per trial.
    with open(evaluator_path) as fh:
        doc = json.load(fh)
    assert doc["authoritative"] is True
    assert doc["evaluator"] == "himrobot-local-proxy@0.1.0"
    assert "proxy" in doc["note"].lower()
    for key in ("legal_return_rate", "target_hit_rate", "landing_error_m", "reaction_latency_ms", "n_trials", "contacts"):
        assert key in doc["metrics"], key
    assert doc["metrics"]["n_trials"] == 2
    assert len(doc["per_trial"]) == 2

    # HoldPolicy never swings -> never a legal return, and never a target hit.
    assert doc["metrics"]["legal_return_rate"] == 0.0
    assert doc["metrics"]["target_hit_rate"] == 0.0
    assert doc["metrics"]["contacts"] == 0
    assert doc["metrics"]["landing_error_m"] is None  # no legal returns to average
    for trial in doc["per_trial"]:
        assert trial["legal_return"] is False
        assert trial["target_hit"] is False

    # main()'s return value is the same metrics dict written into evaluator.json.
    assert metrics == doc["metrics"]


def test_unknown_policy_shortcut_raises_clear_error():
    import pytest

    with pytest.raises(ValueError, match="--policy"):
        evaluate_main(["--policy", "not_a_real_policy", "--trials", "1"])
