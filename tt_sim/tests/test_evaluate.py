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


def test_no_trajectory_skips_only_the_step_log(tmp_path, monkeypatch):
    for var in ("HIM_CANDIDATE_OUTPUT", "HIM_EVALUATION_OUTPUT", "HIM_MEDIA_OUTPUT"):
        monkeypatch.delenv(var, raising=False)
    with_traj = str(tmp_path / "with")
    without = str(tmp_path / "without")
    m1 = evaluate_main(["--policy", "ik_hitter", "--trials", "2", "--seed", "5", "--out", with_traj])
    m2 = evaluate_main(["--policy", "ik_hitter", "--trials", "2", "--seed", "5", "--out", without,
                        "--no-trajectory"])
    assert os.path.isfile(os.path.join(with_traj, "candidate", "trajectory.jsonl"))
    assert not os.path.exists(os.path.join(without, "candidate", "trajectory.jsonl"))
    assert os.path.isfile(os.path.join(without, "evaluation", "evaluator.json"))
    assert m1 == m2  # scoring is untouched by the flag


def test_policy_obs_perturbation_defaults_off_and_changes_only_what_the_policy_sees(tmp_path, monkeypatch):
    for var in ("HIM_CANDIDATE_OUTPUT", "HIM_EVALUATION_OUTPUT", "HIM_MEDIA_OUTPUT"):
        monkeypatch.delenv(var, raising=False)
    base = str(tmp_path / "base")
    clean = str(tmp_path / "clean")
    noisy = str(tmp_path / "noisy")
    args = ["--policy", "ik_hitter", "--trials", "3", "--seed", "9", "--no-trajectory"]
    m_base = evaluate_main(args + ["--out", base])
    m_clean = evaluate_main(args + ["--out", clean, "--obs-noise", "0", "--joint-noise", "0", "--obs-delay-steps", "0"])
    m_noisy = evaluate_main(args + ["--out", noisy, "--obs-noise", "0.05", "--obs-delay-steps", "5"])

    d_base = json.load(open(os.path.join(base, "evaluation", "evaluator.json")))
    d_clean = json.load(open(os.path.join(clean, "evaluation", "evaluator.json")))
    d_noisy = json.load(open(os.path.join(noisy, "evaluation", "evaluator.json")))
    assert m_base == m_clean and d_base["per_trial"] == d_clean["per_trial"]  # explicit zeros == default
    assert d_base["policy_obs"] == {"obs_noise": 0.0, "joint_noise": 0.0, "delay_steps": 0}
    assert d_noisy["policy_obs"] == {"obs_noise": 0.05, "joint_noise": 0.0, "delay_steps": 5}
    # The serves are the same fixed serves; only the policy's view differs.
    assert [t["serve"] for t in d_noisy["per_trial"]] == [t["serve"] for t in d_base["per_trial"]]
    assert d_noisy["per_trial"] != d_base["per_trial"]


def test_obs_perturber_delay_and_noise_semantics():
    import numpy as np
    from tt_sim.evaluate import ObsPerturber

    def obs(t):
        return {"time_s": t, "ball.position": np.array([t, 0.0, 0.0]), "ball.velocity": np.zeros(3),
                "robot.joint_position": np.zeros(29), "robot.joint_velocity": np.zeros(29)}

    off = ObsPerturber()
    o0 = obs(0.0)
    assert off.reset(o0) is o0 and off.step(o0) is o0  # no copy, no draw

    delay = ObsPerturber(delay_steps=2)
    assert delay.reset(obs(0.0))["time_s"] == 0.0
    assert delay.step(obs(1.0))["time_s"] == 0.0
    assert delay.step(obs(2.0))["time_s"] == 0.0
    assert delay.step(obs(3.0))["time_s"] == 1.0  # exactly two steps old from here on

    noise = ObsPerturber(obs_noise=0.01, seed=1)
    seen = noise.reset(obs(0.0))
    assert seen["time_s"] == 0.0
    assert not np.array_equal(seen["ball.position"], o0["ball.position"])
    assert np.array_equal(seen["robot.joint_position"], o0["robot.joint_position"])  # joint_noise off
    assert np.array_equal(o0["ball.position"], np.array([0.0, 0.0, 0.0]))  # the true obs is untouched
