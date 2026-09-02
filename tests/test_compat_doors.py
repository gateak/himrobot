"""The three doors HIM and a published challenge entry load our code by,
unchanged by the himrobot restructure:

    python -m him_adapter.server
    python -m tt_sim.evaluate_stance ...
    tt_sim.policies.param_residual_np:NumpyParamResidualPolicy (a --policy path string)

Run as real subprocesses / a real torch-blocked import, not just "the module
imports" -- see REPORT.md for the full manual verification (bit-exact output
at --trials 10) this pins a fast (--trials 2) version of.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from himrobot.paths import REPO_ROOT

PARAM_RESIDUAL_NP_POLICY = "tt_sim.policies.param_residual_np:NumpyParamResidualPolicy"


def _run(module: str, *args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip())


def test_tt_sim_evaluate_stance_door_runs_the_shipped_policy():
    """python -m tt_sim.evaluate_stance --policy tt_sim.policies.param_residual_np:... """
    metrics = _run("tt_sim.evaluate_stance", "--policy", PARAM_RESIDUAL_NP_POLICY,
                    "--trials", "2", "--no-trajectory")
    assert metrics["n_trials"] == 2
    assert metrics["contacts"] in (0, 1, 2)
    assert isinstance(metrics["legal_return_rate"], float)


def test_himrobot_door_matches_tt_sim_door_bit_for_bit():
    """Same command through the new himrobot.eval.evaluate_stance path: identical output."""
    via_tt_sim = _run("tt_sim.evaluate_stance", "--policy", PARAM_RESIDUAL_NP_POLICY,
                       "--seed", "3", "--trials", "2", "--no-trajectory")
    via_himrobot = _run("himrobot.eval.evaluate_stance", "--policy", PARAM_RESIDUAL_NP_POLICY,
                        "--seed", "3", "--trials", "2", "--no-trajectory")
    assert via_tt_sim == via_himrobot


def test_tt_sim_evaluate_door_runs():
    """python -m tt_sim.evaluate (the plain, un-shifted-stance door) still works."""
    metrics = _run("tt_sim.evaluate", "--policy", "hold", "--trials", "2", "--no-trajectory")
    assert metrics["n_trials"] == 2
    assert metrics["legal_return_rate"] == 0.0  # HoldPolicy never swings
    assert metrics["contacts"] == 0


def test_him_adapter_server_door_runs():
    """python -m him_adapter.server: a minimal health/close stdio round trip."""
    env = {**os.environ, "HIM_POLICY_CHECKPOINT": str(REPO_ROOT / "__no_such_checkpoint__.pt")}
    proc = subprocess.run(
        [sys.executable, "-m", "him_adapter.server"],
        input=json.dumps({"type": "health"}) + "\n" + json.dumps({"type": "close"}) + "\n",
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=30,
    )
    assert proc.returncode == 0
    responses = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    assert responses == [{"ok": True}, {"ok": True}]


def test_policy_path_string_is_torch_free():
    """The published challenge entry's --policy path string, and the door that
    sets its stance, must both import with torch entirely absent (HIM's mujoco
    worker has no torch): sys.modules['torch'] = None makes `import torch`
    raise, the same failure mode as torch not being installed at all."""
    source = (
        "import sys; sys.modules['torch'] = None\n"
        "import tt_sim.policies.param_residual_np\n"
        "import tt_sim.evaluate_stance\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True,
                          cwd=str(REPO_ROOT), timeout=60)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert proc.stdout.strip() == "ok"
