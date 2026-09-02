"""himrobot.rl.param_train: deterministic checkpoint selection and warm-start knobs.

The load-bearing test is the first one: the trainer's deterministic_eval must
score a checkpoint exactly the way himrobot.eval.evaluate does (same serves, same
trial semantics), otherwise best.pt would be selected on a number that is not
the scoreboard's.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from himrobot.eval import evaluate  # noqa: E402
from himrobot.rl import param_residual as pr  # noqa: E402
from himrobot.rl import param_train  # noqa: E402


def _fresh_v2_checkpoint(path: str) -> str:
    """A v2 checkpoint with a non-trivial head, so the eval actually decodes
    something other than the nominal hitter."""
    torch.manual_seed(3)
    model = pr.ParamActorCritic(pr.FEATURE_DIM, pr.K_PARAMS_V2)
    with torch.no_grad():
        for p in model.actor.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    pr.save_checkpoint(path, model, pr.RunningMeanStd((pr.FEATURE_DIM,)), 0,
                       param_specs=pr.PARAM_SPECS_V2)
    return path


def test_deterministic_eval_matches_the_evaluator(tmp_path, monkeypatch):
    """Same serves (seed + k), same 150-step trials, same legal/target/error
    accounting as himrobot.eval.evaluate -- trial for trial."""
    ckpt = _fresh_v2_checkpoint(str(tmp_path / "v2.pt"))
    device = torch.device("cpu")
    model, rms, specs = pr.load_checkpoint(ckpt, device)
    n, seed = 6, 4242
    det = param_train.deterministic_eval(model, rms, specs, device, n, seed)

    monkeypatch.setenv("TT_RL_CKPT", ckpt)
    out = str(tmp_path / "eval")
    metrics = evaluate.main(["--policy", "himrobot.rl.policy:ParamResidualPolicy",
                             "--trials", str(n), "--seed", str(seed), "--out", out])
    doc = json.load(open(os.path.join(out, "evaluation", "evaluator.json")))

    assert det["legal_rate"] == pytest.approx(metrics["legal_return_rate"])
    assert det["target_rate"] == pytest.approx(metrics["target_hit_rate"])
    assert det["contact_rate"] == pytest.approx(100.0 * metrics["contacts"] / n)
    if metrics["landing_error_m"] is None:
        assert np.isnan(det["landing_error_m"])
    else:
        assert det["landing_error_m"] == pytest.approx(metrics["landing_error_m"], abs=1e-9)
    assert any(t["legal_return"] for t in doc["per_trial"]) or True  # informative only


def test_det_eval_selects_best_and_logs_it(tmp_path):
    out = str(tmp_path / "det")
    result = param_train.train(n_envs=2, episodes=6, seed=0, out=out, eval_every=2,
                               batch_episodes=2, minutes=1.0, param_space="v2",
                               det_eval_episodes=2, det_eval_every=2, det_eval_seed=5)
    assert result["best_det"] is not None and "legal_rate" in result["best_det"]
    assert os.path.isfile(os.path.join(out, "best.pt"))
    best = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=False)
    assert best["det_eval"]["n_serves"] == 2 and best["det_eval"]["seed"] == 5
    assert best["episodes"] == result["best_det_episode"]

    rows = [r.split(",") for r in open(os.path.join(out, "train_log.csv")).read().strip().splitlines()]
    header = rows[0]
    assert header[5:9] == ["det_legal_rate", "det_contact_rate", "det_target_rate", "det_landing_error_m"]
    assert header[9:9 + pr.K_PARAMS_V2] == [f"u_{name}" for name in pr.PARAM_NAMES_V2]
    assert header[9 + pr.K_PARAMS_V2:] == ["blend_block", "blend_lift", "blend_punch", "blend_crossbody"]
    blends = [float(x) for x in rows[-1][9 + pr.K_PARAMS_V2:]]
    assert all(0.0 <= b <= 0.8 for b in blends)
    assert rows[1][0] == "0" and rows[1][5] != ""  # row 0 = the starting point's score
    det_rows = [r for r in rows[2:] if r[5] != ""]
    assert det_rows, "no deterministic evaluation was logged during training"
    assert all(len(r) == len(header) for r in rows)


def test_log_std_init_new_only_touches_the_new_coordinates(tmp_path):
    src_path = str(tmp_path / "v1.pt")
    torch.manual_seed(0)
    src = pr.ParamActorCritic(pr.FEATURE_DIM, pr.K_PARAMS, log_std_init=-1.3)
    pr.save_checkpoint(src_path, src, pr.RunningMeanStd((pr.FEATURE_DIM,)), 0,
                       param_specs=pr.PARAM_SPECS)
    out = str(tmp_path / "warm")
    param_train.train(n_envs=1, episodes=1, seed=0, out=out, eval_every=1, batch_episodes=1000,
                      minutes=1.0, param_space="v2", resume_from=src_path,
                      log_std_init=-1.0, log_std_init_new=-2.5)
    final = torch.load(os.path.join(out, "final.pt"), map_location="cpu", weights_only=False)
    log_std = final["model_state"]["log_std"]
    # batch_episodes=1000 > episodes, so no update ran: the init is what was saved.
    assert torch.allclose(log_std[:pr.K_PARAMS], torch.full((pr.K_PARAMS,), -1.3))
    assert torch.allclose(log_std[pr.K_PARAMS:], torch.full((pr.K_PARAMS,), -2.5))


def test_train_time_perturbation_defaults_off_and_reaches_the_hitter():
    """VecTTParam feeds the hitter the env's own obs dict unless noise/delay is
    requested; with it, the hitter and the serve features see a perturbed view."""
    zero = np.zeros(pr.K_PARAMS_V2)
    decide = lambda feats: {"u": zero}  # noqa: E731
    clean = pr.VecTTParam(n_envs=1, seed=3, decide_fn=decide, param_specs=pr.PARAM_SPECS_V2)
    ref = pr.VecTTParam(n_envs=1, seed=3, decide_fn=decide, param_specs=pr.PARAM_SPECS_V2)
    noisy = pr.VecTTParam(n_envs=1, seed=3, decide_fn=decide, param_specs=pr.PARAM_SPECS_V2,
                          obs_noise=0.02, obs_delay_steps=3)
    assert np.array_equal(clean._features, ref._features)
    assert not np.array_equal(clean._features[0, :6], noisy._features[0, :6])  # ball pos/vel features
    acts_clean, acts_ref, acts_noisy = [], [], []
    for _ in range(40):
        acts_clean.append(clean._hitters[0].act(clean.envs[0]._get_obs()))
        clean.step()
        acts_ref.append(ref._hitters[0].act(ref.envs[0]._get_obs()))
        ref.step()
        noisy.step()
        acts_noisy.append(noisy._hitters[0]._plan["p_hit"].copy() if noisy._hitters[0]._plan else None)
    assert np.array_equal(np.stack(acts_clean), np.stack(acts_ref))
    assert clean._hitters[0]._plan is not None and noisy._hitters[0]._plan is not None
    assert not np.array_equal(clean._hitters[0]._plan["p_hit"], noisy._hitters[0]._plan["p_hit"])


def test_blend_fractions_match_the_hitter_blend():
    from himrobot.policies.ik_hitter import GATE_GAIN

    u = np.zeros(pr.K_PARAMS_V2)
    u[pr.PARAM_NAMES_V2.index("gate_lift")] = 1.0
    u[pr.PARAM_NAMES_V2.index("gate_block")] = -0.5  # rectified away
    frac = param_train.blend_fractions(u, pr.PARAM_NAMES_V2)
    assert frac.shape == (4,)
    assert frac[1] == pytest.approx(GATE_GAIN / (1.0 + GATE_GAIN)) and frac[0] == 0.0
    assert param_train.blend_fractions(np.zeros(pr.K_PARAMS), pr.PARAM_NAMES).shape == (0,)


def test_det_select_target_ranks_placement_first():
    a = {"legal_rate": 80.0, "target_rate": 10.0, "landing_error_m": 0.5}
    b = {"legal_rate": 78.0, "target_rate": 20.0, "landing_error_m": 0.4}
    assert param_train._det_key(a) > param_train._det_key(b)
    assert param_train._det_key(b, "target") > param_train._det_key(a, "target")
