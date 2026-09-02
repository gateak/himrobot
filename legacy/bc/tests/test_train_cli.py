"""CLI regression tests for bc.train.main(): run the real arg parser and a
1-epoch loop against the synthetic data fallback (no --root needed) to catch
startup/shape crashes unit tests on individual functions would miss. Cheap: 1
epoch over the default synthetic sample (6000 steps) is well under a second on CPU.
"""

from __future__ import annotations

import sys

import torch

from .. import train as train_mod


def _run_main(monkeypatch, tmp_path, extra_args: list[str]) -> str:
    out = str(tmp_path / "ckpt.pt")
    argv = [
        "train.py",
        "--root", str(tmp_path / "no-such-data-dir"),  # falls back to the synthetic source
        "--epochs", "1",
        "--out", out,
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = train_mod.main()
    assert rc == 0
    return out


def test_transformer_default_context_trains_without_crashing(monkeypatch, tmp_path):
    # Regression: `--model transformer` with no --context used to squeeze the
    # windowed [B,1,obs_dim] input down to [B,obs_dim] and crash inside
    # TinyTransformerPolicy.forward (IndexError on a 2-D tensor).
    out = _run_main(monkeypatch, tmp_path, ["--model", "transformer"])
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["model_kind"] == "transformer"
    assert ckpt["context"] > 1  # sensible default, not the degenerate context=1


def test_transformer_explicit_context1_trains_without_crashing(monkeypatch, tmp_path):
    # The degenerate case must still work when asked for explicitly.
    out = _run_main(monkeypatch, tmp_path, ["--model", "transformer", "--context", "1"])
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["context"] == 1


def test_mlp_default_still_trains_without_crashing(monkeypatch, tmp_path):
    out = _run_main(monkeypatch, tmp_path, ["--model", "mlp"])
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["model_kind"] == "mlp"


def test_joints_mismatch_with_stale_joint_order_warns(monkeypatch, tmp_path, capsys):
    # Regression: --joints N (here 14) used to silently persist the contract's
    # 29-name joint_order into a 14-dim checkpoint with no diagnostic at all.
    _run_main(monkeypatch, tmp_path, ["--joints", "14"])
    printed = capsys.readouterr().out
    assert "[warn]" in printed and "joint_order" in printed and "14" in printed


def test_joints_matching_contract_does_not_warn(monkeypatch, tmp_path, capsys):
    _run_main(monkeypatch, tmp_path, ["--joints", "29"])
    printed = capsys.readouterr().out
    assert "joint_order" not in printed
