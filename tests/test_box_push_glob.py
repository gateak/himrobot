"""tools/box/box_push_results.py's --glob option (WB6): extra glob patterns,
relative to --root, added on top of the fixed <run>/{train_log.csv,best.pt,
final.pt,RESULT.md} + <root>/*.md|*.json|*.log|logs/*.log|scoreboard*/
scoreboard.* set -- needed for an rsl_rl/Isaac Lab run tree
(tools/box/wbc_box_run.sh) that does not follow the legacy
<root>/<run>/train_log.csv convention at all.

Run with: .venv/bin/python -m pytest tests/test_box_push_glob.py -v
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "box" / "box_push_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("box_push_results", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bpr():
    return _load_module()


def _write(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture()
def fake_root(tmp_path: Path) -> Path:
    """<root> = tmp_path/logs/rsl_rl, matching wbc_box_run.sh's own
    `--root "$(dirname "$OUT_DIR")"` (OUT_DIR=logs/rsl_rl/wbc_v1)."""
    root = tmp_path / "logs" / "rsl_rl"

    # The legacy <root>/<run>/train_log.csv tree -- must keep working exactly
    # as before, with or without --glob.
    _write(root / "legacy_run" / "train_log.csv", b"step,reward\n1,2\n")
    _write(root / "legacy_run" / "best.pt", b"legacy-best")

    # The WBC (rsl_rl/Isaac Lab) tree -- no train_log.csv anywhere, so
    # collect()'s run-discovery never sees it without --glob.
    _write(root / "wbc_v1" / "run" / "model_500.pt", b"ckpt-bytes")
    _write(root / "wbc_v1" / "run" / "exported" / "x.npz", b"npz-bytes")
    _write(root / "wbc_v1" / "run" / "params" / "env.yaml", b"env: {}\n")
    _write(root / "wbc_v1" / "smoke_timing.json", b'{"s_per_iteration": 0.1}')

    return root


WBC_GLOBS = (
    "wbc_v1/*/model_*.pt",
    "wbc_v1/*/exported/*.npz",
    "wbc_v1/*/params/*.yaml",
    "wbc_v1/smoke_timing.json",
)


def test_collect_without_glob_behaves_exactly_as_before(bpr, fake_root: Path):
    runs, files = bpr.collect(str(fake_root))
    rel = {Path(f).relative_to(fake_root).as_posix() for f in files}

    assert [Path(r).name for r in runs] == ["legacy_run"]
    assert "legacy_run/train_log.csv" in rel
    assert "legacy_run/best.pt" in rel
    # None of the WBC-tree files are picked up -- no train_log.csv there,
    # and none of the fixed *.md/*.json/*.log/logs/*.log/scoreboard*
    # patterns match model_*.pt / exported/*.npz / params/*.yaml.
    assert not any(f.startswith("wbc_v1/") and f != "wbc_v1/smoke_timing.json" for f in rel)
    # smoke_timing.json DOES match the fixed "*.json" pattern only if it sat
    # at <root> directly -- here it's under wbc_v1/, one level too deep, so
    # it is NOT included without --glob either.
    assert "wbc_v1/smoke_timing.json" not in rel


def test_collect_with_glob_includes_the_new_files_and_keeps_the_old_ones(bpr, fake_root: Path):
    runs, files = bpr.collect(str(fake_root), WBC_GLOBS)
    rel = {Path(f).relative_to(fake_root).as_posix() for f in files}

    assert "legacy_run/train_log.csv" in rel
    assert "legacy_run/best.pt" in rel
    assert "wbc_v1/run/model_500.pt" in rel
    assert "wbc_v1/run/exported/x.npz" in rel
    assert "wbc_v1/run/params/env.yaml" in rel
    assert "wbc_v1/smoke_timing.json" in rel
    # run-discovery (train_log.csv-based) is unaffected by --glob: the WBC
    # tree still isn't counted as a "run" for the summary.json purposes.
    assert [Path(r).name for r in runs] == ["legacy_run"]


def test_collect_glob_ignores_empty_files(bpr, fake_root: Path):
    _write(fake_root / "wbc_v1" / "run" / "model_0.pt", b"")  # 0 bytes
    _, files = bpr.collect(str(fake_root), ("wbc_v1/*/model_*.pt",))
    rel = {Path(f).name for f in files}
    assert "model_500.pt" in rel
    assert "model_0.pt" not in rel


def test_main_dry_run_with_glob_writes_manifest_and_never_invokes_him(bpr, fake_root: Path, tmp_path: Path,
                                                                       monkeypatch):
    out_dir = tmp_path / "push_out"
    argv = [
        "box_push_results.py",
        "--root", str(fake_root),
        "--machine", "ssh_test",
        "--out", str(out_dir),
        "--dry-run",
    ]
    for pattern in WBC_GLOBS:
        argv += ["--glob", pattern]
    monkeypatch.setattr(sys, "argv", argv)

    def _boom(*args, **kwargs):
        raise AssertionError("subprocess.run (the `him` push) must never be called under --dry-run")

    monkeypatch.setattr(subprocess, "run", _boom)

    bpr.main()  # stops before any push -- see --dry-run in main()'s own control flow

    manifest = json.loads((out_dir / "manifest.json").read_text())
    uris = {a["uri"] for a in manifest["artifacts"]}
    assert any(uri.endswith("model_500.pt") for uri in uris)
    assert any(uri.endswith("x.npz") for uri in uris)
    assert any(uri.endswith("env.yaml") for uri in uris)
    assert any(uri.endswith("wbc_v1/smoke_timing.json") for uri in uris)
    assert any(uri.endswith("train_log.csv") for uri in uris)
    assert manifest["dataset_slug"] == "himrobot-rl-checkpoints"

    summary = json.loads((out_dir / "summary.json").read_text())
    assert list(summary["runs"].keys()) == ["legacy_run"]
