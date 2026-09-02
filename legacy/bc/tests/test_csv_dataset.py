"""Tests for the retargeted-G1-CSV data source (bc.dataset._load_csv).

Uses a tiny synthetic CSV in the emitter's layout (root xyz, quat xyzw, J joints;
no header) written to tmp_path. Run: ./.venv/bin/python -m pytest -q legacy/bc/tests
"""

from __future__ import annotations

import numpy as np
import pytest

from ..config import Config, load_contract_joint_order
from ..dataset import build_arrays, check_csv_joint_order, make_windows, split_indices

J = 29
ROOT_COLS = 7


def _write_clip(path, n_frames: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    root_pos = rng.normal(size=(n_frames, 3))
    quat = rng.normal(size=(n_frames, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    joints = np.cumsum(0.01 * rng.normal(size=(n_frames, J)), axis=0)  # smooth-ish walk
    arr = np.hstack([root_pos, quat, joints])
    np.savetxt(path, arr, delimiter=",")
    return arr


def _cfg(root) -> Config:
    cfg = Config()
    cfg.data.root = str(root)
    return cfg


def test_contract_joint_order_has_29_names():
    order = load_contract_joint_order()
    assert len(order) == 29
    assert order[0] == "left_hip_pitch_joint" and order[-1] == "right_wrist_yaw_joint"


def test_csv_source_next_pose_pairs(tmp_path):
    a = _write_clip(tmp_path / "Clip_A.csv", 6, seed=1)
    b = _write_clip(tmp_path / "Clip_B.csv", 4, seed=2)
    obs, act, episodes, source = build_arrays(_cfg(tmp_path))

    assert source.startswith("g1-motion-csv:")
    assert "joint order file absent" in source
    # one episode per file, each of length n_frames - 1
    assert episodes == [(0, 5), (5, 8)]
    assert obs.shape == (8, J) and act.shape == (8, J)
    assert obs.dtype == np.float32 and act.dtype == np.float32
    # obs = joints at t, act = joints at t+1, root columns excluded
    np.testing.assert_allclose(obs[:5], a[:-1, ROOT_COLS:], rtol=1e-6)
    np.testing.assert_allclose(act[:5], a[1:, ROOT_COLS:], rtol=1e-6)
    np.testing.assert_allclose(obs[5:], b[:-1, ROOT_COLS:], rtol=1e-6)
    np.testing.assert_allclose(act[5:], b[1:, ROOT_COLS:], rtol=1e-6)
    # no pair crosses a file boundary
    assert not np.allclose(act[4], obs[5])


def test_csv_root_in_obs_appends_height_and_quat(tmp_path):
    a = _write_clip(tmp_path / "Clip_A.csv", 5, seed=3)
    cfg = _cfg(tmp_path)
    cfg.data.csv_root_in_obs = True
    obs, act, _, source = build_arrays(cfg)
    assert obs.shape == (4, J + 5) and act.shape == (4, J)
    assert cfg.obs_dim == J + 5
    np.testing.assert_allclose(obs[:, :J], a[:-1, ROOT_COLS:], rtol=1e-6)
    np.testing.assert_allclose(obs[:, J:], a[:-1, [2, 3, 4, 5, 6]], rtol=1e-6)
    assert "root(z,quat)" in source


def test_csv_wrong_column_count_raises(tmp_path):
    np.savetxt(tmp_path / "bad.csv", np.zeros((3, ROOT_COLS + J - 1)), delimiter=",")
    with pytest.raises(RuntimeError, match="expected 36 columns"):
        build_arrays(_cfg(tmp_path))


def test_csv_non_finite_raises(tmp_path):
    arr = np.zeros((3, ROOT_COLS + J))
    arr[1, 10] = np.nan
    np.savetxt(tmp_path / "nan.csv", arr, delimiter=",")
    with pytest.raises(RuntimeError, match="non-finite"):
        build_arrays(_cfg(tmp_path))


def test_single_frame_clip_is_skipped(tmp_path):
    _write_clip(tmp_path / "one.csv", 1, seed=4)
    _write_clip(tmp_path / "two.csv", 3, seed=5)
    _, _, episodes, _ = build_arrays(_cfg(tmp_path))
    assert episodes == [(0, 2)]


def test_joint_order_file_checked_against_contract(tmp_path):
    _write_clip(tmp_path / "Clip.csv", 3, seed=6)
    cfg = _cfg(tmp_path)
    good = tmp_path / cfg.data.csv_joint_order_file
    good.write_text("\n".join(cfg.data.joint_order) + "\n")
    assert "verified against contract (29 names)" in check_csv_joint_order(str(tmp_path), cfg)
    _, _, _, source = build_arrays(cfg)
    assert "verified against contract" in source

    swapped = list(cfg.data.joint_order)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    good.write_text("\n".join(swapped) + "\n")
    with pytest.raises(RuntimeError, match="joint order differs"):
        build_arrays(cfg)


def test_no_csv_falls_back_to_synthetic(tmp_path):
    cfg = _cfg(tmp_path / "empty")
    obs, act, _, source = build_arrays(cfg)
    assert source == "synthetic"
    assert obs.shape[1] == J and act.shape[1] == J


def test_windows_and_episode_split_respect_boundaries(tmp_path):
    _write_clip(tmp_path / "A.csv", 7, seed=7)
    _write_clip(tmp_path / "B.csv", 5, seed=8)
    _write_clip(tmp_path / "C.csv", 4, seed=9)
    obs, act, episodes, _ = build_arrays(_cfg(tmp_path))
    X, Y = make_windows(obs, act, episodes, context=3)
    assert X.shape == (len(obs), 3, J) and Y.shape == (len(obs), J)
    # first window of episode 2 is left-padded with its own first frame, not episode 1's last
    s = episodes[1][0]
    np.testing.assert_array_equal(X[s][0], obs[s])
    np.testing.assert_array_equal(X[s][1], obs[s])

    train_idx, val_idx = split_indices(len(X), episodes, 0.34, seed=0, mode="episode")
    assert len(train_idx) + len(val_idx) == len(X)
    assert not set(train_idx) & set(val_idx)
    # every episode is entirely on one side
    for s, e in episodes:
        ids = set(range(s, e))
        assert ids <= set(train_idx) or ids <= set(val_idx)
    assert len(val_idx) > 0 and len(train_idx) > 0
