"""Tests for tools/mocap/mocap_leg_prior.py (work item WB5): the segmenter
(find_events, via detect_swings) on a synthetic clip, and the committed
assets/wbc/mocap_leg_prior.npz against the numbers measured 2026-09-02.

Run with: .venv/bin/python -m pytest tests/test_mocap_leg_prior.py -v
"""
from __future__ import annotations

import numpy as np

from himrobot.paths import ASSETS_DIR, REPO_ROOT
from himrobot.sim import scene
from tools.mocap.mocap_leg_prior import DT, GAP_TOL_FRAMES, SWING_THRESH_MPS, detect_swings, find_events
from tools.mocap.retarget_report import load_robot

NPZ_PATH = ASSETS_DIR / "wbc" / "mocap_leg_prior.npz"
SCENE_PATH = str(ASSETS_DIR / "unitree_g1" / "scene.xml")
ORDER_FILE = str(REPO_ROOT / "work" / "g1" / "gmr_g1_joint_order.txt")

# Measured 2026-09-02 directly from work/g1/Clip_2..8.csv (see the reader
# digest / this test's own docstring history); the npz must reproduce these
# to 0.01 rad regardless of how the swing library it's built from changes.
GLOBAL_MEAN_MEASURED = np.array([
    -0.20, 0.13, 0.22, 0.39, -0.13, -0.05, -0.06, -0.22, -0.37, 0.34, -0.18, 0.04,
])
# Swing events per play clip before any quality mask: the 2026-09-02 reader
# digest (scratchpad mocap_swing_stats.py, right-wrist speed > 2 m/s merged
# across gaps <= 3 frames), 238 in total, quoted in the design doc §4.4.
N_EVENTS_DIGEST = [30, 33, 39, 45, 40, 38, 13]


def _load_npz():
    z = np.load(NPZ_PATH, allow_pickle=False)
    return {k: z[k] for k in z.files}


def test_npz_shapes_and_joint_names():
    z = _load_npz()
    assert z["bins_mean"].shape == (10, 12)
    assert z["bins_std"].shape == (10, 12)
    assert z["global_mean"].shape == (12,)
    assert z["global_std"].shape == (12,)
    assert z["upper_bins_mean"].shape == (10, 17)
    assert z["upper_bins_std"].shape == (10, 17)
    assert z["root_xy_speed_bins"].shape == (10,)
    assert list(z["joint_names"]) == list(scene.JOINT_ORDER[:12])
    assert z["n_events_per_clip"].shape == z["n_events_kept_per_clip"].shape == z["clip_weights"].shape == (7,)
    assert str(z["sat_range_source"]) in ("scene+ik_model", "scene")
    assert "_per_clip_kept_windows" not in z  # build-time scratch, never written


def test_npz_event_counts_match_the_reader_digest_and_bins_are_populated():
    """The segmenter's per-clip counts are mask-independent and pinned to the
    digest; the mask must leave enough swings for the phase bins to mean
    something (an all-rejected build would still pass the shape/finite/floor
    tests, since an empty bin yields mean 0 and the 0.05 std floor)."""
    z = _load_npz()
    assert z["n_events_per_clip"].tolist() == N_EVENTS_DIGEST
    assert list(z["clip_names"]) == [f"Clip_{k}.csv" for k in range(2, 9)]
    kept = z["n_events_kept_per_clip"]
    assert (kept <= z["n_events_per_clip"]).all()
    assert kept.sum() >= 30, kept.tolist()
    # the phase bins are a swing-conditioned view of the same legs, not a
    # different robot: every bin sits within 0.3 rad of the all-frames mean
    assert np.abs(z["bins_mean"] - z["global_mean"][None, :]).max() < 0.3
    assert (z["bins_std"] > 0.05 + 1e-9).any()  # not every bin is sitting on the floor
    assert (z["root_xy_speed_bins"] > 0).all() and (z["root_xy_speed_bins"] < 2.0).all()


def test_npz_values_are_finite_and_std_is_floored():
    z = _load_npz()
    for key in ("bins_mean", "bins_std", "global_mean", "global_std",
                "upper_bins_mean", "upper_bins_std", "root_xy_speed_bins"):
        assert np.isfinite(z[key]).all(), key
    assert (z["bins_std"] >= 0.05 - 1e-12).all()
    assert (z["upper_bins_std"] >= 0.05 - 1e-12).all()


def test_npz_global_mean_matches_measured_value():
    z = _load_npz()
    assert np.abs(z["global_mean"] - GLOBAL_MEAN_MEASURED).max() < 0.01
    assert (z["global_std"] >= 0.10).all() and (z["global_std"] <= 0.30).all()


def test_npz_root_z_medians_are_stance_height():
    """docs/PROJECT_KNOWLEDGE.md's root-z-median claim (0.78-0.81 m), read
    straight from the play clips this npz was built from -- no FK needed."""
    from tools.mocap.mocap_leg_prior import list_play_clips

    for path in list_play_clips(str(REPO_ROOT / "work" / "g1")):
        root_z = np.loadtxt(path, delimiter=",", ndmin=2)[:, 2]
        med = float(np.median(root_z))
        assert 0.78 <= med <= 0.81, (path, med)


def _synthetic_csv(order_names: list[str], n_frames: int, jump_frames: tuple[int, ...], jump_rad: float) -> np.ndarray:
    """A minimal 36-column clip (root at the origin, identity orientation,
    every joint at 0) except right_elbow_joint, which holds a step function:
    0 up to jump_frames[0]-1, then jump_rad*(k+1) from jump_frames[k] onward.
    Forward kinematics turns each isolated step into one high-speed wrist
    frame -- exactly one swing "event" per entry in jump_frames."""
    data = np.zeros((n_frames, 7 + len(order_names)))
    data[:, 6] = 1.0  # quat xyzw identity: x=y=z=0, w=1 (column 6)
    elbow = 7 + order_names.index("right_elbow_joint")
    level = 0.0
    frames = sorted(jump_frames)
    for i, f in enumerate(frames):
        level = jump_rad * (i + 1)
        data[f:, elbow] = level
    return data


def test_synthetic_three_swing_events_are_found_at_the_expected_strike_frames():
    order_names = [line.strip() for line in open(ORDER_FILE) if line.strip()]
    m, d, _names, qadr = load_robot(SCENE_PATH, ORDER_FILE)
    jump_frames = (20, 45, 70)
    data = _synthetic_csv(order_names, n_frames=90, jump_frames=jump_frames, jump_rad=0.6)

    wrist_speed, events, _foot_min = detect_swings(data, m, d, qadr)

    assert len(events) == 3
    assert [e[2] for e in events] == list(jump_frames)  # peak_frame == the jump frame
    for (start, end, peak, peak_speed), f in zip(events, jump_frames):
        assert start == end == peak == f
        assert peak_speed > SWING_THRESH_MPS
    # a lone step produces exactly one frame above threshold -- the frame
    # immediately before and after each jump must be quiet
    for f in jump_frames:
        assert wrist_speed[f - 1] <= SWING_THRESH_MPS
        assert wrist_speed[f + 1] <= SWING_THRESH_MPS


def test_find_events_merges_short_gaps_and_keeps_separate_events_apart():
    """Direct unit test of the merge rule: two above-threshold runs 2 frames
    apart (<= GAP_TOL_FRAMES) merge into one event; two runs (GAP_TOL_FRAMES+1)
    apart do not."""
    speed = np.zeros(30)
    speed[5:8] = 3.0
    speed[10:13] = 3.0   # gap of 2 frames after the first run -> merges
    speed[20] = 5.0       # isolated, far away -> its own event
    events = find_events(speed, SWING_THRESH_MPS, GAP_TOL_FRAMES)
    assert len(events) == 2
    assert events[0][0] == 5 and events[0][1] == 12  # merged run spans 5..12
    assert events[1] == (20, 20, 20, 5.0)

    speed2 = np.zeros(30)
    speed2[5:8] = 3.0
    speed2[13:16] = 3.0  # gap of 5 frames (> GAP_TOL_FRAMES) -> stays separate
    events2 = find_events(speed2, SWING_THRESH_MPS, GAP_TOL_FRAMES)
    assert len(events2) == 2
