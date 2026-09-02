#!/usr/bin/env python3
"""Mocap lower-body pose prior: phase-binned leg-joint statistics from the
retargeted play clips, for a whole-body controller's mocap-prior reward term
(docs/design/whole_body_v1.md §4.4, work item WB5).

    .venv/bin/python tools/mocap/mocap_leg_prior.py
        [--preview N --preview-out work/wbc/preview] [--notes 'text']

Method (matches tools/mocap/retarget_report.py::robot_fk and the scratchpad
swing heuristic, mocap_swing_stats.py, exactly -- both re-verified against
this script's own output before it was written):

  1. work/g1/Clip_2.csv .. Clip_8.csv (36 columns: root xyz, quat xyzw, 29
     joints in gmr_g1_joint_order.txt == scene.JOINT_ORDER, 30 fps).
     Clip.csv (the 17.5 s calibration take -- four slow paddle wobbles, not
     play) is excluded.
  2. Per clip, FK on every frame with assets/unitree_g1/scene.xml gives the
     right_wrist_yaw_link position and the ankle-roll sphere geoms' min z.
     Right-wrist speed (frame-aligned finite difference, leading frame = 0)
     over a 2 m/s threshold, merged across gaps <= 3 frames (0.1 s), marks
     swing events; each event's strike frame is its own peak-speed frame.
  3. A 1.5 s window around each strike frame (0.6 s before, 0.9 s after) is
     normalised into 10 phase bins. A window is dropped if: its event's peak
     speed exceeds 8 m/s (frame 0 of every clip is the rest T-pose, so the
     0->1 step is a 13-22 m/s jump, not a swing; one more such spike sits
     mid-Clip_5); any joint in the window sits within 1e-3 rad of a joint
     limit (retarget_report.py's saturation criterion, checked against the
     tighter of the scene's ranges and GMR's own IK-model ranges when
     --ik-model exists -- GMR clamps at the IK ranges, e.g. hip pitch
     +-1.57 rad, so the scene's wider ranges alone miss those clamps); the
     window's minimum foot z sinks below -0.05 m; or (only if
     --wrist-error-json names a file that exists) any frame's wrist error
     there exceeds 0.20 m. Those are the design's mask rules and nothing
     more: a 5-10 cm pelvis dip during a swing is a knee bend the prior
     should keep, not a reject.
  4. Kept windows are averaged per phase bin, weighted 0.5 for Clip_7 (worst
     joint-limit saturation and the most ambiguous timeline alignment of the
     seven play clips) and 1.0 for the rest.

Writes assets/wbc/mocap_leg_prior.npz (default --out):
  bins_mean[10,12], bins_std[10,12]   -- per-phase-bin leg-joint mean/std,
                                          std floored at 0.05 rad
  global_mean[12], global_std[12]     -- leg-joint mean/std over every play
                                          frame, unmasked and unweighted
  upper_bins_mean[10,17], upper_bins_std[10,17]
                                       -- same phase bins, waist(3) + both
                                          arms(7+7), for later WBC lanes
  root_xy_speed_bins[10]              -- mean planar root speed per bin, for
                                          the stepping lane
  n_events_per_clip, n_events_kept_per_clip, clip_weights, clip_names
                                       -- one entry per play clip, same order
  joint_names                         -- the 12 leg names, JOINT_ORDER order
  sat_range_source                    -- "scene+ik_model" or "scene": which
                                          joint ranges the saturation mask used
  fps, window_s, notes

--preview renders N kept swings (a Clip_4 swing overlapping frames
5521-6121 first, per docs/PROJECT_KNOWLEDGE.md §4c) with
tools/mocap/play_g1_csv.py --video into --preview-out, for a human check of
the strike-frame proxy; --notes writes that verdict into the npz.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/mocap/ -> repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import retarget_report  # noqa: E402  (local module: robot_fk, load_robot, joint_ranges)

FPS = 30.0
DT = 1.0 / FPS
SWING_THRESH_MPS = 2.0        # right-wrist speed that flags a swing frame
GAP_TOL_FRAMES = 3            # merge swing runs separated by <= this many quiet frames (0.1 s)
WINDOW_PRE_S = 0.6
WINDOW_POST_S = 0.9
N_PHASE_BINS = 10
SAT_TOL_RAD = 1e-3            # joint-limit saturation tolerance (retarget_report.py's own criterion)
FOOT_Z_REJECT_M = -0.05       # window rejected if its minimum foot z sinks below this
PEAK_SPEED_REJECT_MPS = 8.0   # event rejected if its peak wrist speed exceeds this (the T-pose->play jump at frame 1, or an FK spike)
WRIST_ERR_REJECT_M = 0.20     # optional --wrist-error-json mask threshold (design doc §4.4); no such file exists today
CALIBRATION_CLIP = "Clip.csv"       # four slow paddle wobbles, not play -- excluded
CLIP_WEIGHT_OVERRIDE = {"Clip_7": 0.5}  # worst saturation + most ambiguous timeline alignment (docs/archive)
N_LEGS = 12
N_UPPER = 17  # waist(3) + left arm(7) + right arm(7)
PREVIEW_CLIP4_FRAMES = (5521, 6121)  # docs/PROJECT_KNOWLEDGE.md §4c's named 20 s segment

DEFAULT_CLIP_DIR = os.path.join(REPO, "work", "g1")
DEFAULT_SCENE = os.environ.get("G1_SCENE") or os.path.join(REPO, "assets", "unitree_g1", "scene.xml")
DEFAULT_IK_MODEL = retarget_report.IK_MODEL  # GMR's g1_mocap_29dof.xml: the ranges the retarget clamps at (outside the repo)
DEFAULT_OUT = os.path.join(REPO, "assets", "wbc", "mocap_leg_prior.npz")
DEFAULT_PREVIEW_OUT = os.path.join(REPO, "work", "wbc", "preview")
PLAY_SCRIPT = os.path.join(_HERE, "play_g1_csv.py")


def list_play_clips(clip_dir: str) -> list[str]:
    """work/g1/Clip_2.csv .. Clip_8.csv, numeric order; the calibration take
    (Clip.csv) is excluded."""
    paths = [p for p in glob.glob(os.path.join(clip_dir, "Clip*.csv"))
             if os.path.basename(p) != CALIBRATION_CLIP]

    def _key(path):
        m = re.search(r"_(\d+)$", os.path.splitext(os.path.basename(path))[0])
        return int(m.group(1)) if m else 0

    return sorted(paths, key=_key)


def frame_speed(pos: np.ndarray) -> np.ndarray:
    """Frame-aligned speed: speed[i] = |pos[i] - pos[i-1]| / dt for i >= 1,
    speed[0] = 0 (no previous frame). The leading zero can never cross the
    swing threshold, so it never changes an event's boundaries relative to
    the plain np.diff the scratchpad heuristic used (verified: both give the
    same per-clip event counts on work/g1/*.csv)."""
    speed = np.zeros(len(pos))
    if len(pos) > 1:
        speed[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1) / DT
    return speed


def find_events(speed: np.ndarray, thresh: float, gap_tol: int) -> list[tuple[int, int, int, float]]:
    """Contiguous speed > thresh runs, merged across gaps <= gap_tol frames
    (the scratchpad mocap_swing_stats.py heuristic). Returns
    (start, end, peak_frame, peak_speed) per event, start/end inclusive frame
    indices into `speed` and peak_frame the argmax of `speed` within them."""
    spans = []
    start = end = None
    gap = 0
    for i, above in enumerate(speed > thresh):
        if above:
            if start is None:
                start = i
            end = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > gap_tol:
                spans.append((start, end))
                start = end = None
    if start is not None:
        spans.append((start, end))
    events = []
    for s, e in spans:
        peak = s + int(np.argmax(speed[s:e + 1]))
        events.append((s, e, peak, float(speed[peak])))
    return events


def detect_swings(data: np.ndarray, m, d, qadr) -> tuple[np.ndarray, list[tuple[int, int, int, float]], np.ndarray]:
    """data: (F, 7 + len(qadr)) raw CSV rows (root xyz, quat xyzw, joints).
    Returns (wrist_speed[F], events, foot_min[F])."""
    out, foot_min = retarget_report.robot_fk(m, d, qadr, data)
    wrist_speed = frame_speed(out["right_wrist_yaw_link"])
    events = find_events(wrist_speed, SWING_THRESH_MPS, GAP_TOL_FRAMES)
    return wrist_speed, events, foot_min


def saturation_mask(joints: np.ndarray, jnt_range: np.ndarray, tol: float) -> np.ndarray:
    """joints: (F, 29). True per frame if ANY joint sits within `tol` of its
    model limit (retarget_report.py's own criterion)."""
    lo, hi = jnt_range[:, 0], jnt_range[:, 1]
    return (np.abs(joints - lo) < tol).any(axis=1) | (np.abs(joints - hi) < tol).any(axis=1)


def load_wrist_error_mask(path: str | None) -> dict[str, np.ndarray]:
    """Optional --wrist-error-json: {clip_filename: [per-frame error_m, ...]}.
    No such file exists today (per-frame wrist error against the human clip
    is only computable with the BVH, which this tool does not load); if the
    path is absent this returns {} and the mask is simply not applied."""
    if not path:
        return {}
    if not os.path.isfile(path):
        print(f"--wrist-error-json {path} not found; skipping the optional wrist-error mask", file=sys.stderr)
        return {}
    with open(path) as h:
        raw = json.load(h)
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def process_clip(path: str, m, d, qadr, order_names: list[str], jnt_range: np.ndarray,
                  wrist_err_by_clip: dict[str, np.ndarray]) -> dict:
    """Loads one clip, detects swings, and quality-masks a phase window
    around each kept one. Returns per-clip data the caller bins across clips."""
    clip = os.path.basename(path)
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    assert data.shape[1] == 7 + len(order_names), (path, data.shape)
    n = data.shape[0]
    joints = data[:, 7:7 + len(order_names)]
    root_speed_xy = frame_speed(data[:, 0:2])
    sat = saturation_mask(joints, jnt_range, SAT_TOL_RAD)
    wrist_err = wrist_err_by_clip.get(clip)
    if wrist_err is not None and len(wrist_err) != n:
        print(f"{clip}: --wrist-error-json entry has {len(wrist_err)} rows, clip has {n}; ignoring", file=sys.stderr)
        wrist_err = None

    _wrist_speed, events, foot_min = detect_swings(data, m, d, qadr)
    pre_frames = int(round(WINDOW_PRE_S * FPS))
    post_frames = int(round(WINDOW_POST_S * FPS))
    win_len = pre_frames + post_frames  # phase-normalisation denominator

    kept = []
    for _s, _e, peak, peak_speed in events:
        if peak_speed > PEAK_SPEED_REJECT_MPS:
            continue
        w_start, w_end = peak - pre_frames, peak + post_frames
        v_start, v_end = max(0, w_start), min(n - 1, w_end)
        if sat[v_start:v_end + 1].any():
            continue
        if foot_min[v_start:v_end + 1].min() < FOOT_Z_REJECT_M:
            continue
        if wrist_err is not None and (wrist_err[v_start:v_end + 1] > WRIST_ERR_REJECT_M).any():
            continue
        kept.append((w_start, v_start, v_end, peak))

    return dict(clip=clip, n=n, joints=joints, root_speed_xy=root_speed_xy,
                n_events=len(events), kept=kept, win_len=win_len)


def saturation_ranges(scene: str, ik_model: str | None, order_names: list[str]) -> tuple[np.ndarray, str]:
    """(29, 2) joint ranges for the saturation mask: the scene's, tightened
    per joint by GMR's IK-model ranges when that file exists (the retarget
    clamps at the IK ranges, which are narrower on hips, waist yaw,
    shoulders and elbows -- docs/PROJECT_KNOWLEDGE.md §4c). Second value
    names the source for the npz."""
    rng = retarget_report.joint_ranges(scene, order_names)
    if ik_model and os.path.isfile(ik_model):
        ik = retarget_report.joint_ranges(ik_model, order_names)
        rng = np.stack([np.maximum(rng[:, 0], ik[:, 0]), np.minimum(rng[:, 1], ik[:, 1])], axis=1)
        return rng, "scene+ik_model"
    print(f"--ik-model {ik_model!r} not found; saturation mask uses the scene's ranges only "
          "(misses the retarget's own clamps at the tighter GMR IK limits)", file=sys.stderr)
    return rng, "scene"


def build_prior(clip_dir: str, scene: str, order_file: str, wrist_error_json: str | None = None,
                ik_model: str | None = DEFAULT_IK_MODEL) -> dict:
    """Runs process_clip over every play clip and assembles the npz payload
    (as a plain dict of arrays -- np.savez(**this) at the call site)."""
    order_names = [line.strip() for line in open(order_file) if line.strip()]
    m, d, _names, qadr = retarget_report.load_robot(scene, order_file)
    jnt_range, sat_range_source = saturation_ranges(scene, ik_model, order_names)
    wrist_err_by_clip = load_wrist_error_mask(wrist_error_json)

    clips = list_play_clips(clip_dir)
    if not clips:
        raise FileNotFoundError(f"no play clips found under {clip_dir}")

    leg_sum_w = np.zeros(N_PHASE_BINS)  # one weight total per bin -- shared across all 12 leg joints
    leg_sum_wx = np.zeros((N_PHASE_BINS, N_LEGS))
    leg_sum_wx2 = np.zeros((N_PHASE_BINS, N_LEGS))
    up_sum_w = np.zeros(N_PHASE_BINS)   # one weight total per bin -- shared across all 17 upper joints
    up_sum_wx = np.zeros((N_PHASE_BINS, N_UPPER))
    up_sum_wx2 = np.zeros((N_PHASE_BINS, N_UPPER))
    root_sum_w = np.zeros(N_PHASE_BINS)
    root_sum_wx = np.zeros(N_PHASE_BINS)

    all_leg_joints = []
    clip_names, n_events_per_clip, n_events_kept_per_clip, clip_weights = [], [], [], []
    per_clip_kept_windows = {}  # clip -> [(w_start, v_start, v_end, peak), ...], for --preview

    for path in clips:
        clip = os.path.basename(path)
        weight = CLIP_WEIGHT_OVERRIDE.get(os.path.splitext(clip)[0], 1.0)
        result = process_clip(path, m, d, qadr, order_names, jnt_range, wrist_err_by_clip)

        all_leg_joints.append(result["joints"][:, :N_LEGS])
        clip_names.append(clip)
        n_events_per_clip.append(result["n_events"])
        n_events_kept_per_clip.append(len(result["kept"]))
        clip_weights.append(weight)
        per_clip_kept_windows[clip] = [(w_start, v_start, v_end, peak) for w_start, v_start, v_end, peak in result["kept"]]

        for w_start, v_start, v_end, peak in result["kept"]:
            for idx in range(v_start, v_end + 1):
                phase = (idx - w_start) / result["win_len"]
                b = min(max(int(phase * N_PHASE_BINS), 0), N_PHASE_BINS - 1)
                legs = result["joints"][idx, :N_LEGS]
                upper = result["joints"][idx, N_LEGS:N_LEGS + N_UPPER]
                leg_sum_w[b] += weight; leg_sum_wx[b] += weight * legs; leg_sum_wx2[b] += weight * legs ** 2
                up_sum_w[b] += weight; up_sum_wx[b] += weight * upper; up_sum_wx2[b] += weight * upper ** 2
                root_sum_w[b] += weight; root_sum_wx[b] += weight * result["root_speed_xy"][idx]

    def _weighted_mean_std(sum_w, sum_wx, sum_wx2):
        """sum_w: (10,); sum_wx/sum_wx2: (10, n_joints). An empty bin (no
        kept frames fell in it) divides by 1 instead of 0, giving mean=std=0
        -- harmless, since bins_std is floored below regardless."""
        w = np.where(sum_w > 0, sum_w, 1.0)[:, None]
        mean = sum_wx / w
        var = np.maximum(sum_wx2 / w - mean ** 2, 0.0)
        return mean, np.sqrt(var)

    bins_mean, bins_std = _weighted_mean_std(leg_sum_w, leg_sum_wx, leg_sum_wx2)
    bins_std = np.maximum(bins_std, 0.05)
    upper_bins_mean, upper_bins_std = _weighted_mean_std(up_sum_w, up_sum_wx, up_sum_wx2)
    upper_bins_std = np.maximum(upper_bins_std, 0.05)
    root_xy_speed_bins = root_sum_wx / np.where(root_sum_w > 0, root_sum_w, 1.0)

    all_legs = np.concatenate(all_leg_joints, axis=0)  # every play frame, unmasked, unweighted
    global_mean = all_legs.mean(axis=0)
    global_std = all_legs.std(axis=0)

    return dict(
        bins_mean=bins_mean, bins_std=bins_std,
        global_mean=global_mean, global_std=global_std,
        upper_bins_mean=upper_bins_mean, upper_bins_std=upper_bins_std,
        root_xy_speed_bins=root_xy_speed_bins,
        n_events_per_clip=np.asarray(n_events_per_clip, dtype=np.int64),
        n_events_kept_per_clip=np.asarray(n_events_kept_per_clip, dtype=np.int64),
        clip_weights=np.asarray(clip_weights, dtype=np.float64),
        clip_names=np.asarray(clip_names),
        joint_names=np.asarray(order_names[:N_LEGS]),
        sat_range_source=sat_range_source,
        fps=np.float64(FPS), window_s=np.float64(WINDOW_PRE_S + WINDOW_POST_S),
        _per_clip_kept_windows=per_clip_kept_windows,  # dropped before np.savez; used only by --preview
    )


def select_preview_swings(per_clip_kept_windows: dict, n: int) -> list[tuple[str, int, int, int]]:
    """Picks up to n kept (clip, v_start, v_end, peak) windows: a Clip_4
    window overlapping PREVIEW_CLIP4_FRAMES first if one was kept, then the
    rest in clip order."""
    picks = []
    clip4 = per_clip_kept_windows.get("Clip_4.csv", [])
    lo, hi = PREVIEW_CLIP4_FRAMES
    for w_start, v_start, v_end, peak in clip4:
        if v_start <= hi and v_end >= lo:
            picks.append(("Clip_4.csv", v_start, v_end, peak))
            break
    for clip, windows in per_clip_kept_windows.items():
        if len(picks) >= n:
            break
        for w_start, v_start, v_end, peak in windows:
            if len(picks) >= n:
                break
            entry = (clip, v_start, v_end, peak)
            if entry in picks:
                continue
            picks.append(entry)
    return picks[:n]


def render_preview(picks: list[tuple[str, int, int, int]], clip_dir: str, order_file: str, scene: str,
                    preview_out: str) -> list[str]:
    """Writes each pick's raw CSV window and renders it with play_g1_csv.py
    --video, for a human to check the strike-frame proxy against."""
    os.makedirs(preview_out, exist_ok=True)
    written = []
    for clip, v_start, v_end, peak in picks:
        data = np.loadtxt(os.path.join(clip_dir, clip), delimiter=",", ndmin=2)
        sub = data[v_start:v_end + 1]
        base = os.path.join(preview_out, f"{os.path.splitext(clip)[0]}_frame{peak}")
        np.savetxt(base + ".csv", sub, delimiter=",")
        subprocess.run([sys.executable, PLAY_SCRIPT, base + ".csv", "--joint-names", order_file,
                        "--scene", scene, "--fps", str(FPS), "--video", base + ".mp4"], check=True)
        written.append(base + ".mp4")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip-dir", default=DEFAULT_CLIP_DIR)
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--ik-model", default=DEFAULT_IK_MODEL,
                    help="GMR IK model whose (tighter) joint ranges the saturation mask also honours; "
                         "skipped with a warning if the file is absent")
    ap.add_argument("--joint-names", default=None, help="default: gmr_g1_joint_order.txt next to the clips")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--wrist-error-json", default=None)
    ap.add_argument("--preview", type=int, default=0, metavar="N", help="render N kept swings to --preview-out")
    ap.add_argument("--preview-out", default=DEFAULT_PREVIEW_OUT)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    order_file = a.joint_names or os.path.join(a.clip_dir, "gmr_g1_joint_order.txt")

    prior = build_prior(a.clip_dir, a.scene, order_file, a.wrist_error_json, a.ik_model)
    per_clip_kept_windows = prior.pop("_per_clip_kept_windows")
    prior["notes"] = a.notes

    print("clips:", prior["clip_names"].tolist())
    print("saturation ranges:", prior["sat_range_source"])
    print("n_events_per_clip:", prior["n_events_per_clip"].tolist())
    print("n_events_kept_per_clip:", prior["n_events_kept_per_clip"].tolist())
    print("total events:", int(prior["n_events_per_clip"].sum()), "kept:", int(prior["n_events_kept_per_clip"].sum()))
    print("global_mean:", np.round(prior["global_mean"], 4).tolist())
    print("global_std:", np.round(prior["global_std"], 4).tolist())
    print("bins_std range:", float(prior["bins_std"].min()), "-", float(prior["bins_std"].max()))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez(a.out, **prior)
    print("wrote", a.out)

    if a.preview > 0:
        picks = select_preview_swings(per_clip_kept_windows, a.preview)
        written = render_preview(picks, a.clip_dir, order_file, a.scene, a.preview_out)
        print("preview videos:", written)


if __name__ == "__main__":
    main()
