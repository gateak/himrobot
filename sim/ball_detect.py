"""Detect the table-tennis ball in Orbbec color frames (perception track, first pass).

Reads frames straight from the session tar (no bulk extraction). The ball is
small, fast and bright against the dark blue table, so we combine frame
differencing (motion) with brightness + roundness, then track by proximity.
Outputs detection stats, the 2D pixel trajectory (.npy), and annotated sample
frames. Metric 3D (aligned depth + the table's ArUco markers) is the next step.

    ./.venv/bin/python sim/ball_detect.py --start 0 --count 600
"""
from __future__ import annotations

import argparse
import os
import tarfile

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_TAR = os.path.join(REPO, "data", "orbbec_20260815_221035_60fps.tar")
DEF_DIR = "orbbec_20260815_221035_60fps/cam0_CP0E8530008F/color"


def read_frames(tar_path, color_dir, start, count):
    want = {f"{color_dir}/{i:06d}.jpg": i for i in range(start, start + count)}
    got = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if m.name in want:
                buf = np.frombuffer(tf.extractfile(m).read(), np.uint8)
                got[want[m.name]] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if len(got) == len(want):
                    break
    idxs = sorted(got)
    return idxs, [got[i] for i in idxs]


# Orange table-tennis ball in HSV (OpenCV H is 0-179). High saturation rejects
# the gray floor/table; the hue band rejects the blue table and skin.
ORANGE_LO = np.array([3, 95, 80], dtype=np.uint8)
ORANGE_HI = np.array([26, 255, 255], dtype=np.uint8)


def detect(frames, min_area=2.0, max_area=500.0, min_circ=0.35):
    last = None
    dets = []  # per frame: (x, y, r) or None
    kernel = np.ones((3, 3), np.uint8)
    for img in frames:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, -1e9
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area or a > max_area:            # small blob; shoes/cables are bigger
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < 1:
                continue
            circ = a / (np.pi * r * r + 1e-6)           # 1.0 = perfect disc
            if circ < min_circ:
                continue
            prox = 0.0 if last is None else -0.1 * float(np.hypot(x - last[0], y - last[1]))
            score = 0.3 * a + 80 * circ + prox          # round, sizable orange blob, near last
            if score > best_score:
                best_score, best = score, (float(x), float(y), float(r))
        dets.append(best)
        if best is not None:
            last = best[:2]
    return dets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tar", default=DEF_TAR)
    ap.add_argument("--color-dir", default=DEF_DIR)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=600)
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()

    idxs, frames = read_frames(args.tar, args.color_dir, args.start, args.count)
    print(f"loaded {len(frames)} frames [{idxs[0]}..{idxs[-1]}]" if idxs else "no frames")
    if not frames:
        return 1

    dets = detect(frames)
    hit = [(idxs[i], d) for i, d in enumerate(dets) if d is not None]
    print(f"ball detected in {len(hit)}/{len(frames)} frames")
    for fi, d in hit[:12]:
        print(f"  frame {fi}: u={d[0]:.0f} v={d[1]:.0f} r={d[2]:.1f}px")

    # 2D trajectory (nan where undetected)
    traj = np.full((len(frames), 3), np.nan)
    for i, d in enumerate(dets):
        if d is not None:
            traj[i] = d
    np.save(os.path.join(REPO, "data", "ball_traj2d.npy"), traj)

    # Save annotated sample frames (prefer detections) for visual check.
    out = "/tmp/ball_samples"
    os.makedirs(out, exist_ok=True)
    picks = [i for i, d in enumerate(dets) if d is not None]
    if len(picks) > args.samples:
        picks = picks[:: max(1, len(picks) // args.samples)][: args.samples]
    for i in picks:
        img = frames[i].copy()
        d = dets[i]
        cv2.circle(img, (int(d[0]), int(d[1])), max(6, int(d[2]) + 4), (0, 0, 255), 2)
        cv2.putText(img, f"f{idxs[i]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(f"{out}/det_{idxs[i]:06d}.jpg", img)
    print(f"saved {len(picks)} annotated samples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
