"""Stitch an Orbbec session's color frames (stored as JPGs inside the tar) into
a playable MP4 so you can watch the raw capture footage.

    ./.venv/bin/python sim/orbbec_video.py --start 0 --count 1800 --fps 30
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tar", default=DEF_TAR)
    ap.add_argument("--color-dir", default=DEF_DIR)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=1800)
    ap.add_argument("--fps", type=int, default=30, help="playback fps (footage is 60; 30 = half-speed)")
    ap.add_argument("--out", default=os.path.join(REPO, "sim", "orbbec_footage.mp4"))
    args = ap.parse_args()

    want = {f"{args.color_dir}/{i:06d}.jpg" for i in range(args.start, args.start + args.count)}
    writer = None
    n = 0
    with tarfile.open(args.tar) as tf:
        for m in tf:  # archive order is ascending frame index
            if m.name not in want:
                continue
            buf = np.frombuffer(tf.extractfile(m).read(), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (w, h))
            writer.write(img)
            n += 1
            if n >= args.count:
                break
    if writer is not None:
        writer.release()
    print(f"wrote {args.out}  ({n} frames @ {args.fps}fps playback)")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
