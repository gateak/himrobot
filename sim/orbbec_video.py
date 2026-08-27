"""Stitch an Orbbec session's colour frames into a playable MP4 so a human can
watch the raw capture. Works on an extracted session directory or the tar.

    ~/him-tools/cv-venv/bin/python sim/orbbec_video.py \\
        --session data/orbbec_20260815_215350 --cam cam0 --fps 0
``--fps 0`` = real-time playback from the host timestamps (median frame gap);
frames are written as-is, so dropped frames are not re-timed.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ball_detect import DEF_CAM, DEF_SESSION, OUT_DIR, Session, iter_frames  # noqa: E402


def open_writer(path, fps, size):
    for fourcc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w, fourcc
        w.release()
    raise SystemExit("no usable MP4 encoder in this OpenCV build")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=DEF_SESSION, help="session directory or .tar")
    ap.add_argument("--tar", help="alias of --session for a .tar (backward compat)")
    ap.add_argument("--cam", default=DEF_CAM)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None, help="default: all frames")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="playback fps; 0 = median host-timestamp rate of the session")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sess = Session(args.tar or args.session)
    cam = sess.resolve_cam(args.cam)
    fps = args.fps
    if fps <= 0:
        t = np.array([r["host_time_s"] for r in sess.frames(cam)])
        fps = float(round(1.0 / np.median(np.diff(t)), 2)) if len(t) > 1 else 30.0
    out = args.out or os.path.join(OUT_DIR, f"{sess.name}_{cam}_color.mp4")
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)

    writer, codec, n = None, None, 0
    for _idx, _t, img in iter_frames(sess, cam, args.start, args.count):
        if writer is None:
            h, w = img.shape[:2]
            writer, codec = open_writer(out, fps, (w, h))
        writer.write(img)
        n += 1
    if writer is not None:
        writer.release()
    print(f"wrote {out}  ({n} frames @ {fps} fps playback, codec {codec})")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
