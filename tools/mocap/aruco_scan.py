"""Scan Orbbec colour frames for ArUco/AprilTag fiducials. Tries the common
OpenCV dictionaries on a sampled set of frames per camera, reports which one
fires with its marker IDs and median pixel corners (for later table-frame
calibration), and writes a JSON summary plus annotated PNGs.

    ~/him-tools/cv-venv/bin/python tools/mocap/aruco_scan.py \\
        --session data/orbbec_20260815_215350 --frames 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ball_detect import DEF_SESSION, OUT_DIR, Session  # noqa: E402

DICTS = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_250", "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]


def sample_rows(sess, cam, n):
    rows = [r for r in sess.frames(cam) if r.get("color_file")]
    if len(rows) <= n:
        return rows
    step = len(rows) / float(n)
    return [rows[int(i * step)] for i in range(n)]


def scan_cam(sess, cam, n_frames, dict_names):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detectors = {d: cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, d)), params)
                 for d in dict_names}
    rows = sample_rows(sess, cam, n_frames)
    per_dict = {d: {"frames_with_markers": 0, "total_markers": 0, "ids": {}} for d in dict_names}
    images = {}
    for r in rows:
        img = sess.read_image(cam, r["color_file"])
        if img is None:
            continue
        images[r["idx"]] = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for d, det in detectors.items():
            corners, ids, _ = det.detectMarkers(gray)
            if ids is None or len(ids) == 0:
                continue
            per_dict[d]["frames_with_markers"] += 1
            per_dict[d]["total_markers"] += int(len(ids))
            for c, i in zip(corners, ids.ravel()):
                per_dict[d]["ids"].setdefault(int(i), []).append(c.reshape(4, 2).tolist())
    # Collapse per-id corner lists into medians + counts.
    for d in per_dict:
        ids = per_dict[d]["ids"]
        per_dict[d]["ids"] = {
            str(i): {"n_frames": len(cs),
                     "corners_px_median": np.median(np.array(cs), axis=0).round(2).tolist(),
                     "center_px": np.median(np.array(cs), axis=0).mean(axis=0).round(2).tolist()}
            for i, cs in sorted(ids.items())}
    return rows, images, per_dict


def annotate(img, detector, label):
    out = img.copy()
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is not None and len(ids):
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        for c, i in zip(corners, ids.ravel()):
            x, y = c.reshape(4, 2).mean(axis=0)
            cv2.putText(out, str(int(i)), (int(x) - 8, int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out, (0 if ids is None else int(len(ids)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=DEF_SESSION)
    ap.add_argument("--cams", nargs="*", default=None, help="default: all cams in the session")
    ap.add_argument("--frames", type=int, default=12, help="frames sampled per camera")
    ap.add_argument("--dicts", nargs="*", default=DICTS)
    ap.add_argument("--annotate", type=int, default=3, help="annotated PNGs per camera")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    if args.frames <= 0:
        ap.error("--frames must be > 0")

    sess = Session(args.session)
    cams = [sess.resolve_cam(c) for c in args.cams] if args.cams else sess.cams()
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {"session": sess.name, "frames_per_cam": args.frames, "dicts": args.dicts, "cams": {}}
    for cam in cams:
        _, images, per_dict = scan_cam(sess, cam, args.frames, args.dicts)
        ranked = sorted(per_dict.items(), key=lambda kv: (kv[1]["total_markers"], len(kv[1]["ids"])),
                        reverse=True)
        best = ranked[0][0] if ranked and ranked[0][1]["total_markers"] else None
        print(f"{sess.name}/{cam}: sampled {len(images)} frames")
        for d, st in ranked:
            if st["total_markers"]:
                print(f"  {d:<24} markers={st['total_markers']:4d} in {st['frames_with_markers']}/{len(images)} "
                      f"frames, ids={sorted(int(i) for i in st['ids'])}")
        summary["cams"][cam] = {"sampled_frames": sorted(images), "best_dict": best,
                                "per_dict": {d: st for d, st in per_dict.items() if st["total_markers"]}}
        if best is None:
            continue
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, best)),
                                      cv2.aruco.DetectorParameters())
        idxs = sorted(images)
        picks = idxs[:: max(1, len(idxs) // args.annotate)][: args.annotate] if args.annotate > 0 else []
        for i in picks:
            out, n = annotate(images[i], det, f"{cam[:4]} f{i} {best}")
            cv2.putText(out, f"{n} markers", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            p = os.path.join(args.out_dir, f"aruco_{sess.name}_{cam}_{i:06d}.png")
            cv2.imwrite(p, out)
            print(f"  wrote {p} ({n} markers)")
    p = os.path.join(args.out_dir, f"aruco_{sess.name}.json")
    with open(p, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
