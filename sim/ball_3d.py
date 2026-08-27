"""Attach aligned depth to the 2D ball track (perception track, 2D -> 2.5D).

For every detection in a ``ball2d_*.csv`` from ``sim/ball_detect.py`` this
opens the depth PNG of the same frame (uint16, ``depth_png_unit`` from
``session.json`` = millimetres, ``align_depth_to_color: true`` so (u, v) is the
same pixel in both images), takes the median of the non-zero depth values
inside the ball disc, and emits ``(frame, host_time_s, u, v, radius, depth_mm,
n_valid, bg_depth_mm)``. ``bg_depth_mm`` is the median of an annulus around
the disc: if ``depth_mm`` is about equal to it, the depth map did not resolve
the ball and you are looking at the table/floor behind it.

Metric XYZ needs the colour intrinsics (fx, fy, cx, cy [, distortion]) which
are NOT stored in the session; nothing here guesses them. If you have them,
pass ``--intrinsics k.json`` (``{"fx":..,"fy":..,"cx":..,"cy":..}``, or a full
``work/calib/intrinsics_<serial>.json`` from ``sim/estimate_intrinsics.py``)
and camera-frame ``X_cam, Y_cam, Z_cam`` (m) are added. If the JSON also
carries a ``table_frame.camera_to_table_4x4`` (the estimate_intrinsics.py
output does), ``X_table, Y_table, Z_table`` (m, table frame: origin at the
net centre, x toward this camera's end line, y across the width, z up) are
added too.

    ~/him-tools/cv-venv/bin/python sim/ball_3d.py \\
        --session data/orbbec_20260815_215350 --cam cam0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ball_detect import DEF_CAM, DEF_SESSION, OUT_DIR, Session  # noqa: E402


def disc_depth(depth, u, v, r, ring=(3, 8)):
    """Median non-zero depth inside the disc and in an annulus around it."""
    h, w = depth.shape
    r_in = max(2, int(round(r)))
    r_out = r_in + ring[1]
    x0, x1 = max(0, int(u) - r_out), min(w, int(u) + r_out + 1)
    y0, y1 = max(0, int(v) - r_out), min(h, int(v) + r_out + 1)
    if x1 <= x0 or y1 <= y0:
        return None, 0, None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - u) ** 2 + (yy - v) ** 2
    patch = depth[y0:y1, x0:x1]
    disc = patch[(d2 <= r_in ** 2) & (patch > 0)]
    ann = patch[(d2 >= (r_in + ring[0]) ** 2) & (d2 <= r_out ** 2) & (patch > 0)]
    dm = float(np.median(disc)) if disc.size else None
    bg = float(np.median(ann)) if ann.size else None
    return dm, int(disc.size), bg


def find_intrinsics(sess):
    """Report anything in the session that looks like calibration (expected: nothing)."""
    keys = ("intrinsic", "fx", "fy", "cx", "cy", "camera_matrix", "distortion", "calib", "extrinsic")
    found = {}
    sj = sess.read_json("session.json") or {}
    hits = [k for k in json.dumps(sj).lower().split('"') if any(x in k for x in keys)]
    if hits:
        found["session.json"] = sorted(set(hits))
    for cam in sess.cams():
        row = sess.frames(cam)[0]
        hits = [k for k in json.dumps(row).lower().split('"') if any(x in k for x in keys)]
        if hits:
            found[f"{cam}/frames.jsonl"] = sorted(set(hits))
    return sj, found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=DEF_SESSION)
    ap.add_argument("--cam", default=DEF_CAM)
    ap.add_argument("--track", default=None, help="ball2d CSV (default: work/perception/ball2d_<tag>.csv)")
    ap.add_argument("--intrinsics", default=None, help="JSON with fx, fy, cx, cy (optional)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sess = Session(args.session)
    cam = sess.resolve_cam(args.cam)
    tag = f"{sess.name}_{cam}"
    track = args.track or os.path.join(OUT_DIR, f"ball2d_{tag}.csv")
    out = args.out or os.path.join(OUT_DIR, f"ball3d_{tag}.csv")

    sj, calib = find_intrinsics(sess)
    unit = sj.get("depth_png_unit")
    print(f"depth_png_unit={unit!r} align_depth_to_color={sj.get('align_depth_to_color')!r}")
    print("calibration-like keys in session files:", calib or "NONE")
    if unit != "millimeters_uint16":
        print(f"WARNING: unexpected depth unit {unit!r}; depth_mm column assumes uint16 mm")
    K = None
    M = None
    if args.intrinsics:
        with open(args.intrinsics) as f:
            K = json.load(f)
        tf = K.get("table_frame")
        if tf and tf.get("camera_to_table_4x4"):
            M = np.array(tf["camera_to_table_4x4"], dtype=np.float64)
            print(f"table_frame found ({tf.get('note', '')}) -> adding X_table/Y_table/Z_table")
        else:
            print("no table_frame in --intrinsics -> X_cam/Y_cam/Z_cam only")

    rows = {int(r["idx"]): r for r in sess.frames(cam)}
    with open(track) as f:
        dets = [r for r in csv.DictReader(f) if r["u"]]
    n_ok = n_nodepth = n_nofile = 0
    depth_scale_seen = set()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["frame", "host_time_s", "u", "v", "radius", "depth_mm", "n_valid", "bg_depth_mm"]
        if K:
            cols += ["X_cam", "Y_cam", "Z_cam"]
        if M is not None:
            cols += ["X_table", "Y_table", "Z_table"]
        w.writerow(cols)
        for d in dets:
            fi = int(d["frame"])
            u, v, r = float(d["u"]), float(d["v"]), float(d["radius"])
            fr = rows.get(fi)
            depth = None
            if fr and fr.get("depth_file"):
                depth = sess.read_image(cam, fr["depth_file"], cv2.IMREAD_UNCHANGED)
                depth_scale_seen.add(fr.get("depth_scale"))
            if depth is None:
                n_nofile += 1
                w.writerow([fi, d["host_time_s"], d["u"], d["v"], d["radius"], "", 0, ""])
                continue
            if depth.dtype != np.uint16:
                raise SystemExit(f"{fr['depth_file']}: dtype {depth.dtype}, expected uint16")
            dm, n, bg = disc_depth(depth, u, v, r)
            if dm is None:
                n_nodepth += 1
            else:
                n_ok += 1
            row = [fi, d["host_time_s"], d["u"], d["v"], d["radius"],
                   "" if dm is None else f"{dm:.0f}", n, "" if bg is None else f"{bg:.0f}"]
            if K and dm is not None:
                z = dm / 1000.0
                x_cam = (u - K["cx"]) / K["fx"] * z
                y_cam = (v - K["cy"]) / K["fy"] * z
                row += [f"{x_cam:.4f}", f"{y_cam:.4f}", f"{z:.4f}"]
                if M is not None:
                    p_table = M[:3, :3] @ np.array([x_cam, y_cam, z]) + M[:3, 3]
                    row += [f"{p_table[0]:.4f}", f"{p_table[1]:.4f}", f"{p_table[2]:.4f}"]
            elif K:
                row += ["", "", ""]
                if M is not None:
                    row += ["", "", ""]
            w.writerow(row)
    print(f"{tag}: {len(dets)} detections -> depth ok {n_ok}, all-zero disc {n_nodepth}, "
          f"no depth file {n_nofile}; frames.jsonl depth_scale values seen: {sorted(depth_scale_seen, key=str)}")
    print(f"wrote {out}")
    if not K:
        print("no intrinsics -> no XYZ (see docs/PROJECT_KNOWLEDGE.md §6a for how to obtain them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
