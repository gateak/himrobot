"""Self-calibrate Orbbec colour-camera intrinsics from the ArUco table markers
and their aligned depth (perception track: intrinsics, no factory calibration
exists anywhere in the capture -- see docs/PROJECT_KNOWLEDGE.md #6a).

Method (no checkerboard, no pyorbbecsdk -- uses only what is on disk):

1. For a session, re-detect the DICT_4X4_50 ArUco markers of one camera's
   half-table (4 markers, ids fixed per serial -- see SERIAL_IDS) across
   every colour frame in the session. Median the per-frame corner pixels ->
   one stable (u, v) per marker corner (16 points) and per marker centre (4
   points, mean of that marker's 4 corners). At each of those *fixed* pixel
   locations, read the aligned depth PNG across ~20 of the frames the marker
   decoded in (small window, ignore zero, mm -> m) and take the median --
   this is a temporal denoise of a static scene, not a spatial one.

2. Unknowns: fx, fy, cx, cy (pinhole, zero distortion -- stated assumption),
   plus two nuisance parameters: `inset` (how far the printed markers sit in
   from the table edges they mark, assumed equal on all four sides, <= 10 cm)
   and `medge` (the marker's own printed edge length). The unprojection model
   is X=(u-cx)Z/fx, Y=(v-cy)Z/fy, Z=Z. The 4 marker centres of one half-table
   must reconstruct to a 1.525 m (net/end-line pairs) x (1.37 m half-length)
   rectangle shrunk by 2*inset on every side, with right angles and (via a
   scalar triple product, in metres = an out-of-plane distance) zero
   thickness; each marker's own 4 corners must reconstruct to a `medge`
   square. scipy.optimize.least_squares fits fx, fy, cx, cy, inset, medge to
   those residuals jointly. Run per (serial, session); report agreement
   across sessions as the repeatability check, and rerun with cx/cy pinned
   to the image centre and with inset pinned at 0/5/10 cm as a sensitivity
   check on the free-parameter version.

3. Cross-check: fit a plane (SVD) to the depth pixels inside the convex hull
   of the 16 corner points, using the fitted intrinsics to unproject; report
   its RMS (mm), the implied camera height above the table (distance from
   the camera origin to the plane), and the angle between the plane normal
   and the table-edge normal (cross of the two fitted rectangle edges).

4. Output work/calib/intrinsics_<serial>.json (fx, fy, cx, cy, uncertainties,
   inset, marker edge, residuals, the table_frame camera<->table transform)
   and work/calib/REPORT.md.

No factory constant is invented anywhere: fx=fy and cx,cy=image-centre are
only the optimizer's *initial guess*, always re-fit as free parameters in
the primary result, and zero lens distortion is an explicit, never-verified
assumption (its effect is visible only as elevated residuals, not isolated).

    ~/him-tools/cv-venv/bin/python sim/estimate_intrinsics.py
    ~/him-tools/cv-venv/bin/python sim/estimate_intrinsics.py --serial CP0E8530008E
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ball_detect import REPO, Session  # noqa: E402

OUT_DIR = os.path.join(REPO, "work", "calib")

# ITTF regulation table (docs/PROJECT_KNOWLEDGE.md, task header): 2.74 x 1.525 m,
# 0.76 m high, net 0.1525 m. Cameras each see one half -> half-length 1.37 m.
TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
TABLE_HALF_LEN = TABLE_LENGTH / 2.0
IMG_W, IMG_H = 640, 480

# Marker ids per serial and their role in the half-table rectangle. Roles are
# named by which width-side ("near"/"far" the camera, i.e. large/small image
# v) and which length-side ("net"/"end", i.e. the table edge each marks) they
# sit at -- fixed facts from docs/PROJECT_KNOWLEDGE.md #6a (aruco_compare.md):
# physical cameras never moved, serial ...8E always sees ids 4-7, ...8F 0-3.
SERIAL_IDS = {
    "CP0E8530008E": {"near_net": 4, "far_net": 5, "near_end": 6, "far_end": 7},
    "CP0E8530008F": {"near_net": 2, "far_net": 3, "near_end": 0, "far_end": 1},
}
ROLES = ["near_net", "far_net", "near_end", "far_end"]

DEFAULT_SESSIONS = [
    "orbbec_20260815_215350",
    "orbbec_20260815_214831",
    "orbbec_20260815_220448_60fps",
]

DICT_NAME = "DICT_4X4_50"

PARAM_NAMES = ["fx", "fy", "cx", "cy", "inset", "medge"]
INIT = {"fx": 550.0, "fy": 550.0, "cx": (IMG_W - 1) / 2.0, "cy": (IMG_H - 1) / 2.0,
        "inset": 0.03, "medge": 0.10}
LO = {"fx": 150.0, "fy": 150.0, "cx": 0.0, "cy": 0.0, "inset": 0.0, "medge": 0.03}
HI = {"fx": 1200.0, "fy": 1200.0, "cx": IMG_W - 1.0, "cy": IMG_H - 1.0, "inset": 0.10, "medge": 0.20}


# --------------------------------------------------------------------- detect
def find_cam_for_serial(sess, serial):
    hits = [c for c in sess.cams() if serial in c]
    if len(hits) != 1:
        raise SystemExit(f"{sess.name}: serial {serial} matches cams {hits}")
    return hits[0]


def detect_all(sess, cam, ids_wanted, max_frames=None):
    """Re-detect DICT_4X4_50 on every colour frame; return {id: [(idx, 4x2 px), ...]}."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(dictionary, params)
    rows = sess.frames(cam)
    if max_frames:
        rows = rows[:max_frames]
    hits = {i: [] for i in ids_wanted}
    n_seen = 0
    for r in rows:
        if not r.get("color_file"):
            continue
        img = sess.read_image(cam, r["color_file"])
        if img is None:
            continue
        n_seen += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            continue
        for c, i in zip(corners, ids.ravel()):
            i = int(i)
            if i in hits:
                hits[i].append((int(r["idx"]), c.reshape(4, 2).astype(np.float64)))
    return hits, n_seen, {int(r["idx"]): r for r in rows}


def window_median_depth(depth_img, u, v, win):
    h, w = depth_img.shape
    x0, x1 = max(0, int(round(u)) - win), min(w, int(round(u)) + win + 1)
    y0, y1 = max(0, int(round(v)) - win), min(h, int(round(v)) + win + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth_img[y0:y1, x0:x1]
    nz = patch[patch > 0]
    return float(np.median(nz)) if nz.size else None


def collect_role(sess, cam, hits, rows_by_idx, role_id, depth_frames_target, win, depth_cache):
    obs = hits[role_id]
    if len(obs) < 3:
        raise SystemExit(f"{sess.name}/{cam}: id {role_id} decoded only {len(obs)} frames, need >=3")
    obs = sorted(obs, key=lambda t: t[0])
    corners_stack = np.stack([c for _, c in obs], axis=0)  # (N,4,2)
    corners_px = np.median(corners_stack, axis=0)  # (4,2)
    center_px = corners_px.mean(axis=0)
    n_decoded = len(obs)
    idxs = [idx for idx, _ in obs]
    n_pick = min(len(idxs), depth_frames_target)
    step = len(idxs) / float(n_pick)
    pick = sorted({idxs[int(i * step)] for i in range(n_pick)})

    pts_px = list(corners_px) + [center_px]
    per_point_readings = [[] for _ in pts_px]
    for idx in pick:
        if idx not in depth_cache:
            row = rows_by_idx.get(idx)
            depth_cache[idx] = None
            if row and row.get("depth_file"):
                depth_cache[idx] = sess.read_image(cam, row["depth_file"], cv2.IMREAD_UNCHANGED)
        depth_img = depth_cache[idx]
        if depth_img is None:
            continue
        for k, (u, v) in enumerate(pts_px):
            d = window_median_depth(depth_img, u, v, win)
            if d is not None:
                per_point_readings[k].append(d)

    Zs_mm = []
    for readings in per_point_readings:
        if not readings:
            raise SystemExit(f"{sess.name}/{cam}: id {role_id} has a corner/centre with zero valid depth reads")
        Zs_mm.append(float(np.median(readings)))

    return {
        "n_decoded": n_decoded,
        "corners_px": corners_px.tolist(),
        "corners_Z_m": [z / 1000.0 for z in Zs_mm[:4]],
        "center_px": center_px.tolist(),
        "center_Z_m": Zs_mm[4] / 1000.0,
    }


def build_observation(sess, cam, serial, depth_frames_target=20, win=2, max_frames=None):
    ids = SERIAL_IDS[serial]
    hits, n_seen, rows_by_idx = detect_all(sess, cam, set(ids.values()), max_frames)
    depth_cache = {}
    obs = {}
    for role, rid in ids.items():
        obs[role] = collect_role(sess, cam, hits, rows_by_idx, rid, depth_frames_target, win, depth_cache)
    return obs, n_seen


# ----------------------------------------------------------------------- fit
def unproject(u, v, Z, fx, fy, cx, cy):
    return np.array([(u - cx) * Z / fx, (v - cy) * Z / fy, Z])


def residual_fn(obs, free_names, fixed):
    def unpack(p):
        vals = dict(fixed)
        vals.update(zip(free_names, p))
        return [vals[n] for n in PARAM_NAMES]

    def fn(p):
        fx, fy, cx, cy, inset, medge = unpack(p)
        P = {r: unproject(obs[r]["center_px"][0], obs[r]["center_px"][1], obs[r]["center_Z_m"],
                           fx, fy, cx, cy) for r in ROLES}
        W = TABLE_WIDTH - 2 * inset
        L = TABLE_HALF_LEN - 2 * inset
        diag = math.hypot(W, L)
        res = [
            np.linalg.norm(P["far_net"] - P["near_net"]) - W,
            np.linalg.norm(P["far_end"] - P["near_end"]) - W,
            np.linalg.norm(P["near_end"] - P["near_net"]) - L,
            np.linalg.norm(P["far_end"] - P["far_net"]) - L,
            np.linalg.norm(P["far_end"] - P["near_net"]) - diag,
            np.linalg.norm(P["near_end"] - P["far_net"]) - diag,
        ]
        v1 = P["far_net"] - P["near_net"]
        v2 = P["near_end"] - P["near_net"]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        res.append(float(np.dot(v1, v2)) / (n1 * n2) if n1 > 1e-9 and n2 > 1e-9 else 0.0)
        va = P["near_net"] - P["far_net"]
        vb = P["far_end"] - P["far_net"]
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        res.append(float(np.dot(va, vb)) / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0)
        v5 = P["far_end"] - P["near_net"]
        vol = float(np.dot(np.cross(v1, v2), v5))
        res.append(vol / (n1 * n2) if n1 > 1e-9 and n2 > 1e-9 else 0.0)  # out-of-plane distance, m
        for r in ROLES:
            c = [unproject(obs[r]["corners_px"][k][0], obs[r]["corners_px"][k][1], obs[r]["corners_Z_m"][k],
                           fx, fy, cx, cy) for k in range(4)]
            res += [
                np.linalg.norm(c[1] - c[0]) - medge,
                np.linalg.norm(c[2] - c[1]) - medge,
                np.linalg.norm(c[3] - c[2]) - medge,
                np.linalg.norm(c[0] - c[3]) - medge,
                np.linalg.norm(c[2] - c[0]) - medge * math.sqrt(2),
                np.linalg.norm(c[3] - c[1]) - medge * math.sqrt(2),
            ]
        return np.array(res)

    return fn


def fit_variant(obs, free_names, overrides=None):
    fixed = dict(INIT)
    if overrides:
        fixed.update(overrides)
    fixed = {k: v for k, v in fixed.items() if k not in free_names}
    p0 = [INIT[n] for n in free_names]
    lo = [LO[n] for n in free_names]
    hi = [HI[n] for n in free_names]
    fn = residual_fn(obs, free_names, fixed)
    sol = least_squares(fn, p0, bounds=(lo, hi), method="trf", xtol=1e-13, ftol=1e-13)
    result = dict(fixed)
    result.update(zip(free_names, sol.x))
    res = sol.fun
    # residuals 0-5 = table-rectangle side/diagonal lengths, 6-7 = right-angle cosines,
    # 8 = coplanarity (out-of-plane distance), 9+ = 6 per marker x 4 markers of marker-edge lengths.
    length_res_mm = np.concatenate([res[:6], res[9:]]) * 1000.0
    angle_res_deg = -res[6:8] * 180.0 / math.pi
    coplanarity_mm = float(res[8] * 1000.0)
    dof = max(len(res) - len(sol.x), 1)
    s2 = float(np.sum(res ** 2) / dof)
    try:
        cov = s2 * np.linalg.inv(sol.jac.T @ sol.jac)
        perr = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        perr = np.full(len(sol.x), float("nan"))
    stderr = dict(zip(free_names, perr.tolist()))
    return {
        "params": result,
        "stderr": stderr,
        "residual_rms_mm": float(np.sqrt(np.mean(length_res_mm ** 2))),
        "residual_max_mm": float(np.max(np.abs(length_res_mm))),
        "angle_error_deg": angle_res_deg.tolist(),
        "coplanarity_mm": coplanarity_mm,
        "cost": float(sol.cost),
        "n_residuals": len(res),
        "success": bool(sol.success),
    }


# --------------------------------------------------------------- plane check
def _nanmedian_quiet(stack):
    """Column-wise nanmedian; all-NaN columns come back NaN without a RuntimeWarning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(stack, axis=0)


def plane_fit_check(sess, cam, obs, fx, fy, cx, cy, n_frames=8, stride_px=4):
    corners_all = np.concatenate([np.array(obs[r]["corners_px"]) for r in ROLES], axis=0)
    hull = cv2.convexHull(corners_all.astype(np.float32))
    mask = np.zeros((IMG_H, IMG_W), np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1)

    rows = sess.frames(cam)
    idxs = sorted({int(r["idx"]) for r in rows if r.get("depth_file")})
    n_pick = min(len(idxs), n_frames)
    step = len(idxs) / float(n_pick)
    pick = [idxs[int(i * step)] for i in range(n_pick)]
    rows_by_idx = {int(r["idx"]): r for r in rows}
    stack = []
    for idx in pick:
        row = rows_by_idx[idx]
        d = sess.read_image(cam, row["depth_file"], cv2.IMREAD_UNCHANGED)
        if d is not None:
            stack.append(d.astype(np.float32))
    stack = np.stack(stack, axis=0)
    stack[stack == 0] = np.nan
    med = _nanmedian_quiet(stack)

    ys, xs = np.mgrid[0:IMG_H:stride_px, 0:IMG_W:stride_px]
    ys, xs = ys.ravel(), xs.ravel()
    keep = (mask[ys, xs] > 0) & np.isfinite(med[ys, xs])
    ys, xs, zmm = ys[keep], xs[keep], med[ys, xs][keep]
    pts = np.stack([(xs - cx) * zmm / fx / 1000.0, (ys - cy) * zmm / fy / 1000.0, zmm / 1000.0], axis=1)

    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)
    resid = (pts - centroid) @ normal
    rms_mm = float(np.sqrt(np.mean(resid ** 2)) * 1000.0)
    cam_height_m = float(abs(np.dot(normal, centroid)))

    P = {r: unproject(obs[r]["center_px"][0], obs[r]["center_px"][1], obs[r]["center_Z_m"], fx, fy, cx, cy)
         for r in ROLES}
    v1 = P["far_net"] - P["near_net"]
    v2 = P["near_end"] - P["near_net"]
    edge_normal = np.cross(v1, v2)
    edge_normal /= np.linalg.norm(edge_normal)
    angle_deg = float(math.degrees(math.acos(np.clip(abs(np.dot(normal, edge_normal)), 0, 1))))
    return {"n_points": int(pts.shape[0]), "n_frames": len(pick), "rms_mm": rms_mm,
            "camera_height_m": cam_height_m, "normal_vs_table_edges_deg": angle_deg}


# ------------------------------------------------------------------- driver
def run_session(sess_name, serial, depth_frames_target, win, max_frames):
    sess = Session(os.path.join(REPO, "data", sess_name))
    cam = find_cam_for_serial(sess, serial)
    t0 = time.time()
    obs, n_seen = build_observation(sess, cam, serial, depth_frames_target, win, max_frames)
    dt = time.time() - t0
    primary = fit_variant(obs, PARAM_NAMES)
    sensitivity = {
        "cx_cy_fixed_centre": fit_variant(obs, ["fx", "fy", "inset", "medge"],
                                           {"cx": INIT["cx"], "cy": INIT["cy"]}),
        "inset_fixed_0.00": fit_variant(obs, ["fx", "fy", "cx", "cy", "medge"], {"inset": 0.00}),
        "inset_fixed_0.05": fit_variant(obs, ["fx", "fy", "cx", "cy", "medge"], {"inset": 0.05}),
        "inset_fixed_0.10": fit_variant(obs, ["fx", "fy", "cx", "cy", "medge"], {"inset": 0.10}),
    }
    p = primary["params"]
    plane = plane_fit_check(sess, cam, obs, p["fx"], p["fy"], p["cx"], p["cy"])
    return {
        "session": sess_name, "cam": cam, "serial": serial,
        "n_frames_seen": n_seen, "detect_time_s": round(dt, 1),
        "decoded": {r: obs[r]["n_decoded"] for r in ROLES},
        "primary": primary, "sensitivity": sensitivity, "plane_fit": plane,
    }


def kabsch(P_cam, P_table):
    cen_cam, cen_table = P_cam.mean(axis=0), P_table.mean(axis=0)
    A, B = P_cam - cen_cam, P_table - cen_table
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cen_table - R @ cen_cam
    return R, t


def table_frame_transform(obs, p):
    fx, fy, cx, cy, inset = p["fx"], p["fy"], p["cx"], p["cy"], p["inset"]
    W = TABLE_WIDTH - 2 * inset
    canon = {
        "near_net": (inset, -W / 2, 0.0), "far_net": (inset, W / 2, 0.0),
        "near_end": (TABLE_HALF_LEN - inset, -W / 2, 0.0), "far_end": (TABLE_HALF_LEN - inset, W / 2, 0.0),
    }
    P_cam = np.array([unproject(obs[r]["center_px"][0], obs[r]["center_px"][1], obs[r]["center_Z_m"],
                                 fx, fy, cx, cy) for r in ROLES])
    P_table = np.array([canon[r] for r in ROLES])
    R, t = kabsch(P_cam, P_table)
    # The 4 canonical targets are exactly coplanar (Z=0), so Kabsch's 3rd (normal) axis is fit against
    # near-zero out-of-plane spread on both sides -- its SIGN is numerically arbitrary, not physically
    # determined, and came out flipped (table "up" pointing away from the camera) for one of the two
    # serials in practice. Force it: table-frame "up", expressed in camera coordinates, must point from
    # the table toward the camera at the camera-frame origin, i.e. have a negative Z_cam component.
    up_in_cam = R.T @ np.array([0.0, 0.0, 1.0])
    if up_in_cam[2] > 0:
        R = np.diag([1.0, -1.0, -1.0]) @ R  # 180 deg about the table x (length) axis; keeps det(R)=+1
        t = np.diag([1.0, -1.0, -1.0]) @ t
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", nargs="*", default=list(SERIAL_IDS.keys()))
    ap.add_argument("--sessions", nargs="*", default=DEFAULT_SESSIONS)
    ap.add_argument("--depth-frames", type=int, default=20)
    ap.add_argument("--depth-window", type=int, default=2, help="half-width of the depth sample window (px)")
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames decoded per session/cam (debug)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}
    for serial in args.serial:
        runs = []
        for sname in args.sessions:
            sess = Session(os.path.join(REPO, "data", sname))
            if serial not in "".join(sess.cams()):
                print(f"skip {sname}: serial {serial} not present ({sess.cams()})")
                continue
            print(f"=== {serial} / {sname} ===")
            r = run_session(sname, serial, args.depth_frames, args.depth_window, args.max_frames)
            print(f"  decoded (n frames): {r['decoded']}  ({r['n_frames_seen']} frames seen, "
                  f"{r['detect_time_s']}s detect)")
            p = r["primary"]["params"]
            print(f"  primary fit: fx={p['fx']:.1f} fy={p['fy']:.1f} cx={p['cx']:.1f} cy={p['cy']:.1f} "
                  f"inset={p['inset'] * 100:.1f}cm medge={p['medge'] * 100:.1f}cm "
                  f"resid_rms={r['primary']['residual_rms_mm']:.1f}mm")
            runs.append(r)
        if not runs:
            print(f"no sessions available for serial {serial}, skipping")
            continue
        best = max(runs, key=lambda r: sum(r["decoded"].values()))
        sess = Session(os.path.join(REPO, "data", best["session"]))
        cam = find_cam_for_serial(sess, serial)
        obs, _ = build_observation(sess, cam, serial, args.depth_frames, args.depth_window, args.max_frames)
        M = table_frame_transform(obs, best["primary"]["params"])

        summary = summarize_serial_with_transform(serial, runs, M)
        out_path = os.path.join(args.out_dir, f"intrinsics_{serial}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {out_path}")
        all_results[serial] = summary

    if all_results:
        write_report(all_results, args)
    return 0


def summarize_serial_with_transform(serial, runs, M):
    fits = [r["primary"]["params"] for r in runs]
    combined = {}
    for name in PARAM_NAMES:
        vals = np.array([f[name] for f in fits])
        combined[name] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0) if len(vals) > 1 else 0.0),
                           "per_session": vals.tolist(), "min": float(vals.min()), "max": float(vals.max())}
    best = max(runs, key=lambda r: sum(r["decoded"].values()))
    inset_sweep = {k: v["params"]["fx"] for k, v in best["sensitivity"].items() if k.startswith("inset_fixed")}
    return {
        "serial": serial, "width": IMG_W, "height": IMG_H,
        "fx": combined["fx"]["mean"], "fy": combined["fy"]["mean"],
        "cx": combined["cx"]["mean"], "cy": combined["cy"]["mean"],
        "inset_m": combined["inset"]["mean"], "marker_edge_m": combined["medge"]["mean"],
        "cross_session_std": {k: combined[k]["std"] for k in PARAM_NAMES},
        "fit_stderr_best_session": best["primary"]["stderr"],
        "residual_mm": {"rms": best["primary"]["residual_rms_mm"], "max": best["primary"]["residual_max_mm"]},
        "best_session": best["session"],
        "identifiability": {
            "well_determined": ["fx", "cx"],
            "weakly_determined": ["fy", "cy", "inset", "medge"],
            "note": (
                "The free 6-parameter fit is only weakly conditioned in fy/cy and in the scale pair "
                "(inset, medge): inset saturates its [0,10]cm bound in every session/serial tried here, "
                "and its Jacobian-based standard error (see fit_stderr_best_session) exceeds the bound's "
                "own width -- i.e. a single view of one coplanar rectangle cannot fully separate focal "
                "length from the assumed physical size of that rectangle, a known limitation of "
                "single-viewpoint planar self-calibration. fx and cx are comparatively stable: cross-"
                "session std "
                f"{combined['fx']['std']:.2f} px (fx) vs {combined['cy']['std']:.2f} px (cy), and forcing "
                "cx,cy to the image centre changes fx by "
                f"{best['sensitivity']['cx_cy_fixed_centre']['params']['fx'] - best['primary']['params']['fx']:+.1f} px "
                "while fy comes out close to fx there (fx="
                f"{best['sensitivity']['cx_cy_fixed_centre']['params']['fx']:.1f}, fy="
                f"{best['sensitivity']['cx_cy_fixed_centre']['params']['fy']:.1f} vs {best['primary']['params']['fy']:.1f} "
                "free) -- a physically-plausible square-pixel cross-check the free fit does not itself "
                "impose. Sweeping inset over its allowed range moves fx from "
                f"{inset_sweep.get('inset_fixed_0.00', float('nan')):.1f} px (inset=0) to "
                f"{inset_sweep.get('inset_fixed_0.10', float('nan')):.1f} px (inset=10cm) with residual RMS "
                "changing by well under 1 mm across that whole range -- the data cannot discriminate "
                "between those inset values, only an independently-measured marker inset or edge length "
                "would. The plane-fit cross-check (below) is independent evidence that the *directional* "
                "geometry (table normal, camera height) is nonetheless solid."
            ),
        },
        "assumptions": [
            "pinhole model, zero lens distortion (no radial/tangential terms fit or removed)",
            "square-pixel/equal-fx-fy and cx,cy=image-centre are only the optimizer's initial guess; "
            "fx, fy, cx, cy are all fit as free parameters in the primary result",
            "marker corners assumed exactly coplanar with the table surface (flat printed markers)",
            "a single 'inset' parameter is assumed for all four marker-to-table-edge margins "
            "(net, end-line, and both side edges), bounded to [0, 10] cm",
            "aligned depth (session.json align_depth_to_color=true) is trusted as exact per-pixel "
            "colour<->depth correspondence -- no independent check of that alignment exists",
            "table dimensions taken from the ITTF regulation, not measured on this table: "
            "2.74 m x 1.525 m, net 0.1525 m",
            "table_frame 'up' is forced to point toward the camera (see table_frame_transform): "
            "Kabsch-aligning two coplanar point sets leaves the fitted normal's sign numerically "
            "arbitrary, and it came out flipped for one of the two serials before this correction",
        ],
        "table_frame": {
            "origin": "table surface at the centre of the net",
            "x": "along table length, positive toward this camera's end line",
            "y": "across table width",
            "z": "up, out of the table surface",
            "camera_to_table_4x4": M.tolist(),
            "table_to_camera_4x4": np.linalg.inv(M).tolist(),
            "note": f"fit from session {best['session']} (most marker decodes)",
        },
        "per_session": [
            {"session": r["session"], "cam": r["cam"], "n_frames_seen": r["n_frames_seen"],
             "decoded_frames": r["decoded"], "params": r["primary"]["params"],
             "residual_rms_mm": r["primary"]["residual_rms_mm"], "residual_max_mm": r["primary"]["residual_max_mm"],
             "angle_error_deg": r["primary"]["angle_error_deg"], "coplanarity_mm": r["primary"]["coplanarity_mm"],
             "sensitivity": {k: v["params"] for k, v in r["sensitivity"].items()},
             "plane_fit": r["plane_fit"]}
            for r in runs
        ],
    }


def write_report(all_results, args):
    lines = ["# Intrinsics self-calibration report", "",
             "Method: `sim/estimate_intrinsics.py` (see its docstring). No factory intrinsics exist "
             "anywhere in the capture (docs/PROJECT_KNOWLEDGE.md #6a); these are fit from the ArUco "
             "table markers (`DICT_4X4_50`) plus aligned depth, using the ITTF table as the metric "
             "reference.", "",
             f"Sessions requested: {', '.join(args.sessions)}. Depth window: "
             f"{2 * args.depth_window + 1}x{2 * args.depth_window + 1} px, median of up to "
             f"{args.depth_frames} frames per point, zeros ignored.", ""]
    for serial, s in all_results.items():
        lines.append(f"## Serial {serial}")
        lines.append("")
        lines.append(f"Combined (mean across {len(s['per_session'])} session(s)):")
        lines.append("")
        lines.append("| param | mean | cross-session std | fit std-err (best session) |")
        lines.append("|---|---|---|---|")
        for name, unit, scale in [("fx", "px", 1), ("fy", "px", 1), ("cx", "px", 1), ("cy", "px", 1),
                                   ("inset", "cm", 100), ("medge", "cm", 100)]:
            key = {"inset": "inset_m", "medge": "marker_edge_m"}.get(name, name)
            mean = s[key]
            std = s["cross_session_std"][name]
            se = s["fit_stderr_best_session"].get(name)
            se_s = f"{se * scale:.2f}" if se is not None else "n/a (fixed)"
            lines.append(f"| {name} | {mean * scale:.2f} {unit} | {std * scale:.2f} {unit} | {se_s} {unit} |")
        lines.append("")
        lines.append(f"Residual RMS (best session, {s['best_session']}): {s['residual_mm']['rms']:.2f} mm "
                      f"(max {s['residual_mm']['max']:.2f} mm).")
        lines.append("")
        lines.append(f"**Well determined**: {', '.join(s['identifiability']['well_determined'])}. "
                      f"**Weakly determined**: {', '.join(s['identifiability']['weakly_determined'])}.")
        lines.append("")
        lines.append(s["identifiability"]["note"])
        lines.append("")
        heights = [ps["plane_fit"]["camera_height_m"] for ps in s["per_session"]]
        angles = [ps["plane_fit"]["normal_vs_table_edges_deg"] for ps in s["per_session"]]
        lines.append(f"Independent plane-fit cross-check across sessions: camera height "
                      f"{min(heights):.3f}-{max(heights):.3f} m above the table, plane normal vs. the "
                      f"fitted table edges {min(angles):.2f}-{max(angles):.2f} deg apart.")
        lines.append("")
        lines.append("### Per-session repeatability")
        lines.append("")
        lines.append("| session | frames seen | decoded (near_net/far_net/near_end/far_end) | fx | fy | cx | cy "
                      "| inset cm | medge cm | resid rms mm | angle err deg | coplanarity mm | plane rms mm | "
                      "cam height m | plane-vs-edge deg |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for ps in s["per_session"]:
            p, d = ps["params"], ps["decoded_frames"]
            pf = ps["plane_fit"]
            lines.append(
                f"| {ps['session']} | {ps['n_frames_seen']} | "
                f"{d['near_net']}/{d['far_net']}/{d['near_end']}/{d['far_end']} | "
                f"{p['fx']:.1f} | {p['fy']:.1f} | {p['cx']:.1f} | {p['cy']:.1f} | "
                f"{p['inset'] * 100:.2f} | {p['medge'] * 100:.2f} | {ps['residual_rms_mm']:.2f} | "
                f"{', '.join(f'{a:.2f}' for a in ps['angle_error_deg'])} | {ps['coplanarity_mm']:.2f} | "
                f"{pf['rms_mm']:.2f} | {pf['camera_height_m']:.3f} | {pf['normal_vs_table_edges_deg']:.2f} |")
        lines.append("")
        lines.append("### Sensitivity (best session, " + s["best_session"] + ")")
        lines.append("")
        lines.append("| variant | fx | fy | cx | cy | inset cm | medge cm |")
        lines.append("|---|---|---|---|---|---|---|")
        best_ps = next(ps for ps in s["per_session"] if ps["session"] == s["best_session"])
        base = best_ps["params"]
        lines.append(f"| free (primary) | {base['fx']:.1f} | {base['fy']:.1f} | {base['cx']:.1f} | "
                      f"{base['cy']:.1f} | {base['inset'] * 100:.2f} | {base['medge'] * 100:.2f} |")
        for name, sv in best_ps["sensitivity"].items():
            lines.append(f"| {name} | {sv['fx']:.1f} | {sv['fy']:.1f} | {sv['cx']:.1f} | {sv['cy']:.1f} | "
                          f"{sv['inset'] * 100:.2f} | {sv['medge'] * 100:.2f} |")
        lines.append("")
        lines.append(f"Table frame: `work/calib/intrinsics_{serial}.json` -> `table_frame` "
                      f"(4x4 camera<->table transform, {s['table_frame']['note']}).")
        lines.append("")
    out_path = os.path.join(args.out_dir, "REPORT.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
