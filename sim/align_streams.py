"""Time-align the three HIM table-tennis capture streams (paddle IMU, Orbbec RGB-D, Rokoko mocap).

Three independent clocks:
  * paddle IMU  - WitMotion iOS export, host wall-clock with NO timezone (HIM inferred America/New_York)
  * Orbbec      - host epoch seconds (frames.jsonl host_time_s, session.json host_time_start_s)
  * Rokoko FBX  - no per-frame absolute time; only an export CreationTimeStamp in the FBX header

What this script does (everything is derived from files on disk, nothing is assumed):
  1. session map   : pair IMU episode <-> Orbbec session <-> Rokoko clip by name/start time and duration.
  2. IMU <-> mocap  : cross-correlate IMU |gyro| against RightHand world speed from the BVH (FK),
                     both resampled to a common rate, over a +/- lag window; plus swing-peak matching.
  3. IMU <-> Orbbec : compare IMU episode start (UTC as inferred by HIM) with Orbbec host_time_start_s;
                     the 3 h discrepancy is a timezone-offset question, reported from the data.
  4. FBX headers   : parse the binary FBX node tree for CreationTimeStamp / take name / take length.

Usage (defaults match the local data layout):
  ~/him-tools/mujoco-venv/bin/python sim/align_streams.py --out work/alignment
  ... --orbbec-motion      # also build a frame-difference motion signal from the Orbbec color
                           # frames and cross-correlate it with the IMU (reads every JPG, ~1-2 min)
  ... --bvh-dir DIR         # Blender BVH exports of the Rokoko clips (Clip*.bvh, *_nofingers.bvh skipped);
                           # default ~/src/him-table-tennis-training/work/bvh

Outputs (under --out): session-map.json, fbx-headers.json, imu-mocap-xcorr.csv,
imu-orbbec-clock.json, xcorr_<episode>_<clip>.png, xcorr-matrix.png.
Needs numpy, scipy, matplotlib (pip-installed into ~/him-tools/mujoco-venv).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
DEFAULT_IMU = HOME / "him-data/him-table-tennis-paddle-imu-20260815"
DEFAULT_ORBBEC = [HOME / "him-data/orbbec_20260815_214619", HOME / "him-data/orbbec_20260815_215350"]
DEFAULT_FBX = HOME / "him-data/rokoko-motion-capture"
DEFAULT_BVH = HOME / "src/him-table-tennis-training/work/bvh"
FBX_KTIME_PER_S = 46186158000  # FBX SDK time unit (KTime ticks per second)
# FBX SDK FbxTime::EMode enum values -> frames per second (eCustom = 14 reads GlobalSettings CustomFrameRate).
FBX_TIME_MODES = {1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0, 5: 48.0, 6: 30.0, 7: 30.0, 8: 29.97, 9: 29.97, 10: 25.0,
                  11: 24.0, 12: 1000.0, 13: 23.976, 15: 96.0, 16: 72.0, 17: 59.94}  # eFrames120 .. eFrames59dot94; 6 = eFrames30, 7 = eFrames30Drop, 8 = eNTSCDropFrame
FBX_TIME_MODE_CUSTOM = 14  # eCustom


# ----------------------------------------------------------------------------- IMU
def load_imu(imu_dir: Path) -> dict:
    """Return {episode_id: {t_utc: epoch s (as inferred by HIM), t_local: naive local s, gyro: |w| dps}}."""
    csv_path = imu_dir / "normalized/paddle-imu-samples.csv"
    eps: dict[str, dict] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            e = eps.setdefault(row["episode_id"], {"t_utc": [], "t_local": [], "gyro": [], "acc": []})
            e["t_utc"].append(dt.datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00")).timestamp())
            e["t_local"].append(dt.datetime.fromisoformat(row["timestamp_local"]).replace(tzinfo=dt.timezone.utc).timestamp())
            w = [float(row[f"angular_velocity_{a}_dps"]) for a in "xyz"]
            a = [float(row[f"accel_{a}_g"]) for a in "xyz"]
            e["gyro"].append(float(np.linalg.norm(w)))
            e["acc"].append(float(np.linalg.norm(a)))
    for e in eps.values():
        for k in list(e):
            e[k] = np.asarray(e[k])
        e["duration"] = float(e["t_utc"][-1] - e["t_utc"][0])
        e["start_utc"] = float(e["t_utc"][0])
        e["start_local_naive"] = float(e["t_local"][0])
    meta = json.load(open(imu_dir / "metadata/episodes.json"))
    for m in meta["episodes"]:
        if m["episode_id"] in eps:
            eps[m["episode_id"]]["meta"] = m
    return eps


# ----------------------------------------------------------------------------- BVH
def parse_bvh(path: Path):
    txt = open(path).read()
    hi = txt.index("MOTION")
    hier = txt[:hi].split()
    mot = txt[hi:].splitlines()
    joints: list[dict] = []
    stack: list[int] = []
    i = 0
    chan_idx = 0
    while i < len(hier):
        tok = hier[i]
        if tok in ("ROOT", "JOINT"):
            j = {"name": hier[i + 1], "parent": stack[-1] if stack else None, "channels": []}
            joints.append(j)
            stack.append(len(joints) - 1)
            i += 2
        elif tok == "End":
            joints.append({"name": joints[stack[-1]]["name"] + "_end", "parent": stack[-1], "channels": []})
            stack.append(len(joints) - 1)
            i += 2
        elif tok == "OFFSET":
            joints[stack[-1]]["offset"] = np.array([float(v) for v in hier[i + 1:i + 4]])
            i += 4
        elif tok == "CHANNELS":
            n = int(hier[i + 1])
            joints[stack[-1]]["channels"] = hier[i + 2:i + 2 + n]
            joints[stack[-1]]["chan_start"] = chan_idx
            chan_idx += n
            i += 2 + n
        elif tok == "}":
            stack.pop()
            i += 1
        else:
            i += 1
    nframes = int(mot[1].split()[-1])
    frame_time = float(mot[2].split()[-1])
    data = np.array([[float(v) for v in line.split()] for line in mot[3:3 + nframes]])
    return joints, frame_time, data


def _rot(axis: str, a: np.ndarray) -> np.ndarray:
    """Batch rotation matrices (N,3,3) about a world axis, right-handed."""
    c, s = np.cos(a), np.sin(a)
    n = a.shape[0]
    R = np.zeros((n, 3, 3))
    if axis == "X":
        R[:, 0, 0] = 1; R[:, 1, 1] = c; R[:, 1, 2] = -s; R[:, 2, 1] = s; R[:, 2, 2] = c
    elif axis == "Y":
        R[:, 1, 1] = 1; R[:, 0, 0] = c; R[:, 0, 2] = s; R[:, 2, 0] = -s; R[:, 2, 2] = c
    else:
        R[:, 2, 2] = 1; R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c
    return R


def bvh_world_transform(joints, data, target: str) -> tuple[np.ndarray, np.ndarray]:
    """World position (N,3) and rotation (N,3,3) of `target` via FK along its chain. Standard BVH
    semantics: local = T(pos) @ R(ch1) @ R(ch2) @ R(ch3). Blender's exporter (rotate_mode=NATIVE,
    root_transform_only=False) writes 6 channels per bone whose position channels already include
    the rest offset, so when position channels exist they REPLACE the OFFSET. Verified against
    Blender's own pose_bone.head on Clip.fbx (work/alignment/verify): Hips exact, RightHand within 1.2 cm mean."""
    names = [j["name"] for j in joints]
    k = names.index(target)
    chain = []
    while k is not None:
        chain.append(k)
        k = joints[k]["parent"]
    chain = chain[::-1]
    n = data.shape[0]
    R = np.tile(np.eye(3), (n, 1, 1))
    p = np.zeros((n, 3))
    for k in chain:
        j = joints[k]
        ch = j["channels"]
        s = j.get("chan_start", 0)
        trans = np.tile(j["offset"], (n, 1))
        Rl = np.tile(np.eye(3), (n, 1, 1))
        for ci, c in enumerate(ch):
            v = data[:, s + ci]
            if c == "Xposition":
                trans[:, 0] = v
            elif c == "Yposition":
                trans[:, 1] = v
            elif c == "Zposition":
                trans[:, 2] = v
            elif c.endswith("rotation"):
                Rl = Rl @ _rot(c[0], np.radians(v))
        p = p + np.einsum("nij,nj->ni", R, trans)
        R = R @ Rl
    return p, R


def bvh_world_position(joints, data, target: str) -> np.ndarray:
    return bvh_world_transform(joints, data, target)[0]


def load_bvh_hand_speed(bvh_dir: Path, bone: str = "RightHand") -> dict:
    """{clip: {fps, duration, t, speed (m/s), ang_speed (deg/s), pos (m)}}. Frame 0 of every clip is a
    T-pose (all rotation channels 0 — Rokoko exports it, Blender keeps it), so it is dropped before
    differentiating. ang_speed is the hand's world angular-speed magnitude, which for a paddle held
    rigidly in the hand is the same physical quantity as the paddle IMU's |gyro|."""
    clips = {}
    for path in sorted(bvh_dir.glob("Clip*.bvh")):
        if path.name.endswith("_nofingers.bvh"):
            continue
        joints, ft, data = parse_bvh(path)
        rot_cols = [j["chan_start"] + ci for j in joints for ci, c in enumerate(j.get("channels", [])) if c.endswith("rotation")]
        rest_frame = bool(np.abs(data[0, rot_cols]).max() < 0.01)  # every rotation channel ~0 deg
        pos, R = bvh_world_transform(joints, data, bone)
        pos = pos / 100.0  # cm -> m (Blender export, cm)
        if rest_frame:
            pos, R = pos[1:], R[1:]
        t = np.arange(pos.shape[0]) * ft
        speed = np.linalg.norm(np.gradient(pos, ft, axis=0), axis=1)
        dR = np.einsum("nji,njk->nik", R[:-1], R[1:])  # R_t^T R_{t+1}
        ang = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1))) / ft
        ang = np.concatenate([ang[:1], ang])
        clips[path.stem] = {"fps": 1.0 / ft, "frames": int(data.shape[0]), "duration": float(data.shape[0] * ft),
                            "t": t, "speed": speed, "ang_speed": ang, "pos": pos, "rest_frame_dropped": rest_frame}
    return clips


# ----------------------------------------------------------------------------- FBX header
def _fbx_read_node(b: bytes, pos: int, v64: bool):
    if v64:
        end, nprops, plen = struct.unpack_from("<QQQ", b, pos); pos += 24
    else:
        end, nprops, plen = struct.unpack_from("<III", b, pos); pos += 12
    nlen = b[pos]; pos += 1
    name = b[pos:pos + nlen].decode("latin1"); pos += nlen
    if end == 0:
        return None, pos
    props = []
    pend = pos + plen
    while pos < pend:
        t = chr(b[pos]); pos += 1
        if t in "CYIFDL":
            fmt = {"C": "<?", "Y": "<h", "I": "<i", "F": "<f", "D": "<d", "L": "<q"}[t]
            props.append(struct.unpack_from(fmt, b, pos)[0]); pos += struct.calcsize(fmt)
        elif t in "SR":
            n = struct.unpack_from("<I", b, pos)[0]; pos += 4
            s = b[pos:pos + n]; pos += n
            props.append(s.decode("latin1", "replace") if t == "S" else f"<raw {n}B>")
        else:  # array property: skip its payload
            n, enc, clen = struct.unpack_from("<III", b, pos); pos += 12
            props.append(f"<array {t} n={n}>"); pos += clen
    children = []
    while pos < end:
        c, pos = _fbx_read_node(b, pos, v64)
        if c is None:
            break
        children.append(c)
    return (name, props, children), end


def parse_fbx_header(path: Path) -> dict:
    """Header / scene-info / take facts from a binary FBX. Only the top-level nodes we need are walked."""
    b = open(path, "rb").read()
    if b[:20] != b"Kaydara FBX Binary  ":
        return {"error": "not a binary FBX"}
    ver = struct.unpack_from("<I", b, 23)[0]
    v64 = ver >= 7500
    pos = 27
    out: dict = {"fbx_version": ver, "file": str(path), "size_bytes": len(b)}
    while pos < len(b):
        node, pos = _fbx_read_node(b, pos, v64)
        if node is None:
            break
        name, props, ch = node
        if name == "FBXHeaderExtension":
            for c in ch:
                if c[0] == "CreationTimeStamp":
                    ts = {cc[0]: cc[1][0] for cc in c[2]}
                    out["creation_timestamp"] = "%04d-%02d-%02dT%02d:%02d:%02d.%03d" % (
                        ts["Year"], ts["Month"], ts["Day"], ts["Hour"], ts["Minute"], ts["Second"], ts["Millisecond"])
                elif c[0] == "Creator":
                    out["creator"] = c[1][0]
                elif c[0] == "SceneInfo":
                    for cc in c[2]:
                        if cc[0] == "Properties70":
                            for p in cc[2]:
                                if p[1][0] in ("Original|DateTime_GMT", "LastSaved|DateTime_GMT", "Original|ApplicationName",
                                               "Original|ApplicationVersion", "Original|FileName"):
                                    out[p[1][0]] = p[1][4] if len(p[1]) > 4 else ""
        elif name == "Creator":
            out["creator_node"] = props[0] if props else ""
        elif name == "GlobalSettings":
            for c in ch:
                if c[0] == "Properties70":
                    for p in c[2]:
                        if p[1][0] in ("TimeMode", "TimeSpanStart", "TimeSpanStop", "CustomFrameRate", "UnitScaleFactor", "UpAxis"):
                            out["global_" + p[1][0]] = p[1][4]
        elif name == "Objects":
            out["anim_stacks"] = [c[1][1].split("\x00")[0] for c in ch if c[0] == "AnimationStack"]
        elif name == "Takes":
            for c in ch:
                if c[0] == "Take":
                    take = {"name": c[1][0]}
                    for cc in c[2]:
                        if cc[0] == "FileName":
                            take["file_name"] = cc[1][0]
                        elif cc[0] in ("LocalTime", "ReferenceTime"):
                            take[cc[0]] = [cc[1][0], cc[1][1]]
                            take[cc[0] + "_s"] = [cc[1][0] / FBX_KTIME_PER_S, cc[1][1] / FBX_KTIME_PER_S]
                    out.setdefault("takes", []).append(take)
    if "global_TimeMode" in out:
        mode = out["global_TimeMode"]
        out["global_fps"] = out.get("global_CustomFrameRate") if mode == FBX_TIME_MODE_CUSTOM else FBX_TIME_MODES.get(mode)
    # Export path strings (Rokoko writes the export folder into the file); keep only the leaf name.
    for key in (b"RokokoStudio/Exports/",):
        i = b.find(key)
        if i >= 0:
            m = re.match(rb"[\x20-\x7e]+", b[i + len(key):i + len(key) + 200])
            out["rokoko_export_folder_leaf"] = m.group(0).decode("ascii").split("/")[0] if m else ""
    return out


# ----------------------------------------------------------------------------- Orbbec
def load_orbbec(session_dir: Path) -> dict:
    sess = json.load(open(session_dir / "session.json"))
    cams = {}
    for cam_dir in sorted(session_dir.glob("cam*")):
        rows = [json.loads(line) for line in open(cam_dir / "frames.jsonl")]
        ht = np.array([r["host_time_s"] for r in rows])
        cams[cam_dir.name] = {"n_frames": len(rows), "first_host_time_s": float(ht[0]), "last_host_time_s": float(ht[-1]),
                              "n_color": sum(1 for r in rows if r.get("color_file")),
                              "median_dt_s": float(np.median(np.diff(ht))) if len(ht) > 1 else None,
                              "host_time_s": ht, "rows": rows}
    name = session_dir.name  # orbbec_YYYYMMDD_HHMMSS[_60fps]
    stamp = name.split("_")[1] + name.split("_")[2]
    return {"dir": str(session_dir), "name": name, "launch_local_naive": dt.datetime.strptime(stamp, "%Y%m%d%H%M%S"),
            "session": sess, "cams": cams}


def orbbec_motion_stats(sess: dict, cam: str, cache_dir: Path) -> np.ndarray:
    """Per-frame frame-difference statistics of the color stream (gray, 4x down-sampled):
    columns = host_time_s, mean|d|, p95|d|, p99|d|, max|d|, fraction(|d|>25). Cached as .npy."""
    cache = cache_dir / f"orbbec-motion-{sess['name']}-{cam}.npy"
    if cache.is_file():
        return np.load(cache)
    import imageio.v2 as iio
    rows = sess["cams"][cam]["rows"]
    base = Path(sess["dir"]) / cam
    prev = None
    out = []
    for r in rows:
        if not r.get("color_file"):
            continue
        g = iio.imread(base / r["color_file"])[::4, ::4].astype(np.float32).mean(axis=2)
        if prev is not None:
            d = np.abs(g - prev)
            out.append([r["host_time_s"], d.mean(), np.percentile(d, 95), np.percentile(d, 99), d.max(), (d > 25).mean()])
        prev = g
    arr = np.array(out)
    np.save(cache, arr)
    return arr


def orbbec_motion_signal(sess: dict, cam: str, rate: float, cache_dir: Path, column: int = 1) -> tuple[np.ndarray, np.ndarray]:
    m = orbbec_motion_stats(sess, cam, cache_dir)
    ts, en = m[:, 0], m[:, column]
    grid = np.arange(ts[0], ts[-1], 1.0 / rate)
    return grid, np.interp(grid, ts, en)


# ----------------------------------------------------------------------------- alignment maths
def resample(t: np.ndarray, x: np.ndarray, rate: float, smooth_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(0.0, t[-1] - t[0], 1.0 / rate)
    y = np.interp(grid, t - t[0], x)
    if smooth_s > 0:
        from scipy.ndimage import gaussian_filter1d
        y = gaussian_filter1d(y, smooth_s * rate)
    return grid, y


def normalise(x: np.ndarray, clip_sigma: float = 4.0) -> np.ndarray:
    x = x - np.median(x)
    s = x.std() or 1.0
    return np.clip(x / s, -clip_sigma, clip_sigma)


def xcorr_lags(a: np.ndarray, b: np.ndarray, rate: float, max_lag_s: float, min_overlap_s: float = 15.0):
    """Pearson correlation of a(t) with b(t - lag) for lag in [-max_lag, max_lag].
    Positive lag = the event appears LATER in `a` than in `b` (a lags b).
    Returns lags (s), corr (nan where the overlap is too short)."""
    max_lag = int(round(max_lag_s * rate))
    min_ov = int(min_overlap_s * rate)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.full(lags.shape, np.nan)
    for i, L in enumerate(lags):
        if L >= 0:
            x, y = a[L:], b[:len(b)]
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
        else:
            x, y = a[:len(a)], b[-L:]
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
        if n < min_ov:
            continue
        xs, ys = x - x.mean(), y - y.mean()
        d = np.sqrt((xs ** 2).sum() * (ys ** 2).sum())
        out[i] = float((xs * ys).sum() / d) if d > 0 else np.nan
    return lags / rate, out


def peak_stats(lags: np.ndarray, corr: np.ndarray, exclusion_s: float = 5.0) -> dict:
    from scipy.signal import find_peaks
    if not np.isfinite(corr).any():
        return {"lag_s": None, "peak_corr": float("nan"), "lobe_width_s": None, "second_lag_s": None, "second_corr": None,
                "secondary_ratio": None, "peak_z": float("nan"), "lag_step_s": float(lags[1] - lags[0])}
    c = np.where(np.isnan(corr), -np.inf, corr)
    best = int(np.argmax(c))
    pk, _ = find_peaks(c)
    dt_ = lags[1] - lags[0]
    others = [p for p in pk if abs(lags[p] - lags[best]) > exclusion_s]
    second = max(others, key=lambda p: c[p]) if others else None
    valid = corr[np.isfinite(corr)]
    z = (c[best] - valid.mean()) / (valid.std() or 1.0)
    # width of the main lobe: contiguous lags around the best whose r >= 0.8 * peak
    lo = hi = best
    while lo > 0 and c[lo - 1] >= 0.8 * c[best]:
        lo -= 1
    while hi < len(c) - 1 and c[hi + 1] >= 0.8 * c[best]:
        hi += 1
    return {"lag_s": float(lags[best]), "peak_corr": float(c[best]), "lobe_width_s": float(lags[hi] - lags[lo]),
            "second_lag_s": float(lags[second]) if second is not None else None,
            "second_corr": float(c[second]) if second is not None else None,
            "secondary_ratio": float(c[second] / c[best]) if second is not None and c[best] > 0 else None,
            "peak_z": float(z), "lag_step_s": float(dt_)}


def swing_peaks(t: np.ndarray, x: np.ndarray, height: float, distance_s: float = 0.4) -> np.ndarray:
    from scipy.signal import find_peaks
    dt_ = t[1] - t[0]
    pk, _ = find_peaks(x, height=height, distance=max(1, int(distance_s / dt_)))
    return t[pk]


def peak_match_lag(ta: np.ndarray, tb: np.ndarray, max_lag_s: float, bin_s: float = 0.2, tol_s: float = 0.3) -> dict:
    """Mode of pairwise (ta_i - tb_j) within +/-max_lag; matched fraction at that lag."""
    if len(ta) == 0 or len(tb) == 0:
        return {"n_peaks_a": int(len(ta)), "n_peaks_b": int(len(tb)), "lag_s": None, "matched_frac": None}
    d = (ta[:, None] - tb[None, :]).ravel()
    d = d[np.abs(d) <= max_lag_s]
    if d.size == 0:
        return {"n_peaks_a": int(len(ta)), "n_peaks_b": int(len(tb)), "lag_s": None, "matched_frac": None}
    edges = np.arange(-max_lag_s, max_lag_s + bin_s, bin_s)
    h, _ = np.histogram(d, edges)
    k = int(np.argmax(h))
    sel = d[(d >= edges[k]) & (d < edges[k + 1])]
    lag = float(np.median(sel))
    matched = sum(1 for x in ta if np.any(np.abs((x - lag) - tb) <= tol_s))
    return {"n_peaks_a": int(len(ta)), "n_peaks_b": int(len(tb)), "lag_s": lag,
            "votes": int(h[k]), "matched_frac": float(matched / len(ta))}


# ----------------------------------------------------------------------------- main
def wall_clock_iso(epoch_naive: float) -> str:
    return dt.datetime.fromtimestamp(epoch_naive, dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="milliseconds")


def orbbec_clock_records(orb: dict, imu: dict, extra_tars: list[Path]) -> list[dict]:
    """Per Orbbec session: host epoch, directory-name wall clock, implied host UTC offset and the IMU
    episodes whose wall-clock window overlaps it. Failed (0-byte) tars contribute name-only records."""
    rows = []
    for name, s in orb.items():
        hs = s["session"]["host_time_start_s"]
        launch = s["launch_local_naive"]
        utc_naive = dt.datetime.fromtimestamp(hs, dt.timezone.utc).replace(tzinfo=None)
        offset_h = round((launch - utc_naive).total_seconds() / 3600.0)
        wall_start = hs + 3600 * offset_h  # host wall clock, expressed as a naive epoch
        cam0 = next(iter(s["cams"].values()))
        first_wall, last_wall = cam0["first_host_time_s"] + 3600 * offset_h, cam0["last_host_time_s"] + 3600 * offset_h
        rec = {"orbbec_session": name, "status": "on disk", "host_time_start_s": hs, "created_utc": s["session"]["created_utc"],
               "host_time_start_utc": dt.datetime.fromtimestamp(hs, dt.timezone.utc).isoformat(),
               "dir_name_wall_clock": launch.isoformat(), "implied_host_utc_offset_h": offset_h,
               "host_start_wall_clock": wall_clock_iso(wall_start), "launch_to_host_start_s": round(wall_start - launch.replace(tzinfo=dt.timezone.utc).timestamp(), 3),
               "first_frame_wall_clock": wall_clock_iso(first_wall), "last_frame_wall_clock": wall_clock_iso(last_wall),
               "n_frames": cam0["n_frames"], "n_color_frames": cam0["n_color"], "median_frame_dt_s": cam0["median_dt_s"],
               "duration_s": round(cam0["last_host_time_s"] - cam0["first_host_time_s"], 3), "imu_candidates": []}
        for ep, e in imu.items():
            imu_start, imu_end = e["start_local_naive"], e["start_local_naive"] + e["duration"]
            overlap = min(last_wall, imu_end) - max(first_wall, imu_start)
            if overlap <= -300:
                continue
            rec["imu_candidates"].append({
                "episode": ep, "imu_start_wall_clock": wall_clock_iso(imu_start), "imu_duration_s": round(e["duration"], 3),
                "imu_start_utc_inferred_ny": dt.datetime.fromtimestamp(e["start_utc"], dt.timezone.utc).isoformat(),
                "orbbec_minus_imu_start_s_ny_inference": round(hs - e["start_utc"], 3),
                "orbbec_minus_imu_start_s_same_wall_clock": round(wall_start - imu_start, 3),
                "first_frame_minus_imu_start_s_same_wall_clock": round(first_wall - imu_start, 3),
                "last_frame_minus_imu_end_s_same_wall_clock": round(last_wall - imu_end, 3),
                "wall_clock_overlap_s": round(overlap, 3),
                "orbbec_inside_imu_episode_same_wall_clock": bool(first_wall >= imu_start and last_wall <= imu_end)})
        rows.append(rec)
    for tar in extra_tars:
        name = tar.stem
        if name in orb or not name.startswith("orbbec_"):
            continue
        stamp = name.split("_")[1] + name.split("_")[2]
        launch = dt.datetime.strptime(stamp, "%Y%m%d%H%M%S")
        rec = {"orbbec_session": name, "status": f"tar on disk is {tar.stat().st_size} bytes (failed download); name only",
               "dir_name_wall_clock": launch.isoformat(), "imu_candidates": []}
        for ep, e in imu.items():
            d = e["start_local_naive"] - launch.replace(tzinfo=dt.timezone.utc).timestamp()
            if -60 <= d <= 60:
                rec["imu_candidates"].append({"episode": ep, "imu_start_wall_clock": wall_clock_iso(e["start_local_naive"]),
                                              "imu_start_minus_orbbec_launch_s": round(d, 3)})
        rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--imu-dir", type=Path, default=DEFAULT_IMU)
    ap.add_argument("--orbbec-dir", type=Path, nargs="*", default=DEFAULT_ORBBEC)
    ap.add_argument("--orbbec-tars", type=Path, nargs="*", default=sorted((HOME / "Downloads").glob("orbbec_*.tar")),
                    help="tar files whose names give the launch wall clock of sessions not extracted (0-byte failed downloads)")
    ap.add_argument("--fbx-dir", type=Path, default=DEFAULT_FBX)
    ap.add_argument("--bvh-dir", type=Path, default=DEFAULT_BVH)
    ap.add_argument("--out", type=Path, default=Path("work/alignment"))
    ap.add_argument("--rate", type=float, default=10.0, help="common resample rate, Hz")
    ap.add_argument("--max-lag", type=float, default=60.0, help="lag window, +/- s")
    ap.add_argument("--smooth", type=float, default=0.4, help="gaussian smoothing sigma (s) after resampling")
    ap.add_argument("--feature", choices=("ang", "lin"), default="ang",
                    help="mocap feature ranked against IMU |gyro|: hand angular speed (ang, same physical quantity) or linear speed (lin)")
    ap.add_argument("--duration-tol", type=float, default=0.15, help="relative duration tolerance for 'similar' pairs")
    ap.add_argument("--peak-dps", type=float, default=400.0, help="swing-peak threshold on |gyro| and hand angular speed (dps)")
    ap.add_argument("--peak-smooth", type=float, default=0.1, help="gaussian sigma (s) used only for swing-peak detection")
    ap.add_argument("--motion-column", type=int, default=1, help="Orbbec motion statistic column (1 mean, 2 p95, 3 p99, 4 max, 5 frac>25)")
    ap.add_argument("--orbbec-motion", action="store_true", help="also cross-correlate Orbbec color motion energy with the IMU")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rate = args.rate

    # ---- load
    imu = load_imu(args.imu_dir)
    clips = load_bvh_hand_speed(args.bvh_dir)
    fbx = {p.stem: parse_fbx_header(p) for p in sorted(args.fbx_dir.glob("*.fbx"))}
    orb = {d.name: load_orbbec(d) for d in args.orbbec_dir if (d / "session.json").is_file()}
    json.dump(fbx, open(out / "fbx-headers.json", "w"), indent=1, sort_keys=True)
    print(f"[load] {len(imu)} IMU episodes, {len(clips)} BVH clips, {len(fbx)} FBX headers, {len(orb)} Orbbec sessions")
    for c, v in clips.items():
        print(f"[bvh] {c}: {v['frames']} frames @ {v['fps']:.3f} fps = {v['duration']:.1f}s; hand speed max {v['speed'].max():.1f} m/s "
              f"p99 {np.percentile(v['speed'], 99):.2f}; hand ang speed max {v['ang_speed'].max():.0f} dps p99 {np.percentile(v['ang_speed'], 99):.0f}; "
              f"T-pose frame dropped={v['rest_frame_dropped']}")
    for ep, e in imu.items():
        print(f"[imu] {ep}: {len(e['gyro'])} samples, {e['duration']:.1f}s, |gyro| max {e['gyro'].max():.0f} dps p99 {np.percentile(e['gyro'], 99):.0f}")

    # ---- 3. IMU <-> Orbbec clocks
    clock_rows = orbbec_clock_records(orb, imu, list(args.orbbec_tars))
    for rec in clock_rows:
        if rec["status"] != "on disk":
            print(f"[clock] {rec['orbbec_session']}: {rec['status']}; IMU episodes starting within 60 s of the launch stamp: "
                  f"{[(c['episode'][-6:], c['imu_start_minus_orbbec_launch_s']) for c in rec['imu_candidates']]}")
            continue
        print(f"[clock] {rec['orbbec_session']}: host epoch -> UTC {rec['host_time_start_utc']}, dir-name wall clock {rec['dir_name_wall_clock']} "
              f"=> host UTC offset {rec['implied_host_utc_offset_h']:+d} h; frames {rec['first_frame_wall_clock']} .. {rec['last_frame_wall_clock']} (wall)")
        for c in rec["imu_candidates"]:
            print(f"        vs IMU {c['episode'][-6:]} ({c['imu_start_wall_clock']} wall, {c['imu_duration_s']}s): Orbbec-IMU start = "
                  f"{c['orbbec_minus_imu_start_s_ny_inference']:+.1f}s with HIM's NY inference, {c['orbbec_minus_imu_start_s_same_wall_clock']:+.1f}s if "
                  f"the hosts share a wall clock; overlap {c['wall_clock_overlap_s']:.1f}s; inside IMU episode: {c['orbbec_inside_imu_episode_same_wall_clock']}")
    clock = {"imu_clock_note": json.load(open(args.imu_dir / "metadata/capture-manifest.json"))["clock"], "sessions": clock_rows}

    # ---- 2. IMU <-> mocap cross-correlation
    imu_rs = {}
    for ep, e in imu.items():
        t, x = resample(e["t_utc"], e["gyro"], rate, args.smooth)
        tp, xp = resample(e["t_utc"], e["gyro"], rate, args.peak_smooth)
        imu_rs[ep] = {"t": t, "x": x, "n": normalise(x), "peaks": swing_peaks(tp, xp, args.peak_dps)}
    clip_rs = {}
    for c, v in clips.items():
        ta, xa = resample(v["t"], v["ang_speed"], rate, args.smooth)
        tl, xl = resample(v["t"], v["speed"], rate, args.smooth)
        tp, xp = resample(v["t"], v["ang_speed"], rate, args.peak_smooth)
        clip_rs[c] = {"t": ta, "ang": normalise(xa), "lin": normalise(xl), "peaks": swing_peaks(tp, xp, args.peak_dps)}
    rows = []
    corr_matrix = np.full((len(imu), len(clips)), np.nan)
    ep_names, clip_names = list(imu), list(clips)
    for i, ep in enumerate(ep_names):
        for j, c in enumerate(clip_names):
            di, dc = imu[ep]["duration"], clips[c]["duration"]
            similar = abs(di - dc) <= args.duration_tol * max(di, dc)
            short_pair = (not similar) and di < 60 and dc < 60 and dc <= di  # short test takes: clip may sit inside the episode
            ir, cr = imu_rs[ep], clip_rs[c]
            res = {}
            for feat in ("ang", "lin"):
                lags, corr = xcorr_lags(ir["n"], cr[feat], rate, args.max_lag, min_overlap_s=10.0 if short_pair else 15.0)
                res[feat] = (lags, corr, peak_stats(lags, corr))
            st = res[args.feature][2]
            alt = res["lin" if args.feature == "ang" else "ang"][2]
            pm = peak_match_lag(ir["peaks"], cr["peaks"], args.max_lag)
            corr_matrix[i, j] = st["peak_corr"]
            row = {"imu_episode": ep, "clip": c, "imu_duration_s": round(di, 2), "clip_duration_s": round(dc, 2),
                   "duration_similar": similar, "short_pair": short_pair, "feature": args.feature,
                   "xcorr_lag_s": st["lag_s"], "xcorr_peak": round(st["peak_corr"], 4),
                   "xcorr_lobe_width_s": st["lobe_width_s"], "xcorr_second_lag_s": st["second_lag_s"], "xcorr_second_peak": None if st["second_corr"] is None else round(st["second_corr"], 4),
                   "secondary_ratio": None if st["secondary_ratio"] is None else round(st["secondary_ratio"], 3),
                   "peak_z": round(st["peak_z"], 2),
                   "alt_feature_lag_s": alt["lag_s"], "alt_feature_peak": round(alt["peak_corr"], 4),
                   "imu_swing_peaks": pm["n_peaks_a"], "hand_swing_peaks": pm["n_peaks_b"], "peakmatch_lag_s": pm["lag_s"],
                   "peakmatch_votes": pm.get("votes"), "peakmatch_matched_frac": None if pm["matched_frac"] is None else round(pm["matched_frac"], 3)}
            rows.append(row)
            if similar or short_pair:
                print(f"[xcorr] {ep[-6:]} x {c}: dur {di:.1f}/{dc:.1f}s{' (short pair)' if short_pair else ''} | {args.feature}: lag {st['lag_s']:+.1f}s r {st['peak_corr']:.3f} "
                      f"lobe {st['lobe_width_s']}s 2nd-ratio {st['secondary_ratio']} z {st['peak_z']:.1f} | alt: lag {alt['lag_s']:+.1f}s r {alt['peak_corr']:.3f} | "
                      f"peaks {pm['n_peaks_a']}/{pm['n_peaks_b']} peak-lag {pm['lag_s']} matched {pm['matched_frac']}")
                if not args.no_plots:
                    lags, corr, _ = res[args.feature]
                    plot_pair(out, ep, c, lags, corr, ir["t"], ir["n"], cr["t"], cr[args.feature], st["lag_s"])
    with open(out / "imu-mocap-xcorr.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if not args.no_plots:
        plot_matrix(out, corr_matrix, ep_names, clip_names, args.feature)

    # ---- optional: Orbbec motion energy vs IMU (independent check of the shared-wall-clock hypothesis)
    if args.orbbec_motion:
        clock["orbbec_motion_xcorr"] = []
        for name, s in orb.items():
            rec = next(r for r in clock_rows if r["orbbec_session"] == name)
            for cam in s["cams"]:
                gt, ge = orbbec_motion_signal(s, cam, rate, out, args.motion_column)
                if len(gt) < 30 * rate:
                    print(f"[orbbec-motion] {name}/{cam}: only {len(gt) / rate:.1f}s of frames, too short for a lag search")
                    continue
                from scipy.ndimage import gaussian_filter1d
                gn = normalise(gaussian_filter1d(ge, args.smooth * rate))
                for cand in rec["imu_candidates"]:
                    if cand["wall_clock_overlap_s"] <= 0:
                        continue
                    ep = cand["episode"]
                    e = imu[ep]
                    ti, xi = resample(e["t_local"], e["gyro"], rate, args.smooth)
                    ni = normalise(xi)
                    expected = (gt[0] + 3600 * rec["implied_host_utc_offset_h"]) - e["start_local_naive"]
                    lags, corr = xcorr_lags(ni, gn, rate, args.max_lag)
                    st = peak_stats(lags, corr)
                    r_exp = float(corr[np.argmin(np.abs(lags - expected))])
                    r = {"orbbec_session": name, "cam": cam, "imu_episode": ep, "orbbec_signal_len_s": round(len(gn) / rate, 1),
                         "expected_lag_s_same_wall_clock": round(expected, 3), "r_at_expected_lag": round(r_exp, 4),
                         "xcorr_lag_s": st["lag_s"], "xcorr_peak": round(st["peak_corr"], 4),
                         "secondary_ratio": st["secondary_ratio"], "peak_z": round(st["peak_z"], 2),
                         "conclusive": bool(st["peak_z"] >= 4 and (st["secondary_ratio"] or 1) < 0.6)}
                    clock["orbbec_motion_xcorr"].append(r)
                    print(f"[orbbec-motion] {name}/{cam} vs IMU {ep[-6:]}: expected lag {expected:+.2f}s if same wall clock, r there {r_exp:.3f}; "
                          f"best lag {st['lag_s']:+.2f}s r {st['peak_corr']:.3f} 2nd-ratio {st['secondary_ratio']} z {st['peak_z']:.1f} -> conclusive={r['conclusive']}")
                    if not args.no_plots:
                        plot_pair(out, ep, f"orbbec-{name.split('_')[-1]}-{cam.split('_')[0]}", lags, corr, ti, ni, gt - gt[0], gn, st["lag_s"])
    json.dump(clock, open(out / "imu-orbbec-clock.json", "w"), indent=1)

    # ---- 1. session map
    session_map = build_session_map(imu, clips, fbx, clock_rows, rows, args)
    json.dump(session_map, open(out / "session-map.json", "w"), indent=1)
    print(f"[out] wrote {out / 'session-map.json'}, imu-mocap-xcorr.csv, imu-orbbec-clock.json, fbx-headers.json")
    return 0


def build_session_map(imu, clips, fbx, clock_rows, xrows, args) -> dict:
    by_pair = {(r["imu_episode"], r["clip"]): r for r in xrows}
    entries = []
    for ep, e in imu.items():
        entry = {"imu_episode": ep, "imu_start_wall_clock": wall_clock_iso(e["start_local_naive"]), "imu_duration_s": round(e["duration"], 2),
                 "imu_start_utc_inferred_ny": dt.datetime.fromtimestamp(e["start_utc"], dt.timezone.utc).isoformat(),
                 "orbbec": None, "rokoko": None}
        # Orbbec: same-wall-clock overlap for extracted sessions, launch-stamp proximity for name-only tars
        for c in clock_rows:
            for cand in c["imu_candidates"]:
                if cand["episode"] != ep:
                    continue
                if c["status"] == "on disk":
                    if cand["wall_clock_overlap_s"] <= 0:
                        continue
                    entry["orbbec"] = {"session": c["orbbec_session"], "status": c["status"], "n_frames": c["n_frames"], "duration_s": c["duration_s"],
                                       "first_frame_minus_imu_start_s_same_wall_clock": cand["first_frame_minus_imu_start_s_same_wall_clock"],
                                       "last_frame_minus_imu_end_s_same_wall_clock": cand["last_frame_minus_imu_end_s_same_wall_clock"],
                                       "wall_clock_overlap_s": cand["wall_clock_overlap_s"],
                                       "inside_imu_episode": cand["orbbec_inside_imu_episode_same_wall_clock"],
                                       "confidence": "high (wall-clock windows overlap; assumes both hosts show the same wall clock, see imu-orbbec-clock.json)"}
                elif entry["orbbec"] is None:
                    entry["orbbec"] = {"session": c["orbbec_session"], "status": c["status"],
                                       "imu_start_minus_orbbec_launch_s": cand["imu_start_minus_orbbec_launch_s"],
                                       "confidence": "medium (launch stamp within 60 s of the IMU episode start; no frames on disk to verify)"}
        # Rokoko: duration similarity, then cross-correlation evidence
        cands = sorted(((abs(by_pair[(ep, c)]["imu_duration_s"] - by_pair[(ep, c)]["clip_duration_s"]), c, by_pair[(ep, c)]) for c in clips), key=lambda x: x[0])
        similar = [x for x in cands if x[2]["duration_similar"]] or [x for x in cands if x[2]["short_pair"]]
        if similar:
            ranked = sorted(similar, key=lambda x: -(x[2]["xcorr_peak"] if np.isfinite(x[2]["xcorr_peak"]) else -1))
            top = ranked[0][2]
            runner = ranked[1][2] if len(ranked) > 1 else None
            margin = None if runner is None else top["xcorr_peak"] - runner["xcorr_peak"]
            small_lag = top["xcorr_lag_s"] is not None and abs(top["xcorr_lag_s"]) <= 15
            conf = "high" if (top["peak_z"] >= 4 and (top["secondary_ratio"] or 1) < 0.6 and small_lag and (margin is None or margin > 0.15)) else \
                   "medium" if (top["peak_z"] >= 3 and small_lag and (margin is None or margin > 0.08)) else "low"
            entry["rokoko"] = {"clip": top["clip"], "clip_duration_s": top["clip_duration_s"],
                               "duration_diff_s": round(abs(top["clip_duration_s"] - top["imu_duration_s"]), 2),
                               "feature": top["feature"], "xcorr_lag_s": top["xcorr_lag_s"], "xcorr_peak": top["xcorr_peak"], "xcorr_lobe_width_s": top["xcorr_lobe_width_s"],
                               "secondary_ratio": top["secondary_ratio"], "peak_z": top["peak_z"],
                               "alt_feature_lag_s": top["alt_feature_lag_s"], "alt_feature_peak": top["alt_feature_peak"],
                               "peakmatch_lag_s": top["peakmatch_lag_s"], "peakmatch_matched_frac": top["peakmatch_matched_frac"],
                               "duration_similar_candidates": [{"clip": x[2]["clip"], "xcorr_peak": x[2]["xcorr_peak"], "xcorr_lag_s": x[2]["xcorr_lag_s"],
                                                                "secondary_ratio": x[2]["secondary_ratio"], "peak_z": x[2]["peak_z"]} for x in ranked],
                               "xcorr_margin_over_runner_up": None if margin is None else round(margin, 4),
                               "ambiguous_by_duration": len(similar) > 1, "confidence": conf,
                               "short_pair_caveat": ("both recordings < 60 s: r is computed over ~10-17 s of low-amplitude motion, so the "
                                                     "absolute r is inflated; the lag is still the best estimate") if top["short_pair"] else None,
                               "fbx_creation_timestamp": fbx.get(top["clip"], {}).get("creation_timestamp")}
        else:
            entry["rokoko"] = {"clip": None, "nearest_by_duration": cands[0][1], "duration_diff_s": round(cands[0][0], 2),
                               "confidence": "none (no clip within duration tolerance)"}
        entries.append(entry)
    # one clip <-> one episode: if two episodes claim the same clip, the higher correlation keeps it
    claims: dict[str, list] = {}
    for en in entries:
        if en["rokoko"] and en["rokoko"].get("clip"):
            claims.setdefault(en["rokoko"]["clip"], []).append(en)
    for clip, ens in claims.items():
        if len(ens) < 2:
            continue
        ens.sort(key=lambda en: -en["rokoko"]["xcorr_peak"])
        for loser in ens[1:]:
            r = loser["rokoko"]
            loser["rokoko"] = {"clip": None, "rejected_candidate": clip, "rejected_xcorr_peak": r["xcorr_peak"], "rejected_xcorr_lag_s": r["xcorr_lag_s"],
                               "confidence": f"none ({clip} is claimed by {ens[0]['imu_episode']} with r={ens[0]['rokoko']['xcorr_peak']})"}
    return {"generated": dt.datetime.now(dt.timezone.utc).isoformat(), "rate_hz": args.rate, "max_lag_s": args.max_lag,
            "smooth_s": args.smooth, "feature": args.feature, "duration_tol": args.duration_tol,
            "notes": [
                "imu_start_utc_inferred_ny = HIM's timestamp_utc (America/New_York assumed by the exporter). imu_start_wall_clock = the raw timestamp_local.",
                "Orbbec host clock is epoch; its directory name is the host wall clock at launch, which implies the host UTC offset (imu-orbbec-clock.json).",
                "Rokoko FBX carries only an export CreationTimeStamp (2026-08-17, two days after capture) and a take named 'clip'; no recording timestamp exists in the file.",
                "xcorr_lag_s: positive = the IMU signal lags the clip, i.e. clip time t corresponds to IMU episode time t + lag; negative = the IMU episode started after the clip.",
                "Rokoko confidence comes from the cross-correlation only; duration-similar candidates are listed so the ambiguity is visible.",
            ],
            "fbx_headers": {k: {kk: v[kk] for kk in ("creation_timestamp", "takes", "global_fps", "rokoko_export_folder_leaf") if kk in v} for k, v in fbx.items()},
            "orbbec_sessions": [{k: v for k, v in c.items() if k != "imu_candidates"} for c in clock_rows],
            "pairs": entries}


# ----------------------------------------------------------------------------- plots
def plot_pair(out: Path, ep: str, clip: str, lags, corr, ti, ni, tc, nc, lag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].plot(lags, corr, lw=0.8)
    ax[0].axvline(lag, color="r", ls="--", lw=0.8)
    ax[0].set_title(f"{ep} vs {clip}: cross-correlation (best lag {lag:+.1f} s)")
    ax[0].set_xlabel("lag (s), + = IMU lags clip")
    ax[0].set_ylabel("Pearson r")
    ax[1].plot(ti, ni, lw=0.6, label=f"IMU |gyro| ({ep})")
    ax[1].set_xlim(min(ti[0], tc[0] + lag), max(ti[-1], tc[-1] + lag))
    ax[1].plot(tc + lag, nc, lw=0.6, alpha=0.7, label=f"{clip} shifted by {lag:+.1f} s")
    ax[1].set_xlabel("IMU episode time (s)")
    ax[1].set_ylabel("normalised")
    ax[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"xcorr_{ep.replace('witmotion_', '')}_{clip}.png", dpi=110)
    plt.close(fig)


def plot_matrix(out: Path, m: np.ndarray, eps, clips, feature: str = "ang"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(m, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(clips)), clips, rotation=45, ha="right")
    ax.set_yticks(range(len(eps)), [e.replace("witmotion_", "") for e in eps])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, ax=ax, label="peak cross-correlation")
    ax.set_title(f"IMU episode x Rokoko clip: peak correlation of |gyro| vs RightHand {'angular' if feature == 'ang' else 'linear'} speed")
    fig.tight_layout()
    fig.savefig(out / "xcorr-matrix.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
