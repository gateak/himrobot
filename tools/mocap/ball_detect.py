"""Detect the table-tennis ball in Orbbec color frames (perception track, 2D pass).

Reads a session from an extracted directory
(``<session>/<cam>/{color,depth,frames.jsonl}`` + ``session.json``) or directly
from the session tar (no bulk extraction). The ball is a small orange disc on
a dark-blue table: an HSV orange mask + blob roundness finds candidates,
optionally gated by frame differencing (``--motion``), a table ROI from ArUco
corner markers (``--roi-json``, from ``tools/mocap/aruco_scan.py``), a static-orange
map (``--static-reject``), and distance to big red/orange blobs (``--near-big``,
paddle rubber) to reject clutter (bottle, sandals, skin, paddle); it then
tracks by proximity.

Outputs per-frame detection stats, a CSV 2D track
(frame, host_time_s, u, v, radius; blank u/v/radius where nothing was found),
and annotated sample frames. Depth lookup: ``tools/mocap/ball_3d.py``.

    ~/him-tools/cv-venv/bin/python tools/mocap/ball_detect.py \\
        --session data/orbbec_20260815_215350 --cam cam0 --motion 8 \\
        --roi-json work/perception/aruco_orbbec_20260815_215350.json \\
        --static-reject 0.2 --near-big 12
    ~/him-tools/cv-venv/bin/python tools/mocap/ball_detect.py \\
        --session data/release/orbbec_20260815_215350.tar --cam cam1_CP0E8530008F
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tarfile

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/mocap/ -> repo root
DEF_SESSION = os.path.join(REPO, "data", "orbbec_20260815_215350")
DEF_CAM = "cam0_CP0E8530008E"
OUT_DIR = os.path.join(REPO, "work", "perception")


# --------------------------------------------------------------------------- I/O
class Session:
    """Uniform access to an Orbbec session stored as a directory or a tar."""

    def __init__(self, path):
        self.path = path
        self.is_tar = os.path.isfile(path)
        if self.is_tar:
            self.tf = tarfile.open(path)
            members = self.tf.getmembers()  # header scan only (uncompressed tar)
            self.index = {m.name: m for m in members if m.isfile()}
            top = sorted({m.name.split("/", 1)[0] for m in members})
            self.name = top[0]
            self._prefix = self.name + "/"
        else:
            self.tf = None
            self.name = os.path.basename(os.path.normpath(path))
            self._prefix = ""

    def cams(self):
        if self.is_tar:
            names = {n.split("/")[1] for n in self.index if n.count("/") >= 2}
        else:
            names = {d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d))}
        return sorted(n for n in names if n.startswith("cam"))

    def resolve_cam(self, cam):
        cams = self.cams()
        hits = [c for c in cams if c == cam or c.startswith(cam)]
        if len(hits) != 1:
            raise SystemExit(f"--cam {cam!r} matches {hits} among {cams}")
        return hits[0]

    def read_bytes(self, rel):
        if self.is_tar:
            m = self.index.get(self._prefix + rel)
            return None if m is None else self.tf.extractfile(m).read()
        p = os.path.join(self.path, rel)
        if not os.path.isfile(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def read_json(self, rel):
        b = self.read_bytes(rel)
        return None if b is None else json.loads(b)

    def frames(self, cam):
        """Rows of <cam>/frames.jsonl (idx, host_time_s, color_file, depth_file, ...)."""
        b = self.read_bytes(f"{cam}/frames.jsonl")
        if b is None:
            raise SystemExit(f"no frames.jsonl for {cam} in {self.path}")
        return [json.loads(l) for l in b.decode().splitlines() if l.strip()]

    def read_image(self, cam, rel, flags=cv2.IMREAD_COLOR):
        b = self.read_bytes(f"{cam}/{rel}")
        if b is None:
            return None
        return cv2.imdecode(np.frombuffer(b, np.uint8), flags)


def iter_frames(sess, cam, start=0, count=None, step=1, stats=None):
    """Yield (idx, host_time_s, bgr_image) for the requested frame rows.

    Skips rows with a null color_file (dropped by the recorder). Also skips
    rows whose color_file fails to decode (corrupt/truncated JPEG), logging
    each one so storage corruption doesn't look like an ordinary recorder
    gap. If `stats` (a dict) is given, "null_color_file" and "decode_failed"
    counts accumulate into it for the caller to report.
    """
    rows = sess.frames(cam)
    end = len(rows) if count is None else min(len(rows), start + count)
    for r in rows[start:end:step]:
        if not r.get("color_file"):
            if stats is not None:
                stats["null_color_file"] = stats.get("null_color_file", 0) + 1
            continue
        img = sess.read_image(cam, r["color_file"])
        if img is None:
            if stats is not None:
                stats["decode_failed"] = stats.get("decode_failed", 0) + 1
            print(f"warn: {cam}/{r['color_file']} (frame {r.get('idx')}) present but failed to decode; skipping",
                  file=sys.stderr)
            continue
        yield r["idx"], float(r["host_time_s"]), img


# ----------------------------------------------------------------------- detector
# Orange table-tennis ball in HSV (OpenCV H is 0-179). High saturation rejects
# the gray floor/table; the hue band rejects the blue table and most skin.
ORANGE_LO = np.array([3, 95, 80], dtype=np.uint8)
ORANGE_HI = np.array([26, 255, 255], dtype=np.uint8)
KERNEL = np.ones((3, 3), np.uint8)


class BallDetector:
    """Per-frame orange-blob detector with proximity tracking.

    motion: if > 0, a candidate blob must show mean |frame - previous frame|
    above this value (gray levels) inside its disc, i.e. it must be moving.
    """

    def __init__(self, min_area=2.0, max_area=500.0, min_circ=0.35, motion=0.0,
                 max_jump=None, roi=None, roi_margin=0.0, static_map=None,
                 static_reject=0.0, near_big=0.0, big_area=800.0):
        self.min_area, self.max_area, self.min_circ = min_area, max_area, min_circ
        self.motion = motion
        self.max_jump = max_jump
        self.roi = None if roi is None else np.asarray(roi, np.float32).reshape(-1, 1, 2)
        self.roi_margin = roi_margin
        self.static_map = static_map          # float32 HxW: fraction of frames orange
        self.static_reject = static_reject
        self.near_big = near_big              # px: reject candidates this close to a big red/orange blob
        self.big_area = big_area
        self.last = None
        self.prev_gray = None

    def _big_blobs(self, hsv):
        """Contours of large red/orange regions (paddle rubber, clothing)."""
        m = cv2.inRange(hsv, np.array([0, 80, 40], np.uint8), np.array([30, 255, 255], np.uint8))
        m |= cv2.inRange(hsv, np.array([165, 80, 40], np.uint8), np.array([179, 255, 255], np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cnts if cv2.contourArea(c) > self.big_area]

    def __call__(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
        mask = cv2.dilate(mask, KERNEL, iterations=1)
        diff = None
        if self.motion > 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is not None:
                diff = cv2.absdiff(gray, self.prev_gray)
            self.prev_gray = gray
            if diff is None:            # first frame: no motion reference
                return None
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = self._big_blobs(hsv) if self.near_big > 0 else []
        best, best_score = None, -1e9
        for c in cnts:
            a = cv2.contourArea(c)
            if a < self.min_area or a > self.max_area:      # shoes/cables are bigger
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < 1:
                continue
            circ = a / (np.pi * r * r + 1e-6)               # 1.0 = perfect disc
            if circ < self.min_circ:
                continue
            if self.roi is not None and \
                    cv2.pointPolygonTest(self.roi, (float(x), float(y)), True) < -self.roi_margin:
                continue                                    # off the table (+margin)
            if self.static_map is not None and self.static_reject > 0:
                yi, xi = min(int(y), self.static_map.shape[0] - 1), min(int(x), self.static_map.shape[1] - 1)
                if self.static_map[yi, xi] >= self.static_reject:
                    continue                                # orange here in most frames: clutter
            if big and any(abs(cv2.pointPolygonTest(b, (float(x), float(y)), True)) <= self.near_big + r
                           or cv2.pointPolygonTest(b, (float(x), float(y)), False) >= 0 for b in big):
                continue                                    # touching the paddle / a red garment
            if diff is not None:
                m = np.zeros(diff.shape, np.uint8)
                cv2.circle(m, (int(x), int(y)), max(2, int(r)), 255, -1)
                if cv2.mean(diff, mask=m)[0] < self.motion:
                    continue
            prox = 0.0
            if self.last is not None:
                d = float(np.hypot(x - self.last[0], y - self.last[1]))
                if self.max_jump is not None and d > self.max_jump:
                    continue
                prox = -0.1 * d
            score = 0.3 * a + 80 * circ + prox              # round, sizable, near last
            if score > best_score:
                best_score, best = score, (float(x), float(y), float(r))
        if best is not None:
            self.last = best[:2]
        return best


def orange_static_map(sess, cam, step=10):
    """Fraction of (subsampled) frames in which each pixel passes the orange mask."""
    acc, n = None, 0
    for _i, _t, img in iter_frames(sess, cam, step=step):
        m = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), ORANGE_LO, ORANGE_HI)
        m = cv2.dilate(m, KERNEL, iterations=2)
        acc = (m > 0).astype(np.float32) if acc is None else acc + (m > 0)
        n += 1
    return None if acc is None else acc / n


def roi_from_aruco(json_path, cam, dict_name=None):
    """Convex hull (Nx2 px) of all marker corners found for `cam` in an aruco_scan JSON."""
    with open(json_path) as f:
        j = json.load(f)
    c = j["cams"][cam]
    key = dict_name or c["best_dict"]
    if key is None:
        raise SystemExit(f"roi_from_aruco: no ArUco markers were found for cam {cam!r} in {json_path} "
                          "(best_dict is null); pass --roi-json for a session/cam where markers were detected")
    if key not in c["per_dict"]:
        raise SystemExit(f"roi_from_aruco: dict {key!r} has no markers for cam {cam!r} in {json_path} "
                          f"(available: {sorted(c['per_dict'])})")
    d = c["per_dict"][key]
    pts = np.array([p for v in d["ids"].values() for p in v["corners_px_median"]], np.float32)
    return cv2.convexHull(pts).reshape(-1, 2)


def detect(frames, **kw):
    """Batch helper kept for the old API: list of images -> list of (x, y, r) | None."""
    det = BallDetector(**kw)
    return [det(img) for img in frames]


# ---------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=DEF_SESSION,
                    help="extracted session directory or session .tar")
    ap.add_argument("--tar", help="alias of --session for a .tar (backward compat)")
    ap.add_argument("--cam", default=DEF_CAM,
                    help="camera folder, full label or unique prefix (cam0 / cam1)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None, help="frames to process (default: all)")
    ap.add_argument("--step", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--motion", type=float, default=0.0,
                    help="require mean frame-diff inside the blob >= this (0 = off)")
    ap.add_argument("--max-jump", type=float, default=None,
                    help="reject candidates farther than this many px from the last hit")
    ap.add_argument("--roi-json", default=None,
                    help="aruco_scan JSON; candidates outside the marker hull (+margin) are dropped")
    ap.add_argument("--roi-margin", type=float, default=40.0, help="px outside the hull still allowed")
    ap.add_argument("--static-reject", type=float, default=0.0,
                    help="drop candidates where the pixel is orange in >= this fraction of frames (0 = off)")
    ap.add_argument("--near-big", type=float, default=0.0,
                    help="drop candidates within this many px of a large red/orange blob (0 = off)")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    sess = Session(args.tar or args.session)
    cam = sess.resolve_cam(args.cam)
    tag = f"{sess.name}_{cam}"
    os.makedirs(args.out_dir, exist_ok=True)

    roi = roi_from_aruco(args.roi_json, cam) if args.roi_json else None
    static_map = None
    if args.static_reject > 0:
        static_map = orange_static_map(sess, cam)
        cv2.imwrite(os.path.join(args.out_dir, f"static_orange_{tag}.png"),
                    (static_map * 255).astype(np.uint8))
    det = BallDetector(motion=args.motion, max_jump=args.max_jump, roi=roi,
                       roi_margin=args.roi_margin, static_map=static_map,
                       static_reject=args.static_reject, near_big=args.near_big)
    rows = []                       # (idx, t, (x, y, r) | None)
    n_img = 0
    frame_stats = {}
    for idx, t, img in iter_frames(sess, cam, args.start, args.count, args.step, stats=frame_stats):
        d = det(img)
        rows.append((idx, t, d))
        n_img += 1
    if not rows:
        print("no frames")
        return 1

    if frame_stats.get("decode_failed"):
        print(f"{tag}: {frame_stats['decode_failed']} color frame(s) present on disk but failed to decode "
              "(corrupt/truncated); skipped, not counted as recorder drops")

    hit = [r for r in rows if r[2] is not None]
    print(f"{tag}: {len(hit)}/{n_img} frames with a ball candidate "
          f"({100.0 * len(hit) / max(1, n_img):.1f}%), frames {rows[0][0]}..{rows[-1][0]}")
    for idx, _t, d in hit[:10]:
        print(f"  frame {idx}: u={d[0]:.1f} v={d[1]:.1f} r={d[2]:.1f}px")

    csv_path = os.path.join(args.out_dir, f"ball2d_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "host_time_s", "u", "v", "radius"])
        for idx, t, d in rows:
            if d is None:
                w.writerow([idx, f"{t:.6f}", "", "", ""])
            else:
                w.writerow([idx, f"{t:.6f}", f"{d[0]:.2f}", f"{d[1]:.2f}", f"{d[2]:.2f}"])
    print(f"wrote {csv_path}")

    # Annotated samples spread evenly over the detections, for a visual check.
    picks = hit
    if args.samples <= 0:
        picks = []
    elif len(picks) > args.samples:
        picks = picks[:: max(1, len(picks) // args.samples)][: args.samples]
    frame_rows = {r["idx"]: r for r in sess.frames(cam)}
    for idx, _t, d in picks:
        img = sess.read_image(cam, frame_rows[idx]["color_file"])
        if roi is not None:
            cv2.polylines(img, [roi.astype(np.int32)], True, (255, 200, 0), 1)
        cv2.circle(img, (int(d[0]), int(d[1])), max(6, int(d[2]) + 4), (0, 0, 255), 2)
        cv2.putText(img, f"{cam[:4]} f{idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.imwrite(os.path.join(args.out_dir, f"det_{tag}_{idx:06d}.jpg"), img)
    print(f"saved {len(picks)} annotated samples to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
