#!/usr/bin/env python3
"""Composite Orbbec footage with the retargeted G1 (MuJoCo) at the aligned time.

Time chain (constants READ FROM work/alignment/*.json at run time, via
--session-map / --clock-json; each has a CLI override):

    wall(frame)  = utc_from_epoch(host_time_s) + host_utc_offset_h     (Orbbec host clock, offset -7 h)
    imu_t        = wall(frame) - imu_start_wall                        (IMU episode time, s)
    clip_t       = imu_t - lag + lag_offset                             (lag: 'clip time t <-> IMU time t + lag')
    csv_row      = 1 + round(clip_t * csv_fps)                          (clip t=0 is BVH frame 1; CSV row 0 is the dropped T-pose)

A frame is used only when 1 <= csv_row <= nrows-1 and its color_file is not null.
MP4 is written at constant --fps on grid t_k = t_first_used + k/fps; each frame pairs the
footage frame with host_time_s nearest t_k with the G1 at THAT frame's own csv_row, so video
time == real elapsed time (footage frames repeat or drop where the recorder ran
slower/faster than --fps).

Modes:
  side     1280x480: footage left, G1 right, caption bar at the bottom.
  overlay  640x480: G1 rendered against a key colour, alpha-masked, scaled by
           --overlay-scale and pasted at --overlay-anchor over the footage; caption bar.
  strip    PNG with N rows of swing peaks (IMU |gyro| maxima, one per equal time bin),
           each row: footage and G1 at t-dt, t, t+dt.

Geometry is a COMPOSITE, not calibrated registration: the MuJoCo camera only imitates the
footage viewpoint (--azimuth/--elevation/--distance); nothing is projected through the real
camera intrinsics/extrinsics (unknown, docs 6a).

Ball (side/overlay only, drawn on the footage panel): for each footage frame with a
detection in --ball-csv (default: auto-located work/perception/ball2d_<session>_<cam>_*.csv),
draws a thin orange circle at (u,v), radius from the CSV (min 6 px), a fading trail of the
previous --ball-trail detections, and -- if the CSV has a depth_mm column (ball3d_*.csv) --
depth in metres as text. Detections are noisy and NOT interpolated: a frame with no CSV row
gets no ball. --no-ball disables all of this.

--start-s/--end-s/--duration-s subset the OUTPUT to a footage/session-elapsed time window
(host_time_s minus the first frame's host_time_s in frames.jsonl): wall/footage time, not
imu_t or clip_t, i.e. "seconds into the recording" rather than into the clip.
"""
import argparse, csv, glob, json, os, sys
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.ndimage import binary_erosion
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/mocap/ -> repo root
# Prefer the vendored, tracked copy (same g1.xml + meshes) over the git-ignored
# third_party/ sparse checkout or the fallback outside the repo.
_SCENES = [os.path.join(REPO, "assets", "unitree_g1", "scene.xml"),
           os.path.join(REPO, "third_party", "mujoco_menagerie", "unitree_g1", "scene.xml"),
           os.path.expanduser("~/him-tools/mujoco_menagerie/unitree_g1/scene.xml")]
SCENE = os.environ.get("G1_SCENE") or next((c for c in _SCENES if os.path.isfile(c)), _SCENES[0])
KEY = np.array([255, 0, 255], dtype=np.int16)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=os.path.join(REPO, "data", "orbbec_20260815_215350"),
                    help="Orbbec session directory")
    ap.add_argument("--cam", default="cam0", help="cam0|cam1 (prefix of the camera sub-directory)")
    ap.add_argument("--csv", default=os.path.join(REPO, "data", "g1", "Clip_3.csv"), help="retargeted G1 CSV")
    ap.add_argument("--joint-names", default=None, help="default: gmr_g1_joint_order.txt next to the CSV")
    ap.add_argument("--csv-fps", type=float, default=30.0)
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--session-map", default=os.path.join(REPO, "work", "alignment", "session-map.json"))
    ap.add_argument("--clock-json", default=os.path.join(REPO, "work", "alignment", "imu-orbbec-clock.json"))
    ap.add_argument("--imu-csv", default=os.path.join(REPO, "data", "him-table-tennis-paddle-imu-20260815",
                                                      "normalized", "paddle-imu-samples.csv"))
    ap.add_argument("--imu-start", default=None, help="override IMU episode start wall clock (ISO, no zone)")
    ap.add_argument("--lag", type=float, default=None, help="override IMU<->clip lag (s) from session-map.json")
    ap.add_argument("--host-utc-offset-h", type=float, default=None, help="override Orbbec host UTC offset (h)")
    ap.add_argument("--lag-offset", type=float, default=0.0,
                    help="seconds ADDED to clip_t (positive = robot shows a LATER part of the clip for the same footage frame)")
    ap.add_argument("--mode", choices=["side", "overlay", "strip"], default="side")
    ap.add_argument("--out", default=None, help="output path (default work/composite/<session>_<cam>_<mode>[_lagXX].mp4|png)")
    ap.add_argument("--fps", type=float, default=20.0, help="constant output fps (nearest-frame sampling)")
    ap.add_argument("--start-s", type=float, default=None,
                    help="only footage from this session-elapsed second (footage-relative: host_time_s minus the "
                         "first frame's host_time_s in frames.jsonl; NOT imu_t/clip_t)")
    ap.add_argument("--end-s", type=float, default=None, help="only footage up to this session-elapsed second (same base as --start-s)")
    ap.add_argument("--duration-s", type=float, default=None,
                    help="alternative to --end-s: only footage for this many seconds after --start-s "
                         "(--start-s defaults to 0 if omitted); ignored if --end-s is given")
    # render camera
    ap.add_argument("--azimuth", type=float, default=155.0, help="MuJoCo free-camera azimuth (deg)")
    ap.add_argument("--elevation", type=float, default=-38.0, help="MuJoCo free-camera elevation (deg, negative looks down)")
    ap.add_argument("--distance", type=float, default=2.6, help="camera distance from lookat (m)")
    ap.add_argument("--lookat-z", type=float, default=0.75)
    ap.add_argument("--no-follow", action="store_true", help="fix lookat xy at the clip's median root xy instead of following the root")
    ap.add_argument("--floor-rgba", default="0.82,0.82,0.80,1")
    # overlay
    ap.add_argument("--overlay-scale", type=float, default=0.55)
    ap.add_argument("--overlay-anchor", default="bottom-right",
                    choices=["bottom-right", "bottom-left", "top-right", "top-left"])
    ap.add_argument("--overlay-margin", type=int, default=8)
    ap.add_argument("--key-tol", type=int, default=60, help="L1 RGB distance from the key colour counted as background")
    # strip
    ap.add_argument("--strip-n", type=int, default=8)
    ap.add_argument("--strip-dt", type=float, default=0.25, help="context offset (s) for the -dt/+dt columns")
    ap.add_argument("--strip-min-dps", type=float, default=400.0)
    ap.add_argument("--strip-scale", type=float, default=0.5)
    # ball overlay (side/overlay modes only; drawn on the footage panel)
    ap.add_argument("--ball-csv", default=None,
                    help="ball track CSV (frame,host_time_s,u,v,radius[,depth_mm,...], see work/perception/ball2d_*.csv "
                         "and ball3d_*.csv); default: auto-locate work/perception/ball2d_<session>_<cam>_*.csv if it exists")
    ap.add_argument("--no-ball", action="store_true", help="disable the ball overlay even if a track CSV is found")
    ap.add_argument("--ball-trail", type=int, default=6, help="number of previous detections to draw as a fading trail")
    # misc
    ap.add_argument("--save-frames", default="", help="comma-separated footage idx list to also dump as PNG")
    ap.add_argument("--report", default=os.path.join(REPO, "work", "composite", "report.json"),
                    help="JSON report to merge this run into")
    ap.add_argument("--frame-map", action="store_true", help="also write <out>.framemap.csv with the per-frame mapping")
    return ap.parse_args()


# ---------------------------------------------------------------- alignment constants
def load_alignment(a, session_name):
    sm = json.load(open(a.session_map))
    ck = json.load(open(a.clock_json))
    pair = next((p for p in sm["pairs"] if (p.get("orbbec") or {}).get("session") == session_name), None)
    if pair is None:
        sys.exit(f"{session_name} not in {a.session_map} pairs")
    sess = next((s for s in ck["sessions"] if s["orbbec_session"] == session_name), None)
    if sess is None:
        sys.exit(f"{session_name} not in {a.clock_json}")
    info = {
        "imu_episode": pair["imu_episode"],
        "imu_start_wall": a.imu_start or pair["imu_start_wall_clock"],
        "clip": pair["rokoko"]["clip"],
        "lag_s": pair["rokoko"]["xcorr_lag_s"] if a.lag is None else a.lag,
        "lag_source": "cli" if a.lag is not None else f"{os.path.relpath(a.session_map, REPO)} rokoko.xcorr_lag_s",
        "host_utc_offset_h": sess["implied_host_utc_offset_h"] if a.host_utc_offset_h is None else a.host_utc_offset_h,
        "first_frame_minus_imu_start_s_from_file": pair["orbbec"].get("first_frame_minus_imu_start_s_same_wall_clock"),
        "lag_convention": sm["notes"][3],
    }
    return info


def host_to_wall(host_time_s, offset_h):
    return datetime.fromtimestamp(host_time_s, tz=timezone.utc).replace(tzinfo=None) + timedelta(hours=offset_h)


# ---------------------------------------------------------------- footage
def load_frames(session_dir, cam):
    # exact-prefix match (cam + "_") so "cam1" cannot ambiguously match "cam10_..."
    cands = sorted(glob.glob(os.path.join(session_dir, cam + "_*")))
    exact = os.path.join(session_dir, cam)
    if os.path.isdir(exact) and exact not in cands:
        cands.append(exact)
    if len(cands) != 1:
        sys.exit(f"--cam {cam!r} matches {[os.path.basename(c) for c in cands]} under {session_dir} "
                 "(need exactly 1)")
    cam_dir = cands[0]
    rows = [json.loads(l) for l in open(os.path.join(cam_dir, "frames.jsonl")) if l.strip()]
    return cam_dir, rows


# ---------------------------------------------------------------- G1
class G1:
    def __init__(self, a):
        self.m = mujoco.MjModel.from_xml_path(a.scene)
        self.d = mujoco.MjData(self.m)
        self.data = np.loadtxt(a.csv, delimiter=",")
        jn = a.joint_names or os.path.join(os.path.dirname(os.path.abspath(a.csv)), "gmr_g1_joint_order.txt")
        names = [l.strip() for l in open(jn) if l.strip()]
        assert self.data.shape[1] == 7 + len(names), (self.data.shape, len(names))
        self.qadr = np.array([self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
        self.fid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.m.geom_matid[self.fid] = -1
        self.floor_rgba = [float(x) for x in a.floor_rgba.split(",")]
        self.m.geom_rgba[self.fid] = self.floor_rgba
        self.r = mujoco.Renderer(self.m, height=480, width=640)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.elevation, self.cam.distance = a.azimuth, a.elevation, a.distance
        self.lookat_z = a.lookat_z
        self.follow = not a.no_follow
        self.median_xy = np.median(self.data[1:, 0:2], axis=0)
        self.key = False

    def set_key(self, key):
        self.key = key
        self.m.geom_rgba[self.fid] = [1, 0, 1, 1] if key else self.floor_rgba

    def render(self, row_i):
        row = self.data[row_i]
        d = self.d
        d.qpos[:] = 0
        d.qpos[0:3] = row[0:3]
        x, y, z, w = row[3:7]
        d.qpos[3:7] = [w, x, y, z]
        d.qpos[self.qadr] = row[7:]
        mujoco.mj_forward(self.m, d)
        xy = row[0:2] if self.follow else self.median_xy
        self.cam.lookat[:] = [xy[0], xy[1], self.lookat_z]
        self.r.update_scene(d, camera=self.cam)
        f = self.r.scene.flags
        f[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        if self.key:
            f[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
            f[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            f[mujoco.mjtRndFlag.mjRND_HAZE] = 0
            f[mujoco.mjtRndFlag.mjRND_FOG] = 0
        return self.r.render().copy()


# ---------------------------------------------------------------- drawing
_FONT = None
def font(size=13):
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.load_default(size=size)
        except TypeError:
            _FONT = ImageFont.load_default()
    return _FONT


def caption_bar(img, text, h=22):
    """Semi-transparent bar drawn over the bottom h px (keeps the frame size)."""
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    dr.rectangle([0, im.height - h, im.width, im.height], fill=(0, 0, 0, 170))
    dr.text((6, im.height - h + 4), text, fill=(255, 255, 255, 255), font=font())
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def overlay_robot(footage, robot_key, scale, anchor, margin, tol):
    mask = (np.abs(robot_key.astype(np.int16) - KEY).sum(axis=2) > tol)
    mask = binary_erosion(mask, iterations=1)   # drop the anti-aliased edge pixels that blend key colour into the robot
    rgb = robot_key.copy()
    rgb[~mask] = (90, 90, 90)          # neutralise the key so resizing cannot bleed magenta into the edges
    rob = Image.fromarray(rgb)
    al = Image.fromarray((mask * 255).astype(np.uint8))
    w, h = int(round(640 * scale)), int(round(480 * scale))
    rob = rob.resize((w, h), Image.BILINEAR)
    al = al.resize((w, h), Image.BILINEAR).point(lambda v: 255 if v > 140 else 0)
    base = Image.fromarray(footage)
    x = margin if "left" in anchor else base.width - w - margin
    y = margin if "top" in anchor else base.height - h - margin
    base.paste(rob, (x, y), al)
    return np.asarray(base), int(mask.sum())


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((4, 4), text, fill=(255, 255, 0), font=font())
    return np.asarray(im)


# ---------------------------------------------------------------- ball track
BALL_COLOR = (255, 140, 0)  # orange
BALL_MIN_RADIUS = 6.0


def find_default_ball_csv(session_name, cam):
    """work/perception/ball2d_<session>_<cam>_*.csv (device-id suffix varies), first match or None."""
    cands = sorted(glob.glob(os.path.join(REPO, "work", "perception", f"ball2d_{session_name}_{cam}_*.csv")))
    return cands[0] if cands else None


def load_ball_track(path):
    """Ball detections keyed by footage frame idx (CSV 'frame' == frames.jsonl 'idx').

    Blank-u/v rows (no detection) are skipped -- callers must NOT interpolate between what
    remains. Handles plain ball2d_*.csv (frame,host_time_s,u,v,radius) and depth-augmented
    ball3d_*.csv (adds depth_mm,n_valid,bg_depth_mm); depth_mm is used automatically when present.
    """
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        has_depth = "depth_mm" in (r.fieldnames or [])
        track = {}
        for row in r:
            if not row["u"].strip():
                continue
            depth_mm = None
            if has_depth and row.get("depth_mm", "").strip():
                depth_mm = float(row["depth_mm"])
            track[int(row["frame"])] = dict(u=float(row["u"]), v=float(row["v"]), r=float(row["radius"]), depth_mm=depth_mm)
    return track, has_depth


def draw_ball(img, idx, track, frames_sorted, has_depth, trail_n=6):
    """Draw the ball detection at footage frame `idx` (if any) plus a fading trail of the
    up-to-trail_n previous detections. No-op if `idx` has no detection -- never interpolates."""
    if idx not in track:
        return img
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    pos = int(np.searchsorted(frames_sorted, idx))
    prev = frames_sorted[max(0, pos - trail_n):pos]
    for k, pi in enumerate(prev):
        age = len(prev) - k  # 1 = most recent previous detection, trail_n = oldest
        alpha = max(30, int(200 * (1 - (age - 1) / max(1, trail_n))))
        d = track[int(pi)]
        dr.ellipse([d["u"] - 2, d["v"] - 2, d["u"] + 2, d["v"] + 2], fill=BALL_COLOR + (alpha,))
    d = track[idx]
    rad = max(d["r"], BALL_MIN_RADIUS)
    dr.ellipse([d["u"] - rad, d["v"] - rad, d["u"] + rad, d["v"] + rad], outline=BALL_COLOR + (255,), width=2)
    if has_depth and d["depth_mm"] is not None:
        dr.text((d["u"] + rad + 3, d["v"] - 6), f"{d['depth_mm'] / 1000.0:.2f}m", fill=BALL_COLOR + (255,), font=font())
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


# ---------------------------------------------------------------- main
def main():
    a = parse_args()
    if a.mode != "strip" and a.fps <= 0:
        sys.exit(f"--fps must be > 0 for --mode {a.mode} (got {a.fps})")
    session_name = os.path.basename(os.path.normpath(a.session))
    al = load_alignment(a, session_name)
    cam_dir, rows = load_frames(a.session, a.cam)
    cam_name = os.path.basename(cam_dir)
    g1 = G1(a)
    nrows = len(g1.data)
    imu_start = datetime.fromisoformat(al["imu_start_wall"])
    lag = al["lag_s"]
    off_h = al["host_utc_offset_h"]

    # per-frame mapping
    fmap = []
    n_null = n_out = 0
    for r in rows:
        wall = host_to_wall(r["host_time_s"], off_h)
        imu_t = (wall - imu_start).total_seconds()
        clip_t = imu_t - lag + a.lag_offset
        row_i = 1 + int(round(clip_t * a.csv_fps))
        in_clip = 1 <= row_i <= nrows - 1
        ok = r.get("color_file") is not None and in_clip
        n_null += r.get("color_file") is None
        n_out += (not in_clip)
        fmap.append(dict(idx=r["idx"], host_time_s=r["host_time_s"], wall=wall.strftime("%H:%M:%S.%f")[:-3],
                         imu_t=imu_t, clip_t=clip_t, csv_row=row_i, color_file=r.get("color_file"), used=ok))
    used = [f for f in fmap if f["used"]]
    if not used:
        sys.exit("no usable frames")
    t_used = np.array([f["host_time_s"] for f in used])
    if a.start_s is not None or a.end_s is not None or a.duration_s is not None:
        t0 = rows[0]["host_time_s"]
        start_s = a.start_s or 0.0
        end_s = a.end_s if a.end_s is not None else (start_s + a.duration_s if a.duration_s is not None else 1e12)
        lo = t0 + start_s; hi = t0 + end_s
        sel = (t_used >= lo) & (t_used <= hi)
        used = [f for f, s in zip(used, sel) if s]; t_used = t_used[sel]

    ball_track, ball_has_depth, ball_csv_used = {}, False, None
    if not a.no_ball:
        ball_csv_used = a.ball_csv or find_default_ball_csv(session_name, a.cam)
        if ball_csv_used and os.path.isfile(ball_csv_used):
            ball_track, ball_has_depth = load_ball_track(ball_csv_used)
        elif a.ball_csv:
            sys.exit(f"--ball-csv {a.ball_csv} not found")
        else:
            ball_csv_used = None
    ball_frames_sorted = np.array(sorted(ball_track), dtype=int)

    tag = f"_lag{a.lag_offset:+.2f}" if a.lag_offset else ""
    out_dir = os.path.join(REPO, "work", "composite")
    os.makedirs(out_dir, exist_ok=True)
    ext = "png" if a.mode == "strip" else "mp4"
    out = a.out or os.path.join(out_dir, f"{session_name}_{a.cam}_{a.mode}{tag}.{ext}")
    if a.frame_map:
        with open(os.path.splitext(out)[0] + ".framemap.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fmap[0].keys())); w.writeheader(); w.writerows(fmap)

    def cap(f, extra=""):
        return (f"{session_name} {a.cam} | {al['clip']} row {f['csv_row']} | wall {f['wall']} | "
                f"clip_t {f['clip_t']:7.2f}s | lag {lag:+.1f}s offset {a.lag_offset:+.2f}s{extra}")

    def read_color(f):
        return np.asarray(Image.open(os.path.join(cam_dir, f["color_file"])).convert("RGB"))

    report = dict(session=session_name, cam=cam_name, mode=a.mode, csv=os.path.relpath(a.csv, REPO), csv_rows=nrows,
                  alignment=al, lag_offset_s=a.lag_offset, out=out,
                  formula="clip_t = (utc(host_time_s) + host_utc_offset_h) - imu_start_wall - lag + lag_offset; csv_row = 1 + round(clip_t*csv_fps)",
                  frames_total=len(rows), frames_null_color=n_null, frames_outside_clip=n_out, frames_used=len(used),
                  clip_t_range_used=[used[0]["clip_t"], used[-1]["clip_t"]],
                  csv_row_range_used=[used[0]["csv_row"], used[-1]["csv_row"]],
                  wall_range_used=[used[0]["wall"], used[-1]["wall"]],
                  camera=dict(azimuth=a.azimuth, elevation=a.elevation, distance=a.distance, lookat_z=a.lookat_z,
                              follow_root=not a.no_follow, floor_rgba=a.floor_rgba, render_size=[640, 480]),
                  output_fps=a.fps, sampling="constant-fps grid from first used frame; nearest footage frame by host_time_s; G1 at that frame's csv_row")

    save_idx = {int(s) for s in a.save_frames.split(",") if s.strip()}

    if a.mode in ("side", "overlay"):
        g1.set_key(a.mode == "overlay")
        writer = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1)
        n_emit = int(np.floor((t_used[-1] - t_used[0]) * a.fps)) + 1
        last_j = -1; frame = None; n_dup = 0; used_ids = set(); mask_px = []; n_ball_hits = 0
        for k in range(n_emit):
            tk = t_used[0] + k / a.fps
            j = int(np.argmin(np.abs(t_used - tk)))
            if j == last_j:
                n_dup += 1
            else:
                f = used[j]; used_ids.add(j)
                foot = read_color(f)
                if f["idx"] in ball_track:
                    n_ball_hits += 1
                foot = draw_ball(foot, f["idx"], ball_track, ball_frames_sorted, ball_has_depth, a.ball_trail)
                rob = g1.render(f["csv_row"])
                if a.mode == "side":
                    frame = np.concatenate([foot, rob], axis=1)
                else:
                    frame, mp = overlay_robot(foot, rob, a.overlay_scale, a.overlay_anchor, a.overlay_margin, a.key_tol)
                    mask_px.append(mp)
                frame = caption_bar(frame, cap(f, f" | video t {k / a.fps:6.2f}s"))
                if f["idx"] in save_idx:
                    Image.fromarray(frame).save(os.path.splitext(out)[0] + f"_frame{f['idx']:06d}.png")
                last_j = j
            writer.append_data(frame)
            if k % 500 == 0:
                print(f"{k}/{n_emit}", flush=True)
        writer.close()
        report.update(video_frames=n_emit, video_duration_s=n_emit / a.fps, footage_frames_repeated=n_dup,
                      footage_frames_skipped=len(used) - len(used_ids),
                      real_duration_used_s=float(t_used[-1] - t_used[0]),
                      ball=dict(csv=os.path.relpath(ball_csv_used, REPO) if ball_csv_used else None,
                                has_depth=ball_has_depth, min_radius_px=BALL_MIN_RADIUS, trail_n=a.ball_trail,
                                footage_frames_with_detection=n_ball_hits))
        if a.mode == "overlay":
            report.update(overlay=dict(scale=a.overlay_scale, anchor=a.overlay_anchor, margin=a.overlay_margin,
                                       key_rgb=[255, 0, 255], key_tol=a.key_tol,
                                       robot_mask_px_median=float(np.median(mask_px))))
    else:  # strip
        g1.set_key(False)
        imu = [r for r in csv.DictReader(open(a.imu_csv)) if r["episode_id"] == al["imu_episode"]]
        if not imu:
            sys.exit(f"no IMU rows for episode_id={al['imu_episode']!r} in {a.imu_csv}; "
                     "check --imu-csv / the session-map episode id")
        ti = np.array([(datetime.fromisoformat(r["timestamp_local"]) - imu_start).total_seconds() for r in imu])
        gy = np.linalg.norm([[float(r[k]) for k in ("angular_velocity_x_dps", "angular_velocity_y_dps",
                                                     "angular_velocity_z_dps")] for r in imu], axis=1)
        # overlap of IMU with used footage (IMU time)
        lo = used[0]["imu_t"] + a.strip_dt; hi = used[-1]["imu_t"] - a.strip_dt
        edges = np.linspace(lo, hi, a.strip_n + 1)
        peaks = []
        for b in range(a.strip_n):
            sel = np.where((ti >= edges[b]) & (ti < edges[b + 1]))[0]
            if not len(sel):
                continue
            i = sel[np.argmax(gy[sel])]
            if gy[i] >= a.strip_min_dps:
                peaks.append((float(ti[i]), float(gy[i])))
        if not peaks:
            sys.exit(f"no IMU |gyro| peak >= {a.strip_min_dps} dps in any of the {a.strip_n} time bins "
                     f"over imu_t [{lo:.2f}, {hi:.2f}]s (imu_episode={al['imu_episode']!r}, {len(imu)} rows); "
                     "try a lower --strip-min-dps or check the episode overlap")
        imu_used = np.array([f["imu_t"] for f in used])
        tiles_rows = []; strip_info = []
        for (tp, gp) in peaks:
            cols = []
            info = dict(imu_t=tp, gyro_dps=gp, wall=(imu_start + timedelta(seconds=tp)).strftime("%H:%M:%S.%f")[:-3], frames=[])
            tw, th = int(round(640 * a.strip_scale)), int(round(480 * a.strip_scale))
            for dt in (-a.strip_dt, 0.0, a.strip_dt):
                j = int(np.argmin(np.abs(imu_used - (tp + dt)))); f = used[j]
                foot = np.asarray(Image.fromarray(read_color(f)).resize((tw, th), Image.BILINEAR))
                rob = np.asarray(Image.fromarray(g1.render(f["csv_row"])).resize((tw, th), Image.BILINEAR))
                foot = label(foot, f"{a.cam} idx {f['idx']}  wall {f['wall']}  imu_t {f['imu_t']:.2f}  ({dt:+.2f}s)")
                rob = label(rob, f"G1 row {f['csv_row']}  clip_t {f['clip_t']:.2f}  ({dt:+.2f}s)")
                cols += [foot, rob]
                info["frames"].append(dict(dt=dt, idx=f["idx"], csv_row=f["csv_row"], clip_t=f["clip_t"]))
            rowimg = np.concatenate(cols, axis=1)
            head = np.full((22, rowimg.shape[1], 3), 30, np.uint8)
            head = label(head, f"IMU |gyro| peak {gp:.0f} dps at imu_t {tp:.2f} s (wall {info['wall']})  |  lag {lag:+.1f} s  offset {a.lag_offset:+.2f} s")
            tiles_rows.append(np.concatenate([head, rowimg], axis=0)); strip_info.append(info)
        strip = np.concatenate(tiles_rows, axis=0)
        strip = caption_bar(strip, f"{session_name} {a.cam} vs {al['clip']} | columns: footage/G1 at peak-{a.strip_dt}s, peak, peak+{a.strip_dt}s | lag {lag:+.1f}s offset {a.lag_offset:+.2f}s")
        Image.fromarray(strip).save(out)
        report.update(strip=dict(n=len(peaks), dt=a.strip_dt, min_dps=a.strip_min_dps, peaks=strip_info))

    # merge into report.json
    rep = {}
    if os.path.isfile(a.report):
        try:
            rep = json.load(open(a.report))
        except Exception:
            rep = {}
    rep.setdefault("runs", {})[os.path.basename(out)] = report
    rep["command_line_last"] = " ".join(sys.argv)
    json.dump(rep, open(a.report, "w"), indent=1)
    print("wrote", out)
    print(json.dumps({k: v for k, v in report.items() if k not in ("alignment", "strip")}, indent=1))


if __name__ == "__main__":
    main()
