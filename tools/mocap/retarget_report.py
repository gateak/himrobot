#!/usr/bin/env python3
"""Retarget fidelity report: LAFAN1-named BVH (human) vs GMR G1 CSV (robot).

  python tools/mocap/retarget_report.py --bvh_dir DIR --csv_dir DIR --out work/retarget/report
      [--clips Clip Clip_2] [--scene .../unitree_g1/scene.xml] [--ik_model .../g1_mocap_29dof.xml]
  python tools/mocap/retarget_report.py --measure_height clip.bvh          # subject height from frame 0

Writes <out>.json and <out>.md. Per clip:
  * right-wrist position error (cm): pelvis-centred heading frame (origin Hips/pelvis; z =
    world up, y = left-hip minus right-hip projected to ground, x = y x z; human vector
    scaled by robot_arm_len / human_arm_len) and shoulder-centred frame (origin RightArm /
    right_shoulder_pitch_link, isolating arm pose from torso-proportion differences).
  * root (pelvis) height min/max; min foot z (foot-class sphere bottoms); joint-limit
    saturation counts (|q - limit| < 1e-3 rad) vs the IK model's ranges (what GMR's
    ConfigurationLimit clamps at) and the Menagerie ranges; geometric elbow angle range
    (0 deg = straight) for human and robot, plus right_elbow_joint q range.
BVH handling: 6-channel joints -> position channels REPLACE the OFFSET (Blender export);
cm -> m and Y-up -> Z-up exactly as GMR's lafan1 loader does ((x, y, z) -> (x, -z, y)).
"""
import argparse, json, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/mocap/ -> repo root
# Prefer the vendored, tracked copy (same g1.xml + meshes) over the git-ignored
# third_party/ sparse checkout or the fallback outside the repo.
_SCENES = [os.path.join(REPO, "assets", "unitree_g1", "scene.xml"),
           os.path.join(REPO, "third_party", "mujoco_menagerie", "unitree_g1", "scene.xml"),
           os.path.expanduser("~/him-tools/mujoco_menagerie/unitree_g1/scene.xml")]
SCENE = os.environ.get("G1_SCENE") or next((c for c in _SCENES if os.path.isfile(c)), _SCENES[0])
IK_MODEL = os.path.expanduser("~/him-tools/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)  # same as GMR lafan1.py


# ---------------------------------------------------------------- BVH
def load_bvh(path):
    names, parents, offsets, channels, end_sites = [], [], [], [], {}
    stack, in_end = [], False
    with open(path) as h:
        lines = h.read().splitlines()
    i = 0
    while i < len(lines):
        parts = lines[i].split(); i += 1
        if not parts:
            continue
        key = parts[0]
        if key in ("ROOT", "JOINT"):
            names.append(parts[1]); parents.append(stack[-1] if stack else -1)
            offsets.append(None); channels.append([])
            stack.append(len(names) - 1)
        elif key == "End" and parts[1] == "Site":
            in_end = True
        elif key == "OFFSET":
            v = np.array([float(x) for x in parts[1:4]])
            if in_end:
                end_sites[names[stack[-1]]] = v
            else:
                offsets[stack[-1]] = v
        elif key == "CHANNELS":
            channels[stack[-1]] = parts[2:2 + int(parts[1])]
        elif key == "}":
            if in_end:
                in_end = False
            else:
                stack.pop()
        elif key == "MOTION":
            break
    nframes = int(lines[i].split(":")[1]); frame_time = float(lines[i + 1].split(":")[1])
    if nframes == 0:
        raise ValueError(f"{path}: BVH motion has 0 frames (empty clip)")
    data = np.loadtxt(lines[i + 2:i + 2 + nframes], dtype=float, ndmin=2)
    data = data.reshape(nframes, -1)
    return {"names": names, "parents": np.array(parents), "offsets": np.array(offsets, dtype=float),
            "channels": channels, "end_sites": end_sites, "data": data, "frame_time": frame_time}


def _rot(axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    o, z = np.ones_like(c), np.zeros_like(c)
    if axis == "X":
        m = [o, z, z, z, c, -s, z, s, c]
    elif axis == "Y":
        m = [c, z, s, z, o, z, -s, z, c]
    else:
        m = [c, -s, z, s, c, z, z, z, o]
    return np.stack(m, axis=-1).reshape(ang.shape + (3, 3))


def bvh_fk(bvh):
    """Global joint positions (frames, J, 3) in metres, Z-up, plus global rotations."""
    data, names, parents = bvh["data"], bvh["names"], bvh["parents"]
    F, J = data.shape[0], len(names)
    lpos = np.repeat(bvh["offsets"][None], F, axis=0)
    lrot = np.repeat(np.eye(3)[None, None], F, axis=0).repeat(J, axis=1)
    col = 0
    for j, chans in enumerate(bvh["channels"]):
        R = np.repeat(np.eye(3)[None], F, axis=0)
        for c in chans:
            v = data[:, col]; col += 1
            if c.endswith("position"):
                lpos[:, j, "XYZ".index(c[0])] = v  # replaces the OFFSET component
            else:
                R = R @ _rot(c[0], np.radians(v))  # BVH: R = R_c1 R_c2 R_c3
        lrot[:, j] = R
    gpos, grot = np.zeros_like(lpos), np.zeros_like(lrot)
    for j in range(J):
        p = parents[j]
        if p < 0:
            gpos[:, j], grot[:, j] = lpos[:, j], lrot[:, j]
        else:
            gpos[:, j] = gpos[:, p] + np.einsum("fij,fj->fi", grot[:, p], lpos[:, j])
            grot[:, j] = grot[:, p] @ lrot[:, j]
    gpos_m = (gpos @ YUP_TO_ZUP.T) / 100.0
    ends = {k: (gpos[:, names.index(k)] + np.einsum("fij,j->fi", grot[:, names.index(k)], v)) @ YUP_TO_ZUP.T / 100.0
            for k, v in bvh["end_sites"].items()}
    return gpos_m, grot, ends


def measure_height(path):
    bvh = load_bvh(path)
    pos, _, ends = bvh_fk(bvh)
    n = bvh["names"]; P = pos[0]
    def z(name): return float(P[n.index(name), 2])
    lo_toe = min(z("LeftToe"), z("RightToe")); lo_foot = min(z("LeftFoot"), z("RightFoot"))
    lo_end = min(float(ends[k][0, 2]) for k in ends if "Toe" in k) if any("Toe" in k for k in ends) else None
    head_end = float(ends["Head"][0, 2]) if "Head" in ends else None
    floor = min(lo_toe, lo_end) if lo_end is not None else lo_toe
    hips_z = z("Hips")
    out = {"file": os.path.basename(path), "frame": 0,
           "hips_z_m": hips_z, "hips_offset_y_cm": float(bvh["offsets"][0][1]),
           "hips_pos_channel_cm": bvh["data"][0, :3].tolist(),
           "head_joint_z_m": z("Head"), "head_end_site_z_m": head_end,
           "head_end_site_offset_cm": bvh["end_sites"].get("Head", np.zeros(3)).tolist(),
           "lowest_toe_joint_z_m": lo_toe, "lowest_foot_joint_z_m": lo_foot, "lowest_toe_end_site_z_m": lo_end,
           "floor_z_m": floor,
           "height_head_joint_above_floor_m": z("Head") - floor,
           "height_gmr_formula_head_plus_0p2_m": z("Head") - floor + 0.2,
           "leg_len_hip_to_ankle_m": float(np.linalg.norm(P[n.index("RightUpLeg")] - P[n.index("RightLeg")])
                                            + np.linalg.norm(P[n.index("RightLeg")] - P[n.index("RightFoot")])),
           "arm_len_shoulder_to_wrist_m": float(np.linalg.norm(P[n.index("RightArm")] - P[n.index("RightForeArm")])
                                                + np.linalg.norm(P[n.index("RightForeArm")] - P[n.index("RightHand")])),
           "shoulder_above_hips_m": z("RightArm") - hips_z,
           "shoulder_width_m": float(np.linalg.norm(P[n.index("RightArm")] - P[n.index("LeftArm")])),
           "hip_width_m": float(np.linalg.norm(P[n.index("RightUpLeg")] - P[n.index("LeftUpLeg")]))}
    # sanity: bone lengths should be constant across frames (position channels == rigid offsets)
    d = np.linalg.norm(pos[:, n.index("RightHand")] - pos[:, n.index("RightForeArm")], axis=1)
    out["forearm_len_std_m"] = float(d.std()); out["frames"] = int(pos.shape[0])
    return out


# ---------------------------------------------------------------- robot
def load_robot(scene, order_file):
    import mujoco
    m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m)
    names = [l.strip() for l in open(order_file) if l.strip()]
    qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    return m, d, names, qadr


def joint_ranges(xml, names):
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    return np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])


def robot_fk(m, d, qadr, rows):
    import mujoco
    body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    bids = {k: body(k) for k in ["pelvis", "torso_link", "right_shoulder_pitch_link", "right_elbow_link",
                                  "right_wrist_yaw_link", "left_hip_pitch_link", "right_hip_pitch_link"]}
    foot_geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                  and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]).endswith("ankle_roll_link")]
    out = {k: np.zeros((len(rows), 3)) for k in bids}; foot_min = np.zeros(len(rows))
    for i, row in enumerate(rows):
        d.qpos[:] = 0; d.qpos[0:3] = row[0:3]
        x, y, z, w = row[3:7]; d.qpos[3:7] = [w, x, y, z]
        d.qpos[qadr] = row[7:]
        mujoco.mj_kinematics(m, d)
        for k, b in bids.items():
            out[k][i] = d.xpos[b]
        foot_min[i] = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in foot_geoms)
    return out, foot_min


def heading_frame(lhip, rhip):
    y = lhip - rhip; y[:, 2] = 0; y /= np.linalg.norm(y, axis=1, keepdims=True)
    z = np.tile([0, 0, 1.0], (len(y), 1)); x = np.cross(y, z)
    R = np.stack([x, y, z], axis=2)  # columns = axes
    return R


def to_frame(R, origin, p):
    return np.einsum("fji,fj->fi", R, p - origin)


def elbow_angle_deg(sh, el, wr):
    a, b = el - sh, wr - el
    c = np.einsum("fi,fi->f", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def robot_arm_length(m, d):
    import mujoco
    d.qpos[:] = 0; d.qpos[3] = 1; mujoco.mj_kinematics(m, d)
    b = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)].copy()
    return float(np.linalg.norm(b("right_elbow_link") - b("right_shoulder_pitch_link"))
                 + np.linalg.norm(b("right_wrist_yaw_link") - b("right_elbow_link")))


# ---------------------------------------------------------------- report
def report_clip(bvh_path, csv_path, scene, ik_model, order_file):
    bvh = load_bvh(bvh_path); n = bvh["names"]
    hpos, _, _ = bvh_fk(bvh)
    rows = np.loadtxt(csv_path, delimiter=",", ndmin=2)
    F = min(len(rows), hpos.shape[0]); rows, hpos = rows[:F], hpos[:F]
    m, d, names, qadr = load_robot(scene, order_file)
    rb, foot_min = robot_fk(m, d, qadr, rows)
    H = lambda k: hpos[:, n.index(k)]
    human_arm = float(np.linalg.norm(H("RightForeArm") - H("RightArm"), axis=1).mean()
                      + np.linalg.norm(H("RightHand") - H("RightForeArm"), axis=1).mean())
    robot_arm = robot_arm_length(m, d)
    s = robot_arm / human_arm
    Rh = heading_frame(H("LeftUpLeg"), H("RightUpLeg"))
    Rr = heading_frame(rb["left_hip_pitch_link"], rb["right_hip_pitch_link"])
    wh = to_frame(Rh, H("Hips"), H("RightHand")) * s
    wr = to_frame(Rr, rb["pelvis"], rb["right_wrist_yaw_link"])
    err = np.linalg.norm(wr - wh, axis=1) * 100
    wh_s = to_frame(Rh, H("RightArm"), H("RightHand")) * s
    wr_s = to_frame(Rr, rb["right_shoulder_pitch_link"], rb["right_wrist_yaw_link"])
    err_s = np.linalg.norm(wr_s - wh_s, axis=1) * 100
    q = rows[:, 7:]
    sat = {}
    for tag, xml in (("ik_model", ik_model), ("menagerie", scene)):
        if not xml or not os.path.isfile(xml):
            continue
        rng = joint_ranges(xml, names)
        hit = (np.abs(q - rng[:, 0]) < 1e-3) | (np.abs(q - rng[:, 1]) < 1e-3)
        sat[tag] = {nm: int(c) for nm, c in zip(names, hit.sum(0)) if c}
        sat[tag + "_total"] = int(hit.sum())
    ei = names.index("right_elbow_joint")
    he = elbow_angle_deg(H("RightArm"), H("RightForeArm"), H("RightHand"))
    re = elbow_angle_deg(rb["right_shoulder_pitch_link"], rb["right_elbow_link"], rb["right_wrist_yaw_link"])
    human_foot_min = float(min(H(k)[:, 2].min() for k in ("LeftToe", "RightToe", "LeftFoot", "RightFoot")))
    return {"clip": os.path.basename(csv_path)[:-4], "frames": int(F), "bvh": bvh_path, "csv": csv_path,
            "arm_len_human_m": human_arm, "arm_len_robot_m": robot_arm, "arm_scale": s,
            "wrist_err_pelvis_cm": {"mean": float(err.mean()), "median": float(np.median(err)), "max": float(err.max()),
                                    "p95": float(np.percentile(err, 95))},
            "wrist_err_shoulder_cm": {"mean": float(err_s.mean()), "median": float(np.median(err_s)),
                                      "max": float(err_s.max()), "p95": float(np.percentile(err_s, 95))},
            "root_z_m": {"min": float(rows[:, 2].min()), "max": float(rows[:, 2].max())},
            "human_hips_z_m": {"min": float(H("Hips")[:, 2].min()), "max": float(H("Hips")[:, 2].max())},
            "min_foot_z_m": float(foot_min.min()), "frames_foot_below_ground": int((foot_min < 0).sum()),
            "human_min_foot_z_m": human_foot_min,
            "joint_saturation": sat,
            "elbow_deg": {"human": [float(he.min()), float(he.max())], "robot": [float(re.min()), float(re.max())],
                          "robot_q_rad": [float(q[:, ei].min()), float(q[:, ei].max())]}}


def markdown(results, title):
    hdr = ("| clip | frames | wrist err pelvis-frame mean/median/max (cm) | wrist err shoulder-frame mean/max (cm) | "
           "root z min/max (m) | min foot z (m) | sat. frames (IK limits) | top saturating joints | "
           "elbow human/robot (deg) |")
    lines = [f"### {title}", "", hdr, "|" + "---|" * 9]
    for r in results:
        sat = r["joint_saturation"]; ik = sat.get("ik_model", {}); top = sorted(ik.items(), key=lambda kv: -kv[1])[:3]
        w, ws, e = r["wrist_err_pelvis_cm"], r["wrist_err_shoulder_cm"], r["elbow_deg"]
        lines.append(f"| {r['clip']} | {r['frames']} | {w['mean']:.1f} / {w['median']:.1f} / {w['max']:.1f} | "
                     f"{ws['mean']:.1f} / {ws['max']:.1f} | {r['root_z_m']['min']:.3f} / {r['root_z_m']['max']:.3f} | "
                     f"{r['min_foot_z_m']:.3f} | {sat.get('ik_model_total', 0)} | "
                     f"{', '.join(f'{k} {v}' for k, v in top) or '-'} | "
                     f"{e['human'][0]:.0f}-{e['human'][1]:.0f} / {e['robot'][0]:.0f}-{e['robot'][1]:.0f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure_height", metavar="BVH")
    ap.add_argument("--bvh_dir"); ap.add_argument("--csv_dir"); ap.add_argument("--out")
    ap.add_argument("--clips", nargs="*"); ap.add_argument("--title", default=None)
    ap.add_argument("--scene", default=SCENE); ap.add_argument("--ik_model", default=IK_MODEL)
    ap.add_argument("--joint_names", default=None, help="default: gmr_g1_joint_order.txt next to the CSVs")
    a = ap.parse_args()
    if a.measure_height:
        print(json.dumps(measure_height(a.measure_height), indent=1)); return
    if not (a.bvh_dir and a.csv_dir and a.out):
        ap.error("need --bvh_dir --csv_dir --out (or --measure_height)")
    order = a.joint_names or os.path.join(a.csv_dir, "gmr_g1_joint_order.txt")
    clips = a.clips or sorted(f[:-4] for f in os.listdir(a.csv_dir) if f.endswith(".csv"))
    results = [report_clip(os.path.join(a.bvh_dir, c + ".bvh"), os.path.join(a.csv_dir, c + ".csv"),
                           a.scene, a.ik_model, order) for c in clips]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    title = a.title or f"{os.path.basename(os.path.abspath(a.csv_dir))}"
    with open(a.out + ".json", "w") as h:
        json.dump({"title": title, "scene": a.scene, "ik_model": a.ik_model, "torso_frame": "pelvis-centred heading frame",
                   "clips": results}, h, indent=1); h.write("\n")
    md = markdown(results, title)
    with open(a.out + ".md", "w") as h:
        h.write(md)
    print(md)


if __name__ == "__main__":
    main()
