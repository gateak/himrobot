#!/usr/bin/env python3
"""Rewrite a Rokoko "Newton" BVH (Blender export, 6 channels/joint, cm, Y-up)
into a BVH whose bone names match what GMR's `--format lafan1` loader expects.

What GMR actually needs (general_motion_retargeting/utils/lafan1.py +
ik_configs/bvh_lafan1_to_g1.json):
  * bones looked up BY NAME in a dict, so extra bones are tolerated;
  * only these names are used: Hips, Spine2, Left/RightUpLeg, Left/RightLeg,
    Left/RightFoot, Left/RightToe, Left/RightArm, Left/RightForeArm,
    Left/RightHand (Spine/Spine1/Neck/Head/Shoulder are NOT used by the G1 config);
  * 3 or 6 channels per joint both parse (6-channel positions override offsets);
  * units are centimetres (loader divides by 100) and the skeleton is Y-up
    (loader rotates Y-up -> Z-up itself);
  * Euler order is taken from the CHANNELS line (Xrotation Yrotation Zrotation ok).

Therefore this script ONLY renames bones; no unit change, no axis change, no
hierarchy surgery. Spine3 is kept under its own name (harmless extra bone) so
the chain Hips>Spine>Spine1>Spine3>Spine2>Neck>Head is intact, and "Spine2"
(the torso target) is Rokoko Spine4 = the chest bone that carries Neck and
both Shoulders, matching LAFAN1 topology. Optionally finger bones can be
dropped (--drop-fingers) which also removes their motion columns.
"""
import argparse, hashlib, json, os, re

RENAME = {
    "Hips": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Spine3": "Spine3",
    "Spine4": "Spine2", "Neck": "Neck", "Head": "Head",
}
for side in ("Left", "Right"):
    RENAME.update({
        f"{side}Shoulder": f"{side}Shoulder", f"{side}Arm": f"{side}Arm",
        f"{side}ForeArm": f"{side}ForeArm", f"{side}Hand": f"{side}Hand",
        f"{side}Thigh": f"{side}UpLeg", f"{side}Shin": f"{side}Leg",
        f"{side}Foot": f"{side}Foot", f"{side}Toe": f"{side}Toe",
    })


def rewrite(src, dst, drop_fingers=False):
    lines = open(src).read().splitlines()
    out, keep_cols = [], []
    depth_drop = None  # brace depth at which a dropped subtree started
    depth = 0
    in_motion = False
    for line in lines:
        if in_motion:
            parts = line.split()
            if parts and not line.startswith(("Frames", "Frame Time")):
                if drop_fingers:
                    parts = [p for k, p in enumerate(parts) if keep_cols[k]]
                out.append(" ".join(parts))
            else:
                out.append(line)
            continue
        if line.strip() == "MOTION":
            in_motion = True; out.append(line); continue
        m = re.match(r"(\s*)(ROOT|JOINT)\s+(\S+)", line)
        if m:
            name = m.group(3)
            if drop_fingers and "Finger" in name and depth_drop is None:
                depth_drop = depth
            if depth_drop is None:
                out.append(f"{m.group(1)}{m.group(2)} {RENAME.get(name, name)}")
            continue
        cm = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if cm:
            n = int(cm.group(1)); keep_cols += [depth_drop is None] * n
        if "{" in line: depth += 1
        if "}" in line:
            depth -= 1
            if depth_drop is not None and depth == depth_drop:
                depth_drop = None; continue
        if depth_drop is None:
            out.append(line)
    open(dst, "w").write("\n".join(out) + "\n")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as h:
        for chunk in iter(lambda: h.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_tree(src_dir, dst_dir, drop_fingers=False):
    """Rewrite every *.bvh in src_dir into dst_dir; write dst_dir/rename-summary.json."""
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".bvh"))
    if not files:
        raise SystemExit(f"no .bvh files under {src_dir}")
    results = []
    for f in files:
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)
        rewrite(src_path, dst_path, drop_fingers)
        results.append({"clip": f, "input_sha256": _sha256(src_path), "output_sha256": _sha256(dst_path)})
    summary = {"clips": files, "count": len(files), "drop_fingers": drop_fingers,
               "rename_map": RENAME, "output": os.path.abspath(dst_dir), "results": results}
    path = os.path.join(dst_dir, "rename-summary.json")
    with open(path, "w") as h:
        json.dump(summary, h, indent=1, sort_keys=True); h.write("\n")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rename Rokoko Newton bones to LAFAN1 names (file or directory).")
    ap.add_argument("src", help="a .bvh file or a directory of .bvh files")
    ap.add_argument("dst", help="output .bvh file, or output directory when src is a directory")
    ap.add_argument("--drop-fingers", action="store_true")
    a = ap.parse_args()
    if os.path.isdir(a.src):
        print("wrote", rewrite_tree(a.src, a.dst, a.drop_fingers))
    else:
        rewrite(a.src, a.dst, a.drop_fingers)
        print("wrote", a.dst)
