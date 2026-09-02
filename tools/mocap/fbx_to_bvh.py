"""Convert Rokoko Studio FBX exports to BVH using a headless Blender.

Usage: python tools/mocap/fbx_to_bvh.py --input <file.fbx|dir> --output <dir> [--blender PATH]

Writes <clip>.bvh and <clip>.skeleton.json (bone names, fps, frame range) per clip.
Pure stdlib; Blender does the parsing so no FBX SDK is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"

BPY_SCRIPT = r'''
import bpy, json, sys
argv = sys.argv[sys.argv.index("--") + 1:]
src, bvh_out, meta_out = argv
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src)
scene = bpy.context.scene
arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
if not arms:
    raise SystemExit("no armature found in " + src)
arm = arms[0]
action = arm.animation_data.action if arm.animation_data else None
if action is None:
    raise SystemExit("armature has no animation in " + src)
start, end = (int(action.frame_range[0]), int(action.frame_range[1]))
bpy.context.view_layer.objects.active = arm
for obj in bpy.data.objects:
    obj.select_set(obj is arm)
bpy.ops.export_anim.bvh(filepath=bvh_out, frame_start=start, frame_end=end,
                        rotate_mode="NATIVE", root_transform_only=False)
meta = {
    "source": src,
    "fps": scene.render.fps / scene.render.fps_base,
    "frame_start": start,
    "frame_end": end,
    "frames": end - start + 1,
    "armature": arm.name,
    "bones": [b.name for b in arm.data.bones],
    "blender_version": bpy.app.version_string,
}
json.dump(meta, open(meta_out, "w"), indent=1, sort_keys=True)
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(blender: str, src: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bvh = out_dir / (src.stem + ".bvh")
    meta = out_dir / (src.stem + ".skeleton.json")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(BPY_SCRIPT)
        script = handle.name
    try:
        completed = subprocess.run(
            [blender, "-b", "--python", script, "--", str(src), str(bvh), str(meta)],
            capture_output=True, text=True, check=False,
        )
    finally:
        os.unlink(script)
    if completed.returncode != 0 or not bvh.is_file() or not meta.is_file():
        sys.stderr.write(completed.stdout[-4000:] + completed.stderr[-4000:])
        raise SystemExit(f"Blender conversion failed for {src}")
    result = json.loads(meta.read_text())
    result["source_path"] = str(src)
    result["source_sha256"] = _sha256(src)
    result["source_size"] = src.stat().st_size
    result["bvh_sha256"] = _sha256(bvh)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    args = parser.parse_args()
    if not Path(args.blender).is_file():
        raise SystemExit(f"Blender not found at {args.blender}; set --blender or $BLENDER")
    sources = sorted(args.input.glob("*.fbx")) if args.input.is_dir() else [args.input]
    if not sources:
        raise SystemExit(f"no .fbx files under {args.input}")
    results = [convert(args.blender, src, args.output) for src in sources]
    per_clip = [{"clip": src.stem, "source_path": r["source_path"], "source_sha256": r["source_sha256"],
                 "source_size": r["source_size"], "frames": r["frames"], "fps": r["fps"],
                 "bvh_sha256": r["bvh_sha256"]} for src, r in zip(sources, results)]
    summary = {"clips": len(results), "frames": sum(r["frames"] for r in results),
               "fps": sorted({r["fps"] for r in results}), "output": str(args.output),
               "results": per_clip}
    (args.output / "conversion-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
