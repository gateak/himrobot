"""Extract per-frame arm-bone rotations from a Rokoko FBX clip.

Runs inside the Blender binary (headless), because the FBX is binary and
Blender's importer is the reliable reader. Blender ships its own Python + numpy,
so no venv is needed. For each frame it records each bone's rotation *relative
to its rest pose* (pose_bone.matrix_basis) as XYZ euler radians — exactly the
per-joint motion we retarget onto the robot.

Usage (note the `--` that separates Blender's args from ours):
    /Applications/Blender.app/Contents/MacOS/Blender -b --python sim/fbx_extract.py -- <clip.fbx> <out.npz>
"""
import sys

import bpy
import numpy as np

# Right-arm chain + a little torso, in the standard Rokoko/Mixamo naming we
# confirmed via `strings` on the FBX.
BONES = ["Hips", "Spine2", "Spine4", "RightShoulder", "RightArm", "RightForeArm", "RightHand"]


def main():
    if "--" not in sys.argv:
        sys.exit("usage: Blender -b --python sim/fbx_extract.py -- <clip.fbx> <out.npz>  (missing '--' separator)")
    argv = sys.argv[sys.argv.index("--") + 1:]
    if len(argv) != 2:
        sys.exit("usage: Blender -b --python sim/fbx_extract.py -- <clip.fbx> <out.npz>")
    fbx_path, out_path = argv

    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.fbx(filepath=fbx_path)
    except (AttributeError, RuntimeError):
        bpy.ops.preferences.addon_enable(module="io_scene_fbx")
        bpy.ops.import_scene.fbx(filepath=fbx_path)

    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    present = [b for b in BONES if b in arm.pose.bones]
    scene = bpy.context.scene
    # The FBX importer does not set the scene range (stays 1..250); take the
    # clip length from the armature's action, like fbx_to_bvh.py does.
    action = arm.animation_data.action if arm.animation_data else None
    if action is not None:
        start, end = int(action.frame_range[0]), int(action.frame_range[1])
    else:
        start, end = scene.frame_start, scene.frame_end
    frames = list(range(start, end + 1))
    fps = scene.render.fps / max(1, scene.render.fps_base)

    # World-space arm landmarks (for robust position-based retargeting):
    # reach where the human's arm reached, rather than copying raw euler.
    LM = [("Hips", "head"), ("Spine4", "head"), ("RightArm", "head"),   # ..., shoulder
          ("RightForeArm", "head"), ("RightHand", "head"), ("RightHand", "tail")]  # elbow, wrist, tip
    LM = [(b, h) for (b, h) in LM if b in arm.pose.bones]
    lm_names = [f"{b}.{h}" for b, h in LM]

    euler = np.zeros((len(frames), len(present), 3), dtype=np.float32)
    pos = np.zeros((len(frames), len(LM), 3), dtype=np.float32)
    for fi, f in enumerate(frames):
        scene.frame_set(f)
        mw = arm.matrix_world
        for bi, bn in enumerate(present):
            e = arm.pose.bones[bn].matrix_basis.to_euler("XYZ")
            euler[fi, bi] = (e.x, e.y, e.z)
        for li, (b, h) in enumerate(LM):
            pb = arm.pose.bones[b]
            p = mw @ (pb.head if h == "head" else pb.tail)
            pos[fi, li] = (p.x, p.y, p.z)

    np.savez(out_path, bones=np.array(present), euler=euler,
             landmarks=np.array(lm_names), pos=pos, fps=float(fps))
    print(f"[fbx] {fbx_path}")
    print(f"[fbx] armature '{arm.name}', {len(frames)} frames @ {fps:.1f} fps")
    print(f"[fbx] bones: {present}")
    print(f"[fbx] landmarks: {lm_names}")
    print(f"[fbx] wrote {out_path}  euler{euler.shape} pos{pos.shape}")


if __name__ == "__main__":
    main()
