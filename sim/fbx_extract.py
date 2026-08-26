"""Extract per-frame arm-bone rotations from a Rokoko FBX clip.

Runs under the python3.11 side-venv with Blender-as-a-module (bpy), because the
FBX is binary and bpy is the reliable reader. For each frame it records each
bone's rotation *relative to its rest pose* (pose_bone.matrix_basis) as XYZ
euler radians — exactly the per-joint motion we retarget onto the K1.

Usage (note the `--` that separates bpy's args from ours):
    .venv-fbx/bin/python sim/fbx_extract.py -- <clip.fbx> <out.npz>
"""
import sys

import bpy
import numpy as np

# Right-arm chain + a little torso, in the standard Rokoko/Mixamo naming we
# confirmed via `strings` on the FBX.
BONES = ["Hips", "Spine2", "Spine4", "RightShoulder", "RightArm", "RightForeArm", "RightHand"]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    fbx_path, out_path = argv[0], argv[1]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.fbx(filepath=fbx_path)
    except (AttributeError, RuntimeError):
        bpy.ops.preferences.addon_enable(module="io_scene_fbx")
        bpy.ops.import_scene.fbx(filepath=fbx_path)

    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    present = [b for b in BONES if b in arm.pose.bones]
    scene = bpy.context.scene
    frames = list(range(scene.frame_start, scene.frame_end + 1))
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
