#!/usr/bin/env python3
"""Headless GMR retarget: LAFAN1-named BVH -> Unitree G1 29-DoF.

Single clip:   --bvh_file clip.bvh --out out/clip      -> out/clip.{pkl,csv}
Directory:     --bvh_dir bvh_lafan1 --out_dir out/g1   -> out/g1/<clip>.{pkl,csv}
Both modes write <dir>/gmr_g1_joint_order.txt (GMR dof order, for play_g1_csv.py);
directory mode also writes <dir>/retarget-summary.json.
Same pipeline as GMR/sim/bvh_to_robot.py minus the RobotMotionViewer.
"""
import argparse, json, os, pickle
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from tqdm import tqdm


def retarget(bvh_file, out, fmt, robot, fps, max_frames=None):
    frames, height = load_bvh_file(bvh_file, format=fmt)
    if max_frames:
        frames = frames[:max_frames]
    rt = GMR(src_human=f"bvh_{fmt}", tgt_robot=robot, actual_human_height=height)
    qs = np.array([rt.retarget(f) for f in tqdm(frames, desc=os.path.basename(bvh_file))])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    root_pos, root_rot, dof = qs[:, :3], qs[:, 3:7][:, [1, 2, 3, 0]], qs[:, 7:]  # quat -> xyzw
    with open(out + ".pkl", "wb") as h:
        pickle.dump({"fps": fps, "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof,
                     "local_body_pos": None, "link_body_list": None}, h)
    np.savetxt(out + ".csv", np.hstack([root_pos, root_rot, dof]), delimiter=",")
    names = sorted(rt.robot_motor_names, key=rt.robot_motor_names.get)
    with open(os.path.join(os.path.dirname(os.path.abspath(out)), "gmr_g1_joint_order.txt"), "w") as h:
        h.write("\n".join(names) + "\n")
    return {"clip": os.path.basename(bvh_file), "frames": int(len(qs)), "dof": int(dof.shape[1]),
            "human_height_m": float(height), "csv": out + ".csv"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bvh_file"); ap.add_argument("--out")
    ap.add_argument("--bvh_dir"); ap.add_argument("--out_dir")
    ap.add_argument("--format", default="lafan1"); ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--fps", type=int, default=30); ap.add_argument("--max_frames", type=int, default=None)
    a = ap.parse_args()
    if a.bvh_dir:
        if not a.out_dir:
            ap.error("--bvh_dir requires --out_dir")
        files = sorted(f for f in os.listdir(a.bvh_dir) if f.endswith(".bvh"))
        if not files:
            ap.error(f"no .bvh files under {a.bvh_dir}")
        results = [retarget(os.path.join(a.bvh_dir, f), os.path.join(a.out_dir, f[:-4]), a.format, a.robot, a.fps, a.max_frames) for f in files]
        summary = {"robot": a.robot, "format": a.format, "fps": a.fps, "clips": results,
                   "total_frames": sum(r["frames"] for r in results), "dof": results[0]["dof"]}
        path = os.path.join(a.out_dir, "retarget-summary.json")
        with open(path, "w") as h:
            json.dump(summary, h, indent=1, sort_keys=True); h.write("\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "clips"}))
    elif a.bvh_file and a.out:
        r = retarget(a.bvh_file, a.out, a.format, a.robot, a.fps, a.max_frames)
        print("frames", r["frames"], "dof", r["dof"], "->", r["csv"])
    else:
        ap.error("use --bvh_file/--out or --bvh_dir/--out_dir")


if __name__ == "__main__":
    main()
