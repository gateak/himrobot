#!/usr/bin/env python3
"""Headless GMR retarget: LAFAN1-named BVH -> Unitree G1 29-DoF.

Single clip:   --bvh_file clip.bvh --out out/clip      -> out/clip.{pkl,csv}
Directory:     --bvh_dir bvh_lafan1 --out_dir out/g1   -> out/g1/<clip>.{pkl,csv}
Both modes write <dir>/gmr_g1_joint_order.txt (GMR dof order, for play_g1_csv.py);
directory mode also writes <dir>/retarget-summary.json.
Same pipeline as GMR/sim/bvh_to_robot.py minus the RobotMotionViewer.

Options that do not touch the upstream GMR checkout:
  --human_height H   passed as GMR's actual_human_height (default: loader's hard-coded
                     1.75 m); GMR multiplies human_scale_table by H / ik_config[
                     "human_height_assumption"] (1.8 for lafan1->g1).
  --ik_config PATH   repo-local IK config JSON used instead of GMR's
                     ik_configs/bvh_<format>_to_<robot>.json (injected into
                     general_motion_retargeting.params.IK_CONFIG_DICT at runtime).
  --jobs N           retarget N clips in parallel (directory mode).
"""
import argparse, json, os, pickle
from multiprocessing import Pool
import numpy as np


def _write_joint_order(out_dir, names):
    with open(os.path.join(out_dir, "gmr_g1_joint_order.txt"), "w") as h:
        h.write("\n".join(names) + "\n")


def retarget(bvh_file, out, fmt, robot, fps, max_frames=None, human_height=None, ik_config=None, progress=True,
             write_order=True):
    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    frames, loader_height = load_bvh_file(bvh_file, format=fmt)
    height = loader_height if human_height is None else float(human_height)
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"{bvh_file}: 0 frames to retarget (clip empty or --max_frames {max_frames} truncated it to none)")
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import params
    if ik_config:
        params.IK_CONFIG_DICT[f"bvh_{fmt}"][robot] = os.path.abspath(ik_config)
    rt = GMR(src_human=f"bvh_{fmt}", tgt_robot=robot, actual_human_height=height, verbose=False)
    it = frames
    if progress:
        from tqdm import tqdm
        it = tqdm(frames, desc=os.path.basename(bvh_file))
    qs = np.array([rt.retarget(f) for f in it])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    root_pos, root_rot, dof = qs[:, :3], qs[:, 3:7][:, [1, 2, 3, 0]], qs[:, 7:]  # quat -> xyzw
    with open(out + ".pkl", "wb") as h:
        pickle.dump({"fps": fps, "root_pos": root_pos, "root_rot": root_rot, "dof_pos": dof,
                     "local_body_pos": None, "link_body_list": None}, h)
    np.savetxt(out + ".csv", np.hstack([root_pos, root_rot, dof]), delimiter=",")
    names = sorted(rt.robot_motor_names, key=rt.robot_motor_names.get)
    if write_order:
        _write_joint_order(os.path.dirname(os.path.abspath(out)), names)
    return {"clip": os.path.basename(bvh_file), "frames": int(len(qs)), "dof": int(dof.shape[1]), "joint_order": names,
            "human_height_m": float(height), "ik_config": os.path.abspath(ik_config) if ik_config else "gmr-default",
            "human_scale_table": {k: float(v) for k, v in rt.human_scale_table.items()}, "csv": out + ".csv"}


def _job(args):
    return retarget(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bvh_file"); ap.add_argument("--out")
    ap.add_argument("--bvh_dir"); ap.add_argument("--out_dir")
    ap.add_argument("--format", default="lafan1"); ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--fps", type=int, default=30); ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--human_height", type=float, default=None)
    ap.add_argument("--ik_config", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    if a.bvh_dir:
        if not a.out_dir:
            ap.error("--bvh_dir requires --out_dir")
        files = sorted(f for f in os.listdir(a.bvh_dir) if f.endswith(".bvh"))
        if not files:
            ap.error(f"no .bvh files under {a.bvh_dir}")
        jobs = [(os.path.join(a.bvh_dir, f), os.path.join(a.out_dir, f[:-4]), a.format, a.robot, a.fps,
                 a.max_frames, a.human_height, a.ik_config, a.jobs == 1, False) for f in files]
        if a.jobs > 1:
            with Pool(a.jobs) as pool:
                results = pool.map(_job, jobs)
        else:
            results = [_job(j) for j in jobs]
        _write_joint_order(a.out_dir, results[0]["joint_order"])  # once, from the parent (not per worker)
        for r in results:
            del r["joint_order"]
        summary = {"robot": a.robot, "format": a.format, "fps": a.fps, "clips": results,
                   "human_height_m": results[0]["human_height_m"], "ik_config": results[0]["ik_config"],
                   "human_scale_table": results[0]["human_scale_table"],
                   "total_frames": sum(r["frames"] for r in results), "dof": results[0]["dof"]}
        path = os.path.join(a.out_dir, "retarget-summary.json")
        with open(path, "w") as h:
            json.dump(summary, h, indent=1, sort_keys=True); h.write("\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "clips"}))
    elif a.bvh_file and a.out:
        r = retarget(a.bvh_file, a.out, a.format, a.robot, a.fps, a.max_frames, a.human_height, a.ik_config)
        print("frames", r["frames"], "dof", r["dof"], "height", r["human_height_m"], "->", r["csv"])
    else:
        ap.error("use --bvh_file/--out or --bvh_dir/--out_dir")


if __name__ == "__main__":
    main()
