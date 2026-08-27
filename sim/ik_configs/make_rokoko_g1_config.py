#!/usr/bin/env python3
"""Derive a GMR IK config for Blender-exported Rokoko BVH -> Unitree G1.

GMR's bvh_lafan1_to_g1.json rotation offsets assume the LAFAN1 bone-frame
convention. Blender's BVH export of the Rokoko "Newton" skeleton has identity
rest rotations for every bone (frame 0 = T-pose, all rotation channels ~0), so
after GMR's Y-up -> Z-up conversion every human bone is oriented Rx(90 deg) in
the T-pose and faces -y (toes point +z_bvh -> -y). The G1 at qpos=0 faces +x.

For every robot body in the IK tables the offset is therefore
    R_offset = Rx(90)^T . Rz(-90) . R_body(q_T)
where q_T is the G1's T-pose (qpos = 0 except right_shoulder_roll = -pi/2,
left_shoulder_roll = +pi/2, both elbows = +pi/2, all read off the model), so
that target = R_human . R_offset reproduces the robot T-pose when the human is
in the T-pose. Weights, position offsets, scale table and height assumption are
copied unchanged from the upstream config.

  ~/him-tools/gmr-venv/bin/python sim/ik_configs/make_rokoko_g1_config.py \
      [--src ~/him-tools/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json] \
      [--model ~/him-tools/GMR/assets/unitree_g1/g1_mocap_29dof.xml] \
      [--out sim/ik_configs/bvh_lafan1_rokoko_to_g1.json]
"""
import argparse, json, os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/him-tools/GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json")
MODEL = os.path.expanduser("~/him-tools/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
T_POSE = {"right_shoulder_roll_joint": -np.pi / 2, "left_shoulder_roll_joint": np.pi / 2,
          "right_elbow_joint": np.pi / 2, "left_elbow_joint": np.pi / 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=SRC); ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=os.path.join(HERE, "bvh_lafan1_rokoko_to_g1.json"))
    a = ap.parse_args()
    cfg = json.load(open(a.src))
    m = mujoco.MjModel.from_xml_path(a.model); d = mujoco.MjData(m)
    d.qpos[:] = 0; d.qpos[3] = 1
    for j, v in T_POSE.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
    mujoco.mj_kinematics(m, d)
    fix = R.from_euler("x", 90, degrees=True).inv() * R.from_euler("z", -90, degrees=True)
    derived = {}
    for table in ("ik_match_table1", "ik_match_table2"):
        for body, entry in cfg[table].items():
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            Rb = R.from_matrix(d.xmat[bid].reshape(3, 3))
            q = (fix * Rb).as_quat(scalar_first=True)
            entry[4] = [round(float(x), 8) for x in q]
            derived[body] = entry[4]
    # T-pose sanity: right wrist should sit straight out to the robot's right (-y) of the shoulder
    b = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)]
    arm = b("right_wrist_yaw_link") - b("right_shoulder_pitch_link")
    cfg["_derived_by"] = os.path.relpath(os.path.abspath(__file__), os.path.dirname(HERE))
    cfg["_source"] = "GMR general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json"  # repo-neutral, not the local path
    cfg["_robot_model"] = "GMR assets/unitree_g1/g1_mocap_29dof.xml"
    cfg["_robot_t_pose_right_arm_vector"] = [round(float(x), 4) for x in arm]
    with open(a.out, "w") as h:
        json.dump(cfg, h, indent=1); h.write("\n")
    print("wrote", a.out); print("robot T-pose right arm vector (shoulder->wrist):", arm.round(4))
    for k, v in derived.items():
        print(f"  {k:26s} {v}")


if __name__ == "__main__":
    main()
