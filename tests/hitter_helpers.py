"""Shared helpers for the hitter parity and geometry tests.

tests/test_hitter_v2.py (the reach-expansion scan) and
tests/test_hitter_v3.py (the stance frame and the operational-band scan
under the collision guard) roll the same policies and classify the same kind of
contact. The rollout recipe, the pose-sampling recipe and the "which contacts
do we actually care about" filter live here so the two cannot drift apart.
"""
from __future__ import annotations

import mujoco
import numpy as np

from himrobot.sim import scene
from himrobot.policies.ik_hitter import IK_INDICES
from himrobot.sim.scene import DEFAULT_PARAMS, TableTennisEnv, sample_serve


def roll_actions(policy, seed=0, steps=150):
    """One serve through a policy; returns the stacked per-step actions."""
    env = TableTennisEnv(seed=seed, params={})
    serve = sample_serve(np.random.default_rng(seed), DEFAULT_PARAMS)
    obs = env.reset(serve=serve)
    policy.reset({"serve": serve, "params": DEFAULT_PARAMS})
    actions = []
    for _ in range(steps):
        action = policy.act(obs)
        actions.append(np.asarray(action, dtype=np.float64).copy())
        obs, _events = env.step(action)
    return np.stack(actions)

# A pre-existing model artifact, not a reachability limit: these two wrist
# links sit two joints apart (roll -> pitch -> yaw), so MuJoCo does not filter
# their contact, and they touch over y in [-0.25, 0.15], z in [0.90, 1.25] at
# x=-1.20 with the v1 hitter at nominal.
WRIST_ARTIFACT_PAIR = frozenset(("right_wrist_roll_link", "right_wrist_yaw_link"))

# The passive left arm -- held at STAND_POSE, off the IK chain, and unable to
# reach the paddle -- brushes the hips when the waist bends toward a far reach.
# Benign, pre-existing, and not what any of these tests is about. The v2 filter
# excused only left_wrist_* against left_hip_pitch_link; the operational-band
# scan (work/tt/design/hitter_v3_reach.md 2.4) found the same family one link
# over -- left_elbow_link against the hips, left_wrist_* against
# left_hip_roll_link / right_hip_roll_link -- on 109-276 poses per plane, which
# would otherwise dominate the v3 scan. So the filter names the whole family.
IDLE_LEFT_ARM = frozenset((
    "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "left_elbow_link",
))
HIP_LINKS = frozenset((
    "pelvis",
    "left_hip_roll_link", "left_hip_pitch_link", "left_hip_yaw_link",
    "right_hip_roll_link", "right_hip_pitch_link", "right_hip_yaw_link",
))


def benign_contact(pair) -> bool:
    """True for a contact that is a known model artifact rather than the right
    arm/torso/table interference these tests exist to catch."""
    if frozenset(pair) == WRIST_ARTIFACT_PAIR:
        return True
    a, b = pair
    return (a in IDLE_LEFT_ARM and b in HIP_LINKS) or (b in IDLE_LEFT_ARM and a in HIP_LINKS)


# The face normals the templates and the pitch/yaw offsets can command.
NORMALS = ((1.0, 0.0, 0.2), (1.0, 0.0, 0.7), (1.0, 0.0, -0.3),
           (1.0, 0.35, 0.2), (1.0, -0.35, 0.2))


def right_arm_collides(pairs) -> bool:
    """True when any contact is the right arm (or the paddle) against something
    that is not the idle left arm -- i.e. the arm driven through the torso.
    Matches work/tt/analysis/reach/v3_guard.py's `is_right`, so the counts the
    design measured are the counts this reproduces."""
    for pair in pairs:
        if any(s.startswith(("left_wrist", "left_elbow", "left_shoulder")) for s in pair):
            continue
        if any(s.startswith("right_") for s in pair) or "paddle" in pair:
            return True
    return False


def ik_pose_at(hitter, data, x, y, z, normal=(1.0, 0.0, 0.2)):
    """Warm-started from stand exactly as the runtime solver is, then forward
    kinematics. Returns (q29, [(body_a, body_b), ...] for every contact)."""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    q = scene.STAND_POSE.copy()
    for _ in range(8):
        q[IK_INDICES] = hitter._solve_ik(q, np.array([x, y, z]), normal)
    data.qpos[hitter._all_qpos_adr] = q
    mujoco.mj_forward(hitter._ik_model, data)
    model = hitter._ik_model
    pairs = []
    for c in data.contact[:data.ncon]:
        body = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])  # noqa: E731
        pairs.append((body(c.geom1), body(c.geom2)))
    return q, pairs
