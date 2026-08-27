# himrobot — humanoid table-tennis policy (HIM Arena)

Work toward a table-tennis policy for HIM Arena, built on the **Booster K1**
humanoid. Per HIM (the dataset owner): *"We built the capture rig and recorded
the data ourselves. Use it to train perception, movement, and control policies
for humanoid table tennis."*

## The three policy tracks

| Track | Learns | From | Status |
|---|---|---|---|
| **Perception** | ball / scene 3D (`ball_state`) | Orbbec RGB-D sessions | not started (needs camera intrinsics + a detector) |
| **Movement** | human swings → G1 reference motions | Rokoko mocap (FBX) | **working, full body** (GMR retarget of all 8 clips onto the 29-DoF G1; `him pipeline run` verified) |
| **Control** | `joints + ball → desired joints` (the deployable policy) | imitation of movement demos + RL in sim | scaffolded (BC trainer + aligned contract) |

Pipeline: `perception (ball) + movement (swings) → control (imitate + RL in sim) → him_adapter package → arena submit`.

## Layout

```
him_adapter/     Deployment wrapper HIM runs: stdio policy server.
                 HoldPolicy (safe fallback) + BCPolicy (loads a trained checkpoint).
.him/            HIM overlay: manifest, runtime contract (aligned to K1's 22 joints),
                 fixtures, smoke test, package lock.
training/        PyTorch behavior-cloning scaffold (config, model, dataset, train).
sim/             Local MuJoCo tools (no HIM servers, un-gated):
                 fbx_to_bvh.py / rokoko_to_lafan1_bvh.py / gmr_bvh_to_g1_headless.py
                                  – full-body mocap → G1 (29 DoF) retarget pipeline
                 play_g1_csv.py   – play a retargeted clip live or render it to MP4
                 view_k1.py       – watch the K1 move (GIF or --live viewer)
                 k1_model.py      – load K1 with a paddle welded to the right hand
                 retarget_imu.py  – movement retarget: real paddle IMU → K1 swing
scripts/         fetch_assets.sh  – re-clone the vendored robot model
```

## Setup

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install torch mujoco imageio numpy pytest
./scripts/fetch_assets.sh          # clones the Booster K1 model into third_party/
```

## Try it

```bash
./scripts/fetch_assets.sh                    # fetch the K1 + G1 robot models
./.venv/bin/python sim/g1_retarget.py        # -> sim/g1_retarget.gif (real human swing on the G1)
./.venv/bin/python sim/view_k1.py --live     # interactive 3D viewer (K1)
~/.local/bin/him validate                    # validate the .him package

# Full-body retarget of every Rokoko clip onto the G1, then a preview MP4 (needs Blender,
# a GMR venv and a MuJoCo venv; see .him/commands/README.md):
~/.local/bin/him pipeline run .him/contracts/training-pipeline.json --yes \
  --set gmr_python=~/him-tools/gmr-venv/bin/python --set mujoco_python=~/him-tools/mujoco-venv/bin/python
~/him-tools/mujoco-venv/bin/python sim/play_g1_csv.py work/g1/Clip_2.csv   # live viewer

# Reading the FBX mocap needs the python3.11 side-venv with Blender-as-a-module:
python3.11 -m venv .venv-fbx && .venv-fbx/bin/pip install bpy numpy
.venv-fbx/bin/python sim/fbx_extract.py -- <clip.fbx> data/mocap_clip4.npz
```

## The robot

**Current target: Unitree G1** (29-DOF humanoid, 7-DOF arm *with wrist* — from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)). The
wrist is why the retargeted swing looks natural.

Earlier reference: HIM's "Adam" = the public **Booster K1** (BSD-3-Clause,
[booster_assets](https://github.com/BoosterRobotics/booster_assets)), 22 joints.
The runtime contract (`.him/contracts/runtime.json`) is bound to the G1's 29
joints in Menagerie/Unitree-SDK order (`extensions.robot.joint_order`).

## Not in this repo (gitignored)

- `data/` — HIM's proprietary licensed capture (paddle IMU + Orbbec RGB-D + Rokoko mocap). Not redistributed.
- `third_party/` — vendored Booster assets; re-fetch with `scripts/fetch_assets.sh`.
- `models/`, `*.pt`, `sim/*.gif` — generated artifacts.

## Honest caveats

- The contract keeps the HIM template's signal names / 50 Hz clock; HIM's Adam
  manifest uses different names @ 60 Hz. **Reconcile with HIM before deploying** —
  the wire format the server-side evaluator expects is not confirmed.
- The movement retarget uses only paddle **orientation** (the readable stream)
  and is not world-calibrated; the body mocap (FBX) needs Blender to unlock.
- The three capture streams were recorded on separate clocks; they need
  time-alignment before the control track can pair ball state with arm motion.
- HIM's hosted simulation/execution is gated server-side (per-package worker);
  local MuJoCo here is the un-gated way to see the robot move.
