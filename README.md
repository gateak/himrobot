# himrobot — humanoid table-tennis policy (HIM Arena)

Work toward a table-tennis policy for HIM Arena, built on the **Unitree G1**
humanoid. Per HIM (the dataset owner): *"We built the capture rig and recorded
the data ourselves. Use it to train perception, movement, and control policies
for humanoid table tennis."*

## The three policy tracks

| Track | Learns | From | Status |
|---|---|---|---|
| **Perception** | ball / scene 3D (`ball_state`) | Orbbec RGB-D sessions | 2D gated orange-ball detection on the two 30 fps sessions on disk (26–32 % of 215350 frames carry a candidate), aligned depth attached per hit, ArUco `DICT_4X4_50` IDs 0–7 found at the half-table corners; **no intrinsics on disk → no metric XYZ yet** (`docs/PROJECT_KNOWLEDGE.md` §6a) |
| **Movement** | human swings → G1 reference motions | Rokoko mocap (FBX) | full body, **kinematic only** (GMR retarget of all 8 clips onto the 29-DoF G1 with a repo-local IK config; the previous CSVs had 50–69 cm mean wrist error / 23–31 % joint-limit saturation, the corrected config gives ~10–12 cm mean wrist error — `work/retarget/adjust2_rotfix_h1p75.md`; retarget stage reproduced under the pipeline contract; full `him pipeline run` not re-executed since the `--ik_config` change) |
| **Control** | `joints + ball → desired joints` (the deployable policy) | imitation of movement demos + RL in sim | scaffolded (BC trainer + aligned contract) |

Pipeline: `perception (ball) + movement (swings) → control (imitate + RL in sim) → him_adapter package → arena submit`.

## Layout

```
him_adapter/     Deployment wrapper HIM runs: stdio policy server.
                 HoldPolicy (safe fallback) + BCPolicy (loads a trained checkpoint).
.him/            HIM overlay: manifest, runtime contract (bound to the G1's 29 joints, Menagerie/SDK order),
                 fixtures, smoke test, package lock.
training/        PyTorch behavior-cloning scaffold (config, model, dataset, train).
sim/             Local MuJoCo tools (no HIM servers, un-gated):
                 fbx_to_bvh.py / rokoko_to_lafan1_bvh.py / gmr_bvh_to_g1_headless.py --ik_config
                                  – full-body mocap → G1 (29 DoF) retarget pipeline
                                    (ik_configs/bvh_lafan1_rokoko_to_g1.json = corrected rotation offsets)
                 retarget_report.py – wrist-error / joint-saturation report for a retargeted set
                 play_g1_csv.py   – play a retargeted clip live or render it to MP4
                 ball_detect.py / ball_3d.py / aruco_scan.py
                                  – 2D orange-ball detector, aligned-depth lookup, table ArUco scan
                 align_streams.py – IMU ↔ Rokoko ↔ Orbbec clock alignment
                 view_k1.py       – watch the K1 move (GIF or --live viewer)
                 k1_model.py      – load K1 with a paddle welded to the right hand
                 retarget_imu.py  – movement retarget: real paddle IMU → K1 swing
scripts/         fetch_assets.sh  – re-clone the vendored robot model
```

## Setup

```bash
python3.12 -m venv .venv           # /usr/local/bin/python3.12 on this machine
./.venv/bin/python -m pip install torch mujoco imageio imageio-ffmpeg numpy pytest
./scripts/fetch_assets.sh          # K1 (booster_assets) + sparse mujoco_menagerie/unitree_g1 -> third_party/
```

`data/` is gitignored and holds symlinks to the licensed capture on this
machine (`~/him-data/...`; `data/g1` → `work/g1`, the corrected retargeted G1 CSVs);
see `docs/PROJECT_KNOWLEDGE.md` §3 for the exact layout. Reading the FBX mocap
needs the Blender binary (5.2.1 LTS at `/Applications/Blender.app`, ships its own
Python + numpy) — no `bpy` wheel or side-venv.

## Try it

```bash
./scripts/fetch_assets.sh                    # fetch the K1 + G1 robot models
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
$BLENDER -b --python sim/fbx_extract.py -- data/rokoko-motion-capture/Clip_4.fbx data/mocap_clip4.npz
./.venv/bin/python sim/g1_retarget.py        # -> sim/g1_retarget.gif (real human swing on the G1)
./.venv/bin/python sim/view_k1.py            # -> sim/k1_wave.gif (add --live for the interactive viewer)
./.venv/bin/python sim/retarget_imu.py       # -> sim/k1_retarget.gif (paddle IMU orientation -> K1 arm)
~/.local/bin/him validate                    # validate the .him package

# Full-body retarget of every Rokoko clip onto the G1, then a preview MP4 (needs Blender,
# a GMR venv and a MuJoCo venv; see .him/commands/README.md):
~/.local/bin/him pipeline run .him/contracts/training-pipeline.json --yes \
  --set gmr_python=~/him-tools/gmr-venv/bin/python --set mujoco_python=~/him-tools/mujoco-venv/bin/python
~/him-tools/mujoco-venv/bin/python sim/play_g1_csv.py work/g1/Clip_2.csv   # live viewer
```

## The robot

**Current target: Unitree G1** (29-DOF humanoid, 7-DOF arm *with wrist* — from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)). The
7-DoF arm lets the retarget place the wrist: with the corrected IK config the
mean wrist error is ~10–12 cm per clip, versus 50–69 cm (and 23–31 % of
joint-frames at IK limits) with the upstream offsets
(`work/retarget/adjust2_rotfix_h1p75.md`). That is a kinematic measure only —
nothing has been checked for dynamic feasibility.

Earlier reference: HIM's "Adam" = the public **Booster K1** (BSD-3-Clause,
[booster_assets](https://github.com/BoosterRobotics/booster_assets)), 22 joints.
The runtime contract (`.him/contracts/runtime.json`) is bound to the G1's 29
joints in Menagerie/Unitree-SDK order (`extensions.robot.joint_order`).

## Not in this repo (gitignored)

- `data/` — HIM's proprietary licensed capture (paddle IMU + Orbbec RGB-D + Rokoko mocap + the verified release tars), kept as real files inside the repo folder. Not redistributed.
- `third_party/` — vendored Booster assets; re-fetch with `scripts/fetch_assets.sh`.
- `models/`, `*.pt`, `sim/*.gif` — generated artifacts.

## Honest caveats

- The contract keeps the HIM template's signal names / 50 Hz clock; HIM's Adam
  manifest uses different names @ 60 Hz. **Reconcile with HIM before deploying** —
  the wire format the server-side evaluator expects is not confirmed.
- The retarget is kinematic (poses only, no dynamics/balance); the older
  `sim/retarget_imu.py` uses only paddle orientation and is not world-calibrated.
- The three capture streams were recorded on separate clocks; IMU↔Rokoko↔Orbbec
  alignment is done for the sessions on disk (PROJECT_KNOWLEDGE.md §6b) and still
  open for the six Orbbec rally sessions whose tars failed to download.
- HIM's hosted simulation/execution is gated server-side (per-package worker);
  local MuJoCo here is the un-gated way to see the robot move.
