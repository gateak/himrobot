# himrobot — humanoid table-tennis policy (HIM Arena)

This repository contains work toward a table-tennis policy for HIM Arena. The
policy uses the **Unitree G1** humanoid robot. HIM is the source of the
dataset. HIM tells us this: *"We built the capture rig and recorded
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

Git does not track `data/`. This folder contains symlinks to the licensed
capture on this machine, at `~/him-data/...`. The symlink `data/g1` points to
`work/g1`, the corrected, retargeted G1 CSV files. `docs/PROJECT_KNOWLEDGE.md`,
§3, gives the exact layout.

The Blender binary (version 5.2.1 LTS, at `/Applications/Blender.app`) is
necessary to read the FBX mocap. This copy of Blender includes Python and
numpy. A `bpy` wheel or a different venv is not necessary.

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

## HIM Arena entrypoints

Besides the stdio `inference` server, `.him/him.yaml` declares `evaluate`
(`python -m tt_sim.evaluate ...`) and `train` (`python -m tt_sim.rl.ppo
--residual --residual-scale 0.2 --vel-penalty 1e-5 --legal-bonus 10
--n-envs 64 --minutes 420 --eval-every-steps 100000`), both `execution_mode:
python` against the `tt-runtime.json` contract. `train` omits `--out`: when
run under HIM, `tt_sim/rl/ppo.py` writes checkpoints (`best.pt`, `final.pt`)
and `train_log.csv` to `$HIM_CANDIDATE_OUTPUT`; run by hand with no `--out`
and no `HIM_CANDIDATE_OUTPUT` set, it still defaults to `work/tt/rl/run` as
before.

## The robot

This project uses the **Unitree G1** as the target robot (29-DOF humanoid,
7-DOF arm *with wrist*, from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)). The
7-DoF arm lets the retarget place the wrist. With the corrected IK config, the
mean wrist error is approximately 10–12 cm for each clip. With the upstream
offsets, the mean wrist error is 50–69 cm, and 23–31 % of joint-frames are at
IK limits (`work/retarget/adjust2_rotfix_h1p75.md`). This measurement is only
kinematic. A check of dynamic feasibility is necessary.

The previous target robot was HIM's "Adam," the open-source **Booster K1**
(BSD-3-Clause license,
[booster_assets](https://github.com/BoosterRobotics/booster_assets)), 22
joints. The runtime contract (`.him/contracts/runtime.json`) uses the G1's 29
joints, in Menagerie and Unitree-SDK order (`extensions.robot.joint_order`).

## Not in this repo (gitignored)

- `data/` — HIM's licensed capture data (paddle IMU, Orbbec RGB-D, Rokoko
  mocap, and the verified release tars). This data is on disk inside the
  repository folder. This repository does not redistribute the data.
- `third_party/` — vendored Booster assets. `scripts/fetch_assets.sh` fetches
  them again.
- `models/`, `*.pt`, `sim/*.gif` — generated artifacts.

## Honest caveats

- The contract keeps the HIM template's signal names and its 50 Hz clock.
  HIM's Adam manifest uses different names, at 60 Hz. **Agree with HIM on the
  correct format before you deploy the policy.** No one knows the wire format
  that the server-side evaluator must use.
- The retarget is kinematic. It gives only poses, not dynamics or balance. The
  `sim/retarget_imu.py` script is a previous version. It uses only paddle
  orientation, and it is not world-calibrated.
- HIM recorded the three capture streams on different clocks. The team aligned
  the IMU, Rokoko, and Orbbec streams for all eight Orbbec sessions on disk
  (PROJECT_KNOWLEDGE.md §6b).
- A per-package worker on the HIM side gates HIM's hosted simulation and
  execution. The local MuJoCo tool here is the un-gated procedure to see the
  robot move.
