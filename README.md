# himrobot — humanoid table-tennis policy (HIM Arena)

Work toward a table-tennis policy for HIM Arena, built on the **Booster K1**
humanoid. Per HIM (the dataset owner): *"We built the capture rig and recorded
the data ourselves. Use it to train perception, movement, and control policies
for humanoid table tennis."*

## The three policy tracks

| Track | Learns | From | Status |
|---|---|---|---|
| **Perception** | ball / scene 3D (`ball_state`) | Orbbec RGB-D sessions | not started (needs camera intrinsics + a detector) |
| **Movement** | human swings → K1 22-joint reference motions | Rokoko mocap + paddle IMU | **first pass working** (IK retarget from paddle IMU) |
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
./.venv/bin/python sim/view_k1.py            # -> sim/k1_wave.gif (K1 waving)
./.venv/bin/python sim/view_k1.py --live     # interactive 3D viewer
./.venv/bin/python sim/retarget_imu.py       # -> sim/k1_retarget.gif (real swing on the K1)
~/.local/bin/him validate                    # validate the .him package
```

## The robot

HIM's "Adam" is built on the public **Booster K1** (BSD-3-Clause,
[github.com/BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets)):
22 joints (2 head, 4+4 arms, 6+6 legs). The runtime contract
(`.him/contracts/runtime.json`) is bound to this 22-joint layout; the canonical
joint order lives in its `extensions.robot` block.

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
