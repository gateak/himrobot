# Project knowledge — himrobot (HIM humanoid table tennis)

Canonical, version-controlled project knowledge. Keep this current as decisions
and findings land — this is the source of truth (not any external notes).
_Last updated: 2026-08-26._

---

## 1. Goal & HIM's intent

Build a table-tennis policy for **HIM Arena** on a humanoid robot. Per HIM (the
dataset owner): *"We built the capture rig and recorded the data ourselves. Use
it to train perception, movement, and control policies for humanoid table
tennis."* So the multimodal human capture **is the intended training set** —
not a mismatch to work around.

## 2. Current status & next steps

| Track | Meaning | Status |
|---|---|---|
| **Movement** | human swings → robot reference motions | 🟢 **working on the G1** (position retarget, ~1.4 cm wrist error) |
| **Perception** | ball / scene 3D (`ball_state`) from cameras | ⚪ not started (needs Orbbec intrinsics + a ball detector) |
| **Control** | `joints + ball → desired joints` (deployable hitter) | 🟡 scaffolded; blueprint = PPO in sim + ball predictor + mocap refs |

**Pending / loose ends**
- Migrate `.him/contracts/runtime.json` from the K1's **22** joints to the G1's **29**.
- Reconcile contract signal names / rate with HIM's server-side evaluator (unconfirmed).
- Time-align the three capture streams (separate clocks) before the control track.
- Compute at scale is gated (see §4); local MuJoCo is the un-gated path.

## 3. Environment & tooling

- `him` CLI = `him_sdk 0.1.3` in a **Python 3.12** venv at `~/.local/share/him/venv` (binary `~/.local/bin/him`). Endpoint default is `http://127.0.0.1:5173` (dead) — **always** `export HIM_API_URL=https://arena.himrobotics.com`. Authenticated (account `him_pat_…`, creds at `~/.config/him/credentials.json` — do not read/commit).
- Main dev venv: `.venv/` (Python 3.12) — torch, mujoco 3.12, imageio, numpy, pytest.
- FBX side-venv: `.venv-fbx/` (**Python 3.11**) — `bpy 5.0.1` (Blender-as-a-module). Needed because `bpy` has no 3.12 wheel and the Rokoko FBX is **binary**.
- Vendored assets (git-ignored, re-fetch via `scripts/fetch_assets.sh`): `third_party/booster_assets` (K1), `third_party/mujoco_menagerie` (G1).
- `data/` (git-ignored) holds HIM's proprietary licensed capture — never commit.

## 4. HIM platform mechanics

- **Data:** `him data download` is hard-restricted to the **table-tennis** release only; no catalog, no robot obs→action dataset to pull. `capture`/`dataset`/`train` consume LOCAL fixtures you author (`him train rehearse` is a deterministic stub — no real training).
- **Simulator:** server-side only (no local sim). Run a policy against a sim bot via `him arena submit <challenge_version_id> --package <approved_pkg> --entrypoint <ep>` (Community challenge, isaac-sim; scored on legal-return/target-hit/landing-error/latency), or `him job launch simulation` (mujoco-simulation-contract, `mujoco_mjx`, gpu-standard) + `him job watch`. Output is **metrics/events, not video**.
- **Gating:** real package execution is blocked server-side by a per-package worker gate (`package_worker_disabled`). Our first `him run` **failed** (reconcile-timeout); a built-in fixture control run succeeded, proving the plumbing is healthy — real code needs HIM to enable the worker. Package `my-policy@0.1.0` id `pkg_110ce5d185fa9ba3d703`; failed job `job_6be550b37b3337434d9a`.
- **Approval gates (ask the user first):** `him login/run/push/publish`, `job launch`, `arena submit`, `dataset push` — anything that uploads source, spends credits, or launches compute.

## 5. The robot

- **Current target: Unitree G1** (user directive). 29-DOF, **7-DOF arm with wrist** (`right_shoulder_pitch/roll/yaw`, `right_elbow`, `right_wrist_roll/pitch/yaw`), hand at `right_wrist_yaw_link`. Model: `mujoco_menagerie/unitree_g1/g1.xml` (+ `g1_mjx.xml` matches HIM's `mujoco_mjx`). The wrist is why retargeting looks natural. G1 also matches our contract's signal names.
- **Earlier reference: Booster K1** = the real robot behind HIM's proprietary "Adam" fixture (public, **BSD-3**, `github.com/BoosterRobotics/booster_assets`, `robots/K1/K1_22dof.{urdf,xml}`). 22 DOF, wrist-less 4-DOF arm — orientation IK on it looked "crazy", which motivated the G1 switch. HIM's Adam manifest is `fixture_only`, `run_on_adam_enabled:false`.
- **Contract:** `.him/contracts/runtime.json` obs `robot.joint_position` → action `robot.desired_joint_position`, currently bound to **[22]** (K1) with joint order in `extensions.robot`; 50 Hz, fallback "hold". Pending migration to G1's 29. `him validate` = valid.

## 6. The data (`data/`, git-ignored, cleaned 2026-08-26)

Raw human table-tennis capture — three unsynchronized streams (separate clocks; all carry absolute UTC/epoch timestamps so software alignment is feasible):
- **Paddle IMU** (`him-table-tennis-paddle-imu-*.tar`): WitMotion, human-held paddle, ~11.11 Hz, 10 episodes, 20,741 samples. 12 real channels (accel×3 g, gyro×3 dps, euler×3 deg, mag×3 µT). `quaternion_*`/`chip_time` all null; `firmware`/`battery` constant.
- **Orbbec RGB-D** (8 `orbbec_*.tar`): 2 cams, 640×480 @30fps, color JPG + depth PNG (mm, `align_depth_to_color:true`). **No intrinsics/extrinsics in the archive** (recover intrinsics from Orbbec SDK).
- **Rokoko mocap** (`rokoko-fbx.tar`): 8 binary FBX clips, standard bone names (`Hips`, `Spine*`, `RightShoulder`→`RightArm`→`RightForeArm`→`RightHand`).
- There is **no robot / no joint data** in the capture — `robot_version_id: null`. (Cleanup removed ~13 GB browser-duplicate tars + `.crdownload` partials; kept one canonical of each.)

## 7. Pipeline & prior art

Pipeline: `perception(ball) + movement(swings) → control(imitate + RL in sim) → him_adapter package → arena submit`.

Reference approaches (both: **model-based ball prediction → RL whole-body controller**):
- **HITTER** (Berkeley; `humanoid-table-tennis.github.io`, arXiv 2508.21043): model-based ball/racket planning + RL whole-body controller, with **human motion references** for naturalness. Real-world: up to 106 consecutive shots. → validates our mocap-retarget-as-reference plan.
- **PACE** (Purdue TraceLab, ICRA 2026; `github.com/purdue-tracelab/PACE-ICRA2026`, **MIT**): Booster T1, Isaac Sim + Isaac Lab; end-to-end **PPO** + learned ball-landing predictor + physics **dense** rewards; ≥96% hit. Repo = training/eval scripts + TT sim env (no pretrained/models/datasets). Usable harness to adapt to the G1.

## 8. Repo map & key commands

```
him_adapter/   Deployment policy server HIM runs (HoldPolicy fallback + BCPolicy).
.him/          HIM overlay: manifest, runtime contract, fixtures, smoke test, lock.
training/      PyTorch behavior-cloning scaffold.
sim/           Local MuJoCo (un-gated): view_k1, k1_model, g1_model, fbx_extract, retarget_imu, g1_retarget.
scripts/       fetch_assets.sh (re-clone K1 + G1 models).
docs/          THIS knowledge doc.
```
```bash
./scripts/fetch_assets.sh
./.venv/bin/python sim/g1_retarget.py     # real human swing on the G1 -> sim/g1_retarget.gif
.venv-fbx/bin/python sim/fbx_extract.py -- <clip.fbx> data/mocap_clip4.npz
~/.local/bin/him validate
```

## 9. Standing constraints (operating rules)

Do not invent specs (signals, units, joint counts, licenses, evaluator behavior).
Edit only `.him` by default; ask before changing native project files and before
anything that uploads/spends/launches. Never weaken checks to pass validation.
Don't commit weights/data/creds. Keep this doc updated as the project's memory.
