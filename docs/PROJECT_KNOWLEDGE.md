# Project knowledge — himrobot (HIM humanoid table tennis)

Canonical, version-controlled project knowledge. Keep this current as decisions
and findings land — this is the source of truth (not any external notes).
_Last updated: 2026-08-27 (reviewer corrections: corrected retarget numbers, perception/alignment facts, tooling)._

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
| **Movement** | human swings → robot reference motions | 🟢 **G1, full body (29 DoF), all 8 clips** via GMR (`sim/gmr_bvh_to_g1_headless.py --ik_config sim/ik_configs/bvh_lafan1_rokoko_to_g1.json`), **kinematic only**: the previous CSVs (`~/him-tools/out/all`, upstream offsets) had 50–69 cm mean wrist error and 23–31 % joint-limit saturation; the corrected config (`work/g1/`) gives ~10–12 cm mean wrist error (`work/retarget/adjust2_rotfix_h1p75.md`, §4c). No dynamics/feasibility check yet. The older right-arm-only IK (`sim/g1_retarget.py`) is kept as a reference |
| **Perception** | ball / scene 3D (`ball_state`) from cameras | 🟡 2D orange-ball tracking + aligned-depth lookup on the two 30 fps sessions on disk (`sim/ball_detect.py`, `sim/ball_3d.py`; 215350: 26–32 % of frames carry a candidate at the documented settings (gated, ROI margin 40 px), ~1/2–2/3 of sampled candidates are the real ball, rest = sandals at the far edge); ArUco DICT_4X4 IDs 0–7 verified at the half-table corners (`sim/aruco_scan.py`). **No intrinsics anywhere on disk → no metric XYZ yet**; the `*_60fps` rally tars are 0-byte downloads. See §6a. |
| **Control** | `joints + ball → desired joints` (deployable hitter) | 🟡 scaffolded; blueprint = PPO in sim + ball predictor + mocap refs. `training/` currently trains a **next-pose motion prior** (BC, joints[t] → joints[t+1]) on the corrected `work/g1` CSVs — not a hitter; numbers in `training/README.md` |

**Pending / loose ends**
- ~~Migrate `.him/contracts/runtime.json` from the K1's 22 joints to the G1's 29.~~ Done 2026-08-26 (Menagerie/SDK order in `extensions.robot.joint_order`).
- Train a G1 tracking policy on the retargeted CSVs (mjlab / BeyondMimic recipe; needs an NVIDIA machine — the CSV format `sim/gmr_bvh_to_g1_headless.py` emits is the Unitree/LAFAN1 CSV those recipes ingest).
- Reconcile contract signal names / rate with HIM's server-side evaluator (unconfirmed).
- ~~Time-align the three capture streams (separate clocks) before the control track.~~ Done 2026-08-26 (`sim/align_streams.py`, §6b): IMU↔Rokoko lags per session from cross-correlation (|gyro| vs hand angular speed; the 7 play sessions r 0.36–0.64, lags −6.0…+0.9 s; the 17 s test take 214615↔Clip separately at r 0.987, lag +6.8 s), IMU↔Orbbec by wall clock (both hosts agree to seconds; HIM's America/New_York UTC inference is 3 h off the Orbbec host's UTC−7). Open: only 2 of 8 Orbbec sessions are on disk; Rokoko has no absolute clock, so it is anchored only via the IMU.
- Compute at scale is gated (see §4); local MuJoCo is the un-gated path.

## 3. Environment & tooling

- `him` CLI = `him_sdk 0.1.3` in a **Python 3.12** venv at `~/.local/share/him/venv` (binary `~/.local/bin/him`). Endpoint default is `http://127.0.0.1:5173` (dead) — **always** `export HIM_API_URL=https://arena.himrobotics.com`. Authenticated (account `him_pat_…`, creds at `~/.config/him/credentials.json` — do not read/commit).
- Main dev venv: `.venv/` (`/usr/local/bin/python3.12` → 3.12.8) — torch 2.13.0, mujoco 3.12.0, imageio 2.37.4, imageio-ffmpeg 0.6.0, numpy 2.5.2, pytest 9.1.1 (installed 2026-08-26).
- FBX reading: the **Blender binary** (`/Applications/Blender.app/Contents/MacOS/Blender`, 3.4.1, bundled Python 3.10.8 + numpy 2.2.6, `io_scene_fbx` enabled) — `Blender -b --python sim/fbx_extract.py -- <clip.fbx> <out.npz>`. No `bpy` wheel / `.venv-fbx` needed (the old side-venv path is retired). The importer leaves the scene range at 1..250, so `fbx_extract.py` takes the frame range from the armature action (Clip_4 = 9569 frames @ 30 fps).
- Other tool venvs outside the repo: `~/him-tools/mujoco-venv` (python 3.12.8: mujoco 3.12.0, imageio 2.37.4, imageio-ffmpeg 0.6.0, numpy 2.5.2, pillow 12.3.0, **scipy 1.18.1 + matplotlib 3.11.1** for `sim/align_streams.py` / `sim/retarget_report.py`), `~/him-tools/cv-venv` (python 3.12.8: numpy 2.5.2, opencv-contrib-python 5.0.0.93, imageio 2.37.4, pillow 12.3.0 — used by `sim/ball_detect.py`, `sim/ball_3d.py`, `sim/aruco_scan.py`, `sim/orbbec_video.py`), `~/him-tools/gmr-venv` (GMR, torch, mujoco, mink), `~/him-tools/GMR` (upstream, read-only), pyenv 3.11.9 at `~/.pyenv/versions/3.11.9`.
- Vendored assets (git-ignored, re-fetch via `scripts/fetch_assets.sh`): `third_party/booster_assets` (K1, 49 MB) and `third_party/mujoco_menagerie` (sparse checkout of `unitree_g1` only, 56 MB). Fallback copy of the G1 lives at `~/him-tools/mujoco_menagerie/unitree_g1`.
- `data/` (git-ignored) holds HIM's proprietary licensed capture — never commit. Consolidated as REAL files (no symlinks, user directive 2026-08-27): `him-table-tennis-paddle-imu-20260815/` (+ `paddle-imu-samples.csv` copy that `sim/retarget_imu.py` reads), `rokoko-motion-capture/` (8 FBX), `orbbec_20260815_214619/`, `orbbec_20260815_215350/` (extracted sessions), `release/` (the 10 verified tars from `him data download table-tennis`), `manifests/` (drafted HIM dataset manifests + packed artifacts), `preview/` (Blender .blend), `g1/` (copy of the corrected retarget CSVs from `work/g1`, what `training/` reads), `mocap_clip4.npz`. Derived BVH lives in `work/bvh` and `work/bvh_lafan1`. Tool venvs stay in `~/him-tools/`.

## 4. HIM platform mechanics

- **Data:** `him data download` is hard-restricted to the **table-tennis** release only; no catalog, no robot obs→action dataset to pull. `capture`/`dataset`/`train` consume LOCAL fixtures you author (`him train rehearse` is a deterministic stub — no real training).
- **Simulator:** server-side only (no local sim). Run a policy against a sim bot via `him arena submit <challenge_version_id> --package <approved_pkg> --entrypoint <ep>` (Community challenge, isaac-sim; scored on legal-return/target-hit/landing-error/latency), or `him job launch simulation` (mujoco-simulation-contract, `mujoco_mjx`, gpu-standard) + `him job watch`. Output is **metrics/events, not video**.
- **Gating:** real package execution is blocked server-side by a per-package worker gate (`package_worker_disabled`). Our first `him run` **failed** (reconcile-timeout); a built-in fixture control run succeeded, proving the plumbing is healthy — real code needs HIM to enable the worker. Package `my-policy@0.1.0` id `pkg_110ce5d185fa9ba3d703`; failed job `job_6be550b37b3337434d9a`.
- **Approval gates (ask the user first):** `him login/run/push/publish`, `job launch`, `arena submit`, `dataset push` — anything that uploads source, spends credits, or launches compute.

## 4b. HIM datasets (private, registered from this machine 2026-08-26)

- `him-table-tennis-paddle-imu@1.0.0` → **`dataset_f410d4b7a08c0ab25d88`** (status ready; usable as `him run --dataset <id>`). Pushed from HIM's own manifest with `license.spdx_or_name` added.
- Orbbec 214619 / 215350 and Rokoko mocap: manifests drafted + `him dataset validate` clean under `~/him-data/manifests/`, **not pushed** — license for locally captured data and the Rokoko session timestamp need confirmation.
- `him capture inspect` / `dataset build` / `train rehearse` accept only `fixture_only` JSON in 0.1.3; raw-capture manifests must be hand-written.

## 4c. Retargeting pipeline (`.him/contracts/training-pipeline.json`, `rokoko-to-g1-motion-tracking-v1`)

Blender headless FBX→BVH (`sim/fbx_to_bvh.py`, ~1 min/clip) → `sim/validate_motion.py` → rename Rokoko "Newton" bones to LAFAN1 names (`sim/rokoko_to_lafan1_bvh.py`: Spine1→Spine, Spine2→Spine1, Spine4→Spine2, Thigh→UpLeg, Shin→Leg; GMR looks bones up by name, extra bones tolerated, cm + Y-up expected) → GMR `--robot unitree_g1 --format lafan1` (`sim/gmr_bvh_to_g1_headless.py`, ~350 fps on M4 Pro CPU; `--jobs N` parallelises clips) → Unitree CSV (root xyz, quat **xyzw**, 29 joints in SDK order) → `sim/play_g1_csv.py` (live `launch_passive` or `--video`). Full `him pipeline run` verified 2026-08-26 (pre-`--ik_config`); after the contract change the retarget stage command was reproduced on one clip (byte-identical CSV) but the full run was not re-executed: 8 clips, 59,685 frames, 33 min of motion.
Tool venvs live outside the repo (`~/him-tools/{gmr-venv,mujoco-venv,GMR,mujoco_menagerie}`); pass them via `--set gmr_python=… --set mujoco_python=…`.

**Retarget fidelity (2026-08-26, `sim/retarget_report.py`; reports in `work/retarget/`, CSVs in `work/g1/`).** `work/g1/` is the **corrected** set (adjustment 2 below) and is what `data/g1` and `training/` use; `~/him-tools/out/all` is the **broken baseline** (upstream rotation offsets) and is kept only for comparison.
- **GMR internals** (`~/him-tools/GMR`, read-only): `utils/lafan1.py:load_bvh_file` returns a hard-coded `human_height = 1.75` (the `Head − lowest foot + 0.2` formula is commented out). `GeneralMotionRetargeting.__init__` computes `ratio = actual_human_height / ik_config["human_height_assumption"]` (1.8 in `bvh_lafan1_to_g1.json`) and multiplies every `human_scale_table` entry by it (defaults 0.9 hips/legs, 0.75 arms → 0.875 / 0.729 at 1.75 m). The config path comes from `params.IK_CONFIG_DICT[src][robot]`, patchable at runtime — `gmr_bvh_to_g1_headless.py --ik_config PATH --human_height H` does that; `rt.human_scale_table` is also a plain dict attribute. Nothing upstream is edited.
- **BVH facts** (Blender export, 6 channels/joint): the position channels *replace* the OFFSET (GMR's vendored `read_bvh` does exactly that; our own FK in `retarget_report.py` matches GMR's loader to 1e-15 m; non-root position channels equal their OFFSET lines; bone lengths are constant to 1e-7 m over all frames). Frame 0 of every clip is the rest T-pose with all rotation channels ≈ 0. Hips sit at **0.917 m** (root position channel 91.72 cm; the OFFSET line says 94.0 cm but is overridden). Head joint 1.593 m, lowest toe joint −0.010 m → **head-joint-above-floor 1.603 m**. The `Head` and finger End Sites are degenerate (Head offset (0, −11.1, 0) = back to the Neck; each fingertip End Site is the negative of its parent's offset), but the Toe End Sites are real (OFFSET (0, 0, −14.57 cm) from each `LeftToe`/`RightToe`, not the negative of the Toe's own (0, −6.24, +13.17) offset); true stature is still **not derivable** from the BVH; GMR's commented-out formula would give 1.803 m. Segments: arm shoulder→wrist 0.554 m (27.1 + 28.3 cm), leg hip→ankle 0.865 m, shoulder 0.487 m above Hips, shoulder width 0.412 m. G1 (`right_shoulder_pitch_link→right_elbow_link→right_wrist_yaw_link`, Menagerie) 0.377 m → arm ratio **0.68** vs GMR's 0.729.
- **Metric**: right wrist (`RightHand` vs `right_wrist_yaw_link`) in a pelvis-centred heading frame (origin `Hips`/`pelvis`, z = world up, y = left hip − right hip; torso = pelvis, not `torso_link`), human vector scaled by 0.68; also a shoulder-centred variant. Robot FK = MuJoCo on Menagerie `scene.xml`. Joint-limit saturation is counted against GMR's own `g1_mocap_29dof.xml` ranges (what its IK clamps at; tighter than Menagerie, e.g. shoulder_pitch ≤ 1.149 vs 2.67, elbow ≤ 1.7 vs 2.09).
- **Root cause found**: the upstream `bvh_lafan1_to_g1.json` rotation offsets assume LAFAN1 bone frames. Blender/Rokoko bones have identity rest rotations, so the pelvis orientation target is 120° off (upstream offset (0.5, 0.5, 0.5, 0.5) vs derived (0.5, −0.5, −0.5, −0.5)); the IK fights its orientation tasks. **Baseline** (`~/him-tools/out/all`, 1.75 m; `work/retarget/baseline_1p75.json`): wrist error mean **50–69 cm** (Clip 50.2, Clip_2 55.8), **23.0–30.8 %** of joint-frames at IK limits (per clip `ik_model_total / (frames × 29)`), feet 6–19 cm under the floor, root z down to 0.46 m, robot elbow never beyond 59–67°.
- **Adjustment 1** — `--human_height 1.6032` with the upstream config (`work/g1/h1p60/`, `work/retarget/adjust1_h1p60.json`): no gain (Clip 50.3, Clip_2 55.8 cm mean), saturation 25.2–33.0 %, pelvis lower (0.25–0.82 m), penetration worse.
- **Adjustment 2 (kept)** — `sim/ik_configs/bvh_lafan1_rokoko_to_g1.json`, generated by `sim/ik_configs/make_rokoko_g1_config.py`: rotation offsets re-derived as `Rx(90)ᵀ·Rz(−90)·R_body(q_T)` from the GMR model in a G1 T-pose (shoulder_roll ∓π/2, elbows +π/2), everything else copied from upstream, height kept at 1.75 m. Wrist error mean **10.0–11.7 cm** (Clip 10.0, Clip_2 10.6; medians 9.4–11.2; max 16–25 cm, Clip_5 50 cm), shoulder-frame 8.9–11.6 cm, saturation 60 (Clip) to 4,521 (Clip_7) joint-frames (0.3–1.7 %, mostly `ankle_roll`), min foot z +0.008 m (Clip) … −0.083 m (Clip_5), root z 0.53–0.89 m, robot elbow range 0–123° vs human 0–130°. `work/g1/*.csv` and `work/retarget/{Clip_g1.mp4, Clip_4_g1_20s.mp4}` are from this run (Clip_4 = best long clip: 10.1 cm mean, 988 saturated joint-frames; segment = frames 5521–6121, the 20 s with the most wrist travel). Still at the 10 cm mark: the human shoulder sits 0.49 m above the hips vs 0.29 m on the G1, which a per-bone scale about the root cannot absorb.
- Commands: `~/him-tools/gmr-venv/bin/python sim/gmr_bvh_to_g1_headless.py --bvh_dir work/bvh_lafan1 --out_dir work/g1 --ik_config sim/ik_configs/bvh_lafan1_rokoko_to_g1.json --jobs 5`; `~/him-tools/mujoco-venv/bin/python sim/retarget_report.py --bvh_dir ~/him-tools/bvh_lafan1 --csv_dir work/g1 --out work/retarget/report` (`--measure_height clip.bvh` for the anthropometrics). The pipeline contract passes `--ik_config sim/ik_configs/bvh_lafan1_rokoko_to_g1.json --jobs 4` (updated 2026-08-27).

## 5. The robot

- **Current target: Unitree G1** (user directive). 29-DOF, **7-DOF arm with wrist** (`right_shoulder_pitch/roll/yaw`, `right_elbow`, `right_wrist_roll/pitch/yaw`), hand at `right_wrist_yaw_link`. Model: `mujoco_menagerie/unitree_g1/g1.xml` (+ `g1_mjx.xml` matches HIM's `mujoco_mjx`). The 7-DoF arm lets the IK place the wrist: ~10–12 cm mean wrist error with the corrected config vs 50–69 cm before (§4c); kinematic only, no dynamics check. G1 also matches our contract's signal names.
- **Earlier reference: Booster K1** = the real robot behind HIM's proprietary "Adam" fixture (public, **BSD-3**, `github.com/BoosterRobotics/booster_assets`, `robots/K1/K1_22dof.{urdf,xml}`). 22 DOF, wrist-less 4-DOF arm — orientation IK on it looked "crazy", which motivated the G1 switch. HIM's Adam manifest is `fixture_only`, `run_on_adam_enabled:false`.
- **Contract:** `.him/contracts/runtime.json` obs `robot.joint_position` → action `robot.desired_joint_position`, bound to **[29]** (G1) with the Menagerie/SDK joint order in `extensions.robot.joint_order`; 50 Hz, fallback "hold". Signal names/rate still unconfirmed against HIM's evaluator. `him validate` = valid.

## 6. The data (`data/`, git-ignored, cleaned 2026-08-26)

Raw human table-tennis capture — three unsynchronized streams (separate clocks). The IMU CSV carries wall-clock timestamps and the Orbbec sessions epoch seconds, so those two align in software; the Rokoko FBX carries **only an export `CreationTimeStamp`** (two days after capture) and no recording time, so it is anchored through the IMU by cross-correlation (§6b):
- **Paddle IMU** (`him-table-tennis-paddle-imu-*.tar`): WitMotion, human-held paddle, nominal 11.11 Hz / effective ~10.1 Hz (§6b), 10 episodes, 20,741 samples. 12 real channels (accel×3 g, gyro×3 dps, euler×3 deg, mag×3 µT). `quaternion_*`/`chip_time` all null; `firmware`/`battery` constant.
- **Orbbec RGB-D** (8 `orbbec_*.tar`): **2 synced cams** (cam0/cam1 stereo pair), 640×480, color JPG + depth PNG (mm, `align_depth_to_color:true`); 60fps sessions are the rally sessions. Table has **ArUco markers at the corners** → metric table-frame calibration is possible (solvePnP + triangulation) without factory intrinsics. **The ball is ORANGE** on a dark-blue table → HSV color detection works (see `sim/ball_detect.py`). Beware a second held/spare orange ball + net-post clutter.
- **Rokoko mocap** (`rokoko-fbx.tar`): 8 binary FBX clips, standard bone names (`Hips`, `Spine*`, `RightShoulder`→`RightArm`→`RightForeArm`→`RightHand`).
- There is **no robot / no joint data** in the capture — `robot_version_id: null`. (Cleanup removed ~13 GB browser-duplicate tars + `.crdownload` partials; kept one canonical of each.)

### 6a. Perception status (2026-08-26, Track C)

**What is on disk.** Only two Orbbec sessions are usable: `orbbec_20260815_214619` (extracted + 59.7 MB tar; 125 rows/cam, 5.3 s, host-time ≈23–24 fps, **no ball — a player adjusts a paddle, nothing is in flight**) and `orbbec_20260815_215350` (extracted + 1.9 GB tar in `~/Downloads`; 3,983 rows/cam, 193.5 s, host-time **≈20.6 fps** effective vs. `fps_prefer: 30`, ~1,820 frames/cam short of 30 fps per the manifest quality report; `session.json` has no `host_time_end`/`frame_counts`, i.e. the recorder was interrupted). 215350 contains real play: serves and rallies from about frame 100 onward, ball clearly visible (see `work/perception/precision_check_*.jpg`, `work/perception/orbbec_20260815_215350_cam0_CP0E8530008E_color.mp4` = raw cam0 colour, 3,980 frames at 20.6 fps). `~/Downloads/orbbec_20260815_{220448,221035,221649,222542}_60fps.tar` and `orbbec_20260815_{214831,215750}.tar` are **0-byte failed downloads** — re-download before anything about the 60 fps rally sessions can be claimed. A few `frames.jsonl` rows have `color_file: null` / `depth_file: null` (215350: 3+5 colour, 2+3 depth; 214619: 2+4 colour, 0+1 depth) — the recorder dropped that stream for the row; the detector skips them. 214619 cam1 frame 10 decodes with a corrupted lower third.

**Camera geometry (from the frames, not from metadata).** The two cams are *not* a classic overlapping stereo pair: cam0 (`CP0E8530008E`) looks down one half of the table with the net at its left image edge; cam1 (`CP0E8530008F`) looks down the other half with the net at its right edge. Overlap is only the net strip. ArUco: `DICT_4X4_50` fires (also its 100/250/1000 supersets, no 5×5/6×6/7×7/AprilTag), **4 markers per half at the half's four corners**: cam0 sees IDs 4 (near/net ≈ (86,425)), 5 (far/net ≈ (190,101)), 6 (near/end-line ≈ (567,427)), 7 (far/end-line ≈ (472,86)); cam1 sees IDs 0 (near/end ≈ (87,408)), 1 (far/end ≈ (203,97)), 2 (near/net ≈ (564,435), partly behind the net post), 3 (far/net ≈ (481,103)). Decode counts over the 12 sampled frames (`work/perception/aruco_orbbec_20260815_215350.json`, `n_frames`): cam0 IDs 4 → 12, 6 → 12, 5 → 9, 7 → 3; cam1 IDs 0 → 12, 3 → 8, 1 → 7, 2 → 3 — the ~20 px far markers decode in 3–9/12, the near ones in 12/12 except ID 2 (net post) at 3/12. Median pixel corners per ID are in `work/perception/aruco_orbbec_20260815_*.json`. The "markers at the table corners" claim in §6 is **confirmed** (per half; the neighbouring half's markers are visible across the net but not decoded). Marker physical size and exact offset from the table corner are **unknown** — needed for solvePnP.

**2D detector (`sim/ball_detect.py`).** HSV orange mask + roundness, gated by frame differencing (`--motion 8`), the ArUco-hull table ROI (`--roi-json`, `--roi-margin`), a static-orange map (`--static-reject 0.2`, kills the bottle by the stool and a small orange object on the floor by the far player) and distance to large red/orange blobs (`--near-big 12`, the paddle rubber). Reads a directory or the tar, `--cam cam0|cam1`. Measured colour (OpenCV HSV, median over a disc of 0.6× the detected radius around the `ball2d` centre, 215350 cam0, re-checked 2026-08-27 with `~/him-tools/cv-venv`): six confirmed ball hits (frames 255/1799/2518/2954/3058/3638) H 18–20, S 159–194, V 213–234; four sandal hits (frames 2067/2246/2406/2623) H 16–17, S 112–125, V 99–145. Hue overlaps, and over the full detected disc the motion-blurred ball edge pulls its median S down to 97–122 — onto the sandal range — so colour alone cannot separate them. Rates over the full sessions (gated, ROI margin 40 px):

| session / cam | frames | candidates | sampled precision |
|---|---|---|---|
| 215350 cam0 | 3,980 | 1,270 (31.9 %) | 8/12 real ball (4 = far player's sandals) |
| 215350 cam1 | 3,978 | 1,043 (26.2 %) | 5/12 real ball (7 = sandals/feet at the far edge) |
| 214619 cam0 | 123 | 20 (16.3 %) | 0/20 — all the far player's sandal |
| 214619 cam1 | 121 | 0 | — |

With `--roi-margin 15` the 215350 counts drop to 860 (21.6 %) / 631 (15.9 %) (`work/perception/roi15/`) at the cost of balls in the air past the far edge. Ungated colour-only gives ~100 % "candidates" (pure clutter): the old "119/121 frames in a rally" figure is of unreproducible provenance — ungated 214619 cam1 (a session with **no ball in flight**) gives exactly 119/121 (`work/perception/nomotion/log_214619_cam1.txt`), so that figure was clutter, not a rally. Tracks: `work/perception/ball2d_<session>_<cam>.csv` (frame, host_time_s, u, v, radius; blank when nothing found). Ball in flight moves 50–80 px/frame at 20 fps and is motion-blurred; net rally length in the images is limited by the ~20 fps rate.

**Depth (`sim/ball_3d.py`).** Depth PNGs are plain 16-bit greyscale (`I;16`, no PNG text chunks, no EXIF in the JPGs), `depth_png_unit: millimeters_uint16`, `depth_scale: 1.0` in every row, ~99 % non-zero. Median depth inside the ball disc vs. an annulus around it: 215350 cam0 median disc depth 1,649 mm, disc is >40 mm nearer than the annulus in 67 % of hits (cam1: 1,754 mm, 50 %) — so the aligned depth **does resolve the ball roughly half to two-thirds of the time**; the rest reads the table behind it (and false positives read the floor, ~3.5 m). Output `work/perception/ball3d_<session>_<cam>.csv` (adds depth_mm, n_valid, bg_depth_mm).

**What blocks metric XYZ / `ball_state`.** No intrinsics anywhere: `session.json`, `frames.jsonl`, the tar (only images + the two JSON files), the PNG/JPEG headers and the HIM manifests (`calibration: []`, "No intrinsics/extrinsics available") all lack fx/fy/cx/cy, distortion and the cam0↔cam1 extrinsics. Ways to get them (pick one, none done yet): (a) `pyorbbecsdk` (not on PyPI — build from `github.com/orbbec/pyorbbecsdk`) with the same two cameras plugged in, `pipeline.get_camera_param()` on serials `CP0E8530008E` / `CP0E8530008F` at the 640×480 colour profile (aligned depth shares the colour intrinsics); (b) checkerboard / ChArUco calibration of each camera at 640×480; (c) if the ArUco marker edge length and their positions on the table are measured, `solvePnP` per camera gives the table-frame pose *and* (with ≥4 non-coplanar views or `calibrateCamera` on the 8 markers over many frames) an intrinsics estimate — weaker, but it uses only the existing footage. Also still needed: marker size + table offsets, cam0↔cam1 relation (via the shared table frame), a clock offset between the Orbbec host clock and the IMU/Rokoko clocks.

Tooling: `~/him-tools/cv-venv` (python 3.12.8: numpy 2.5.2, opencv-contrib-python 5.0.0.93, imageio 2.37.4, pillow 12.3.0); all outputs under `work/perception/` (git-ignored).

### 6b. Stream alignment (2026-08-26, Track D)

Script: `sim/align_streams.py` (run with `~/him-tools/mujoco-venv/bin/python`, which now also has scipy 1.18.1 + matplotlib 3.11.1). Outputs under `work/alignment/` (git-ignored): `session-map.json`, `imu-mocap-xcorr.csv` (all 10×8 pairs), `imu-orbbec-clock.json`, `fbx-headers.json`, `xcorr_<episode>_<clip>.png`, `xcorr-matrix.png`; `--orbbec-motion` adds the (inconclusive) video check. Everything below is read from the files; nothing is assumed.

**Clocks.** The IMU CSV `timestamp_local` is the raw iOS wall clock (no zone); HIM's `timestamp_utc` = local + 4 h (America/New_York, stated as *inferred* in `capture-manifest.json`). The Orbbec `session.json`/`frames.jsonl` carry epoch seconds; the session directory name is the host wall clock at launch. `orbbec_20260815_214619` → `host_time_start` 04:46:36.48 UTC, `215350` → 04:53:55.87 UTC, i.e. the Orbbec host displayed **UTC−7** (launch→`host_time_start` 17.5 s and 5.9 s of camera init). The IMU and Orbbec hosts show the **same wall clock**: Orbbec 215350 frames run 21:53:56.43–21:57:09.97, IMU episode 215358 runs 21:53:58.29–21:57:09.61 (starts 1.9 s apart, ends 0.4 s apart, 191.3 s overlap); Orbbec 214619 (21:46:36.83–21:46:42.14) lies inside IMU 214615 (21:46:15.93–21:46:43.50); the six 0-byte tars' launch stamps precede the matching IMU episode starts by −0.5…+8.3 s. So the "3 h" between HIM's `timestamp_utc` (01:46 Z) and the Orbbec `created_utc` (04:46 Z) is exactly UTC−4 vs UTC−7: the two hosts agree on wall time, and the Orbbec host's zone was UTC−7. **Either the IMU's America/New_York inference is wrong by 3 h (true UTC = `timestamp_local` + 7 h) or the Orbbec host's epoch clock was 3 h off while both hosts shared a zone setting — the files cannot distinguish these**, but for alignment it does not matter: align on wall clock (`timestamp_local` ↔ `host_time_s` − 7 h; `imu-orbbec-clock.json` has every offset). Residual Orbbec−IMU start offsets are recorder start times, not drift: 215350/215358 −1.87 s (first frame vs first sample), 214619/214615 +20.9 s. An independent check via frame-difference motion energy of the Orbbec colour stream vs |gyro| is **inconclusive** (r ≈ 0.10 at the expected lag, no peak above z 2.7 in ±60 s, both cams) — the whole-frame signal is dominated by the other player/ball.

**Rokoko FBX headers** (binary FBX 7500 parsed node-by-node): `CreationTimeStamp` 2026-08-17 13:08:23 (Clip) … 13:31:04 (Clip_8), i.e. **export time two days after capture, in the export host's unstated local zone**; `Original|DateTime_GMT` and `LastSaved|DateTime_GMT` are empty; one take named `clip` (`clip.tak`), take length = clip length (17.43, 252.87, 192.90, 318.93, 295.37, 298.60, 305.23, 307.90 s), `TimeMode` 6 = 30 fps, Y-up, cm. Clip_8 was exported from a Rokoko folder named `Clip (1)`. **No recording timestamp exists in the FBX**; Rokoko is anchored only through the IMU.

**IMU↔Rokoko method.** Frame 0 of every Blender BVH is a T-pose (all rotation channels 0) and is dropped, so **clip t = 0 is BVH frame 1** (all lags below use that origin). BVH FK uses standard semantics with the 6-channel position values *replacing* the OFFSET (Blender exporter writes rest-offset+translation); verified against Blender's own `pose_bone.head` on Clip.fbx (Hips exact, RightHand mean 1.2 cm / max 2.7 cm). Signal = RightHand **world angular-speed magnitude** (from the FK rotation matrices) vs IMU |gyro| — the same physical quantity for a paddle held rigidly (p99: hand 520–615 dps, IMU 670–740 dps); linear hand speed is kept as the `alt_feature` (lower r everywhere). Both resampled to 10 Hz, Gaussian σ 0.4 s, z-scored and clipped at ±4σ, Pearson r over the overlap for lags ±60 s (min 15 s overlap). Lag sign: clip time t ↔ IMU episode time t + lag; negative = IMU recording started after the clip. `secondary_ratio` = best local maximum more than 5 s away / peak; `lobe` = lag span with r ≥ 0.8·peak; peak-match = mode of pairwise swing-peak differences (>400 dps, σ 0.1 s).

| IMU episode (wall, s) | Orbbec session | Rokoko clip (s) | lag s | r | 2nd ratio | z | lobe s | peak-match lag | runner-up clip (r) | confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 214615 (21:46:15, 27.6) | 214619 (on disk, 5.3 s inside) | Clip (17.5) | +6.8 | 0.987 | 0.39 | 2.3 | 1.1 | — (no peaks >400 dps) | — | test take: four slow paddle wobbles match exactly, but only ~17 s of signal |
| 214836 (21:48:36, 257.3) | 214831 (0-byte) | Clip_2 (252.9) | −3.8 | 0.636 | 0.32 | 4.3 | 2.1 | −4.2 | Clip_5 (0.244) | high |
| 215358 (21:53:58, 191.3) | 215350 (on disk, 193.5 s) | Clip_3 (192.9) | +0.9 | 0.484 | 0.46 | 3.6 | 1.7 | −14.3 (disagrees) | none by duration | medium |
| 215749 (21:57:49, 335.3) | 215750 (0-byte) | Clip_4 (319.0) | −1.2 | 0.446 | 0.41 | 3.9 | 2.0 | −1.3 | Clip_7 (0.223) | medium |
| 220452 (22:04:52, 301.5) | 220448_60fps (0-byte) | Clip_5 (295.4) | +0.3 | 0.452 | 0.47 | 3.5 | 2.7 | −19.4 (disagrees) | Clip_7 (0.233) | medium |
| 221043 (22:10:43, 300.9) | 221035_60fps (0-byte) | Clip_6 (298.6) | −2.7 | 0.364 | 0.81 | 3.0 | 3.0 | −28.9 (disagrees) | Clip_4 (0.201) | medium (weakest) |
| 221654 (22:16:54, 303.7) | 221649_60fps (0-byte) | Clip_7 (305.3) | −4.8 (twin peak −1.6, r 0.39; linear-speed feature picks −1.6) | 0.422 | 0.51 | 2.7 | 4.2 | −1.8 | Clip_5 (0.196) | low (lag ambiguous between −4.8 and −1.6) |
| 222546 (22:25:46, 311.2) | 222542_60fps (0-byte) | Clip_8 (307.9) | −6.0 | 0.624 | 0.46 | 4.3 | 2.7 | −6.2 | Clip_5 (0.257) | high |
| 192602 (19:26:02, 21.7) | — | — (Clip is taken by 214615; r 0.58 here but IMU peaks 2087 dps vs 271 dps in the clip) | | | | | | | | none |
| 215351 (21:53:51, 3.8) | — (false start before 215358) | — | | | | | | | | none |

The four ~300 s sessions (220452/221043/221654/222546 vs Clip_5…8) are ambiguous by duration alone, but the 4×4 correlation block is diagonal both row- and column-wise (diagonal r 0.36–0.62, every off-diagonal ≤ 0.28), so the chronological pairing IMU→clip in recording order holds. Implied start/stop consistency (IMU end − clip end = IMU duration − lag − clip duration): Clip_2 +8.2 s, Clip_3 −2.5 s, Clip_4 +17.5 s, Clip_5 +5.8 s, Clip_6 +5.0 s, Clip_7 +3.2 s (0.0 s at the −1.6 s lag), Clip_8 +9.3 s, Clip +3.3 s — i.e. the IMU was usually started a few seconds after and stopped a few seconds after the Rokoko take, consistent with one operator working two apps. Chain for the one fully-tripled session: Clip_3 frame 0 = IMU 215358 t = 0.9 s = wall 21:53:59.2, Orbbec 215350 frame 0 = wall 21:53:56.43 → `clip3_t = orbbec_host_time_s − 7 h − 21:53:59.2`.

**Caveats.** IMU is nominally 11.11 Hz but effectively **~10.1 Hz** (samples/duration in `work/alignment/run.log`, e.g. 1,932 samples / 191.3 s for 215358, 3,384 / 335.3 s for 215749), so swings (~0.2 s) are aliased — r is moderate (0.36–0.64) and lag lobes are 1.7–4.2 s wide; treat lags as ±0.5 s and refine per swing if needed. Peak matching finds ~2× more IMU peaks than hand peaks and agrees with the cross-correlation only for Clip_2/4/8. The Orbbec-motion check does not confirm the shared-wall-clock hypothesis (nor contradict it). Six of eight Orbbec sessions are 0-byte downloads, so IMU↔Orbbec offsets exist only for 214619 and 215350. Rokoko carries no absolute clock; a 0.1 s lag grid was used.

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
sim/           MuJoCo (un-gated): view_k1, k1_model, g1_model, fbx_extract, retarget_imu, g1_retarget;
               full-body pipeline: fbx_to_bvh, validate_motion, rokoko_to_lafan1_bvh, gmr_bvh_to_g1_headless (--ik_config),
               play_g1_csv, retarget_report.py (wrist-error / saturation report), ik_configs/;
               perception: ball_detect.py (orange ball, 2D), ball_3d.py (aligned depth), aruco_scan.py (table markers),
               orbbec_video.py (frames->mp4);
               alignment: align_streams.py (IMU <-> Rokoko <-> Orbbec clocks).
scripts/       fetch_assets.sh (re-clone K1 + G1 models).
docs/          THIS knowledge doc.
```
```bash
./scripts/fetch_assets.sh
./.venv/bin/python sim/g1_retarget.py     # real human swing on the G1 -> sim/g1_retarget.gif
/Applications/Blender.app/Contents/MacOS/Blender -b --python sim/fbx_extract.py -- <clip.fbx> data/mocap_clip4.npz
~/.local/bin/him validate
```

## 9. Standing constraints (operating rules)

Do not invent specs (signals, units, joint counts, licenses, evaluator behavior).
Edit only `.him` by default; ask before changing native project files and before
anything that uploads/spends/launches. Never weaken checks to pass validation.
Don't commit weights/data/creds. Keep this doc updated as the project's memory.
