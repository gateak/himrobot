# Project knowledge — himrobot (HIM humanoid table tennis)

This document gives the facts of the project. The project keeps this document
under version control. Change this document each time the team makes a new
decision or finds a new fact. Use this document for the facts of the project.
Do not use notes from outside the project.
_Last updated: 2026-08-29 (hitter v2 landed behind parity; deterministic checkpoint selection; scoreboard and noise tooling)._

---

## 1. Goal & HIM's intent

The goal is to make a table-tennis policy for **HIM Arena** on a humanoid
robot. HIM, the owner of the dataset, tells the team: *"We built the capture
rig and recorded the data ourselves. Use it to train perception, movement, and
control policies for humanoid table tennis."* So the multimodal human capture
**is the training set for this project**, not a problem to correct.

## 2. Current status & next steps

| Track | Meaning | Status |
|---|---|---|
| **Movement** | human swings → robot reference motions | 🟢 **G1, full body (29 DoF), all 8 clips**, done with GMR (`sim/gmr_bvh_to_g1_headless.py --ik_config sim/ik_configs/bvh_lafan1_rokoko_to_g1.json`). **Kinematic only** — no one has examined the dynamics of this motion, and no one has made sure that a real robot can do it. The earlier CSVs (`~/him-tools/out/all`, upstream offsets) had a mean wrist error of 50–69 cm and 23–31% joint-limit saturation. The corrected config (`work/g1/`) gives a mean wrist error of about 10–12 cm (`work/retarget/adjust2_rotfix_h1p75.md`, §4c). The team keeps the older right-arm-only IK (`sim/g1_retarget.py`) as a reference |
| **Perception** | ball / scene 3D (`ball_state`) from cameras | 🟡 The team tracks the orange ball in 2D and looks up its depth on all eight sessions (`sim/ball_detect.py`, `sim/ball_3d.py`, with results in `work/perception/summary.md`). The candidate rate is 23–47% per camera. Random-sample precision is 96/144 = 67%. Depth identifies the ball in 45–71% of hits. For session 215350, at the documented settings (gated, ROI margin 40 px), 26–32% of frames carry a candidate. About 1/2 to 2/3 of the sampled candidates are the real ball. The rest are sandals at the far edge. The team found ArUco DICT_4X4 IDs 0–7 at the half-table corners (`sim/aruco_scan.py`). **No intrinsics are on disk, so there is no metric XYZ yet.** See §6a. |
| **Control** | `joints + ball → desired joints` (the deployable hitter) | 🟡 First policy exists. `tt_sim/` is the fixed-base G1 MuJoCo dev scene with a local proxy evaluator (`python -m tt_sim.evaluate`). The model-based IK hitter (`tt_sim/policies/ik_hitter.py`) scores 60% legal returns on seeds 0 to 3 (10 trials each, paddle contact 8 of 10, landing error 0.7 m, reaction 0.5 s). The hold policy scores 0%. The `evaluate` entrypoint in `.him/him.yaml` runs it. Over 200 serves (seed 11) the hitter scores 46.5% legal, 3.5% target hits, 534 ms reaction. **RL v3 (2026-08-28):** residual PPO over the hitter (`tt_sim/rl/ppo.py --residual --residual-scale 0.2 --vel-penalty 1e-5 --legal-bonus 10`, 85 min on a HIM box) scores 53.5% legal / 6.5% target (seed 32) and 45.5% / 12.5% (seed 31) on the same 200 serves, with 24–35 ms reaction. Checkpoints: HIM dataset `himrobot-rl-checkpoints@0.1.29798103`, local `work/tt/rl/box/v3/`, write-up `work/tt/rl/v3/RESULT.md`. The v2 reward (velocity penalty 0.0005) made PPO swing slower (legal 47% → 3%); the intercept-shaped cold PPO stayed at 0–1%. **RL v4 (2026-08-28, box #3, 6 seeds × 425 min, ~11.5 M steps each):** no improvement — fresh seeds 36–49 % (s40 final.pt 51.5 %), seeds resumed from the 53.5 % checkpoint *fell* to 32–48.5 % while their noisy train-time rate rose 22 → 31 %. Cause: PPO optimizes returns under its own exploration noise (log_std −1.6, never annealed) and `best.pt` was selected by a 20-episode noisy window. Fix (commit 381faea): `--det-eval-episodes N` selects `best.pt` by a mean-action evaluation on N fixed serves (logged as `det_legal_rate`), `--log-std-init`/`--log-std-final` anneal the noise. Write-up `work/tt/rl/v4/RESULT.md`. Also landed (f66c109, c23c6a1): serve-range params + `serve_scale`, per-episode `physics_jitter`, `--obs-noise/--joint-noise/--obs-delay-steps`, a `train` entrypoint in `.him/him.yaml`, and the per-serve parameter residual (`tt_sim/rl/param_residual.py`, `tt_sim.rl.param_train`, `ParamResidualPolicy`; K=10 bounded offsets on the IK hitter, step-zero parity tested). **Parameter residual result (2026-08-29):** `python -m tt_sim.rl.param_train --n-envs 32 --minutes 45 --vel-penalty 1e-5 --legal-bonus 10` on the Mac (no credits) → **67.5 % / 68.0 %** legal (seeds 0/1) on 200 serves, confirmed 67.5 % / 68.0 % on two unseen 200-serve sets where the hitter scores 46.0 % / 46.5 %; best.pt ≈ final.pt (no drift); target hits still ~1 %. Write-up `work/tt/rl/param/RESULT.md`. This is the policy to package next (`ParamResidualPolicy` into `him_adapter`; needs ball state in the runtime contract). **Best policy (2026-08-29): `work/tt/rl/param/s3/best.pt` = `assets/policies/param_residual_s3.npz`, 70.5 / 70.5 / 71.0 % legal, 10 % target, 0.52 m on seeds 11 / 7 / 3.** **Hitter v2 (2026-08-29, commit pending):** swing library (five templates blended by four rectified gates), coarse hitting-plane depth, reach expansion, face-weight, dwell and swing-direction knobs — a 20-knob `v2` parameter space on top of the unchanged 10-knob `v1` (`--param-space v2`, warm start from a v1 checkpoint via `--resume-from`). Nominal behaviour is bit-identical (golden fixture `tt_sim/tests/data/param_s3_actions.npz`; hitter 42.5 %/26 contacts on 40 serves seed 7; v1 policy 70.5 % on seed 11). Measured: exploration noise on the ten new knobs is destructive (std 0.37 → 22 % during training; use `--log-std-init-new -2.5/-3.0`); the reach-expansion and face-weight knobs change nothing on the nominal serve set; the crossbody template ships without its +y box (the solver self-collides there) and without yaw biases. Write-up `work/tt/rl/param_v2/RESULT.md`, design `work/tt/design/hitter_v2.md` (with an implementation-deviations section). **First v2 training (nine warm-started runs, 20 000 episodes each, Mac): legal returns did not improve on unseen serves (v1 73.8 % vs best v2 73.0–73.5 % on seeds 2000/4000, ±3 pts), but the placement-reward run `wa0/final` (`--aim-bonus 8 --aim-sigma 0.30 --target-hit-bonus 4`) doubles target hits (12 → 26.5 %) and cuts landing error (0.50 → 0.39 m) at the same legal rate; on seed 11: 73.5 % / 30.7 % / 0.39 m. Shipped as `assets/policies/param_residual_v2_wa0.npz` (the `evaluate` entrypoint's default weights). A placement phase 2 (target-first selection, 200-serve selection set, 2 seeds × 8 000 episodes, 2026-08-30) scored 72.5–73.5 % / 27.8–29.5 % / 0.37–0.38 m — the same as wa0 within noise: placement is saturated near 28–30 % with this hitter. The ten new knobs were not discovered in 20 000 episodes (window-mean u within ±0.2 of nominal); the reach ceiling (20–24 % of serves unreachable with the pelvis welded) caps legal returns near 76–80 % whatever the knobs do. |

**Free-pelvis probe (2026-08-30, 20 serves, seed 2000+, `scene.build_model(fixed_base=False)` with the pelvis placed at
(−1.77, 0, 0.793)).** With NO balance controller — position actuators holding STAND_POSE for the legs and the shipped
v2 hitter driving waist + arm exactly as on the bolted pelvis — the G1 stayed upright in 15/20 serves (5 falls) and made
paddle contact with the ball crossing the net in 14/20 (bolted reference ≈ 84 % contact, 76 % legal on these serves).
Pelvis drift at contact: ≈2 cm; the hitter's paddle-position belief (bolted-pelvis kinematics) was off by 11 cm median,
40 cm when falling. Read: a "stand still and swing" whole-body policy is within reach of the current hitter plus a
base-pose-aware IK and a light balance controller; stepping/moving is the real whole-body problem (Isaac lane).

**Free base, scored (2026-08-30, 200 serves × seeds 11/2000/4000 = 600 trials per row, `tt_sim.evaluate` mean action).**
The probe's optimism did not survive scoring, and the base-aware IK it recommended **made things worse**. The option
shipped: `TableTennisEnv(fixed_base=False)`, `tt_sim.evaluate --free-base`, `eval_seeds --free-base`, an additive
`metrics["falls"]`, four `base.*` observation keys emitted only on the free base, and base-aware IK in the hitter
(`IKHitterPolicy(base_aware=True)` / `tt_sim.evaluate --base-aware`; off by default) — all of it gated so the welded scene is
bit-identical (welded reference re-measured at exactly 73.5 % / 28.5 % / 0.39 m). Numbers: the shipped
`param_residual_v2_wa0` scores **54.0 % legal with base awareness OFF and 39.3 % with it ON**, against 73.5 % welded;
the plain hitter 33.7 % OFF / 28.3 % ON against 45.3 % welded. Stand rates 80.7 % / 63.0 % (wa0) and 90.7 % / 80.0 %
(hitter). Why awareness loses even though it works: it *does* fix the kinematics — blade-to-target error at contact
drops 0.167 m → 0.066 m and contacts rise (52.9 % → 68.3 % among trials that stayed upright, for the hitter) — but
chasing the drifting pelvis keeps the arm tracking through the disturbance and feeds the reaction torque back into a
robot with nothing holding it up: pelvis-tilt p90 rises 58.9° → 73.4° and falls roughly double (19.3 % → 37.0 % for
wa0). Read: unwelding costs ~20 points, not ~5; base-aware IK is the right kinematics in the wrong order and should
stay OFF on the free base until a balance controller (Isaac whole-body lane) absorbs the arm's reaction, then be
re-measured there. Caveat: `fell` latches anywhere in the 3 s episode, including after a good return, so `falls` is
an upper bound. Write-up `work/tt/rl/free_base/RESULT.md`; ~5 min local CPU, no credits.

**Hitter v3 stance (2026-08-30) — where the robot stands is worth more than every hitter knob put together.**
Nine uniform knob directions through the real evaluator buy nothing (best: a 5 % slower swing, +1.25 points of legal
returns for −10.25 points of target hits); the hitter's own parameter space is exhausted. Sliding the **welded pelvis
0.10 m toward −y** — one constant, `scene.ROBOT_Y`, read once at import from `TT_ROBOT_Y`, plus a stance-relative
reach frame in `ik_hitter.py` (`STANCE_Y` / `_stance_y` slide `Y_REACH`, `REACH_MAX_Y` and the `lift`/`punch` y boxes;
x, z and every face/direction bias are untouched) — moves the **shipped, unretrained** policy this far
(`tt_sim.evaluate` mean action, 200 serves × seeds 11 / 2000 / 4000):

| policy | stance | seed 11 | seed 2000 | seed 4000 | legal / 600 | legal % | target % | err |
|---|---|---|---|---|---|---|---|---|
| `param_residual_v2_wa0` | y = 0.00 (today) | 147 | 153 | 141 | 441 | 73.5 % | 28.5 % | 0.39 m |
| `param_residual_v2_wa0` | **y = −0.10** | **161** | **154** | **152** | **467** | **77.8 %** | 30.5 % | 0.39 m |
| plain `ik_hitter` | y = 0.00 | 93 | 94 | 85 | 272 | 45.3 % | 2.2 % | 0.75 m |
| plain `ik_hitter` | y = −0.10 | 88 | 104 | 91 | 283 | 47.2 % | 2.2 % | 0.75 m |

+4.33 points, all three disjoint seeds up, McNemar exact two-sided p = 0.013 (64 trials gained, 38 lost); contacts
80.0 % → 87.8 %. **No retrain has happened yet** — the 38 lost trials and the 55 % conversion on the new contacts
(against 91 % on the ones the policy was trained for) are what a warm-started retrain at the new stance is for.
Read it as tuned to *this* forehand-only serve distribution: on a symmetric one it would cost as much as it gains,
and there is no left-arm backhand. Whether HIM's evaluator lets a policy choose its base pose is an open question
below. Also landed, both defaulting to today's behaviour exactly: `VecTT(depth_bonus=)` / `param_train
--depth-bonus`, a reward term that prices only the **landing-depth shortfall** (every legal return lands 0.276 m
short, and the aim knobs move it by 3 % of the requested amount because the swing-speed cap binds on 400/400
trials — this term's gradient points at `swing_speed_scale` instead); and `collision_guard`, a per-serve hitter knob
(`SERVE_PARAM_NOMINAL_V3`, **not** in any `PARAM_SPACES` entry — `K_PARAMS_V2` stays 20) that removes the real
latent defect of the solver driving the upper arm through the chest in the deep/inner band (86 of 950 poses at
x = −1.31, 306 at x = −1.50 → 0 / 5 with the guard on), worth 0 legal returns at either stance today and −8 points
of target hits when forced on, so it ships **off** as insurance for a widened serve set. Design
`work/tt/design/hitter_v3_reach.md`, write-up `work/tt/rl/v3_stance/RESULT.md`.

**HIM answers (2026-08-30, via Akbar).** Ethan: the `publishing_outputs` / `provider_reconciliation_timeout` job failure
is a legacy flow being cleaned up; the core upload loop should be stable — retry a `him push` + run and report any new
failure. Still open: the challenge version id, the evaluator's observation contract (ball state? base pose?), control
rate, trial count, and whether the challenge scene is free-standing (Akbar: "the bot needs to stand and move").

**CLI 0.1.7 — the guided-run flow is gone (2026-09-01).** HIM's newest SDK wheel is 0.1.7
(`https://arena.himrobotics.com/cli/him_sdk-0.1.7-py3-none-any.whl`; 0.1.8+ do not exist), now installed. It
**removes `him run` and `him job` entirely** (the functions remain in `cli.py` as unregistered dead code) — this is
the "legacy flow" Ethan meant; the `publishing_outputs` / `provider_reconciliation_timeout` deaths were that flow
being decommissioned. What remains: `him push` (careful: **visibility now defaults to public** — pass
`--visibility private`), `him watch <pkgrun_id>`, `him compute` (= `machine launch`), and `him arena
submit <challenge_version_id> --package <digest> --entrypoint <ep>`. `POST /api/v1/packages/<digest>/runs` still
answers but is a **contract rehearsal only** (provider `fake-local-package-v1`: "did not execute uploaded user
code or claim task performance" — pkgrun_196031dfa954bcf859cc). **So HIM's server no longer executes user code
outside an arena submission**; every other result is package-reported (run it yourself, report the numbers), which
matches Ethan's own entry. A
final legacy job (`job_0372111e7b28f1790d1d`, 2026-09-01) ran its simulation, reached `publishing_outputs`, and
became unreadable (404) under the new API — uncharged; a cancel attempt returned `cancel_window_closed`. The v3
package is pushed and approved-private under 0.1.7 (digest `sha256:14f2708283…`, commit 7e4c86b, all four declared
tests passed server-side).

**THE ENTRY PATH IS `him upload` (found 2026-09-01; it existed since CLI 0.1.3 and was missed).** The Table Tennis
challenge is **not** an arena community challenge — every candidate id 404s, including the open-build challenge's
own id, and no community-challenge listing endpoint exists. Site entries (Ethan's, Kyle Jeffrey's crawler) are
**source-upload listings**: `him upload <zip> --video <mp4> --name … --caption … --run-command …`, whose defaults
are literally `--challenge table-tennis-targeted-forehand --robot unitree-g1 --simulator mujoco`. Video is a
required MP4 ≤ 50 MB; the listing is public and quarantined (`/api/v1/source-uploads`). `arena submit` targets
user-created community challenges (via `him catalog create-challenge`) and is a dead end for this challenge today.
Stop looking for a challenge version id.

**ENTERED (2026-09-01).** `him upload` of hitter v3 published:
`https://arena.himrobotics.com/uploads/source_cfefeaf6ba8370a5e61d` (public, quarantined, downloads allowed).
Archive: `work/tt/entry/himrobot_v3_entry.zip` (weights `param_residual_v3_s0final.npz` + `param_residual_np.py`
+ README, 28 KB, sha256 a633644c…). Video: the 10-serve `summary.mp4` (9/10 legal, 4/10 target). Caption metrics:
**1,000 serves, seed 2000, mean action: 78.6 % legal / 47.0 % target / 0.288 m / 88.3 % contact / 452 ms.** The
listing's challenge object carries the internal id `challengev_tt_forehand_0_1_0`, which the community-challenges
API still rejects — it is not an `arena submit` target. No credits spent (uploads are free).

**What "submitting" is (2026-08-30, from Akbar reading the Arena page).** The table-tennis challenge page shows no
challenge version, interface or trial count; Ethan's own entry is a package-reported run: "Policy-driven fixed-base
MuJoCo demo for a 29-DoF Unitree G1 on one qualified center-lane feed, followed by measured 1,000-feed simulation
metrics. Simulation only; physical commissioning and the >=50 human-rally goal remain unverified." So the current
path is `him push` + `him run --entrypoint evaluate` (10 serves, video) and `evaluate-1000` (1000 serves, seed 2000,
metrics), not `him arena submit`. Standing/moving is a goal Akbar sets, not a published requirement; Isaac Lab has
not been used yet (`scripts/gpu_box_setup.sh` is ready, never run).

**Hitter v3 retrain + ship (2026-08-30 evening).** Two warm-started runs at the stance with `--depth-bonus 3`
settled the pre-registered bar: legal returns are stance-limited (best 79.8 % vs 77.8 % unretrained -- the welded-scene
reach ceiling ~81 % is effectively reached), and placement broke open: **s0/final = 77.0 % legal / 49.5 % target hits /
0.28 m** on seeds 11/2000/4000 (s1/final 77.5/46.0/0.31). **Shipped:** the `evaluate`/`evaluate-1000` entrypoints now run
`tt_sim.evaluate_stance` (sets `TT_ROBOT_Y=-0.10` + the v3 weights before the scene import) with
`assets/policies/param_residual_v3_s0final.npz`; `param_residual_v3_s1best.npz` is the legal-max alternative
(79.8/30.7/0.39). The welded default scene, `tt_sim.evaluate`, and every parity fixture are unchanged.
Full log `work/tt/rl/v3_stance/RESULT.md`.

**Serve sets (found 2026-08-29).** `tt_sim.evaluate` draws serve k from `default_rng(seed + k)`, so seeds 11 / 7 / 3
share 192–196 of 200 serves: every "confirmed on three independent sets" statement before this date was one set
measured three times. Independent sets need seeds ≥ `--trials` apart; the scoreboard is now **seeds 11 / 2000 / 4000**
(`tt_sim.rl.eval_seeds` refuses overlapping seeds) and the trainer's selection set is seed 1001 (serves 1001–1080,
disjoint from all three).

**Harness (2026-08-29).** `tt_sim.rl.param_train --det-eval-episodes N --det-eval-every M --det-eval-seed S` selects `best.pt` by the
mean action on N fixed serves scored exactly like `tt_sim.evaluate` (verified: s3 on seed 11 → 70.5 / 10.0 / 0.5202, identical);
keep S off 11/7/3. `train_log.csv` carries `det_*` columns, the window mean of every knob (`u_<name>`) and the blend fraction per
template (`blend_<template>`). `tt_sim.rl.eval_seeds --ckpt … --npz … --hitter --out DIR` is the scoreboard (seeds 11/7/3 × 200,
parallel, `scoreboard.md/json`). `tt_sim.evaluate` gained `--no-trajectory` and policy-side `--obs-noise/--joint-noise/--obs-delay-steps`
(recorded under `policy_obs` in `evaluator.json`; scoring untouched); `param_train` accepts the same three for training under noise.
Box scripts: `scripts/box_param_sweep.sh` (parallel sweep, 30-min pushes, self-stop) and `scripts/box_push_results.py --root --machine`.
`scripts/make_golden_param_actions.py --check` proves the current code still reproduces the golden fixture. Previous boxes: L4,
128 cores, 51 GB cgroup cap, 93 credits/h, 8 h max (744 credits); a `cpu` profile exists (price not yet seen).

**Pending / loose ends**
- ~~Migrate `.him/contracts/runtime.json` from the K1's 22 joints to the G1's 29.~~ Done 2026-08-26 (Menagerie/SDK order in `extensions.robot.joint_order`).
- Train a G1 tracking policy on the retargeted CSVs (mjlab / BeyondMimic recipe. This needs an NVIDIA machine — the CSV format that `sim/gmr_bvh_to_g1_headless.py` gives is the Unitree/LAFAN1 CSV that those recipes take as input).
- Make sure that the contract's signal names and rate agree with HIM's server-side evaluator (not yet confirmed).
- ~~Time-align the three capture streams (separate clocks) before the control track.~~ Done 2026-08-26 (`sim/align_streams.py`, §6b). The IMU-to-Rokoko lag per session comes from cross-correlation (|gyro| against hand angular speed). The 7 play sessions show r 0.36–0.64, with lags −6.0…+0.9 s. The 17 s test take, 214615 against Clip, shows r 0.987 alone, with lag +6.8 s. The IMU-to-Orbbec lag comes from the wall clock: both hosts agree to the second. HIM's America/New_York UTC guess is 3 h off the Orbbec host's UTC−7. The team ran this again on 2026-08-27 with **all 8 Orbbec sessions on disk**. First frame minus the paired IMU episode start is −1.9…+5.4 s for the 7 play sessions (+20.9 s for the 5 s test take 214619). Frame spans agree with the IMU durations within −9.3…+5.5 s (start/stop offsets, not drift). The `_60fps` sessions in fact recorded at **about 19 fps, effective** (`work/alignment/orbbec-imu-offsets.md`). Open item: Rokoko has no absolute clock, so it is anchored only through the IMU.
- Compute at scale still has a gate on it (see §4). Local MuJoCo is the path with no gate.

## 3. Environment & tooling

- The `him` CLI is `him_sdk 0.1.3`, in a **Python 3.12** venv at `~/.local/share/him/venv` (binary `~/.local/bin/him`). The default endpoint is `http://127.0.0.1:5173`. This endpoint does not operate. **Always** run `export HIM_API_URL=https://arena.himrobotics.com` first. The team is logged in (account `him_pat_…`, credentials at `~/.config/him/credentials.json` — do not read or commit this file).
- The main development venv is `.venv/` (`/usr/local/bin/python3.12` → 3.12.8). It has torch 2.13.0, mujoco 3.12.0, imageio 2.37.4, imageio-ffmpeg 0.6.0, numpy 2.5.2, and pytest 9.1.1 (installed 2026-08-26).
- For FBX reading, the team uses the **Blender binary** (`/Applications/Blender.app/Contents/MacOS/Blender`), version 5.2.1 LTS since 2026-08-27 (version 3.4.1 before that). The BVH export from both versions is the same to within ≤7e-5. The team made sure of this. The Blender binary has a bundled Python 3.10.8 with numpy 2.2.6, and `io_scene_fbx` is on. Command: `Blender -b --python sim/fbx_extract.py -- <clip.fbx> <out.npz>`. No `bpy` wheel and no `.venv-fbx` are necessary (the team discarded the old side-venv path). The importer leaves the scene range at 1..250, so `fbx_extract.py` gets the frame range from the armature action instead (Clip_4 = 9569 frames at 30 fps).
- Other tool venvs are outside the repo:
  - `~/him-tools/mujoco-venv` — python 3.12.8: mujoco 3.12.0, imageio 2.37.4, imageio-ffmpeg 0.6.0, numpy 2.5.2, pillow 12.3.0, **scipy 1.18.1 and matplotlib 3.11.1**, for `sim/align_streams.py` and `sim/retarget_report.py`.
  - `~/him-tools/cv-venv` — python 3.12.8: numpy 2.5.2, opencv-contrib-python 5.0.0.93, imageio 2.37.4, pillow 12.3.0, used by `sim/ball_detect.py`, `sim/ball_3d.py`, `sim/aruco_scan.py`, `sim/orbbec_video.py`.
  - `~/him-tools/gmr-venv` — GMR, torch, mujoco, mink.
  - `~/him-tools/GMR` — the upstream copy, read-only.
  - pyenv 3.11.9, at `~/.pyenv/versions/3.11.9`.
- Vendored assets are git-ignored. Download them again with `scripts/fetch_assets.sh`. The assets are `third_party/booster_assets` (K1, 49 MB) and `third_party/mujoco_menagerie` (a sparse checkout of `unitree_g1` only, 56 MB). A fallback copy of the G1 is at `~/him-tools/mujoco_menagerie/unitree_g1`.
- `data/` (git-ignored) holds HIM's proprietary, licensed capture. Never commit this directory. On 2026-08-27, per a user directive, the team put the data together as real files, with no symlinks:
  - `him-table-tennis-paddle-imu-20260815/` (plus a `paddle-imu-samples.csv` copy that `sim/retarget_imu.py` reads)
  - `rokoko-motion-capture/` (8 FBX files)
  - `orbbec_20260815_214619/`, `orbbec_20260815_215350/` (extracted sessions)
  - `release/` (the 10 verified tar files from `him data download table-tennis`)
  - `manifests/` (prepared HIM dataset manifests, plus packed artifacts)
  - `preview/` (a Blender `.blend` file)
  - `g1/` (a copy of the corrected retarget CSVs from `work/g1`, which `training/` reads)
  - `mocap_clip4.npz`

  Derived BVH files are in `work/bvh` and `work/bvh_lafan1`. Tool venvs stay in `~/him-tools/`.

## 4. HIM platform mechanics

- **Data:** `him data download` only works for the **table-tennis** release. There is no catalog, and there is no robot obs→action dataset to get. `capture`, `dataset`, and `train` use LOCAL fixtures that you write (`him train rehearse` is a stub that always gives the same result — it does not do real training).
- **Simulator:** the simulator runs only on the server. There is no local simulator. To run a policy against a simulated robot, use `him arena submit <challenge_version_id> --package <approved_pkg> --entrypoint <ep>` (the Community challenge, on isaac-sim, scored on legal-return, target-hit, landing-error, and latency), or use `him job launch simulation` (mujoco-simulation-contract, `mujoco_mjx`, gpu-standard) with `him job watch`. The output is **metrics and events, not video**.
- **Gating:** the server has a per-package worker gate (`package_worker_disabled`). This gate stops the real package now. The team's first `him run` failed, with status `reconcile-timeout`. A control run, with a built-in fixture, succeeded — this shows that the connection between the team's system and HIM's system operates correctly. HIM must open the gate before the team's real code can operate. **Update 2026-08-28:** the gate is open. Run #3 (`job_bba9697ea3091b2f53ba`) failed inside the simulation because the gitignored G1 model was not in the source snapshot; the model is now vendored as plain files in `assets/unitree_g1/` (a nested git clone is uploaded as a gitlink) and `tt_sim/__init__.py` probes for a MuJoCo GL backend that imports. Run #5 (`job_b314d2cca72c57d8306e`, 138 files, 15.9 MB) ran the simulation and reached `publishing_outputs`, then failed with `provider_reconciliation_timeout` (no charge); two retries failed the same way within minutes, and `him job watch` on it returns HTTP 400 "Compute attestation identity or digest verification failed" — a HIM-side issue to report with the job id. `him run` reuses the job when the lock digest is unchanged; pass `--idempotency-key` to force a new job. Package `my-policy@0.1.0`, id `pkg_110ce5d185fa9ba3d703`. Failed job `job_6be550b37b3337434d9a`.
- **Approval gates:** get the user's permission first for `him login`, `him run`, `him push`, `him publish`, `job launch`, `arena submit`, and `dataset push` — and for anything that uploads source code, spends credits, or starts compute resources.

## 4b. HIM datasets (private, registered from this machine 2026-08-26)

- `him-table-tennis-paddle-imu@1.0.0` → **`dataset_f410d4b7a08c0ab25d88`** (status ready, usable as `him run --dataset <id>`). The team added this from HIM's own manifest, with `license.spdx_or_name` added.
- Orbbec 214619 / 215350 and Rokoko mocap: the team prepared manifests, and `him dataset validate` shows no error, under `~/him-data/manifests/`. The team has **not** pushed these manifests — the license for locally captured data, and the Rokoko session timestamp, still need confirmation.
- `him capture inspect`, `dataset build`, and `train rehearse` take only `fixture_only` JSON in version 0.1.3. Someone must write raw-capture manifests by hand.

## 4c. Retargeting pipeline (`.him/contracts/training-pipeline.json`, `rokoko-to-g1-motion-tracking-v1`)

The pipeline has these stages:
- Blender headless FBX→BVH (`sim/fbx_to_bvh.py`, about 1 minute per clip).
- `sim/validate_motion.py`.
- Rename the Rokoko "Newton" bones to LAFAN1 names (`sim/rokoko_to_lafan1_bvh.py`: Spine1→Spine, Spine2→Spine1, Spine4→Spine2, Thigh→UpLeg, Shin→Leg). GMR looks bones up by name. Extra bones are permitted. cm and Y-up are expected.
- GMR `--robot unitree_g1 --format lafan1` (`sim/gmr_bvh_to_g1_headless.py`, about 350 fps on an M4 Pro CPU). `--jobs N` runs clips in parallel.
- Unitree CSV (root xyz, quat **xyzw**, 29 joints in SDK order).
- `sim/play_g1_csv.py` (live `launch_passive` or `--video`).

The team made sure of a full `him pipeline run` on 2026-08-26 (before the `--ik_config` change). After the contract change, the team ran the retarget stage command again on one clip and got the same CSV, byte for byte. It did not re-run the full pipeline. The full pipeline covers 8 clips, 59,685 frames, and 33 minutes of motion.
Tool venvs live outside the repo (`~/him-tools/{gmr-venv,mujoco-venv,GMR,mujoco_menagerie}`). Pass them with `--set gmr_python=… --set mujoco_python=…`.

**Retarget fidelity (2026-08-26, `sim/retarget_report.py`, reports in `work/retarget/`, CSVs in `work/g1/`).** `work/g1/` is the **corrected** set (adjustment 2, below), and `data/g1` and `training/` use it. `~/him-tools/out/all` is the **broken baseline** (upstream rotation offsets). The team keeps it only for comparison.
- **GMR internals** (`~/him-tools/GMR`, read-only): `utils/lafan1.py:load_bvh_file` gives a fixed `human_height = 1.75` (the `Head − lowest foot + 0.2` formula is commented out, not active). `GeneralMotionRetargeting.__init__` calculates `ratio = actual_human_height / ik_config["human_height_assumption"]` (1.8 in `bvh_lafan1_to_g1.json`). It multiplies every `human_scale_table` entry by this ratio (defaults 0.9 hips/legs, 0.75 arms → 0.875 / 0.729 at 1.75 m). The config path comes from `params.IK_CONFIG_DICT[src][robot]`. A user can change this path at runtime. `gmr_bvh_to_g1_headless.py --ik_config PATH --human_height H` does that. `rt.human_scale_table` is also a plain dict attribute. The team did not change anything in the upstream code.
- **BVH facts** (from the Blender export, 6 channels per joint): the position channels *replace* the OFFSET value (GMR's own `read_bvh` does exactly this. The team's own FK code in `retarget_report.py` agrees with GMR's loader to 1e-15 m. The non-root position channels equal their OFFSET lines. Bone lengths stay the same to 1e-7 m over all frames). Frame 0 of every clip is the rest T-pose, with all rotation channels near 0. The hips sit at **0.917 m** (root position channel 91.72 cm — the OFFSET line says 94.0 cm, but the position channel takes priority). The head joint is at 1.593 m, the lowest toe joint at −0.010 m, so the **head joint sits 1.603 m above the floor**. The `Head` and finger End Sites are not useful for a height measurement. The Head offset (0, −11.1, 0) points back to the Neck. Each fingertip End Site is the negative of its parent's offset. But the Toe End Sites are real: OFFSET (0, 0, −14.57 cm) from each `LeftToe`/`RightToe`. This is not the negative of the Toe's own (0, −6.24, +13.17) offset. So the correct stature is still **not available** from the BVH data. GMR's commented-out formula would give 1.803 m. Segment lengths:
  - Arm (shoulder to wrist): 0.554 m (27.1 + 28.3 cm).
  - Leg (hip to ankle): 0.865 m.
  - Shoulder: 0.487 m above the Hips.
  - Shoulder width: 0.412 m.

  The G1 (`right_shoulder_pitch_link→right_elbow_link→right_wrist_yaw_link`, Menagerie) measures 0.377 m, so the arm ratio is **0.68**, not GMR's 0.729.
- **Metric**: the right wrist (`RightHand` against `right_wrist_yaw_link`). The team used a pelvis-centered heading frame:
  - Origin: `Hips`/`pelvis`.
  - z: world up.
  - y: left hip minus right hip.
  - Torso: pelvis, not `torso_link`.

  The team scaled the human vector by 0.68. The team also computed a shoulder-centered variant. The robot FK is MuJoCo, on the Menagerie `scene.xml`. Joint-limit saturation is counted against GMR's own `g1_mocap_29dof.xml` ranges. These are the ranges its IK clamps to. They are tighter than the Menagerie ranges — for example, shoulder_pitch ≤ 1.149 against 2.67, elbow ≤ 1.7 against 2.09.
- **Root cause found**: the upstream `bvh_lafan1_to_g1.json` rotation offsets assume LAFAN1 bone frames. The Blender/Rokoko bones have identity rest rotations instead, so the pelvis orientation target is 120° wrong (the upstream offset is (0.5, 0.5, 0.5, 0.5). The correct, derived offset is (0.5, −0.5, −0.5, −0.5)). This wrong target makes the IK fight its own orientation tasks. **Baseline** (`~/him-tools/out/all`, 1.75 m, `work/retarget/baseline_1p75.json`): mean wrist error is **50–69 cm** (Clip 50.2, Clip_2 55.8). **23.0–30.8%** of joint-frames are at IK limits (per clip: `ik_model_total / (frames × 29)`). Feet sit 6–19 cm under the floor. Root z runs as low as 0.46 m. The robot elbow never goes past 59–67°.
- **Adjustment 1** — `--human_height 1.6032` with the upstream config (`work/g1/h1p60/`, `work/retarget/adjust1_h1p60.json`). This gave no gain: mean wrist error is 50.3 cm for Clip, 55.8 cm for Clip_2. Saturation is 25.2–33.0%. The pelvis sits lower (0.25–0.82 m). Floor penetration is worse.
- **Adjustment 2 (kept)** — `sim/ik_configs/bvh_lafan1_rokoko_to_g1.json`, made by `sim/ik_configs/make_rokoko_g1_config.py`. The rotation offsets are re-derived as `Rx(90)ᵀ·Rz(−90)·R_body(q_T)`, from the GMR model in a G1 T-pose (shoulder_roll ∓π/2, elbows +π/2). The team copied everything else from upstream. Height stays at 1.75 m. Mean wrist error **10.0–11.7 cm** (Clip 10.0, Clip_2 10.6, medians 9.4–11.2, max 16–25 cm, Clip_5 50 cm). Shoulder-frame error 8.9–11.6 cm. Saturation ranges from 60 joint-frames (Clip) to 4,521 joint-frames (Clip_7) — 0.3–1.7%, mostly at `ankle_roll`. Minimum foot z runs from +0.008 m (Clip) to −0.083 m (Clip_5). Root z is 0.53–0.89 m. The robot elbow range is 0–123°, against a human range of 0–130°. `work/g1/*.csv` and `work/retarget/{Clip_g1.mp4, Clip_4_g1_20s.mp4}` come from this run (Clip_4 is the best long clip: 10.1 cm mean, 988 saturated joint-frames. The segment is frames 5521–6121, the 20 s with the most wrist travel). Even at the 10 cm mark, the human shoulder sits 0.49 m above the hips, against 0.29 m on the G1. A per-bone scale about the root cannot correct this difference.
- Commands: `~/him-tools/gmr-venv/bin/python sim/gmr_bvh_to_g1_headless.py --bvh_dir work/bvh_lafan1 --out_dir work/g1 --ik_config sim/ik_configs/bvh_lafan1_rokoko_to_g1.json --jobs 5`. Then `~/him-tools/mujoco-venv/bin/python sim/retarget_report.py --bvh_dir ~/him-tools/bvh_lafan1 --csv_dir work/g1 --out work/retarget/report` (add `--measure_height clip.bvh` for the anthropometrics). The pipeline contract passes `--ik_config sim/ik_configs/bvh_lafan1_rokoko_to_g1.json --jobs 4` (updated 2026-08-27).
- The contract's per-stage `cache.key_fields` is a label only. The HIM local runner keys each stage's cache on its resolved command and parameters, plus its dependencies' `output_digest` (a hash of each dependency's *declared* outputs). So `sim/fbx_to_bvh.py`, `sim/rokoko_to_lafan1_bvh.py`, and `sim/validate_motion.py` write a per-clip source and output sha256 into their summary JSON files, for anyone who audits what actually changed.

## 5. The robot

- **Current target: Unitree G1** (user directive). 29-DOF, **7-DOF arm with wrist** (`right_shoulder_pitch/roll/yaw`, `right_elbow`, `right_wrist_roll/pitch/yaw`), hand at `right_wrist_yaw_link`. Model: `mujoco_menagerie/unitree_g1/g1.xml` (+ `g1_mjx.xml`, matches HIM's `mujoco_mjx`). The 7-DoF arm lets the IK place the wrist: about 10–12 cm mean wrist error with the corrected config, against 50–69 cm before (§4c). Kinematic only, with no dynamics check. The G1 also matches our contract's signal names.
- **Earlier reference: Booster K1** — this is the real robot behind HIM's proprietary "Adam" fixture (public, **BSD-3**, `github.com/BoosterRobotics/booster_assets`, `robots/K1/K1_22dof.{urdf,xml}`). 22 DOF, wrist-less 4-DOF arm — orientation IK on it looked "crazy", which is why the team moved to the G1. HIM's Adam manifest is `fixture_only`, `run_on_adam_enabled:false`.
- **Contract:** `.him/contracts/runtime.json` maps obs `robot.joint_position` to action `robot.desired_joint_position`, bound to **[29]** (G1) with the Menagerie/SDK joint order in `extensions.robot.joint_order`. Rate 50 Hz, fallback "hold". Signal names and rate still need confirmation against HIM's evaluator. `him validate` shows no error.

## 6. The data (`data/`, git-ignored, cleaned 2026-08-26)

This is raw human table-tennis capture, from three streams with separate
clocks. The IMU CSV carries wall-clock timestamps, and the Orbbec sessions
carry epoch seconds, so software can align these two streams. The Rokoko FBX
carries only an export `CreationTimeStamp` (two days after capture) and no
recording time, so the team anchors it through the IMU by cross-correlation
(§6b):
- **Paddle IMU** (`him-table-tennis-paddle-imu-*.tar`): WitMotion, a human-held paddle, nominal 11.11 Hz, effective about 10.1 Hz (§6b), 10 episodes, 20,741 samples. 12 real channels (accel×3 g, gyro×3 dps, euler×3 deg, mag×3 µT). `quaternion_*` and `chip_time` are all null. `firmware` and `battery` stay constant.
- **Orbbec RGB-D** (8 `orbbec_*.tar`): **2 synced cams** (cam0/cam1, a stereo pair), 640×480, color JPG plus depth PNG (mm, `align_depth_to_color:true`). The 60fps sessions are the rally sessions. The table has **ArUco markers at the corners**, so metric table-frame calibration is possible (solvePnP plus triangulation) with no factory intrinsics. **The ball is orange**, on a dark-blue table, so HSV color detection works (see `sim/ball_detect.py`). Watch for a second held or spare orange ball, plus clutter around the net post.
- **Rokoko mocap** (`rokoko-fbx.tar`): 8 binary FBX clips, with standard bone names (`Hips`, `Spine*`, `RightShoulder`→`RightArm`→`RightForeArm`→`RightHand`).
- There is **no robot data and no joint data** in the capture — `robot_version_id: null`. (Cleanup removed about 13 GB of browser-duplicate tar files, plus `.crdownload` partial files. The team kept one copy of each.)

### 6a. Perception status (2026-08-26 Track C, updated 2026-08-27 Track P)

**What is on disk (2026-08-27).** All eight Orbbec sessions are now real directories, `data/orbbec_20260815_*/`, taken from the ten verified tar files in `data/release/` (`data/release/extract.log`, extracted 2026-08-27). None of them is a 0-byte download any more. Per camera (`work/perception/sessions.json`):
  - `214619` — 125 rows, 5.3 s, 23.3–24.0 fps host (**no ball — a player adjusts a paddle**).
  - `215350` — 3,983 rows, 193.5 s, 20.6 fps (no `host_time_end` and no `frame_counts` — the recorder stopped early. It carries serves and rallies from about frame 100).
  - `214831` — 6,578 rows, 260.3 s, 25.3 fps.
  - `215750` — 6,554 rows, 326.0 s, 20.1 fps.
  - `220448_60fps` — 5,673 rows, 297.1 s, 19.1 fps.
  - `221035_60fps` — 5,803 rows, 301.8 s, 19.2 fps.
  - `221649_60fps` — 5,927 rows, 309.2 s, 19.2 fps.
  - `222542_60fps` — 5,773/5,772 rows, 306.8 s, 18.8 fps.

**The `_60fps` sessions are not 60 fps on disk.** The cameras did run at 59.9 Hz. The color `device_timestamp_us` steps 50.09 ms while the device frame `index` steps by 3. That is, 16.7 ms per device frame (`work/perception/device_index_rate.json`). But the recorder kept only about 1 of every 3.1 device frames (host `dt` median 50.3–51.0 ms, p95 61–64 ms, max 0.33–0.56 s). So the effective rate is 18.8–19.2 fps. This is *lower* than the 30 fps sessions:
  - 214831 kept 1 of every 1.18 frames, giving 25.3 fps.
  - 215350/215750 kept 1 of every 1.45–1.49 frames, giving 20.1–20.6 fps.

The `_60fps` sessions give no time advantage. The ball still jumps 50–80 px between recorded frames. **The camera labels do not point to the same serial number across sessions.** Sessions 214619/215350/220448/221649 have `cam0_CP0E8530008E` plus `cam1_CP0E8530008F`. Sessions 214831/215750/221035/222542 have `cam0_CP0E8530008F` plus `cam1_CP0E8530008E`. This depends on USB enumeration order. So identify each camera only by its serial number, never by the cam index. The physical cameras never moved. Serial …8E always sees ArUco IDs 4–7, and …8F always sees IDs 0–3. Marker centers stay within 3.7 px of 215350's centers in all eight sessions (`work/perception/aruco_compare.md`). Null rows are rare (0–9 for color, 0–5 for depth, per camera. `idx` stays continuous everywhere) — the recorder did not record that stream for that row, and the detector does not use these rows. Frame 10 of camera 1 in session 214619 has damage in the lower third of the picture.

**Camera geometry (from the frames, not from the metadata).** The two cameras are *not* a classic overlapping stereo pair. cam0 (`CP0E8530008E`) looks down one half of the table, with the net at its left image edge. cam1 (`CP0E8530008F`) looks down the other half, with the net at its right image edge. The only overlap is the net strip. ArUco: `DICT_4X4_50` decodes. Its 100/250/1000 supersets also decode. No 5×5, 6×6, 7×7, or AprilTag marker decodes. There are **4 markers at each half's four corners**:
  - cam0 sees IDs 4 (near/net, about (86,425)), 5 (far/net, about (190,101)), 6 (near/end-line, about (567,427)), 7 (far/end-line, about (472,86)).
  - cam1 sees IDs 0 (near/end, about (87,408)), 1 (far/end, about (203,97)), 2 (near/net, about (564,435), partly behind the net post), 3 (far/net, about (481,103)).

Over 12 sampled frames (`work/perception/aruco_orbbec_20260815_215350.json`, `n_frames`), the decode counts are: cam0 IDs 4 → 12, 6 → 12, 5 → 9, 7 → 3. cam1 IDs 0 → 12, 3 → 8, 1 → 7, 2 → 3. The far markers, at about 20 px, decode in 3 to 9 of 12 frames. The near markers decode in 12 of 12, except ID 2 (at the net post), which decodes in 3 of 12. The median pixel corners for each ID are in `work/perception/aruco_orbbec_20260815_*.json`. The claim in §6, that markers sit at the table corners, is **confirmed** per half. The markers on the far half are visible across the net, but the detector does not decode them. The marker's physical size, and its exact offset from the table corner, are **not yet known** — both values are necessary for solvePnP.

**2D detector (`sim/ball_detect.py`).** This uses an HSV orange mask plus roundness. It gates candidates with:
  - Frame differencing (`--motion 8`).
  - The ArUco-hull table ROI (`--roi-json`, `--roi-margin`).
  - A static-orange map (`--static-reject 0.2`), which removes the bottle by the stool and a small orange object on the floor near the far player.
  - Distance from large red or orange blobs (`--near-big 12`, the paddle rubber).

It reads a directory or the tar file, with `--cam cam0|cam1`. The team measured ball color in OpenCV HSV: the median over a disc of 0.6× the detected radius, around the `ball2d` center. This was for session 215350 cam0, checked again on 2026-08-27 with `~/him-tools/cv-venv`. Two sets of hits show:
  - Six confirmed ball hits (frames 255/1799/2518/2954/3058/3638): H 18–20, S 159–194, V 213–234.
  - Four sandal hits (frames 2067/2246/2406/2623): H 16–17, S 112–125, V 99–145.

The hue values overlap. Over the full detected disc, the motion-blur at the ball's edge pulls its median S down to 97–122 — into the sandal range. So color alone cannot tell the two apart. Rates over the full sessions (gated, ROI margin 40 px):

| session / cam | frames | candidates | sampled precision |
|---|---|---|---|
| 215350 cam0 | 3,980 | 1,270 (31.9 %) | 8/12 real ball (4 = far player's sandals) |
| 215350 cam1 | 3,978 | 1,043 (26.2 %) | 5/12 real ball (7 = sandals/feet at the far edge) |
| 214619 cam0 | 123 | 20 (16.3 %) | 0/20 — all the far player's sandal |
| 214619 cam1 | 121 | 0 | — |

The team ran six new sessions on 2026-08-27 (Track P), with the same gated
settings and `--step 1` (every frame). Each session has its own ROI file
(`work/perception/roi_orbbec_20260815_<s>.json`). The 12-frame ArUco sample
misses ID 2 on serial …8F in sessions 214831 and 221035, and misses ID 1 in
222542. For those, the team filled the corners in from 215350's medians for the
same serial. The driver is `work/perception/run_track_p.sh`, 35–43 s per session per
camera. All numbers are in `work/perception/summary.md`, `summary.json`, and
`precision_counts.json`. Precision here means 12 random candidates per
camera, seed 0, checked by eye on
`precision_check_<s>_<cam>_random12.jpg`. "Depth resolved" means the disc
median is more than 40 mm nearer than the annulus around it:

| session / cam (serial) | frames | candidates | depth resolved | random-12 real / FP (FP kinds) |
|---|---|---|---|---|
| 214831 cam0 (…8F) | 6,574 | 1,506 (22.9 %) | 881 (59 %) | 9 / 3 (sandal 2, foot 1) |
| 214831 cam1 (…8E) | 6,572 | 1,608 (24.5 %) | 1,102 (69 %) | 8 / 4 (sandal 3, hand 1) |
| 215750 cam0 (…8F) | 6,545 | 1,896 (29.0 %) | 1,221 (64 %) | 10 / 2 (sandal 1, hand 1) |
| 215750 cam1 (…8E) | 6,552 | 2,796 (42.7 %) | 1,587 (57 %) | 5 / 7 (sandal 7 — one standing far player's sandal at ≈(192–206, 80) px) |
| 220448_60fps cam0 (…8E) | 5,672 | 2,232 (39.4 %) | 1,029 (46 %) | 6 / 6 (sandal 6) |
| 220448_60fps cam1 (…8F) | 5,672 | 1,476 (26.0 %) | 970 (66 %) | 12 / 0 |
| 221035_60fps cam0 (…8F) | 5,803 | 2,117 (36.5 %) | 1,508 (71 %) | 10 / 2 (hand/paddle 1, sandal 1) |
| 221035_60fps cam1 (…8E) | 5,803 | 1,893 (32.6 %) | 1,075 (57 %) | 9 / 3 (sandal 3) |
| 221649_60fps cam0 (…8E) | 5,925 | 2,003 (33.8 %) | 1,304 (65 %) | 6 / 6 (wrist above a black sleeve 5, sandal 1) |
| 221649_60fps cam1 (…8F) | 5,926 | 2,793 (47.1 %) | 1,259 (45 %) | 6 / 6 (sandal 5, net post 1) |
| 222542_60fps cam0 (…8F) | 5,769 | 1,826 (31.6 %) | 963 (53 %) | 7 / 5 (sandal 4, hand 1) |
| 222542_60fps cam1 (…8E) | 5,769 | 2,126 (36.9 %) | 1,499 (71 %) | 8 / 4 (sandal 3, hand 1) |

**Every one of the six new sessions has rallies in both cameras.**
Consecutive-frame candidate pairs that move 15–150 px number 777–1,291 per
camera. Runs of 6 or more such pairs number 22–92 per camera. No 30 s
window, in any camera, drops below a 0.079 candidate rate (0.0789, in
214831 cam1's final partial window, from the `summary.json` timeline). Over all 144
sampled candidates, 96 (66.7 %) are the real ball. The 48 false positives break down as follows:
  - 37 sandals or feet at the table edges.
  - 10 hands, wrists, or paddle edges.
  - 1 net post.

These are the same failure types found in 215350. So the next gain in
precision should come from a filter for sandals, skin, or track consistency,
not from the color band. Median disc depth is 1,451–2,000 mm. Depth
identifies the ball in 45–71 % of candidates, which agrees with 215350. The
median candidate radius is 6.4–9.2 px. The detector operates at about 240
frames per second, in one process, so a full pass over a session needs no
stride. Note: the file
`ball2d_orbbec_20260815_215350_cam0_CP0E8530008E.csv` was found overwritten
by a 5-row test run (09:24, 2026-08-27). The team made it again with the
documented command, and got the same result, 1,270/3,980
(`log_gated_215350_cam0_rerun.txt`).

With `--roi-margin 15`, the 215350 counts drop to 860 (21.6 %) / 631 (15.9 %)
(`work/perception/roi15/`). This comes at the cost of missing balls in the air past the
far edge. Ungated color-only detection gives near 100% "candidates" (pure
clutter). The old figure of "119/121 frames in a rally" has no clear source.
Ungated 214619 cam1 (a session with **no ball in flight**) gives exactly
119/121 (`work/perception/nomotion/log_214619_cam1.txt`). So that figure
described clutter, not a rally. Tracks are in
`work/perception/ball2d_<session>_<cam>.csv` (frame, host_time_s, u, v,
radius. Blank when nothing is found). A ball in flight moves 50–80 px per
frame at 20 fps and shows motion blur. The net rally length in the images is
limited by the roughly 20 fps rate.

**Depth (`sim/ball_3d.py`).** The depth PNGs are plain 16-bit greyscale
(`I;16`, no PNG text chunks, no EXIF data in the JPGs). `depth_png_unit` is
`millimeters_uint16`, `depth_scale` is 1.0 in every row, and about 99% of
values are non-zero. Median depth inside the ball disc, against an annulus
around it:
  - For 215350 cam0: the median disc depth is 1,649 mm. The disc reads more than 40 mm nearer than the annulus in 67% of hits.
  - For cam1: 1,754 mm, 50% of hits.

So the aligned depth **does identify the ball roughly half to
two-thirds of the time**. The rest of the time it reads the table behind the
ball (and false positives read the floor, at about 3.5 m). Output:
`work/perception/ball3d_<session>_<cam>.csv` (adds depth_mm, n_valid,
bg_depth_mm).

**What blocks metric XYZ / `ball_state`.** No intrinsics exist anywhere. These sources all lack fx, fy, cx, cy, the distortion values, and the cam0-to-cam1 extrinsics:
  - `session.json`.
  - `frames.jsonl`.
  - The tar file (which holds only images and the two JSON files).
  - The PNG/JPEG headers.
  - The HIM manifests (`calibration: []`, "No intrinsics/extrinsics available").

There are three ways to get them, and the team has done none of them yet:
  - (a) `pyorbbecsdk` (not on PyPI — build it from `github.com/orbbec/pyorbbecsdk`), with the same two cameras plugged in. Use `pipeline.get_camera_param()` on serials `CP0E8530008E` and `CP0E8530008F` at the 640×480 color profile (the aligned depth shares the color intrinsics).
  - (b) Checkerboard or ChArUco calibration of each camera at 640×480.
  - (c) If someone measures the ArUco marker edge length and the marker positions on the table, `solvePnP` per camera gives the table-frame pose. With 4 or more non-coplanar views, or `calibrateCamera` on the 8 markers over many frames, this also gives an estimate of the intrinsics. This is a weaker method, but it uses only the existing footage.

The team also still needs:
  - The marker size and table offsets.
  - The cam0-to-cam1 relation (through the shared table frame).
  - The clock offset between the Orbbec host clock and the IMU/Rokoko clocks.

Tooling: `~/him-tools/cv-venv` (python 3.12.8: numpy 2.5.2,
opencv-contrib-python 5.0.0.93, imageio 2.37.4, pillow 12.3.0). All outputs
are under `work/perception/` (git-ignored).

**Self-calibrated intrinsics (2026-08-27, `sim/estimate_intrinsics.py`,
results in `work/calib/`).** The team used the four `DICT_4X4_50` markers, plus the
aligned depth, plus the ITTF table size, per serial. From these, the team found the
camera height above the table (1.9 m, both serials) and the table plane to
about 1°. The team also found a focal length whose value depends on an unknown scale. The
team did **not** find the focal length on its own. A single view of one flat
rectangle cannot separate it from the unknown marker inset or size. The free
fit drifts (fy≈300, cy≈0). Changing the inset from 0 to 15 cm moves fx from
416 to 531 px. The change in the residual is less than 1 mm
(a verifier re-derived this). Use the center-fixed, square-pixel variant:
  - Camera …8E gives fx≈fy≈444 px.
  - Camera …8F gives fx≈482–501 px.
  - Both are about 67° HFOV, a possible value for an Orbbec RGB camera.

This has a ±15% scale caveat.
`sim/ball_3d.py --intrinsics work/calib/intrinsics_<serial>.json` then gives
X/Y/Z, in the camera frame and in the table frame. One tape measurement would settle this scale
question to within about 1%. This could be a measurement of a marker's printed edge
(the fit gives about 9.3 cm), or of its distance from the table corner. The camera model
(serial prefix CP0E85…) is still not confirmed.

### 6b. Stream alignment (2026-08-26, Track D, Orbbec part re-run 2026-08-27 with all 8 sessions)

Script: `sim/align_streams.py` (run it with
`~/him-tools/mujoco-venv/bin/python`, which now also has scipy 1.18.1 plus
matplotlib 3.11.1). By default, it finds every `data/orbbec_*/` directory
(sorted, though `--orbbec-dir` overrides this) and records each
`data/release/orbbec_*.tar` file's byte size. Outputs are under
`work/alignment/` (git-ignored): `session-map.json`,
`imu-mocap-xcorr.csv` (all 10×8 pairs), `imu-orbbec-clock.json`,
`fbx-headers.json`, `xcorr_<episode>_<clip>.png`, `xcorr-matrix.png`,
`run.log`. `work/alignment/summarize_orbbec_clock.py` condenses the clock
JSON into `orbbec-imu-offsets.{csv,md}` (the table below). `--orbbec-motion`
adds the (inconclusive) video check. The 2026-08-27 re-run, with 8 sessions,
left `imu-mocap-xcorr.csv` the same, byte for byte, as the 2-session run
(`prev-imu-mocap-xcorr.csv`, sha256 `55555e44…`). Everything below comes from
the files. The team assumed nothing.

**Clocks.** The IMU CSV `timestamp_local` is the raw iOS wall clock, with no
zone marked. HIM's `timestamp_utc` equals local time plus 4 h
(America/New_York — stated as a *guess* in `capture-manifest.json`). The
Orbbec `session.json` and `frames.jsonl` carry epoch seconds. The session
directory name is the host wall clock at the moment the session starts.
`orbbec_20260815_214619` gives a `host_time_start` of 04:46:36.48 UTC.
`215350` gives 04:53:55.87 UTC — that is, the Orbbec host showed **UTC−7**.
Camera setup took 17.5 s and 5.9 s, between the session start and `host_time_start`.
The IMU host and the Orbbec host show the **same wall clock**.
Orbbec session 215350's frames run 21:53:56.43–21:57:09.97, and IMU episode
215358 runs 21:53:58.29–21:57:09.61. They start 1.9 s apart, and stop 0.4 s
apart, with 191.3 s of overlap. Orbbec session 214619
(21:46:36.83–21:46:42.14) falls inside IMU episode 214615
(21:46:15.93–21:46:43.50). The first frames of the other six sessions fall
−1.9…+5.4 s from their IMU episode starts (see the table below). So the "3 h"
gap between HIM's `timestamp_utc` (01:46 Z) and the Orbbec `created_utc`
(04:46 Z) is exactly UTC−4 against UTC−7. The two hosts agree on wall time,
and the Orbbec host's zone setting was UTC−7. **Two explanations both agree with the
data, and the files cannot show which one is true.** Either the IMU's
America/New_York guess is wrong by 3 h (the true UTC equals `timestamp_local`
plus 7 h). Or the Orbbec host's epoch clock was 3 h off, while both hosts
shared one zone setting. But for alignment purposes this does not matter:
align on wall clock
(`timestamp_local` against `host_time_s` minus 7 h. `imu-orbbec-clock.json`
has every offset). The remaining Orbbec-minus-IMU offsets are recorder
start/stop times, not drift (`work/alignment/orbbec-imu-offsets.md`, for the
first camera of each session. The second camera's first frame comes 0.07–0.17 s
later, and its last frame comes 0.01–0.04 s later):

| Orbbec session (tar bytes) | cam0 / cam1 serial | frames/cam | frame span s | eff. fps (prefer) | IMU episode (dur s) | first frame − IMU start s | last frame − IMU end s | span − IMU dur s | inside IMU |
|---|---|---|---|---|---|---|---|---|---|
| 214619 (59,699,200) | …8E / …8F | 125 | 5.315 | 23.5 (30) | 214615 (27.571) | +20.897 | −1.359 | −22.256 | yes |
| 214831 (3,175,403,520) | …8F / …8E | 6,578 | 260.330 | 25.3 (30) | 214836 (257.341) | −1.901 | +1.088 | +2.989 | no (starts 1.9 s early) |
| 215350 (1,910,609,920) | …8E / …8F | 3,983 | 193.540 | 20.6 (30) | 215358 (191.311) | −1.866 | +0.363 | +2.229 | no (starts 1.9 s early) |
| 215750 (3,139,819,520) | …8F / …8E | 6,554 | 325.966 | 20.1 (30) | 215749 (335.283) | +5.394 | −3.923 | −9.317 | yes |
| 220448_60fps (2,738,247,680) | …8E / …8F | 5,673 | 297.057 | 19.1 (60) | 220452 (301.526) | +3.779 | −0.690 | −4.469 | yes |
| 221035_60fps (2,808,535,040) | …8F / …8E | 5,803 | 301.837 | 19.2 (60) | 221043 (300.932) | +0.456 | +1.361 | +0.905 | no (ends 1.4 s late) |
| 221649_60fps (2,871,398,400) | …8E / …8F | 5,927 | 309.195 | 19.2 (60) | 221654 (303.722) | +0.776 | +6.249 | +5.473 | no (ends 6.2 s late) |
| 222542_60fps (2,813,112,320) | …8F / …8E | 5,773 / 5,772 | 306.769 | 18.8 (60) | 222546 (311.222) | +4.249 | −0.204 | −4.453 | yes |

Wall-clock first frames: 214831 21:48:34.43, 215750 21:57:54.93, 220448
22:04:56.35, 221035 22:10:43.72, 221649 22:16:55.35, 222542 22:25:50.76 (all
on the UTC−7 host wall clock. `implied_host_utc_offset_h` is −7 for every
session. The gap from session start to `host_time_start` is 2.9–8.3 s, except
17.5 s for 214619). The `_60fps` sessions' `frames.jsonl` spans agree with
their IMU episode durations to within −4.5…+5.5 s. The first/last-frame
offsets explain the whole-session mismatch — for example, session 221649
started 0.8 s after the IMU episode, and stopped 6.2 s after it. The data
shows no clock drift at this resolution. **The target `fps_prefer: 60` was
not reached**: the four `_60fps` sessions have a median frame dt of
0.050–0.051 s (18.8–19.2 fps, effective. 5,673–5,927 frames, over about 300
s), and the 30 fps sessions have a median frame dt of 0.037–0.050 s
(20.1–25.3 fps). `session.json`'s `frame_counts` equal the `frames.jsonl` row
counts for the 7 sessions that have this field (215350 stopped early and has
none). Camera labels follow USB enumeration order, **not** the serial number.
Sessions 214831, 215750, 221035_60fps, and 222542_60fps have
`cam0_CP0E8530008F` and `cam1_CP0E8530008E` — the reverse of the other four.
So identify cameras by serial number, not by `cam0`/`cam1`. An independent
check used frame-difference motion energy from the Orbbec color stream
against |gyro|. This check gives no clear answer. For 215350, r ≈ 0.10 at the expected
lag, with no peak above z 2.7 in ±60 s, on both cameras. The whole-frame
signal is dominated by the other player and the ball. The team repeated this
on 2026-08-27, on session 220448_60fps (all 5,673 color JPGs per camera, with
no frame subsampling. Results in `work/alignment/motion60/`): camera 1 (…8F) has its
best lag at **+4.10 s, against an expected +4.13 s** (r 0.195, z 2.8,
secondary ratio 0.83). Camera 0 (…8E) shows r −0.23 at the expected +3.99 s,
with a false peak at +58.0 s (r 0.18, z 2.5). Camera 1's result supports the
shared-wall-clock rule. But neither camera clears the script's own bar for a
clear answer (z ≥ 4, secondary ratio < 0.6). So the result stays
inconclusive.

**Rokoko FBX headers** (binary FBX 7500, parsed node by node).
`CreationTimeStamp` runs 2026-08-17 13:08:23 (Clip) through 13:31:04
(Clip_8) — that is, export time, two days after capture. This is in the export
host's own, unstated time zone. `Original|DateTime_GMT` and
`LastSaved|DateTime_GMT` are empty. One take is named `clip` (`clip.tak`).
Take length equals clip length (17.43, 252.87, 192.90, 318.93, 295.37,
298.60, 305.23, 307.90 s). `TimeMode` is 6, meaning 30 fps. The data is Y-up,
in cm. Clip_8 was exported from a Rokoko folder named `Clip (1)`. **No
recording timestamp exists in the FBX file**. The team anchors Rokoko only
through the IMU.

**IMU-to-Rokoko method.** Frame 0 of every Blender BVH file is a T-pose (all
rotation channels are 0). The team drops it, so **clip time t = 0 is BVH
frame 1**. Every lag below uses this origin. The BVH FK uses standard
semantics, with the 6-channel position values *replacing* the OFFSET value
(the Blender exporter writes rest-offset plus translation). The team checked
this against Blender's own `pose_bone.head`, on Clip.fbx (exact for Hips,
mean 1.2 cm and max 2.7 cm for RightHand). The signal is the RightHand
**world angular-speed magnitude** (from the FK rotation matrices), against
IMU |gyro|. This is the same physical quantity, for a paddle held rigidly (p99: hand
520–615 dps, IMU 670–740 dps). The team kept linear hand speed as the
`alt_feature` (this gives a lower r everywhere). The team resampled both signals
to 10 Hz, with a Gaussian σ of 0.4 s. The team z-scored and clipped them at ±4σ. The team
took the Pearson r over the overlap, for lags of ±60 s (with a minimum 15 s
overlap). Lag sign: clip time t lines up with IMU episode time t + lag. A
negative lag means the IMU recording started after the clip.
`secondary_ratio` is the best local maximum more than 5 s away, divided by
the peak. `lobe` is the lag span where r ≥ 0.8 of the peak. `peak-match` is
the most common value among pairwise swing-peak differences (peaks over 400
dps, σ 0.1 s).

| IMU episode (wall, s) | Orbbec session | Rokoko clip (s) | lag s | r | 2nd ratio | z | lobe s | peak-match lag | runner-up clip (r) | confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 214615 (21:46:15, 27.6) | 214619 (on disk, 5.3 s inside, +20.9 s) | Clip (17.5) | +6.8 | 0.987 | 0.39 | 2.3 | 1.1 | — (no peaks >400 dps) | — | test take: four slow paddle wobbles match exactly, but only ~17 s of signal |
| 214836 (21:48:36, 257.3) | 214831 (on disk, first frame −1.9 s) | Clip_2 (252.9) | −3.8 | 0.636 | 0.32 | 4.3 | 2.1 | −4.2 | Clip_5 (0.244) | high |
| 215358 (21:53:58, 191.3) | 215350 (on disk, 193.5 s, −1.9 s) | Clip_3 (192.9) | +0.9 | 0.484 | 0.46 | 3.6 | 1.7 | −14.3 (disagrees) | none by duration | medium |
| 215749 (21:57:49, 335.3) | 215750 (on disk, +5.4 s) | Clip_4 (319.0) | −1.2 | 0.446 | 0.41 | 3.9 | 2.0 | −1.3 | Clip_7 (0.223) | medium |
| 220452 (22:04:52, 301.5) | 220448_60fps (on disk, +3.8 s) | Clip_5 (295.4) | +0.3 | 0.452 | 0.47 | 3.5 | 2.7 | −19.4 (disagrees) | Clip_7 (0.233) | medium |
| 221043 (22:10:43, 300.9) | 221035_60fps (on disk, +0.5 s) | Clip_6 (298.6) | −2.7 | 0.364 | 0.81 | 3.0 | 3.0 | −28.9 (disagrees) | Clip_4 (0.201) | medium (weakest) |
| 221654 (22:16:54, 303.7) | 221649_60fps (on disk, +0.8 s) | Clip_7 (305.3) | −4.8 (twin peak −1.6, r 0.39, linear-speed feature picks −1.6) | 0.422 | 0.51 | 2.7 | 4.2 | −1.8 | Clip_5 (0.196) | low (lag ambiguous between −4.8 and −1.6) |
| 222546 (22:25:46, 311.2) | 222542_60fps (on disk, +4.2 s) | Clip_8 (307.9) | −6.0 | 0.624 | 0.46 | 4.3 | 2.7 | −6.2 | Clip_5 (0.257) | high |
| 192602 (19:26:02, 21.7) | — | — (Clip is taken by 214615, r 0.58 here but IMU peaks 2087 dps vs 271 dps in the clip) | | | | | | | | none |
| 215351 (21:53:51, 3.8) | — (false start before 215358) | — | | | | | | | | none |

The four sessions of about 300 s each (220452/221043/221654/222546, against
Clip_5…8) cannot be told apart by duration alone. But the 4×4 correlation
block is diagonal, both by row and by column (diagonal r 0.36–0.62, every
off-diagonal value ≤ 0.28). So the chronological pairing — IMU to clip, in
recording order — holds. The implied start/stop consistency is IMU end minus
clip end, equal to IMU duration minus lag minus clip duration. The values are:
  - Clip_2 +8.2 s.
  - Clip_3 −2.5 s.
  - Clip_4 +17.5 s.
  - Clip_5 +5.8 s.
  - Clip_6 +5.0 s.
  - Clip_7 +3.2 s (0.0 s at the −1.6 s lag).
  - Clip_8 +9.3 s.
  - Clip +3.3 s.

That is, the
IMU usually started a few seconds after the Rokoko take, and stopped a few
seconds after it too. This agrees with one operator working two apps. Chain
for the one fully tripled session: Clip_3 frame 0 equals IMU 215358 t = 0.9
s, equal to wall time 21:53:59.2. Orbbec 215350 frame 0 equals wall time
21:53:56.43. So `clip3_t = orbbec_host_time_s − 7 h − 21:53:59.2`.

**Caveats.** The IMU runs at a nominal 11.11 Hz, but effectively at **about
10.1 Hz** (samples divided by duration, in `work/alignment/run.log`). For
example, 1,932 samples over 191.3 s for 215358, and 3,384 samples over 335.3
s for 215749. So swings, which last about 0.2 s, are under-sampled: r stays
moderate (0.36–0.64), and the lag lobes are 1.7–4.2 s wide. Treat every lag
as accurate to only ±0.5 s, and refine it per swing if this is necessary.
Peak matching finds about twice as many IMU peaks as hand peaks, and it
agrees with the cross-correlation result only for Clip_2, Clip_4, and Clip_8.
The Orbbec-motion check neither confirms nor contradicts the shared-wall-clock
idea. All eight Orbbec sessions are now on disk (tar files 59.7 MB to 3.18
GB, extracted under `data/`). So IMU-to-Orbbec offsets exist for every one of
them. But these offsets rest on the shared-wall-clock idea, accurate to
about ±0.1 s, from frame timing. The two cameras of one session start
0.07–0.17 s apart. The 214619 `frames.jsonl` span (5.315 s) is shorter than
its `session.json` `duration_s` (5.885 s). This is because the second value runs from
`host_time_start_s` to `host_time_end_s`, not from the first frame to the
last frame. Rokoko carries no absolute clock. The team used a 0.1 s lag grid.

## 7. Pipeline & prior art

**Open-source survey (2026-08-27, every repo fetched and checked):
`docs/research/humanoid-table-tennis-survey.md`.** Main points: HITTER
(Berkeley) and SMASH (HKU, ego-camera only) ship no code. The only open,
trainable, table-tennis-specific humanoid RL codebase is Purdue PACE (MIT,
Booster T1, Isaac Sim 4.5 / Isaac Lab 2.1). `hitchopen/HOPE` (Apache-2.0) is
the platform that ran the table-tennis event at the 2026 World Humanoid Robot
Games (Agibot A3. It ships a checkpoint, plus a train → ONNX →
MuJoCo-check → C++ deploy pipeline. The G1 support is docs-only).
BeyondMimic (`HybridRobotics/whole_body_tracking`, MIT) is the whole-body
control base under HITTER and HOPE. Recommended build order: the PACE
environment and predictor, plus the HOPE pipeline pattern, plus BeyondMimic's
whole-body control and deploy code. Booster's own RL scaffolding repos carry
a NOASSERTION license.

Pipeline: `perception(ball) + movement(swings) → control(imitate + RL in sim) → him_adapter package → arena submit`.

Reference approaches (both use **model-based ball prediction, then an RL
whole-body controller**):
- **HITTER** (Berkeley, `humanoid-table-tennis.github.io`, arXiv 2508.21043): model-based ball and racket planning, plus an RL whole-body controller, with **human motion references** for natural movement. Real-world result: up to 106 shots in a row. This supports the team's plan to use mocap-retarget data as a reference.
- **PACE** (Purdue TraceLab, ICRA 2026, `github.com/purdue-tracelab/PACE-ICRA2026`, **MIT**): Booster T1, Isaac Sim plus Isaac Lab. An end-to-end **PPO** policy, plus a learned ball-landing predictor, plus dense physics rewards, gives a 96% or more hit rate. The repo has training and evaluation scripts, plus a table-tennis simulation environment, but no pretrained models and no datasets. This is a usable base to adapt for the G1.

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

Do not give a spec value (a signal, a unit, a joint count, a license, or
evaluator behavior) unless the team knows it is correct. By default, edit
only files under `.him`. Get the user's permission before you change a native
project file. Get the user's permission before any action that uploads
source code, spends credits, or starts compute resources. Do not decrease
any check only to make validation pass. Do not commit weights, data, or
credentials. Keep this document up to date as the project's memory.
