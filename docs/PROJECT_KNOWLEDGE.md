# Project knowledge — himrobot (HIM humanoid table tennis)

This document gives the facts of the project. The project keeps this document
under version control. Change this document each time the team makes a new
decision or finds a new fact. Use this document for the facts of the project.
Do not use notes from outside the project.
_Last updated: 2026-09-01 (docs/results restructure: §6a/§6b moved to `docs/archive/`, paths updated
to the `himrobot/` layout, results ledger added at `results/`; no fact or number changed)._

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
| **Movement** | human swings → robot reference motions | 🟢 **G1, full body (29 DoF), all 8 clips**, done with GMR (`tools/mocap/gmr_bvh_to_g1_headless.py --ik_config tools/mocap/ik_configs/bvh_lafan1_rokoko_to_g1.json`). **Kinematic only** — no one has examined the dynamics of this motion, and no one has made sure that a real robot can do it. The earlier CSVs (`~/him-tools/out/all`, upstream offsets) had a mean wrist error of 50–69 cm and 23–31% joint-limit saturation. The corrected config (`work/g1/`) gives a mean wrist error of about 10–12 cm (`work/retarget/adjust2_rotfix_h1p75.md`, §4c). The team keeps the older right-arm-only IK (`tools/mocap/g1_retarget.py`) as a reference |
| **Perception** | ball / scene 3D (`ball_state`) from cameras | 🟡 The team tracks the orange ball in 2D and looks up its depth on all eight sessions (`tools/mocap/ball_detect.py`, `tools/mocap/ball_3d.py`, with results in `work/perception/summary.md`). The candidate rate is 23–47% per camera. Random-sample precision is 96/144 = 67%. Depth identifies the ball in 45–71% of hits. For session 215350, at the documented settings (gated, ROI margin 40 px), 26–32% of frames carry a candidate. About 1/2 to 2/3 of the sampled candidates are the real ball. The rest are sandals at the far edge. The team found ArUco DICT_4X4 IDs 0–7 at the half-table corners (`tools/mocap/aruco_scan.py`). **No intrinsics are on disk, so there is no metric XYZ yet.** See §6a. |
| **Control** | `joints + ball → desired joints` (the deployable hitter) | 🟡 First policy exists. `himrobot/` is the fixed-base G1 MuJoCo dev scene with a local proxy evaluator (`python -m himrobot.eval.evaluate`). The model-based IK hitter (`himrobot/policies/ik_hitter.py`) scores 60% legal returns on seeds 0 to 3 (10 trials each, paddle contact 8 of 10, landing error 0.7 m, reaction 0.5 s). The hold policy scores 0%. The `evaluate` entrypoint in `.him/him.yaml` runs it. Over 200 serves (seed 11) the hitter scores 46.5% legal, 3.5% target hits, 534 ms reaction. **RL v3 (2026-08-28):** residual PPO over the hitter (`himrobot/rl/ppo.py --residual --residual-scale 0.2 --vel-penalty 1e-5 --legal-bonus 10`, 85 min on a HIM box) scores 53.5% legal / 6.5% target (seed 32) and 45.5% / 12.5% (seed 31) on the same 200 serves, with 24–35 ms reaction. Checkpoints: HIM dataset `himrobot-rl-checkpoints@0.1.29798103`, local `work/tt/rl/box/v3/`, write-up `work/tt/rl/v3/RESULT.md` (local; versioned copy `results/v3/RESULT.md`). The v2 reward (velocity penalty 0.0005) made PPO swing slower (legal 47% → 3%); the intercept-shaped cold PPO stayed at 0–1%. **RL v4 (2026-08-28, box #3, 6 seeds × 425 min, ~11.5 M steps each):** no improvement — fresh seeds 36–49 % (s40 final.pt 51.5 %), seeds resumed from the 53.5 % checkpoint *fell* to 32–48.5 % while their noisy train-time rate rose 22 → 31 %. Cause: PPO optimizes returns under its own exploration noise (log_std −1.6, never annealed) and `best.pt` was selected by a 20-episode noisy window. Fix (commit 381faea): `--det-eval-episodes N` selects `best.pt` by a mean-action evaluation on N fixed serves (logged as `det_legal_rate`), `--log-std-init`/`--log-std-final` anneal the noise. Write-up `work/tt/rl/v4/RESULT.md` (local; versioned copy `results/v4/RESULT.md`). Also landed (f66c109, c23c6a1): serve-range params + `serve_scale`, per-episode `physics_jitter`, `--obs-noise/--joint-noise/--obs-delay-steps`, a `train` entrypoint in `.him/him.yaml`, and the per-serve parameter residual (`himrobot/rl/param_residual.py`, `himrobot.rl.param_train`, `ParamResidualPolicy`; K=10 bounded offsets on the IK hitter, step-zero parity tested). **Parameter residual result (2026-08-29):** `python -m himrobot.rl.param_train --n-envs 32 --minutes 45 --vel-penalty 1e-5 --legal-bonus 10` on the Mac (no credits) → **67.5 % / 68.0 %** legal (seeds 0/1) on 200 serves, confirmed 67.5 % / 68.0 % on two unseen 200-serve sets where the hitter scores 46.0 % / 46.5 %; best.pt ≈ final.pt (no drift); target hits still ~1 %. Write-up `work/tt/rl/param/RESULT.md` (local; versioned copy `results/param/RESULT.md`). This is the policy to package next (`ParamResidualPolicy` into `him_adapter`; needs ball state in the runtime contract). **Best policy (2026-08-29): `work/tt/rl/param/s3/best.pt` = `assets/policies/param_residual_s3.npz`, 70.5 / 70.5 / 71.0 % legal, 10 % target, 0.52 m on seeds 11 / 7 / 3.** **Hitter v2 (2026-08-29, commit pending):** swing library (five templates blended by four rectified gates), coarse hitting-plane depth, reach expansion, face-weight, dwell and swing-direction knobs — a 20-knob `v2` parameter space on top of the unchanged 10-knob `v1` (`--param-space v2`, warm start from a v1 checkpoint via `--resume-from`). Nominal behaviour is bit-identical (golden fixture `tests/data/param_s3_actions.npz`; hitter 42.5 %/26 contacts on 40 serves seed 7; v1 policy 70.5 % on seed 11). Measured: exploration noise on the ten new knobs is destructive (std 0.37 → 22 % during training; use `--log-std-init-new -2.5/-3.0`); the reach-expansion and face-weight knobs change nothing on the nominal serve set; the crossbody template ships without its +y box (the solver self-collides there) and without yaw biases. Write-up `work/tt/rl/param_v2/RESULT.md` (local; versioned copy `results/param_v2/RESULT.md`), design `work/tt/design/hitter_v2.md` (with an implementation-deviations section). **First v2 training (nine warm-started runs, 20 000 episodes each, Mac): legal returns did not improve on unseen serves (v1 73.8 % vs best v2 73.0–73.5 % on seeds 2000/4000, ±3 pts), but the placement-reward run `wa0/final` (`--aim-bonus 8 --aim-sigma 0.30 --target-hit-bonus 4`) doubles target hits (12 → 26.5 %) and cuts landing error (0.50 → 0.39 m) at the same legal rate; on seed 11: 73.5 % / 30.7 % / 0.39 m. Shipped as `assets/policies/param_residual_v2_wa0.npz` (the `evaluate` entrypoint's default weights). A placement phase 2 (target-first selection, 200-serve selection set, 2 seeds × 8 000 episodes, 2026-08-30) scored 72.5–73.5 % / 27.8–29.5 % / 0.37–0.38 m — the same as wa0 within noise: placement is saturated near 28–30 % with this hitter. The ten new knobs were not discovered in 20 000 episodes (window-mean u within ±0.2 of nominal); the reach ceiling (20–24 % of serves unreachable with the pelvis welded) caps legal returns near 76–80 % whatever the knobs do. |

**Free-pelvis probe (2026-08-30, 20 serves, seed 2000+, `scene.build_model(fixed_base=False)` with the pelvis placed at
(−1.77, 0, 0.793)).** With NO balance controller — position actuators holding STAND_POSE for the legs and the shipped
v2 hitter driving waist + arm exactly as on the bolted pelvis — the G1 stayed upright in 15/20 serves (5 falls) and made
paddle contact with the ball crossing the net in 14/20 (bolted reference ≈ 84 % contact, 76 % legal on these serves).
Pelvis drift at contact: ≈2 cm; the hitter's paddle-position belief (bolted-pelvis kinematics) was off by 11 cm median,
40 cm when falling. Read: a "stand still and swing" whole-body policy is within reach of the current hitter plus a
base-pose-aware IK and a light balance controller; stepping/moving is the real whole-body problem (Isaac lane).

**Free base, scored (2026-08-30, 200 serves × seeds 11/2000/4000 = 600 trials per row, `himrobot.eval.evaluate` mean action).**
The probe's optimism did not survive scoring, and the base-aware IK it recommended **made things worse**. The option
shipped: `TableTennisEnv(fixed_base=False)`, `himrobot.eval.evaluate --free-base`, `eval_seeds --free-base`, an additive
`metrics["falls"]`, four `base.*` observation keys emitted only on the free base, and base-aware IK in the hitter
(`IKHitterPolicy(base_aware=True)` / `himrobot.eval.evaluate --base-aware`; off by default) — all of it gated so the welded scene is
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
an upper bound. Write-up `work/tt/rl/free_base/RESULT.md` (local; versioned copy `results/free_base/RESULT.md`); ~5 min local CPU, no credits.

**Hitter v3 stance (2026-08-30) — where the robot stands is worth more than every hitter knob put together.**
Nine uniform knob directions through the real evaluator buy nothing (best: a 5 % slower swing, +1.25 points of legal
returns for −10.25 points of target hits); the hitter's own parameter space is exhausted. Sliding the **welded pelvis
0.10 m toward −y** — one constant, `scene.ROBOT_Y`, read once at import from `TT_ROBOT_Y`, plus a stance-relative
reach frame in `ik_hitter.py` (`STANCE_Y` / `_stance_y` slide `Y_REACH`, `REACH_MAX_Y` and the `lift`/`punch` y boxes;
x, z and every face/direction bias are untouched) — moves the **shipped, unretrained** policy this far
(`himrobot.eval.evaluate` mean action, 200 serves × seeds 11 / 2000 / 4000):

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
`work/tt/design/hitter_v3_reach.md`, write-up `work/tt/rl/v3_stance/RESULT.md` (local; versioned copy `results/v3_stance/RESULT.md`).

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
not been used yet (`tools/box/gpu_box_setup.sh` is ready, never run).

**Balance v1 (2026-09-01/02) — first free-standing gain.** Free base at the v3 stance, no balance controller: the
shipped v3 policy scores **45.0 % legal / 5.0 % target / 0.61 m, falls 21.8 %** (600 trials, seeds 11/2000/4000;
worse than v2 wa0's 54.0 % because v3's faster placement swings kick harder). A leg-only PPO residual
(`himrobot/rl/balance_env.py` VecTTBalance: 12 leg joints, ±0.15 rad around STAND_POSE, obs 99 + 12-dim base
block, the v3 expert unchanged on waist+arm, zero-init head so step 0 == baseline; `himrobot.rl.balance_train`;
torch-free runtime `himrobot/policies/balance_np.py`) trained 3 seeds × 4.6 h on the Mac scores **52.3 / 59.2 /
62.7 % legal** with contacts 62 % → 80–85 % — but s2 (62.7 %) falls **25.5 %**, more than the baseline, while s0
(52.3 %) falls 11.8 %: selection by legal rate with a fall penalty equal to the legal bonus buys returns with
falls. `results/balance_v1/RESULT.md`, design `docs/design/balance_v1.md`.
**Correction (2026-09-02, Akbar spotted it on video): every free-base "stand" number above is table-supported.**
`fell` only fires below `FALL_PELVIS_Z`, and the robot topples forward onto the table, which holds it up. With
the body made intangible to the table and net (`params.robot_table_collision=false`, new, default true; ball and
paddle still collide) the same 40 serves (seed 11) give: no balance 52.5 % legal / **32/40 falls**; v1 s2 55 % /
**32/40 falls** (13/40 with the table solid). The honest no-balance stand rate on the 40-serve selection set is
**27.5 %** (77.5 % with the crutch). Balance v1 changed nothing about falling. v1.1 (fall penalty 30, legal × stand
selection, solid table) was stopped after 1 h for that reason. **v1.2 (running, 4 × ~4.5 h):** phase-through scene,
fall penalty 30, legal × stand selection, stay-in-box pelvis term (`--pos-penalty 2`, box ±0.15 × ±0.30 m, no pull
to a point — "stay in a general place, free to shuffle sideways"), leg authority 0.15 (bracing) vs 0.40 (weight
shift), fresh vs warm-started from s2. **Result (600 serves each, honest scene): no leg-only policy fixes
falling** — v3 alone 61.5 % legal / **75.7 % falls**; best balance run c 39.2 % / 60.3 % falls, a 56.7 % / 73.7 %.
The 55 % stand rates during training were overfits of the 40-serve selection set. Read: the swing's reaction
momentum is not absorbable by ±0.15–0.40 rad leg corrections around STAND_POSE; balance here is whole-body
(Isaac Lab lane, mocap prior). `results/balance_v1_2/RESULT.md`.
**Root cause found (2026-09-02, `results/waist_lock/RESULT.md`): the falls are the hitter's torso, not the legs.**
Holding STAND_POSE the G1 stands 3 s at 0.1° tilt; swing speed 35–100 % changes nothing; falls land ~1 s after
the return; in every falling serve the IK drove waist roll/pitch to their 0.52 rad limits mid swing. New knobs,
all default = today: `TT_WAIST_LIMIT="roll,pitch"` (solver bounds), `TT_LEG_POSE` (stance), `TT_SWING_SPEED`
(global cap multiplier); and on the honest scene the paddle now meets only the ball (no propping). **Waist locked,
honest free base, 600 serves: 45.3 % legal / 3.5 % target / 0.68 m, falls 14.8 %** (waist free: 61.5 % / 75.7 %
falls). Even ±0.1 rad of waist still topples it (waist *motion*, not lean). Stance sweep: crouch+wide lifts
returns (57 → 72 %) but not standing; with the waist locked the crouch is worse (44 % falls). **Retrained with the waist locked on the honest free base (`results/free_v4/RESULT.md`, 2 × 74 min Mac): free_v4
s1/best = 59.7 % legal / 18.0 % target / 0.45 m / 16.8 % falls** on 600 serves (from 45.3 / 3.5 / 0.68 / 14.8).
Shipped as `assets/policies/param_residual_free_v4_s1best.npz` (run with `TT_WAIST_LIMIT=0,0`, free base,
`robot_table_collision=false`). The ~17-point gap to the welded 77 % is the torso's reach; the ~17 % falls have no
knob to fix them — both are the whole-body lane (design in progress: `docs/design/whole_body_v1.md`).

**Hitter v3 retrain + ship (2026-08-30 evening).** Two warm-started runs at the stance with `--depth-bonus 3`
settled the pre-registered bar: legal returns are stance-limited (best 79.8 % vs 77.8 % unretrained -- the welded-scene
reach ceiling ~81 % is effectively reached), and placement broke open: **s0/final = 77.0 % legal / 49.5 % target hits /
0.28 m** on seeds 11/2000/4000 (s1/final 77.5/46.0/0.31). **Shipped:** the `evaluate`/`evaluate-1000` entrypoints now run
`himrobot.eval.evaluate_stance` (sets `TT_ROBOT_Y=-0.10` + the v3 weights before the scene import) with
`assets/policies/param_residual_v3_s0final.npz`; `param_residual_v3_s1best.npz` is the legal-max alternative
(79.8/30.7/0.39). The welded default scene, `himrobot.eval.evaluate`, and every parity fixture are unchanged.
Full log `work/tt/rl/v3_stance/RESULT.md` (local; versioned copy `results/v3_stance/RESULT.md`).

**Serve sets (found 2026-08-29).** `himrobot.eval.evaluate` draws serve k from `default_rng(seed + k)`, so seeds 11 / 7 / 3
share 192–196 of 200 serves: every "confirmed on three independent sets" statement before this date was one set
measured three times. Independent sets need seeds ≥ `--trials` apart; the scoreboard is now **seeds 11 / 2000 / 4000**
(`himrobot.eval.eval_seeds` refuses overlapping seeds) and the trainer's selection set is seed 1001 (serves 1001–1080,
disjoint from all three).

**Harness (2026-08-29).** `himrobot.rl.param_train --det-eval-episodes N --det-eval-every M --det-eval-seed S` selects `best.pt` by the
mean action on N fixed serves scored exactly like `himrobot.eval.evaluate` (verified: s3 on seed 11 → 70.5 / 10.0 / 0.5202, identical);
keep S off 11/7/3. `train_log.csv` carries `det_*` columns, the window mean of every knob (`u_<name>`) and the blend fraction per
template (`blend_<template>`). `himrobot.eval.eval_seeds --ckpt … --npz … --hitter --out DIR` is the scoreboard (seeds 11/7/3 × 200,
parallel, `scoreboard.md/json`). `himrobot.eval.evaluate` gained `--no-trajectory` and policy-side `--obs-noise/--joint-noise/--obs-delay-steps`
(recorded under `policy_obs` in `evaluator.json`; scoring untouched); `param_train` accepts the same three for training under noise.
Box scripts: `tools/box/box_param_sweep.sh` (parallel sweep, 30-min pushes, self-stop) and `tools/box/box_push_results.py --root --machine`.
`tools/make_golden_param_actions.py --check` proves the current code still reproduces the golden fixture. Previous boxes: L4,
128 cores, 51 GB cgroup cap, 93 credits/h, 8 h max (744 credits); a `cpu` profile exists (price not yet seen).

**Pending / loose ends**
- ~~Migrate `.him/contracts/runtime.json` from the K1's 22 joints to the G1's 29.~~ Done 2026-08-26 (Menagerie/SDK order in `extensions.robot.joint_order`).
- Train a G1 tracking policy on the retargeted CSVs (mjlab / BeyondMimic recipe. This needs an NVIDIA machine — the CSV format that `tools/mocap/gmr_bvh_to_g1_headless.py` gives is the Unitree/LAFAN1 CSV that those recipes take as input).
- Make sure that the contract's signal names and rate agree with HIM's server-side evaluator (not yet confirmed).
- ~~Time-align the three capture streams (separate clocks) before the control track.~~ Done 2026-08-26 (`tools/mocap/align_streams.py`, §6b). The IMU-to-Rokoko lag per session comes from cross-correlation (|gyro| against hand angular speed). The 7 play sessions show r 0.36–0.64, with lags −6.0…+0.9 s. The 17 s test take, 214615 against Clip, shows r 0.987 alone, with lag +6.8 s. The IMU-to-Orbbec lag comes from the wall clock: both hosts agree to the second. HIM's America/New_York UTC guess is 3 h off the Orbbec host's UTC−7. The team ran this again on 2026-08-27 with **all 8 Orbbec sessions on disk**. First frame minus the paired IMU episode start is −1.9…+5.4 s for the 7 play sessions (+20.9 s for the 5 s test take 214619). Frame spans agree with the IMU durations within −9.3…+5.5 s (start/stop offsets, not drift). The `_60fps` sessions in fact recorded at **about 19 fps, effective** (`work/alignment/orbbec-imu-offsets.md`). Open item: Rokoko has no absolute clock, so it is anchored only through the IMU.
- Compute at scale still has a gate on it (see §4). Local MuJoCo is the path with no gate.

## 3. Environment & tooling

- The `him` CLI is `him_sdk 0.1.3`, in a **Python 3.12** venv at `~/.local/share/him/venv` (binary `~/.local/bin/him`). The default endpoint is `http://127.0.0.1:5173`. This endpoint does not operate. **Always** run `export HIM_API_URL=https://arena.himrobotics.com` first. The team is logged in (account `him_pat_…`, credentials at `~/.config/him/credentials.json` — do not read or commit this file).
- The main development venv is `.venv/` (`/usr/local/bin/python3.12` → 3.12.8). It has torch 2.13.0, mujoco 3.12.0, imageio 2.37.4, imageio-ffmpeg 0.6.0, numpy 2.5.2, and pytest 9.1.1 (installed 2026-08-26).
- For FBX reading, the team uses the **Blender binary** (`/Applications/Blender.app/Contents/MacOS/Blender`), version 5.2.1 LTS since 2026-08-27 (version 3.4.1 before that). The BVH export from both versions is the same to within ≤7e-5. The team made sure of this. The Blender binary has a bundled Python 3.10.8 with numpy 2.2.6, and `io_scene_fbx` is on. Command: `Blender -b --python tools/mocap/fbx_extract.py -- <clip.fbx> <out.npz>`. No `bpy` wheel and no `.venv-fbx` are necessary (the team discarded the old side-venv path). The importer leaves the scene range at 1..250, so `fbx_extract.py` gets the frame range from the armature action instead (Clip_4 = 9569 frames at 30 fps).
- Other tool venvs are outside the repo:
  - `~/him-tools/mujoco-venv` — python 3.12.8: mujoco 3.12.0, imageio 2.37.4, imageio-ffmpeg 0.6.0, numpy 2.5.2, pillow 12.3.0, **scipy 1.18.1 and matplotlib 3.11.1**, for `tools/mocap/align_streams.py` and `tools/mocap/retarget_report.py`.
  - `~/him-tools/cv-venv` — python 3.12.8: numpy 2.5.2, opencv-contrib-python 5.0.0.93, imageio 2.37.4, pillow 12.3.0, used by `tools/mocap/ball_detect.py`, `tools/mocap/ball_3d.py`, `tools/mocap/aruco_scan.py`, `tools/mocap/orbbec_video.py`.
  - `~/him-tools/gmr-venv` — GMR, torch, mujoco, mink.
  - `~/him-tools/GMR` — the upstream copy, read-only.
  - pyenv 3.11.9, at `~/.pyenv/versions/3.11.9`.
- Vendored assets are git-ignored. Download them again with `tools/fetch_assets.sh`. The assets are `third_party/booster_assets` (K1, 49 MB) and `third_party/mujoco_menagerie` (a sparse checkout of `unitree_g1` only, 56 MB). A fallback copy of the G1 is at `~/him-tools/mujoco_menagerie/unitree_g1`.
- `data/` (git-ignored) holds HIM's proprietary, licensed capture. Never commit this directory. On 2026-08-27, per a user directive, the team put the data together as real files, with no symlinks:
  - `him-table-tennis-paddle-imu-20260815/` (plus a `paddle-imu-samples.csv` copy that `tools/mocap/retarget_imu.py` reads)
  - `rokoko-motion-capture/` (8 FBX files)
  - `orbbec_20260815_214619/`, `orbbec_20260815_215350/` (extracted sessions)
  - `release/` (the 10 verified tar files from `him data download table-tennis`)
  - `manifests/` (prepared HIM dataset manifests, plus packed artifacts)
  - `preview/` (a Blender `.blend` file)
  - `g1/` (a copy of the corrected retarget CSVs from `work/g1`, which `legacy/bc/` reads)
  - `mocap_clip4.npz`

  Derived BVH files are in `work/bvh` and `work/bvh_lafan1`. Tool venvs stay in `~/him-tools/`.

## 4. HIM platform mechanics

- **Data:** `him data download` only works for the **table-tennis** release. There is no catalog, and there is no robot obs→action dataset to get. `capture`, `dataset`, and `train` use LOCAL fixtures that you write (`him train rehearse` is a stub that always gives the same result — it does not do real training).
- **Simulator:** the simulator runs only on the server. There is no local simulator. To run a policy against a simulated robot, use `him arena submit <challenge_version_id> --package <approved_pkg> --entrypoint <ep>` (the Community challenge, on isaac-sim, scored on legal-return, target-hit, landing-error, and latency), or use `him job launch simulation` (mujoco-simulation-contract, `mujoco_mjx`, gpu-standard) with `him job watch`. The output is **metrics and events, not video**.
- **Gating:** the server has a per-package worker gate (`package_worker_disabled`). This gate stops the real package now. The team's first `him run` failed, with status `reconcile-timeout`. A control run, with a built-in fixture, succeeded — this shows that the connection between the team's system and HIM's system operates correctly. HIM must open the gate before the team's real code can operate. **Update 2026-08-28:** the gate is open. Run #3 (`job_bba9697ea3091b2f53ba`) failed inside the simulation because the gitignored G1 model was not in the source snapshot; the model is now vendored as plain files in `assets/unitree_g1/` (a nested git clone is uploaded as a gitlink) and `himrobot/__init__.py` probes for a MuJoCo GL backend that imports. Run #5 (`job_b314d2cca72c57d8306e`, 138 files, 15.9 MB) ran the simulation and reached `publishing_outputs`, then failed with `provider_reconciliation_timeout` (no charge); two retries failed the same way within minutes, and `him job watch` on it returns HTTP 400 "Compute attestation identity or digest verification failed" — a HIM-side issue to report with the job id. `him run` reuses the job when the lock digest is unchanged; pass `--idempotency-key` to force a new job. Package `my-policy@0.1.0`, id `pkg_110ce5d185fa9ba3d703`. Failed job `job_6be550b37b3337434d9a`.
- **Approval gates:** get the user's permission first for `him login`, `him run`, `him push`, `him publish`, `job launch`, `arena submit`, and `dataset push` — and for anything that uploads source code, spends credits, or starts compute resources.

## 4b. HIM datasets (private, registered from this machine 2026-08-26)

- `him-table-tennis-paddle-imu@1.0.0` → **`dataset_f410d4b7a08c0ab25d88`** (status ready, usable as `him run --dataset <id>`). The team added this from HIM's own manifest, with `license.spdx_or_name` added.
- Orbbec 214619 / 215350 and Rokoko mocap: the team prepared manifests, and `him dataset validate` shows no error, under `~/him-data/manifests/`. The team has **not** pushed these manifests — the license for locally captured data, and the Rokoko session timestamp, still need confirmation.
- `him capture inspect`, `dataset build`, and `train rehearse` take only `fixture_only` JSON in version 0.1.3. Someone must write raw-capture manifests by hand.

## 4c. Retargeting pipeline (`.him/contracts/training-pipeline.json`, `rokoko-to-g1-motion-tracking-v1`)

The pipeline has these stages:
- Blender headless FBX→BVH (`tools/mocap/fbx_to_bvh.py`, about 1 minute per clip).
- `tools/mocap/validate_motion.py`.
- Rename the Rokoko "Newton" bones to LAFAN1 names (`tools/mocap/rokoko_to_lafan1_bvh.py`: Spine1→Spine, Spine2→Spine1, Spine4→Spine2, Thigh→UpLeg, Shin→Leg). GMR looks bones up by name. Extra bones are permitted. cm and Y-up are expected.
- GMR `--robot unitree_g1 --format lafan1` (`tools/mocap/gmr_bvh_to_g1_headless.py`, about 350 fps on an M4 Pro CPU). `--jobs N` runs clips in parallel.
- Unitree CSV (root xyz, quat **xyzw**, 29 joints in SDK order).
- `tools/mocap/play_g1_csv.py` (live `launch_passive` or `--video`).

The team made sure of a full `him pipeline run` on 2026-08-26 (before the `--ik_config` change). After the contract change, the team ran the retarget stage command again on one clip and got the same CSV, byte for byte. It did not re-run the full pipeline. The full pipeline covers 8 clips, 59,685 frames, and 33 minutes of motion.
Tool venvs live outside the repo (`~/him-tools/{gmr-venv,mujoco-venv,GMR,mujoco_menagerie}`). Pass them with `--set gmr_python=… --set mujoco_python=…`.

**Retarget fidelity (2026-08-26, `tools/mocap/retarget_report.py`, reports in `work/retarget/`, CSVs in `work/g1/`).** `work/g1/` is the **corrected** set (adjustment 2, below), and `data/g1` and `legacy/bc/` use it. `~/him-tools/out/all` is the **broken baseline** (upstream rotation offsets). The team keeps it only for comparison.
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
- **Adjustment 2 (kept)** — `tools/mocap/ik_configs/bvh_lafan1_rokoko_to_g1.json`, made by `tools/mocap/ik_configs/make_rokoko_g1_config.py`. The rotation offsets are re-derived as `Rx(90)ᵀ·Rz(−90)·R_body(q_T)`, from the GMR model in a G1 T-pose (shoulder_roll ∓π/2, elbows +π/2). The team copied everything else from upstream. Height stays at 1.75 m. Mean wrist error **10.0–11.7 cm** (Clip 10.0, Clip_2 10.6, medians 9.4–11.2, max 16–25 cm, Clip_5 50 cm). Shoulder-frame error 8.9–11.6 cm. Saturation ranges from 60 joint-frames (Clip) to 4,521 joint-frames (Clip_7) — 0.3–1.7%, mostly at `ankle_roll`. Minimum foot z runs from +0.008 m (Clip) to −0.083 m (Clip_5). Root z is 0.53–0.89 m. The robot elbow range is 0–123°, against a human range of 0–130°. `work/g1/*.csv` and `work/retarget/{Clip_g1.mp4, Clip_4_g1_20s.mp4}` come from this run (Clip_4 is the best long clip: 10.1 cm mean, 988 saturated joint-frames. The segment is frames 5521–6121, the 20 s with the most wrist travel). Even at the 10 cm mark, the human shoulder sits 0.49 m above the hips, against 0.29 m on the G1. A per-bone scale about the root cannot correct this difference.
- Commands: `~/him-tools/gmr-venv/bin/python tools/mocap/gmr_bvh_to_g1_headless.py --bvh_dir work/bvh_lafan1 --out_dir work/g1 --ik_config tools/mocap/ik_configs/bvh_lafan1_rokoko_to_g1.json --jobs 5`. Then `~/him-tools/mujoco-venv/bin/python tools/mocap/retarget_report.py --bvh_dir ~/him-tools/bvh_lafan1 --csv_dir work/g1 --out work/retarget/report` (add `--measure_height clip.bvh` for the anthropometrics). The pipeline contract passes `--ik_config tools/mocap/ik_configs/bvh_lafan1_rokoko_to_g1.json --jobs 4` (updated 2026-08-27).
- The contract's per-stage `cache.key_fields` is a label only. The HIM local runner keys each stage's cache on its resolved command and parameters, plus its dependencies' `output_digest` (a hash of each dependency's *declared* outputs). So `tools/mocap/fbx_to_bvh.py`, `tools/mocap/rokoko_to_lafan1_bvh.py`, and `tools/mocap/validate_motion.py` write a per-clip source and output sha256 into their summary JSON files, for anyone who audits what actually changed.

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
- **Orbbec RGB-D** (8 `orbbec_*.tar`): **2 synced cams** (cam0/cam1, a stereo pair), 640×480, color JPG plus depth PNG (mm, `align_depth_to_color:true`). The 60fps sessions are the rally sessions. The table has **ArUco markers at the corners**, so metric table-frame calibration is possible (solvePnP plus triangulation) with no factory intrinsics. **The ball is orange**, on a dark-blue table, so HSV color detection works (see `tools/mocap/ball_detect.py`). Watch for a second held or spare orange ball, plus clutter around the net post.
- **Rokoko mocap** (`rokoko-fbx.tar`): 8 binary FBX clips, with standard bone names (`Hips`, `Spine*`, `RightShoulder`→`RightArm`→`RightForeArm`→`RightHand`).
- There is **no robot data and no joint data** in the capture — `robot_version_id: null`. (Cleanup removed about 13 GB of browser-duplicate tar files, plus `.crdownload` partial files. The team kept one copy of each.)

### 6a. Perception status (2026-08-26 Track C, updated 2026-08-27 Track P)

**Summary (full write-up moved to `docs/archive/perception-2026-08.md`, verbatim, tables included).**
All eight Orbbec sessions are on disk. The 2D ball detector's candidate rate is 16–47 % of frames
per camera (gated: motion + ArUco-hull ROI + static-orange reject + near-big-blob reject).
Random-sample precision across all six Track-P sessions is 96/144 = 66.7 % real ball (the other
33 % is mostly sandals/feet at the table edges, some hands/wrists, one net post). Aligned depth
identifies the ball (disc reads >40 mm nearer than its annulus) in 45–71 % of candidates. ArUco
`DICT_4X4_50` markers at each half-table's four corners give table geometry, but **no camera
intrinsics exist on disk anywhere** (not in the manifests, tar files, or image headers); a
self-calibration from the markers plus aligned depth gives focal lengths (fx≈fy≈444 px / 482–501
px per camera) with a **±15 % scale caveat** — one tape measurement of a marker edge would settle
it. So there is still no metric XYZ / `ball_state`. See the archive for per-session tables, the
camera-serial/enumeration gotcha, and the three ways to get real intrinsics.

### 6b. Stream alignment (2026-08-26, Track D, Orbbec part re-run 2026-08-27 with all 8 sessions)

**Summary (full write-up moved to `docs/archive/alignment-2026-08.md`, verbatim, tables included).**
Three capture streams, three clocks, no shared absolute reference. **The clock-offset rule**: the
IMU host and the Orbbec host share one wall clock (the Orbbec host's zone was UTC−7; HIM's stated
America/New_York guess for the IMU is off by 3 h either way — the files can't show which side is
wrong, but it doesn't matter for alignment) — so align on wall clock, `timestamp_local` against
`host_time_s` minus 7 h; the remaining Orbbec-minus-IMU offsets (−22.3 s … +5.5 s across the 8
sessions) are recorder start/stop slop, not clock drift. Rokoko has no absolute clock at all, so
it is anchored only through cross-correlating IMU |gyro| against the retargeted hand's angular
speed (the **lag table**): 2 sessions high confidence, 4 medium, 1 low (twin-peaked, ambiguous
between −4.8 s and −1.6 s), r 0.36–0.64 throughout because the IMU's effective 10.1 Hz
under-samples 0.2 s swings — treat every lag as accurate to only ±0.5 s. See the archive for the
per-session table and the full method.

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

**The restructure (2026-09-01).** `tt_sim/` moved to `himrobot/` (layered: `himrobot/sim` — scene,
media, obs_perturb; `himrobot/policies`; `himrobot/rl` — ppo, param_train, balance_train, env, …;
`himrobot/eval` — evaluate, evaluate_stance, eval_seeds; `himrobot/serve` — the HIM stdio server,
formerly `him_adapter/`). **`tt_sim.*` and `him_adapter.*` remain as thin compatibility shims**
that re-export the `himrobot.*` code — this is deliberate: it is what HIM's already-approved
package and every old dotted command (`python -m tt_sim.evaluate_stance`,
`tt_sim.policies.param_residual_np:NumpyParamResidualPolicy`) still load, so neither the shims nor
the old names get deleted. New code and new docs use `himrobot.*`.

```
himrobot/      The project code. sim/ (scene, media, obs_perturb), policies/ (ik_hitter,
               param_residual_np, …), rl/ (ppo, param_train, balance_train, env, …),
               eval/ (evaluate, evaluate_stance, eval_seeds), serve/ (HIM stdio server).
tt_sim/        Compatibility shim: re-exports himrobot.* under the old dotted paths HIM loads.
him_adapter/   Compatibility shim: re-exports himrobot.serve under the old path.
.him/          HIM overlay: manifest, runtime contract, fixtures, smoke test, lock.
legacy/bc/     PyTorch behavior-cloning scaffold (formerly training/).
tools/
  mocap/       MuJoCo (un-gated): view_k1, k1_model, g1_model, fbx_extract, retarget_imu, g1_retarget;
               full-body pipeline: fbx_to_bvh, validate_motion, rokoko_to_lafan1_bvh, gmr_bvh_to_g1_headless (--ik_config),
               play_g1_csv, retarget_report.py (wrist-error / saturation report), ik_configs/;
               perception: ball_detect.py (orange ball, 2D), ball_3d.py (aligned depth), aruco_scan.py (table markers),
               orbbec_video.py (frames->mp4);
               alignment: align_streams.py (IMU <-> Rokoko <-> Orbbec clocks).
  box/         box_param_sweep.sh, box_push_results.py, gpu_box_setup.sh (rented-machine scripts).
  fetch_assets.sh, make_golden_param_actions.py.
tests/         Pytest suite (formerly tt_sim/tests): golden fixtures, parity, collision-guard tests.
results/       Versioned run ledger: one RESULT.md per scored run, plus SCOREBOARD.md.
assets/        Vendored robot models (assets/unitree_g1/) and shipped policy weights (assets/policies/).
docs/          THIS knowledge doc, docs/design/ (design docs per feature), docs/archive/ (superseded
               detail sections, moved out verbatim).
pyproject.toml `pip install -e .` installs himrobot as an editable package.
```
```bash
pip install -e .
pytest

# himrobot (primary):
./.venv/bin/python -m himrobot.eval.evaluate
./.venv/bin/python -m himrobot.eval.evaluate_stance
./.venv/bin/python -m himrobot.rl.eval_seeds --ckpt … --npz … --hitter --out DIR

# tt_sim (still works, same code, via the compatibility shim):
./.venv/bin/python -m tt_sim.evaluate
./.venv/bin/python -m tt_sim.evaluate_stance

./tools/fetch_assets.sh
./.venv/bin/python tools/mocap/g1_retarget.py     # real human swing on the G1 -> tools/mocap/g1_retarget.gif
/Applications/Blender.app/Contents/MacOS/Blender -b --python tools/mocap/fbx_extract.py -- <clip.fbx> data/mocap_clip4.npz
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
