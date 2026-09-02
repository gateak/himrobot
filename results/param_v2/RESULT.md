# Hitter v2 (swing library + 20-knob parameter space) — result log (2026-08-29)

Baselines, deterministic (`tt_sim.evaluate`, mean action, 200 serves): hitter 46.5 / 46.0 / 46.5 % legal on
seeds 11 / 7 / 3; v1 parameter residual (`work/tt/rl/param/s3/best.pt` = `assets/policies/param_residual_s3.npz`)
**70.5 / 70.5 / 71.0 % legal, 10 % target hits, 0.52 m landing error**.

## Parity (v2 code, everything at nominal) — green

| check | expected | measured |
|---|---|---|
| hitter, 40 serves, seed 7 | 42.5 % legal, 26 contacts | 42.5 % / 26 (`parity_hitter_seed7/`) |
| v1 `.npz`, 200 serves, seed 11 | 70.5 % / 10 % / 0.520 m / 154 contacts | 70.5 % / 10.0 % / 0.5203 m / 154 (`parity_v1npz_seed11/`) |
| `pytest tt_sim/tests` | all green | 88 → 103+ passed (v2, trainer, evaluator and perturbation tests) |
| golden fixture `tt_sim/tests/data/param_s3_actions.npz` | committed | was git-ignored by the `data/` rule (test silently skipped); un-ignored |

## Attempt 1 (killed after 800 episodes): exploration noise on the new knobs destroys the rollouts

Warm-started v2 runs with the trainer's default `--log-std-init -1.0` (std 0.37 in pre-squash z) on the ten new
coordinates showed noisy train-time legal rates of 24.5–32.5 % at 800 episodes, against 65–68 % at the end of the v1
run they were resumed from. Measured cause (`noise_sensitivity.json`, 96 random serves per cell, sampled actions,
v1 head from s3/best.pt, every other coordinate silent):

| noise on | std 0.10 | std 0.20 | std 0.37 |
|---|---|---|---|
| nothing (baseline) | 70.8 % | | |
| the ten v1 knobs at their trained noise | 66.7 % | | |
| **all ten new knobs** | **52.1 %** | **34.4 %** | **21.9 %** |
| gate_block | 67.7 | 64.6 | 54.2 |
| gate_lift | 62.5 | 51.0 | 45.8 |
| gate_punch | 66.3 | 61.9 | 58.3 |
| gate_crossbody | 68.8 | 63.5 | 56.2 |
| x_hit_coarse | 71.9 | 68.8 | 63.5 |
| reach_expand_y | 70.8 | 70.8 | 70.8 |
| reach_expand_z | 70.8 | 70.8 | 70.8 |
| normal_weight_scale | 70.8 | 69.8 | 70.8 |
| swing_window_scale | 74.0 | 66.7 | 54.6 |
| swing_dir_pitch | 69.8 | 61.5 | 53.1 |

Read: the gates (lift worst), the swing-window scale and the swing-direction tilt are noise-sensitive; the two
reach-expansion knobs and the face-weight knob change *nothing* on this serve distribution (the clamped serves are
physically out of reach whatever the clamp says, and the mean policy never binds the z clamp). Ten mildly
perturbed knobs compound: std 0.1 on each costs 19 points. Fix: `--log-std-init-new` (exploration std for the new
coordinates only; the v1 coordinates keep their learned noise) and deterministic checkpoint selection
(`--det-eval-episodes`, see below).

## Attempt 2: gentle exploration + deterministic selection (nine runs, 20 000 episodes each)

Recipe: `python -m tt_sim.rl.param_train --n-envs 32 --minutes 75 --eval-every 200 --vel-penalty 1e-5
--legal-bonus 10 --param-space v2 --resume-from work/tt/rl/param/s3/best.pt --det-eval-episodes 80
--det-eval-every 2000 --det-eval-seed 1001 --log-std-init-new {-2.5,-3.0,-2.0}`; `wa0` adds
`--aim-bonus 8 --aim-sigma 0.30 --target-hit-bonus 4`. Started 17:40 EDT on the Mac under heavy competing load
(~1.4–5 episodes/s instead of 11/s).

| run | new-knob std | seed | reward | episodes | best det (sel. set) legal / target / err @ep | final det legal / target | seed 11 / 7 / 3 (200 serves) |
|---|---|---|---|---|---|---|---|
| w0 | 0.082 (−2.5) | 0 | legal | 20 000 | 80.00 / 13.75 / 0.49 @16k | 78.75 / 10.00 | see scoreboard |
| w1 | 0.082 (−2.5) | 1 | legal | 20 000 | 80.00 / 13.75 / 0.49 @14k | 78.75 / 13.75 | |
| w2 | 0.050 (−3.0) | 0 | legal | 20 000 | 80.00 / 7.50 / 0.59 @14k | 78.75 / 6.25 | |
| w3 | 0.135 (−2.0) | 0 | legal | 20 000 | 80.00 / 18.75 / 0.48 @4k | 80.00 / 10.00 | |
| wa0 | 0.082 (−2.5) | 0 | legal + aim 8 / σ 0.30 / hit +4 | 20 000 | 81.25 / 10.00 / 0.51 @6k | 76.25 / 17.50 | |
| w4 | 0.082 (−2.5) | 2 | legal | 20 000 | 81.25 / 8.75 / 0.54 @14k | 73.75 / 7.50 | |
| w5 | 0.082 (−2.5) | 3 | legal | 20 000 | 81.25 / 23.75 / 0.46 @4k | 76.25 / 12.50 | |
| w6 | 0.050 (−3.0) | 1 | legal | 20 000 | 80.00 / 6.25 / 0.56 @18k | 78.75 / 10.00 | |
| wa1 | 0.082 (−2.5) | 1 | legal + aim | 20 000 | 80.00 / 21.25 / 0.46 @12k | 77.50 / 30.00 | |

Selection set = seed 1001, 80 serves, deterministic; start point 78.75 / 15.00 / 0.50 for every run. Every run ended
on the 20 000-episode default cap (~63–65 min wall each, 9 runs sharing the Mac). Read: on the selection set the
legal rate never left the noise band around its start (76–81 % vs 78.75, ±4.5 pts at n = 80); the placement-reward
runs (wa0, wa1) and w5 moved target hits up (21–30 % vs 15 %) at no measurable legal cost. The window-mean knob
usage (`u_*` columns) shows every run still driving the same v1 knobs (flight_time_scale pinned fast, pitch_offset,
aim_y_offset, t_hit_offset); the ten new knobs stay within ±0.2 of nominal — 20 000 episodes at std 0.05–0.08 is
not enough for them to be discovered.

(The 78.75 % at episode 0 is the v1 policy on the 80-serve selection set — seed 1001 is an easier set than
seed 11; it is the reference every later deterministic number in `train_log.csv` is compared with.)

## Harness changes made alongside (all default-off / behaviour-preserving)

- `tt_sim.rl.param_train`: `deterministic_eval` (reproduces `tt_sim.evaluate` trial for trial: s3 on seed 11 →
  70.5 / 10.0 / 0.5202, identical), `--det-eval-episodes/--det-eval-every/--det-eval-seed`, `--log-std-init-new`,
  `u_<name>` window-mean columns in `train_log.csv`, row 0 = the starting point's deterministic score.
- `tt_sim.rl.eval_seeds`: one command → seeds 11/7/3 scoreboard (`scoreboard.md/json`), parallel subprocesses.
- `tt_sim.evaluate`: `--no-trajectory`; `--obs-noise/--joint-noise/--obs-delay-steps` (policy-side sensor noise
  and latency, scoring untouched, `policy_obs` recorded in `evaluator.json`).
- `scripts/box_param_sweep.sh` + generalized `scripts/box_push_results.py` for the robustness sweep on a box.

## Serve-set overlap (found while reading the batch-1 scoreboard)

Every policy scored within ±0.5 % across seeds 11 / 7 / 3 — too tight for independent 200-serve sets. Cause:
`tt_sim.evaluate` draws serve k from `default_rng(seed + k)`, so seeds 11/7/3 share 192–196 of 200 serves. The
v1 result's "confirmed on unseen sets" was one set measured three times. From now on the scoreboard is seeds
11 / 2000 / 4000 (disjoint; `eval_seeds` refuses seeds closer than the trial count) and the trainer's selection set
(seed 1001) is disjoint from all of them.

## Scoreboards (`tt_sim.rl.eval_seeds`, mean action, 200 serves per seed)

### Seeds 11 / 7 / 3 (one serve set measured three times — see the overlap note above)

| policy | seed 11 legal / target / err | seed 7 legal / target / err | seed 3 legal / target / err | mean legal | mean target | mean err | contacts |
|---|---|---|---|---|---|---|---|
| w0/best | 73.5 % / 9.0 % / 0.50 m | 73.0 % / 8.0 % / 0.50 m | 73.5 % / 8.0 % / 0.50 m | **73.3 %** | 8.3 % | 0.50 m | 158/157/158 |
| w0/final | 74.5 % / 7.0 % / 0.58 m | 74.5 % / 7.0 % / 0.58 m | 75.0 % / 7.5 % / 0.58 m | **74.7 %** | 7.2 % | 0.58 m | 160/159/160 |
| w1/best | 72.0 % / 12.5 % / 0.52 m | 71.5 % / 11.5 % / 0.52 m | 72.0 % / 11.5 % / 0.52 m | **71.8 %** | 11.8 % | 0.52 m | 154/153/154 |
| w1/final | 72.0 % / 9.5 % / 0.50 m | 72.0 % / 9.5 % / 0.50 m | 72.5 % / 9.5 % / 0.50 m | **72.2 %** | 9.5 % | 0.50 m | 155/154/155 |
| w2/best | 72.0 % / 3.5 % / 0.59 m | 72.0 % / 3.5 % / 0.60 m | 72.5 % / 4.0 % / 0.60 m | **72.2 %** | 3.7 % | 0.60 m | 158/157/158 |
| w2/final | 73.0 % / 4.5 % / 0.57 m | 72.5 % / 4.5 % / 0.57 m | 73.0 % / 4.5 % / 0.57 m | **72.8 %** | 4.5 % | 0.57 m | 156/155/156 |
| w3/best | 72.5 % / 11.0 % / 0.52 m | 72.0 % / 10.5 % / 0.53 m | 72.0 % / 10.0 % / 0.53 m | **72.2 %** | 10.5 % | 0.53 m | 160/159/160 |
| w3/final | 74.5 % / 9.0 % / 0.50 m | 73.5 % / 8.5 % / 0.51 m | 74.0 % / 9.0 % / 0.50 m | **74.0 %** | 8.8 % | 0.50 m | 158/157/158 |
| wa0/best | 74.0 % / 12.0 % / 0.51 m | 73.0 % / 11.5 % / 0.52 m | 73.5 % / 11.5 % / 0.52 m | **73.5 %** | 11.7 % | 0.52 m | 157/156/157 |
| wa0/final | 73.5 % / 32.0 % / 0.39 m | 73.0 % / 30.0 % / 0.40 m | 73.5 % / 30.0 % / 0.40 m | **73.3 %** | 30.7 % | 0.39 m | 157/156/157 |

200 serves per seed; mean action (`tt_sim.evaluate`); contacts listed per seed.

| w4/best | 73.5 % / 6.0 % / 0.55 m | 73.0 % / 5.5 % / 0.56 m | 74.0 % / 5.5 % / 0.56 m | **73.5 %** | 5.7 % | 0.55 m | 153/152/153 |
| w4/final | 71.0 % / 8.0 % / 0.54 m | 70.5 % / 7.5 % / 0.54 m | 70.5 % / 7.0 % / 0.54 m | **70.7 %** | 7.5 % | 0.54 m | 153/152/153 |
| w5/best | 74.0 % / 9.5 % / 0.52 m | 73.5 % / 9.0 % / 0.52 m | 74.0 % / 9.0 % / 0.52 m | **73.8 %** | 9.2 % | 0.52 m | 157/156/157 |
| w5/final | 71.0 % / 8.5 % / 0.52 m | 70.5 % / 8.0 % / 0.52 m | 71.0 % / 8.5 % / 0.52 m | **70.8 %** | 8.3 % | 0.52 m | 156/155/156 |
| w6/best | 73.5 % / 6.5 % / 0.59 m | 73.0 % / 6.0 % / 0.59 m | 73.5 % / 6.0 % / 0.59 m | **73.3 %** | 6.2 % | 0.59 m | 156/155/156 |
| w6/final | 72.0 % / 9.5 % / 0.52 m | 71.5 % / 9.0 % / 0.52 m | 72.0 % / 9.0 % / 0.52 m | **71.8 %** | 9.2 % | 0.52 m | 152/151/152 |
| wa1/best | 65.5 % / 12.0 % / 0.48 m | 65.0 % / 11.5 % / 0.48 m | 65.5 % / 11.5 % / 0.48 m | **65.3 %** | 11.7 % | 0.48 m | 156/155/156 |
| wa1/final | 72.0 % / 19.5 % / 0.46 m | 70.5 % / 19.0 % / 0.46 m | 71.0 % / 19.5 % / 0.46 m | **71.2 %** | 19.3 % | 0.46 m | 157/156/157 |

200 serves per seed; mean action (`tt_sim.evaluate`); contacts listed per seed.

### Disjoint seeds 2000 / 4000 (genuinely unseen serves)

| policy | seed 2000 legal / target / err | seed 4000 legal / target / err | mean legal | mean target | mean err | contacts |
|---|---|---|---|---|---|---|
| wa0/final | 76.5 % / 29.5 % / 0.39 m | 70.5 % / 23.5 % / 0.40 m | **73.5 %** | 26.5 % | 0.39 m | 168/155 |
| w0/final | 76.0 % / 8.5 % / 0.58 m | 70.0 % / 7.5 % / 0.56 m | **73.0 %** | 8.0 % | 0.57 m | 168/156 |
| w3/final | 77.5 % / 12.0 % / 0.49 m | 69.5 % / 9.0 % / 0.51 m | **73.5 %** | 10.5 % | 0.50 m | 170/155 |
| policies/param_residual_s3 | 78.0 % / 12.0 % / 0.51 m | 69.5 % / 12.0 % / 0.49 m | **73.8 %** | 12.0 % | 0.50 m | 165/151 |
| ik_hitter | 47.0 % / 2.0 % / 0.78 m | 42.5 % / 1.5 % / 0.74 m | **44.8 %** | 1.8 % | 0.76 m | 152/137 |

200 serves per seed; mean action (`tt_sim.evaluate`); contacts listed per seed.


## Conclusion (2026-08-29)

1. **Legal returns did not improve on unseen serves.** v1 (`param_residual_s3`) 73.8 %, the best v2 runs 73.0–73.5 %,
   all inside the ±3-point noise of two 200-serve sets. The +4 points `w0/final` shows on seed 11 is fitting to that
   one set. The kinematic-ceiling sweep in the design (20–24 % of serves unreachable with the pelvis welded) puts the
   ceiling near 76–80 %; we are 2–5 points under it, and nothing in the 20-knob space changes reachability.
2. **Placement is real and robust.** `wa0/final` (`--aim-bonus 8 --aim-sigma 0.30 --target-hit-bonus 4`):
   target hits 12 % → 26.5 %, landing error 0.50 → 0.39 m on the disjoint sets, at an unchanged legal rate; 30.7 % /
   0.39 m on seed 11. Exported as `assets/policies/param_residual_v2_wa0.npz` (torch-free runtime).
3. **20 000 episodes at std 0.05–0.08 is not enough for the new knobs to be discovered** (window-mean u within ±0.2
   of nominal for all ten). Their value, if any, needs 5–10× the episodes — a box, not the Mac.
4. **An 80-serve selection set is too small**: `best.pt` beat `final.pt` in only 4 of 9 runs on the real scoreboard
   (wa1: 65.3 % vs 71.2 %). Use ≥ 200 selection serves on a box (`--det-eval-episodes 200 --det-eval-every 4000`).
5. Seeds 11/7/3 overlap (192–196 of 200 serves); the scoreboard is now 11 / 2000 / 4000.

## Shipped weights

`assets/policies/param_residual_v2_wa0.npz` = `work/tt/rl/param_v2/wa0/final.pt` through `scripts/export_param_policy.py`.
Numpy runtime vs torch: decoded parameters agree to 3e-8; joint targets agree to 1e-4 until the follow-through, where
the warm-started IK amplifies float32 rounding on some serves (up to 0.16 rad after step 46, i.e. after contact).
Scores are identical on seeds 11 / 2000 / 4000 except one target hit in 600 trials (seed 4000: 24.0 % vs 23.5 %).
`NumpyParamResidualPolicy` now defaults to this file; the v1 file stays shipped as `S3_WEIGHTS` (golden-fixture guard).

## Placement phase 2 (2026-08-30): saturated

Resumed `wa0/final` with `--det-select target --det-eval-episodes 200 --det-eval-every 4000` (200-serve selection
set, seed 1001), same aim reward, two seeds, 8 000 episodes each (`work/tt/rl/param_v2_phase2/`). Seeds 11 / 2000 / 4000:
p0 best = final 73.5 % / 27.8 % / 0.38 m; p1/best 72.5 % / 29.5 % / 0.37 m; p1/final 73.0 % / 21.5 % / 0.43 m.
Against the shipped wa0 (73.5 % / 28.5 % / 0.39 m) that is noise: placement is saturated at ~28–30 % target hits with
this hitter. wa0 stays the shipped policy. (The 200-serve selection set behaved: best.pt == final.pt for p0.)
