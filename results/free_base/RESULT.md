# Free base (unwelded pelvis) — first scored result

_2026-08-30. Scene: `tt_sim.scene.TableTennisEnv(fixed_base=False)` — the vendor pelvis free joint is kept, the
pelvis is replanted at (ROBOT_X, 0, ROBOT_Z) with identity orientation every reset, and the G1 must hold itself
up. 200 serves × seeds 11 / 2000 / 4000 = 600 trials per row, mean action, `tt_sim.evaluate` only._

## Scoreboard

| policy | scene | seed 11 legal / target / err | seed 2000 | seed 4000 | mean legal | mean target | mean err | contacts | falls | stood |
|---|---|---|---|---|---|---|---|---|---|---|
| param_residual_v2_wa0 | **welded (reference)** | 73.5 % / 32.0 % / 0.39 m | 76.5 % / 29.5 % / 0.39 m | 70.5 % / 24.0 % / 0.40 m | **73.5 %** | 28.5 % | 0.39 m | 157/168/155 | 0/0/0 | 100 % |
| ik_hitter | welded (reference) | 46.5 % / 3.5 % / 0.71 m | 47.0 % / 2.0 % / 0.78 m | 42.5 % / 1.5 % / 0.74 m | **45.3 %** | 2.3 % | 0.74 m | 139/152/137 | 0/0/0 | 100 % |
| param_residual_v2_wa0 | free, base-aware **ON** | 39.5 % / 2.0 % / 0.76 m | 41.0 % / 4.5 % / 0.72 m | 37.5 % / 3.5 % / 0.75 m | **39.3 %** | 3.3 % | 0.74 m | 151/152/139 | 83/63/76 | 63.0 % |
| param_residual_v2_wa0 | free, base-aware **OFF** | 53.5 % / 2.0 % / 0.77 m | 56.5 % / 2.0 % / 0.77 m | 52.0 % / 2.0 % / 0.78 m | **54.0 %** | 2.0 % | 0.77 m | 143/154/135 | 41/31/44 | 80.7 % |
| ik_hitter | free, base-aware **ON** | 28.0 % / 2.0 % / 0.71 m | 29.0 % / 4.5 % / 0.65 m | 28.0 % / 2.0 % / 0.75 m | **28.3 %** | 2.8 % | 0.70 m | 137/141/134 | 49/37/34 | 80.0 % |
| ik_hitter | free, base-aware **OFF** | 29.0 % / 1.0 % / 0.91 m | 36.5 % / 0.0 % / 0.91 m | 35.5 % / 1.5 % / 0.91 m | **33.7 %** | 0.8 % | 0.91 m | 105/119/103 | 14/23/19 | 90.7 % |

"stood" = trials that never latched `fell` (pelvis never went below 0.45 m), over all 600.

## The five-line read

1. **Unwelding the pelvis costs about 20 points of legal returns, not 5.** The shipped policy drops 73.5 % → 54.0 %
   at its best free-base setting, and stands in 81 % of serves. Whole-body play is not a small correction to the
   bolted policy.
2. **Base-aware IK made it worse, not better: 39.3 % against 54.0 %.** This is the opposite of what the free-pelvis
   probe predicted, and it is the main finding here.
3. **Base awareness works exactly as designed — that is *why* it hurts.** Measured over 60 free-base serves: it cuts
   the blade-to-target error at contact from 0.167 m to 0.066 m and lifts contacts from 33/60 to 40/60. It really
   does fix the 11 cm paddle-position error the probe found.
4. **The accuracy is bought with balance.** Chasing the drifting pelvis keeps the arm tracking through the
   disturbance instead of settling, and the reaction torque feeds straight back into an unbalanced base: pelvis-tilt
   p90 rises 58.9° → 73.4° and falls roughly double (wa0 19.3 % → 37.0 %, hitter 9.3 % → 20.0 %). More contacts, more
   falls, fewer legal returns.
5. **The conclusion is sequencing, not abandonment.** Base-aware IK is the right kinematics and the wrong thing to
   ship alone; it pays only once a balance controller absorbs the arm's reaction. Default it OFF on the free base
   until the whole-body lane exists, and re-measure it there.

## Caveat on the `falls` metric

`fell` latches at any point in the 3 s episode, including *after* a good return has already landed. That is why
base-unaware wa0 still scores 50.9 % legal on the trials it counts as falls. `falls` is therefore an upper bound on
"fell before it mattered", and the stand rates above read a little pessimistically. The aware-vs-unaware *difference*
is unaffected — both are measured the same way.

## Per-trial split (600 trials each)

| run | legal | falls | legal \| stood | contact \| stood |
|---|---|---|---|---|
| wa0, base-aware ON | 39.3 % | 37.0 % | 45.8 % | 78.8 % |
| wa0, base-aware OFF | 54.0 % | 19.3 % | 54.8 % | 72.7 % |
| ik_hitter, base-aware ON | 28.3 % | 20.0 % | 28.5 % | 68.3 % |
| ik_hitter, base-aware OFF | 33.7 % | 9.3 % | 32.4 % | 52.9 % |

Base awareness raises the contact rate *among trials that stayed upright* in both policies (+6.1 pts for wa0,
+15.4 pts for the hitter) and raises falls in both. The reach improves; the robot pays for it in balance.

## Exact commands

Run from the repo root with `PYTHONPATH=.` and `OMP_NUM_THREADS=1`, `.venv/bin/python`:

```bash
# (i) shipped policy, free base, base awareness ON  (+ the free-base hitter, row 5)
python -m tt_sim.rl.eval_seeds --npz assets/policies/param_residual_v2_wa0.npz --hitter \
    --seeds 11 2000 4000 --trials 200 --jobs 6 --free-base --out work/tt/rl/free_base/aware

# (ii) shipped policy, free base, base awareness OFF
TT_BASE_AWARE=0 python -m tt_sim.rl.eval_seeds --npz assets/policies/param_residual_v2_wa0.npz \
    --seeds 11 2000 4000 --trials 200 --jobs 6 --free-base --out work/tt/rl/free_base/unaware

# (iii) free-base hitter, base awareness OFF (the A/B partner for row 5)
TT_BASE_AWARE=0 python -m tt_sim.rl.eval_seeds --hitter \
    --seeds 11 2000 4000 --trials 200 --jobs 6 --free-base --out work/tt/rl/free_base/unaware_hitter

# welded reference, same seeds, same command minus --free-base
python -m tt_sim.rl.eval_seeds --npz assets/policies/param_residual_v2_wa0.npz --hitter \
    --seeds 11 2000 4000 --trials 200 --jobs 6 --out work/tt/rl/free_base/welded
```

The welded reference reproduces the recorded 73.5 % / 28.5 % / 0.39 m exactly, which is the 1200-trial check that
the free-base code changed nothing in the scored scene.

Cost: ~5 minutes of local CPU across all four runs (no credits). Wall time per run ≈ 55 s at `--jobs 6`.

## What is not answered here

- Whether a balance controller recovers the 20 lost points. That needs the whole-body lane (Isaac / MJX), not this
  scene.
- Whether the policy retrained *on* the free base closes the gap. Everything above runs a policy trained on the
  welded scene, unchanged — this measures transfer, not the free base's ceiling.
- Stepping or moving the feet. The robot is asked to stand still and swing; the legs hold `STAND_POSE` throughout.
