# Balance v1.2 — honest free-base scene (table intangible to the body) — 2026-09-02

Follows `results/balance_v1/RESULT.md`. Trigger: on video the free-standing robot toppled forward onto the
table and was held up by it; `fell` (pelvis below `FALL_PELVIS_Z`) never fired. New scene param
`robot_table_collision=false` (commit 83dbc9e; ball and paddle still collide with table and net) removes the
crutch. Everything below is on that scene.

## Setup

Same VecTTBalance as v1 (v3 expert on waist+arm, 12 leg joints) with: fall penalty 30 (was 10), `best.pt` by
legal × stand on 40 serves (seed 1001), stay-in-box pelvis term `--pos-penalty 2` (box ±0.15 × ±0.30 m), leg
authority 0.15 or 0.40 rad, 16 envs, 300 min cap (~13.5 k episodes each), `TT_ROBOT_Y=-0.10`.

| run | authority | start | best det (40 serves) legal / stand |
|---|---|---|---|
| a_s015_warm | 0.15 | warm from v1 s2 | 67.5 % / 55.0 % @ep 7336 |
| b_s040_fresh | 0.40 | fresh (zero head) | 45.0 % / 42.5 % @ep 2408 |
| c_s040_fresh | 0.40 | fresh | 42.5 % / 55.0 % @ep 1767 |
| d_s015_fresh | 0.15 | fresh | 45.0 % / 45.0 % @ep 9089 |
| no balance, same 40 serves | — | — | 32.5 % / 27.5 % |

## Result (200 serves × seeds 11 / 2000 / 4000 = 600 trials, mean action, `eval_seeds --free-base --params '{"robot_table_collision": false}'`)

| policy | seed 11 | seed 2000 | seed 4000 | legal | target | err | contacts | falls |
|---|---|---|---|---|---|---|---|---|
| v3 expert alone (no balance) | 57.0 % | 65.0 % | 62.5 % | **61.5 %** | 4.8 % | 0.69 m | 470/600 (78 %) | 454/600 (**75.7 %**) |
| a_s015_warm | 55.0 % | 59.5 % | 55.5 % | 56.7 % | 6.5 % | 0.62 m | 486 (81 %) | 442 (73.7 %) |
| b_s040_fresh | 44.5 % | 54.0 % | 47.0 % | 48.5 % | 1.8 % | 0.73 m | 409 (68 %) | 388 (64.7 %) |
| c_s040_fresh | 36.5 % | 40.5 % | 40.5 % | 39.2 % | 4.0 % | 0.68 m | 418 (70 %) | 362 (60.3 %) |
| d_s015_fresh | 34.5 % | 43.5 % | 40.0 % | 39.3 % | 4.3 % | 0.64 m | 506 (84 %) | 454 (75.7 %) |

For reference, the same expert with the table solid: 45.0 % legal, 21.8 % "falls" (table-supported).

## Read

1. No leg-only policy reduces falls materially on unseen serves: best 75.7 % → 60.3 % (c), at −22 points of
   legal returns. The 55 % stand rates seen during training were overfits of the 40-serve selection set.
2. The plain v3 expert returns 61.5 % free-standing once its body no longer collides with the table (45.0 %
   with the solid table) — the table was also blocking swings — and falls in 3 of 4 serves.
3. Conclusion: the swing's reaction momentum is not absorbable by ±0.15–0.40 rad leg corrections around
   STAND_POSE. Balance for this swing is a whole-body problem (waist/arm compensation, stepping) — the Isaac
   Lab lane with the mocap prior. Before that: a stance sweep (wider / lower STAND_POSE) of the plain expert,
   free, to change the tip-over physics with zero training.

Artifacts (local, git-ignored): `work/tt/rl/balance_v1_2/<run>/{best.pt,final.pt,balance_v1_2_<run>.npz,train_log.csv}`
and `score_<run>/scoreboard.{md,json}`.
