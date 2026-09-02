# free_v4 — the hitter relearned with a locked waist on the honest free base — 2026-09-02

Follows `results/waist_lock/RESULT.md`: with waist roll/pitch locked the free-standing G1 stops falling
(14.8 %) but loses reach (45.3 % legal). Here the 20-knob parameter residual (v2 space) is warm-started from
v3 `s0/final` and retrained ON the honest free base with the waist locked, so the arm learns the reach the
torso used to provide.

Command (per seed, Mac CPU, ~74 min to the 20 000-episode budget):

    TT_ROBOT_Y=-0.10 TT_WAIST_LIMIT=0,0 python -m himrobot.rl.param_train --n-envs 32 --seed <s> --minutes 150 \
      --param-space v2 --resume-from work/tt/rl/v3_stance/s0/final.pt --vel-penalty 1e-5 --legal-bonus 10 \
      --aim-bonus 8 --aim-sigma 0.30 --target-hit-bonus 4 --depth-bonus 3 --fall-penalty 10 \
      --params '{"free_base": true, "robot_table_collision": false}' \
      --det-eval-episodes 80 --det-eval-every 200 --det-eval-seed 1001

Selection set (80 serves, seed 1001): s0 best 60.0 % @ep 9814, s1 best 62.5 % @ep 18615 (start 46.2 %).

## Result (200 serves × seeds 11 / 2000 / 4000, mean action, honest free base, waist locked)

| policy | seed 11 | seed 2000 | seed 4000 | legal | target | err | contacts | falls |
|---|---|---|---|---|---|---|---|---|
| v3 s0/final, waist locked (no retrain) | 41.5 % | 51.5 % | 43.0 % | 45.3 % | 3.5 % | 0.68 m | 57 % | 14.8 % |
| free_v4 s0/best | 55.5 % | 59.0 % | 55.0 % | 56.5 % | 19.5 % | 0.43 m | 69 % | 17.8 % (36/40/31) |
| **free_v4 s1/best** | 56.5 % | 64.0 % | 58.5 % | **59.7 %** | **18.0 %** | **0.45 m** | 70 % | **16.8 %** (38/33/30) |
| welded v3 (reference) | | | | 77.0 % | 49.5 % | 0.28 m | 88 % | — |

Scoring: `TT_ROBOT_Y=-0.10 TT_WAIST_LIMIT=0,0 TT_PARAM_NPZ=<npz> python -m himrobot.eval.eval_seeds --policy
x=himrobot.policies.param_residual_np:NumpyParamResidualPolicy --free-base --params '{"robot_table_collision": false}'`.

Shipped: `assets/policies/param_residual_free_v4_s1best.npz` (needs `TT_WAIST_LIMIT=0,0` and the honest free base).

## Read

- +14.4 points legal, target hits 3.5 → 18 %, error 0.68 → 0.45 m, falls unchanged within noise. The arm can
  recover most of the reach a locked torso costs; what it cannot recover is the ~17 points to the welded ceiling,
  which is the torso's contribution — the whole-body lane's job (waist free + counterbalance + stepping).
- Falls (~17 %) did not go down: the fall penalty priced them but the 20 knobs have no balance authority. The
  remaining falls are the whole-body lane's other job.
