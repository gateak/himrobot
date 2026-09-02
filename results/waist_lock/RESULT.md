# Waist lock: the free-base falls were the hitter's torso, not the legs — 2026-09-02

Chain of diagnosis on the honest free-base scene (`robot_table_collision=false`, and since this commit the paddle
meets only the ball, so nothing props the body up):

1. Holding STAND_POSE with no policy: the G1 stands 3 s at 0.1° tilt — the physics is stable.
2. Swing speed sweep (`TT_SWING_SPEED` 1.0 → 0.35): falls stay at 78–86 % (straight legs) and 96–100 % (crouch)
   — not a momentum problem.
3. Per-serve trace: falls land at t ≈ 1.8–2.9 s, ~1 s after contact and after the return crossed the net; a
   serve with no contact at all fell too. In every falling serve the IK solver drove waist roll and pitch to
   their 0.52 rad limits mid swing; the two survivors returned the waist to 0.
4. `TT_WAIST_LIMIT="roll,pitch"` (new; tightens the solver's waist bounds, default = model limits, bit-exact):

| waist roll/pitch bound | legal | target | contacts | falls (200 serves, seed 11) |
|---|---|---|---|---|
| model limits (shipped) | 57.0 % | 3.5 % | 77 % | 85.5 % |
| ±0.3 | 57.0 % | 12.5 % | 74 % | 86.0 % |
| ±0.2 | 51.0 % | 7.5 % | 71.5 % | 92.0 % |
| ±0.1 | 51.0 % | 7.5 % | 65 % | 75.5 % |
| **0 (locked)** | **41.5 %** | 2.5 % | 55.5 % | **13.5 %** |
| 0 + crouch/wide stance | 42.0 % | 3.5 % | 61 % | 44.0 % |

Even ±0.1 rad topples the robot: it is the waist *motion* during the swing that throws the body, not a static
lean. Only a rigid torso survives; the crouch is worse again.

## Scored (200 × seeds 11 / 2000 / 4000, mean action, honest scene, straight legs)

| policy | seed 11 | seed 2000 | seed 4000 | legal | target | err | contacts | falls |
|---|---|---|---|---|---|---|---|---|
| v3, waist free (`results/balance_v1_2`, paddle still solid then) | 57.0 % | 65.0 % | 62.5 % | 61.5 % | 4.8 % | 0.69 m | 78 % | 75.7 % |
| **v3, waist locked** | 41.5 % | 51.5 % | 43.0 % | **45.3 %** | 3.5 % | 0.68 m | 57 % | **14.8 %** (27/36/26) |
| welded v3 (reference) | | | | 77.0 % | 49.5 % | 0.28 m | 88 % | — |

Command: `TT_ROBOT_Y=-0.10 TT_WAIST_LIMIT=0,0 python -m himrobot.eval.eval_seeds --policy
v3_waistlocked=himrobot.policies.param_residual_np:NumpyParamResidualPolicy --free-base --params
'{"robot_table_collision": false}' --jobs 6`.

## Read

- First free-standing number that means what it says: 45.3 % legal, 85 % standing, no training.
- The cost is reach: the hitter was tuned on a bolted pelvis where the torso was free; with the waist locked
  contacts fall 78 → 57 %. Next (free): retrain the 20-knob parameter residual with the waist locked on the
  honest free-base scene (`param_train`, warm start from v3), so the arm learns the reach the torso used to give.
- Two nights of leg-only balance RL (`results/balance_v1*`) attacked the wrong joint. Lesson: trace one failure
  frame by frame before training anything.
