# Leg-stance sweep on the honest free-base scene — 2026-09-02

Question: does a crouched / wider standing pose change the tip-over physics enough to reduce falls without
any training? Knob: `TT_LEG_POSE` (twelve radians for JOINT_ORDER slots 0–11, read once at import by
`himrobot.sim.scene`; absent = the vendor "stand" keyframe, straight legs, bit-exact parity).

Setup: plain v3 expert (`param_residual_v3_s0final.npz`), `TT_ROBOT_Y=-0.10`, `--free-base`,
`robot_table_collision=false`, 200 serves, seed 11, mean action, `himrobot.eval.evaluate`.

| stance (`TT_LEG_POSE`, left leg hp,hr,hy,k,ap,ar; right mirrored on roll) | legal | contacts | falls |
|---|---|---|---|
| straight legs (default) | 57.0 % | 154 (77 %) | 154 (77.0 %) |
| crouch1: hp −0.1, k 0.3, ap −0.2 | 66.5 % | 162 (81 %) | 173 (86.5 %) |
| crouch2: hp −0.2, k 0.5, ap −0.3 | 70.0 % | 159 (80 %) | 165 (82.5 %) |
| wide1: hr ±0.1 | 61.0 % | 163 (82 %) | 142 (71.0 %) |
| crouch1 + wide1 | 69.5 % | 171 (86 %) | 170 (85.0 %) |
| crouch2 + wide2 (hr ±0.2) | **72.0 %** | 166 (83 %) | 153 (76.5 %) |

## Read

- No stance reduces falls materially (best 71 %, wide feet; crouching alone raises them). The swing tips the
  robot regardless of how the legs are set: whole-body compensation or a step is required.
- Crouching lifts legal returns +13–15 points (57 → 70–72 %) on the free base, within 5 points of the welded
  77 %: a lower pelvis puts the shipped hitter's reach in a better place for this serve distribution. Returns
  happen before falls, so the free-standing robot already scores near the welded one on the return metric.
- Next instrument: per-trial fall timing in the evaluator (before vs after the return crosses the net) to
  decide whether the whole-body lane's first job is "recover after the swing" or "do not tip during it".

Artifacts: `work/tt/stance_sweep/<stance>/` (local).
