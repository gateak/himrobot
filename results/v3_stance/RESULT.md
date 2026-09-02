# v3 stance — the shipped policy re-scored at a pelvis slid 0.10 m toward −y

**No training has happened yet.** Everything below is the *shipped, unretrained*
`assets/policies/param_residual_v2_wa0.npz` (and the plain hitter) flown at the new stance, which
is the mid-point gate of `work/tt/design/hitter_v3_reach.md` §3.6: the in-package stance frame must
reproduce the scratch harness (`work/tt/analysis/reach/v3_stance_task.py`) exactly, or nothing
downstream is trustworthy.

Landed in this change (all defaulting to today's behaviour):

* `tt_sim/scene.py` — `ROBOT_Y = float(os.environ.get("TT_ROBOT_Y", "0.0"))`, used by
  `BASE_NOMINAL_POS` and by `_build_xml`'s pelvis line as `{ROBOT_Y:g}`. `f"{0.0:g}"` renders `0`,
  so at the default the XML string is byte-identical to the pre-stance tree.
* `tt_sim/policies/ik_hitter.py` — `STANCE_Y` / `_stance_y`, applied to `Y_REACH`, `REACH_MAX_Y`
  and the `lift` / `punch` y boxes. At the nominal stance `_stance_y` returns its argument object,
  so every template dict holds the identical float objects it held before.
* `tt_sim/rl/env.py`, `tt_sim/rl/param_train.py` — `depth_bonus` / `--depth-bonus` (design §2.3),
  default 0.0 = today's reward bit-for-bit.
* `tt_sim/policies/ik_hitter.py` — `collision_guard` (design §2.4), a `SERVE_PARAM_NOMINAL_V3` knob,
  default 0.0. Deliberately not in any `PARAM_SPACES` entry: `K_PARAMS_V2` stays 20.

## Re-score at `TT_ROBOT_Y=-0.10` (200 serves × seeds 11 / 2000 / 4000, `tt_sim.evaluate` mean action)

| policy | seed 11 | seed 2000 | seed 4000 | legal / 600 | legal % | target % | err | contacts |
|---|---|---|---|---|---|---|---|---|
| `param_residual_v2_wa0` @ y = 0.00 (scoreboard) | 147 | 153 | 141 | 441 | 73.5 % | 28.5 % | 0.39 m | 157/168/155 |
| **`param_residual_v2_wa0` @ y = −0.10** | **161** | **154** | **152** | **467** | **77.8 %** | 30.5 % | 0.39 m | 176/174/177 |
| plain `ik_hitter` @ y = 0.00 (scoreboard) | 93 | 94 | 85 | 272 | 45.3 % | 2.3 % | 0.74 m | 139/152/137 |
| plain `ik_hitter` @ y = −0.10 | 88 | 104 | 91 | 283 | 47.2 % | 2.2 % | 0.75 m | 151/156/146 |

**Gate met exactly**: the design measured 161 / 154 / 152 for the policy from the scratch harness, and the
in-package stance frame reproduces it to the trial, contacts included (the design's paired table records
157 → 176, 168 → 174, 155 → 177).

Commands (from the repo root of the worktree; `PYTHONPATH` is the repo root):

```
export OMP_NUM_THREADS=1 PYTHONPATH=. TT_ROBOT_Y=-0.10

./.venv/bin/python -m tt_sim.rl.eval_seeds \
    --npz assets/policies/param_residual_v2_wa0.npz \
    --seeds 11 2000 4000 --trials 200 --jobs 4 \
    --out work/tt/rl/v3_stance/rescore_npz

./.venv/bin/python -m tt_sim.rl.eval_seeds --hitter \
    --seeds 11 2000 4000 --trials 200 --jobs 4 \
    --out work/tt/rl/v3_stance/rescore_hitter
```

Raw output: `rescore_npz/scoreboard.{md,json}`, `rescore_hitter/scoreboard.{md,json}`, plus one
`evaluation/evaluator.json` per (policy, seed).

The y = 0.00 rows were re-measured on this same tree (same command, `TT_ROBOT_Y` unset,
`--out work/tt/rl/v3_stance/rescore_default_stance`) and came back at exactly the pre-change
scoreboard — 73.5 / 76.5 / 70.5 % for the policy and 46.5 / 47.0 / 42.5 % for the hitter — which is
what the byte-identical XML and the golden fixture already promised.

## The collision guard, geometrically

The operational band the balls actually occupy (y ∈ [−0.85, +0.05] step 0.05, z ∈ [0.90, 1.35] step 0.05,
the five face normals, `hitter_helpers.ik_pose_at`'s recipe: 950 poses per plane), counting poses where the
right arm (or the paddle) contacts something that is not the idle left arm. Pinned in
`tt_sim/tests/test_hitter_v3.py::test_the_operational_band_is_collision_free_under_the_guard`, and it
reproduces the design's numbers to the pose:

| plane | guard off | guard on |
|---|---|---|
| −1.20 | 3 | **0** |
| −1.31 | 86 | **0** |
| −1.41 | 227 | **0** |
| −1.50 | 306 | **5** |

The guard is off by default and buys 0 legal returns on today's serve set at either stance; it is insurance
for a widened serve set (`serve_scale` > 1) or a `+y` crossbody box.

## Filter change worth recording

`_benign_contact` (now `tt_sim/tests/hitter_helpers.py::benign_contact`, shared by the v2 and v3
collision tests) was widened to excuse the whole idle-left-arm-against-the-hips family, not just
`left_wrist_* | left_hip_pitch_link`: the left arm is held at `STAND_POSE`, is off the IK chain and
cannot reach the paddle, and on the operational band that family alone accounts for 109-276 poses
per plane. Under the old filter the v2 collision test's deep-plane counts measured exactly its
pinned 27 / 57 / 90 / 277; under the widened one they are **22 / 45 / 66 / 184**, and the pins were
tightened to match rather than left as a slacker ceiling.

## Next

Design §4: warm-start a retrain at the new stance from `wa0` (Mac, ~65 min per seed, no credits),
with `--depth-bonus 3`, selection on seed 1001 × 200 serves, scored on seeds 11 / 2000 / 4000.
Pre-registered acceptance bar in §4.4; the fallback is exactly the 467/600 row above, which needs no
training at all.


## Retrain scoreboard (2026-08-30, TT_ROBOT_Y=-0.10, seeds 11/2000/4000, 200 serves each, mean action)

| policy | seed 11 legal / target / err | seed 2000 legal / target / err | seed 4000 legal / target / err | mean legal | mean target | mean err | contacts | falls |
|---|---|---|---|---|---|---|---|---|
| s0/best | 77.5 % / 30.5 % / 0.37 m | 78.0 % / 29.5 % / 0.38 m | 80.0 % / 32.0 % / 0.39 m | **78.5 %** | 30.7 % | 0.38 m | 175/171/176 | 0/0/0 |
| s0/final | 75.0 % / 51.0 % / 0.27 m | 81.0 % / 47.0 % / 0.29 m | 75.0 % / 50.5 % / 0.30 m | **77.0 %** | 49.5 % | 0.28 m | 177/172/178 | 0/0/0 |
| s1/best | 79.0 % / 31.5 % / 0.38 m | 81.0 % / 32.0 % / 0.40 m | 79.5 % / 28.5 % / 0.40 m | **79.8 %** | 30.7 % | 0.39 m | 178/174/176 | 0/0/0 |
| s1/final | 76.5 % / 44.5 % / 0.31 m | 79.5 % / 47.0 % / 0.31 m | 76.5 % / 46.5 % / 0.31 m | **77.5 %** | 46.0 % | 0.31 m | 179/175/176 | 0/0/0 |
| policies/param_residual_v2_wa0 | 80.5 % / 34.0 % / 0.37 m | 77.0 % / 30.5 % / 0.40 m | 76.0 % / 27.0 % / 0.40 m | **77.8 %** | 30.5 % | 0.39 m | 176/174/177 | 0/0/0 |

200 serves per seed; mean action (`tt_sim.evaluate`); contacts and falls listed per seed (falls is 0 unless --free-base).


## Verdict against the pre-registered bar (design section 4.4)

- **Legal: "the stance is the whole gain."** Best retrain s1/best 79.8 % vs 77.8 % unretrained -- inside the
  77.8-79 band (s1/best grazes 80 but two of nine numbers under 80 means it is not a conversion). The reach lane
  on the welded scene is closed; the ceiling (~81 %) is effectively reached.
- **Placement: the depth bonus worked, decisively.** s0/final: 77.0 % legal, **49.5 % target hits, 0.28 m** --
  the bar was >=35 % at >=77 %. s1/final 77.5 / 46.0 / 0.31. The depth-shortfall term moved the one knob with
  authority (swing speed) instead of the inert aim knobs.
- **Shipped:** `assets/policies/param_residual_v3_s0final.npz` (= s0/final; torch/numpy decode parity 6e-8)
  behind `tt_sim.evaluate_stance` (sets TT_ROBOT_Y=-0.10 and the npz before the scene import). s1/best exported
  as `param_residual_v3_s1best.npz`, the legal-max alternative (79.8 / 30.7 / 0.39).
