# Training plan — HIM targeted-forehand policy

Sources for every fact/command below: `work/pace/PACE_ANALYSIS.md`,
`work/pace/pace_pins.json`, `work/pace/BOX_OPTIONS.md` (this repo). Do not
contradict them without re-reading first.

## Goal

A policy scored against HIM Arena's own metrics — legal-return rate,
target-hit rate, landing error, latency (`docs/PROJECT_KNOWLEDGE.md` §4) —
for a **targeted forehand**: return a served ball to a chosen zone, not just
any legal return. PACE's `t1_tt` task already optimizes the closest available
proxy (`reward_future_pass_net` + `reward_table_success`,
`PACE_ANALYSIS.md` §4); targeting a *specific* landing zone instead of "any
legal return" is new reward-shaping work, not present in PACE today (§4's
reward table has no target-zone term).

## Robot decision — pending HIM

`docs/PROJECT_KNOWLEDGE.md` §5 currently retargets to **Unitree G1** (user
directive). But HIM's own hardware fixture, "Adam," is Booster-**K1**-based
(`fixture_only`, `run_on_adam_enabled:false`) — which robot HIM will actually
score against is unconfirmed. Both forks are file-scoped
(`PACE_ANALYSIS.md` §5):
- **K1** (22 DoF, no waist, 4-DoF wrist-less arm): needs a USD import (none
  exists), a `BOOSTER_K1_TT_CFG`, fixed `A`-prefixed joint names, dropped
  `Waist` reward terms, `paddle_index` moved to a name lookup. Matches HIM's
  real fixture and PACE's own T1 joint-naming convention most closely.
- **G1** (29 DoF, 3-DoF waist, 7-DoF arm+wrist): bigger action/obs rewrite
  (21→31 joints), new feet/termination body names
  (`.*_ankle_roll_link`, `pelvis`), unmeasured paddle offset — but reuses our
  existing retargeted mocap (`work/g1/`) and BeyondMimic's G1 config directly
  (`pace_pins.json` `.motion_imitation`, has `robots/g1.py` already).
Stage 1 below proceeds on G1 per current project direction; confirm with HIM
before Stage 2 sinks GPU-hours into a robot they won't score.

## Stage 0 — smoke (unmodified PACE T1)

Verify the install only; no port work.
```
python -m legged_lab.scripts.train --task=t1_tt --num_envs=256 \
  --max_iterations=200 --headless --logger=tensorboard
```
(`PACE_ANALYSIS.md` §8; run by `scripts/gpu_box_setup.sh` step 8.)
**Exit criteria**: non-zero, non-NaN `[Play] Hits X/Y` after eval warmup;
`eval_results.csv` populated; `exported/policy.onnx` and `policy.pt` non-empty
(§8 points 1–4).
**Est. GPU-hours**: <1 (200 iterations is far fewer than the README's own
`--num_envs=64` verify-install default of 10,000 iterations, even though this
smoke run uses more parallel envs, 256 vs 64). **Cost**: BOX_OPTIONS.md §6's "6h
smoke session" anchor, $4.74 at RunPod Community L40S ($0.79/hr) — covers
install time too, not just the run. No author wall-clock number exists to
benchmark against (`PACE_ANALYSIS.md` §1/§2 — unstated in README and paper).

## Stage 1 — G1 port + our serve distribution

Files (`PACE_ANALYSIS.md` §5): new `legged_lab/assets/unitree/` G1
`ArticulationCfg`; new `legged_lab/envs/g1_tt/g1_tt_config.py` (29-entry
joint lists, `num_actions=29`); rewritten joint-deviation reward keys for
`waist_yaw/roll/pitch` + wrist; `paddle_index` → name lookup on
`right_wrist_yaw_link` with a newly measured offset; `feet_body_names` →
`.*_ankle_roll_link`; termination body → `pelvis`. Also swap
`tt_env_config.py:84-111`'s serve-randomization constants (currently
calibrated to the PACE authors' Vicon rig, per its own code comment) for our
own serve range — derived from the paddle-IMU swing data and `ball3d_*`
tracks (`work/perception/`), flagged as approximate (see Risks: no
intrinsics).
**Exit criteria**: G1 loads and trains without NaN/divergence at reduced
scale (`--num_envs=512`, a few thousand iterations); joint limits/PD gains
hold without runaway saturation; non-zero hit rate on the ported robot.
**Est. GPU-hours**: ~10–15 (several short validation runs at reduced scale,
not full-paper-scale — this is port debugging, not final training).
**Cost estimate**: ~$8–12 at RunPod L40S $0.79/hr (BOX_OPTIONS.md §4/§6).

## Stage 2 — mocap-imitation prior + domain randomization + ONNX export

Graft BeyondMimic's `MotionCommand` (`.../mdp/commands.py`) and imitation
reward terms (`.../mdp/rewards.py`:
`motion_global_anchor_position_error_exp` et al.) into the G1 task, using
`scripts/csv_to_npz.py` to convert `work/g1/*.csv` (`pace_pins.json`
`.motion_imitation.beyondmimic_files_to_graft`); reuse
`.../robots/g1.py`'s existing G1 `ArticulationCfg` as a cross-check.
Extend PACE's existing `EventCfg` domain randomization
(`tt_env_config.py:168-246`) — perception delay already randomizes 2–5
substeps per env at reset (§3), but the additive observation noise is a
fixed scale applied uniformly, *not* randomized per-episode (§4) — widen
that for sim-to-real robustness. Export: `eval.py` writes
`exported/policy.onnx` unconditionally (§8 point 4); wire that file into
`him_adapter/policy.py` (today only `HoldPolicy`/`BCPolicy` exist — an
ONNX-loading policy class is new work) behind the `runtime.json`
`clock_domains[0].deadline_ms: 15` budget.
**Exit criteria**: hit/success rate trending toward PACE's own sim numbers
(≥96% hit, ≥92% success, `README.md:23`) on the ported G1; exported ONNX
loads in `him_adapter/server.py`'s infer loop and returns within 15 ms.
**Est. GPU-hours**: ~80–120 (full-scale runs, `num_envs≈4096`, plus repeats
for the new imitation-reward weights — 2–3× BOX_OPTIONS.md §6's "5-run pass"
anchor of ~40 GPU-hr, since fusing two reward systems that have never been
combined in either codebase is unproven work, not a rerun of a known config).
**Cost estimate**: ~$63–95 at RunPod L40S $0.79/hr.

*All GPU-hour figures above are planning estimates, not sourced
measurements — neither PACE's README nor its paper states a wall-clock
training time (`PACE_ANALYSIS.md` §1/§2).*

## Risks

1. **Isaac Sim version pin**: PACE's README states, verbatim, that upgrading
   past 4.5.0 "degrades success rate" (`README.md:23,51`,
   `pace_pins.json` `.software_pins`) — the box must pin exactly 4.5.0, never
   "latest."
2. **Sim-to-eval gap**: PACE has no simulated paddle rigid body or collision
   mesh at all — a hit is a distance check against a kinematic offset point
   (`PACE_ANALYSIS.md` §5). Whether HIM's own Arena evaluator models contact
   the same way is unconfirmed.
3. **K1 asset import**: no K1 USD exists yet; Isaac Sim's URDF importer has
   to be run offline first, and K1's per-joint PD/effort limits are
   unauthored (§5) — real work if HIM picks K1 over G1.
4. **No intrinsics**: `work/perception` ball tracks carry no camera
   intrinsics (`docs/PROJECT_KNOWLEDGE.md` §6a), so Stage 1's "our serve
   distribution" is an approximation, not a calibrated metric range.
