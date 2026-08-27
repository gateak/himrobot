# Training plan — HIM targeted-forehand policy

These files are the source of each fact and each command in this document:
`work/pace/PACE_ANALYSIS.md`, `work/pace/pace_pins.json`, and
`work/pace/BOX_OPTIONS.md` (this repo). Read the files again before you write
information that is different from them.

## Goal

The goal is a policy for a **targeted forehand**. HIM Arena scores this
policy against its own metrics: legal-return rate, target-hit rate, landing
error, and latency (`docs/PROJECT_KNOWLEDGE.md` §4). The policy must return a
served ball to a selected zone. A legal return to any zone is not sufficient.
PACE's `t1_tt` task optimizes the nearest available proxy reward:
`reward_future_pass_net` and `reward_table_success` (`PACE_ANALYSIS.md` §4).
A reward for a selected landing zone is different from a reward for any legal
return. This is new reward-shaping work. PACE does not have this work at this
time. PACE's reward table in §4 has no term for a target zone.

## Robot decision — pending HIM

`docs/PROJECT_KNOWLEDGE.md` §5 changes the target robot to **Unitree G1**
(user directive). But HIM's own hardware fixture, "Adam," is
Booster-**K1**-based (`fixture_only`, `run_on_adam_enabled:false`). It is not
yet known which robot HIM will actually score against. Both forks are
file-scoped (`PACE_ANALYSIS.md` §5):
- **K1** (22 DoF, no waist, 4-DoF wrist-less arm): For this fork, a new USD
  import is necessary (not created yet). A new `BOOSTER_K1_TT_CFG`, fixed
  `A`-prefixed joint names, and dropped `Waist` reward terms are also
  necessary. This fork must also move `paddle_index` to a name lookup. K1 is
  nearer than G1 to HIM's real fixture and to PACE's own T1 joint-naming
  convention.
- **G1** (29 DoF, 3-DoF waist, 7-DoF arm+wrist): For this fork, a larger
  rewrite of the action and the observation is necessary (from 21 to 31
  joints). New feet and termination body names are also necessary
  (`.*_ankle_roll_link`, `pelvis`), with an unmeasured paddle offset. But G1
  uses our retargeted mocap data (`work/g1/`) and BeyondMimic's G1 config
  directly (`pace_pins.json` `.motion_imitation`). The codebase has a
  `robots/g1.py` file.

Stage 1 below continues with G1, as given above. Before Stage 2 uses many
GPU-hours on a robot HIM will not score, ask HIM which robot they will score.

## Stage 0 — smoke (PACE T1, no changes)

Do a check of the installation only. Do not do the port work yet.
```
python -m legged_lab.scripts.train --task=t1_tt --num_envs=256 \
  --max_iterations=200 --headless --logger=tensorboard
```
(`PACE_ANALYSIS.md` §8. Run by `scripts/gpu_box_setup.sh` step 8.)

**Exit criteria**: a non-zero, non-NaN `[Play] Hits X/Y` value after the eval
warmup period. `eval_results.csv` contains data. `exported/policy.onnx` and
`policy.pt` are non-empty (§8 points 1–4).

**Est. GPU-hours**: fewer than 1. The README's own install check, for
`--num_envs=64`, uses a 10,000-iteration default. 200 iterations is far fewer
than that. This smoke run uses more parallel envs than the install check, 256
vs 64. **Cost**: BOX_OPTIONS.md §6 gives this cost for a "6h smoke
session": $4.74 at RunPod Community L40S ($0.79/hr). This cost is for the
installation and the run together, not only the run. There is no author
wall-clock number to compare against (`PACE_ANALYSIS.md` §1/§2). The README
and the paper do not give this number.

## Stage 1 — G1 port + our serve distribution

Files (`PACE_ANALYSIS.md` §5):
- a new `legged_lab/assets/unitree/` G1 `ArticulationCfg`
- a new `legged_lab/envs/g1_tt/g1_tt_config.py` (29-entry joint lists,
  `num_actions=29`)
- rewritten joint-deviation reward keys for `waist_yaw/roll/pitch` and the
  wrist
- `paddle_index` moved to a name lookup on `right_wrist_yaw_link`, with a
  newly measured offset
- `feet_body_names` changed to `.*_ankle_roll_link`
- the termination body changed to `pelvis`

Also interchange the serve-randomization constants at
`tt_env_config.py:84-111` for our own serve range. A comment in the code
shows that these constants are for the PACE authors' Vicon rig. Calculate the
new range from the paddle-IMU swing data and the `ball3d_*` tracks
(`work/perception/`). This range is only approximate (see Risks: no camera
intrinsics).

**Exit criteria**: G1 loads and trains with no NaN and no divergence, at a
reduced scale (`--num_envs=512`, a few thousand iterations). The joint limits
and the PD gains stay within range, with no runaway saturation. The ported
robot has a non-zero hit rate.

**Est. GPU-hours**: approximately 10 to 15. This number is for several short
validation runs at a reduced scale, not full-paper-scale training. This is
port debugging work, not final training.

**Cost estimate**: approximately $8 to $12 at RunPod L40S $0.79/hr
(BOX_OPTIONS.md §4/§6).

## Stage 2 — mocap-imitation prior + domain randomization + ONNX export

Add BeyondMimic's `MotionCommand` (`.../mdp/commands.py`) and its imitation
reward terms (`.../mdp/rewards.py`: `motion_global_anchor_position_error_exp`
and others) to the G1 task. Use `scripts/csv_to_npz.py` to change the
`work/g1/*.csv` files. `pace_pins.json`'s
`.motion_imitation.beyondmimic_files_to_graft` field shows these files. Use
`.../robots/g1.py`'s G1 `ArticulationCfg` as a cross-check.

Extend PACE's `EventCfg` domain randomization (`tt_env_config.py:168-246`).
The perception delay randomizes 2–5 substeps for each env at reset (§3). But
the additive observation noise is a fixed scale applied uniformly. It is not
randomized for each episode (§4). Extend that scale for sim-to-real transfer.

Export: `eval.py` always writes `exported/policy.onnx` (§8 point 4). Connect
that file to `him_adapter/policy.py`. At this time, `him_adapter/policy.py`
has only `HoldPolicy` and `BCPolicy`. A policy class that loads ONNX files is
new work. This work is within the `runtime.json`
`clock_domains[0].deadline_ms: 15` budget.

**Exit criteria**: a hit rate and a success rate near PACE's own sim numbers
(at least 96% hit rate, at least 92% success rate, `README.md:23`) on the
ported G1. The exported ONNX loads in `him_adapter/server.py`'s infer loop and
returns a result within 15 ms.

**Est. GPU-hours**: approximately 80 to 120. These are full-scale runs,
`num_envs≈4096`, with repeats for the new imitation-reward weights. This is 2
to 3 times BOX_OPTIONS.md §6's "5-run pass" figure of approximately 40
GPU-hr. No one put these two reward systems together before, in either
codebase. This is new work. The result is not yet known. This does not simply
run a known configuration again.

**Cost estimate**: approximately $63 to $95 at RunPod L40S $0.79/hr.

*All GPU-hour figures above are planning estimates, not measurements from a
real run. Neither PACE's README nor its paper gives a wall-clock training
time (`PACE_ANALYSIS.md` §1/§2).*

## Risks

1. **Isaac Sim version pin**: PACE's README gives this exact statement: an
   upgrade past 4.5.0 "degrades success rate" (`README.md:23,51`,
   `pace_pins.json` `.software_pins`). The box must use only version 4.5.0,
   never "latest."
2. **Sim-to-eval gap**: PACE has no simulated paddle rigid body and no
   collision mesh. A hit is a distance check against a kinematic offset point
   (`PACE_ANALYSIS.md` §5). It is not known if HIM's own Arena evaluator
   models contact in the same way.
3. **K1 asset import**: There is no K1 USD file yet. Run Isaac Sim's URDF importer offline first. K1 has no per-joint PD and effort
   limits yet (§5). This is real work if HIM selects K1, not G1.
4. **No intrinsics**: The `work/perception` ball tracks contain no camera
   intrinsics (`docs/PROJECT_KNOWLEDGE.md` §6a). So Stage 1's own serve
   distribution is an approximate value, not a calibrated range of values.
