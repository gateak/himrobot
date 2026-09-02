# Whole-body v1 "Anchor" — a learned lower body that holds its ground while the shipped hitter swings

_Design, 2026-09-02. Status: design only; nothing here has been trained, no box has been rented, no file in
this design exists yet. Numbers marked (measured) come from files in this repo or commands run while writing
this document; everything else is a stated assumption._

## In plain language

Today the free-standing G1 falls over in 76 % of serves. `results/waist_lock/RESULT.md` traced why: it is not
the legs and not the swing's speed — it is the hitter's own **waist**. The scripted hitter twists and bends the
torso hard during every swing (waist yaw up to 1.9 rad, roll and pitch pinned at their 0.52 rad stops) while the
legs stand perfectly still, so the body is thrown and nothing catches it. Bolting the waist rigid stops the falls
(14.8 %) but costs reach (45 % legal instead of 61 %). Two nights of leg-only balance training changed nothing,
because ±0.15–0.40 rad of leg twitching around a fixed pose cannot absorb a torso that swings to its limits.

The fix proposed here is the smallest thing that can work: **keep the hitter exactly as it is** (it still owns the
waist and both arms, it still aims, it still ships), and train one new controller — a *whole-body controller*
(WBC), a small neural network — that owns the twelve leg joints with real authority, plus a small correction on
the three waist joints, and whose only job is **to keep the pelvis where the hitter assumes it is** while the
hitter's swing plays out. Its secret weapon is that it can see the future: the hitter's upcoming joint commands
0.1 s and 0.2 s ahead are handed to it as input, so it braces *before* the torso moves instead of reacting after.

Why this is the right target: the hitter's arm maths assumes the pelvis never moves (the welded scene). Every
centimetre the WBC holds the pelvis is paddle accuracy for free, and letting the waist move again returns the
reach that locking it cost. A human stands through a table-tennis swing the same way — legs and hips absorb the
torso — and our 33 minutes of human mocap tell the WBC what a plausible lower body looks like.

Training happens on a rented GPU box in Isaac Lab (NVIDIA's massively parallel robot simulator), because the
controller has to see millions of swings; the *scoring* happens on the Mac, in the same MuJoCo evaluator that
produced every number on the scoreboard, at no cost. The first result is either a pass or a fail against a bar
written down here before any credit is spent.

**The bar (600 serves, honest free-base scene, mean action):** falls ≤ 5 % **and** legal returns ≥ 55 % with the
waist free — beating both of today's honest rows (45.3 % / 14.8 % falls locked; 61.5 % / 75.7 % falls free) at once.

**Cost:** 372 credits for a 4 h chain-proof box that may already produce the result, plus 744 credits for an 8 h
training box if it does not — 1,116 credits on the L4 class, nothing launched without an explicit "launch it".

---

## 0. Decision record

Four designs were drafted and judged (three independent judges). The tally shipped with the brief double-counted
by name string; summed per design over the three judges the totals are: **Anchor 89.5**, Swing-reel tracker 88.5,
HITTER-G1 62, PACE end-to-end 47, and two of the three judges picked Anchor as the winner. This document is Anchor
with the judges' grafts folded in and every fatal flaw they found fixed:

| Flaw found | Where | Fix in this design |
|---|---|---|
| "The scored MuJoCo scene has no torque limit" — false. Every joint carries `jnt_actfrcrange` (±88 hip pitch/yaw + waist yaw, ±139 hip roll/knee, ±50 ankles + waist roll/pitch, ±25 shoulders/elbow/wrist roll, ±5 wrist pitch/yaw N·m) with `jnt_actfrclimited=True` (measured, §3.2). | HITTER-G1 | The Isaac actuator table is generated from the compiled MuJoCo model and includes the joint-level clamps as `effort_limit`; a torque regulariser stays in the reward. A test pins the table against the model. |
| Scoring the result under a different actuator model than the baselines (BeyondMimic gains). | Swing-reel | The scored scene is never changed. Isaac mirrors MuJoCo, never the reverse. Rows are comparable to `results/waist_lock/RESULT.md` by construction. |
| "PACE's USD carries a fused paddle" — contradicts `work/pace/PACE_ANALYSIS.md` §5. | PACE e2e | PACE is not used here at all. |
| Hitter contact time "typically 0.5–1.05 s" — the measured distribution is 0.48–1.37 s, median 0.79 s. | Swing-reel | Preview horizons and library time-jitter are sized from the measured bank (§4.3). |

Grafts taken from the other three designs (judges' lists, deduplicated): a two-sided open-loop replay gate before
training (§7.2); a held-out library built from the exact 600 evaluator serves (§4.2); scoring the last three
checkpoints on the box before it stops (§7.3); per-trial diagnostics next to the headline number (§8.3); a
numpy-vs-torch observation parity test (§9, item 3); a version gate in the first ten minutes with a hard abort
(§7.2); iterations sized from a 200-iteration smoke run (§7.2); `--max-credits` with slack (§7.1); resets from
the policy's own post-swing states plus idle rows (§6.1); the measured planner clock bank as the sizing source
for previews and for the later stepping command (§4.3, §10); public `plan()` and `preview()` accessors on the
hitter, parity-tested (§5.2); the mocap quality mask and a five-swing human preview before trusting the swing
labels (§4.4); pre-strike gating of the station term as the bridge to stepping (§5.4); the BeyondMimic compliant
gain table recorded as a *later* real-G1 option, not swapped in (§10).

---

## 1. Thesis

The free-base falls are caused by the torso the IK drives kinematically with nothing coupling it to the legs.
Measured on the honest free-base scene (`robot_table_collision=false`, paddle meets only the ball), 600 serves,
seeds 11/2000/4000, mean action (`results/waist_lock/RESULT.md`):

| policy | legal | target | err | contacts | falls |
|---|---|---|---|---|---|
| v3 hitter, waist free | 61.5 % | 4.8 % | 0.69 m | 78 % | **75.7 %** |
| v3 hitter, waist locked (`TT_WAIST_LIMIT=0,0`) | 45.3 % | 3.5 % | 0.68 m | 57 % | **14.8 %** |
| best leg-only balance run (v1.2 c) | 39.2 % | 4.0 % | 0.68 m | — | 60.3 % |
| welded v3 (reference, pelvis bolted) | 77.0 % | 49.5 % | 0.28 m | 88 % | — |

Three facts about the shipped v3 policy, re-measured while writing this (40 welded serves at `TT_ROBOT_Y=-0.10`,
scratchpad prototype `swing_lib_proto.npz`):

- Its 29-dim joint target moves **only the ten IK joints** (waist yaw/roll/pitch + right arm). All twelve leg
  joints and the seven left-arm joints are exactly `STAND_POSE` on every control step.
- The swing window (first to last step where the target differs from the rest pose) is **27–38 control steps**
  (0.54–0.76 s), starting between step 3 and step 40 (0.06–0.80 s after the serve).
- Waist excursions: |yaw| up to 1.93 rad, |roll| and |pitch| pinned at 0.52 rad in essentially every serve
  (means 1.04 / 0.518 / 0.519).

Because the hitter's IK runs on a private *welded* shadow model (`ik_hitter.py`, `_solve_ik_stock`), its arm
solution is right exactly when the real pelvis sits at `scene.BASE_NOMINAL_POS/QUAT`. So the controller's task
reduces to **station-keeping under a previewed disturbance**: hold the pelvis at the nominal pose and heading
while a known 17-channel upper-body trajectory plays. That needs no ball, no goal, no body-pose tracker and no
motion-capture retargeting into Isaac — only the hitter's own recorded commands (free, MuJoCo, minutes) and a
lower-body pose prior from the human clips.

---

## 2. Recipe — codebases and exactly what is taken

**Simulator / RL stack (box side).** Isaac Lab v2.1.0 on Isaac Sim 4.5.0, Python 3.10, rsl_rl — the pins in
`tools/box/gpu_box_setup.sh` lines 29–38 and `work/pace/pace_pins.json`. If the `him-robotics` image's
preinstall reports Isaac Lab 2.1.x or 2.2.x (`pip show isaaclab isaacsim rsl-rl-lib`), use it; otherwise run
`gpu_box_setup.sh` steps 1–5 (driver / RT-core check, OS check, apt, venv + torch 2.5.1 + isaacsim 4.5.0, Isaac
Lab v2.1.0). Steps 6–9 (PACE) are not used.

**Robot asset.** Convert **our own** MJCF, not BeyondMimic's URDF and not Isaac Lab's 23-DoF `g1.usd`
(`work/pace/g1_port/PORT.md` §2 — that body has no waist roll/pitch and no wrist pitch/yaw):
`assets/wbc/g1_29dof_paddle_free.xml` = the vendored `assets/unitree_g1/g1.xml` with `scene.PADDLE_XML` injected
at `scene._PADDLE_MARKER` (the same 0.17 kg paddle, blade centre 0.205 m along wrist +x, as the scored scene),
freejoint kept, vendor keyframe dropped, no table/ball. Converted on the box with Isaac Lab's
`scripts/tools/convert_mjcf.py` (`MjcfConverter`). Fallback if conversion fails its checks: BeyondMimic's
`unitree_description/urdf/g1/main.urdf` (verified reachable, 55.6 MB) with a fixed 0.17 kg paddle link added —
then link masses may differ slightly from the scored scene and `check_model_parity` (item 4) says by how much.

**Actuators (the mirror).** Copied from the **compiled scored model**, never from BeyondMimic and never typed by
hand: `assets/wbc/actuators_mujoco.json` (item 4) → Isaac `ImplicitActuatorCfg` per joint. See §3.2.

**From BeyondMimic (`HybridRobotics/whole_body_tracking` @ `cd65172032`, MIT; cloned on the box only):** the
exp-kernel reward form, `mdp/events.py` (material randomisation, default-joint-pos jitter, CoM offset, velocity
push), `mdp/terminations.py` (bad orientation, root height), the PPO hyperparameters of
`config/g1/agents/rsl_rl_ppo_cfg.py` ([512,256,128] ELU, lr 1e-3 adaptive with desired KL 0.01, 24 steps/env,
5 epochs, 4 minibatches, γ 0.99, λ 0.95, entropy 0.005, save every 500) and the adaptive-sampling idea (failure-
weighted EMA, floor 0.1) applied to *library rows* instead of motion-time bins. **Not taken:** MotionCommand / npz
motion tracking, the 160-D tracking observation, WandB plumbing, the ONNX-with-baked-motion exporter.

**From HOPE (`hitchopen/HOPE` @ `52fce34`, Apache-2.0):** two ideas only — "the base-position term is dominant
before the strike and relaxed after it" (§5.4) and the ring buffer of the policy's own post-swing states as reset
states (§6.1). The MuJoCo sim-to-sim gate pattern (`scripts/mujoco_eval_onnx.py`) is re-implemented as
`himrobot/wbc/replay_library.py` against our own scene.

**From HITTER (arXiv 2508.21043):** the observation form "base forward vector in world axes + goal in world
axes" for the heading slot, and nothing else in v1; its station command becomes v2's stepping interface (§10).

**From himrobot:** `himrobot/rl/balance_env.py`'s pattern B (RL owns a disjoint joint slice; the expert's
command passes through) with `body_frame_gravity`, `tilt_angle`, `box_excursion_sq` reused; `himrobot/policies/
balance_np.py`'s torch-free runtime shape (npz with `W*/b*`, `rms_mean/var`, `obs_version`, scales, `expert_npz`);
`himrobot/rl/balance_train.py::export_npz` as the export convention; `himrobot/rl/demos.py::target_to_action`;
`himrobot/eval/evaluate.py --free-base --params` and `himrobot/eval/eval_seeds.py --policy label=module:Class`
unchanged; `tools/box/box_param_sweep.sh` (push timer, SIGINT before expiry, self-stop) and
`tools/box/box_push_results.py` as the box conventions; `work/pace/g1_port/stub_import_smoke.py` + `checks.py`
as the Mac-side gate pattern for Isaac config files; `tools/mocap/retarget_report.py::robot_fk/load_robot` and
the scratchpad `mocap_swing_stats.py` heuristic for the mocap prior.

**New package:** `himrobot/wbc/` (runtime-side: library recorder, observation layout, action mapping, replay
gate, export, diagnostics) and `himrobot/wbc/isaac/` (imports `isaaclab` only on the box; importable on the Mac
through the stub gate). Task id `Himrobot-WBC-G1-v0`.

**Fallback engine (named, not primary):** `mujocolab/mjlab` (MuJoCo-Warp, Isaac-Lab-style manager API, loads
our MJCF directly, runs on the `standard` image) if the PhysX→MuJoCo replay gate cannot be made to pass.

---

## 3. Interfaces

### 3.1 Observation — layout `wbc_obs_v1`, 141 floats

Defined once in `himrobot/wbc/obs.py` (numpy) and mirrored slot-for-slot by
`himrobot/wbc/isaac/mdp/observations.py` (torch); a dims-and-order test pins both, and a parity test runs both
builders on 1,000 random poses to 1e-6.

| slots | content | source (evaluator obs dict) | train noise (uniform) |
|---|---|---|---|
| 0:3 | gravity unit vector in the pelvis frame | `balance_env.body_frame_gravity(base.orientation)` | ±0.05 |
| 3:6 | base angular velocity, pelvis frame | `base.angular_velocity` **as is** — MuJoCo free-joint `qvel[3:6]` is already body-frame | ±0.2 |
| 6:9 | base linear velocity, pelvis frame | `R_baseᵀ · base.linear_velocity` — MuJoCo free-joint `qvel[0:3]` is **world**-frame | ±0.5 |
| 9:12 | pelvis position − `scene.BASE_NOMINAL_POS`, world | `base.position` | ±0.02 |
| 12:14 | heading (cos ψ, sin ψ) of the pelvis +x axis in world xy | `base.orientation` | ±0.02 |
| 14:43 | joint position − default pose (29) | `robot.joint_position`; default = concat(`leg_default`, `STAND_POSE[12:]`) | ±0.01 |
| 43:72 | joint velocity × 0.1 (29) | `robot.joint_velocity` (the `env.flatten_obs` scaling) | ±0.5 (pre-scale) |
| 72:87 | previous WBC action (15) | runtime state | — |
| 87:104 | hitter command now: u(t) − `STAND_POSE` on `JOINT_ORDER[12:29]` (waist 3 + left arm 7 + right arm 7) | live expert `act()` at test; library row at train | — |
| 104:121 | u(t+5) − `STAND_POSE` (0.1 s preview) | `IKHitterPolicy.preview(obs, 10)[4]` at test; library at train | — |
| 121:138 | u(t+10) − `STAND_POSE` (0.2 s preview) | `preview(...)[9]` at test; library at train | — |
| 138:141 | swing clock: in-window flag, seconds since window start, seconds to predicted contact (each clipped to ±1) | `IKHitterPolicy.plan(obs)` (`t_contact`, `t_pre`, `speed_buffer`, `t_follow`); row metadata at train | — |

The left arm is constant in every row (measured) but is kept so the command slice is literally
`JOINT_ORDER[12:29]`. No ball state, no paddle state, no history stacking. Critic (asymmetric): the 141 without
noise plus u(t+15), u(t+25) (34) and the row's `serve_scale` (1) = 176. Observation normalisation: rsl_rl
`empirical_normalization=True`, exported as `rms_mean`/`rms_var` exactly as `balance_np` consumes them.

Variant `wbc_obs_v1_noimu` drops slots 6:12 for a HIM evaluator that streams no base state; a flag, not trained
first (open question §11).

### 3.2 Action — 15 floats, and the actuator mirror

Raw MLP output, clipped to ±1 before scaling (runtime and Isaac action term alike):

- **Legs `a[0:12]`:** `q_des_leg = leg_default + 0.5 rad · a`, clipped to `jnt_range`. 0.5 rad is the authority
  balance v1 lacked (mocap hip/knee excursions span ~0.6–0.9 rad between p5 and p95).
- **Waist `a[12:15]`:** `q_des_waist = u_waist(t) + 0.2 rad · a`, clipped to the model limits (±0.52 roll/pitch,
  ±2.618 yaw). A bounded *compensation residual* on the hitter's own waist command, penalised so it damps rather
  than re-plans.
- **Arms:** `u_arm(t)` passthrough — the hitter's 14 values, untouched.

Assembled into the 29-vector by `himrobot/wbc/action.py::joint_target_wbc(a, u, jnt_range, leg_default)`
(numpy) and `himrobot/wbc/isaac/mdp/actions.py::WBCJointAction` (torch, same arithmetic). **Parity anchor:**
`a = 0` with `leg_default = STAND_POSE[:12]` reproduces today's free-base hitter bit-for-bit; the final actor layer
is zero-initialised so iteration 0 *is* the 61.5 % / 75.7 %-falls row. `leg_default` is a run flag: `stand`
(`STAND_POSE`, parity; box 1) or `mocap` (the play-clip mean lower body, §4.4; box 2 candidate).

**Actuator mirror (measured on `scene.build_model(fixed_base=False, robot_table_collision=False)`):** every joint
is a MuJoCo `<position>` actuator with kp = 500, `dampratio = 1` (compiled kv per joint, e.g. hip pitch 43.0, hip
roll 39.6, hip yaw 9.8, knee 15.8, ankle pitch 5.1, ankle roll 4.6, waist yaw 24.3, waist roll 37.5, waist pitch
35.8, shoulders 10–18, elbows 9–11, wrists 4.6–6.5; the right arm differs from the left because the paddle mass
is in the reflected inertia), armature 0.01, frictionloss 0.3, `actuator_forcerange` unlimited **but
`jnt_actfrcrange` set and enforced on all 29 joints** (±88 / ±139 / ±50 / ±25 / ±5 N·m as in §0), total mass
33.51 kg. Isaac's `ImplicitActuatorCfg` per joint takes stiffness = kp, damping = kv, `effort_limit` =
`jnt_actfrcrange`, armature and friction from the same JSON; `soft_joint_pos_limit_factor = 1.0` (the hitter
commands the model limits). Physics rate: Isaac `sim.dt = 0.002`, decimation 10 — the evaluator's exact
`SIM_DT × DECIMATION`; only if the smoke run shows this costs more than 2× the throughput of `0.004 × 5` does the
latter become the run setting (recorded in `smoke_timing.json`, never silently).

### 3.3 Reward, terminations, episode

Per 50 Hz step; exp kernels as in BeyondMimic; weights are the pre-registered starting point.

| term | weight × kernel | note |
|---|---|---|
| station | 1.0 × exp(−|p_xy − p_nom,xy|² / 0.05²) | **this term is paddle accuracy** for a nominal-pelvis IK; full weight until `t_contact + t_follow` |
| station, post-swing | 0.3 × same + (−1.0 × `box_excursion_sq`, box ±0.15 × ±0.30 m) | HOPE's pre-strike gating: after the swing the robot may shuffle inside the box (bridge to stepping) |
| height | 0.5 × exp(−(z − 0.793)² / 0.05²) | |
| upright | 1.0 × exp(−tilt² / 0.15²) | `tilt = balance_env.tilt_angle(gravity_body)` |
| heading | 0.5 × exp(−ψ_err² / 0.2²) | |
| base velocity | 0.5 × exp(−|v_lin|² / 0.5²) + 0.3 × exp(−|ω|² / 1.0²) | |
| feet | +0.3 if both `ankle_roll_link`s in contact (> 1 N); −0.5 × foot slip (xy speed of a foot in contact) | |
| mocap leg prior | 0.3 × exp(−|q_legs\ankles − q_prior(phase)|² / 0.4²) | phase from the swing clock; ankles excluded (§4.4) |
| regularisers | −0.1 action-rate², −1e-4 leg joint-vel², **−2e-5 leg torque²** (torques are clamped, so the term is meaningful), −10 joint-limit violation, −0.5 × |waist residual|², −0.1 undesired contact (any body but the two feet touching the ground, > 1 N) | |

Terminations: pelvis z < 0.45 m (= `scene.FALL_PELVIS_Z`, so Isaac and the evaluator agree on what a fall is),
tilt > 0.7 rad, time-out at 150 steps (3 s = `evaluate.EPISODE_CONTROL_STEPS`); a termination costs −10 once.
No ball, no return reward: return quality is measured only by `himrobot.eval.evaluate` on MuJoCo. Isaac
diagnostics to watch: mean episode length (must leave 1 within a few hundred iterations and approach 150),
held-out fall rate on the eval library, pelvis-xy error at each row's contact step.

---

## 4. Data: the swing library and the motion prior

### 4.1 The hitter's swing library (train-time stand-in for the hitter)

`himrobot/wbc/record_library.py` runs the shipped `assets/policies/param_residual_v3_s0final.npz` on the
**welded** scene at `TT_ROBOT_Y=-0.10` (the shipped stance) for 150 control steps per serve and stores, per row:
`u[150, 17]` float32 (`JOINT_ORDER[12:29]` targets), `swing_start_step`, `contact_step`, `t_contact`, `t_pre`,
`t_follow`, the serve dict, and the welded legal/contact outcome. Serves from `scene.sample_serve` with seeds
≥ 5000 (disjoint from the scoreboard's 11/2000/4000 and the trainer's 1001). Three libraries: **nominal**
(N = 4000, `serve_scale` 1.0), **wide** (4000, `serve_scale` 1.3), **slow** (2000, `TT_SWING_SPEED=0.5`);
4000 rows = 40.8 MB float32; the 40-row prototype took 4.7 s, so ≈ 8 min per 4000 rows. Each Isaac env samples a
row at reset, plays `u(t)` as the arm/waist command through `WBCJointAction`, and reads `u(t+5)`, `u(t+10)` for
the preview slots — the library is one GPU tensor, one gather per step.

### 4.2 The held-out library (graft)

`swing_lib_eval.npz`: the **exact 600 evaluator serves** — `default_rng(seed + k)` for seeds 11/2000/4000,
k < 200, drawn as `evaluate.run_trial` draws them — so the Isaac held-out fall rate and the Mac scoreboard share
serves and the sim-to-sim gap is one line in `RESULT.md`. Never sampled in training.

### 4.3 Sizing from the measured planner clock (graft)

On 1,000 seed-2000 serves with the shipped v3 policy (`scratchpad/hitter_targets_seed2000.npz`, measured):
`t_contact` 0.48–1.37 s (p10 0.60, median 0.79, p90 1.08); 99.4 % ≥ 0.5 s, 89.5 % ≥ 0.6 s; `t_pre` 0.27–0.43 s
(median 0.38), `t_follow` 0.18–0.28 s; hit point world y −0.85…−0.10 (median −0.47, i.e. 0.37 m to the right of
the −0.10 stance), z 0.78–1.36 m. Consequences: preview horizons of 5 and 10 steps sit inside the swing window
for most of it; library time-jitter is ±3 steps; the swing clock's "seconds to contact" slot is clipped at ±1 s.

### 4.4 The mocap lower-body prior (`tools/mocap/mocap_leg_prior.py`)

`work/g1/Clip_2..8.csv` (36 columns: root xyz, quat xyzw, 29 joints in `gmr_g1_joint_order.txt` =
`scene.JOINT_ORDER`, 30 fps; 59,161 frames = 32.9 min, measured) are used as a **pose prior, not a tracking
target** — no npz/FK export into Isaac. Per clip: FK with `retarget_report.py::robot_fk`; swing events = right-
wrist speed > 2 m/s merged across gaps ≤ 3 frames (≈ 238 events across the seven play clips per the reader
digest); a 1.5 s window per event (0.6 s before to 0.9 s after the speed peak) normalised into 10 phase bins;
per-bin mean/std of the 12 leg joints. Quality mask (graft): frames with pelvis-frame wrist error > 0.20 m
rejected, windows containing joint-limit saturation rejected, windows with min foot z < −0.05 m rejected;
`Clip.csv` (17.5 s calibration take) excluded; `Clip_7` (worst saturation, ambiguous alignment) weighted 0.5.
Ankles are kept in the file but excluded from the reward (foot penetration of −0.03…−0.08 m makes them the least
trustworthy). Output `assets/wbc/mocap_leg_prior.npz` with the global mean/std as well — measured today over all
play frames: mean `[−0.20, 0.13, 0.22, 0.39, −0.13, −0.05, −0.06, −0.22, −0.37, 0.34, −0.18, 0.04]` rad, std
0.12–0.25 rad, root z medians 0.78–0.81 m — so the `mocap` leg default keeps the pelvis near the hitter's assumed
0.793 m. The same script stores, for later lanes and at no extra cost, the phase-binned 17-joint upper-body
swing bank and root-xy velocity statistics. Before trusting the 2 m/s label, five flagged swings are rendered
with `tools/mocap/play_g1_csv.py --video` (Clip_4 frames 5521–6121 first) and the verdict is written into the
npz's `notes`.

---

## 5. Hitter composition

### 5.1 Train time (Isaac)

The hitter is not in the loop. Its behaviour is the library (§4.1). Nothing in `himrobot/policies/` runs on the
box except `scene.py` constants imported for the joint order and `STAND_POSE`.

### 5.2 Two pure additions to `IKHitterPolicy` (graft)

- `plan(obs) -> dict | None`: runs `_maybe_replan` and returns `self._plan` (keys `t_contact`, `p_hit`, `v_in`,
  `swing_vel`, `n_des`, `t_pre`, `speed_buffer`, `t_follow`, …). Touches no IK.
- `preview(obs, horizon) -> np.ndarray[horizon, 29]`: rolls `_swing_target(t + k·dt)` + `_solve_ik` forward
  from the measured `q29` on a **private** `MjData`, seeding each step with the previous solution (open loop);
  `STAND_POSE` rows outside the window or with no plan. Never writes `self._ik_data` or `self._plan`.

Both are proven not to change `act()`: `tools/make_golden_param_actions.py --check` still passes, and a test
interleaves `plan()`/`preview()` calls between `act()` calls and asserts bit-identical actions.

### 5.3 Test time (MuJoCo — our evaluator and the HIM package)

`himrobot/policies/wbc_np.py::NumpyWBCPolicy(Policy)` wraps an **unchanged** `NumpyParamResidualPolicy`.
Per step: `u = expert.act(obs)` (29), `pre = expert.hitter.preview(obs, 10)`, `plan = expert.hitter.plan(obs)`,
observation by `wbc_obs.flatten`, normaliser + MLP [512,256,128] ELU from the npz, `joint_target_wbc` → the
29-vector `TableTennisEnv.step` consumes. `reset()` forwards to the expert. The hitter keeps the ball, the aim,
the swing and the arm; the WBC sees only joint targets and its own body. Loaded by
`himrobot.eval.evaluate --policy himrobot.policies.wbc_np:NumpyWBCPolicy --free-base` with `TT_WBC_NPZ` (the
`TT_BALANCE_NPZ` pattern). `TT_WAIST_LIMIT` stays unset (waist free); `TT_BASE_AWARE` stays off in v1.

Known train/test gap, measured before the box: the library's `u(t+k)` is the closed-loop recorded future; the
runtime preview is an open-loop IK roll-out from the measured state. `himrobot/wbc/preview_error.py` reports the
joint-space RMS at k = 5, 10 over 200 welded serves; if it exceeds 0.05 rad the libraries are re-recorded storing
`preview(t)` instead of the true future (free, ~8 min per library). The hitter re-solves its IK from measured joints
every step, so a waist residual is partly re-absorbed by the arm live — hence the 0.2 rad bound and the residual
penalty.

### 5.4 Why not the other compositions (stated so nobody re-derives them)

Legs-only under the hitter (balance v1/v1.2): measured not to work — the disturbance is the waist. A full 29-dim
residual over the hitter (`VecTTResidual` pattern): inherits the waist-to-limit swings as the mean action. A
goal-conditioned tracker that owns the arm (HITTER-G1) or end-to-end PPO with a ball (PACE): the right long-term
architectures, but they discard the best-tuned asset we have (77 % welded) and need several tuning rounds; they
are the v2/v3 of this lane, not v1 (§10).

---

## 6. Curriculum and randomisation

### 6.1 Resets and row sampling

- Library mix by iteration: 0–1000: 70 % slow / 30 % nominal (learn to stand under a half-speed swing; the
  parity start already stands ~25 % of nominal rows); 1000–3000: 20 / 60 / 20 (slow/nominal/wide); 3000+:
  10 / 50 / 40, plus adaptive sampling over rows (EMA of terminations per row, α 0.001, uniform floor 0.1).
- Episode start (graft): 25 % of resets at row step 0 from the reset pose; 45 % at a random step in
  `[0, swing_start − 5]` with the robot at the nominal pose plus noise (xy ±0.03 m, yaw ±0.1 rad, joints
  ±0.05 rad); **25 % from the policy's own post-swing states** (ring buffer of 8192 states captured at
  `contact_step + 25`, HOPE's mechanism — the falls land 1.8–2.9 s into the episode, in the recovery phase, so
  that phase needs its own samples); **5 % idle rows** (u ≡ `STAND_POSE` for 150 steps) so standing still stays
  in-distribution and the freeze-and-never-move collapse is visible as a separate curve.
- Library time-jitter ±3 steps so the preview is not perfectly phase-locked.

### 6.2 Domain randomisation (startup unless noted)

Ground friction static 0.4–1.0 / dynamic 0.3–0.8 (our floor is 0.6), restitution 0; link mass scale 0.9–1.1 on
all bodies; paddle mass 0.15–0.20 kg; PD stiffness and damping scale 0.9–1.1 (covers PhysX-vs-MuJoCo gain
realisation); armature × 0.8–1.2; default joint pos ±0.01 rad; torso CoM offset x ±0.025, y/z ±0.05 m; velocity
push every 2–4 s, lin ±0.3 m/s, ang ±0.3 rad/s (smaller than BeyondMimic's ±0.5: the swing *is* the
disturbance); actuator delay 0–1 control step (50 % of envs); observation noise per §3.1. **Not randomised:** the
swing itself (the library is the true distribution) and `leg_default` within a run.

---

## 7. Box plan

### 7.1 Machines and credit arithmetic

Prices from `him machine capabilities` (measured): `gpu-l4-workspace` 1×24 GB, 93 credits/h;
`gpu-l40s-workspace` 1×48 GB, 186 credits/h; 24 h lifetime cap; image `him-robotics` (CUDA + Isaac Sim + Isaac
Lab + MuJoCo + GMR, versions unknown). The default `--max-credits 0` ceiling equals `creditsPerHour × hours`
exactly (confirmed from `him machine list`), so every launch passes an explicit ceiling with ~8 % slack.

| box | class | hours | credits | ceiling passed |
|---|---|---|---|---|
| 1 — chain proof + smoke + short train + on-box score | L4 | 4 | **372** | `--max-credits 400` |
| 2 — main run (`--resume` from box 1) | L4 | 8 | **744** | `--max-credits 800` |
| 2 alt — main run, 12 h | L4 | 12 | 1,116 | `--max-credits 1200` |
| 1 on L40S (only if VRAM forces it) | L40S | 4 | 744 | `--max-credits 800` |
| 2 on L40S | L40S | 8 | 1,488 | `--max-credits 1600` |
| 2 on L40S, 12 h | L40S | 12 | 2,232 | `--max-credits 2400` |

**Quoted for the first result: 372 + 744 = 1,116 credits** on the L4 class (worst case with L40S for both:
2,232). The L40S costs exactly 2× for 2× VRAM; it is justified only if the smoke run shows 4096 envs do not fit
in 24 GB *and* 2048 envs gives < 0.6× the throughput. No dollar rate for credits is known (open item in
`work/pace/BOX_OPTIONS.md`).

Launch shape (only after "launch it"):
`export HIM_API_URL=https://arena.himrobotics.com; ~/.local/bin/him machine launch --profile gpu-l4-workspace
--environment him-robotics --hours 4 --max-credits 400 --yes --json`.

### 7.2 Box 1 timeline (4 h), all under `MACHINE_ID=ssh_xxx HOURS=4 nohup bash tools/box/wbc_box_run.sh`

Every Python call is `/workspace/venv/bin/python`; the script resolves which interpreter imports `isaaclab` and
symlinks it to `/workspace/venv` if the image puts it elsewhere.

- **0:00–0:10** rsync the repo (`himrobot/`, `tools/`, `tests/`, `assets/`; `work/wbc/*.npz` libraries), `nvidia-smi`,
  `pip show isaaclab isaacsim rsl-rl-lib`, `python -c "import isaaclab"`. **Gate 1 (version):** 2.1.x/2.2.x
  continue; anything else runs `gpu_box_setup.sh` steps 1–5 with a **hard 90-minute abort** that pushes
  `STATUS.md` and stops the box (worst case ≈ 1.7 h = 160 credits lost, never 4 h).
- **0:10–0:30** `convert_mjcf.py assets/wbc/g1_29dof_paddle_free.xml → /workspace/assets/g1_29dof_paddle.usd`;
  verify 29 joint names in `JOINT_ORDER`, paddle body pose at `STAND_POSE` within 5 mm and blade-normal dot
  > 0.999 of the MuJoCo truth (`check_model_parity` output shipped with the repo), 10 zero-action steps with no NaN.
- **0:30–0:45 Gate 0 (two-sided replay, graft):** `replay_zero_action.py --swing_lib swing_lib_eval.npz
  --num_envs 600 --headless` plays the 600 held-out rows open-loop with `a = 0` in Isaac; the Mac has already
  produced the MuJoCo side with `himrobot/wbc/replay_library.py` on the same rows. Pass = STAND_POSE-only rows
  stand 3 s (0 falls) and the library rows' fall rate is within ±15 points of MuJoCo's. Fail → stop the box
  (actuator/asset problem, fix locally; ≈ 70 credits spent).
- **0:45–1:00 smoke:** `train.py --task Himrobot-WBC-G1-v0 --num_envs 4096 --max_iterations 200 --headless
  --logger tensorboard --run_name smoke --swing_lib …nominal.npz` → `smoke_timing.json` (s/iteration,
  nvidia-smi peak). OOM → 2048 envs. Iterations for the main run are sized from this number, never guessed.
- **1:00–3:05 train:** `--run_name wbc_v1_a --leg_default stand --curriculum default`, `model_<it>.pt` every
  500, wall-clock capped by the script; export to npz every 30 min; push every 30 min.
- **3:05–3:25 (graft):** SIGINT, export the last three checkpoints, score each with
  `himrobot.eval.eval_seeds --free-base --params '{"robot_table_collision": false}' --jobs $(nproc)` on the box
  CPU (the previous L4 boxes had 128 cores; ≈ 5 min per policy), pick by legal × stand, write `RESULT.md`, final
  push, `him machine stop $MACHINE_ID --yes` (STOP_BEFORE_MIN = 35).

If box 1's scored policy already meets the bar (§8), box 2 is not launched.

### 7.3 Box 2 (8 h, 744 credits)

Same setup (~30 min), `--resume` from box 1's best checkpoint (downloaded from the HIM dataset on the box), full
curriculum, `leg_default` chosen from box 1's per-trial diagnostics (§8.3), same 30-min push / SIGINT at T−35 /
score-last-three / self-stop sequence.

### 7.4 Persistence convention (decision needed)

Recommended: HIM-push + self-stop as on every CPU box so far — `tools/box/box_push_results.py --root
logs/rsl_rl/wbc_v1 --machine $MACHINE_ID` (gaining a `--glob` option for `model_*.pt`, `exported/*.npz`,
`params/*.yaml`, `events*`, `smoke_timing.json`, `STATUS.md`, `RESULT.md`; dataset `himrobot-rl-checkpoints`, a
new version per push). This needs `him login` on the box, i.e. the owner's explicit approval per `CLAUDE.md`. The
alternative is `work/pace/g1_port/isaac_box_run.sh`'s rsync-pull design, which cannot stop the box and needs a
manual alarm.

---

## 8. First-result criterion (pre-registered)

### 8.1 The command

```
TT_ROBOT_Y=-0.10 TT_WBC_NPZ=work/wbc/v1/best.npz ./.venv/bin/python -m himrobot.eval.eval_seeds \
  --policy wbc_v1=himrobot.policies.wbc_np:NumpyWBCPolicy --free-base \
  --params '{"robot_table_collision": false}' --seeds 11 2000 4000 --trials 200 --jobs 6 \
  --out results/whole_body_v1/scoreboard
```

200 serves × seeds 11/2000/4000 = 600 trials, mean action, the honest free-base scene, straight-leg reset — the
exact command form of `results/waist_lock/RESULT.md`. Numbers come from `scoreboard.json`
(`legal_return_rate`, `target_hit_rate`, `landing_error_m`, `contacts`, `falls`; `falls` = `events["fell"]`
latched anywhere in the 3 s, an upper bound). The waist-locked comparison row is re-run in the same table with
`TT_WAIST_LIMIT=0,0` on the plain v3 policy. Scored on the Mac, ~10 min CPU, no credits. **Nothing about the
scored scene changes**: vendor actuators, kp = 500, the joint torque clamps, `FALL_PELVIS_Z = 0.45`.

### 8.2 The bar

| verdict | falls (600) | legal (600) | meaning |
|---|---|---|---|
| **PASS** | ≤ 5.0 % (≤ 30) | ≥ 55.0 % (≥ 330) | beats both honest baselines on both axes with the waist free; recovers at least half the gap to the welded 77.0 % |
| GO for box 2 | ≤ 10.0 % | ≥ 50.0 % | worth another 744 credits |
| FAIL | anything else | | lane stops; write-up with the Isaac held-out fall rate next to the MuJoCo one (the sim-to-sim gap is the first thing to read); credits go to the waist-locked param retrain |

All three seeds must move in the same direction on falls; legal vs the waist-locked row by McNemar exact test
(p < 0.05), as in `results/v3_stance`. Box 1's policy only has to prove the chain (loads, 600 trials without NaN,
falls < 75.7 %) unless it also happens to pass.

Reference rows carried into `results/whole_body_v1/RESULT.md`: welded v3 77.0 / 49.5 / 0.28 m; waist-free v3
61.5 % legal / 75.7 % falls; waist-locked v3 45.3 % / 3.5 % / 0.68 m / 14.8 % falls; the first sanity line is
"hold `STAND_POSE` 3 s in the mirrored Isaac model: 0 falls" (graft), next to MuJoCo's own 0.1° tilt hold.

### 8.3 Secondary diagnostics (graft), from the evaluator's `candidate/trajectory.jsonl`

`himrobot/wbc/diagnostics.py` reports per trial: pelvis-xy error p90 at the contact step (target < 3 cm), tilt
p90, fall time relative to contact, peak |waist roll| and |pitch| actually executed (the waist must be *moving*,
≥ 0.2 rad, for the "stands while swinging hard" claim; ±0.1 rad already topples today's hitter), paddle speed at
contact, contacts (expect the waist-free 78 % to return), and the fell-after-return fraction. Also the ball-free
"stand under library replay" fall rate (`replay_library.py`) as a standing metric that isolates balance from
return quality.

---

## 9. Free local work before any box (the six items that matter)

See the work-item list returned with this design for file paths, specs and verification recipes:

1. `himrobot/wbc/record_library.py` — swing libraries (nominal / wide / slow / eval-600).
2. `IKHitterPolicy.plan()` + `preview()` accessors, parity-tested; `himrobot/wbc/preview_error.py`.
3. `himrobot/wbc/obs.py`, `himrobot/wbc/action.py`, `himrobot/policies/wbc_np.py`, `himrobot/wbc/export_npz.py`
   + parity tests (a = 0 ≡ today's free-base hitter; velocity-frame conventions; numpy-vs-torch obs).
4. `himrobot/wbc/mujoco_mirror.py` (MJCF for conversion + `actuators_mujoco.json` with the torque clamps) and
   `himrobot/wbc/replay_library.py` (the MuJoCo half of Gate 0, and the per-checkpoint sim-to-sim check).
5. `tools/mocap/mocap_leg_prior.py` → `assets/wbc/mocap_leg_prior.npz`.
6. `himrobot/wbc/isaac/` task package + stub-import gate, `tools/box/wbc_box_run.sh`, `box_push_results.py
   --glob`, `docs/runbooks/whole_body_box.md`.

Long local work (library recording ≈ 30 min total, prior building ≈ 10 min) dies if the laptop sleeps.

---

## 10. After v1 — what this design deliberately leaves for v2/v3

- **Stepping (v2, box 3).** The action already owns all 12 leg joints at 0.5 rad and the observation already
  carries pelvis position and heading. Stepping is a *command* change: replace the fixed station with a commanded
  pelvis xy per row (HITTER's `p̂_base`; from the bank, `p̂_base,y = p_hit,y + 0.37 m` clipped to ±0.35 m), switch
  the hitter's base-aware IK on (`TT_BASE_AWARE=1`, measured to fix the kinematics once something balances), and
  use the root-xy velocity prior `mocap_leg_prior.py` already stores. Not a new controller.
- **A WBC that owns the arm (v3).** HITTER's 104-D interface with the IK hitter as planner (`plan()` already
  exists after v1). Needs a racket-target reward and several tuning rounds; only after v1/v2 stand.
- **Real-G1 transfer.** The kp = 500 stiff PD and 5 N·m wrists are the evaluator's, not the real G1's.
  BeyondMimic's compliant table (kp = armature·(2π·10 Hz)², ζ = 2; effort 88/139/50/25/5) is recorded here as
  the actuator model for a later hardware pass — it is **not** swapped into the scored scene for this lane.

---

## 11. Risks

- **PhysX → MuJoCo gap.** Even with kp/kv/effort/armature/friction copied from the compiled model, PhysX's
  implicit PD, contact model and solver differ from MuJoCo `implicitfast` at 0.002 s. Gate 0 and
  `replay_library.py` on every checkpoint are the defence; a policy that stands in Isaac and falls in MuJoCo is a
  plausible outcome of box 1 — then the fallback is mjlab (same MJCF, no conversion) or CPU MuJoCo PPO.
- **MJCF → USD conversion never run** (`PORT.md` §5 item 3): paddle inertia, joint limits or collision meshes may
  not survive; 20 min budgeted with the FK-truth check, URDF fallback named.
- **kp = 500 PD in PhysX** at 0.002 s may chatter or explode; the smoke run is the only source, `0.004 × 5` the
  documented fallback.
- **Unknown image versions** (Isaac Sim / Isaac Lab / rsl_rl not in `machine_caps.txt`); 2.3.x and 3.0 break
  Isaac-Lab-2.1-era code. Gate 1 with a 90-minute abort caps the loss.
- **Unknown s/iteration and VRAM at 4096 envs** — the smoke run answers; the main run is wall-clock capped and
  checkpoints every 500 iterations so a partial policy is resumable on box 2.
- **Preview mismatch** (open-loop roll-out vs closed-loop future) — measured before the box; re-record if
  > 0.05 rad RMS.
- **Waist residual fights the hitter** — bounded at 0.2 rad and penalised; the pass bar demands both falls and
  legal, so a policy that buys standing with returns fails.
- **Reach loss from holding station** — the waist-free hitter's 61.5 % was partly bought by falling into the
  ball; expect legal between 45 % and 65 %, the bar is 55 %.
- **Library validity** — tied to the shipped v3 weights at `TT_ROBOT_Y=-0.10`; any hitter or stance change means
  re-recording (free, minutes).
- **HIM evaluator contract** — the actor uses base linear velocity and world pelvis position; if HIM's runtime
  gives joint state only, `wbc_obs_v1_noimu` must be trained (a second run).
- **`falls` latches anywhere in 3 s**, including after a good return — counted anyway (conservative).
- **Credit exposure** — box 1 is 372 credits with two early-abort gates; a full 4 h that fails the MuJoCo gate
  still costs 372.

## 12. Open questions

1. **Owner:** go on 372 credits (L4, 4 h, `him-robotics`) for box 1, and approval to run `him login` on the box
   for the push + self-stop convention (alternative: rsync-pull with a manual stop)?
2. **HIM (Ethan):** does the scored runtime provide base pose / base velocity to the policy, and is the
   challenge scene free-standing? Decides `wbc_obs_v1` vs `wbc_obs_v1_noimu` before box 2.
3. **Owner/lead:** `leg_default` for box 1 — `stand` (parity start, recommended) or the `mocap` crouch?
4. Which Isaac Sim / Isaac Lab / rsl_rl versions are on the `him-robotics` image (only `pip show` on a launched
   box, or HIM, can answer)?
5. Engine fallback if Gate 0 fails: is porting the same task to mjlab (MuJoCo-Warp, `standard` image)
   acceptable as the box-2 fallback instead of CPU MuJoCo PPO?
6. Strike/swing label: is peak wrist speed within one or two frames of the real stroke on the five previewed
   mocap swings? If not, the leg prior's phase alignment is shifted, not the mechanism.
7. Should the scoreboard for this lane also carry the ball-free "stand under library replay" fall rate as a
   standing metric next to legal/falls?
8. Stepping (v2): plan it as box 3, or fold the commanded-station variant into box 2 if box 1 passes?
