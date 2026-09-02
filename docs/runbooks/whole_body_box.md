# Runbook: the whole-body-controller (WBC) training box

Design: `docs/design/whole_body_v1.md` (read it first — this runbook only
repeats the numbers needed to actually launch and babysit the box). Nothing
in this runbook authorizes spending credits by itself — it is the plan
`wbc_box_run.sh` executes once the owner says "launch it" per `CLAUDE.md`'s
guardrails.

## 1. The quote

| box | class | hours | credits | ceiling to pass |
|---|---|---|---|---|
| 1 — chain proof + smoke + short train + on-box score | `gpu-l4-workspace` | 4 | **372** | `--max-credits 400` |
| 2 — main run (`--resume` from box 1) | `gpu-l4-workspace` | 8 | **744** | `--max-credits 800` |

**First result: 372 credits.** Box 2 (744 more, 1,116 total) is launched only
if box 1's own scored policy does not already clear the bar (§4 below), or
per the design doc's "GO for box 2" tier. Image `him-robotics` (CUDA + Isaac
Sim + Isaac Lab + MuJoCo + GMR — exact versions unconfirmed, Gate 1 below
checks them in the first ten minutes). `gpu-l40s-workspace` (2x the
credits/hour for 2x VRAM) is a fallback only if the smoke run shows 4096
envs do not fit in 24 GB.

Launch (only after an explicit "launch it"):

```
export HIM_API_URL=https://arena.himrobotics.com
~/.local/bin/him machine launch --profile gpu-l4-workspace --environment him-robotics \
    --hours 4 --max-credits 400 --yes --json
```

The `workspaceId` in that JSON response is `MACHINE_ID` below.

## 2. Before the box: free local prep (no credits)

These produce the files the box reads; do them on the Mac first and rsync
the results up (`work/wbc/*.npz`, `assets/wbc/*`). Every step runs at the
shipped stance: `export TT_ROBOT_Y=-0.10` first (the libraries, the Mac-side
Gate 0 half and the on-box scoring must all agree; `wbc_box_run.sh` exports
the same value itself).

1. `himrobot/wbc/record_library.py` (WB1) → `work/wbc/{nominal,wide,slow,swing_lib_eval}.npz`.
   **Not in the tree as of 2026-09-02** — until it lands there are no libraries, and the
   box cannot run past Gate 1. The schema it must write is
   `himrobot.wbc.replay_library.LIBRARY_KEYS`.
2. `TT_ROBOT_Y=-0.10 python -m himrobot.wbc.replay_library --lib work/wbc/swing_lib_eval.npz --mode ball-parked`
   (WB4) → `work/wbc/replay_swing_lib_eval_ball-parked.json` — the MuJoCo half of Gate 0
   below. `ball-parked`, not `with-ball`: the Isaac task has no ball, so the two halves
   must simulate the same thing (`MUJOCO_GATE0` overrides the path).
3. `himrobot/wbc/mujoco_mirror.py` (WB4, already committed) →
   `assets/wbc/{g1_29dof_paddle_free.xml,actuators_mujoco.json}` — regenerate only if
   `assets/unitree_g1/g1.xml` or `himrobot.sim.scene`'s paddle/pose constants changed.
4. `tools/mocap/mocap_leg_prior.py` (WB5, already committed) → `assets/wbc/mocap_leg_prior.npz`.
5. `.venv/bin/python -m pytest tests/test_wbc_isaac_stub.py tests/test_wbc_mdp_torch.py tests/test_box_push_glob.py`
   and `DRY=1 MACHINE_ID=ssh_test HOURS=4 bash tools/box/wbc_box_run.sh` — the Mac-side gate.

`tools/make_refs_tarball.sh`-style packing is optional; a plain
`rsync -avz work/wbc assets/wbc himrobot tools tests <box>:/workspace/himrobot/`
is enough.

## 3. The command

```
MACHINE_ID=ssh_xxx HOURS=4 nohup bash tools/box/wbc_box_run.sh \
    > /workspace/wbc_box.log 2>&1 &
```

Every knob is an env var with a default — `bash tools/box/wbc_box_run.sh`
prints them via its own header comment; the ones worth setting explicitly:

| var | default | meaning |
|---|---|---|
| `HOURS` | 4 | must match the box's own `--hours` |
| `LEG_DEFAULT` | `stand` | `stand` (parity anchor, recommended for box 1) or `mocap` |
| `RUN_NAME` | `wbc_v1_a` | rsl_rl experiment name, also the log/checkpoint subdirectory |
| `NUM_ENVS` | 4096 | if the smoke run fails (OOM), retried once at half, automatically |
| `CURRICULUM` | `default` | pass `off` to freeze the library mix at iteration 0 (sec 6.1's schedule advances once per PPO update otherwise) |
| `TT_ROBOT_Y` | `-0.10` | the stance every library was recorded at and every reference row is scored at |
| `RESUME` | (empty) | box 2: a `model_<it>.pt` to `--resume` from |
| `MUJOCO_GATE0` | `work/wbc/replay_swing_lib_eval_ball-parked.json` | the Mac-side Gate 0 half |

`DRY=1 MACHINE_ID=ssh_test HOURS=4 bash tools/box/wbc_box_run.sh` prints the
exact command sequence below with nothing executed — read it before the
first real launch.

## 4. The sequence and its gates

1. **venv probe.** If `/workspace/venv/bin/python` doesn't exist, look for an
   interpreter that already imports `isaaclab` (`/isaac-sim/python.sh` and
   similar) and symlink it in. Every later step uses `$PY =
   /workspace/venv/bin/python`, never bare `python`.
2. **Gate 1 — version check.** `pip show isaaclab isaacsim rsl-rl-lib`; pass
   if `isaaclab` reports `2.1.x`/`2.2.x` (docs/design/whole_body_v1.md §2, §11 —
   `2.3.x`/`3.0` need patches this repo doesn't carry). Fail →
   `tools/box/gpu_box_setup.sh` steps 1–5 with a **hard 90-minute abort**
   (`GATE1_ABORT_MIN`) that writes `STATUS.md`, pushes, and stops the box —
   worst case ≈1.7 h ≈160 credits lost, never the full 4 h.
3. **MJCF → USD conversion + FK-truth check.**
   `himrobot.wbc.isaac.scripts.convert_mjcf` (the installed
   `isaaclab.sim.converters.MjcfConverter` API — Isaac Lab's own
   `scripts/tools/convert_mjcf.py` only exists in a repo checkout, whose
   location on the image is unknown) on `assets/wbc/g1_29dof_paddle_free.xml`,
   then `himrobot.wbc.isaac.scripts.verify_usd_parity` builds the real task env
   with one robot, forces `STAND_POSE` at the nominal pelvis pose and diffs the
   paddle-blade centre/normal and both feet against
   `assets/wbc/actuators_mujoco.json`'s `fk_truth` block (5 mm, blade-normal
   dot ≥ 0.999), checks the 29 USD joint names cover `scene.JOINT_ORDER`, and
   runs 10 zero-action steps (finite 141/176-dim observations, no
   termination) → `usd_parity.json`. A mismatch OR an exception inside the
   Isaac-side code stops the box (exit 2 / 3) — never a silent pass. **This
   conversion has never been run** — docs/design/whole_body_v1.md §11 names
   it the first real risk of any box session.
4. **Gate 0 — two-sided replay.** `a = 0` on the held-out 600-row
   `swing_lib_eval.npz`, once in Isaac
   (`himrobot.wbc.isaac.scripts.replay_zero_action`: row *i* in env *i* from
   step 0, exact nominal resets, no randomisation, falls read off the
   termination manager's `fell` term at `scene.FALL_PELVIS_Z`) and once
   already on the Mac (`himrobot.wbc.replay_library --mode ball-parked`,
   step 2 above). Pass = idle (`STAND_POSE`-only) rows show 0% falls in
   Isaac, and the library rows' fall rate is within **±15 points** of the
   MuJoCo number. Fail → stop the box (an actuator/asset problem to fix
   locally, ≈70 credits spent).
5. **200-iteration smoke** at `NUM_ENVS` (one automatic retry at half on
   failure) → `smoke_timing.json` (s/iteration). The main run's
   `--max_iterations` is sized from this number and the lifetime **actually
   left when training starts** minus `STOP_BEFORE_MIN` — never guessed, and
   never from a fixed setup allowance.
6. **Timed training** (`--max_minutes` = that budget, enforced inside
   `train.py` by a SIGALRM → `KeyboardInterrupt` → `final.pt`, and by the
   script's own `SIGINT` at the same deadline, polled every minute);
   checkpoints every 500 iterations regardless; pushes every 30 min.
7. **Export + score** the last 3 checkpoints
   (`himrobot.wbc.isaac.scripts.export` → `<run>/exported/model_<it>.npz` →
   `TT_ROBOT_Y=-0.10 TT_WBC_NPZ=… himrobot.eval.eval_seeds --free-base
   --params '{"robot_table_collision": false}'`, the honest scene, 600
   serves); pick by legal × stand, where stand = 100 − falls % and falls %
   is the fall **count** over 600 (the `scoreboard.json` `falls` field is a
   count per seed, not a rate); write `RESULT.md`.
8. **Final push, `him machine stop`.** Any other exit — a failed gate, a
   crash — takes the same path through the script's `EXIT` trap: `STATUS.md`
   with the failing step, a push, `him machine stop`.

If box 1's scored policy already meets the pass bar (below), box 2 is not
launched — say so and stop, per `CLAUDE.md`'s "quote cost first" guardrail.

## 5. The bar (unchanged from the design doc — repeated here for convenience)

600 serves (seeds 11/2000/4000 × 200), honest free-base scene, mean action:

| verdict | falls | legal | meaning |
|---|---|---|---|
| **PASS** | ≤ 5.0 % | ≥ 55.0 % | beats both honest baselines (waist-free 61.5 %/75.7 % falls, waist-locked 45.3 %/14.8 % falls) at once |
| GO for box 2 | ≤ 10.0 % | ≥ 50.0 % | worth the next 744 credits |
| FAIL | anything else | | write up the Isaac-vs-MuJoCo gap first; stop the lane |

## 6. What to read in the pushed dataset

Every push (`tools/box/box_push_results.py --root logs/rsl_rl --glob ...`,
dataset `himrobot-rl-checkpoints`, a new version each call) carries:

- `wbc_v1/<run>/model_*.pt`, `final.pt` — rsl_rl checkpoints, every 500 iterations.
- `wbc_v1/<run>/exported/*.npz` — the last 3 checkpoints, converted to the
  torch-free runtime format (`himrobot.wbc.export_npz`).
- `wbc_v1/<run>/params/*.yaml` — the resolved env/agent config for that run;
  `wbc_v1/<run>/events*` — the TensorBoard log.
- `wbc_v1/versions.txt` — Gate 1's `pip show` output (the image's real versions).
- `wbc_v1/usd_parity.json` — the FK-truth check, with the USD's own joint/body order.
- `wbc_v1/smoke_timing.json` — s/iteration and the sizing decision.
- `wbc_v1/gate0_isaac.json` (and, on the Mac, `replay_swing_lib_eval_ball-parked.json`)
  — the Gate 0 numbers both sides actually compared.
- `wbc_v1/scoreboard_<checkpoint>/scoreboard.{md,json}` — the on-box
  `eval_seeds` scores per exported checkpoint (this is what "pick by legal ×
  stand" reads); `wbc_v1/scores.txt` — the legal × stand table.
- `wbc_v1/RESULT.md` — the run summary, in the same shape as
  `results/waist_lock/RESULT.md`, `results/balance_v1*/RESULT.md`.
- `wbc_v1/STATUS.md` — present whenever the script exited early (which step,
  exit code, lifetime left).

Pull with `him dataset pull himrobot-rl-checkpoints@<version>` (or browse via
`him dataset list`/`him dataset show`), then re-score any exported `.npz`
locally with `himrobot.eval.evaluate --policy
himrobot.policies.wbc_np:NumpyWBCPolicy --free-base` before trusting a
number from a box.

## 7. The token / self-stop decision

**Recommended (what `wbc_box_run.sh` implements): HIM-push + self-stop**,
the same convention as every CPU box so far
(`tools/box/box_param_sweep.sh`) — `him login` on the box, pushes on a
30-minute timer via `tools/box/box_push_results.py --glob`, `him machine
stop $MACHINE_ID --yes` at the end. This needs the owner's explicit approval
to copy the HIM token to the box (`CLAUDE.md`'s guardrail), **and a reminder
to rotate that token once the box stops.**

The alternative, `work/pace/g1_port/isaac_box_run.sh`'s rsync-pull design
(no token on the box, a background `snapshot.sh` mirrors results, the Mac
pulls, and a human stops the box manually with no automatic safety net), was
not chosen here: it trades the token-on-box risk for an unattended box that
cannot stop itself, which is worse for a 4–8 hour unattended run. Revisit
this choice if the owner's risk preference says otherwise — it is a one-line
change (swap the `push()`/`stop_machine()` calls in `wbc_box_run.sh` for a
`snapshot.sh`-style rsync target and drop the token entirely).

## 8. What the task package does NOT yet implement (design gaps, verified 2026-09-02)

Read `himrobot/wbc/isaac/env_cfg.py`'s module docstring for the live list. As of
this runbook: the 25 % "policy's own post-swing states" resets (the ring buffer
exists, nothing captures/restores physical state into it — that share folds into
pre-swing resets); the torso-CoM / paddle-mass / armature / actuator-delay
randomisation of §6.2 (friction, link-mass, PD-gain scale, pushes and reset
pose/joint noise are in). Neither blocks box 1; both are box-2 items.

## 9. Open questions this runbook does not answer

Carried from docs/design/whole_body_v1.md §12, unchanged:

1. Owner: go on 372 credits for box 1, and approval for `him login` on the
   box (§7 above)?
2. HIM (Ethan): does the scored runtime provide base pose/velocity, and is
   the challenge scene free-standing? Decides `wbc_obs_v1` vs
   `wbc_obs_v1_noimu` before box 2.
3. Owner/lead: `LEG_DEFAULT` for box 1 — `stand` (recommended) or `mocap`?
4. Which Isaac Sim/Isaac Lab/rsl_rl versions are actually on the
   `him-robotics` image — only Gate 1 on a real box answers this.
5. Engine fallback if Gate 0 fails: is porting to `mjlab` (no MJCF→USD
   conversion needed) acceptable as the box-2 fallback?
