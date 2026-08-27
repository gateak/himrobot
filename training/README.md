# Behavior-cloning training

Trains a small PyTorch policy that maps robot joint state → desired joint
positions and saves a checkpoint that `him_adapter.policy` loads at deploy time.
When no checkpoint (or no torch) is present, the deployment path falls back to
`HoldPolicy`, so the HIM package stays valid and the smoke test stays green.

## Data provenance (read this first)

There is **no robot data** in this project. HIM's capture is human data
(paddle IMU, Orbbec RGB-D, Rokoko mocap; `robot_version_id: null`). What the
trainer consumes by default is the **movement track's output**: Rokoko mocap
retargeted onto the Unitree G1 with GMR (`sim/gmr_bvh_to_g1_headless.py`),
i.e. *kinematic* poses, not measured robot states or commanded targets.

| field | value | where it comes from | status |
|---|---|---|---|
| joints | **29** (Unitree G1) | `.him/contracts/runtime.json` `observations[0].shape`, `extensions.robot.joint_order` | bound in our contract |
| joint order | Menagerie `unitree_g1` actuator order = Unitree SDK2 `G1JointIndex` | contract `extensions.robot.joint_order`; `data/g1/gmr_g1_joint_order.txt` is checked against it at load time | verified equal on disk |
| training data | `data/g1/*.csv` → **`work/g1/`**, the corrected retarget — 8 clips, 59,685 frames @ 30 fps; row = root xyz (m), root quat **xyzw**, 29 joint angles (rad) | GMR retarget of the 8 Rokoko FBX clips with the repo-local IK config `sim/ik_configs/bvh_lafan1_rokoko_to_g1.json` (derived rotation offsets, human height 1.75 m; `docs/PROJECT_KNOWLEDGE.md` §4c). The earlier `~/him-tools/out/all` CSVs (upstream offsets, 50–69 cm wrist error) are the broken baseline and are no longer the training input | **not robot data** |
| obs | `robot.joint_position` = joints[t] (29) | root pose excluded, see below | — |
| act | `robot.desired_joint_position` = joints[t+1] (29) | the pose 1/30 s later | — |
| deploy signal names / 50 Hz clock | `robot.joint_position` → `robot.desired_joint_position` | our contract (copied from HIM's single-policy template) | **UNCONFIRMED against HIM's evaluator** — HIM has not confirmed names, order, or rate |
| ball state | 6-dim, optional (`--ball`) | only meaningful with a parquet/synthetic source | not present in any real data here |

**What the BC model actually learns:** a *next-pose predictor* on retargeted
human motion — a motion prior ("given this pose, where does a human-like G1
body go 33 ms later"). It is **not** a table-tennis controller: it never sees
the ball, the data are 30 fps kinematics (not the contract's 50 Hz), and the
actions are not commands that were ever executed on a robot. It exists to
prove the train → checkpoint → `him_adapter` path end-to-end on the real joint
layout. The control policy proper is the PPO-in-sim + ball-predictor plan in
`docs/PROJECT_KNOWLEDGE.md` §7.

Observation design choice: **root position/orientation are excluded from obs**
by default. The runtime contract only provides `robot.joint_position` (no
base pose signal), so a policy that needs the root could not be served, and
the root xy is world-frame mocap placement with no meaning to a proprioceptive
policy. `--csv-root-obs` appends root height z + quat xyzw (+5 dims) for
offline experiments only; such a checkpoint is refused at deploy time by the
shape check in `BCPolicy.infer` and falls back to hold.

Model note: the heads are **residual** (`output = current joints + net(x)`,
zero-initialised, in normalized space) so training starts exactly at the hold
policy. Without this a plain MLP was ~12× worse than hold after 3 epochs on
this data. `--no-residual` restores the plain head. The "exact hold" identity
requires the obs and action normalization stats to agree on the shared joint
columns; `training.dataset.compute_norm_stats` enforces that (see
`training/tests/test_residual_hold.py`).

`--model transformer` defaults to `--context 8` (a plain `--model transformer`
with no `--context` is a reasonable first try and must not crash or silently
degrade to something useless); `--context 1` also works, it just collapses
the window to single-step attention with no history, same as the MLP. `mlp`
always trains/serves at context 1 regardless of `--context`.

`--joints N` that disagrees with the runtime contract's `joint_order` length
(29) prints a `[warn]`: the checkpoint still trains and saves fine, but its
`joint_order` field is informational-only metadata and won't label the
`act_dim=N` outputs correctly in that case.

## Data sources (auto-selected from `--root`)

1. LeRobot-v3 parquet episodes (`**/*.parquet`, needs `pip install pyarrow`) — priority.
2. Retargeted G1 motion CSVs (`*.csv`, no header, 36 columns) — one episode per file; episode boundaries are kept for windowed (transformer) models.
3. Synthetic sample when the root has neither — a fixed linear "expert", only so the loop runs anywhere.

## Run

From the repo root, using `.venv` (torch 2.13, Python 3.12):

```bash
./.venv/bin/python -m training.train --root data/g1 --epochs 15 --val-split episode   # -> models/bc_policy.pt
./.venv/bin/python -m training.train --root data/g1 --model transformer --context 8 --epochs 8 --out work/train/bc_transformer_c8.pt
./.venv/bin/python -m training.train --root /nonexistent --epochs 5 --out work/train/synthetic.pt   # synthetic fallback
./.venv/bin/python -m pytest -q .him/tests training/tests
```

`--val-split episode` holds out whole clips (with 8 clips and `val_frac 0.1`
that is 1 clip); the default `random` split puts adjacent frames on both sides
and is optimistic for next-pose data. The printed *hold baseline* is the MSE of
"desired = current" on the same validation set, in the same normalized action
space; a useful model must be below it.

Result of the run above on 2026-08-26 on the **corrected (rotation-fixed)
retarget** — `data/g1` is a symlink to `../work/g1`, produced with
`sim/ik_configs/bvh_lafan1_rokoko_to_g1.json` (log: `work/train/retrain.log`;
Apple MPS, 35.2 s wall for 15 epochs; MLP 256-256, lr 1e-3, batch 256, seed 0;
59,677 (obs, act) pairs over 8 episodes; val clip = 5,787 frames, train = 53,890):

| | val MSE (normalized) |
|---|---|
| hold baseline (episode split) | 0.02298 |
| BC MLP, best epoch (epoch 2) | **0.02066** (the deployed `models/bc_policy.pt`) |
| BC MLP, epoch 15 | 0.02603 (overfits from epoch ~3; the checkpoint keeps the best-val epoch) |
| BC transformer, context 8, 8 epochs, episode split (42.6 s) | **0.00502** (`work/train/bc_transformer_c8_episode.pt`, not the deployed file) |
| BC transformer, context 8, 8 epochs, random split (48.7 s) | 0.00313 against its own hold baseline of 0.01639 (`work/train/bc_transformer_c8.pt`; the random split is optimistic and not comparable to the rows above) |

The MSEs are in the action space normalized on the training data, so they are
**not comparable** with the numbers of the previous run on `~/him-tools/out/all`
(hold 0.00201, MLP 0.00189, transformer 0.00105): that retarget used the wrong
(upstream) rotation offsets, is the broken baseline, and must not be used for
training or as a reference. Every number here is from `work/train/retrain.log`.

Reading: from a single 30 fps frame the velocity is unobservable, so a
feedforward next-pose model can only marginally beat "hold". A windowed model
(`--model transformer --context 8`) sees velocity and cuts the hold error by
about 4.6× (0.00502 vs 0.02298, episode split). `BCPolicy` supports windowed checkpoints (it keeps the last `context`
observations); copy it to `models/bc_policy.pt` to deploy it instead.

The checkpoint (`models/bc_policy.pt`, git-ignored, ~330 kB) is self-contained:
model arch + weights, normalization stats, the 29-name joint order, deploy
signal names, a provenance string, and `extra` (source, split, hold baseline,
best epoch). `him_adapter.BCPolicy` reloads it via
`training.model.build_model_from_ckpt`; verified with
`.him/fixtures/observation.json` → 29 finite values, and `HoldPolicy` when the
checkpoint is absent.

## Deployment note

The `.him` runtime declares `frameworks: [custom-python, mujoco]`, a stdlib
entrypoint (`python -m him_adapter.server`) and `artifacts: []`, so the hosted
worker has **no torch and no checkpoint** — it loads `HoldPolicy`. To deploy the
trained net you must (a) declare a pytorch runtime/dependency in `.him/him.yaml`,
and (b) ship `models/bc_policy.pt` as a package artifact (weights are
git-ignored and have never been committed or uploaded). That is a deliberate,
separate change that also needs the contract's signal names/rate confirmed by
HIM first.
