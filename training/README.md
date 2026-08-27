# Behavior-cloning training

The trainer trains a small PyTorch policy. This policy calculates desired
joint positions from the robot joint state. The trainer saves a checkpoint.
`him_adapter.policy` loads this checkpoint at deploy time.

If no checkpoint or no torch package is present, the deployment path uses
`HoldPolicy` as an alternative. This keeps the HIM package valid, and the
smoke test stays green.

## Data provenance (read this first)

There is **no robot data** in this project. HIM's capture is human data:
paddle IMU, Orbbec RGB-D, and Rokoko mocap. The value of `robot_version_id`
is null for this capture. By default, the trainer uses the **movement
track's output**. This output is Rokoko mocap retargeted onto the Unitree G1
with GMR (`sim/gmr_bvh_to_g1_headless.py`). These are *kinematic* poses, not
measured robot states or commanded targets.

| field | value | where it comes from | status |
|---|---|---|---|
| joints | **29** (Unitree G1) | `.him/contracts/runtime.json` `observations[0].shape`, `extensions.robot.joint_order` | given in our contract |
| joint order | Menagerie `unitree_g1` actuator order = Unitree SDK2 `G1JointIndex` | This comes from the contract field `extensions.robot.joint_order`. The trainer checks `data/g1/gmr_g1_joint_order.txt` against it at load time. | the same on disk |
| training data | `data/g1/*.csv` → **`work/g1/`**, the corrected retarget — 8 clips, 59,685 frames @ 30 fps; row = root xyz (m), root quat **xyzw**, 29 joint angles (rad) | GMR retarget of the 8 Rokoko FBX clips with the repo-local IK config `sim/ik_configs/bvh_lafan1_rokoko_to_g1.json` (derived rotation offsets, human height 1.75 m). See `docs/PROJECT_KNOWLEDGE.md` §4c. The earlier `~/him-tools/out/all` CSVs (upstream offsets, 50–69 cm wrist error) are the broken baseline. The trainer does not use them for training. | **not robot data** |
| obs | `robot.joint_position` = joints[t] (29) | This does not include the root pose. See below. | — |
| act | `robot.desired_joint_position` = joints[t+1] (29) | the pose 1/30 s later | — |
| deploy signal names / 50 Hz clock | `robot.joint_position` → `robot.desired_joint_position` | our contract (from HIM's single-policy template) | **UNCONFIRMED against HIM's evaluator** — HIM did not give approval for the names, order, or rate |
| ball state | 6-dim, optional (`--ball`) | This has meaning only with a parquet or synthetic source. | missing from all real data here |

**What the BC model actually learns:** a *next-pose predictor* for
retargeted human motion, a motion prior ("given this pose, where does a
human-like G1 body go 33 ms later"). The BC model is **not** a table-tennis
controller. It never sees the ball, and the data are 30 fps kinematics, not
the contract's 50 Hz rate. Nobody executed the actions as commands on a
robot. This model shows that the train → checkpoint → `him_adapter` path
works end-to-end on the real joint layout. The real control policy is the
PPO-in-sim and ball-predictor plan in `docs/PROJECT_KNOWLEDGE.md` §7.

Observation design choice: by default, **root position and orientation are
not included in obs**. The runtime contract gives only
`robot.joint_position`, with no base pose signal. The root position is
necessary for some policies. This contract could not run such a policy.
Also, the root xy value is world-frame mocap placement. A proprioceptive
policy cannot use it.

`--csv-root-obs` adds root height z and quat xyzw (+5 dims), for offline
experiments only. At deploy time, the shape check in `BCPolicy.infer`
rejects this checkpoint. The system uses the hold policy as an alternative.

Model note: the heads are **residual** (`output = current joints + net(x)`,
zero-initialized, in normalized space). So, training starts exactly at the
hold policy. Without this design, a plain MLP had ~12× the error of hold
after 3 epochs on this data. With `--no-residual`, the trainer uses the
plain head as an alternative. For the "exact hold" identity, the obs and
action normalization stats must agree on the same joint columns.
`training.dataset.compute_norm_stats` makes sure of this (see
`training/tests/test_residual_hold.py`).

The default value of `--context` for `--model transformer` is 8. You can try
`--model transformer` with no `--context` flag as a first test. It must not
crash, and it must not give a bad result without a warning.

`--context 1` also works. With `--context 1`, the window becomes a
single-step attention window, with no history. This is the same as the MLP
model. `mlp` always trains and serves at context 1. The value of `--context`
does not change this.

If `--joints N` is different from the runtime contract's `joint_order`
length (29), the trainer prints a `[warn]` message. The checkpoint still
trains and saves without error. But its `joint_order` field is only for
information. In that case, this field will not give the correct label for
the `act_dim=N` outputs.

## Data sources (auto-selected from `--root`)

1. LeRobot-v3 parquet episodes (`**/*.parquet`). The `pyarrow` package is necessary (`pip install pyarrow`). This source has priority.
2. Retargeted G1 motion CSVs (`*.csv`, no header, 36 columns). Each file is one episode. Windowed (transformer) models keep the episode boundaries.
3. A synthetic sample, used when the root has neither of the above. This is a fixed linear "expert". It lets the loop run anywhere.

## Run

From the repo root, you use `.venv` (torch 2.13, Python 3.12):

```bash
./.venv/bin/python -m training.train --root data/g1 --epochs 15 --val-split episode   # -> models/bc_policy.pt
./.venv/bin/python -m training.train --root data/g1 --model transformer --context 8 --epochs 8 --out work/train/bc_transformer_c8.pt
./.venv/bin/python -m training.train --root /nonexistent --epochs 5 --out work/train/synthetic.pt   # synthetic fallback
./.venv/bin/python -m pytest -q .him/tests training/tests
```

`--val-split episode` puts whole clips only in the validation set. With 8
clips and `val_frac 0.1`, this is 1 clip. The default `random` split puts
adjacent frames on both sides. This makes the result look better than it is
for next-pose data.

The printed *hold baseline* is the MSE of "desired = current" on the same
validation set. This MSE uses the same normalized action space. A useful
model must be below it.

This is the result of the run above, done on 2026-08-26 on the **corrected
(rotation-fixed) retarget**. `data/g1` is a symlink to `../work/g1`, made
with `sim/ik_configs/bvh_lafan1_rokoko_to_g1.json`. See the log at
`work/train/retrain.log`. The run used Apple MPS, 35.2 s wall time for 15
epochs, an MLP 256-256, learning rate 1e-3, batch 256, and seed 0. The
training set had 59,677 (obs, act) pairs over 8 episodes. The validation
clip had 5,787 frames, and the training clips had 53,890 frames:

| | val MSE (normalized) |
|---|---|
| hold baseline (episode split) | 0.02298 |
| BC MLP, best epoch (epoch 2) | **0.02066** (the deployed `models/bc_policy.pt`) |
| BC MLP, epoch 15 | 0.02603 (overfits after epoch ~3. The checkpoint keeps the best-val epoch.) |
| BC transformer, context 8, 8 epochs, episode split (42.6 s) | **0.00502** (`work/train/bc_transformer_c8_episode.pt`, not the deployed file) |
| BC transformer, context 8, 8 epochs, random split (48.7 s) | 0.00313 against its own hold baseline of 0.01639 (`work/train/bc_transformer_c8.pt`. This split is not comparable to the rows above.) |

The MSEs are in the action space normalized on the training data. So, they
are **not comparable** with the numbers from the earlier run on
`~/him-tools/out/all` (hold 0.00201, MLP 0.00189, transformer 0.00105). That
retarget used the incorrect (upstream) rotation offsets. It is the broken
baseline. Nobody must use it for training or as a reference. Every number
here is from `work/train/retrain.log`.

You cannot see the velocity in a single 30 fps frame. So, a
feedforward next-pose model can only be a little better than "hold". A
windowed model (`--model transformer --context 8`) sees velocity. It
decreases the hold error by approximately 4.6× (0.00502 versus 0.02298,
episode split).

`BCPolicy` can use windowed checkpoints. It keeps the last `context`
observations. You can deploy this checkpoint as an alternative: make a copy
of it in `models/bc_policy.pt`.

The checkpoint (`models/bc_policy.pt`, git-ignored, ~330 kB) has everything
in one file. It has the model arch and weights, normalization stats, the
29-name joint order, deploy signal names, a provenance string, and `extra`
(source, split, hold baseline, best epoch). `him_adapter.BCPolicy` loads it
again with `training.model.build_model_from_ckpt`. A check with
`.him/fixtures/observation.json` shows 29 finite values. The system uses
`HoldPolicy` when the checkpoint is missing.

## Deployment note

The `.him` runtime gives `frameworks: [custom-python, mujoco]`. It also
gives a stdlib entrypoint (`python -m him_adapter.server`) and
`artifacts: []`. So, the hosted worker has **no torch and no checkpoint**.
It loads `HoldPolicy`.

To deploy the trained net, you must do two things. First, add a pytorch
runtime and a dependency to `.him/him.yaml`. Second, add
`models/bc_policy.pt` as a package artifact. The weights are git-ignored.
Nobody ever committed or uploaded them.

This is a deliberate, separate change. HIM's approval of the contract's
signal names and rate is necessary first.
