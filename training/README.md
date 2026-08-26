# Behavior-cloning training

Trains a small PyTorch policy that maps robot joint state → desired joint
targets, and saves a checkpoint that `him_adapter.policy` loads at deploy time.
When no checkpoint (or no torch) is present, the deployment path falls back to
`HoldPolicy`, so the HIM package stays valid and the smoke test stays green.

## ⚠️ Data provenance (important)

The real HIM public table-tennis dataset was **not downloadable** during setup
(the artifact endpoint returns SPA HTML instead of the tar). So the shapes used
here come from the **example** manifests shipped inside `him_sdk`
(`dataset-adam-table-tennis-example.json`, `robot-adam-table-tennis-v0.json`),
which are `example_only` / `fixture_only` tier:

| field | example value | status |
|---|---|---|
| joints | **22** (Booster K1 / Adam) | placeholder — reconfirm |
| observation.state.joints | float32 [T,22] @100Hz rad | placeholder |
| action.joint_targets | float32 [T,22] @100Hz rad | placeholder |
| ball_state | 6-dim | optional input (`--ball`) |
| front camera | uint8 480×640×3 @60Hz | **not used** by this joint-only baseline |

Everything is parameterised in `config.py`. **Do not treat 22 or the key names
as confirmed** — swap in the real values once the dataset is in hand.

## Run

From the repo root, using the torch venv created during setup:

```bash
./.venv/bin/python -m training.train                    # MLP on a synthetic sample
./.venv/bin/python -m training.train --model transformer --context 8 --epochs 30
./.venv/bin/python -m training.train --root data --ball  # real LeRobot-v3 parquet + ball_state
```

With no parquet under `--root`, a **synthetic** sample is generated (a fixed,
mildly non-linear joint→target "expert") so the loop runs end-to-end today; a
trained model should beat the printed *hold baseline* MSE. Reading real parquet
requires `pip install pyarrow pandas` into `.venv`.

The checkpoint (default `models/bc_policy.pt`, git-ignored) is self-contained:
model arch + weights, normalization stats, and the deployment contract signal
names. `him_adapter.BCPolicy` reloads it via `training.model.build_model_from_ckpt`.

## Deployment note

The current `.him` runtime declares `frameworks: [custom-python]` and a
stdlib-only entrypoint, so the hosted worker has **no torch** — it will load
`HoldPolicy`. To actually deploy the trained net you must (a) declare a
pytorch runtime + dependency, and (b) ship the checkpoint as a package artifact.
That is a deliberate, separate change (weights are intentionally git-ignored and
were never committed/uploaded).
