# Native command references

The deployment entrypoint is the command `python -m him_adapter.server`. This
command uses stdio. HoldPolicy is the default policy. If the file
`models/bc_policy.pt` is available, the system uses BCPolicy. If not, the
system uses HoldPolicy.

The retargeting pipeline uses the file `.him/contracts/training-pipeline.json`.
All stages run on the local CPU.

| stage | command | needs |
|---|---|---|
| convert_fbx | `sim/fbx_to_bvh.py` | Blender (headless) |
| validate_motion | `sim/validate_motion.py` | stdlib |
| rename_bones | `sim/rokoko_to_lafan1_bvh.py` | stdlib |
| retarget_g1 | `sim/gmr_bvh_to_g1_headless.py --ik_config sim/ik_configs/bvh_lafan1_rokoko_to_g1.json --jobs 4` | GMR venv (`gmr_python` parameter). The repo-local IK config corrects the rotation. The upstream GMR offsets are incorrect for Rokoko/Blender bone frames. |
| render_preview | `sim/play_g1_csv.py --video` | MuJoCo venv + `third_party/mujoco_menagerie` (`mujoco_python`) |

To run the pipeline, use this command: `him pipeline run .him/contracts/training-pipeline.json --yes --set gmr_python=... --set mujoco_python=...`

`cache.key_fields` for each stage gives information only. The system does not
use it to make the cache key. The local runner (`him_sdk/local_pipeline.py`)
makes the cache key. It uses the resolved command of the stage, the
parameters of the stage, and the `output_digest` value of each dependency.

The `output_digest` value is a hash of the list of specified outputs of each
dependency. Thus, the JSON reports that follow show a hash for each clip. You
can use these hashes to find what changed.

These items are not configured: policy training, export, and evaluation. An
NVIDIA runner is necessary for policy training.
