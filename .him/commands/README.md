# Native command references

Deployment entrypoint: `python -m him_adapter.server` (stdio; HoldPolicy fallback,
BCPolicy when `models/bc_policy.pt` exists).

Retargeting pipeline (`.him/contracts/training-pipeline.json`, all local CPU):

| stage | command | needs |
|---|---|---|
| convert_fbx | `sim/fbx_to_bvh.py` | Blender (headless) |
| validate_motion | `sim/validate_motion.py` | stdlib |
| rename_bones | `sim/rokoko_to_lafan1_bvh.py` | stdlib |
| retarget_g1 | `sim/gmr_bvh_to_g1_headless.py --ik_config sim/ik_configs/bvh_lafan1_rokoko_to_g1.json --jobs 4` | GMR venv (`gmr_python` parameter); the repo-local IK config is the rotation-fixed one (upstream GMR offsets are wrong for Rokoko/Blender bone frames) |
| render_preview | `sim/play_g1_csv.py --video` | MuJoCo venv + `third_party/mujoco_menagerie` (`mujoco_python`) |

Run: `him pipeline run .him/contracts/training-pipeline.json --yes --set gmr_python=... --set mujoco_python=...`

Not configured: policy training (needs an NVIDIA runner), export, evaluation.
