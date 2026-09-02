# Native command references

The deployment entrypoint is the command `python -m him_adapter.server`. This
command uses stdio. HoldPolicy is the default policy. If the file
`models/bc_policy.pt` is available, the system uses BCPolicy. If not, the
system uses HoldPolicy.

The retargeting pipeline uses the file `.him/contracts/training-pipeline.json`.
All stages run on the local CPU.

| stage | command | needs |
|---|---|---|
| convert_fbx | `tools/mocap/fbx_to_bvh.py` | Blender (headless) |
| validate_motion | `tools/mocap/validate_motion.py` | stdlib |
| rename_bones | `tools/mocap/rokoko_to_lafan1_bvh.py` | stdlib |
| retarget_g1 | `tools/mocap/gmr_bvh_to_g1_headless.py --ik_config tools/mocap/ik_configs/bvh_lafan1_rokoko_to_g1.json --jobs 4` | GMR venv (`gmr_python` parameter). The repo-local IK config corrects the rotation. The upstream GMR offsets are incorrect for Rokoko/Blender bone frames. |
| render_preview | `tools/mocap/play_g1_csv.py --video` | MuJoCo venv + `assets/unitree_g1` (`mujoco_python`) |

To run the pipeline, use this command: `him pipeline run .him/contracts/training-pipeline.json --yes --set gmr_python=... --set mujoco_python=...`

`cache.key_fields` for each stage gives information only. The system does not
use it to make the cache key. The local runner (`him_sdk/local_pipeline.py`)
makes the cache key. It uses the resolved command of the stage, the
parameters of the stage, and the `output_digest` value of each dependency.

The `output_digest` value is a hash of the list of specified outputs of each
dependency. Thus, the JSON reports that follow show a hash for each clip. You
can use these hashes to find what changed.

The `train` entrypoint runs `python -m tt_sim.rl.ppo --residual
--residual-scale 0.2 --vel-penalty 1e-5 --legal-bonus 10 --n-envs 64
--minutes 420 --eval-every-steps 100000`. This command does not pass `--out`.
`tt_sim/rl/ppo.py` writes checkpoints (`best.pt`, `final.pt`) and
`train_log.csv` to `$HIM_CANDIDATE_OUTPUT` when HIM sets it, else to
`work/tt/rl/run`. `tt_sim.rl.ppo.pick_device` uses MPS if available, else
CPU; an NVIDIA runner is not necessary for this entrypoint.

These items are not configured: export.
