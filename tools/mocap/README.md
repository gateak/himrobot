# Motion-capture, retargeting & perception tools

Local, un-gated MuJoCo/CV/Blender scripts — no HIM servers involved. Not part of the
`himrobot` package; nothing under `himrobot/` imports this directory.

- **Retargeting**: `fbx_to_bvh.py`, `fbx_extract.py`, `rokoko_to_lafan1_bvh.py`,
  `gmr_bvh_to_g1_headless.py` (+ `ik_configs/`), `g1_retarget.py`, `retarget_imu.py`,
  `retarget_report.py`, `validate_motion.py` — Rokoko human mocap (FBX) -> BVH -> LAFAN1
  bone names -> GMR retarget onto the Unitree G1 / Booster K1.
- **Playback**: `g1_model.py`, `k1_model.py`, `play_g1_csv.py`, `view_k1.py`,
  `composite_g1_video.py` — load the robot models and preview/render clips in MuJoCo.
- **Perception**: `ball_detect.py`, `ball_3d.py`, `aruco_scan.py`, `estimate_intrinsics.py`,
  `orbbec_video.py` — orange-ball detection, aligned-depth lookup, table ArUco corners,
  colour-camera intrinsics self-calibration, and session-to-MP4 review, on Orbbec RGB-D captures.
- **Alignment**: `align_streams.py` — pairs IMU episodes, Orbbec sessions, and Rokoko
  clips and cross-correlates paddle gyro against BVH hand speed.

These scripts run under external venvs, not `.venv`: `~/him-tools/gmr-venv` (GMR, torch,
mujoco, mink) for the GMR retarget step; `~/him-tools/mujoco-venv` (mujoco, imageio, scipy,
matplotlib) for `align_streams.py` and `retarget_report.py`; `~/him-tools/cv-venv` (numpy,
opencv-contrib, imageio, pillow) for `ball_detect.py`, `ball_3d.py`, `aruco_scan.py`, and
`orbbec_video.py`. See `docs/PROJECT_KNOWLEDGE.md` §3 (venv contents) and §4c (pipeline
stages) for the exact versions and commands.
