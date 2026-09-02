#!/usr/bin/env bash
# Fetch third-party robot assets (gitignored; not committed).
#
# Booster K1 is the real public robot behind HIM's "Adam" table-tennis fixture
# (BSD-3-Clause, (c) Booster Robotics); its 22-joint layout matches HIM's Adam
# fixture, so we vendor it under third_party/ for the tools/mocap scripts to load.
#
# Unitree G1 (current target robot) is NOT fetched here: the model is vendored
# as tracked files at assets/unitree_g1/ (same g1.xml + meshes as
# google-deepmind/mujoco_menagerie's unitree_g1), so no clone is needed.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p third_party
if [ -d third_party/booster_assets/.git ]; then
  echo "booster_assets already present."
else
  git clone --depth 1 https://github.com/BoosterRobotics/booster_assets third_party/booster_assets
fi

echo
echo "K1 MuJoCo: third_party/booster_assets/robots/K1/K1_22dof.xml"
echo "G1 MuJoCo: assets/unitree_g1/g1.xml  (current target, vendored, no fetch needed)"
echo "Done. Try:  ./.venv/bin/python tools/mocap/g1_retarget.py   (real swing retargeted onto the G1)"
