#!/usr/bin/env bash
# Fetch third-party robot assets (gitignored; not committed).
#
# The Booster K1 is the real, public robot behind HIM's "Adam" table-tennis
# fixture (BSD-3-Clause, (c) Booster Robotics). Its 22-joint layout matches
# HIM's fixture. We vendor it under third_party/ so the sim scripts can load it.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p third_party
if [ -d third_party/booster_assets/.git ]; then
  echo "booster_assets already present."
else
  git clone --depth 1 https://github.com/BoosterRobotics/booster_assets third_party/booster_assets
fi

echo
echo "K1 URDF:   third_party/booster_assets/robots/K1/K1_22dof.urdf"
echo "K1 MuJoCo: third_party/booster_assets/robots/K1/K1_22dof.xml"
echo "Done. Try:  ./.venv/bin/python sim/view_k1.py   (renders sim/k1_wave.gif)"
