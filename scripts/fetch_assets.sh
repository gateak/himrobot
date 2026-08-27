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

# Unitree G1 (current target robot): 29-DOF humanoid with a 7-DOF arm + wrist.
# MuJoCo model from google-deepmind/mujoco_menagerie (sparse checkout).
#
# Clone + sparse-checkout is two commands; check for the actual asset (not
# just .git) and clone into a scratch dir first, so an interruption between
# them (Ctrl-C, network drop, disk full) can never leave a .git-only
# directory that this script would then treat as "already present" forever.
if [ -f third_party/mujoco_menagerie/unitree_g1/g1.xml ]; then
  echo "mujoco_menagerie already present."
else
  tmp_menagerie="$(mktemp -d third_party/.mujoco_menagerie.XXXXXX)"
  trap 'rm -rf "$tmp_menagerie"' EXIT
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie "$tmp_menagerie"
  git -C "$tmp_menagerie" sparse-checkout set unitree_g1
  if [ ! -f "$tmp_menagerie/unitree_g1/g1.xml" ]; then
    echo "fetch_assets.sh: clone of mujoco_menagerie did not produce unitree_g1/g1.xml" >&2
    exit 1
  fi
  rm -rf third_party/mujoco_menagerie   # replace any stale/partial directory from a prior interruption
  mv "$tmp_menagerie" third_party/mujoco_menagerie
  trap - EXIT
fi

echo
echo "K1 MuJoCo: third_party/booster_assets/robots/K1/K1_22dof.xml"
echo "G1 MuJoCo: third_party/mujoco_menagerie/unitree_g1/g1.xml  (current target)"
echo "Done. Try:  ./.venv/bin/python sim/g1_retarget.py   (real swing retargeted onto the G1)"
