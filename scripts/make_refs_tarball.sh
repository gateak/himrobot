#!/usr/bin/env bash
#
# make_refs_tarball.sh — packs reference data (retargeted G1 motion, IK
# config, runtime contract, paddle-IMU, ball 3D tracks) into one tarball to
# scp onto a GPU box, for scripts/gpu_box_setup.sh (step 7) / Stage 1-2 of
# docs/TRAINING_PLAN.md. Idempotent: rebuilds the staging dir and tarball
# from scratch each run.
#
# Usage:  bash scripts/make_refs_tarball.sh
# Output: work/pace/himrobot-refs.tgz
#         work/pace/himrobot-refs.manifest.sha256 (sha256 per packed file, also inside the tarball)

set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="work/pace"
STAGE_DIR="$(mktemp -d)"
TARBALL="${OUT_DIR}/himrobot-refs.tgz"
MANIFEST="himrobot-refs.manifest.sha256"

cleanup() { rm -rf "$STAGE_DIR"; }
trap cleanup EXIT

mkdir -p "$OUT_DIR"
mkdir -p "$STAGE_DIR/work/g1" "$STAGE_DIR/sim/ik_configs" "$STAGE_DIR/.him/contracts" \
  "$STAGE_DIR/data/him-table-tennis-paddle-imu-20260815/normalized" "$STAGE_DIR/work/perception"

copy_required() {
  # $1 = source path, $2 = destination path (real-file copy, no symlinks)
  if [ ! -e "$1" ]; then
    echo "ERROR: required source file missing: $1" >&2
    exit 1
  fi
  cp -L "$1" "$2"
}

echo "--- staging work/g1/*.csv + gmr_g1_joint_order.txt ---"
shopt -s nullglob
g1_csvs=(work/g1/*.csv)
shopt -u nullglob
if [ "${#g1_csvs[@]}" -eq 0 ]; then
  echo "ERROR: no CSVs found matching work/g1/*.csv" >&2
  exit 1
fi
for f in "${g1_csvs[@]}"; do
  copy_required "$f" "$STAGE_DIR/work/g1/$(basename "$f")"
done
copy_required "work/g1/gmr_g1_joint_order.txt" "$STAGE_DIR/work/g1/gmr_g1_joint_order.txt"

echo "--- staging sim/ik_configs/ ---"
cp -RL sim/ik_configs/. "$STAGE_DIR/sim/ik_configs/"
find "$STAGE_DIR/sim/ik_configs" -name '__pycache__' -type d -prune -exec rm -rf {} +

echo "--- staging .him/contracts/runtime.json ---"
copy_required ".him/contracts/runtime.json" "$STAGE_DIR/.him/contracts/runtime.json"

echo "--- staging paddle-IMU normalized CSV ---"
copy_required \
  "data/him-table-tennis-paddle-imu-20260815/normalized/paddle-imu-samples.csv" \
  "$STAGE_DIR/data/him-table-tennis-paddle-imu-20260815/normalized/paddle-imu-samples.csv"

echo "--- staging work/perception/ball3d_*.csv ---"
shopt -s nullglob
ball3d_csvs=(work/perception/ball3d_*.csv)
shopt -u nullglob
if [ "${#ball3d_csvs[@]}" -eq 0 ]; then
  echo "ERROR: no files found matching work/perception/ball3d_*.csv" >&2
  exit 1
fi
for f in "${ball3d_csvs[@]}"; do
  copy_required "$f" "$STAGE_DIR/work/perception/$(basename "$f")"
done

echo "--- computing sha256 manifest ---"
( cd "$STAGE_DIR" && find . -type f ! -name "$MANIFEST" | sort | xargs shasum -a 256 ) > "$STAGE_DIR/$MANIFEST"

echo "--- packing ${TARBALL} ---"
tar -czf "$TARBALL" -C "$STAGE_DIR" .
cp "$STAGE_DIR/$MANIFEST" "${OUT_DIR}/${MANIFEST}"

echo "--- contents ---"
tar -tzf "$TARBALL" | sort

echo "--- manifest (also inside the tarball as ./${MANIFEST}) ---"
cat "${OUT_DIR}/${MANIFEST}"

echo "--- size ---"
ls -la "$TARBALL"
du -h "$TARBALL"

echo "Done: ${TARBALL}"
echo "Upload with:  scp ${TARBALL} <user>@<gpu-box>:~/himrobot-refs.tgz"
