#!/usr/bin/env bash
#
# gpu_box_setup.sh — bootstraps a fresh Ubuntu 22.04 NVIDIA box for PACE-style
# Isaac Sim / Isaac Lab table-tennis RL training (himrobot G1/K1 port).
#
# Idempotent (each step checks for its target before acting); every version
# below is pinned, nothing floats to "latest". Assumes the NVIDIA driver is
# already installed — verifies via nvidia-smi and aborts with a clear message
# if not (does not install a driver itself).
#
# Pins/commands sourced from, must not contradict:
#   - work/pace/PACE_ANALYSIS.md  §1 (software pins), §8 (smoke cmd + checkpoint math)
#   - work/pace/pace_pins.json    (machine-readable pins, PACE repo commit)
#   - work/pace/BOX_OPTIONS.md    §3 (Isaac Sim 4.5 hardware requirements)
#   - https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html
#     (linked from PACE README.md:51; fetched 2026-08-27 for the venv/pip/isaaclab.sh
#     command sequence used below)
#
# Usage (on the GPU box, as a sudo-capable non-root user):
#   bash scripts/gpu_box_setup.sh
# Env var overrides: TT_HOME, REFS_TARBALL, CUDA_VARIANT (cu121|cu118),
# SMOKE_ITERS, SMOKE_ENVS. See docs/TRAINING_PLAN.md for when/why to run this.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Pins — every version lives here, nothing floats to "latest".
# ---------------------------------------------------------------------------
ISAACSIM_VERSION="4.5.0"                                  # PACE_ANALYSIS.md §1 / pace_pins.json .software_pins.isaac_sim
ISAACLAB_TAG="v2.1.0"                                      # PACE_ANALYSIS.md §1 / pace_pins.json .software_pins.isaac_lab
ISAACLAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
PYTHON_BIN="python3.10"                                     # pace_pins.json .software_pins.python (inherited from the Isaac Lab v2.1.0 pip guide; not an independent PACE pin)
TORCH_VERSION="2.5.1"                                        # Isaac Lab v2.1.0 pip-install page (fetched 2026-08-27), matches pace_pins.json .software_pins.cuda.torch_version_with_cuda
TORCHVISION_VERSION="0.20.1"                                  # same source
CUDA_VARIANT="${CUDA_VARIANT:-cu121}"                          # override CUDA_VARIANT=cu118 for a CUDA-11 box; pace_pins.json lists both CUDA 11 and 12 as supported, torch pin is the same either way
MIN_DRIVER_VERSION="535.129.03"                                 # BOX_OPTIONS.md §3, citing the Isaac Sim 4.5 requirements page
PACE_REPO="https://github.com/purdue-tracelab/PACE-ICRA2026"
PACE_COMMIT="8cab1d59fe96a9cc3b80eba19f109b94fbeb013f"           # pace_pins.json .repo.commit (2026-08-24T16:22:12-04:00)
TT_HOME="${TT_HOME:-$HOME/tt}"                                    # task-specified workspace root
ISAAC_LAB_DIR="$TT_HOME/IsaacLab"
PACE_DIR="$TT_HOME/PACE"
VENV_DIR="$TT_HOME/env_isaaclab"
REFS_TARBALL="${REFS_TARBALL:-$HOME/himrobot-refs.tgz}"           # produced by scripts/make_refs_tarball.sh, scp'd in by the user
REFS_DIR="$TT_HOME/refs"
LOG_FILE="$TT_HOME/setup.log"
SMOKE_ITERS="${SMOKE_ITERS:-200}"                                  # PACE_ANALYSIS.md §8 smoke command
SMOKE_ENVS="${SMOKE_ENVS:-256}"                                     # PACE_ANALYSIS.md §8 smoke command

mkdir -p "$TT_HOME"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== gpu_box_setup.sh started $(date -u +%FT%TZ) ==="
echo "TT_HOME=$TT_HOME  CUDA_VARIANT=$CUDA_VARIANT"

# ---------------------------------------------------------------------------
# 1. GPU / driver check — abort immediately if the driver isn't already usable.
# ---------------------------------------------------------------------------
echo "--- [1/9] Checking NVIDIA driver (nvidia-smi) ---"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found on PATH." >&2
  echo "This script assumes the NVIDIA driver is already installed (it does not" >&2
  echo "install one). Install a driver >= ${MIN_DRIVER_VERSION} (BOX_OPTIONS.md §3)" >&2
  echo "for this box's GPU, then re-run this script." >&2
  exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is on PATH but failed to run — no GPU visible or the" >&2
  echo "driver kernel module isn't loaded. Output:" >&2
  nvidia-smi || true
  exit 1
fi
nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
echo "GPU: ${GPU_NAME}   driver: ${DRIVER_VERSION}"

case "$GPU_NAME" in
  *A100*|*H100*)
    echo "ERROR: ${GPU_NAME} has no RT cores. BOX_OPTIONS.md §3 (citing the Isaac" >&2
    echo "Sim 4.5 requirements page) states A100/H100 are explicitly NOT supported" >&2
    echo "for Isaac Sim's RTX renderer, even headless. Use an RT-core GPU instead" >&2
    echo "(L4, L40S, RTX 4090, RTX 5090 — BOX_OPTIONS.md §3/§6)." >&2
    exit 1
    ;;
esac

if [ "$(printf '%s\n%s\n' "$MIN_DRIVER_VERSION" "$DRIVER_VERSION" | sort -V | head -1)" != "$MIN_DRIVER_VERSION" ]; then
  echo "WARNING: driver ${DRIVER_VERSION} is older than the ${MIN_DRIVER_VERSION} floor" >&2
  echo "documented in BOX_OPTIONS.md §3. nvidia-smi runs, so continuing, but Isaac" >&2
  echo "Sim's first launch may fail — upgrade the driver if step 4/8 below fails." >&2
fi

# ---------------------------------------------------------------------------
# 2. OS check — Isaac Lab v2.1.0's pip-install path recommends Ubuntu 22.04.
# ---------------------------------------------------------------------------
echo "--- [2/9] Checking OS ---"
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  echo "OS: ${PRETTY_NAME:-unknown}"
  if [ "${ID:-}" != "ubuntu" ] || [ "${VERSION_ID:-}" != "22.04" ]; then
    echo "WARNING: pace_pins.json .software_pins.os and the Isaac Lab v2.1.0 pip" >&2
    echo "guide recommend Ubuntu 22.04 (20.04 needs the binary install path" >&2
    echo "instead, GLIBC < 2.34). Detected '${PRETTY_NAME:-unknown}' — continuing" >&2
    echo "anyway, but apt package names/behavior below are only verified for 22.04." >&2
  fi
else
  echo "WARNING: /etc/os-release not found; cannot confirm Ubuntu 22.04." >&2
fi

# ---------------------------------------------------------------------------
# 3. System dependencies.
# ---------------------------------------------------------------------------
echo "--- [3/9] apt dependencies ---"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  git curl ca-certificates \
  cmake build-essential \
  "${PYTHON_BIN}" "${PYTHON_BIN}-venv" "${PYTHON_BIN}-dev"
# (cmake/build-essential per the Isaac Lab v2.1.0 pip guide's "system prerequisites"
# step; apt does not carry per-package version pins for these in that guide, so none
# are pinned here — pinning a number the source doesn't give would be inventing one.)

# ---------------------------------------------------------------------------
# 4. Python venv + PyTorch + Isaac Sim (pip route, per PACE_ANALYSIS.md §1:
#    "installed via the official Isaac Lab v2.1.0 pip guide").
# ---------------------------------------------------------------------------
echo "--- [4/9] venv + torch ${TORCH_VERSION} + isaacsim ${ISAACSIM_VERSION} ---"
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

if ! python -c "import torch" >/dev/null 2>&1; then
  pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_VARIANT}"
else
  echo "torch already installed: $(python -c 'import torch; print(torch.__version__)')"
fi

if ! python -c "import isaacsim" >/dev/null 2>&1; then
  pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" --extra-index-url https://pypi.nvidia.com
else
  echo "isaacsim already installed."
fi
echo "NOTE: the FIRST time any Isaac Sim/Isaac Lab process actually launches the app"
echo "(e.g. the smoke train in step 8 below), it pulls extensions from the registry"
echo "and prompts for EULA acceptance — the Isaac Lab v2.1.0 pip guide says this can"
echo "take upwards of 10 minutes. That is not driven non-interactively here; if step 8"
echo "hangs, attach to the box's console and accept the prompt once."

# ---------------------------------------------------------------------------
# 5. Isaac Lab, pinned to the exact tag PACE was validated against.
# ---------------------------------------------------------------------------
echo "--- [5/9] Isaac Lab ${ISAACLAB_TAG} ---"
if [ ! -d "$ISAAC_LAB_DIR/.git" ]; then
  git clone "$ISAACLAB_REPO" "$ISAAC_LAB_DIR"
fi
git -C "$ISAAC_LAB_DIR" fetch --tags
git -C "$ISAAC_LAB_DIR" checkout "$ISAACLAB_TAG"
ISAACLAB_HEAD=$(git -C "$ISAAC_LAB_DIR" rev-parse HEAD)
echo "IsaacLab checked out at ${ISAACLAB_TAG} (${ISAACLAB_HEAD})"

pushd "$ISAAC_LAB_DIR" >/dev/null
./isaaclab.sh --install
popd >/dev/null

# ---------------------------------------------------------------------------
# 6. PACE, pinned to the exact commit recorded in pace_pins.json.
# ---------------------------------------------------------------------------
echo "--- [6/9] PACE @ ${PACE_COMMIT} ---"
if [ ! -d "$PACE_DIR/.git" ]; then
  git clone "$PACE_REPO" "$PACE_DIR"
fi
git -C "$PACE_DIR" fetch origin "$PACE_COMMIT" || git -C "$PACE_DIR" fetch --all --tags
git -C "$PACE_DIR" checkout "$PACE_COMMIT"
PACE_HEAD=$(git -C "$PACE_DIR" rev-parse HEAD)
if [ "$PACE_HEAD" != "$PACE_COMMIT" ]; then
  echo "ERROR: PACE checked out at ${PACE_HEAD}, expected pinned commit ${PACE_COMMIT}" >&2
  echo "(pace_pins.json .repo.commit). Aborting rather than training against an" >&2
  echo "unrecorded revision." >&2
  exit 1
fi
echo "PACE pinned commit recorded: ${PACE_COMMIT} (matches work/pace/pace_pins.json)"

# README.md:58-67 (per PACE_ANALYSIS.md §1): outer repo first, then the vendored
# rsl_rl fork — order matters, rsl_rl is not on PyPI as this fork.
pip install -e "$PACE_DIR"
pip install -e "$PACE_DIR/rsl_rl"

# ---------------------------------------------------------------------------
# 7. Reference-data workspace: symlink-free copy extracted from a tarball the
#    user uploads separately (scripts/make_refs_tarball.sh builds it locally).
# ---------------------------------------------------------------------------
echo "--- [7/9] Reference data workspace (${REFS_DIR}) ---"
mkdir -p "$REFS_DIR"
if [ ! -f "$REFS_TARBALL" ]; then
  echo "NOTE: ${REFS_TARBALL} not found on this box yet."
  echo "Before Stage 1/2 of docs/TRAINING_PLAN.md, upload it from your workstation:"
  echo "  bash scripts/make_refs_tarball.sh   # builds work/pace/himrobot-refs.tgz locally"
  echo "  scp work/pace/himrobot-refs.tgz <user>@<this-box>:${REFS_TARBALL}"
  echo "It packs work/g1/*.csv + gmr_g1_joint_order.txt, sim/ik_configs/,"
  echo ".him/contracts/runtime.json, the paddle-IMU CSV, and work/perception/ball3d_*.csv"
  echo "with a sha256 manifest (see the tarball's manifest.sha256 to verify after scp)."
  echo "Stage 0 (unmodified PACE T1 smoke, below) does not need this — PACE has no"
  echo "consumer for our data yet (PACE_ANALYSIS.md §7) — so continuing without it."
elif [ -n "$(ls -A "$REFS_DIR" 2>/dev/null)" ]; then
  echo "${REFS_DIR} already populated; skipping extraction (idempotent)."
  echo "Remove ${REFS_DIR} and re-run to force a fresh extract."
else
  tar -xzf "$REFS_TARBALL" -C "$REFS_DIR"
  echo "Extracted ${REFS_TARBALL} -> ${REFS_DIR}"
fi

# ---------------------------------------------------------------------------
# 8. Smoke command (PACE_ANALYSIS.md §8): short, headless, a few iterations.
# ---------------------------------------------------------------------------
echo "--- [8/9] Smoke-train: --num_envs=${SMOKE_ENVS} --max_iterations=${SMOKE_ITERS} ---"
cd "$PACE_DIR"
python -m legged_lab.scripts.train \
  --task=t1_tt \
  --num_envs="${SMOKE_ENVS}" \
  --max_iterations="${SMOKE_ITERS}" \
  --headless \
  --logger=tensorboard

SMOKE_RUN_DIR=$(find "$PACE_DIR/logs/t1_table_tennis" -maxdepth 1 -mindepth 1 -type d | sort | tail -1)
echo "Smoke run directory: ${SMOKE_RUN_DIR:-<not found — check logs above for errors>}"
if [ -n "${SMOKE_RUN_DIR:-}" ]; then
  # PACE_ANALYSIS.md §8: save_interval=250 only ever fires at it=0 for a run
  # shorter than 250 iterations; the only other checkpoint is the unconditional
  # save after the loop exits, model_{last_it}.pt where last_it = SMOKE_ITERS-1.
  LAST_IT=$((SMOKE_ITERS - 1))
  echo "Expected checkpoints: ${SMOKE_RUN_DIR}/model_0.pt and ${SMOKE_RUN_DIR}/model_${LAST_IT}.pt"
  echo "(do not expect intermediate model_<N*250>.pt files for a ${SMOKE_ITERS}-iteration run)"
  ls -la "${SMOKE_RUN_DIR}" || true
fi

# ---------------------------------------------------------------------------
# 9. Print the exact command for the real (paper-scale) training run.
#    NOT executed here — this script only smoke-tests the install.
# ---------------------------------------------------------------------------
echo "--- [9/9] Real training command (paste to run; not executed by this script) ---"
cat <<CMD_EOF

  source ${VENV_DIR}/bin/activate
  cd ${PACE_DIR}
  python -m legged_lab.scripts.train \\
    --task=t1_tt \\
    --num_envs=4096 \\
    --headless \\
    --logger=tensorboard

(num_envs=4096 is TTEnvCfg.scene.num_envs's own default, PACE_ANALYSIS.md §2/§3;
omitting --max_iterations runs the config default of 10,000,
T1TableTennisAgentCfg.max_iterations, t1_tt_config.py:325.)

Logs:        ${PACE_DIR}/logs/t1_table_tennis/<timestamp>/
Checkpoints: model_<it>.pt every save_interval=250 iterations, plus the final
             unconditional save at exit (model_9999.pt for a full 10,000-iteration run).
Eval + ONNX/policy.pt export (PACE_ANALYSIS.md §8) only happens when you run:
  python -m legged_lab.scripts.eval --task=t1_tt_eval --num_envs=16 --headless \\
    --load_run <timestamp_dir_name> --checkpoint model_9999.pt
CMD_EOF

echo "=== gpu_box_setup.sh finished $(date -u +%FT%TZ) ==="
echo "Full log: ${LOG_FILE}"
