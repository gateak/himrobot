#!/usr/bin/env bash
#
# box_param_sweep.sh — hitter-v2 robustness sweep on a rented HIM SSH box (RunPod, 128 cores).
#
# What it does, unattended:
#   1. builds /workspace/venv (never bare `python`), runs the parity tests;
#   2. launches every (config x seed) parameter-residual run in parallel, one process each, warm-started
#      from the shipped v1 checkpoint, best.pt selected by deterministic evaluation;
#   3. pushes results to HIM (dataset himrobot-rl-checkpoints) every PUSH_EVERY_MIN minutes;
#   4. STOP_BEFORE_MIN minutes before the machine's lifetime ends: interrupts training (SIGINT, so final.pt
#      is written), scores every best.pt on seeds 11/7/3 (nominal and widened serves), pushes once more,
#      and stops the machine itself.
#
# Run on the box, from a fresh shell, after `git clone`/rsync of the repo into /workspace/himrobot and
# `him login` (the token copied here must be rotated once the box stops):
#     MACHINE_ID=ssh_xxx HOURS=8 nohup bash tools/box/box_param_sweep.sh > /workspace/sweep.log 2>&1 &
#
# Every knob below is an environment variable with a default.
set -euo pipefail

REPO="${REPO:-/workspace/himrobot}"
VENV="${VENV:-/workspace/venv}"
PY="$VENV/bin/python"
HIM="${HIM:-$HOME/.local/bin/him}"
export HIM_API_URL="${HIM_API_URL:-https://arena.himrobotics.com}"
MACHINE_ID="${MACHINE_ID:?set MACHINE_ID=ssh_... (from him machine launch --json) so the box can stop itself}"
HOURS="${HOURS:-8}"                       # the --hours the machine was launched with
STOP_BEFORE_MIN="${STOP_BEFORE_MIN:-35}"  # leave this much of the lifetime for final eval + push + stop
PUSH_EVERY_MIN="${PUSH_EVERY_MIN:-30}"
EVAL_MIN="${EVAL_MIN:-18}"                # budget for the final scoreboard (128 cores: 3 seeds x ~40 ckpts fits)
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
RESUME="${RESUME:-work/tt/rl/param/s3/best.pt}"   # v1 warm start; must exist on the box (rsync it)
OUT="${OUT:-work/tt/rl/param_v2_sweep}"
N_ENVS="${N_ENVS:-32}"
DET="${DET:---det-eval-episodes 100 --det-eval-every 4000 --det-eval-seed 1001}"
COMMON="${COMMON:---n-envs $N_ENVS --eval-every 200 --vel-penalty 1e-5 --legal-bonus 10 --param-space v2 --log-std-init-new -2.5 $DET}"
MUJOCO_PIN="${MUJOCO_PIN:-mujoco==3.12.0}"
HIM_SDK_PIN="${HIM_SDK_PIN:-him-sdk==0.1.3}"      # verify the PyPI name before the first box; see docs

# Sweep configs: label|extra trainer args. Widened serves + physics jitter are the robustness axes;
# "nominal" replicates the Mac recipe at box scale so the two can be compared.
CONFIGS=(
  "nominal|"
  "wide13_jit10|--params {\"serve_scale\":1.3,\"physics_jitter\":0.10}"
  "wide15_jit15|--params {\"serve_scale\":1.5,\"physics_jitter\":0.15}"
  "aim|--aim-bonus 8 --aim-sigma 0.30 --target-hit-bonus 4"
  "wide13_jit10_noise|--params {\"serve_scale\":1.3,\"physics_jitter\":0.10} --obs-noise 0.01 --obs-delay-steps 2"
)

TRAIN_MIN=$(( HOURS * 60 - STOP_BEFORE_MIN - 12 ))   # 12 min for setup
START_TS=$(date +%s)
LIFETIME_END=$(( START_TS + HOURS * 3600 ))
log() { echo "[$(date -u +%FT%TZ)] $*"; }

cd "$REPO"
mkdir -p "$OUT/logs"

# ---------------------------------------------------------------- 1. environment + parity
if [ ! -x "$PY" ]; then
  log "creating $VENV"
  python3 -m venv "$VENV"
  "$PY" -m pip install --upgrade pip >/dev/null
  "$PY" -m pip install "$MUJOCO_PIN" "numpy>=2" torch imageio imageio-ffmpeg pytest >/dev/null
fi
if [ ! -x "$HIM" ]; then
  log "installing the him CLI ($HIM_SDK_PIN) into its own venv"
  python3 -m venv /workspace/him-venv && /workspace/him-venv/bin/pip install "$HIM_SDK_PIN" >/dev/null
  HIM=/workspace/him-venv/bin/him
fi
"$HIM" whoami >/dev/null 2>&1 || { log "him CLI is not logged in on this box (run: $HIM login)"; exit 2; }
[ -f "$RESUME" ] || { log "warm-start checkpoint $RESUME missing"; exit 2; }
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1
log "nproc=$(nproc) train_min=$TRAIN_MIN configs=${#CONFIGS[@]} seeds=($SEEDS)"
"$PY" -m pytest -q tt_sim/tests/test_hitter_v2.py tt_sim/tests/test_param_residual_np.py -x 2>&1 | tail -3

# ---------------------------------------------------------------- 2. launch every run
PIDS=()
for cfg in "${CONFIGS[@]}"; do
  label="${cfg%%|*}"; extra="${cfg#*|}"
  for seed in $SEEDS; do
    run="$OUT/${label}_s${seed}"
    # shellcheck disable=SC2086
    nohup "$PY" -m tt_sim.rl.param_train $COMMON $extra --seed "$seed" --minutes "$TRAIN_MIN" \
      --resume-from "$RESUME" --out "$run" > "$OUT/logs/${label}_s${seed}.log" 2>&1 &
    PIDS+=($!)
  done
done
log "launched ${#PIDS[@]} runs"

push() {
  "$PY" tools/box/box_push_results.py --root "$OUT" --machine "$MACHINE_ID" --him "$HIM" \
    --description "hitter-v2 robustness sweep ($MACHINE_ID)" || log "push failed (will retry next tick)"
}

# ---------------------------------------------------------------- 3. push on a timer until training ends
TRAIN_END=$(( START_TS + (TRAIN_MIN + 12) * 60 ))
while [ "$(date +%s)" -lt "$TRAIN_END" ]; do
  sleep $(( PUSH_EVERY_MIN * 60 ))
  alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
  log "tick: $alive/${#PIDS[@]} runs alive"; push
  [ "$alive" -eq 0 ] && break
done

# ---------------------------------------------------------------- 4. stop training, score, push, stop machine
for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done   # SIGINT -> KeyboardInterrupt -> final.pt
sleep 90
for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null || true; done

CKPTS=(); for f in "$OUT"/*/best.pt; do [ -f "$f" ] && CKPTS+=(--ckpt "$f"); done
if [ "${#CKPTS[@]}" -gt 0 ]; then
  log "scoring ${#CKPTS[@]} checkpoints on seeds 11/7/3 (nominal, then widened serves + jitter)"
  timeout "${EVAL_MIN}m" "$PY" -m tt_sim.rl.eval_seeds "${CKPTS[@]}" --npz assets/policies/param_residual_s3.npz \
    --hitter --jobs 96 --out "$OUT/scoreboard_nominal" || log "nominal scoreboard timed out/failed"
  timeout "${EVAL_MIN}m" "$PY" -m tt_sim.rl.eval_seeds "${CKPTS[@]}" --npz assets/policies/param_residual_s3.npz \
    --hitter --jobs 96 --params '{"serve_scale":1.3,"physics_jitter":0.10}' --out "$OUT/scoreboard_wide13_jit10" \
    || log "widened scoreboard timed out/failed"
  timeout "${EVAL_MIN}m" "$PY" -m tt_sim.rl.eval_seeds "${CKPTS[@]}" --npz assets/policies/param_residual_s3.npz \
    --hitter --jobs 96 --obs-noise 0.01 --obs-delay-steps 2 --out "$OUT/scoreboard_noise01_delay2" \
    || log "noisy scoreboard timed out/failed"
fi
push
left=$(( LIFETIME_END - $(date +%s) ))
log "stopping machine $MACHINE_ID with $((left/60)) min of lifetime left"
"$HIM" machine stop "$MACHINE_ID" --yes || log "machine stop failed -- STOP IT MANUALLY"
