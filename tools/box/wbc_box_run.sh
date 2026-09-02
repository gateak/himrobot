#!/usr/bin/env bash
#
# wbc_box_run.sh -- whole-body-controller (WBC) training on a rented HIM
# Isaac Lab box, docs/design/whole_body_v1.md sec 7 ("Box plan"). Adapted
# from tools/box/box_param_sweep.sh's own push-timer / SIGINT-before-expiry /
# score / push / self-stop sequence (see that script's header) -- the two
# differences this task needs: a version gate with a hard abort before any
# GPU time is spent (Gate 1), and a two-sided MuJoCo-vs-Isaac replay
# comparison before training starts at all (Gate 0).
#
# Sequence, unattended:
#   0. venv probe (symlink to whichever interpreter already has isaaclab);
#   1. Gate 1: isaaclab/isaacsim/rsl-rl-lib version check, hard abort at
#      GATE1_ABORT_MIN if the box needs a from-scratch Isaac Lab install;
#   2. MJCF -> USD (himrobot.wbc.isaac.scripts.convert_mjcf, the installed
#      MjcfConverter API) + the FK-truth parity check against
#      assets/wbc/actuators_mujoco.json (himrobot.wbc.isaac.scripts.verify_usd_parity);
#   3. Gate 0: replay the held-out swing_lib_eval.npz with a=0 in Isaac
#      (himrobot.wbc.isaac.scripts.replay_zero_action) and diff its fall rate
#      against the MuJoCo half (himrobot.wbc.replay_library --mode ball-parked,
#      run on the Mac beforehand) at +-GATE0_TOLERANCE_POINTS;
#   4. a SMOKE_ITERATIONS smoke run -> smoke_timing.json (s/iteration); an
#      OOM/crash at NUM_ENVS retries once at NUM_ENVS/2;
#   5. the main training run, sized from the smoke timing and the lifetime
#      ACTUALLY left when it starts, SIGINT'd at T-STOP_BEFORE_MIN so
#      final.pt is written;
#   6. export + score the last three checkpoints on the box CPU
#      (himrobot.eval.eval_seeds --free-base, the honest scene, at the
#      TT_ROBOT_Y stance the library was recorded at), pick by legal x stand,
#      write RESULT.md;
#   7. final push, `him machine stop`.
#
# ANY exit -- a failed gate, a crash, set -e -- goes through on_exit, which
# writes STATUS.md, pushes what exists and stops the machine, so no failure
# mode leaves the box burning credits until its lifetime cap.
#
# Every python call is $PY = $VENV/bin/python, never bare `python` (CLAUDE.md
# box convention). Results are pushed to HIM on a PUSH_EVERY_MIN timer via
# tools/box/box_push_results.py --glob (dataset himrobot-rl-checkpoints), and
# the box stops itself with STOP_BEFORE_MIN of lifetime still budgeted for
# the final export/score/push/stop sequence -- same convention as
# box_param_sweep.sh, see docs/runbooks/whole_body_box.md for the token/
# self-stop decision this presumes (HIM-push, not rsync-pull).
#
# Run on the box, from a fresh shell, after rsync of the repo into
# /workspace/himrobot and `him login`:
#     MACHINE_ID=ssh_xxx HOURS=4 nohup bash tools/box/wbc_box_run.sh \
#         > /workspace/wbc_box.log 2>&1 &
#
# DRY=1 prints every command this script would run (`+ <cmd>`) and exits 0
# without invoking him, python, or any GPU/Isaac tool -- use it to read the
# full sequence before spending credits.
#
# Every knob below is an environment variable with a default.
set -euo pipefail

REPO="${REPO:-/workspace/himrobot}"
VENV="${VENV:-/workspace/venv}"
PY="$VENV/bin/python"
HIM="${HIM:-$HOME/.local/bin/him}"
export HIM_API_URL="${HIM_API_URL:-https://arena.himrobotics.com}"
MACHINE_ID="${MACHINE_ID:?set MACHINE_ID=ssh_... (from him machine launch --json) so the box can stop itself}"
HOURS="${HOURS:-4}"                       # the --hours the machine was launched with
STOP_BEFORE_MIN="${STOP_BEFORE_MIN:-35}"  # leave this much of the lifetime for export + score + push + stop
PUSH_EVERY_MIN="${PUSH_EVERY_MIN:-30}"
GATE1_ABORT_MIN="${GATE1_ABORT_MIN:-90}"  # docs/design/whole_body_v1.md sec 7.2's worst-case-loss cap
EVAL_JOBS="${EVAL_JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)}"
# The stance every swing library is recorded at and every reference row in
# docs/design/whole_body_v1.md sec 8 is scored at (himrobot.sim.scene reads it
# once at import). Exported so the on-box eval_seeds scoring, the Isaac spawn
# pose and the MuJoCo libraries all agree.
export TT_ROBOT_Y="${TT_ROBOT_Y:--0.10}"

NUM_ENVS="${NUM_ENVS:-4096}"
SMOKE_ITERATIONS="${SMOKE_ITERATIONS:-200}"
RUN_NAME="${RUN_NAME:-wbc_v1_a}"
LEG_DEFAULT="${LEG_DEFAULT:-stand}"       # stand (parity, recommended for box 1) | mocap
CURRICULUM="${CURRICULUM:-default}"
OUT_DIR="${OUT_DIR:-logs/rsl_rl/wbc_v1}"
SWING_LIB_DIR="${SWING_LIB_DIR:-work/wbc}"  # nominal.npz / wide.npz / slow.npz / swing_lib_eval.npz (WB1 output)
USD_PATH="${USD_PATH:-/workspace/assets/g1_29dof_paddle.usd}"
GATE0_NUM_ENVS="${GATE0_NUM_ENVS:-600}"
GATE0_TOLERANCE_POINTS="${GATE0_TOLERANCE_POINTS:-15}"  # +-15 points, docs/design/whole_body_v1.md sec 7.2
MUJOCO_GATE0="${MUJOCO_GATE0:-$SWING_LIB_DIR/replay_swing_lib_eval_ball-parked.json}"  # the Mac-side half
N_EXPORT_CKPTS="${N_EXPORT_CKPTS:-3}"
RESUME="${RESUME:-}"                      # box 2: a model_<it>.pt to --resume from
DRY="${DRY:-0}"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

run() {
  if [ "$DRY" = "1" ]; then
    echo "+ $*"
  else
    log "+ $*"
    "$@"
  fi
}

START_TS=$(date +%s)
LIFETIME_END=$(( START_TS + HOURS * 3600 ))
OUT_BASE="$(basename "$OUT_DIR")"   # the push globs are relative to --root = dirname(OUT_DIR)
TRAIN_PID=0
STATUS_NOTE="running"

minutes_left() { echo $(( (LIFETIME_END - $(date +%s)) / 60 )); }

if [ "$DRY" = "1" ]; then
  echo "+ cd $REPO"
  echo "+ mkdir -p $OUT_DIR $SWING_LIB_DIR"
else
  cd "$REPO"
  mkdir -p "$OUT_DIR" "$SWING_LIB_DIR"
fi
log "wbc_box_run.sh starting: hours=$HOURS stop_before=$STOP_BEFORE_MIN dry=$DRY leg_default=$LEG_DEFAULT run_name=$RUN_NAME TT_ROBOT_Y=$TT_ROBOT_Y"

push() {
  run "$PY" tools/box/box_push_results.py --root "$(dirname "$OUT_DIR")" --machine "$MACHINE_ID" --him "$HIM" \
    --description "whole-body-controller training ($MACHINE_ID, $RUN_NAME)" \
    --glob "$OUT_BASE/*/model_*.pt" --glob "$OUT_BASE/*/final.pt" --glob "$OUT_BASE/*/exported/*.npz" \
    --glob "$OUT_BASE/*/params/*.yaml" --glob "$OUT_BASE/*/events*" \
    --glob "$OUT_BASE/*.json" --glob "$OUT_BASE/*.md" --glob "$OUT_BASE/*.txt" \
    --glob "$OUT_BASE/scoreboard_*/scoreboard.*" \
    || log "push failed (will retry next tick)"
}

stop_machine() {
  log "stopping machine $MACHINE_ID with $(minutes_left) min of lifetime left"
  run "$HIM" machine stop "$MACHINE_ID" --yes || log "machine stop failed -- STOP IT MANUALLY"
}

FINISHED=0
on_exit() {
  local rc=$?
  [ "$FINISHED" = "1" ] && return
  FINISHED=1
  log "on_exit: rc=$rc note=$STATUS_NOTE"
  if [ "$DRY" != "1" ]; then
    if [ "$TRAIN_PID" != "0" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
      kill -INT "$TRAIN_PID" 2>/dev/null || true; sleep 30; kill -9 "$TRAIN_PID" 2>/dev/null || true
    fi
    mkdir -p "$OUT_DIR"
    printf '# STATUS\n\nexit code %s at %s (%s min of lifetime left)\n\n%s\n' \
      "$rc" "$(date -u +%FT%TZ)" "$(minutes_left)" "$STATUS_NOTE" > "$OUT_DIR/STATUS.md"
    push
    stop_machine
  fi
}
trap on_exit EXIT

# ---------------------------------------------------------------- 0. venv probe
if [ ! -x "$PY" ]; then
  log "no venv at $VENV; probing candidate interpreters for isaaclab"
  FOUND=""
  for cand in /isaac-sim/python.sh /workspace/isaaclab/_isaac_sim/python.sh /workspace/*/python.sh; do
    [ -x "$cand" ] || continue
    if "$cand" -c "import isaaclab" >/dev/null 2>&1; then FOUND="$cand"; break; fi
  done
  if [ -n "$FOUND" ]; then
    log "found isaaclab under $FOUND -- symlinking $VENV to it"
    run mkdir -p "$(dirname "$VENV")"
    run ln -sfn "$(dirname "$FOUND")" "$VENV"
  else
    log "no interpreter with isaaclab found yet -- Gate 1 below will install one"
  fi
fi

# ---------------------------------------------------------------- 1. Gate 1: version check
log "Gate 1: isaaclab / isaacsim / rsl-rl-lib versions"
GATE1_OK=0
if [ "$DRY" = "1" ]; then
  echo "+ $PY -m pip show isaaclab isaacsim rsl-rl-lib"
  GATE1_OK=1
else
  VERSIONS="$("$PY" -m pip show isaaclab isaacsim rsl-rl-lib 2>&1 || true)"
  echo "$VERSIONS" | tee "$OUT_DIR/versions.txt"
  # isaaclab's OWN version line (pip show prints one block per package; a
  # 2.x rsl-rl-lib line must not pass a 3.x isaaclab).
  ISAACLAB_VERSION="$(echo "$VERSIONS" | awk '/^Name: isaaclab$/{f=1} f&&/^Version:/{print $2; exit}')"
  log "isaaclab version: ${ISAACLAB_VERSION:-<absent>}"
  case "$ISAACLAB_VERSION" in 2.1.*|2.2.*) GATE1_OK=1 ;; esac
fi
if [ "$GATE1_OK" != "1" ]; then
  log "Gate 1 FAILED (isaaclab is not 2.1.x/2.2.x) -- falling back to tools/box/gpu_box_setup.sh steps 1-5," \
      "hard abort at ${GATE1_ABORT_MIN} min so a bad image never eats the whole box"
  STATUS_NOTE="Gate 1: isaaclab not 2.1.x/2.2.x; gpu_box_setup.sh fallback"
  if [ "$DRY" = "1" ]; then
    echo "+ timeout ${GATE1_ABORT_MIN}m bash tools/box/gpu_box_setup.sh"
  else
    if ! timeout "${GATE1_ABORT_MIN}m" bash tools/box/gpu_box_setup.sh; then
      STATUS_NOTE="Gate 1 FAILED: could not get isaaclab 2.1.x/2.2.x working within ${GATE1_ABORT_MIN} min"
      log "$STATUS_NOTE"
      exit 2
    fi
  fi
fi
log "Gate 1 passed"

# ---------------------------------------------------------------- 2. MJCF -> USD + FK-truth check
log "converting assets/wbc/g1_29dof_paddle_free.xml -> $USD_PATH"
STATUS_NOTE="MJCF -> USD conversion"
run mkdir -p "$(dirname "$USD_PATH")"
run "$PY" -m himrobot.wbc.isaac.scripts.convert_mjcf --headless "assets/wbc/g1_29dof_paddle_free.xml" "$USD_PATH"

log "verifying the converted USD against assets/wbc/actuators_mujoco.json's fk_truth block"
STATUS_NOTE="USD parity check (fk_truth)"
run env G1_WBC_USD_PATH="$USD_PATH" "$PY" -m himrobot.wbc.isaac.scripts.verify_usd_parity --headless \
  --swing_lib "$SWING_LIB_DIR/swing_lib_eval.npz" --out "$OUT_DIR/usd_parity.json"
log "USD parity passed"

# ---------------------------------------------------------------- 3. Gate 0: two-sided replay
log "Gate 0: replaying swing_lib_eval.npz with a=0 in Isaac (${GATE0_NUM_ENVS} rows) and diffing against MuJoCo"
STATUS_NOTE="Gate 0 (Isaac replay)"
if [ "$DRY" != "1" ] && [ ! -f "$MUJOCO_GATE0" ]; then
  STATUS_NOTE="Gate 0 FAILED: $MUJOCO_GATE0 not found -- run himrobot.wbc.replay_library --mode ball-parked on the Mac and rsync it up"
  log "$STATUS_NOTE"
  exit 2
fi
run env G1_WBC_USD_PATH="$USD_PATH" "$PY" -m himrobot.wbc.isaac.scripts.replay_zero_action \
  --swing_lib "$SWING_LIB_DIR/swing_lib_eval.npz" --num_envs "$GATE0_NUM_ENVS" --headless \
  --out "$OUT_DIR/gate0_isaac.json"

if [ "$DRY" = "1" ]; then
  echo "+ $PY -c '<compare $OUT_DIR/gate0_isaac.json against $MUJOCO_GATE0 at +-$GATE0_TOLERANCE_POINTS points>'"
else
  "$PY" - "$OUT_DIR/gate0_isaac.json" "$MUJOCO_GATE0" "$GATE0_TOLERANCE_POINTS" <<'PYEOF'
import json
import sys

isaac_path, mujoco_path, tol_points = sys.argv[1], sys.argv[2], float(sys.argv[3])
isaac = json.load(open(isaac_path))
mujoco = json.load(open(mujoco_path))

isaac_idle_rate = 100.0 * isaac["idle"]["falls"]["rate"]
if isaac_idle_rate > 0.0:
    print(f"Gate 0 FAILED: STAND_POSE-only rows fell {isaac_idle_rate:.1f}% of the time in Isaac (must be 0%)")
    sys.exit(2)

isaac_rate = 100.0 * isaac["falls"]["rate"]
mujoco_rate = mujoco["falls_pct"]
diff = abs(isaac_rate - mujoco_rate)
print(f"Gate 0: Isaac falls {isaac_rate:.1f}%, MuJoCo falls {mujoco_rate:.1f}% ({mujoco.get('mode')}), "
      f"diff {diff:.1f} points (tolerance {tol_points:.0f})")
if diff > tol_points:
    print("Gate 0 FAILED: sim-to-sim fall-rate gap exceeds tolerance")
    sys.exit(2)
print("Gate 0 passed")
PYEOF
fi
log "Gate 0 passed"

# ---------------------------------------------------------------- 4. smoke run (OOM -> retry at NUM_ENVS/2)
LIB_ARGS=(--swing_lib "nominal=$SWING_LIB_DIR/nominal.npz" --swing_lib "wide=$SWING_LIB_DIR/wide.npz" \
          --swing_lib "slow=$SWING_LIB_DIR/slow.npz")
smoke() {
  run env G1_WBC_USD_PATH="$USD_PATH" "$PY" -m himrobot.wbc.isaac.scripts.train --task Himrobot-WBC-G1-v0 \
    --num_envs "$NUM_ENVS" --max_iterations "$SMOKE_ITERATIONS" --headless --logger tensorboard \
    --run_name smoke --leg_default "$LEG_DEFAULT" --out_dir "$OUT_DIR" "${LIB_ARGS[@]}"
}
STATUS_NOTE="smoke run"
log "smoke: ${SMOKE_ITERATIONS} iterations at ${NUM_ENVS} envs"
SMOKE_START=$(date +%s)
if ! smoke; then
  NUM_ENVS=$(( NUM_ENVS / 2 ))
  log "smoke failed at $(( NUM_ENVS * 2 )) envs (OOM?) -- retrying once at ${NUM_ENVS}"
  SMOKE_START=$(date +%s)
  smoke
fi
SMOKE_ELAPSED=$(( $(date +%s) - SMOKE_START ))

if [ "$DRY" = "1" ]; then
  S_PER_IT="0.100000"
  echo "+ write $OUT_DIR/smoke_timing.json"
else
  S_PER_IT=$("$PY" -c "print(${SMOKE_ELAPSED} / ${SMOKE_ITERATIONS})")
  cat > "$OUT_DIR/smoke_timing.json" <<JSON
{"iterations": ${SMOKE_ITERATIONS}, "num_envs": ${NUM_ENVS}, "elapsed_s": ${SMOKE_ELAPSED},
 "s_per_iteration": ${S_PER_IT}, "note": "elapsed includes Isaac Sim startup and the USD load"}
JSON
fi
log "smoke done: ${SMOKE_ELAPSED}s for ${SMOKE_ITERATIONS} iterations (${S_PER_IT} s/iteration)"

# ---------------------------------------------------------------- 5. size + run the main training
# Sized from the lifetime ACTUALLY left now (setup + gates + smoke took what
# they took), not from a fixed setup allowance.
TRAIN_MIN=$(( $(minutes_left) - STOP_BEFORE_MIN ))
if [ "$DRY" = "1" ]; then
  ITERATIONS=1000
elif [ "$TRAIN_MIN" -lt 10 ]; then
  STATUS_NOTE="only ${TRAIN_MIN} min left for training after setup -- not starting the main run"
  log "$STATUS_NOTE"
  exit 2
else
  ITERATIONS=$("$PY" -c "
train_s = ${TRAIN_MIN} * 60
s_per_it = max(${S_PER_IT}, 1e-6)
print(max(500, int(train_s / s_per_it)))
")
fi
log "sized main run: ${ITERATIONS} iterations for a ${TRAIN_MIN}-minute budget ($(minutes_left) min of lifetime left)"

RESUME_ARGS=()
[ -n "$RESUME" ] && RESUME_ARGS=(--resume "$RESUME")
STATUS_NOTE="main training run ($RUN_NAME)"
# Backgrounded directly (not through run()) so $! is the real training
# process's own PID -- SIGINT below must reach the actual python process for
# scripts/train.py's KeyboardInterrupt handler (which writes final.pt).
if [ "$DRY" = "1" ]; then
  echo "+ $PY -m himrobot.wbc.isaac.scripts.train --task Himrobot-WBC-G1-v0 --num_envs $NUM_ENVS" \
       "--max_iterations $ITERATIONS --headless --logger tensorboard --run_name $RUN_NAME" \
       "--leg_default $LEG_DEFAULT --curriculum $CURRICULUM --max_minutes $TRAIN_MIN --out_dir $OUT_DIR" \
       "${LIB_ARGS[*]} ${RESUME_ARGS[*]:-} (backgrounded)"
  TRAIN_PID=0
else
  G1_WBC_USD_PATH="$USD_PATH" "$PY" -m himrobot.wbc.isaac.scripts.train --task Himrobot-WBC-G1-v0 \
    --num_envs "$NUM_ENVS" --max_iterations "$ITERATIONS" --headless --logger tensorboard \
    --run_name "$RUN_NAME" --leg_default "$LEG_DEFAULT" --curriculum "$CURRICULUM" \
    --max_minutes "$TRAIN_MIN" --out_dir "$OUT_DIR" "${LIB_ARGS[@]}" "${RESUME_ARGS[@]}" &
  TRAIN_PID=$!
fi
log "training pid $TRAIN_PID"

# Push every PUSH_EVERY_MIN; poll every minute so the SIGINT lands within a
# minute of T-STOP_BEFORE_MIN, not up to a whole push interval late.
TRAIN_END=$(( $(date +%s) + TRAIN_MIN * 60 ))
if [ "$DRY" != "1" ]; then
  LAST_PUSH=$(date +%s)
  while [ "$(date +%s)" -lt "$TRAIN_END" ]; do
    sleep 60
    kill -0 "$TRAIN_PID" 2>/dev/null || { log "tick: training exited"; break; }
    if [ $(( $(date +%s) - LAST_PUSH )) -ge $(( PUSH_EVERY_MIN * 60 )) ]; then
      log "tick: training alive, pushing"
      push
      LAST_PUSH=$(date +%s)
    fi
  done
fi

# ---------------------------------------------------------------- 6. stop training, export, score
STATUS_NOTE="export + score"
if [ "$DRY" = "1" ]; then
  echo "+ kill -INT $TRAIN_PID   # SIGINT -> KeyboardInterrupt -> final.pt (see scripts/train.py)"
  echo "+ kill -9 $TRAIN_PID"
else
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -INT "$TRAIN_PID" 2>/dev/null || true
    sleep 90
    kill -9 "$TRAIN_PID" 2>/dev/null || true
  fi
  wait "$TRAIN_PID" 2>/dev/null || true
fi

log "exporting the last ${N_EXPORT_CKPTS} checkpoints"
if [ "$DRY" = "1" ]; then
  echo "+ export the last ${N_EXPORT_CKPTS} model_*.pt under $OUT_DIR/$RUN_NAME to $OUT_DIR/$RUN_NAME/exported/"
  echo "+ TT_ROBOT_Y=$TT_ROBOT_Y TT_WBC_NPZ=<npz> $PY -m himrobot.eval.eval_seeds --policy <label>=himrobot.policies.wbc_np:NumpyWBCPolicy --free-base --params '{\"robot_table_collision\": false}' --jobs $EVAL_JOBS --out $OUT_DIR/scoreboard_<label>"
else
  mapfile -t CKPTS < <(ls -t "$OUT_DIR/$RUN_NAME"/model_*.pt 2>/dev/null | head -n "$N_EXPORT_CKPTS")
  EXPORT_DIR="$OUT_DIR/$RUN_NAME/exported"   # matches the push glob and the runbook
  mkdir -p "$EXPORT_DIR"
  SCORES_FILE="$OUT_DIR/scores.txt"
  : > "$SCORES_FILE"
  for ckpt in "${CKPTS[@]:-}"; do
    [ -n "$ckpt" ] || continue
    label="$(basename "${ckpt%.pt}")"
    npz="$EXPORT_DIR/$label.npz"
    if ! run "$PY" -m himrobot.wbc.isaac.scripts.export --checkpoint "$ckpt" --out "$npz" --leg_default "$LEG_DEFAULT"; then
      log "export failed for $ckpt -- skipping"; continue
    fi
    out_dir="$OUT_DIR/scoreboard_$label"
    if ! run env TT_WBC_NPZ="$npz" "$PY" -m himrobot.eval.eval_seeds \
      --policy "$label=himrobot.policies.wbc_np:NumpyWBCPolicy" --free-base \
      --params '{"robot_table_collision": false}' --jobs "$EVAL_JOBS" --out "$out_dir"; then
      log "eval_seeds failed for $label -- skipping"; continue
    fi
    # scoreboard.json rows: legal_return_rate in PERCENT, falls a COUNT out
    # of d["trials"] serves per seed (himrobot.eval.evaluate metrics).
    "$PY" -c "
import json
d = json.load(open('$out_dir/scoreboard.json'))
rows = d['rows']
trials = int(d['trials'])
legal = sum(r['legal_return_rate'] for r in rows) / len(rows)
falls_pct = 100.0 * sum((r.get('falls') or 0) for r in rows) / (len(rows) * trials)
stand = 100.0 - falls_pct
print(f'$label {legal * stand / 100.0:.2f} {legal:.2f} {stand:.2f} {falls_pct:.2f}')
" >> "$SCORES_FILE"
  done

  BEST_LABEL="$(sort -k2 -n -r "$SCORES_FILE" | head -1 | awk '{print $1}')"
  log "best checkpoint by legal x stand: ${BEST_LABEL:-<none scored>}"
  {
    echo "# WBC box run: $RUN_NAME ($MACHINE_ID)"
    echo
    echo "leg_default=$LEG_DEFAULT curriculum=$CURRICULUM iterations=$ITERATIONS num_envs=$NUM_ENVS TT_ROBOT_Y=$TT_ROBOT_Y"
    echo "600 serves (seeds 11/2000/4000 x 200), honest free-base scene (robot_table_collision=false), mean action."
    echo
    echo "| checkpoint | legal x stand / 100 | legal % | stand % | falls % |"
    echo "|---|---|---|---|---|"
    awk '{printf "| %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5}' "$SCORES_FILE"
    echo
    echo "Best: **${BEST_LABEL:-none}**"
    echo
    echo "Gate 0 / USD parity: see gate0_isaac.json, usd_parity.json, smoke_timing.json next to this file."
    echo "Reference rows (docs/design/whole_body_v1.md sec 8.2): welded v3 77.0/49.5/0.28m;"
    echo "waist-free v3 61.5% legal / 75.7% falls; waist-locked v3 45.3%/3.5%/0.68m/14.8% falls."
  } > "$OUT_DIR/RESULT.md"
fi

# ---------------------------------------------------------------- 7. final push + stop
STATUS_NOTE="completed"
FINISHED=1
push
stop_machine
