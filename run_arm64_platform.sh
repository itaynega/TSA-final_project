#!/bin/bash
# Cross-platform replication of the live Stage-2 arms on macOS/arm64.
#
# Why this exists: every Stage-2 arm was measured on Amitay's x86 machine. This
# machine is the arm64 one docs/STATUS.md G2 recorded as unavailable, and its five
# Stage-1 checkpoints survive here. Re-running the arms on it measures how much of
# the noise the project has been fighting is platform rather than method.
#
# This is NOT a new arm and it changes no pre-registered prediction. It is
# post-hoc by construction (report/prereg-improvement.md sec 4 was applied on
# 2026-08-10 before this ran) and belongs in F6.
#
# The hazard this script is built around: with the default --model_id, Arm B
# resolves --cycle auto to 24 and writes into
# ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024, and Arm D's criterion
# fires "drop" and writes into ..._tq0ca0. Both are protected checkpoints on this
# machine -- the second is what closed G2. Every run below therefore carries a
# distinct --model_id, and the five protected checkpoints are hashed before the
# first run and after every run, with a hard stop on any change (CLAUDE.md).
#
# Writes: TQNet/checkpoints/ETTh1_96_96_arm64_*, TQNet/results/ETTh1_96_96_arm64_*,
# and TQNet/result_arm64.txt (deliberately not the tracked result_ours.txt).
# Touches no existing checkpoint, no results/runs/ record, no report file.
#
# Run it so the output is captured:
#
#     bash run_arm64_platform.sh 2>&1 | tee -a arm64_platform.log
#
# This script deliberately does NOT tee itself. `exec > >(tee -a "$LOG") 2>&1`, as
# used by run_w_curve.sh, leaves the tee child unreaped; under a sandboxed shell it
# was killed before flushing and left a zero-byte log while the runs themselves
# completed normally (observed 2026-08-10). Redirecting from the caller cannot fail
# that way.

set -uo pipefail

PYTHON="${PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

BEFORE=/tmp/arm64_ckpt_before.txt
AFTER=/tmp/arm64_ckpt_after.txt

echo "=============================================================="
echo "arm64 platform replication started: $(date)"
echo "uname: $(uname -m) / $(uname -s)"
echo "=============================================================="

# The five Stage-1 checkpoints that exist only on this machine.
protected_hashes() {
  find TQNet/checkpoints -name checkpoint.pth \
    -not -path '*arm64*' -exec shasum -a 256 {} \; | sort -k2
}

echo
echo "--- Precondition: pytest ---"
"$PYTHON" -m pytest -q | tail -3
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "!!! pytest failed. STOP. Train nothing. !!!"
  exit 1
fi

echo
echo "--- Recording the protected checkpoints BEFORE any training ---"
protected_hashes > "$BEFORE"
cat "$BEFORE"
echo "protected checkpoint count: $(wc -l < "$BEFORE")"
if [ "$(wc -l < "$BEFORE")" -ne 5 ]; then
  echo "!!! Expected 5 protected checkpoints, found $(wc -l < "$BEFORE"). STOP. !!!"
  exit 1
fi

check_protected() {
  protected_hashes > "$AFTER"
  if ! diff -q "$BEFORE" "$AFTER" > /dev/null; then
    echo
    echo "!!! A PROTECTED CHECKPOINT CHANGED. STOP IMMEDIATELY. !!!"
    diff "$BEFORE" "$AFTER"
    echo "Per CLAUDE.md this is a stop-and-report event. Nothing further runs."
    exit 1
  fi
  echo "  [guard] 5 protected checkpoints unchanged."
}

run_one() {
  local model_id="$1"; local seed="$2"; shift 2
  echo
  echo "=== ${model_id} seed=${seed}  ($(date)) ==="
  local start; start=$(date +%s)
  ( cd TQNet && "$PYTHON" -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv \
      --model_id "$model_id" --model TQNet --data ETTh1 --features M \
      --seq_len 96 --pred_len 96 --enc_in 7 \
      --train_epochs 30 --patience 5 --dropout 0.5 --accelerator cpu \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed "$seed" \
      --result_path ./result_arm64.txt --save_outputs 1 \
      "$@" )
  local status=$?
  echo "${model_id} seed=${seed} exit=${status} wall=$(( $(date +%s) - start ))s"
  if [ $status -ne 0 ]; then
    echo "!!! Run failed. Stopping. !!!"
    exit 1
  fi
  check_protected
}

SWEEP_START=$(date +%s)

echo
echo "########## Arm B -- period estimated from the training split ##########"
for SEED in 2024 2025 2026; do
  run_one ETTh1_96_96_arm64_armB "$SEED" --cycle auto
done

echo
echo "########## Arm D -- channel-count-conditional attention ##########"
for SEED in 2024 2025 2026; do
  run_one ETTh1_96_96_arm64_armD "$SEED" --cycle 24 --channel_criterion 1
done

echo
echo "########## Arm A -- damped-trend instance normalisation, phi=0.8 ##########"
for SEED in 2024 2025 2026; do
  run_one ETTh1_96_96_arm64_armA "$SEED" --cycle 24 --use_damped_trend 1 --damped_phi 0.8
done

echo
echo "Total wall-clock: $(( $(date +%s) - SWEEP_START ))s"

echo
echo "--- Final protected-checkpoint verification ---"
check_protected

echo
echo "--- New checkpoint directories created ---"
find TQNet/checkpoints -name checkpoint.pth -path '*arm64*' -exec shasum -a 256 {} \; | sort -k2

echo
echo "--- Final pytest ---"
"$PYTHON" -m pytest -q | tail -3

echo
echo "=============================================================="
echo "arm64 platform replication finished: $(date)"
echo "=============================================================="
