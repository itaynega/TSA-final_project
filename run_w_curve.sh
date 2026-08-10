#!/bin/bash
# J-15, Arm B: W-sensitivity sweep.
# Run from the repo root in MINGW64:  bash run_w_curve.sh
#
# Encodes: Precondition 1 (anchor re-derivation, hard stop on any drift),
# Precondition 2 (pytest gate), the seed-major 21-run sweep, the --cycle auto
# demonstration under a distinct model_id, before/after checkpoint capture,
# and a final pytest + git status read. Everything is teed to w_curve.log.
#
# Does NOT write report/w_curve.md — that's assembled afterward from the
# resulting results/validation/*.json files, once this finishes.

set -uo pipefail

PYTHON="${PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG="w_curve.log"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo "J-15 Arm B W-curve run started: $(date)"
echo "=============================================================="

echo
echo "--- Precondition 1: re-derive the anchor ---"
"$PYTHON" tools/validation_metrics.py
p1_status=$?
if [ $p1_status -ne 0 ]; then
  echo "validation_metrics.py exited nonzero ($p1_status). STOP. Train nothing."
  exit 1
fi

"$PYTHON" - <<'EOF'
import json, sys
# NOTE (2026-08-10, post Gate-1 incident): the original reconstruction anchor
# (0.6712632722155959) was lost when the seed2024/H=96 checkpoint was
# overwritten during J-14 return verification and could not be restored byte-
# exact (no backup existed in any _snapshots/*.tar.gz, and CPU training here
# is not bit-reproducible even under a fixed seed -- confirmed by the smoke
# run and the repair run themselves landing 10 sig figs apart on identical
# config). Per standing order 5 ("the run wins"), Amitay decided to adopt the
# current, reproducibly-observed value as the new anchor going forward. No
# report/docs file ever cited the old number, so nothing else needed correcting.
mse1_expected = 0.6869550701723053
n1_expected = 661640
mse2_expected = 0.6795092456048932  # tq0ca0 anchor, never affected, unchanged

p1 = json.load(open("results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json"))
p2 = json.load(open("results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_tq0ca0.json"))
mse1, n1 = p1["val_MSE"], p1["n_params"]
mse2 = p2["val_MSE"]
ok = (mse1 == mse1_expected) and (n1 == n1_expected) and (mse2 == mse2_expected)
print("ANCHOR CHECK (post-incident anchor): mse1={!r} n_params={!r} mse2(tq0ca0)={!r} -> {}".format(
    mse1, n1, mse2, "OK" if ok else "MISMATCH"))
sys.exit(0 if ok else 1)
EOF
if [ $? -ne 0 ]; then
  echo
  echo "!!! PRECONDITION 1 FAILED: anchor drifted. STOP. Train nothing. !!!"
  echo "Return to the PM — this is a Gate-1 problem, not a J-15 problem."
  exit 1
fi
echo "Precondition 1 passed."

echo
echo "--- Precondition 2: pytest ---"
"$PYTHON" -m pytest -q | tee /tmp/pytest_pre.txt
if ! grep -q "178 passed" /tmp/pytest_pre.txt; then
  echo
  echo "!!! PRECONDITION 2 FAILED: pytest did not report '178 passed'. STOP. !!!"
  exit 1
fi
echo "Precondition 2 passed."

echo
echo "--- Recording protected checkpoints BEFORE any training ---"
find TQNet/checkpoints -name checkpoint.pth -newermt '2000-01-01' \
  -printf '%T@ %p\n' | sort > /tmp/ckpt_before.txt
wc -l /tmp/ckpt_before.txt

echo
echo "--- 21-run W-curve sweep (seed-major) ---"
cd TQNet
SWEEP_START=$(date +%s)
for SEED in 2024 2025 2026; do
  for W in 6 8 12 23 25 48 168; do
    echo
    echo "=== seed=${SEED} W=${W}  ($(date)) ==="
    RUN_START=$(date +%s)
    "$PYTHON" -u run.py \
      --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv \
      --model_id ETTh1_96_96 --model TQNet --data ETTh1 --features M \
      --seq_len 96 --pred_len 96 --enc_in 7 --cycle ${W} \
      --train_epochs 30 --patience 5 --dropout 0.5 --accelerator cpu \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed ${SEED}
    RUN_STATUS=$?
    echo "seed=${SEED} W=${W} exit=${RUN_STATUS} wall=$(( $(date +%s) - RUN_START ))s"
    if [ $RUN_STATUS -ne 0 ]; then
      echo "!!! Training run failed (seed=${SEED} W=${W}). Stopping sweep. !!!"
      exit 1
    fi
  done
done
echo "Sweep wall-clock: $(( $(date +%s) - SWEEP_START ))s"

echo
echo "--- --cycle auto demonstration (distinct model_id: ETTh1_96_96_armB_auto) ---"
"$PYTHON" -u run.py \
  --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv \
  --model_id ETTh1_96_96_armB_auto --model TQNet --data ETTh1 --features M \
  --seq_len 96 --pred_len 96 --enc_in 7 --cycle auto \
  --train_epochs 30 --patience 5 --dropout 0.5 --accelerator cpu \
  --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed 2024
AUTO_STATUS=$?
cd "$REPO_ROOT"
if [ $AUTO_STATUS -ne 0 ]; then
  echo "!!! --cycle auto run failed. !!!"
  exit 1
fi

echo
echo "--- Post-training validation metrics ---"
"$PYTHON" tools/validation_metrics.py

echo
echo "--- Re-reading the two anchors after all training ---"
"$PYTHON" - <<'EOF'
import json
p1 = json.load(open("results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json"))
p2 = json.load(open("results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_tq0ca0.json"))
print("POST-TRAINING ANCHOR: mse1={!r} n_params={!r} mse2(tq0ca0)={!r}".format(
    p1["val_MSE"], p1["n_params"], p2["val_MSE"]))
EOF

echo
echo "--- Checkpoint diff (protected set must be additions-only) ---"
find TQNet/checkpoints -name checkpoint.pth -printf '%T@ %p\n' | sort > /tmp/ckpt_after.txt
diff /tmp/ckpt_before.txt /tmp/ckpt_after.txt
echo "(diff exit code: $?; 0/1 both fine here, 1 just means differences exist)"

echo
echo "--- Final pytest ---"
"$PYTHON" -m pytest -q

echo
echo "--- git status (read-only) ---"
git --no-optional-locks status --porcelain

echo
echo "=============================================================="
echo "J-15 Arm B W-curve run finished: $(date)"
echo "=============================================================="
