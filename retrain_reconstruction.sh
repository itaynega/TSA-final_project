#!/bin/bash
# Clean re-derivation of the reconstruction checkpoint at cycle=24, seed=2024, H=96,
# after the original was lost (STAGE2_WORKPLAN_2026-08-09.md §7j/§7k).
#
# Run from the repo root in MINGW64:  bash retrain_reconstruction.sh
#
# Uses a DISTINCT model_id (ETTh1_96_96_reconstruction_v2) so it cannot collide with
# or overwrite ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024 (still protected)
# or any of the other 26 protected checkpoints. Implements standing order 14: sha256
# every checkpoint.pth before and after, diff, must be additions-only.
#
# Rationale for expecting this to land near the lost anchor (0.6712632722155959):
# the W-curve sweep's --cycle auto run (ETTh1_96_96_armB_auto, cycle=24, seed=2024,
# same everything else) already landed at 0.6712632724477633 -- 2.3e-10 from the lost
# anchor, vs. -0.0157 from the checkpoint currently sitting in the protected directory.
# This run is a second, independent, purpose-built confirmation of that.

set -uo pipefail

PYTHON="${PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG="retrain_reconstruction.log"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo "Reconstruction re-derivation started: $(date)"
echo "=============================================================="

echo
echo "--- Pre-image: sha256 of every checkpoint.pth (standing order 14) ---"
find TQNet/checkpoints -name checkpoint.pth -exec sha256sum {} \; | sort -k2 > /tmp/ckpt_sha_before.txt
wc -l /tmp/ckpt_sha_before.txt

echo
echo "--- Training: model_id=ETTh1_96_96_reconstruction_v2, cycle=24, seed=2024 ---"
cd TQNet
"$PYTHON" -u run.py \
  --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv \
  --model_id ETTh1_96_96_reconstruction_v2 --model TQNet --data ETTh1 --features M \
  --seq_len 96 --pred_len 96 --enc_in 7 --cycle 24 \
  --train_epochs 30 --patience 5 --dropout 0.5 --accelerator cpu \
  --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed 2024
TRAIN_STATUS=$?
cd "$REPO_ROOT"
if [ $TRAIN_STATUS -ne 0 ]; then
  echo "!!! Training failed. Stopping. !!!"
  exit 1
fi

echo
echo "--- Post-image: sha256 of every checkpoint.pth ---"
find TQNet/checkpoints -name checkpoint.pth -exec sha256sum {} \; | sort -k2 > /tmp/ckpt_sha_after.txt

echo
echo "--- Diff (must be additions-only -- one new line, nothing changed or removed) ---"
diff /tmp/ckpt_sha_before.txt /tmp/ckpt_sha_after.txt
echo "(diff exit code: $?; 1 just means differences exist, check by eye that they are additions only)"

echo
echo "--- Validation metrics ---"
"$PYTHON" tools/validation_metrics.py

echo
echo "--- Read back the new checkpoint's val_MSE and compare ---"
"$PYTHON" - <<'EOF'
import json, glob
paths = glob.glob("results/validation/ETTh1_96_96_reconstruction_v2_*seed2024.json")
if not paths:
    print("!!! No result file found for ETTh1_96_96_reconstruction_v2 seed2024 !!!")
else:
    p = json.load(open(paths[0]))
    v2 = p["val_MSE"]
    lost_anchor = 0.6712632722155959
    adopted_anchor = 0.6869550701723053
    print(f"file: {paths[0]}")
    print(f"v2 val_MSE = {v2!r}  n_params = {p['n_params']!r}")
    print(f"diff vs LOST original anchor    ({lost_anchor!r}) = {v2 - lost_anchor!r}")
    print(f"diff vs ADOPTED anchor          ({adopted_anchor!r}) = {v2 - adopted_anchor!r}")
EOF

echo
echo "--- Final pytest ---"
"$PYTHON" -m pytest -q

echo
echo "--- git status (read-only) ---"
git --no-optional-locks status --porcelain

echo
echo "=============================================================="
echo "Reconstruction re-derivation finished: $(date)"
echo "=============================================================="
echo "NOTE: this does NOT overwrite the protected checkpoint directory."
echo "It creates a new, independently-verifiable checkpoint under a distinct model_id."
echo "Whether/how to formally supersede the adopted anchor with this one is a Gate-1"
echo "decision, not something this script does automatically."
