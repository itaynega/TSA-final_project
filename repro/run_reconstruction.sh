#!/bin/bash
# Stage 1: reproduce TQNet's published ETTh1 result at horizon 96.
#
# Target cell:  ETTh1, multivariate (7 -> 7), L=96 -> H=96, seed 2024
# Published:    mse 0.3712165653705597, mae 0.3928201496601105
#               (TQNet/result_authors_reference.txt, the authors' own run)
#
# Every flag below is copied verbatim from TQNet/scripts/TQNet/etth1.sh, except that
# the upstream script loops over pred_len 96 192 336 720 and this runs only 96. Nothing
# is tuned here. The whole point of this cell is that the authors pinned every
# hyperparameter, so there is nothing left to guess -- and any number we get that
# differs from theirs is a fact about our environment, not about our choices.
#
# Run from the repository root:
#
#   bash repro/run_reconstruction.sh                 # the target cell, seed 2024
#   SEEDS="2024 2025 2026" bash repro/run_reconstruction.sh
#   PRED_LENS="96 192 336 720" bash repro/run_reconstruction.sh
#
# Outputs, per run:
#   TQNet/checkpoints/<setting>/checkpoint.pth   best-validation weights
#   TQNet/results/<setting>/pred.npy, true.npy   test windows, for the report
#   TQNet/result_ours.txt                        appended one-line summary
#
# Then:  python3 tools/collect_results.py && python3 tools/make_report.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/TQNet"

SEQ_LEN=${SEQ_LEN:-96}
PRED_LENS=${PRED_LENS:-96}
SEEDS=${SEEDS:-2024}
ACCELERATOR=${ACCELERATOR:-auto}

# matplotlib writes a font cache on first import; on a locked-down home directory
# that import fails, and it fails inside exp_main rather than anywhere obvious.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [ ! -f ./dataset/ETTh1.csv ]; then
  echo "ETTh1.csv is missing. Run:  python3 tools/get_data.py" >&2
  exit 1
fi

# Guard the authors' reference numbers. Our runs write to result_ours.txt, but a
# clobbered result.txt would destroy the only full-precision copy of the published
# figures, so refuse to start rather than discover it afterwards.
if [ ! -f ./result_authors_reference.txt ]; then
  echo "result_authors_reference.txt is missing -- restore it before running." >&2
  exit 1
fi

for pred_len in ${PRED_LENS}; do
for seed in ${SEEDS}; do

  echo "=============================================================="
  echo "ETTh1  L=${SEQ_LEN} -> H=${pred_len}  seed=${seed}  device=${ACCELERATOR}"
  echo "=============================================================="

  start=$(date +%s)

  python3 -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${SEQ_LEN}_${pred_len} \
    --model TQNet \
    --data ETTh1 \
    --features M \
    --seq_len ${SEQ_LEN} \
    --pred_len ${pred_len} \
    --enc_in 7 \
    --cycle 24 \
    --train_epochs 30 \
    --patience 5 \
    --dropout 0.5 \
    --accelerator ${ACCELERATOR} \
    --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed ${seed}

  echo "wall-clock: $(( $(date +%s) - start ))s"
  echo

done
done

echo "=============================================================="
echo "Done. Ingest and compare with:"
echo "  python3 tools/collect_results.py"
echo "  python3 tools/make_report.py"
echo "=============================================================="
