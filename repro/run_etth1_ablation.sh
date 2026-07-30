#!/bin/bash
# Does the Temporal Query mechanism do anything on ETTh1?
#
# The paper never answers this. Every ablation and integration study in it runs on
# Electricity, PEMS03 and PEMS04 only -- all with more than 100 channels. ETTh1 has 7,
# so its channel attention map is 7x7, and TQ's contribution at that width is
# unmeasured. Two runs of ~30 seconds each settle it.
#
# The switches exist in upstream's source as plain booleans at models/TQNet.py:17-18,
# but upstream expects you to *edit the file* between variants, which is not
# reproducible. They are exposed here as --use_tq / --channel_aggre; the defaults
# reproduce the published model exactly (verified: 661,640 parameters either way).
#
# Non-default runs get a _tq<N>ca<N> suffix in the checkpoint path, the results
# directory and the result_ours.txt label, so they cannot collide with the
# reconstruction.
#
# Run from the repository root:
#
#   bash repro/run_etth1_ablation.sh
#   python3 tools/collect_results.py && python3 tools/make_report.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/TQNet"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [ ! -f ./dataset/ETTh1.csv ]; then
  echo "ETTh1.csv is missing. Run:  python3 tools/get_data.py" >&2
  exit 1
fi

# label : --use_tq : --channel_aggre
#   tq0ca1  TQ removed, attention kept -> plain channel self-attention, iTransformer-like.
#           The gap from the reconstruction IS TQ's entire contribution on ETTh1.
#   tq0ca0  both removed -> pure MLP. The floor: how much of 0.3712 needs neither mechanism.
# Add "tq1ca0:1:0" for the fourth cell (TQ kept, attention removed -> a channel
# identifier) if you want the full 2x2 rather than the two decisive runs.
VARIANTS=(
  "tq0ca1:0:1"
  "tq0ca0:0:0"
)

SEQ_LEN=${SEQ_LEN:-96}
PRED_LEN=${PRED_LEN:-96}
SEED=${SEED:-2024}
ACCELERATOR=${ACCELERATOR:-auto}

for variant in "${VARIANTS[@]}"; do
  IFS=':' read -r label use_tq channel_aggre <<< "$variant"

  echo "=============================================================="
  echo "ETTh1 ablation ${label}: --use_tq ${use_tq} --channel_aggre ${channel_aggre}"
  echo "=============================================================="

  start=$(date +%s)

  # Every flag other than --use_tq / --channel_aggre is copied verbatim from
  # scripts/TQNet/etth1.sh, so these runs are comparable to the reconstruction
  # by construction rather than by argument.
  python3 -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${SEQ_LEN}_${PRED_LEN} \
    --model TQNet \
    --data ETTh1 \
    --features M \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --enc_in 7 \
    --cycle 24 \
    --train_epochs 30 \
    --patience 5 \
    --dropout 0.5 \
    --use_tq ${use_tq} \
    --channel_aggre ${channel_aggre} \
    --accelerator ${ACCELERATOR} \
    --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed ${SEED}

  echo "${label} wall-clock: $(( $(date +%s) - start ))s"
  echo
done

echo "=============================================================="
echo "Done. Compare against the reconstruction cell (published: mse 0.3712, mae 0.3928):"
echo "  python3 tools/collect_results.py"
echo "  python3 tools/make_report.py"
echo "=============================================================="
