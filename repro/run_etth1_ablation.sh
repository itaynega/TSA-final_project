#!/bin/bash
# Does the Temporal Query mechanism do anything on ETTh1?
#
# The paper never answers this. Every ablation and integration study in the paper
# runs on Electricity, PEMS03 and PEMS04 only -- all with >100 channels. ETTh1 has 7,
# so the channel attention map is 7x7 and TQ's contribution there is unmeasured.
#
# Run this from the root of a TQNet clone that has repro/tqnet-ablation-flags.patch applied:
#
#   git apply /path/to/TSA-final_project/repro/tqnet-ablation-flags.patch
#   cp result.txt result_authors_reference.txt      # exp_main.py APPENDS to result.txt
#   bash /path/to/TSA-final_project/repro/run_etth1_ablation.sh
#
# Everything except --use_tq / --channel_aggre is copied verbatim from
# scripts/TQNet/etth1.sh, so these runs are comparable to the reproduction by construction.

set -euo pipefail

# variant label : --use_tq : --channel_aggre
#   tq0ca1  TQ removed, attention kept  -> becomes plain channel self-attention (iTransformer-like).
#                                          The difference from the baseline IS the TQ contribution.
#   tq0ca0  both removed                -> pure MLP. The floor: how much of 0.3712 needs neither.
# Add "tq1ca0:1:0" below for the fourth cell (TQ kept, attention removed -> channel identifier)
# if you want the complete 2x2 rather than the two decisive runs.
VARIANTS=(
  "tq0ca1:0:1"
  "tq0ca0:0:0"
)

SEQ_LEN=96
PRED_LEN=96
SEED=2024

for variant in "${VARIANTS[@]}"; do
  IFS=':' read -r label use_tq channel_aggre <<< "$variant"

  echo "=============================================================="
  echo "ETTh1 ablation ${label}: --use_tq ${use_tq} --channel_aggre ${channel_aggre}"
  echo "=============================================================="

  start=$(date +%s)

  python -u run.py \
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
    --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed ${SEED}

  echo "${label} wall-clock: $(( $(date +%s) - start ))s"
done

echo
echo "Done. New lines are appended to ./result.txt, labelled with the _tq<N>ca<N> suffix."
echo "Compare against the baseline cell ETTh1_96_96 ... _seed2024 (mse 0.3712, mae 0.3928)."
