#!/usr/bin/env bash
# Ablation matrix for the writeup.
# Every run logs to results/runs/$DATASET/$EXP/.
set -euo pipefail
DATASET="${1:?usage: $0 <dataset_name>}"

cd TAPE
export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=False

# Feature ablations: which input gives the biggest lift?
for feat in ogb TA E P TA_P_E; do
  for gnn in GCN SAGE; do
    for seed in 0 1 2 3; do
      python -m core.trainGNN dataset "$DATASET" \
          gnn.model.name "$gnn" \
          gnn.train.feature_type "$feat" \
          seed "$seed"
    done
  done
done
