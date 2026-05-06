#!/usr/bin/env bash
# A1 -- drop-one ensemble ablation.
#
# Re-runs the ensemble trainer with each pair of views (TA+P, TA+E, P+E) and
# the full triple (TA_P_E) so we can read off how much accuracy each view
# contributes inside the ensemble. Reuses the LM .emb files that
# scripts/03_run_new_dataset.sh produced; no new LM fine-tunes here.
#
# Outputs:
#   results/runs/<DATASET>/seed_<S>/<feat>_<gnn>.json
# (consumed by scripts/aggregate_results.py)
#
# Wall-clock: about 30-60 min on Colab T4 for goodreads_children (12 cells
# x ~1 min each, plus the per-view inner GNN training that ensemble_trainer
# runs internally).
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:?usage: $0 <dataset_name>}"
SEEDS=(0 1 2 3)
GNNS=(MLP GCN SAGE)
ENSEMBLE_VARIANTS=(TA_P TA_E P_E TA_P_E)

export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=False

for SEED in "${SEEDS[@]}"; do
    for GNN in "${GNNS[@]}"; do
        for FEAT in "${ENSEMBLE_VARIANTS[@]}"; do
            python scripts/_run_gnn_cell.py \
                --dataset "${DATASET}" --seed "${SEED}" \
                --gnn "${GNN}" --feature "${FEAT}"
        done
    done
done
