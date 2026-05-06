#!/usr/bin/env bash
# Replicate TAPE on ogbn-arxiv (validation anchor for the rest of the pipeline).
# 4 seeds x 4 backbones (MLP, GCN, SAGE, RevGAT) using the upstream cached
# gpt_responses + the upstream LM trainer.
#
# Prerequisites (one-time, manual; see TAPE/UPSTREAM_README.md sec. 1 for the
# canonical download links):
#   1. TAPE/dataset/ogbn_arxiv_orig/titleabs.tsv  -- raw text
#   2. TAPE/gpt_responses/ogbn-arxiv/*.json       -- GPT-3.5 explanations
#
# Wall-clock budget: ~3-4 h on a single A100 (most of it is the 2 LM fine-tunes
# x 4 seeds). Each GNN cell is ~1 min once the .emb files exist.
set -euo pipefail
cd "$(dirname "$0")/.."

GNN_ONLY=0
if [[ "${1:-}" == "--gnn-only" ]]; then
    GNN_ONLY=1
fi

DATASET=ogbn-arxiv
SEEDS=(0 1 2 3)
GNNS=(MLP GCN SAGE RevGAT)

export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=False

if [[ "${GNN_ONLY}" -eq 0 ]]; then
    for SEED in "${SEEDS[@]}"; do
        # TA: fine-tune DeBERTa on raw title+abstract
        (cd TAPE && python -m core.trainLM dataset "${DATASET}" seed "${SEED}")
        # E:  fine-tune DeBERTa on the LLM explanation text
        (cd TAPE && python -m core.trainLM dataset "${DATASET}" seed "${SEED}" lm.train.use_gpt True)
    done
fi

for GNN in "${GNNS[@]}"; do
    EXTRA_ARGS=()
    if [[ "${GNN}" == "RevGAT" ]]; then
        # The hyperparameters TAPE upstream uses for RevGAT.
        EXTRA_ARGS=(--extra gnn.train.lr 0.002 gnn.train.dropout 0.5)
    fi
    for SEED in "${SEEDS[@]}"; do
        # TA_P_E ensemble: trains TA-GNN + P-GNN + E-GNN, averages logits.
        # The wrapper drops a JSON record at results/runs/<dataset>/seed_<S>/.
        python scripts/_run_gnn_cell.py \
            --dataset "${DATASET}" --seed "${SEED}" \
            --gnn "${GNN}" --feature TA_P_E "${EXTRA_ARGS[@]}"
    done
done
