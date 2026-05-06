#!/usr/bin/env bash
# Run the full TAPE pipeline on a new dataset.
# This is the script that, given a built TAG and its LLM explanations, produces
# the headline 60-cell matrix in the report:
#     5 features (ogb, TA, P, E, TA_P_E) x 3 GNNs (MLP, GCN, SAGE) x 4 seeds.
#
# Prerequisites:
#   1. new_dataset/data/<DATASET>/{graph.pt, node_texts.jsonl, labels.txt}
#      (produce with new_dataset/prep/build_<DATASET>.py)
#   2. TAPE/gpt_responses/<DATASET>/<i>.json + TAPE/gpt_preds/<DATASET>.csv
#      (produce with llm_explanations/generate.py)
#
# Wall-clock budget on a single A100 GPU:
#   - LM fine-tunes: 2 fine-tunes x 4 seeds ~= 12 GPU-hours total
#   - GNN sweep:     60 runs x ~1 min       ~= 1 hour total
# On a Colab T4 the LM step is ~1.5 h per seed per view; we cache .emb files
# on Drive and resume across sessions via notebooks/e2_goodreads.ipynb.
#
# Outputs:
#   - LM artifacts: TAPE/prt_lm/<DATASET>{,2}/<lm>/<lm>-seed<S>.{ckpt,emb}
#   - GNN cells:    results/runs/<DATASET>/seed_<S>/<feat>_<gnn>.json
#                   (consumed by scripts/aggregate_results.py --dataset <DATASET>)
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:?usage: $0 <dataset_name>}"
SEEDS=(0 1 2 3)
GNNS=(MLP GCN SAGE)
SINGLE_FEATURES=(ogb TA P E)

export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=False

# 1. Two LM fine-tunes per seed (TA on raw text, E on LLM explanations).
for SEED in "${SEEDS[@]}"; do
    (cd TAPE && python -m core.trainLM dataset "${DATASET}" seed "${SEED}")
    (cd TAPE && python -m core.trainLM dataset "${DATASET}" seed "${SEED}" lm.train.use_gpt True)
done

# 2. GNN sweep. The wrapper at scripts/_run_gnn_cell.py runs the upstream
#    trainer and dumps a JSON record at the path aggregate_results.py expects.
for SEED in "${SEEDS[@]}"; do
    for GNN in "${GNNS[@]}"; do
        for FEAT in "${SINGLE_FEATURES[@]}"; do
            python scripts/_run_gnn_cell.py \
                --dataset "${DATASET}" --seed "${SEED}" \
                --gnn "${GNN}" --feature "${FEAT}"
        done
        python scripts/_run_gnn_cell.py \
            --dataset "${DATASET}" --seed "${SEED}" \
            --gnn "${GNN}" --feature TA_P_E
    done
done
