#!/usr/bin/env bash
# Reproduce TAPE on ogbn-arxiv using the upstream GPT-3.5 cached responses
# and pre-trained checkpoints (download manually first - see TAPE/UPSTREAM_README.md).
#
# Prereqs:
#   1. Download dataset/ogbn_arxiv_orig (titleabs.tsv) into TAPE/dataset/ogbn_arxiv_orig/
#   2. Download gpt_responses/ogbn-arxiv into TAPE/gpt_responses/ogbn-arxiv/
#   3. (Optional) Download TAPE author checkpoints to skip LM fine-tune.
#
# Recommended runtime: Colab T4 ~3-4 hours end-to-end, or CPU for GNN-only path.
set -euo pipefail
cd TAPE

export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=False

# 1) Fine-tune LM on raw title+abstract
python -m core.trainLM dataset ogbn-arxiv seed 0

# 2) Fine-tune LM on GPT explanations
python -m core.trainLM dataset ogbn-arxiv seed 0 lm.train.use_gpt True

# 3) Train GNN ensemble using TA + P + E features (the headline TAPE config)
python -m core.trainEnsemble dataset ogbn-arxiv gnn.model.name GCN \
    gnn.train.feature_type TA_P_E
