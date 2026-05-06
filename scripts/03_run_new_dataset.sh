#!/usr/bin/env bash
# Apply the TAPE pipeline to a new dataset (we use goodreads_children).
#
# Workflow:
#   1) new_dataset/prep/build_<DATASET>.py  -> produces TAPE-compatible files
#   2) llm_explanations/generate.py         -> produces explanation JSONs
#   3) python -m core.trainLM   dataset $DATASET lm.train.use_gpt True
#   4) python -m core.trainEnsemble dataset $DATASET gnn.train.feature_type TA_P_E
set -euo pipefail
DATASET="${1:?usage: $0 <dataset_name>}"

cd TAPE
export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=False

python -m core.trainLM      dataset "$DATASET" seed 0
python -m core.trainLM      dataset "$DATASET" seed 0 lm.train.use_gpt True

for gnn in MLP GCN SAGE; do
  python -m core.trainEnsemble dataset "$DATASET" gnn.model.name "$gnn" \
      gnn.train.feature_type TA_P_E
done
