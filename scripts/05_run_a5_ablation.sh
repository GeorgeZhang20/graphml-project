#!/usr/bin/env bash
# A5 / sec. 4.5 -- frozen vs fine-tuned LM for the E view.
#
# Runs three GNN sweeps against a frozen-DeBERTa E embedding so we can
# compare against the fine-tuned E rows that Phase 3 already produced.
# Saves results in results/runs/a5_frozen/<dataset>_<gnn>_seed<S>/, then
# aggregates with the existing E rows and re-renders Figure 4.
#
# Implementation trick: the GNN trainer always reads E features from the
# fixed path TAPE/prt_lm/<ds>2/<lm>/<lm>.emb. So instead of patching the
# upstream trainer, we:
#   1. write the frozen embedding to a sibling tree (...2_frozen/)
#   2. symlink the canonical path -> the frozen tree, run the sweep
#   3. restore the original symlink so the existing fine-tuned E numbers
#      stay reproducible.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-goodreads_children}"
N_NODES="${2:-76349}"

LM_NAME="microsoft/deberta-base"
CANON_DIR="TAPE/prt_lm/${DATASET}2/${LM_NAME%/*}"
CANON_EMB="${CANON_DIR}/${LM_NAME##*/}.emb"
FROZEN_DIR="TAPE/prt_lm/${DATASET}2_frozen/${LM_NAME%/*}"
FROZEN_EMB="${FROZEN_DIR}/${LM_NAME##*/}.emb"
GPT_DIR="TAPE/gpt_responses/${DATASET}"

# 1. Build the frozen E embedding (one-shot, no fine-tune).
mkdir -p "${FROZEN_DIR}"
if [[ ! -f "${FROZEN_EMB}" ]]; then
    python scripts/05_frozen_deberta_emb.py \
        --dataset "${DATASET}" \
        --n_nodes "${N_NODES}" \
        --gpt_responses "${GPT_DIR}" \
        --use explanation \
        --out "${FROZEN_EMB}"
else
    echo "[A5] reusing existing ${FROZEN_EMB}"
fi

# 2. Stash the fine-tuned canonical emb and swap in the frozen one.
[[ -f "${CANON_EMB}" ]] && mv "${CANON_EMB}" "${CANON_EMB}.finetuned.bak"
ln -sf "$(realpath "${FROZEN_EMB}")" "${CANON_EMB}"

cleanup() {
    rm -f "${CANON_EMB}"
    [[ -f "${CANON_EMB}.finetuned.bak" ]] && mv "${CANON_EMB}.finetuned.bak" "${CANON_EMB}"
}
trap cleanup EXIT

# 3. Run the GNN sweep against the swapped emb.
for GNN in MLP GCN SAGE; do
    for SEED in 0 1 2 3; do
        OUT="results/runs/a5_frozen/${DATASET}_${GNN}_seed${SEED}"
        mkdir -p "${OUT}"
        python -m core.trainGNN \
            dataset "${DATASET}" \
            gnn.model.name "${GNN}" \
            gnn.train.feature_type "E" \
            lm.train.use_gpt True \
            lm.model.name "${LM_NAME}" \
            seed "${SEED}" \
            output_dir "${OUT}/"
    done
done

cleanup
trap - EXIT

# 4. Aggregate frozen rows with the fine-tuned E rows from Phase 3.
python scripts/aggregate_results.py \
    --pattern "results/runs/a5_frozen/${DATASET}_*/metrics.json" \
    --tag frozen \
    --out "results/${DATASET}_a5_frozen_vs_finetuned.csv"

python scripts/aggregate_results.py \
    --pattern "results/runs/${DATASET}/seed_*/E_*.json" \
    --tag finetuned \
    --append "results/${DATASET}_a5_frozen_vs_finetuned.csv"

# 5. Render Figure 4.
python scripts/plot_a5.py \
    --csv "results/${DATASET}_a5_frozen_vs_finetuned.csv" \
    --out "paper/figures/a5_frozen_vs_finetuned.png"

echo "[A5] done."
