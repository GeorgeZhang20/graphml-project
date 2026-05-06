#!/usr/bin/env bash
# A5 / sec. 4.5 -- frozen vs fine-tuned LM for the E view.
#
# Builds a frozen-DeBERTa CLS embedding from the explanation text, drops it
# at the same path the upstream GNN trainer expects (TAPE/prt_lm/<ds>2/), and
# runs the GNN sweep against it. The fine-tuned-E rows (the comparison) come
# from scripts/03_run_new_dataset.sh, so make sure that pipeline has finished
# first.
#
# Implementation trick: the GNN trainer always reads E features from the
# fixed path TAPE/prt_lm/<ds>2/<lm>/<lm>.emb. Instead of patching the upstream
# trainer, we
#   1. write the frozen embedding to a sibling tree (...2_frozen/),
#   2. symlink the canonical path -> the frozen tree, run the sweep,
#   3. restore the original .emb so the existing fine-tuned numbers stay
#      reproducible (a trap fires on early exit too).
#
# Wall-clock: ~20 min on a single A100 to embed 76k nodes through DeBERTa-base,
# plus 12 GNN cells (~1 min each).
#
# Outputs:
#   results/runs/<DATASET>_a5frozen/seed_<S>/E_<gnn>.json   (frozen E rows)
#   results/<DATASET>_a5_frozen_vs_finetuned.csv            (merged frozen + fine-tuned)
#   results/figures/full_76k/a5_frozen_vs_finetuned.{png,pdf}
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

DATASET="${1:-goodreads_children}"
# Default node count matches what build_goodreads_children.py emits on the
# full CS-TAG Children dump. Override via positional arg 2 if you've subsampled.
N_NODES="${2:-76875}"

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

# 2. Stash the fine-tuned canonical .emb and swap in the frozen one.
[[ -f "${CANON_EMB}" ]] && mv "${CANON_EMB}" "${CANON_EMB}.finetuned.bak"
ln -sf "${PROJECT_ROOT}/${FROZEN_EMB}" "${CANON_EMB}"

cleanup() {
    rm -f "${CANON_EMB}"
    [[ -f "${CANON_EMB}.finetuned.bak" ]] && mv "${CANON_EMB}.finetuned.bak" "${CANON_EMB}"
}
trap cleanup EXIT

# 3. Run the GNN sweep against the swapped .emb. We tag this as a separate
#    "dataset" slug so the JSON outputs land in their own subtree and don't
#    clobber the fine-tuned rows already in results/runs/<DATASET>/.
export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=False
for SEED in 0 1 2 3; do
    for GNN in MLP GCN SAGE; do
        python scripts/_run_gnn_cell.py \
            --dataset "${DATASET}" --seed "${SEED}" \
            --gnn "${GNN}" --feature E \
            --out_dir "results/runs/${DATASET}_a5frozen/seed_${SEED}"
    done
done

cleanup
trap - EXIT

# 4. Merge the frozen E rows we just produced with the fine-tuned E rows from
#    Phase 3 into one paper-ready CSV.
python - <<PY
import json
from pathlib import Path

import pandas as pd

def collect(runs_dir: Path, lm_status: str):
    rows = []
    for jp in sorted(runs_dir.glob("seed_*/E_*.json")):
        with jp.open() as f:
            rec = json.load(f)
        if rec.get("returncode") != 0 or rec.get("test_acc") is None:
            continue
        rows.append({
            "dataset": "${DATASET}",
            "gnn": rec["gnn"],
            "lm_status": lm_status,
            "seed": rec["seed"],
            "test_acc": rec["test_acc"],
        })
    return rows

frozen = collect(Path("results/runs/${DATASET}_a5frozen"), "frozen")
finetuned = collect(Path("results/runs/${DATASET}"), "finetuned")
out = pd.DataFrame(frozen + finetuned)
out_path = Path("results/${DATASET}_a5_frozen_vs_finetuned.csv")
out.to_csv(out_path, index=False)
print(f"[A5] wrote {out_path} ({len(out)} rows)")
PY

# 5. Render Figure 4. Output stays inside the public results tree.
python scripts/plot_a5.py \
    --csv "results/${DATASET}_a5_frozen_vs_finetuned.csv" \
    --out "results/figures/full_76k/a5_frozen_vs_finetuned.png"

echo "[A5] done."
