# LLM-as-Feature on Text-Attributed Graphs: a Cross-Domain Replication of TAPE

Code for a class project (COMP 459 / Machine Learning with Graphs, Spring 2026)
re-running [TAPE (He et al., ICLR 2024)](https://arxiv.org/abs/2305.19523) on
**Goodreads-Children**, a non-academic text-attributed graph (TAG) where the
node text is a publisher's book description rather than a paper abstract. We
replicate the headline ogbn-arxiv numbers, then test whether TAPE's per-view
ranking transfers off citation graphs.

**Authors:** George Zhang, Haoran Sun, Yiwen Zhu (Rice University).

This repo is a fork of the upstream TAPE code with a small set of additions
(new-dataset build, our LLM-call wrapper, plotting and aggregation utilities).
Upstream code lives untouched under `TAPE/`; everything we wrote sits in
sibling top-level folders.

## Headline findings

We frame three claims (each backed by a numbered table or figure in the report):

- **C1.** The full `TA_P_E` ensemble lifts shallow node features by **8 to 16 pp**
  on Goodreads-Children across MLP, GCN, and GraphSAGE — the same magnitude
  TAPE reports on academic graphs. The recipe transfers at the headline level.
- **C2.** The strongest singleton view of TAPE **flips between domains.** On
  Goodreads, fine-tuned `TA` (raw title+description) beats the
  LLM-explanation view `E` on every backbone we tried. On TAPE's four academic
  graphs the ordering is `E ≥ TA`. The flip is consistent across MLP / GCN /
  GraphSAGE and survives a drop-one ablation.
- **C3.** The marginal value of stacking the explanation branch on top of
  fine-tuned `TA` collapses to **0.5–1.9 pp** on Goodreads, vs ~3–5 pp on
  academic graphs. Frozen-vs-fine-tuned DeBERTa on `E` shows the same shape
  (1.3–1.9 pp gap on Goodreads, see
  `results/figures/full_76k/a5_frozen_vs_finetuned.png`).

The plausible reading is that publisher's descriptions already use the words
the genre label uses (e.g. "picture book," "early-reader chapter book," "YA
fantasy"), so the LLM's explanation text adds little vocabulary that `TA` did
not already see. On dense, jargon-heavy paper abstracts the situation is
reversed.

## Repository layout

```
graphml-project/
├── TAPE/                  # upstream TAPE code (mostly untouched)
│   ├── core/              # LM, GNN, ensemble trainers + dataset registry
│   ├── gpt_preds/         # small cached top-k LLM label preds (kept)
│   ├── dataset/           # raw graph data; gitignored
│   ├── gpt_responses/     # per-node LLM JSONs; gitignored (large)
│   └── prt_lm/            # DeBERTa fine-tune outputs; gitignored
├── new_dataset/
│   ├── prep/              # build_<DATASET>.py — TAPE-compatible artifacts
│   └── data/              # outputs of build_*.py; gitignored
├── llm_explanations/      # per-node LLM call wrapper (resumable)
│   ├── generate.py        # main entry point
│   └── prompts/           # zero-shot prompt templates
├── scripts/               # shell + python helpers (env, sweeps, plots)
├── data/                  # paper Table 2 reference numbers (CSV)
├── results/
│   ├── *_long.csv         # per-seed results (one row per (seed, feature, gnn))
│   ├── *_summary.csv      # mean ± std per (feature, gnn)
│   └── figures/full_76k/  # rendered figures used in the report
├── notebooks/
│   └── e2_goodreads.ipynb # per-seed Colab runner with Drive checkpointing
├── requirements.txt
└── README.md
```

## Setup

```bash
bash scripts/00_setup_env.sh        # creates conda env `tape-proj`, ~5 min
conda activate tape-proj
bash scripts/01_smoke_test.sh       # GCN-on-Cora smoke test, ~3 sec on CPU
```

`requirements.txt` pins `torch==2.2.2` (CPU). On Colab/CUDA reinstall torch
from the matching CUDA wheel — the first cell of
`notebooks/e2_goodreads.ipynb` shows the exact command.

## Reproducing the results

The pipeline is split into four phases; each phase is a single shell script.
**Each phase's prerequisites are listed at the top of the script** (also
summarised below). Outputs land in standard places that the next phase
reads from, so you can run a phase, inspect its output, and then move on.

The wall-clock numbers are for a single A100; on a Colab T4 the LM
fine-tunes are roughly 3x slower. We did all of our runs through
`notebooks/e2_goodreads.ipynb` because it caches `.emb` and `.ckpt`
artifacts on Drive and resumes cleanly across Colab sessions; the shell
scripts below mirror that pipeline for non-Colab use.

### Phase 1 — replicate TAPE on ogbn-arxiv (~6–10 h on A100, ~24 h on Colab T4)

This is the validation anchor. If our pipeline's `TA_P_E` numbers on
ogbn-arxiv are within ±1% of TAPE Table 2, the rest of the runs are
trustworthy.

**Manual prereq:** TAPE upstream does not ship the ogbn-arxiv text or the
GPT-3.5 explanations. Follow `TAPE/UPSTREAM_README.md` §1 to download:

- `TAPE/dataset/ogbn_arxiv_orig/titleabs.tsv` (raw text)
- `TAPE/gpt_responses/ogbn-arxiv/*.json` (cached GPT-3.5 explanations)

Then:

```bash
bash scripts/02_reproduce_arxiv.sh
# or, if you already have the LM .emb files (e.g. from the authors' checkpoints):
bash scripts/02_reproduce_arxiv.sh --gnn-only
```

This loops over 4 seeds × 4 backbones (MLP, GCN, SAGE, RevGAT). Each backbone
is invoked with `gnn.train.feature_type TA_P_E`, which triggers the upstream
ensemble trainer (3 GNNs trained, logits averaged). RevGAT uses TAPE's
upstream hyperparameters (`lr=0.002 dropout=0.5`).

### Phase 2 — build Goodreads-Children + LLM call (~$15, resumable)

```bash
# Builds graph.pt + node_texts.jsonl + labels.txt under new_dataset/data/.
# Auto-downloads the CS-TAG Children.csv from HuggingFace if needed.
python new_dataset/prep/build_goodreads_children.py

# Costs ~$15 on gpt-4o-mini for the full 76,875-node dump. Resumable:
# re-running picks up where it left off, so interrupting + relaunching is safe.
export OPENAI_API_KEY=sk-...
python llm_explanations/generate.py \
    --dataset goodreads_children \
    --node_texts new_dataset/data/goodreads_children/node_texts.jsonl \
    --labels    new_dataset/data/goodreads_children/labels.txt \
    --model     gpt-4o-mini
```

Outputs: `TAPE/gpt_responses/goodreads_children/<i>.json` (one per node) and
`TAPE/gpt_preds/goodreads_children.csv` (top-5 label IDs per node).

### Phase 3 — DeBERTa fine-tunes + 60-cell GNN matrix (~5 h on A100, ~13 h on Colab T4)

```bash
bash scripts/03_run_new_dataset.sh goodreads_children
```

Two DeBERTa fine-tunes per seed (TA on raw text, E on explanation text), then
the full sweep:

- 5 features (`ogb`, `TA`, `P`, `E`, `TA_P_E`) × 3 GNNs (`MLP`, `GCN`, `SAGE`)
  × 4 seeds = **60 runs total**.

Single-feature cells go through `core.trainGNN`; the `TA_P_E` cells go through
`core.trainEnsemble`, which trains a per-view GNN and averages their logits.
Both invocations are wrapped by `scripts/_run_gnn_cell.py`, which dumps a
JSON record per cell at `results/runs/<DATASET>/seed_<S>/<feat>_<gnn>.json`
(the layout that `scripts/aggregate_results.py` reads).

### Phase 4 — drop-one (A1) and frozen-LM (A5) ablations

A1 reuses the LM `.emb` files Phase 3 produced; A5 builds a fresh frozen
embedding for the E view, swaps it into the canonical path via a symlink,
and restores the original at exit (a `trap` handles early termination too).

```bash
# A1: TA+P, TA+E, P+E, TA_P_E across 3 GNNs x 4 seeds.
# Outputs: results/runs/<DATASET>/seed_<S>/{TA_P,TA_E,P_E,TA_P_E}_<gnn>.json
bash scripts/04_ablations.sh goodreads_children

# A5: frozen vs fine-tuned DeBERTa for E, 3 GNNs x 4 seeds.
# Outputs: results/runs/<DATASET>_a5frozen/seed_<S>/E_<gnn>.json,
#          results/<DATASET>_a5_frozen_vs_finetuned.csv (merged with Phase 3),
#          results/figures/full_76k/a5_frozen_vs_finetuned.{png,pdf}
bash scripts/05_run_a5_ablation.sh goodreads_children
```

## Aggregating results and rendering figures

After every seed finishes (Phase 3 dumps per-seed JSONs into
`results/runs/<dataset>/seed_*/`), aggregate them into one tidy CSV and
render the figures:

```bash
python scripts/aggregate_results.py --dataset goodreads_children
python scripts/make_plots.py        --dataset goodreads_children
python scripts/plot_gnn_curves.py   --dataset goodreads_children   # optional: training curves from logs
```

The A5 wrapper aggregates and re-renders Figure 4 itself, so you don't need
to call `plot_a5.py` by hand unless you want to regenerate the figure from a
hand-edited CSV:

```bash
python scripts/plot_a5.py \
    --csv results/goodreads_children_a5_frozen_vs_finetuned.csv \
    --out results/figures/full_76k/a5_frozen_vs_finetuned.png
```

`scripts/make_plots.py` handles the headline figures (heatmap, singleton bars,
H3 reversal, cross-dataset comparison, Pareto frontier). All output figures
land in `results/figures/full_76k/`.

## Hardware notes

- **Local Mac (Intel, 16 GB, no CUDA).** Fine for `00_setup` / `01_smoke` and
  for all post-hoc plotting and aggregation. Not enough for any DeBERTa
  fine-tune.
- **Colab T4 / Kaggle GPU.** Required for the LM fine-tunes (TA + E per seed).
  Each fine-tune is ~1.5 h on a T4. `notebooks/e2_goodreads.ipynb` caches
  `.emb` and `.ckpt` artifacts on Drive so a re-attached Colab session picks
  up where the previous one left off.
- **A100 / H100.** Same scripts; LM fine-tunes drop to ~30 min each.

## Implementation deviations from the released TAPE code

We document three differences from the upstream pipeline. Each is conservative
for our headline finding (C2: `TA > E` on Goodreads).

- **LLM:** `gpt-4o-mini` instead of `gpt-3.5-turbo` (deprecated). A stronger
  LLM can only narrow the C2 gap by improving the explanation view, so this
  is a stress test against the finding rather than for it.
- **Precision:** fp32 fine-tuning of DeBERTa. HuggingFace's mixed-precision
  path overflows on long Goodreads descriptions; fp32 if anything helps the
  E branch.
- **Backbones for Goodreads:** MLP, GCN, GraphSAGE; no RevGAT (DGL install
  issues on Colab). The C2 reversal is verified across all three backbones we
  do run. The ogbn-arxiv replication in Phase 1 still includes RevGAT.

## Acknowledgments

This project builds directly on the released TAPE code. The full upstream
README is preserved at `TAPE/UPSTREAM_README.md`, and the upstream license
is at `TAPE/LICENSE`.

```bibtex
@inproceedings{he2024harnessing,
  title  = {Harnessing Explanations: {LLM}-to-{LM} Interpreter for Enhanced
            Text-Attributed Graph Representation Learning},
  author = {He, Xiaoxin and Bresson, Xavier and Laurent, Thomas and
            Perold, Adam and LeCun, Yann and Hooi, Bryan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year   = {2024}
}
```

The Goodreads-Children TAG comes from
[CS-TAG](https://github.com/sktsherlock/TAG-Benchmark); the underlying
Goodreads dump is from Wan & McAuley
[(RecSys 2018)](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home).
