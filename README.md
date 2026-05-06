# LLM-as-Feature on Text-Attributed Graphs: a Cross-Domain Replication of TAPE

Code for a class project (COMP 459 / Machine Learning with Graphs, Spring 2026)
re-running [TAPE (He et al., ICLR 2024)](https://arxiv.org/abs/2305.19523) on
**Goodreads-Children**, a non-academic text-attributed graph (TAG) where node
text is a publisher's book description rather than a paper abstract. We
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
  on Goodreads-Children across MLP, GCN, and GraphSAGE, the same magnitude
  TAPE reports on academic graphs. The recipe transfers at the headline level.
- **C2.** The strongest singleton view of TAPE **flips between domains.** On
  Goodreads, fine-tuned `TA` (raw title+description) beats the
  LLM-explanation view `E` on every backbone we tried. On TAPE's four academic
  graphs the ordering is `E ≥ TA`. The flip is consistent across MLP / GCN /
  GraphSAGE and survives a drop-one ablation.
- **C3.** The marginal value of stacking the explanation branch on top of
  fine-tuned `TA` collapses to **0.5–1.9 pp** on Goodreads, vs ~3–5 pp on
  academic graphs. Frozen-vs-fine-tuned DeBERTa on `E` shows the same shape
  (1.3–1.9 pp gap on Goodreads, see `results/figures/full_76k/a5_frozen_vs_finetuned.png`).

The plausible reading is that publisher's descriptions already use the words
the genre label uses (e.g. "picture book," "early-reader chapter book," "YA
fantasy"), so the LLM's explanation text adds little vocabulary that `TA`
did not already see. On dense, jargon-heavy paper abstracts the situation
is reversed.

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
│   ├── *_long.csv         # per-seed results
│   ├── *_summary.csv      # mean ± std per (feature, GNN)
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

`requirements.txt` pins torch 2.2.2 (CPU); on Colab/CUDA you'll want to
reinstall torch from the matching CUDA wheel — see the first cell of
`notebooks/e2_goodreads.ipynb` for the exact command.

## Reproducing the results

We split the work into four phases. Each phase is one shell script.

```bash
# Phase 1 — replicate TAPE on ogbn-arxiv (validation anchor, ~3-4 h on Colab T4).
# Within ±1% of the paper's headline number on every backbone.
bash scripts/02_reproduce_arxiv.sh

# Phase 2 — build Goodreads-Children and call the LLM on it (~$15 on
# gpt-4o-mini for ~76k nodes; resumable, see llm_explanations/README.md).
python new_dataset/prep/build_goodreads_children.py
python llm_explanations/generate.py \
    --dataset goodreads_children \
    --node_texts new_dataset/data/goodreads_children/node_texts.jsonl \
    --labels    new_dataset/data/goodreads_children/labels.txt \
    --model     gpt-4o-mini

# Phase 3 — DeBERTa fine-tunes + GNN sweep on Goodreads-Children
# (5 features × 3 GNNs × 4 seeds = 60 runs).
bash scripts/03_run_new_dataset.sh goodreads_children
bash scripts/04_ablations.sh       goodreads_children

# Phase 4 — A5 ablation (frozen vs fine-tuned DeBERTa for the E view).
bash scripts/05_run_a5_ablation.sh goodreads_children
```

For day-to-day work we used `notebooks/e2_goodreads.ipynb` instead of the
shell scripts: it's the same pipeline but with Drive-cached checkpoints so
that interrupted runs resume cleanly across Colab sessions.

## Results and plots

After every seed finishes, dump per-seed JSONs into a single tidy CSV and
re-render the figures:

```bash
python scripts/aggregate_results.py --dataset goodreads_children
python scripts/make_plots.py        --dataset goodreads_children
python scripts/plot_gnn_curves.py   --dataset goodreads_children   # optional
python scripts/plot_a5.py \
    --csv results/goodreads_children_a5_frozen_vs_finetuned.csv \
    --out results/figures/full_76k/a5_frozen_vs_finetuned.png
```

`scripts/make_plots.py` takes care of the headline figures (heatmap,
singleton bars, H3 reversal, cross-dataset comparison, Pareto frontier).
All output figures land in `results/figures/full_76k/`.

## Hardware notes

- **Local Mac (Intel, 16 GB, no CUDA).** Fine for GNN training on
  Cora / ogbn-arxiv-sized data and for all post-hoc plotting. Not enough
  for a DeBERTa fine-tune on the full Goodreads graph.
- **Colab T4 / Kaggle GPU.** Required for DeBERTa fine-tunes (one TA + one
  E per seed) and for any local-LLM explanation generation. Each fine-tune
  takes roughly 1.5 h on a T4. `notebooks/e2_goodreads.ipynb` is set up to
  cache `.emb` and `.ckpt` artifacts on Drive so that a re-attached Colab
  session picks up where the previous one left off.

## Implementation deviations from the released TAPE code

We document three differences from the upstream pipeline. Each is conservative
for our headline finding (C2: TA > E on Goodreads).

- **LLM:** `gpt-4o-mini` instead of `gpt-3.5-turbo` (deprecated). A stronger
  LLM can only narrow the C2 gap by improving the explanation view, so
  this is a stress test against the finding rather than for it.
- **Precision:** fp32 fine-tuning of DeBERTa. HuggingFace's mixed-precision
  path overflows on long Goodreads descriptions; fp32 if anything helps the
  E branch.
- **Backbones:** MLP, GCN, GraphSAGE; no RevGAT (DGL install issues on Colab).
  The C2 reversal is verified across all three backbones we do run.

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

The Goodreads-Children TAG comes from CS-TAG~\cite{yan2023comprehensive};
the underlying Goodreads dump is from Wan & McAuley
[(RecSys 2018)](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home).
