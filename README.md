# GraphML Spring 2026 Final Project — TAPE on a New Text-Attributed Graph

We extend [TAPE (He et al., ICLR 2024)](https://arxiv.org/abs/2305.19523) — an
LLM-as-feature-enhancer for text-attributed graphs — beyond the four
paper-citation datasets it was originally evaluated on, and run a controlled
study of where the "explanations-as-features" idea actually helps.

This is a fork of the upstream TAPE code, organized for our experiments.

## Repository layout
```
graphml-project/
├── TAPE/                    # upstream code (mostly untouched; UPSTREAM_README.md preserved)
│   ├── core/                # models + trainers (LM, GNN, ensemble)
│   ├── dataset/             # raw data, gitignored
│   ├── gpt_preds/           # small cached top-k LLM label preds (kept)
│   └── gpt_responses/       # large per-node LLM JSONs, gitignored
├── new_dataset/             # our extension: build a new TAG
│   ├── prep/                # build_<DATASET>.py scripts
│   └── data/                # outputs, gitignored
├── llm_explanations/        # generate per-node LLM explanations for new TAG
│   ├── prompts/
│   ├── responses/           # gitignored
│   └── generate.py
├── scripts/                 # reproducible run wrappers
├── configs/                 # custom YACS overrides
├── notebooks/               # exploration + plotting
├── results/                 # logs + metrics, gitignored
├── docs/
├── requirements.txt
├── Plan.md                  # <-- READ THIS
└── README.md
```

## Quickstart
```bash
# 1. set up env (creates conda env tape-proj, ~5 min)
bash scripts/00_setup_env.sh
conda activate tape-proj

# 2. confirm pipeline works on Cora (~3 sec)
bash scripts/01_smoke_test.sh
```

After that, follow `Plan.md` phase by phase.

## Hardware
- Local Mac (Intel, 16 GB, no CUDA): fine for GNN training on Cora / ogbn-arxiv-sized data.
- **Colab T4 / Kaggle GPU is required** for DeBERTa fine-tuning and for any
  Llama-based explanation generation.

## Citing the work we build on
```bibtex
@inproceedings{he2024harnessing,
  title={Harnessing Explanations: LLM-to-LM Interpreter for Enhanced
         Text-Attributed Graph Representation Learning},
  author={He, Xiaoxin and Bresson, Xavier and Laurent, Thomas and
          Perold, Adam and LeCun, Yann and Hooi, Bryan},
  booktitle={ICLR},
  year={2024}
}
```

License of upstream TAPE code: see `TAPE/LICENSE`.
