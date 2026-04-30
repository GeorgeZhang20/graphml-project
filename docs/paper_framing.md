# Paper Skeleton — COMP 459 Final Project

NeurIPS 2020, 8 pp body + refs. Due 2026-05-05.
Authors: George, Haoran, Yiwen.

## Status legend

| symbol | meaning |
|---|---|
| ✅ | done |
| 🟡 | in flight |
| ❌ | not started |

---

## Section skeleton

| § | Title | Pages | Primary claim | Supporting evidence | Status |
|---|---|---|---|---|---|
| 1 | Introduction | ~1.0 | Motivates TAGs + LLM-as-feature; states C1/C2/C3 up front | — | ❌ writing |
| 2 | Related work | ~0.75 | Position vs TAPE / GLEM / GIANT / GraphAdapter / CS-TAG | citations only | ❌ writing |
| 3 | Method | ~1.0 | We faithfully reproduce TAPE pipeline modulo 3 documented changes (gpt-4o-mini, fp32, no RevGAT) | Figure 12 (pipeline diagram); Goodreads schema table | ❌ writing |
| 4.1 | Replication on ogbn-arxiv | ~0.5 | Our pipeline reproduces TAPE's headline ogbn-arxiv number within ±1% | Table 1 | 🟡 Haoran (E1) |
| 4.2 | Goodreads-Children matrix | ~1.5 | **C1**: TAPE > shallow by 8-16 pp · **C2**: TA > E across 3 GNNs · **C3**: TAPE − TA = 0.4-1.3 pp | Table 2; Fig 2, Fig 6, Fig 8 | ✅ n=1; 🟡 n=4 |
| 4.3 | Cross-dataset comparison | ~0.75 | H3 reversal is a *dataset effect*, not a *GNN effect* | Fig 1, Fig 4, Fig 5, Fig 7 | ✅ paper data + n=1 ours |
| 4.4 | Drop-one ablation | ~0.5 | Removing E from `TA_P_E` costs less than removing TA on Goodreads (clinches C2) | Table 3, Fig 9 | ❌ Haoran (A1) |
| 4.5 | Frozen-LM controls (E3 + A5) | ~0.5 | Isolates "fine-tune contribution" from "LM encoder contribution" — supports C3 | Table 2 extra rows + Fig 13 | ❌ George (E3+A5) |
| 4.6 | Low-label regime | ~0.5 | TAPE's lift over `ogb` *grows* as labels shrink (or doesn't) — practical takeaway | Table 5, Fig 14 | ❌ Yiwen (A6) |
| 4.7 *(optional)* | Top-k for `P` | ~0.25 | `P` ablation is robust to k ∈ {1, 3, 5} — secondary | Table 4 | ❌ Yiwen (A2) |
| 5 | Discussion | ~1.5 | Why TA wins on Goodreads (§5.1); why graph adds little (§5.2 — C4); practitioner takeaways (§5.3) | qualitative spot-check of explanations + lift table from §4 | ❌ writing |
| 6 | Limitations | ~0.5 | One dataset, one LLM, fp32, no RevGAT, n=4 seeds | — | ❌ writing |
| 7 | Conclusion | ~0.25 | Restate C1/C2/C3; future work (Amazon-Books, Reddit-TAG) | — | ❌ writing |
| 8 | Author contributions | ~0.1 | Per-person breakdown (course requirement) | — | ❌ writing |

**Total: ~8 pages body** (4.5 fits if we tighten elsewhere; otherwise drop 4.5).

---

## Claims (one place to find them all)

| Tag | Claim | Strength of evidence we'd have at submission |
|---|---|---|
| **C1** | TAPE ensemble lifts shallow features by 8-16 pp on a non-academic TAG | strong: 4-seed ± std on full 76k |
| **C2** ⭐ | The strongest singleton view of TAPE flips between datasets — `TA > E` on Goodreads, `E ≥ TA` on TAPE's academic graphs | strong: directional consistent across MLP/GCN/SAGE + drop-one ablation (A1) |
| **C3** | The marginal value of the LLM-explanation pipeline (`TA_P_E - TA`) is small (<1.5 pp) on description-style text | strong: visible in Table 2 + cross-dataset comparison Fig 5 |
| C4 *(secondary)* | Goodreads's co-shelf graph adds <1.5 pp over MLP — graph signal is much weaker than citation networks | medium: clear from Table 2 / Fig 8 but only one comparison point |

---

## Experiment status board

| ID | Description | Owner | Status | Outputs |
|---|---|---|---|---|
| **E1** | Reproduce TAPE on ogbn-arxiv (use authors' `.ckpt` + `.emb`) | Haoran | ❌ | Table 1 |
| **E2** | Goodreads full pipeline, 5 features × 3 GNNs × 4 seeds | George | ✅ seed 0; 🟡 seeds 1-3 | Table 2 + Fig 1-8 |
| **E3** | **Control**: frozen DeBERTa on raw text (no fine-tune) → GNN | **George** | ❌ | extra row in Table 2; supports C3 |
| **A1** | Drop-one ablation: `TA+P`, `TA+E`, `P+E` vs `TA_P_E` | Haoran | ❌ | Table 3, Fig 9 |
| **A2** *(optional)* | Top-k sweep for `P`: k ∈ {1, 3, 5} | Yiwen | ❌ | Table 4 |
| **A5** | **Control**: frozen DeBERTa on **explanation text** (no fine-tune) → GNN | **George** | ❌ | extra row in Table 2; supports C3 (paired with E3) |
| **A6** | Low-label regime: train_mask ∈ {25%, 50%, 100%} × {`ogb`, `TA_P_E`} | Yiwen | ❌ | Table 5, Fig 13 |

---

## Figures status board

| Tag | File | Supports | Status |
|---|---|---|---|
| Fig 1 | `full_76k/cross_landscape.png` | C1 | ✅ rendered (paper data + ours n=1) |
| Fig 2 | `full_76k/goodreads_children_singletons.png` | C2 | ✅ rendered |
| Fig 3 | `full_76k/goodreads_children_heatmap.png` | C1 + C2 + C3 (table view) | ✅ rendered |
| Fig 4 | `full_76k/cross_lift_over_shallow.png` | C1 | ✅ rendered |
| Fig 5 | `full_76k/cross_lift_over_LM.png` | **C3** | ✅ rendered |
| Fig 6 | `full_76k/goodreads_children_h3_singleton_gap.png` | **C2** | ✅ rendered |
| Fig 7 | `full_76k/h3_TA_minus_E_paper_vs_ours.png` | **C2** (key) | ✅ rendered |
| Fig 8 | `full_76k/goodreads_children_gnn_comparison.png` | C4 | ✅ rendered |
| Fig 9 | drop-one ablation bars | C2 | ❌ pending A1 |
| Fig 12 | TAPE method pipeline diagram | §3 | ❌ to draw (matplotlib or draw.io) |
| Fig 13 | 6-bar component contribution: shallow → frozen-text → frozen-expl → TA → E → TAPE | **C3** | ❌ pending E3 + A5 |
| Fig 14 | TAPE − ogb lift vs train-label fraction | C1 in low-label regime | ❌ pending A6 |
| Fig 10 *(stretch)* | confusion matrix on Goodreads | C2 mechanistic | ❌ needs per-pred dump from GNN |
| Fig 11 *(stretch)* | per-class TAPE-vs-shallow lift | C2 mechanistic | ❌ needs per-pred dump |

All figures live under `results/figures/full_76k/`. n=1 currently; will re-render once seeds 1-3 land.

---

## Tables status board

| Tag | Description | Source | Status |
|---|---|---|---|
| Table 1 | E1 ogbn-arxiv replication, our row alongside paper Table 2 | Haoran's E1 | ❌ |
| Table 2 | Goodreads feature × GNN, mean ± std | E2 4 seeds | ✅ n=1; 🟡 n=4 |
| Table 3 | Drop-one ablation on Goodreads | A1 | ❌ |
| Table 4 *(optional)* | Top-k sweep for `P` | A2 | ❌ |

---

## What we owe to ship "Target tier"

In rough priority order:

1. 🟡 **E2 seeds 1-3** (George, ~5 h × 3 on A100, can chain across sessions)
2. ❌ **E1 ogbn-arxiv replication** (Haoran, ~1-2 h CPU)
3. ❌ **A1 drop-one ablation** (Haoran, ~30 min CPU after seed 0 .emb downloaded)
4. ❌ **NeurIPS LaTeX skeleton** with empty section files
5. ❌ **Figure 12 — pipeline diagram** (manual matplotlib/draw.io)
6. ❌ **Re-render figures** once seed 1-3 land
7. ❌ **A2 top-k for `P`** (Yiwen, ~1 h CPU) — *optional, not blocking*
8. ❌ **E3 frozen-DeBERTa baseline** (~1 h GPU + GNN sweep) — *optional, not blocking*
9. ❌ **Section drafts** by each owner per §8 contributions
10. ❌ **Final integration + proofread + submission**

---

## Risk register (for §6 Limitations + reviewer responses)

| Risk | Defense |
|---|---|
| One non-academic dataset only | Frame as *case study*; future work lists Amazon-Books and Reddit-TAG |
| n=4 seeds | Standard for TAPE paper; quote ± std |
| GPT-4o-mini ≠ paper's GPT-3.5-turbo | Stronger LLM only narrows H3 gap (E gets better) → finding is conservative |
| fp32 instead of mixed precision | Document HF DeBERTa overflow bug; fp32 *helps* `E` if anything → conservative |
| No RevGAT | DGL install issues on Colab; H3 reversal pattern verified on MLP/GCN/SAGE |
| One LLM | A4 (open-source LLM swap) is in original Plan, dropped due to time |
