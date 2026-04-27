# Project Plan — TAPE on a New Text-Attributed Graph

**Course:** COMP 549 / Machine Learning with Graphs (Spring 2026, Arlei Silva)
**Team:** 3 students (assign names in §6)
**Final report due:** 2026-05-05 (NeurIPS format, 8 pages excluding refs)
**Today:** 2026-04-27 → **8 days remaining**

---

## 1. Research question

> The TAPE method (He et al., ICLR 2024) shows that prompting an LLM to **predict
> a label *and explain its reasoning*** for each node, then encoding the
> explanation with a small LM, yields strong node features for downstream GNNs.
> The paper evaluates this only on **paper-citation graphs** (Cora, PubMed,
> ogbn-arxiv, arxiv-2023).
>
> **Does the LLM-explanation feature still help on a non-academic
> text-attributed graph, and which component of the TAPE pipeline is doing the
> work?**

This positions the project under **Category 4** of the call-for-projects
("Novel synergies between Large Language Models and graphs") and partially
under **Category 2** ("evaluating a recent GraphML model" on a new dataset).

## 2. Hypotheses to test

- **H1 (replication).** TAPE's "TA + P + E" features beat shallow OGB features on
  ogbn-arxiv by the margin reported in the paper.
- **H2 (transfer).** On a non-academic TAG (e.g., Goodreads-Children),
  TAPE features still beat shallow + plain-BERT features, **but the lift is
  smaller** than on academic graphs.
- **H3 (component attribution).** Most of TAPE's gain on the new dataset comes
  from the *predicted label* (P), not the *explanation* (E) — i.e., on
  description-style text the LLM's reasoning is less judgmentally informative
  than on academic abstracts.

H3 is the headline finding if it holds; if the data goes the other way, that's
also a publishable observation.

## 3. Method (what we are actually running)

For every dataset $\mathcal{D} \in \{\text{ogbn-arxiv}, \text{NEW}\}$ and every
GNN $g \in \{\text{MLP}, \text{GCN}, \text{SAGE}\}$:

| feature | meaning                                                        | comes from                |
|---------|----------------------------------------------------------------|---------------------------|
| `ogb`   | original shallow node features                                 | OGB / dataset             |
| `TA`    | DeBERTa(text)                                                  | fine-tune LM on labels    |
| `P`     | one-hot of LLM top-k predicted label                           | `gpt_preds/<DATASET>.csv` |
| `E`     | DeBERTa(LLM explanation text)                                  | fine-tune LM on `use_gpt` |
| `TA_P_E`| concatenation                                                  | TAPE headline             |

We compute test accuracy mean ± std over 4 seeds {0,1,2,3}.

## 4. Datasets

### 4.1 Required to ship
- **ogbn-arxiv** (TAPE replication anchor). 169k nodes, 40 classes.

### 4.2 Pick exactly one new TAG (Phase 1, day 1)
Top candidates listed in [`new_dataset/README.md`](new_dataset/README.md). Default
choice: **Goodreads-Children** (~76k nodes, book-genre classification, clear
text attribute, clearly *not* a paper graph).
Backup: **Amazon-Books** (~36k, smaller → faster turnaround if blocked).

## 5. Phased timeline (8 days)

Each phase has a definition-of-done. If a phase slips, the *backup* row tells
us what to drop.

| # | Phase                          | Days        | Deliverable                                                         | If slipping, drop... |
|---|--------------------------------|-------------|---------------------------------------------------------------------|----------------------|
| 0 | Setup                          | 04-27       | env builds; `01_smoke_test.sh` passes; team has access              | —                    |
| 1 | Reproduce TAPE on ogbn-arxiv   | 04-28→04-29 | A table row matching paper ±1% on ogbn-arxiv (any one GNN)          | use authors' checkpoints instead of fine-tuning from scratch |
| 2 | Build new dataset + LLM calls  | 04-29→05-01 | `new_dataset/data/<DATASET>/{graph.pt, node_texts.jsonl, labels.txt}` + `TAPE/gpt_responses/<DATASET>/*.json` for ALL nodes | shrink to a 30k-node subgraph |
| 3 | Train pipeline on new dataset  | 05-01→05-02 | `TA`, `E`, `TA_P_E` rows for {GCN, SAGE} × 4 seeds                  | drop SAGE, keep GCN  |
| 4 | Ablations (H3)                 | 05-02→05-03 | feature × GNN matrix on new dataset (table 2)                       | drop MLP row         |
| 5 | Plots + report                 | 05-03→05-05 | 8-page NeurIPS-format PDF + repo cleanup                            | shorten discussion   |

### Concrete commands per phase

```bash
# Phase 0
bash scripts/00_setup_env.sh
conda activate tape-proj
bash scripts/01_smoke_test.sh

# Phase 1
# Download data per TAPE/UPSTREAM_README.md §1, then:
bash scripts/02_reproduce_arxiv.sh

# Phase 2
python new_dataset/prep/build_goodreads_children.py     # TODO (write in Phase 2)
python llm_explanations/generate.py \
    --dataset goodreads_children \
    --node_texts new_dataset/data/goodreads_children/node_texts.jsonl \
    --labels    new_dataset/data/goodreads_children/labels.txt

# Phase 3 + 4
bash scripts/03_run_new_dataset.sh goodreads_children
bash scripts/04_ablations.sh       goodreads_children
```

## 6. Division of labor

> Each contribution must be summarized in the final report (course requirement).

| Person   | Owns                                                                 | Backup for     |
|----------|----------------------------------------------------------------------|----------------|
| **A**    | Env, TAPE replication, GNN training, results tables                  | Person C       |
| **B**    | New dataset construction + label list + train/val/test split + DGL/PyG glue (`load_<DATASET>.py`) | Person A |
| **C**    | LLM explanation generation (prompt design, API/Llama calls, caching, JSON formatting) + ablations on prompt template | Person B |
| **all**  | Report writing, plotting, error analysis                             |                |

Fill in actual names in the final report under "Author Contributions".

## 7. Key files / where things live

| What                         | Path                                                |
|------------------------------|-----------------------------------------------------|
| Upstream config (yacs)       | [TAPE/core/config.py](TAPE/core/config.py)          |
| Dataset registry             | [TAPE/core/data_utils/load.py](TAPE/core/data_utils/load.py) |
| LM trainer (DeBERTa)         | [TAPE/core/LMs/lm_trainer.py](TAPE/core/LMs/lm_trainer.py)   |
| GNN trainer                  | [TAPE/core/GNNs/gnn_trainer.py](TAPE/core/GNNs/gnn_trainer.py) |
| Ensemble entry point         | [TAPE/core/trainEnsemble.py](TAPE/core/trainEnsemble.py)     |
| Original GPT preds (cached)  | [TAPE/gpt_preds/](TAPE/gpt_preds/)                  |
| New-dataset build            | [new_dataset/prep/](new_dataset/prep/)              |
| LLM call code                | [llm_explanations/generate.py](llm_explanations/generate.py) |

To register a new dataset in TAPE we will need to:
1. add `load_<DATASET>.py` under `TAPE/core/data_utils/` exposing `get_raw_text_<dataset>(use_text, seed)`,
2. add one branch to `load_data` in [TAPE/core/data_utils/load.py:26-43](TAPE/core/data_utils/load.py),
3. drop label preds at `TAPE/gpt_preds/<DATASET>.csv`,
4. drop per-node JSONs at `TAPE/gpt_responses/<DATASET>/*.json`.

## 8. Risk register

| Risk                                                           | Likelihood | Mitigation |
|----------------------------------------------------------------|------------|------------|
| TAPE doesn't reproduce on ogbn-arxiv in a Colab session        | medium     | use the authors' provided `*.ckpt` and `*.emb`; load and just train the GNN head |
| LLM calls cost more than expected                              | low        | start with a 2k-node dry run; estimate $/call before full run |
| New dataset has no public train/val/test split                 | medium     | use a fixed 60/20/20 random split, seed 0 |
| Colab T4 OOMs on DeBERTa for ogbn-products-scale text          | medium     | cap `lm.train.batch_size` to 4 + `grad_acc_steps 4`; or freeze DeBERTa and only train pooler |
| Numpy 1.x ↔ 2.x conflict on Intel Mac                          | confirmed  | run heavy training in Colab/Linux; local Mac is for GNN-only steps |
| Dropouts: a teammate is blocked                                | medium     | each task has a backup owner (§6) |

## 9. What the final report needs (NeurIPS format, 8 pp)

1. **Abstract** (~150 words). State H1/H2/H3 and the verdict.
2. **Introduction.** Why LLM × graph is interesting; what TAPE does.
3. **Related work.** TAPE, GLEM, GraphAdapter, GLBench, plus traditional GNN-on-TAG.
4. **Method.** Re-explain TAPE concisely; describe new dataset; describe the
   ablation grid.
5. **Experiments.**
   - Table 1: replication on ogbn-arxiv.
   - Table 2: feature × GNN matrix on the new dataset.
   - Table 3: prompt-template ablation (if Phase 4 has time).
6. **Discussion.** Verdict on H1/H2/H3, plus error analysis on misclassified
   nodes (which prompts confuse the LLM?).
7. **Conclusion + limitations.**
8. **References.**
9. **Author contributions.**

## 10. Open questions to resolve in Phase 1
- [ ] Final dataset choice (Goodreads-Children vs. Amazon-Books vs. Reddit)
- [ ] LLM backend (GPT-4o-mini vs. Llama-3-8B on Colab)
- [ ] Whether to also include Cora/PubMed in Table 1 for completeness
- [ ] Who has the OpenAI API key / billing set up
