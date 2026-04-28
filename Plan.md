# Project Plan — TAPE on a New Text-Attributed Graph

**Course:** COMP 459 / Machine Learning with Graphs (Spring 2026, Arlei Silva)
**Team:** George Zhang, Haoran Sun, Yiwen Zhu
**Final report due:** 2026-05-05 (NeurIPS format, 8 pages excluding refs)

---

## 1. Background: what TAPE does

> **Paper:** *Harnessing Explanations: LLM-to-LM Interpreter for Enhanced
> Text-Attributed Graph Representation Learning* — He, Bresson, Laurent,
> Perold, LeCun, Hooi (ICLR 2024).
> [arXiv:2305.19523](https://arxiv.org/abs/2305.19523) · authors' figure at
> [TAPE/overview.svg](TAPE/overview.svg).

**Setup.** A *text-attributed graph* (TAG) puts a chunk of text on every node
(e.g., title+abstract on a citation graph). Standard GNNs need fixed-size
features, so prior work either fed shallow bag-of-words to the GNN or
fine-tuned a small LM on the text — both throw away signal.

**TAPE's idea.** For each node, prompt an LLM (GPT-3.5) with the node's text
and ask it to (a) predict the top-k labels and (b) **explain its reasoning** in
a few sentences. The explanation is shorter than the abstract but written in
*label-aligned* language, so a small DeBERTa fine-tuned on it produces strong
features. The LLM is acting as a feature engineer.

**Three views per node** (computed once, reused for every GNN):

| view | what it is                                          | code path                                                                  |
|------|-----------------------------------------------------|----------------------------------------------------------------------------|
| `TA` | DeBERTa fine-tuned on raw title+abstract (768-d)    | [`gnn_trainer.py:41-49`](TAPE/core/GNNs/gnn_trainer.py#L41-L49)            |
| `P`  | one-hot of LLM's top-k label guesses                | [`load.py:8-23`](TAPE/core/data_utils/load.py#L8-L23)                      |
| `E`  | DeBERTa fine-tuned on LLM explanation (768-d)       | [`gnn_trainer.py:50-58`](TAPE/core/GNNs/gnn_trainer.py#L50-L58)            |

Each view trains an independent GNN; **logits are averaged at test time**.
That ensemble is `TA_P_E` (it is *not* feature concatenation — see
[`ensemble_trainer.py:60-72`](TAPE/core/GNNs/ensemble_trainer.py#L60-L72)).

**Reported result on ogbn-arxiv** (paper §5; verify in their Table 2 before
quoting): shallow OGB → RevGAT ≈ 73.5%, plain DeBERTa(text) → RevGAT ≈ 75–76%,
**TAPE (TA+P+E) → RevGAT ≈ 77.5%**. Their ablations show **`E` carries the
load**: removing it hurts more than removing `TA`.

**Where TAPE sits in the landscape.** Before stress-testing it, here is how
TAPE compares to other ways of using text on graphs. The vocabulary is fuzzy
in the literature, so for this project we adopt:

- **LM** = small text encoder, ~100M params, e.g., BERT, DeBERTa, SBERT.
  Cheap to fine-tune; you train your own copy per dataset.
- **LLM** = chat-grade model, >10B params, e.g., GPT-3.5/4, Llama-3-8B. Used
  frozen, prompted (often via paid API), no per-task fine-tune.

| Family                       | Examples                          | How node features are produced                | Uses LLM?         | Uses LM?           | Sees graph? |
|------------------------------|-----------------------------------|------------------------------------------------|-------------------|--------------------|-------------|
| Shallow text                 | TF-IDF, skip-gram, OGB defaults   | hand-crafted vector                            | no                | no                 | no          |
| Frozen pretrained LM         | BERT-CLS, SBERT                   | LM forward pass, no training                   | no                | yes (frozen)       | no          |
| Fine-tuned LM                | GIANT, plain DeBERTa-FT           | LM trained on the label task                   | no                | yes (fine-tuned)   | no          |
| LLM-as-classifier (zero-shot)| prompt GPT-4 directly             | LLM outputs the label; **no GNN at all**       | yes (frozen)      | no                 | no          |
| **LLM-as-feature engineer**  | **TAPE**, GraphAdapter            | LLM writes explanation → LM encodes it → GNN   | yes (frozen)      | yes (fine-tuned)   | yes         |
| LLM + GNN co-training        | GLEM, LLaGA                       | LLM and GNN trained jointly                    | yes (fine-tuned)  | yes                | yes         |

**Where each TAPE view lives in this map:**
- `TA` = the *fine-tuned LM* row (DeBERTa on raw text, no LLM).
- `P`  = the *LLM-as-classifier* row (just the LLM's guessed labels).
- `E`  = the *LLM-as-feature-engineer* row (LLM writes, LM encodes).

So `TA_P_E` is effectively an **ensemble of three different rows of this
table**, which is part of why it generalises better than any single approach.

**Why stress-test it.** TAPE was only evaluated on paper-citation graphs, so
two assumptions are untested: that LLM explanations remain informative on
non-academic text (→ H3), and that the recipe transfers off the academic
regime at all (→ H2).

---

## 2. Research question

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

## 3. Hypotheses to test

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

## 4. Method (what we are actually running)

For every dataset $\mathcal{D} \in \{\text{ogbn-arxiv}, \text{NEW}\}$ and every
GNN $g \in \{\text{MLP}, \text{GCN}, \text{SAGE}\}$, we feed the GNN one of the
following five feature configurations:

| feature  | plain-language meaning                                                                                               | dim    | how to get it                                              |
|----------|----------------------------------------------------------------------------------------------------------------------|--------|------------------------------------------------------------|
| `ogb`    | the **original shallow features that ship with the dataset** (TF-IDF / skip-gram-style); no language model involved   | dataset-default | OGB / dataset, no model needed                  |
| `TA`     | a 768-d DeBERTa embedding, **fine-tuned on labels using the raw title+abstract as input** ("Title-and-Abstract")     | 768    | `core.trainLM` (default) → written to `prt_lm/<ds>/`       |
| `P`      | the **top-k label IDs the LLM guessed** for this node; the GNN learns a small embedding per label and concatenates k of them | k integers | `gpt_preds/<DATASET>.csv` (cached for upstream datasets) |
| `E`      | a 768-d DeBERTa embedding, **fine-tuned on labels using the LLM's *explanation text* as input** instead of the original text | 768    | `core.trainLM ... lm.train.use_gpt True` → `prt_lm/<ds>2/`  |
| `TA_P_E` | **not** a single concatenated vector — train one GNN per view (TA, P, E) and **average their logits at test time** (the headline TAPE config) | —      | `core.trainEnsemble`                                       |

The four input vocabularies for "what text the LM saw":
1. **`ogb`** — no text at all, just shallow features.
2. **`TA`** — *raw* node text (title + abstract).
3. **`E`** — text the *LLM wrote* about this node (the explanation).
4. **`P`** — no text at the GNN side, just label IDs.

### 4.1 What `TA_P_E` actually means (step-by-step)

> **It is not a feature, it is a training recipe.** This is the single most
> common point of confusion when reading the TAPE paper.

The string `TA_P_E` is a config flag. The upstream code at
[`ensemble_trainer.py:60-72`](TAPE/core/GNNs/ensemble_trainer.py#L60-L72) splits
it on `_` → `["TA", "P", "E"]` and runs the following loop:

```python
# Pseudocode of what cfg.gnn.train.feature_type = "TA_P_E" actually does
for view in ["TA", "P", "E"]:                # three SEPARATE GNNs
    gnn_view  = fresh GCN/SAGE/MLP            # different random init each time
    features  = load(view)                    # 768-d, k-int, or 768-d resp.
    train(gnn_view, features, train_labels)   # full training run per view
    logits[view] = gnn_view.predict(all_nodes)

# Test-time blending: average the three GNNs' output logits
final_logits = (logits["TA"] + logits["P"] + logits["E"]) / 3
prediction   = argmax(final_logits)
```

So at the end of training you have **three GNN models sitting in memory**,
not one. There is no `768 + k + 768`-dimensional super-feature anywhere — the
views are kept apart all the way through, and only the *output predictions*
get blended.

**Concrete numerical example** for one test node in a 3-class problem:

| GNN trained on | softmax output      | argmax says |
|----------------|---------------------|-------------|
| `TA` features  | [0.70, 0.20, 0.10]  | class 0     |
| `P`  features  | [0.40, 0.50, 0.10]  | class 1     |
| `E`  features  | [0.20, 0.70, 0.10]  | class 1     |
| **average**    | [0.43, 0.47, 0.10]  | **class 1** |

The TA-GNN was outvoted; the P-GNN and E-GNN agreed and won. That's the
ensemble. The single number reported as "TAPE accuracy" comes from this
final averaged prediction across all test nodes.

**Why this matters for our hypotheses:** Because each view trains its own
GNN independently, *removing* a view just means dropping its GNN from the
average. That is exactly what ablation **A1** in §7 does — `TA+P` is the same
recipe with the E-GNN deleted, etc. — and is what lets us tell which of the
three GNNs is doing the heavy lifting on the new dataset.

We compute test accuracy mean ± std over 4 seeds {0,1,2,3}.

## 5. Datasets

### 5.1 Required to ship
- **ogbn-arxiv** (TAPE replication anchor). 169k nodes, 40 classes.

### 5.2 Pick exactly one new TAG (Phase 1, day 1)
Top candidates listed in [`new_dataset/README.md`](new_dataset/README.md). Default
choice: **Goodreads-Children** (~76k nodes, book-genre classification, clear
text attribute, clearly *not* a paper graph).
Backup: **Amazon-Books** (~36k, smaller → faster turnaround if blocked).

## 6. Experiments (main results)

Every run reports test accuracy as **mean ± std over seeds {0,1,2,3}**, using
the upstream entry points (`core.trainGNN` for single-feature rows,
`core.trainEnsemble` for `TA_P_E`). All runs land in `results/runs/`.

| ID | Hypothesis | Dataset           | Features                          | GNNs        | # runs | Output                          |
|----|------------|-------------------|-----------------------------------|-------------|--------|---------------------------------|
| E1 | H1         | ogbn-arxiv        | `ogb`, `TA_P_E`                   | GCN         | 8      | Table 1, replication anchor     |
| E2 | H2 + H3    | new TAG (Goodreads) | `ogb`, `TA`, `P`, `E`, `TA_P_E` | GCN, SAGE   | 40     | Table 2, full feature × GNN     |
| E3 | H2 control | new TAG           | `frozen-DeBERTa(text)` (no fine-tune) | GCN     | 4      | "is it just the LM?" baseline   |
| E4 | sanity     | Cora *(opt)*      | `ogb`, `TA_P_E`                   | GCN         | 8      | second replication anchor if E1 misbehaves |

**Definition of done for the main results:** Table 1 reproduces ogbn-arxiv
within ±1% of the paper, and Table 2 contains all 5 × 2 = 10 feature/GNN cells
on the new dataset with std ≤ 1.5%.

### Why each experiment exists

**E1 — ogbn-arxiv replication (H1).** Sanity check on the whole pipeline. If
our env, configs, DeBERTa fine-tunes, and GNN training are wired correctly, a
GCN with `TA_P_E` features should land within ±1% of the paper's GCN+TAPE
number on ogbn-arxiv. If E1 fails, **nothing else we report is trustworthy**
— so this gate must pass before we touch the new dataset. Comparing `ogb` (the
shallow OGB feature baseline) against `TA_P_E` also confirms the lift goes in
the right direction.

**E2 — feature × GNN matrix on Goodreads (H2 + H3).** The headline experiment
of the project. We run *every* feature type (`ogb`, `TA`, `P`, `E`, `TA_P_E`)
crossed with two GNN architectures (GCN, SAGE) on the new dataset. Two things
fall out of the same matrix:
- *H2 — does TAPE transfer?* Compare `TA_P_E` against `ogb`. If `TA_P_E` wins
  by a smaller margin than on academic graphs, H2 is supported.
- *H3 — which view carries the ensemble?* Compare the singletons `TA`, `P`,
  `E` against each other. The paper found `E` strongest on academic text. If
  on Goodreads `P` ≥ `E`, that flips the picture and supports H3.

This matrix is exactly Table 2 of the report.

**E3 — frozen-DeBERTa baseline on Goodreads (H2 control).** Without this we
can't tell whether TAPE's lift is *"the LLM helps"* or just *"any deep LM
helps"*. We extract DeBERTa CLS embeddings from the raw text **without
fine-tuning** and feed those to a GCN. Expected ordering: shallow OGB <
frozen DeBERTa < TAPE. If frozen DeBERTa already matches TAPE, the LLM
explanation pipeline isn't doing useful work on this dataset — that's a
publishable negative result.

**E4 — Cora sanity (optional).** A second, much smaller replication anchor.
We only run it if E1 misbehaves: Cora trains in minutes on CPU, so it helps
localize whether a bug lives in the LM fine-tune, the ensemble trainer, or
somewhere else. Skip if E1 lands within ±1% of the paper.

## 7. Ablation experiments (do them in order; stop when out of time)

All ablations run **on the new TAG only** unless noted. They feed §12 Tables 3+.

| ID | Question (focus)                          | Levers                                                  | Effort | Drop if... |
|----|-------------------------------------------|---------------------------------------------------------|--------|------------|
| A1 | Which two views are enough? (H3)          | `TA+P`, `TA+E`, `P+E` vs full `TA_P_E`                  | low    | never — this is the H3 evidence beyond E2 |
| A2 | Does top-k matter for `P`?                | k ∈ {1, 3, 5}                                           | low    | timeline slips by 1 day |
| A3 | Does prompt template matter for `E`?      | zero-shot classify vs +CoT vs +keyword-list (1k subset) | medium | timeline slips by 2 days |
| A4 | Open-source LLM swap                      | GPT-4o-mini vs Llama-3-8B on a 2k-node subset           | medium | budget allows only one backend |
| A5 | Frozen vs fine-tuned LM for `E`           | DeBERTa frozen vs fine-tuned                            | low    | timeline slips by 1 day |
| A6 | Low-label regime                          | 25% / 50% / 100% of train labels                        | low    | timeline slips by 2 days |

**Priority order:** A1 > A2 > A5 > A3 > A4 > A6. A1 is the single most
important ablation because it directly tests H3 (P-only vs E-only contribution
to the ensemble); A4 is most expensive and should be last.

### Why each ablation exists

**A1 — which two views are enough? (H3 evidence beyond E2).** Run the
ensemble trainer with `TA+P`, `TA+E`, `P+E` and compare to full `TA_P_E`. If
`TA+P` ≈ `TA_P_E` but `TA+E` is noticeably worse, **`P` is carrying the
ensemble** (and `E` is mostly noise on this dataset) — clean evidence for H3.
The opposite ordering refutes H3. *Non-droppable.*

**A2 — top-k for `P`.** The `P` feature is a multi-hot over the LLM's top-k
guesses. The paper uses k=5 for arxiv and k=3 for pubmed; on a new label
space we don't know the right k, so we sweep k ∈ {1, 3, 5}. If accuracy is
flat across k, k is not a sensitive knob; if it's not, we tune it.

**A3 — prompt template for `E`.** `E`'s quality depends on the prompt we
send the LLM. We try the current zero-shot prompt, a CoT ("let's think step
by step") variant, and a structured-keyword variant on a 1k-node subset.
This tells us how much of `E`'s value is just prompt engineering — a real
concern for anyone trying to reuse TAPE on their own data.

**A4 — open-source LLM swap.** GPT-4o-mini costs money; Llama-3-8B is free
and runs locally. If Llama produces competitive explanations, that's a
meaningful practical finding (TAPE doesn't actually need a paid API). If
Llama is noticeably worse, that's also worth flagging — it caps how
reproducible TAPE really is. Run on a 2k-node subset to keep cost down.

**A5 — frozen vs fine-tuned LM for `E`.** The `E` feature comes from a
DeBERTa *fine-tuned* on labels. The natural question: does fine-tuning even
matter, or is the LLM explanation already so label-aligned that a frozen
sentence encoder is enough? If frozen ≈ fine-tuned, we can drop a multi-hour
fine-tune step from the pipeline.

**A6 — low-label regime.** TAPE uses the full train split (~60%). The
LLM-explanation feature should help **more** when labels are scarce, since
the LLM doesn't need our labels to write explanations. We sweep
{25%, 50%, 100%} of the train labels and watch whether TAPE's lift over
`ogb` *grows* as labels shrink. If yes, that's a real-world advantage worth
highlighting.

## 8. Phased timeline (8 days)

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

## 9. Division of labor

> Mapped to the experiment IDs in §6 and §7. Each person's contribution must be
> summarized in the final report under "Author Contributions" (course requirement).

| Person      | Owns (code + data)                                                                                                                                                                  | Experiments | Backs up |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|----------|
| **George**  | E2 + all of E2's prerequisites: Goodreads build (`new_dataset/prep/build_goodreads_children.py`), `load_goodreads_children.py`, registry edit, `llm_explanations/generate.py`, both DeBERTa fine-tunes (TA + E × 4 seeds on Colab Pro), the 5 × 2 × 4 GNN sweep, results aggregation, plots | **E2**      | Haoran   |
| **Haoran**  | ogbn-arxiv replication harness, downloading the authors' cached `gpt_responses` + `.emb` checkpoints, running E1, plus the drop-one ensemble ablation (A1) on Goodreads once George's TA/E features are written | E1, A1      | George   |
| **Yiwen**   | Prompt-template ablation infra (a thin variant on `generate.py`), top-k sweep for `P` (re-runs `core.trainGNN` with topk ∈ {1, 3, 5}), low-label regime experiment (subsetting `train_mask`)                       | A2, A6      | Haoran   |
| **unassigned** | A3 (prompt template), A4 (Llama backend), A5 (frozen-LM), E3 (frozen-DeBERTa baseline), E4 (Cora sanity) — pick up if a phase finishes early; drop if not                                                                      | —           | —        |
| **all**     | Final report writing, error analysis on misclassified nodes                                                                                                                          | —           | —        |

Suggested initial assignment based on workstream coverage; swap if anyone
prefers a different fit. The "Backs up" column kicks in if a teammate is
blocked or out — see Risk register §11.

## 10. Key files / where things live

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

## 11. Risk register

| Risk                                                           | Likelihood | Mitigation |
|----------------------------------------------------------------|------------|------------|
| TAPE doesn't reproduce on ogbn-arxiv in a Colab session        | medium     | use the authors' provided `*.ckpt` and `*.emb`; load and just train the GNN head |
| LLM calls cost more than expected                              | low        | start with a 2k-node dry run; estimate $/call before full run |
| New dataset has no public train/val/test split                 | medium     | use a fixed 60/20/20 random split, seed 0 |
| Colab T4 OOMs on DeBERTa for ogbn-products-scale text          | medium     | cap `lm.train.batch_size` to 4 + `grad_acc_steps 4`; or freeze DeBERTa and only train pooler |
| Numpy 1.x ↔ 2.x conflict on Intel Mac                          | confirmed  | run heavy training in Colab/Linux; local Mac is for GNN-only steps |
| Dropouts: a teammate is blocked                                | medium     | each task has a backup owner (§9) |

## 12. What the final report needs (NeurIPS format, 8 pp)

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

## 13. Open questions to resolve in Phase 1
- [ ] Final dataset choice (Goodreads-Children vs. Amazon-Books vs. Reddit)
- [ ] LLM backend (GPT-4o-mini vs. Llama-3-8B on Colab)
- [ ] Whether to also include Cora/PubMed in Table 1 for completeness
- [ ] Who has the OpenAI API key / billing set up
