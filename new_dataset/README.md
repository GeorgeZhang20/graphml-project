# New Dataset

We apply TAPE to a Text-Attributed Graph (TAG) that the original paper did
**not** evaluate. Final dataset to be locked in Phase 1 of `Plan.md`.

## Candidate shortlist
| Name                 | Nodes  | Source                                                       | Why we'd pick it |
|----------------------|--------|--------------------------------------------------------------|------------------|
| Goodreads-Children   | ~76k   | [CS-TAG](https://github.com/sktsherlock/TAG-Benchmark)       | Clear text + book genre labels; not papers |
| Amazon-Books         | ~36k   | CS-TAG                                                       | Smaller, faster turnaround |
| Reddit (TAG version) | ~33k   | [GLBench](https://github.com/NineAbyss/GLBench/tree/main/datasets) | Different domain (forum posts) |

## Files we must produce
For each chosen dataset, `new_dataset/data/<DATASET>/` should contain:
- `graph.pt` — PyG `Data` object with `x`, `edge_index`, `y`, `train_mask`, `val_mask`, `test_mask`
- `node_texts.jsonl` — `{id, text}` per line, indexed 0..N-1
- `labels.txt` — one label name per line (index = label id)

The TAPE loader needs us to add a `load_<DATASET>.py` module mirroring the API of
[TAPE/core/data_utils/load_arxiv_2023.py](../TAPE/core/data_utils/load_arxiv_2023.py),
exposing `get_raw_text_<dataset>(use_text=False, seed=0)`.

## Build script
Place data prep scripts in `new_dataset/prep/build_<DATASET>.py`. Stub TBD.
