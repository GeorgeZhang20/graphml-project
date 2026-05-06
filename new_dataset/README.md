# New Dataset

We apply TAPE to a Text-Attributed Graph (TAG) that the original paper did
**not** evaluate. We use **Goodreads-Children** from CS-TAG as our cross-domain
probe; build script is at `prep/build_goodreads_children.py`.

## Other TAGs the build pattern can be reused for
| Name                 | Nodes  | Source                                                       | Why this one |
|----------------------|--------|--------------------------------------------------------------|--------------|
| Goodreads-Children   | ~76k   | [CS-TAG](https://github.com/sktsherlock/TAG-Benchmark)       | the one we run |
| Amazon-Books         | ~36k   | CS-TAG                                                       | smaller, faster turnaround |
| Reddit (TAG version) | ~33k   | [GLBench](https://github.com/NineAbyss/GLBench/tree/main/datasets) | different domain (forum posts) |

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
