"""TAPE-side loader for the Goodreads-Children TAG.

Mirrors the API of load_arxiv.get_raw_text_arxiv: returns (data, text_or_None).
Reads the artifacts produced by `new_dataset/prep/build_goodreads_children.py`.

The optional `dataset` kwarg lets the same loader serve the subsample slug
(`goodreads_children_n1500`) and the full one (`goodreads_children`) — the
registry in `load.py` passes whichever was requested, and we look in
`new_dataset/data/<dataset>/` accordingly.
"""
import json
from pathlib import Path

import torch


def _data_dir_candidates(dataset: str):
    return [
        # When run from TAPE/ (the upstream entrypoint convention): new_dataset/ is a sibling.
        Path(__file__).resolve().parents[3] / "new_dataset" / "data" / dataset,
        # When run from project root:
        Path("new_dataset") / "data" / dataset,
    ]


def _resolve_data_dir(dataset: str) -> Path:
    candidates = _data_dir_candidates(dataset)
    for cand in candidates:
        if cand.exists() and (cand / "graph.pt").exists():
            return cand
    raise FileNotFoundError(
        f"{dataset}/graph.pt not found in any of:\n  - "
        + "\n  - ".join(str(c) for c in candidates)
        + f"\nRun `python new_dataset/prep/build_goodreads_children.py "
        + (f"--limit_nodes {dataset.split('_n')[-1]}` " if "_n" in dataset else "` ")
        + "first."
    )


def get_raw_text_goodreads_children(use_text=False, seed=0, dataset="goodreads_children"):
    """Return (data, text_list_or_None) matching the contract used elsewhere in TAPE.

    `seed` is accepted for API parity but is ignored: the train/val/test split is
    fixed at build time (60/20/20, seed 42) so all four GNN seeds see the same split.

    `dataset` is the slug (e.g. ``goodreads_children`` or ``goodreads_children_n1500``);
    set by the registry in `load.py` so we read from the right artifact directory.
    """
    data_dir = _resolve_data_dir(dataset)

    data = torch.load(data_dir / "graph.pt", weights_only=False)

    # PyG Data already carries x, edge_index, y, train/val/test_mask.
    # TAPE's gnn_trainer.py expects data.y[i] to be a scalar; squeeze for safety.
    if data.y.dim() > 1:
        data.y = data.y.squeeze(-1)

    if not use_text:
        return data, None

    text_path = data_dir / "node_texts.jsonl"
    text = [None] * data.num_nodes
    with open(text_path, "r") as f:
        for line in f:
            row = json.loads(line)
            text[row["id"]] = row["text"]
    if any(t is None for t in text):
        raise ValueError(f"node_texts.jsonl missing entries; check {text_path}")
    return data, text
