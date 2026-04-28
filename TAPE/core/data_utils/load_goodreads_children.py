"""TAPE-side loader for the Goodreads-Children TAG.

Mirrors the API of load_arxiv.get_raw_text_arxiv: returns (data, text_or_None).
Reads the artifacts produced by `new_dataset/prep/build_goodreads_children.py`.
"""
import json
import os
from pathlib import Path

import torch

# Resolve paths from ANY cwd. TAPE entrypoints `cd TAPE` before running, so the
# new_dataset/ folder is one level up from there.
_DATA_DIR_CANDIDATES = [
    # When run from TAPE/ (the upstream entrypoint convention)
    Path(__file__).resolve().parents[3] / "new_dataset" / "data" / "goodreads_children",
    # When run from project root
    Path("new_dataset") / "data" / "goodreads_children",
]


def _resolve_data_dir() -> Path:
    for cand in _DATA_DIR_CANDIDATES:
        if cand.exists() and (cand / "graph.pt").exists():
            return cand
    # last-resort error message points at the build script
    raise FileNotFoundError(
        "goodreads_children/graph.pt not found in any of:\n  - "
        + "\n  - ".join(str(c) for c in _DATA_DIR_CANDIDATES)
        + "\nRun `python new_dataset/prep/build_goodreads_children.py` first."
    )


def get_raw_text_goodreads_children(use_text=False, seed=0):
    """Return (data, text_list_or_None) matching the contract used elsewhere in TAPE.

    `seed` is accepted for API parity but is ignored: the train/val/test split is
    fixed at build time (60/20/20, seed 42) so all four GNN seeds see the same split.
    """
    data_dir = _resolve_data_dir()

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
