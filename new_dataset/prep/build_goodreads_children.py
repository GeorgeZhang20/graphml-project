"""Build the Goodreads-Children TAG from the CS-TAG `Children.csv` dump.

Source: https://huggingface.co/datasets/Sherirto/CSTAG/tree/main/Children
        (one CSV: 110 MB, columns = category, text, label, node_id, neighbour)

Outputs (consumed by TAPE/core/data_utils/load_goodreads_children.py):
  new_dataset/data/goodreads_children/graph.pt        PyG Data with x, edge_index, y, train/val/test_mask
  new_dataset/data/goodreads_children/node_texts.jsonl one {"id", "text"} per line
  new_dataset/data/goodreads_children/labels.txt       one label name per line, idx = label id

Run:
  python new_dataset/prep/build_goodreads_children.py [--csv path/to/Children.csv] [--out_dir ...]

Auto-downloads the CSV from HuggingFace if --csv is omitted.
"""
from __future__ import annotations
import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

CSV_URL = "https://huggingface.co/datasets/Sherirto/CSTAG/resolve/main/Children/Children.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "goodreads_children"
DEFAULT_RAW = Path(__file__).resolve().parents[1] / "raw" / "Children.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=None,
                   help="path to Children.csv; auto-downloads to new_dataset/raw/ if omitted")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--shallow_dim", type=int, default=128,
                   help="SVD-reduced TF-IDF dimensionality for the `x` (shallow OGB-equivalent) feature")
    p.add_argument("--tfidf_min_df", type=int, default=3,
                   help="min_df for TF-IDF; lower for tiny test corpora")
    p.add_argument("--split_seed", type=int, default=42,
                   help="seed for 60/20/20 split; 42 matches CS-TAG's split_graph()")
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--limit_nodes", type=int, default=None,
                   help="if set, keep only the first N rows (and edges among them) — "
                        "useful for fast end-to-end pipeline shake-out before the full run")
    return p.parse_args()


def download_csv(dst: Path):
    import urllib.request
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {CSV_URL} -> {dst}")
    urllib.request.urlretrieve(CSV_URL, dst)
    print(f"[download] {dst.stat().st_size / 1e6:.1f} MB")


def reformat_text(raw: str) -> str:
    """CSV text is 'Description: {desc}; Title: {title}'; flip to TAPE's 'Title: ...\\nDescription: ...'."""
    if not isinstance(raw, str):
        return ""
    if "; Title: " in raw:
        desc_part, title_part = raw.split("; Title: ", 1)
        desc = desc_part.removeprefix("Description: ").strip()
        title = title_part.strip()
        return f"Title: {title}\nDescription: {desc}"
    return raw


def build_edge_index(df: pd.DataFrame) -> torch.Tensor:
    """neighbour column is a string-encoded Python list of ints."""
    src, dst = [], []
    n = len(df)
    for nid, neigh_str in zip(df["node_id"].tolist(), df["neighbour"].tolist()):
        try:
            neigh = ast.literal_eval(neigh_str) if isinstance(neigh_str, str) else []
        except (ValueError, SyntaxError):
            neigh = []
        for j in neigh:
            if 0 <= j < n and j != nid:
                src.append(nid)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # symmetrise + dedupe (the CSV adjacency may be one-sided)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index.t(), dim=0).t().contiguous()
    return edge_index


def build_shallow_features(texts: list[str], dim: int, min_df: int) -> torch.Tensor:
    """TF-IDF -> TruncatedSVD; this is the `ogb`-equivalent shallow feature for E2."""
    print(f"[features] TF-IDF on {len(texts)} docs (min_df={min_df})...")
    tfidf = TfidfVectorizer(
        max_features=20000, stop_words="english",
        sublinear_tf=True, min_df=min_df, max_df=0.95,
    )
    X = tfidf.fit_transform(texts)
    print(f"[features] TF-IDF matrix: {X.shape}, nnz={X.nnz}")
    n_components = min(dim, min(X.shape) - 1)
    if n_components < dim:
        print(f"[features] capping SVD components to {n_components} (corpus too small for dim={dim})")
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    Xr = svd.fit_transform(X)
    print(f"[features] explained variance ratio sum: {svd.explained_variance_ratio_.sum():.3f}")
    return torch.tensor(Xr, dtype=torch.float32)


def random_split(n: int, train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = torch.zeros(n, dtype=torch.bool)
    val = torch.zeros(n, dtype=torch.bool)
    test = torch.zeros(n, dtype=torch.bool)
    train[idx[:n_train]] = True
    val[idx[n_train:n_train + n_val]] = True
    test[idx[n_train + n_val:]] = True
    return train, val, test


def main():
    args = parse_args()

    csv_path = args.csv or DEFAULT_RAW
    if not csv_path.exists():
        if args.csv is not None:
            sys.exit(f"--csv path does not exist: {csv_path}")
        download_csv(csv_path)

    print(f"[load] reading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[load] {len(df)} rows, columns: {list(df.columns)}")

    required = {"category", "text", "label", "node_id", "neighbour"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV missing required columns: {missing}")

    df = df.sort_values("node_id").reset_index(drop=True)
    n_full = len(df)
    if not (df["node_id"] == np.arange(n_full)).all():
        sys.exit("node_id is not a contiguous 0..N-1 range; aborting")

    # label mapping computed on the FULL dataset, before any subsampling, so the
    # number of classes is stable across subsample and full runs (the GNN output
    # head will just have some unused slots when subsampling).
    full_label_to_name: dict[int, str] = {}
    for lbl, cat in zip(df["label"].tolist(), df["category"].tolist()):
        full_label_to_name.setdefault(int(lbl), str(cat))
    num_classes = max(full_label_to_name) + 1
    if set(full_label_to_name) != set(range(num_classes)):
        sys.exit(f"label space is not contiguous: {sorted(full_label_to_name)}")
    label_names = [full_label_to_name[i] for i in range(num_classes)]
    print(f"[labels] {num_classes} classes (full dataset): {label_names}")

    if args.limit_nodes is not None and args.limit_nodes < n_full:
        keep = args.limit_nodes
        print(f"[subsample] keeping first {keep} of {n_full} nodes")
        df = df.iloc[:keep].copy()
        n = keep
        present = sorted(int(l) for l in df["label"].unique())
        print(f"[subsample] classes present in subset: {len(present)}/{num_classes} → {present}")
    else:
        n = n_full

    # texts
    texts = [reformat_text(t) for t in df["text"].tolist()]
    print(f"[text] avg chars/doc = {np.mean([len(t) for t in texts]):.0f}, "
          f"max chars = {max(len(t) for t in texts)}")

    # graph
    edge_index = build_edge_index(df)
    print(f"[graph] |V|={n}, |E|={edge_index.size(1)} (symmetrised)")

    # shallow `x` feature for the `ogb` row of E2
    x = build_shallow_features(texts, dim=args.shallow_dim, min_df=args.tfidf_min_df)

    # labels + masks
    y = torch.tensor(df["label"].astype(int).values, dtype=torch.long)
    train_mask, val_mask, test_mask = random_split(
        n, args.train_ratio, args.val_ratio, args.split_seed)
    print(f"[split] train={int(train_mask.sum())}, val={int(val_mask.sum())}, "
          f"test={int(test_mask.sum())} (seed={args.split_seed})")

    # write outputs
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    from torch_geometric.data import Data
    data = Data(
        x=x, edge_index=edge_index, y=y,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        num_nodes=n,
    )
    torch.save(data, out / "graph.pt")
    print(f"[write] {out / 'graph.pt'}")

    with open(out / "node_texts.jsonl", "w") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"id": i, "text": t}) + "\n")
    print(f"[write] {out / 'node_texts.jsonl'}")

    with open(out / "labels.txt", "w") as f:
        for name in label_names:
            f.write(name + "\n")
    print(f"[write] {out / 'labels.txt'}")

    print("[done]")


if __name__ == "__main__":
    main()
