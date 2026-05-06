"""Frozen-DeBERTa featurizer for the E3/A5 controls (sec. 4.5).

E3: frozen DeBERTa CLS embedding on raw node text       --> sibling of TA, no fine-tune
A5: frozen DeBERTa CLS embedding on LLM explanation text --> sibling of E,  no fine-tune

Both produce a `.emb` file in the same on-disk format the upstream LM trainer
emits (raw float32 matrix, shape (n_nodes, hidden_dim)) so the existing GNN
trainer can pick it up without any code change. The wrapper at
`scripts/05_run_a5_ablation.sh` swaps the canonical path via a symlink so
nothing in TAPE/core/ has to be patched.

Usage:
  # E3 (raw text, no fine-tune)
  python scripts/05_frozen_deberta_emb.py \
      --dataset goodreads_children \
      --n_nodes 76349 \
      --node_texts_jsonl new_dataset/data/goodreads_children/node_texts.jsonl \
      --use raw \
      --out TAPE/prt_lm/goodreads_children_frozen/microsoft/deberta-base.emb

  # A5 (LLM explanation, no fine-tune)
  python scripts/05_frozen_deberta_emb.py \
      --dataset goodreads_children \
      --n_nodes 76349 \
      --gpt_responses TAPE/gpt_responses/goodreads_children \
      --use explanation \
      --out TAPE/prt_lm/goodreads_children2_frozen/microsoft/deberta-base.emb
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def parse_explanation(raw_json: dict) -> str:
    """Pull explanation text out of one TAPE per-node response JSON."""
    try:
        return raw_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def load_explanations(gpt_responses_dir: Path, n_nodes: int) -> list[str]:
    """Load per-node explanations in node-id order. Missing nodes get ''."""
    texts = [""] * n_nodes
    for p in sorted(gpt_responses_dir.glob("*.json")):
        try:
            node_id = int(p.stem)
        except ValueError:
            continue
        if node_id >= n_nodes:
            continue
        with p.open() as f:
            texts[node_id] = parse_explanation(json.load(f))
    return texts


def load_raw_texts(jsonl_path: Path, n_nodes: int) -> list[str]:
    texts = [""] * n_nodes
    with jsonl_path.open() as f:
        for line in f:
            row = json.loads(line)
            texts[int(row["node_id"])] = row.get("text", "")
    return texts


@torch.no_grad()
def embed(texts: list[str], model_name: str, batch_size: int, device: str,
          max_length: int = 512) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        # CLS-pooled hidden state to match the fine-tuned recipe.
        h = model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
        out[i:i + len(chunk)] = h
        if (i // batch_size) % 10 == 0:
            print(f"  embedded {i + len(chunk)}/{len(texts)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_nodes", type=int, required=True,
                    help="number of graph nodes; must match Data.num_nodes")
    ap.add_argument("--gpt_responses",
                    help="dir of per-node JSONs (use with --use explanation)")
    ap.add_argument("--node_texts_jsonl",
                    help="JSONL of {node_id, text} (use with --use raw)")
    ap.add_argument("--use", choices=["explanation", "raw"], required=True)
    ap.add_argument("--out", required=True,
                    help="output .emb path; mirror prt_lm/<ds>(2)/<model>.emb")
    ap.add_argument("--model", default="microsoft/deberta-base")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.use == "explanation":
        assert args.gpt_responses, "--gpt_responses required for --use explanation"
        texts = load_explanations(Path(args.gpt_responses), args.n_nodes)
    else:
        assert args.node_texts_jsonl, "--node_texts_jsonl required for --use raw"
        texts = load_raw_texts(Path(args.node_texts_jsonl), args.n_nodes)

    n_empty = sum(1 for t in texts if not t)
    print(f"[frozen-deberta] {args.use}: {len(texts) - n_empty}/{len(texts)} "
          f"non-empty texts on {args.dataset}")

    emb = embed(texts, args.model, args.batch_size, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    emb.astype(np.float32).tofile(out)
    print(f"[frozen-deberta] wrote {out} ({emb.shape[0]} x {emb.shape[1]} fp32)")


if __name__ == "__main__":
    main()
