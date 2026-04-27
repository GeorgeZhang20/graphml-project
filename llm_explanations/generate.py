"""
Generate LLM explanations + top-k label predictions for every node in a TAG.

Outputs (to match TAPE's expected layout):
  TAPE/gpt_responses/<DATASET>/<i>.json   -- one file per node, OpenAI chat format
  TAPE/gpt_preds/<DATASET>.csv            -- CSV: row i = comma-separated top-5 label ids

Run after new_dataset/prep/build_<DATASET>.py has produced node texts + label list.
This is a stub. Filling it in is part of Phase 2 of Plan.md.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

# Choose ONE backend:
#   - openai: best quality, ~$5-15 for 30k nodes with gpt-4o-mini
#   - llama:  free, run on Colab T4 with llama-cpp-python or transformers
BACKEND = "openai"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="dataset name; matches TAPE folder names")
    p.add_argument("--node_texts", required=True,
                   help="path to a .jsonl file of {id, text} per node")
    p.add_argument("--labels", required=True,
                   help="path to a .txt file with one label per line")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--max_nodes", type=int, default=None,
                   help="for dry runs")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir or f"../TAPE/gpt_responses/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "TODO(phase 2): implement OpenAI / Llama call loop with caching, "
        "retry on rate-limit, and write per-node JSON in OpenAI chat format "
        "(TAPE reads json_data['choices'][0]['message']['content']).")


if __name__ == "__main__":
    main()
