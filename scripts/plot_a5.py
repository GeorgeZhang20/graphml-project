"""Plot for sec. 4.5 / A5.

Reads the per-seed CSV produced by `05_run_a5_ablation.sh` and renders a
grouped bar chart of test accuracy: frozen vs fine-tuned DeBERTa for the
E view, on each backbone (MLP, GCN, SAGE).

Usage:
    python scripts/plot_a5.py \
        --csv results/goodreads_children_a5_frozen_vs_finetuned.csv \
        --out results/figures/full_76k/a5_frozen_vs_finetuned.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    agg = (df.groupby(["gnn", "lm_status"])["test_acc"]
             .agg(["mean", "std", "count"])
             .reset_index())

    gnns = ["MLP", "GCN", "SAGE"]
    statuses = ["frozen", "finetuned"]
    width = 0.36
    x = np.arange(len(gnns))

    colors = {"frozen": "#a6bddb", "finetuned": "#1f77b4"}
    labels = {"frozen": "Frozen DeBERTa (no fine-tune)",
              "finetuned": "Fine-tuned DeBERTa (TAPE default)"}

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for i, status in enumerate(statuses):
        means, stds = [], []
        for g in gnns:
            row = agg[(agg.gnn == g) & (agg.lm_status == status)].iloc[0]
            means.append(row["mean"] * 100)
            stds.append(row["std"] * 100)
        ax.bar(x + (i - 0.5) * width, means, width,
               yerr=stds, capsize=3, color=colors[status], label=labels[status])
        for xi, m, s in zip(x + (i - 0.5) * width, means, stds):
            ax.text(xi, m + s + 0.18, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=8.5)

    ax.set_xticks(x, gnns)
    ax.set_ylabel("Test accuracy on Goodreads-Children (%)")
    ax.set_ylim(55, 63)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, fontsize=9, frameon=False, handlelength=1.6)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.5, -0.18,
            r"E view, 4 seeds, mean $\pm$ std",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, style="italic", color="#444")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[plot_a5] wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
