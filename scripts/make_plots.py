"""Render E2 figures from results/<DATASET>_long.csv.

Produces under results/figures/:
  <DATASET>_heatmap.{png,pdf}    feature x GNN, cells = mean test acc + std
  <DATASET>_bars.{png,pdf}       grouped bars, seed-error bars
  <DATASET>_scatter.{png,pdf}    one dot per seed (variance shown honestly)
  <DATASET>_lift.{png,pdf}       per-feature delta vs `ogb` baseline (the H2 picture)

Same logic as notebooks/plots.ipynb, headless so the auto-push cell can call it.

Usage:
  python scripts/make_plots.py --dataset goodreads_children
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_ORDER = ["ogb", "TA", "P", "E", "TA_P_E"]
GNN_ORDER = ["GCN", "SAGE"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    args = parse_args()
    long_csv = args.results_dir / f"{args.dataset}_long.csv"
    if not long_csv.exists():
        sys.exit(f"{long_csv} not found. Run aggregate_results.py first.")
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(long_csv)
    df = df[(df["returncode"] == 0) & df["test_acc"].notna()].copy()
    if df.empty:
        sys.exit(f"{long_csv} has no successful runs to plot.")
    df["feature"] = pd.Categorical(df["feature"], FEATURE_ORDER, ordered=True)
    df["gnn"] = pd.Categorical(df["gnn"], GNN_ORDER, ordered=True)
    n_seeds = df["seed"].nunique()
    print(f"[plot] dataset={args.dataset}  n_seeds={n_seeds}  rows={len(df)}")

    # Aggregate for heatmap / bars / lift
    agg = (df.groupby(["feature", "gnn"], observed=True)["test_acc"]
             .agg(["mean", "std", "count"])
             .reset_index())
    mean_mat = agg.pivot(index="feature", columns="gnn", values="mean").reindex(FEATURE_ORDER)[GNN_ORDER]
    std_mat = agg.pivot(index="feature", columns="gnn", values="std").reindex(FEATURE_ORDER)[GNN_ORDER]

    # ---------- 1. heatmap ----------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(mean_mat.values, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(GNN_ORDER))); ax.set_xticklabels(GNN_ORDER)
    ax.set_yticks(range(len(FEATURE_ORDER))); ax.set_yticklabels(FEATURE_ORDER)
    for i, _feat in enumerate(FEATURE_ORDER):
        for j, _gnn in enumerate(GNN_ORDER):
            m = mean_mat.values[i, j]
            s = std_mat.values[i, j]
            if pd.isna(m):
                txt = "—"
            elif pd.isna(s) or n_seeds <= 1:
                txt = f"{m:.3f}"
            else:
                txt = f"{m:.3f}\n±{s:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10)
    ax.set_title(f"E2: test acc on {args.dataset}  (n={n_seeds} seed{'s' if n_seeds != 1 else ''})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="test acc")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(fig_dir / f"{args.dataset}_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- 2. grouped bars ----------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(FEATURE_ORDER))
    width = 0.38
    for k, gnn in enumerate(GNN_ORDER):
        means = [mean_mat.loc[f, gnn] if f in mean_mat.index else np.nan for f in FEATURE_ORDER]
        stds = [std_mat.loc[f, gnn] if f in std_mat.index else 0 for f in FEATURE_ORDER]
        stds = [0 if (pd.isna(s) or n_seeds <= 1) else s for s in stds]
        ax.bar(x + (k - 0.5) * width, means, width, yerr=stds, capsize=4, label=gnn)
    ax.set_xticks(x); ax.set_xticklabels(FEATURE_ORDER)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"E2 ({args.dataset}): feature × GNN, n={n_seeds} seed{'s' if n_seeds != 1 else ''}")
    ax.legend(title="GNN")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(fig_dir / f"{args.dataset}_bars.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- 3. per-seed scatter ----------
    fig, axes = plt.subplots(1, len(GNN_ORDER), figsize=(8, 4), sharey=True)
    rng = np.random.default_rng(0)
    for ax_, gnn in zip(axes, GNN_ORDER):
        sub = df[df["gnn"] == gnn]
        for j, feat in enumerate(FEATURE_ORDER):
            ys = sub.loc[sub["feature"] == feat, "test_acc"].values
            if len(ys) == 0:
                continue
            xs = np.full_like(ys, j, dtype=float) + rng.uniform(-0.08, 0.08, len(ys))
            ax_.scatter(xs, ys, alpha=0.85)
            ax_.hlines(np.mean(ys), j - 0.25, j + 0.25, colors="black", linewidth=2)
        ax_.set_xticks(range(len(FEATURE_ORDER))); ax_.set_xticklabels(FEATURE_ORDER, rotation=20)
        ax_.set_title(gnn)
        ax_.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("test accuracy")
    fig.suptitle(f"E2 per-seed scatter ({args.dataset})")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(fig_dir / f"{args.dataset}_scatter.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- 4. lift vs ogb (the H2 picture) ----------
    if "ogb" in mean_mat.index:
        ogb_means = mean_mat.loc["ogb"]
        lift = mean_mat.subtract(ogb_means, axis=1).drop(index="ogb")

        fig, ax = plt.subplots(figsize=(6.5, 4))
        x = np.arange(len(lift.index))
        width = 0.38
        for k, gnn in enumerate(GNN_ORDER):
            ax.bar(x + (k - 0.5) * width, lift[gnn].values, width, label=gnn)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(lift.index)
        ax.set_ylabel("test acc − ogb baseline")
        ax.set_title(f"E2 lift over shallow features ({args.dataset}, n={n_seeds})")
        ax.legend(title="GNN")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(fig_dir / f"{args.dataset}_lift.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        print("[plot] no `ogb` row in long.csv; skipping lift plot")

    written = sorted(fig_dir.glob(f"{args.dataset}_*"))
    print(f"[plot] wrote {len(written)} files to {fig_dir}/")
    for p in written:
        print(f"  - {p.relative_to(args.results_dir)}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
