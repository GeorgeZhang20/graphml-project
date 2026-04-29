"""Render E2 figures from results/<DATASET>_long.csv.

Outputs under results/figures/, all PNG. Set of figures:

Per-dataset (depend only on results/<DATASET>_long.csv):
  <DATASET>_heatmap.png         feature × GNN, mean ± std table
  <DATASET>_bars.png            grouped bars with seed-error bars
  <DATASET>_scatter.png         per-seed dots — variance shown honestly
  <DATASET>_lift.png            feature − ogb baseline (within-dataset H2 lift)
  <DATASET>_singletons.png      TA / P / E ordering — the H3 picture
  <DATASET>_time_vs_acc.png     wall-time vs test acc (cost/quality scatter)

Cross-dataset (also pull data/tape_paper_table2.csv with the paper's reported numbers):
  cross_landscape.png           shallow vs LM_finetune vs TAPE bars across all datasets
  cross_lift_over_shallow.png   how big is TAPE's lift on each dataset?
  cross_lift_over_LM.png        and how much does adding P+E to plain LM-finetune buy?

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
# GNN axis is dataset-driven now (we may run MLP/GCN/SAGE)
DEFAULT_GNN_ORDER = ["MLP", "GCN", "SAGE"]
# our project's "pretty name" for our dataset, used in cross-dataset plots
OUR_DATASET_LABEL = "Goodreads-Children\n(ours)"

# Mapping between paper column names and ours, for cross-dataset comparison.
# - h_shallow == ogb (raw OGB / TF-IDF features)
# - LM_finetune == TA (DeBERTa fine-tuned on raw text)
# - h_TAPE == TA_P_E (the ensemble)
# - h_GIANT and LLM are paper-only (not in our pipeline)
PAPER_TO_OURS = {"h_shallow": "ogb", "LM_finetune": "TA", "h_TAPE": "TA_P_E"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--paper_table", type=Path,
                   default=Path("data/tape_paper_table2.csv"),
                   help="Reported TAPE Table 2 numbers; used by cross-dataset plots.")
    p.add_argument("--cross_gnn", default="SAGE",
                   help="GNN architecture to use for cross-dataset plots (must exist for "
                        "every dataset including ours). Default SAGE.")
    return p.parse_args()


def load_our_long(args) -> pd.DataFrame:
    long_csv = args.results_dir / f"{args.dataset}_long.csv"
    if not long_csv.exists():
        sys.exit(f"{long_csv} not found. Run aggregate_results.py first.")
    df = pd.read_csv(long_csv)
    df = df[(df["returncode"] == 0) & df["test_acc"].notna()].copy()
    if df.empty:
        sys.exit(f"{long_csv} has no successful runs to plot.")
    return df


def get_gnn_order(df: pd.DataFrame) -> list[str]:
    present = [g for g in DEFAULT_GNN_ORDER if g in df["gnn"].unique()]
    extras = [g for g in df["gnn"].unique() if g not in DEFAULT_GNN_ORDER]
    return present + sorted(extras)


# ============================================================================
# Per-dataset plots
# ============================================================================


def plot_heatmap(df, fig_dir, dataset, gnn_order, n_seeds, mean_mat, std_mat):
    fig, ax = plt.subplots(figsize=(0.9 + 1.0 * len(gnn_order), 5))
    im = ax.imshow(mean_mat.values, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(gnn_order))); ax.set_xticklabels(gnn_order)
    ax.set_yticks(range(len(FEATURE_ORDER))); ax.set_yticklabels(FEATURE_ORDER)
    for i, _ in enumerate(FEATURE_ORDER):
        for j, _ in enumerate(gnn_order):
            m, s = mean_mat.values[i, j], std_mat.values[i, j]
            if pd.isna(m):
                txt = "—"
            elif pd.isna(s) or n_seeds <= 1:
                txt = f"{m:.3f}"
            else:
                txt = f"{m:.3f}\n±{s:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10)
    ax.set_title(f"E2: test acc on {dataset}  (n={n_seeds} seed{'s' if n_seeds != 1 else ''})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="test acc")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{dataset}_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bars(fig_dir, dataset, gnn_order, n_seeds, mean_mat, std_mat):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(FEATURE_ORDER))
    width = 0.8 / len(gnn_order)
    palette = plt.cm.tab10.colors
    for k, gnn in enumerate(gnn_order):
        means = [mean_mat.loc[f, gnn] if f in mean_mat.index else np.nan for f in FEATURE_ORDER]
        stds = [std_mat.loc[f, gnn] if f in std_mat.index else 0 for f in FEATURE_ORDER]
        stds = [0 if (pd.isna(s) or n_seeds <= 1) else s for s in stds]
        ax.bar(x + (k - (len(gnn_order) - 1) / 2) * width, means, width,
               yerr=stds, capsize=3, label=gnn, color=palette[k])
    ax.set_xticks(x); ax.set_xticklabels(FEATURE_ORDER)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"E2 ({dataset}): feature × GNN, n={n_seeds} seed{'s' if n_seeds != 1 else ''}")
    ax.legend(title="GNN")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{dataset}_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(df, fig_dir, dataset, gnn_order):
    fig, axes = plt.subplots(1, len(gnn_order), figsize=(3.2 * len(gnn_order), 4),
                             sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax_, gnn in zip(axes[0], gnn_order):
        sub = df[df["gnn"] == gnn]
        for j, feat in enumerate(FEATURE_ORDER):
            ys = sub.loc[sub["feature"] == feat, "test_acc"].values
            if len(ys) == 0:
                continue
            xs = np.full_like(ys, j, dtype=float) + rng.uniform(-0.08, 0.08, len(ys))
            ax_.scatter(xs, ys, alpha=0.85)
            ax_.hlines(np.mean(ys), j - 0.25, j + 0.25, colors="black", linewidth=2)
        ax_.set_xticks(range(len(FEATURE_ORDER)))
        ax_.set_xticklabels(FEATURE_ORDER, rotation=20)
        ax_.set_title(gnn)
        ax_.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("test accuracy")
    fig.suptitle(f"E2 per-seed scatter ({dataset})")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{dataset}_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lift_within_dataset(fig_dir, dataset, gnn_order, n_seeds, mean_mat):
    if "ogb" not in mean_mat.index:
        return
    ogb_means = mean_mat.loc["ogb"]
    lift = mean_mat.subtract(ogb_means, axis=1).drop(index="ogb")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(lift.index))
    width = 0.8 / len(gnn_order)
    palette = plt.cm.tab10.colors
    for k, gnn in enumerate(gnn_order):
        ax.bar(x + (k - (len(gnn_order) - 1) / 2) * width,
               lift[gnn].values, width, label=gnn, color=palette[k])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(lift.index)
    ax.set_ylabel("Δ test acc vs `ogb` (shallow features)")
    ax.set_title(f"Lift over shallow features ({dataset}, n={n_seeds})")
    ax.legend(title="GNN")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{dataset}_lift.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_singleton_ordering(fig_dir, dataset, gnn_order, n_seeds, mean_mat, std_mat):
    """The H3 picture — TA vs P vs E head-to-head, plus TA_P_E for context."""
    feats = [f for f in ["TA", "P", "E", "TA_P_E"] if f in mean_mat.index]
    if not feats:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(feats))
    width = 0.8 / len(gnn_order)
    palette = plt.cm.tab10.colors
    for k, gnn in enumerate(gnn_order):
        means = [mean_mat.loc[f, gnn] for f in feats]
        stds = [std_mat.loc[f, gnn] for f in feats]
        stds = [0 if (pd.isna(s) or n_seeds <= 1) else s for s in stds]
        ax.bar(x + (k - (len(gnn_order) - 1) / 2) * width, means, width,
               yerr=stds, capsize=3, label=gnn, color=palette[k])
    ax.set_xticks(x); ax.set_xticklabels(feats)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Singleton view comparison on {dataset}\n"
                 f"(H3: which view of TAPE carries the ensemble?)")
    ax.legend(title="GNN")
    ax.grid(axis="y", alpha=0.3)
    # Annotate the winning singleton on each GNN with a tiny star
    for k, gnn in enumerate(gnn_order):
        means = [mean_mat.loc[f, gnn] for f in feats[:-1]]   # exclude TA_P_E from "winner"
        if all(pd.isna(m) for m in means):
            continue
        wins = int(np.nanargmax(means))
        x_pos = wins + (k - (len(gnn_order) - 1) / 2) * width
        ax.annotate("★", (x_pos, mean_mat.loc[feats[wins], gnn] + 0.005),
                    ha="center", va="bottom", fontsize=12, color=palette[k])
    plt.tight_layout()
    plt.savefig(fig_dir / f"{dataset}_singletons.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_time_vs_acc(df, fig_dir, dataset, gnn_order):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    palette = plt.cm.tab10.colors
    feat_marker = {"ogb": "o", "TA": "s", "P": "D", "E": "^", "TA_P_E": "*"}
    for k, gnn in enumerate(gnn_order):
        sub = df[df["gnn"] == gnn]
        for feat in FEATURE_ORDER:
            sub2 = sub[sub["feature"] == feat]
            if sub2.empty:
                continue
            ax.scatter(sub2["wall_seconds"], sub2["test_acc"],
                       marker=feat_marker.get(feat, "o"), s=80,
                       color=palette[k], alpha=0.8,
                       label=f"{gnn}·{feat}" if feat == FEATURE_ORDER[0] else None)
            for _, r in sub2.iterrows():
                ax.annotate(feat, (r["wall_seconds"], r["test_acc"]),
                            fontsize=7, alpha=0.7, ha="left", va="bottom")
    ax.set_xlabel("wall-clock seconds (one GNN training run)")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Cost vs quality on {dataset}")
    # build a separate legend that only shows GNN colors
    handles = [plt.Line2D([0], [0], marker="o", linestyle="none",
                          color=palette[k], label=gnn, markersize=8)
               for k, gnn in enumerate(gnn_order)]
    ax.legend(handles=handles, title="GNN", loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{dataset}_time_vs_acc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Cross-dataset plots: take the paper's reported numbers and bolt our row on
# ============================================================================


def build_cross_dataset_table(df_ours, mean_mat, std_mat, gnn_for_cross, paper_csv):
    """Return a tidy long-format DF with columns:
       dataset, gnn, method, acc_mean, acc_std
    Methods are: h_shallow, LM_finetune, h_TAPE (the 3 we can compare).
    Includes the paper's 5 datasets at the chosen `gnn_for_cross`, plus our row.
    """
    if not paper_csv.exists():
        sys.exit(f"{paper_csv} not found — needed for cross-dataset plots.")
    paper = pd.read_csv(paper_csv)
    paper = paper[paper["gnn"] == gnn_for_cross]
    paper = paper[paper["method"].isin(PAPER_TO_OURS.keys())].copy()

    # Map our singletons to the paper's column names so we can concat.
    rows = []
    for paper_name, ours_name in PAPER_TO_OURS.items():
        if ours_name in mean_mat.index and gnn_for_cross in mean_mat.columns:
            m = float(mean_mat.loc[ours_name, gnn_for_cross])
            s = float(std_mat.loc[ours_name, gnn_for_cross]) if not pd.isna(
                std_mat.loc[ours_name, gnn_for_cross]) else 0.0
            rows.append({"dataset": OUR_DATASET_LABEL, "gnn": gnn_for_cross,
                         "method": paper_name, "acc_mean": m, "acc_std": s})
    ours = pd.DataFrame(rows)

    return pd.concat([paper, ours], ignore_index=True)


def plot_cross_landscape(fig_dir, cross_df, gnn_for_cross):
    """Bar chart: one group per dataset, three bars (shallow / LM-finetune / TAPE)."""
    methods = ["h_shallow", "LM_finetune", "h_TAPE"]
    method_labels = {"h_shallow": "Shallow features",
                     "LM_finetune": "LM fine-tune (TA)",
                     "h_TAPE": "TAPE (TA+P+E)"}
    method_colors = {"h_shallow": "#aac8ff", "LM_finetune": "#ffb86c", "h_TAPE": "#5cb85c"}

    datasets_order = [d for d in cross_df["dataset"].unique() if d != OUR_DATASET_LABEL]
    datasets_order = ["Cora", "PubMed", "ogbn-arxiv", "ogbn-products", "tape-arxiv23"]
    datasets_order = [d for d in datasets_order if d in cross_df["dataset"].unique()]
    datasets_order.append(OUR_DATASET_LABEL)

    x = np.arange(len(datasets_order))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, m in enumerate(methods):
        means, stds = [], []
        for d in datasets_order:
            row = cross_df[(cross_df["dataset"] == d) & (cross_df["method"] == m)]
            means.append(float(row["acc_mean"].iloc[0]) if not row.empty else np.nan)
            stds.append(float(row["acc_std"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + (k - 1) * width, means, width, yerr=stds, capsize=3,
               label=method_labels[m], color=method_colors[m])

    # vertical separator between paper datasets and ours
    ax.axvline(len(datasets_order) - 1.5, color="gray", linestyle="--", alpha=0.6)
    ax.text(len(datasets_order) - 1.05, ax.get_ylim()[1] * 0.97,
            "← paper datasets   |   ours →", ha="center", va="top",
            fontsize=9, color="gray", fontstyle="italic")

    ax.set_xticks(x); ax.set_xticklabels(datasets_order, rotation=15)
    ax.set_ylabel(f"test accuracy ({gnn_for_cross})")
    ax.set_title(f"TAPE landscape: shallow / LM-finetune / TAPE-ensemble across datasets\n"
                 f"(paper Table 2 + ours; GNN = {gnn_for_cross})")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "cross_landscape.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cross_lift(fig_dir, cross_df, gnn_for_cross, baseline_method, fname, title):
    """Bar chart of TAPE_acc - baseline_acc per dataset."""
    datasets_order = ["Cora", "PubMed", "ogbn-arxiv", "ogbn-products", "tape-arxiv23"]
    datasets_order = [d for d in datasets_order if d in cross_df["dataset"].unique()]
    datasets_order.append(OUR_DATASET_LABEL)

    lifts = []
    for d in datasets_order:
        sub = cross_df[cross_df["dataset"] == d]
        try:
            tape = float(sub[sub["method"] == "h_TAPE"]["acc_mean"].iloc[0])
            base = float(sub[sub["method"] == baseline_method]["acc_mean"].iloc[0])
            lifts.append(tape - base)
        except (IndexError, ValueError):
            lifts.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#5cb85c"] * (len(datasets_order) - 1) + ["#d9534f"]  # ours in red
    bars = ax.bar(range(len(datasets_order)), lifts, color=colors)
    for i, v in enumerate(lifts):
        if not pd.isna(v):
            ax.annotate(f"{v:+.3f}", (i, v),
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(datasets_order))); ax.set_xticklabels(datasets_order, rotation=15)
    ax.set_ylabel(f"TAPE − {baseline_method} (test acc)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    # a discreet legend explaining the red bar
    ax.bar([], [], color="#5cb85c", label="paper-reported")
    ax.bar([], [], color="#d9534f", label="ours (Goodreads)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_dir / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_our_long(args)
    gnn_order = get_gnn_order(df)
    n_seeds = df["seed"].nunique()
    print(f"[plot] dataset={args.dataset}  n_seeds={n_seeds}  rows={len(df)}  gnns={gnn_order}")

    df["feature"] = pd.Categorical(df["feature"], FEATURE_ORDER, ordered=True)
    df["gnn"] = pd.Categorical(df["gnn"], gnn_order, ordered=True)
    agg = (df.groupby(["feature", "gnn"], observed=True)["test_acc"]
             .agg(["mean", "std", "count"]).reset_index())
    mean_mat = agg.pivot(index="feature", columns="gnn", values="mean").reindex(FEATURE_ORDER)[gnn_order]
    std_mat = agg.pivot(index="feature", columns="gnn", values="std").reindex(FEATURE_ORDER)[gnn_order]

    # --- per-dataset ---
    plot_heatmap(df, fig_dir, args.dataset, gnn_order, n_seeds, mean_mat, std_mat)
    plot_bars(fig_dir, args.dataset, gnn_order, n_seeds, mean_mat, std_mat)
    plot_scatter(df, fig_dir, args.dataset, gnn_order)
    plot_lift_within_dataset(fig_dir, args.dataset, gnn_order, n_seeds, mean_mat)
    plot_singleton_ordering(fig_dir, args.dataset, gnn_order, n_seeds, mean_mat, std_mat)
    plot_time_vs_acc(df, fig_dir, args.dataset, gnn_order)

    # --- cross-dataset (if paper table exists and our row has the chosen GNN) ---
    if args.paper_table.exists() and args.cross_gnn in gnn_order:
        cross_df = build_cross_dataset_table(df, mean_mat, std_mat,
                                              args.cross_gnn, args.paper_table)
        plot_cross_landscape(fig_dir, cross_df, args.cross_gnn)
        plot_cross_lift(fig_dir, cross_df, args.cross_gnn, "h_shallow",
                        "cross_lift_over_shallow.png",
                        f"How big is TAPE's lift over shallow features?  (GNN = {args.cross_gnn})")
        plot_cross_lift(fig_dir, cross_df, args.cross_gnn, "LM_finetune",
                        "cross_lift_over_LM.png",
                        f"What does P+E ensembling buy on top of plain LM fine-tune?  "
                        f"(GNN = {args.cross_gnn})")
    else:
        print(f"[plot] skipping cross-dataset plots "
              f"(paper_table exists={args.paper_table.exists()}, "
              f"{args.cross_gnn} in gnn_order={args.cross_gnn in gnn_order})")

    written = sorted(fig_dir.glob("*.png"))
    print(f"[plot] wrote {len(written)} files to {fig_dir}/")
    for p in written:
        print(f"  - {p.relative_to(args.results_dir)}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
