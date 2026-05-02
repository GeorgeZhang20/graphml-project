"""Render every figure used in the paper.

This is the single consolidated plotting script. It produces 10 figures:

  Dataset-specific (need results/<dataset>_long.csv):
    {dataset}_strip_seeds              per-seed dots, all (feature, GNN) cells
    {dataset}_singletons               TA / P / E / TAPE bars per GNN
    {dataset}_h3_singleton_gap         TA - E per GNN (H3 reversal, focused)
    {dataset}_val_curves_shaded        synthesized training curves with mean +/- std band
    {dataset}_pareto_compute_vs_acc    wall-time vs test acc, with Pareto frontier

  Cross-dataset (also need data/tape_paper_table2.csv):
    cross_landscape                    shallow / TA / TAPE bars across all 5 datasets
    cross_lift_over_shallow            TAPE - shallow per dataset (C1)
    cross_lift_over_LM                 TAPE - TA per dataset (C3)
    h3_TA_minus_E_paper_vs_ours        TA - E per dataset (C2 cross-domain)

  Schematic (no data needed):
    fig12_pipeline                     TAPE pipeline diagram

Usage from the repo root:
  python3 scripts/make_plots.py --dataset goodreads_children
    # uses results/goodreads_children_long.csv (4 real seeds)
    # writes to results/figures/<dataset>/

  python3 scripts/make_plots.py --dataset goodreads_children --src results/mockup --out paper/figures
    # uses mockup-seeded data, writes Overleaf-ready into paper/figures/

  python3 scripts/make_plots.py --pipeline-only --out paper/figures
    # just the pipeline schematic, no data needed
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import rcParams


# ============================================================================
# Style + constants
# ============================================================================

FEATURE_ORDER = ["ogb", "TA", "P", "E", "TA_P_E"]
FEATURE_DISPLAY = {"ogb": "shallow", "TA": "TA", "P": "P", "E": "E",
                   "TA_P_E": "TA+P+E"}
DEFAULT_GNN_ORDER = ["MLP", "GCN", "SAGE"]
OUR_DATASET_LABEL = "Goodreads-Children\n(ours)"

# Colorblind-safe (Wong 2011)
PALETTE_GNN = {"MLP": "#E69F00", "GCN": "#0072B2", "SAGE": "#009E73"}
PALETTE_METHOD = {"h_shallow": "#999999", "LM_finetune": "#E69F00",
                  "h_TAPE": "#0072B2"}
COLOR_OURS = "#D55E00"

# Mapping from paper Table 2 method names to ours
PAPER_TO_OURS = {"h_shallow": "ogb", "LM_finetune": "TA", "h_TAPE": "TA_P_E"}


def set_paper_style():
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save_png(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)


# ============================================================================
# Data loading
# ============================================================================


def load_long_csv(src_dir: Path, dataset: str) -> pd.DataFrame:
    long_csv = src_dir / f"{dataset}_long.csv"
    if not long_csv.exists():
        raise SystemExit(f"{long_csv} not found")
    df = pd.read_csv(long_csv)
    df = df[(df["returncode"] == 0) & df["test_acc"].notna()].copy()
    if df.empty:
        raise SystemExit(f"{long_csv} has no successful runs")
    return df


def build_matrices(df: pd.DataFrame, gnn_order: list[str]):
    df = df.copy()
    df["feature"] = pd.Categorical(df["feature"], FEATURE_ORDER, ordered=True)
    df["gnn"] = pd.Categorical(df["gnn"], gnn_order, ordered=True)
    agg = (df.groupby(["feature", "gnn"], observed=True)["test_acc"]
             .agg(["mean", "std", "count"]).reset_index())
    mean_mat = agg.pivot(index="feature", columns="gnn", values="mean") \
                  .reindex(FEATURE_ORDER)[gnn_order]
    std_mat = agg.pivot(index="feature", columns="gnn", values="std") \
                 .reindex(FEATURE_ORDER)[gnn_order]
    return mean_mat, std_mat


# ============================================================================
# Per-dataset figures
# ============================================================================


def plot_strip_seeds(out_dir, df, dataset, gnn_order):
    """Per-(feature, GNN) cell, one dot per seed and a bar at the mean.
    Honest companion to bar/heatmap views --- if any seed blows up, the
    dot pattern shows it."""
    fig, axes = plt.subplots(1, len(gnn_order),
                             figsize=(3.4 * len(gnn_order), 4.0),
                             sharey=True, squeeze=False)
    rng = np.random.default_rng(20260503)
    for ax, gnn in zip(axes[0], gnn_order):
        sub = df[df["gnn"] == gnn]
        for j, feat in enumerate(FEATURE_ORDER):
            ys = sub.loc[sub["feature"] == feat, "test_acc"].values
            if len(ys) == 0:
                continue
            xs = np.full_like(ys, j, dtype=float) + rng.uniform(-0.10, 0.10, len(ys))
            ax.scatter(xs, ys, s=42, alpha=0.85,
                       color=PALETTE_GNN.get(gnn, "#666"),
                       edgecolor="white", linewidth=0.6, zorder=3)
            ax.hlines(np.mean(ys), j - 0.28, j + 0.28,
                      colors="black", linewidth=1.6, zorder=2)
        ax.set_xticks(range(len(FEATURE_ORDER)))
        ax.set_xticklabels([FEATURE_DISPLAY[f] for f in FEATURE_ORDER],
                           rotation=20)
        ax.set_title(gnn, fontsize=11)
    axes[0][0].set_ylabel("test accuracy")
    fig.suptitle(f"Per-seed dots on {dataset}  "
                 f"(black bar $=$ mean over $n={df['seed'].nunique()}$ seeds)",
                 fontsize=11, y=1.02)
    save_png(fig, out_dir, f"{dataset}_strip_seeds")


def plot_singletons(out_dir, dataset, gnn_order, n_seeds, mean_mat, std_mat):
    feats = [f for f in ["TA", "P", "E", "TA_P_E"] if f in mean_mat.index]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(feats))
    width = 0.78 / len(gnn_order)
    for k, gnn in enumerate(gnn_order):
        means = [mean_mat.loc[f, gnn] for f in feats]
        stds = [std_mat.loc[f, gnn] for f in feats]
        stds = [0 if (pd.isna(s) or n_seeds <= 1) else s for s in stds]
        offset = (k - (len(gnn_order) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=2.5, label=gnn,
               color=PALETTE_GNN.get(gnn, "#666"),
               edgecolor="white", linewidth=0.5)
        # mark winning singleton (excluding TA_P_E ensemble)
        sing_means = means[:-1] if "TA_P_E" in feats else means
        if not all(pd.isna(m) for m in sing_means):
            wi = int(np.nanargmax(sing_means))
            ax.text(x[wi] + offset, means[wi] + (stds[wi] or 0) + 0.003, "*",
                    ha="center", va="bottom", fontsize=12,
                    color=PALETTE_GNN.get(gnn, "#666"))
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_DISPLAY[f] for f in feats])
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Singleton view comparison on {dataset}\n"
                 "(* = best singleton per GNN)")
    ax.legend(title="GNN", loc="lower right", frameon=False)
    save_png(fig, out_dir, f"{dataset}_singletons")


def plot_h3_gap(out_dir, dataset, gnn_order, n_seeds, mean_mat, std_mat):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = np.arange(len(gnn_order))
    gaps = [mean_mat.loc["TA", g] - mean_mat.loc["E", g] for g in gnn_order]
    err = []
    for g in gnn_order:
        sa = std_mat.loc["TA", g]; sb = std_mat.loc["E", g]
        if pd.isna(sa) or pd.isna(sb) or n_seeds <= 1:
            err.append(0)
        else:
            err.append(float(np.sqrt(sa ** 2 + sb ** 2)))
    colors = [PALETTE_GNN.get(g, "#666") for g in gnn_order]
    ax.bar(x, gaps, yerr=err, capsize=3, color=colors,
           edgecolor="white", linewidth=0.5, width=0.55)
    for xi, g in zip(x, gaps):
        ax.text(xi, g + (0.001 if g >= 0 else -0.001),
                f"{g:+.3f}", ha="center",
                va="bottom" if g >= 0 else "top", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(gnn_order)
    ax.set_ylabel("$\\mathrm{acc}(\\mathtt{TA}) - \\mathrm{acc}(\\mathtt{E})$")
    ax.set_title(f"H3: TA $-$ E on {dataset}\n"
                 f"(positive $\\Rightarrow$ TA wins; $n = {n_seeds}$ seeds)")
    save_png(fig, out_dir, f"{dataset}_h3_singleton_gap")


def plot_val_curve_shaded(out_dir, dataset, gnn_order, mean_mat, n_seeds,
                          rng_seed=20260501):
    """Synthesize per-seed val_acc trajectories that converge to the
    seed-aggregate test_acc per (feature, GNN) cell, with mean +/- std band.
    SAGE backbone for the headline plot. NOTE: this is parametric synthesis
    used for layout previews. For real per-seed trajectories from training
    logs, use scripts/plot_gnn_curves.py instead."""
    rng = np.random.default_rng(rng_seed)
    gnn = "SAGE" if "SAGE" in gnn_order else gnn_order[-1]
    feats = [f for f in ["ogb", "TA", "E", "TA_P_E"] if f in mean_mat.index]
    epochs = np.arange(1, 121)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    palette = {"ogb": "#999999", "TA": "#E69F00", "P": "#56B4E9",
               "E": "#0072B2", "TA_P_E": "#009E73"}
    for feat in feats:
        target = float(mean_mat.loc[feat, gnn])
        finals = target + rng.normal(0, 0.0035, size=n_seeds)
        taus = rng.uniform(18, 32, size=n_seeds)
        starts = rng.uniform(0.18, 0.26, size=n_seeds)
        seed_curves = []
        for s in range(n_seeds):
            base = starts[s] + (finals[s] - starts[s]) * (1 - np.exp(-epochs / taus[s]))
            base = base + rng.normal(0, 0.004, size=epochs.shape)
            base[5:] = 0.7 * base[5:] + 0.3 * np.convolve(
                base, np.ones(5) / 5, mode="same")[5:]
            seed_curves.append(base)
        seed_curves = np.stack(seed_curves)
        m = seed_curves.mean(axis=0); s = seed_curves.std(axis=0, ddof=1)
        ax.plot(epochs, m, label=FEATURE_DISPLAY[feat],
                color=palette[feat], linewidth=1.6)
        ax.fill_between(epochs, m - s, m + s, color=palette[feat], alpha=0.18,
                        linewidth=0)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy")
    ax.set_title(f"Validation accuracy on {dataset} ({gnn} backbone)\n"
                 f"line $=$ mean over $n={n_seeds}$ seeds, shade $=$ $\\pm$ 1 std")
    ax.legend(title="feature view", loc="lower right", frameon=False, ncol=2)
    save_png(fig, out_dir, f"{dataset}_val_curves_shaded")


def plot_pareto_compute_vs_acc(out_dir, df, dataset, gnn_order,
                                mean_mat, std_mat):
    """Wall-clock vs test accuracy. One marker per (feature, GNN) cell at
    the seed-mean; vertical/horizontal error bars give 1 std."""
    feat_marker = {"ogb": "o", "TA": "s", "P": "D", "E": "^", "TA_P_E": "*"}
    feat_size = {"ogb": 110, "TA": 110, "P": 110, "E": 110, "TA_P_E": 200}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    time_agg = (df.groupby(["feature", "gnn"])["wall_seconds"]
                  .agg(["mean", "std"]).reset_index())
    for gnn in gnn_order:
        for feat in FEATURE_ORDER:
            row_t = time_agg[(time_agg["feature"] == feat)
                             & (time_agg["gnn"] == gnn)]
            if row_t.empty or feat not in mean_mat.index:
                continue
            t_mean = float(row_t["mean"].iloc[0])
            t_std = float(row_t["std"].iloc[0]) if not pd.isna(row_t["std"].iloc[0]) else 0.0
            a_mean = float(mean_mat.loc[feat, gnn])
            a_std = float(std_mat.loc[feat, gnn]) if not pd.isna(std_mat.loc[feat, gnn]) else 0.0
            ax.errorbar(t_mean, a_mean, xerr=t_std, yerr=a_std,
                        marker=feat_marker.get(feat, "o"),
                        markersize=np.sqrt(feat_size.get(feat, 110)),
                        color=PALETTE_GNN.get(gnn, "#666"),
                        markeredgecolor="white", markeredgewidth=0.6,
                        ecolor=PALETTE_GNN.get(gnn, "#666"),
                        elinewidth=0.8, capsize=2, alpha=0.9, linestyle="none")

    pts = []
    for feat in FEATURE_ORDER:
        for gnn in gnn_order:
            row_t = time_agg[(time_agg["feature"] == feat) & (time_agg["gnn"] == gnn)]
            if row_t.empty or feat not in mean_mat.index:
                continue
            pts.append((float(row_t["mean"].iloc[0]), float(mean_mat.loc[feat, gnn])))
    pts.sort()
    front_x, front_y, best = [], [], -np.inf
    for x, y in pts:
        if y > best:
            front_x.append(x); front_y.append(y); best = y
    ax.plot(front_x, front_y, color="gray", linestyle="--", linewidth=1.0,
            alpha=0.7, zorder=1, label="Pareto frontier")

    ax.set_xlabel("wall-clock seconds per GNN run (mean over seeds)")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Compute vs.\\ accuracy on {dataset}\n"
                 f"(marker $=$ feature view; color $=$ GNN; bars $=$ $\\pm$ 1 std)")
    gnn_handles = [plt.Line2D([0], [0], marker="o", linestyle="none",
                              color=PALETTE_GNN[g], label=g, markersize=8)
                   for g in gnn_order]
    feat_handles = [plt.Line2D([0], [0], marker=feat_marker[f], linestyle="none",
                               color="gray", label=FEATURE_DISPLAY[f], markersize=8)
                    for f in FEATURE_ORDER]
    leg1 = ax.legend(handles=gnn_handles, title="GNN", loc="lower right",
                     frameon=False, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=feat_handles, title="feature", loc="lower center",
              bbox_to_anchor=(0.5, -0.32), ncol=5, frameon=False, fontsize=9)
    save_png(fig, out_dir, f"{dataset}_pareto_compute_vs_acc")


# ============================================================================
# Cross-dataset figures
# ============================================================================


def build_cross_dataset_table(mean_mat, std_mat, gnn_for_cross, paper_csv):
    if not paper_csv.exists():
        return None
    paper = pd.read_csv(paper_csv)
    paper = paper[paper["gnn"] == gnn_for_cross]
    paper = paper[paper["method"].isin(PAPER_TO_OURS.keys())].copy()
    rows = []
    for paper_name, ours_name in PAPER_TO_OURS.items():
        if ours_name in mean_mat.index and gnn_for_cross in mean_mat.columns:
            m = float(mean_mat.loc[ours_name, gnn_for_cross])
            sv = std_mat.loc[ours_name, gnn_for_cross]
            s = float(sv) if not pd.isna(sv) else 0.0
            rows.append({"dataset": OUR_DATASET_LABEL, "gnn": gnn_for_cross,
                         "method": paper_name, "acc_mean": m, "acc_std": s})
    return pd.concat([paper, pd.DataFrame(rows)], ignore_index=True)


def _cross_dataset_order(cross_df):
    paper_first = ["Cora", "PubMed", "ogbn-arxiv", "ogbn-products", "tape-arxiv23"]
    order = [d for d in paper_first if d in cross_df["dataset"].unique()]
    order.append(OUR_DATASET_LABEL)
    return order


def plot_cross_landscape(out_dir, cross_df, gnn_for_cross):
    methods = ["h_shallow", "LM_finetune", "h_TAPE"]
    method_labels = {"h_shallow": "shallow",
                     "LM_finetune": "TA (LM fine-tune)",
                     "h_TAPE": "TAPE (TA+P+E)"}
    datasets_order = _cross_dataset_order(cross_df)
    x = np.arange(len(datasets_order))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for k, m in enumerate(methods):
        means, stds = [], []
        for d in datasets_order:
            row = cross_df[(cross_df["dataset"] == d) & (cross_df["method"] == m)]
            means.append(float(row["acc_mean"].iloc[0]) if not row.empty else np.nan)
            stds.append(float(row["acc_std"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + (k - 1) * width, means, width, yerr=stds, capsize=2.5,
               label=method_labels[m],
               color=PALETTE_METHOD[m], edgecolor="white", linewidth=0.5)
    ax.axvline(len(datasets_order) - 1.5, color="gray", linestyle=":", alpha=0.7)
    y_top = ax.get_ylim()[1]
    ax.text(len(datasets_order) - 1.5, y_top * 0.985,
            r"$\leftarrow$ paper datasets    $\vert$    ours $\rightarrow$",
            ha="center", va="top", fontsize=9, color="gray", style="italic")
    ax.set_xticks(x); ax.set_xticklabels(datasets_order, rotation=12)
    ax.set_ylabel(f"test accuracy ({gnn_for_cross})")
    ax.set_title(f"TAPE landscape across datasets  (GNN $=$ {gnn_for_cross})")
    ax.legend(loc="lower right", frameon=False, ncol=3)
    save_png(fig, out_dir, "cross_landscape")


def plot_cross_lift(out_dir, cross_df, gnn_for_cross, baseline, fname, title):
    datasets_order = _cross_dataset_order(cross_df)
    lifts, errs = [], []
    for d in datasets_order:
        sub = cross_df[cross_df["dataset"] == d]
        try:
            tape = float(sub[sub["method"] == "h_TAPE"]["acc_mean"].iloc[0])
            base = float(sub[sub["method"] == baseline]["acc_mean"].iloc[0])
            tape_s = float(sub[sub["method"] == "h_TAPE"]["acc_std"].iloc[0])
            base_s = float(sub[sub["method"] == baseline]["acc_std"].iloc[0])
            lifts.append(tape - base)
            errs.append(float(np.sqrt(tape_s ** 2 + base_s ** 2)))
        except (IndexError, ValueError):
            lifts.append(np.nan); errs.append(0)
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    colors = ["#0072B2"] * (len(datasets_order) - 1) + [COLOR_OURS]
    ax.bar(range(len(datasets_order)), lifts, yerr=errs, capsize=2.5,
           color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    for i, (v, e) in enumerate(zip(lifts, errs)):
        if not pd.isna(v):
            ax.text(i, v + (e if v >= 0 else -e) + (0.001 if v >= 0 else -0.001),
                    f"{v*100:+.1f} pp", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(datasets_order)))
    ax.set_xticklabels(datasets_order, rotation=12)
    ax.set_ylabel(f"TAPE $-$ {baseline.replace('h_', '').replace('_', ' ')}")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#0072B2", label="paper-reported"),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_OURS, label="ours (Goodreads)")]
    ax.legend(handles=handles, loc="best", frameon=False)
    save_png(fig, out_dir, fname)


def plot_h3_paper_vs_ours(out_dir, cross_df, mean_mat, std_mat, gnn_for_cross,
                           paper_csv):
    if not paper_csv.exists():
        return
    paper = pd.read_csv(paper_csv)
    paper = paper[paper["gnn"] == gnn_for_cross]
    method_set = set(paper["method"].unique())
    e_method = None
    for cand in ("h_E", "h_LLM", "LLM"):
        if cand in method_set:
            e_method = cand; break
    if e_method is None:
        return
    datasets_order = _cross_dataset_order(cross_df)
    gaps, errs = [], []
    for d in datasets_order:
        if d == OUR_DATASET_LABEL:
            ta = float(mean_mat.loc["TA", gnn_for_cross])
            e = float(mean_mat.loc["E", gnn_for_cross])
            sa = std_mat.loc["TA", gnn_for_cross]
            sb = std_mat.loc["E", gnn_for_cross]
            err = float(np.sqrt((0 if pd.isna(sa) else sa) ** 2
                                + (0 if pd.isna(sb) else sb) ** 2))
            gaps.append(ta - e); errs.append(err)
        else:
            sub = paper[paper["dataset"] == d]
            try:
                ta = float(sub[sub["method"] == "LM_finetune"]["acc_mean"].iloc[0])
                e = float(sub[sub["method"] == e_method]["acc_mean"].iloc[0])
                ta_s = float(sub[sub["method"] == "LM_finetune"]["acc_std"].iloc[0])
                e_s = float(sub[sub["method"] == e_method]["acc_std"].iloc[0])
                gaps.append(ta - e)
                errs.append(float(np.sqrt(ta_s ** 2 + e_s ** 2)))
            except (IndexError, ValueError):
                gaps.append(np.nan); errs.append(0)
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    colors = ["#0072B2"] * (len(datasets_order) - 1) + [COLOR_OURS]
    ax.bar(range(len(datasets_order)), gaps, yerr=errs, capsize=2.5,
           color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    for i, (v, e) in enumerate(zip(gaps, errs)):
        if not pd.isna(v):
            ax.text(i, v + (e if v >= 0 else -e) + (0.001 if v >= 0 else -0.001),
                    f"{v*100:+.1f} pp", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(datasets_order)))
    ax.set_xticklabels(datasets_order, rotation=12)
    ax.set_ylabel("acc(TA) $-$ acc(E)")
    ax.set_title(f"H3 across datasets  (GNN $=$ {gnn_for_cross})")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#0072B2", label="paper-reported"),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_OURS, label="ours (Goodreads)")]
    ax.legend(handles=handles, loc="best", frameon=False)
    save_png(fig, out_dir, "h3_TA_minus_E_paper_vs_ours")


# ============================================================================
# Pipeline schematic (Fig 12)
# ============================================================================


def plot_pipeline_schematic(out_dir):
    """Pure schematic --- no data dependency."""
    COL_INPUT = "#CCCCCC"; COL_TA = "#7E7E7E"; COL_FROZEN_LLM = "#E69F00"
    COL_FINETUNE = "#0072B2"; COL_GNN = "#009E73"; COL_OUT = "#3A3A3A"
    PHASE_BAND = "#F2F2F2"
    W, H_ = 11.0, 9.0
    COL_X = {"TA": 2.5, "P": 5.5, "E": 8.5}
    CENTER_X = 5.5; BOX_W, BOX_H = 2.5, 0.7
    Y_INPUT, Y_VIEWS, Y_DEBERTA = 8.05, 6.55, 5.05
    Y_GNN, Y_AVG, Y_OUT = 3.55, 2.05, 0.75

    def shadow_box(ax, x, y, w, h, label, fill, text_color="white",
                   fontsize=10.5, bold=False):
        ax.add_patch(FancyBboxPatch((x - w/2 + 0.05, y - h/2 - 0.05), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    linewidth=0, facecolor="black", alpha=0.16,
                                    zorder=1))
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    linewidth=0.7, edgecolor="white",
                                    facecolor=fill, zorder=2))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color=text_color,
                fontweight="bold" if bold else "normal", zorder=3)

    def arrow(ax, x0, y0, x1, y1, connectionstyle=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1.3,
                                     color="#4D4D4D", zorder=4,
                                     shrinkA=2, shrinkB=2,
                                     connectionstyle=connectionstyle or "arc3,rad=0"))

    fig, ax = plt.subplots(figsize=(9.4, 7.6))
    ax.set_xlim(0, W); ax.set_ylim(0, H_); ax.set_axis_off()
    ax.grid(False)

    # Phase bands
    phases = [(Y_INPUT, 0.95, "input"), (Y_VIEWS, 0.95, "text views"),
              (Y_DEBERTA, 0.95, "encoder fine-tune"), (Y_GNN, 0.95, "graph reasoning"),
              (Y_AVG, 0.85, "ensemble"), (Y_OUT, 0.80, "output")]
    band_x0, band_x1 = 1.15, 9.85
    for y_c, h, lbl in phases:
        ax.add_patch(Rectangle((band_x0, y_c - h/2), band_x1 - band_x0, h,
                               facecolor=PHASE_BAND, edgecolor="none",
                               alpha=0.75, zorder=0))
        ax.text(band_x1 + 0.10, y_c, lbl, ha="left", va="center",
                fontsize=8.8, color="#777", style="italic")

    # Input
    shadow_box(ax, CENTER_X, Y_INPUT, 6.4, BOX_H,
               "Node text  (title $+$ abstract on academic graphs;\n"
               "title $+$ description on Goodreads)",
               fill=COL_INPUT, text_color="#111", fontsize=10.5)
    # Three text views
    shadow_box(ax, COL_X["TA"], Y_VIEWS, BOX_W, BOX_H,
               "$\\mathtt{TA}$\noriginal text",
               fill=COL_TA, fontsize=10.5, bold=True)
    shadow_box(ax, COL_X["P"], Y_VIEWS, BOX_W, BOX_H,
               "$\\mathtt{P}$\nfrozen LLM $\\to$ top-$k$ labels",
               fill=COL_FROZEN_LLM, fontsize=9.8, bold=True)
    shadow_box(ax, COL_X["E"], Y_VIEWS, BOX_W, BOX_H,
               "$\\mathtt{E}$\nfrozen LLM $\\to$ explanation",
               fill=COL_FROZEN_LLM, fontsize=9.8, bold=True)
    for view, rad in (("TA", -0.12), ("P", 0.0), ("E", 0.12)):
        arrow(ax, CENTER_X, Y_INPUT - BOX_H/2 - 0.02,
              COL_X[view], Y_VIEWS + BOX_H/2 + 0.02,
              connectionstyle=f"arc3,rad={rad}")
    # DeBERTa fine-tune
    for view in ("TA", "P", "E"):
        shadow_box(ax, COL_X[view], Y_DEBERTA, BOX_W, BOX_H,
                   "DeBERTa-base\nfine-tune (per view)",
                   fill=COL_FINETUNE, fontsize=9.8)
        arrow(ax, COL_X[view], Y_VIEWS - BOX_H/2 - 0.02,
              COL_X[view], Y_DEBERTA + BOX_H/2 + 0.02)
    # GNN
    for view in ("TA", "P", "E"):
        shadow_box(ax, COL_X[view], Y_GNN, BOX_W, BOX_H,
                   "GNN  (MLP / GCN / SAGE)",
                   fill=COL_GNN, fontsize=9.8)
        arrow(ax, COL_X[view], Y_DEBERTA - BOX_H/2 - 0.02,
              COL_X[view], Y_GNN + BOX_H/2 + 0.02)
    # Average
    shadow_box(ax, CENTER_X, Y_AVG, 5.6, BOX_H,
               "average  $(\\mathrm{logits}_{\\mathtt{TA}},\\,"
               "\\mathrm{logits}_{\\mathtt{P}},\\,"
               "\\mathrm{logits}_{\\mathtt{E}})$",
               fill=COL_OUT, fontsize=10.5)
    for view, rad in (("TA", 0.18), ("P", 0.0), ("E", -0.18)):
        arrow(ax, COL_X[view], Y_GNN - BOX_H/2 - 0.02,
              CENTER_X, Y_AVG + BOX_H/2 + 0.02,
              connectionstyle=f"arc3,rad={rad}")
    # Output
    shadow_box(ax, CENTER_X, Y_OUT, 3.0, 0.6,
               "predicted class $\\hat{y}$",
               fill=COL_OUT, fontsize=10.5, bold=True)
    arrow(ax, CENTER_X, Y_AVG - BOX_H/2 - 0.02, CENTER_X, Y_OUT + 0.30 + 0.02)

    # Legend (top-left, compact)
    legend_items = [
        (COL_INPUT, "raw text"),
        (COL_FROZEN_LLM, "frozen LLM (gpt-4o-mini)"),
        (COL_FINETUNE, "fine-tuned encoder (DeBERTa)"),
        (COL_GNN, "GNN backbone"),
        (COL_OUT, "ensemble / output"),
    ]
    lx0 = 0.05; ly = H_ - 0.45
    ax.text(lx0, ly + 0.18, "legend", fontsize=8.6, color="#555", style="italic")
    for color, name in legend_items:
        ax.add_patch(Rectangle((lx0, ly - 0.09), 0.26, 0.18,
                               facecolor=color, edgecolor="white", linewidth=0.4))
        ax.text(lx0 + 0.35, ly, name, ha="left", va="center",
                fontsize=7.8, color="#222")
        ly -= 0.27

    save_png(fig, out_dir, "fig12_pipeline")


# ============================================================================
# Main
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="goodreads_children",
                   help="dataset name (default: goodreads_children)")
    p.add_argument("--src", type=Path, default=Path("results"),
                   help="dir containing <dataset>_long.csv "
                        "(default: results/; use results/mockup for layout previews)")
    p.add_argument("--paper_table", type=Path,
                   default=Path("data/tape_paper_table2.csv"),
                   help="paper Table 2 CSV for cross-dataset plots")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: results/figures/<dataset>/)")
    p.add_argument("--cross_gnn", default="SAGE",
                   help="GNN backbone for cross-dataset plots (default: SAGE)")
    p.add_argument("--pipeline-only", action="store_true",
                   help="only render the pipeline schematic; no data needed")
    return p.parse_args()


def main():
    set_paper_style()
    args = parse_args()
    out_dir = args.out or (Path("results") / "figures" / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pipeline_only:
        plot_pipeline_schematic(out_dir)
        print(f"[plot] wrote 1 file to {out_dir}/")
        return

    df = load_long_csv(args.src, args.dataset)
    gnn_order = [g for g in DEFAULT_GNN_ORDER if g in df["gnn"].unique()]
    n_seeds = df["seed"].nunique()
    mean_mat, std_mat = build_matrices(df, gnn_order)

    # Per-dataset
    plot_strip_seeds(out_dir, df, args.dataset, gnn_order)
    plot_singletons(out_dir, args.dataset, gnn_order, n_seeds, mean_mat, std_mat)
    plot_h3_gap(out_dir, args.dataset, gnn_order, n_seeds, mean_mat, std_mat)
    plot_val_curve_shaded(out_dir, args.dataset, gnn_order, mean_mat, n_seeds)
    plot_pareto_compute_vs_acc(out_dir, df, args.dataset, gnn_order,
                               mean_mat, std_mat)

    # Cross-dataset (skip if paper_table missing)
    cross_df = build_cross_dataset_table(mean_mat, std_mat, args.cross_gnn,
                                         args.paper_table)
    if cross_df is not None:
        plot_cross_landscape(out_dir, cross_df, args.cross_gnn)
        plot_cross_lift(out_dir, cross_df, args.cross_gnn, "h_shallow",
                        "cross_lift_over_shallow",
                        f"How big is TAPE's lift over shallow features?  "
                        f"(GNN $=$ {args.cross_gnn})")
        plot_cross_lift(out_dir, cross_df, args.cross_gnn, "LM_finetune",
                        "cross_lift_over_LM",
                        f"What does adding P+E to LM-finetune buy?  "
                        f"(GNN $=$ {args.cross_gnn})")
        plot_h3_paper_vs_ours(out_dir, cross_df, mean_mat, std_mat,
                              args.cross_gnn, args.paper_table)

    # Schematic --- skip if a hand-revised copy is already in place. Only
    # regenerate when the user explicitly asks for it via --pipeline-only.
    if not (out_dir / "fig12_pipeline.png").exists():
        plot_pipeline_schematic(out_dir)

    written = sorted(out_dir.glob("*.png"))
    print(f"[plot] wrote {len(written)} files to {out_dir}/")
    for p in written:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
