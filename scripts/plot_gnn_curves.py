"""Parse GNN training stdout logs and plot val/train curves.

Reads `results/runs/<DATASET>/seed_*/<feat>_<gnn>.log` files (which contain
the LOG_FREQ=10 progress lines printed by `core.GNNs.gnn_trainer.GNNTrainer.train`),
extracts (epoch, loss, train_acc, val_acc) per epoch, and produces:

  results/figures/<dataset_subdir>/<DATASET>_gnn_val_curves.png   val_acc vs epoch (5 features × 3 GNNs grid)
  results/figures/<dataset_subdir>/<DATASET>_gnn_loss_curves.png  train loss vs epoch (same grid)

Run from project root:
  python scripts/plot_gnn_curves.py --dataset goodreads_children
"""
from __future__ import annotations
import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LINE_RX = re.compile(
    r"Epoch:\s+(?P<epoch>\d+),\s+Time:\s+(?P<time>[\d.]+),\s+"
    r"Loss:\s+(?P<loss>[\d.]+),\s+"
    r"TrainAcc:\s+(?P<train_acc>[\d.]+),\s+"
    r"ValAcc:\s+(?P<val_acc>[\d.]+)"
)

FEATURE_ORDER = ["ogb", "TA", "P", "E", "TA_P_E"]
DEFAULT_GNN_ORDER = ["MLP", "GCN", "SAGE"]


def parse_log(path: Path) -> dict:
    """Return {'epoch': [...], 'loss': [...], 'train_acc': [...], 'val_acc': [...]}."""
    out = defaultdict(list)
    text = path.read_text(errors="ignore")
    for m in LINE_RX.finditer(text):
        out["epoch"].append(int(m.group("epoch")))
        out["loss"].append(float(m.group("loss")))
        out["train_acc"].append(float(m.group("train_acc")))
        out["val_acc"].append(float(m.group("val_acc")))
    return dict(out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--runs_dir", type=Path, default=None,
                   help="default: results/runs/<DATASET>/")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="default: results/figures/full_76k/ (or n1500/ if dataset name contains _n)")
    return p.parse_args()


def grid_plot(curves, metric: str, ylim, ylabel, title, save_path: Path,
              gnn_order: list[str]):
    """One subplot per (feature, gnn). Multiple seeds overlaid with thin lines."""
    n_rows = len(FEATURE_ORDER)
    n_cols = len(gnn_order)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols, 1.8 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    palette = plt.cm.tab10.colors
    for i, feat in enumerate(FEATURE_ORDER):
        for j, gnn in enumerate(gnn_order):
            ax = axes[i][j]
            cells = curves.get((feat, gnn), [])
            if not cells:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                if i == 0:
                    ax.set_title(gnn, fontsize=10)
                if j == 0:
                    ax.set_ylabel(feat, fontsize=10, rotation=0,
                                  labelpad=20, va="center")
                continue
            for k, (seed, data) in enumerate(sorted(cells)):
                ax.plot(data["epoch"], data[metric],
                        color=palette[k % len(palette)], alpha=0.7,
                        linewidth=1.2, label=f"seed {seed}")
            if i == 0:
                ax.set_title(gnn, fontsize=10)
            if j == 0:
                ax.set_ylabel(feat, fontsize=10, rotation=0, labelpad=20,
                              va="center")
            ax.grid(alpha=0.3)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if i == n_rows - 1:
                ax.set_xlabel("epoch", fontsize=9)
    # legend on the figure (just one)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right",
                   bbox_to_anchor=(1.0, 1.0), fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.005, ylabel, ha="center", fontsize=10)
    plt.tight_layout(rect=[0.03, 0.02, 0.97, 0.97])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    runs_dir = args.runs_dir or (Path("results") / "runs" / args.dataset)
    if not runs_dir.exists():
        raise SystemExit(f"{runs_dir} does not exist; run cell 15 on Colab first.")

    # Decide output subfolder by dataset slug
    if args.out_dir is not None:
        fig_dir = args.out_dir
    elif "_n" in args.dataset:
        # e.g. goodreads_children_n1500
        slug = args.dataset.split("_n")[-1]
        fig_dir = Path("results") / "figures" / f"n{slug}"
    else:
        fig_dir = Path("results") / "figures" / "full_76k"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Collect curves: (feature, gnn) -> [(seed, parsed_dict), ...]
    curves: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    gnns_seen = set()
    seeds_seen = set()
    for log_path in sorted(runs_dir.glob("seed_*/*.log")):
        # filename: <feat>_<gnn>.log
        feat_gnn = log_path.stem  # e.g. "TA_P_E_GCN"
        # split on last _ to extract gnn
        parts = feat_gnn.rsplit("_", 1)
        if len(parts) != 2:
            print(f"[skip] cannot parse {log_path.name}")
            continue
        feat, gnn = parts
        seed_dir = log_path.parent.name  # "seed_0"
        try:
            seed = int(seed_dir.split("_")[-1])
        except Exception:
            continue
        data = parse_log(log_path)
        if not data.get("epoch"):
            print(f"[empty] {log_path}")
            continue
        curves[(feat, gnn)].append((seed, data))
        gnns_seen.add(gnn)
        seeds_seen.add(seed)

    if not curves:
        raise SystemExit(f"no parseable logs found under {runs_dir}/seed_*/")

    gnn_order = [g for g in DEFAULT_GNN_ORDER if g in gnns_seen]
    extras = sorted(g for g in gnns_seen if g not in DEFAULT_GNN_ORDER)
    gnn_order = gnn_order + extras
    print(f"[plot] dataset={args.dataset}  seeds={sorted(seeds_seen)}  "
          f"gnns={gnn_order}  cells={len(curves)}")

    # ----- val accuracy grid -----
    grid_plot(
        curves, metric="val_acc", ylim=None,
        ylabel="validation accuracy",
        title=f"GNN val-acc training curves on {args.dataset}",
        save_path=fig_dir / f"{args.dataset}_gnn_val_curves.png",
        gnn_order=gnn_order,
    )
    print(f"[plot] wrote {fig_dir}/{args.dataset}_gnn_val_curves.png")

    # ----- train loss grid -----
    grid_plot(
        curves, metric="loss", ylim=None,
        ylabel="train loss",
        title=f"GNN train-loss curves on {args.dataset}",
        save_path=fig_dir / f"{args.dataset}_gnn_loss_curves.png",
        gnn_order=gnn_order,
    )
    print(f"[plot] wrote {fig_dir}/{args.dataset}_gnn_loss_curves.png")


if __name__ == "__main__":
    main()
