"""Glob results/runs/<DATASET>/seed_*/<feat>_<gnn>.json and produce a tidy CSV.

Idempotent. Works with any number of seeds present (1 → 4). Used after each
seed's notebook run so we can watch H2/H3 build up incrementally.

Output layout:
  results/<DATASET>_long.csv     one row per (seed, feature, gnn) with val/test acc
  results/<DATASET>_summary.csv  one row per (feature, gnn) with mean ± std and n_seeds
  prints a markdown table to stdout (paste into the report draft as-is)

Usage:
  python scripts/aggregate_results.py --dataset goodreads_children
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--runs_dir", type=Path, default=None,
                   help="default: results/runs/<DATASET>/")
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    args = parse_args()
    runs_dir = args.runs_dir or (Path("results") / "runs" / args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        print(f"[warn] {runs_dir} does not exist yet; nothing to aggregate.")
        return

    rows = []
    for jp in sorted(runs_dir.glob("seed_*/*.json")):
        with open(jp) as f:
            rec = json.load(f)
        rows.append(rec)

    if not rows:
        print(f"[warn] no result JSONs under {runs_dir}/seed_*/")
        return

    df = pd.DataFrame(rows)
    df = df[["dataset", "seed", "feature", "gnn", "val_acc", "test_acc",
             "wall_seconds", "returncode"]]
    long_path = args.out_dir / f"{args.dataset}_long.csv"
    df.to_csv(long_path, index=False)
    print(f"[long] wrote {long_path} ({len(df)} rows)")

    # Drop failures from the summary view
    ok = df[(df["returncode"] == 0) & df["test_acc"].notna()]

    summary = (
        ok.groupby(["feature", "gnn"])
          .agg(
              n_seeds=("seed", "nunique"),
              test_acc_mean=("test_acc", "mean"),
              test_acc_std=("test_acc", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
              val_acc_mean=("val_acc", "mean"),
          )
          .reset_index()
          .sort_values(["gnn", "feature"])
    )
    summary_path = args.out_dir / f"{args.dataset}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[summary] wrote {summary_path}")

    # ---- pretty markdown table for pasting into the report draft ----
    feature_order = ["ogb", "TA", "P", "E", "TA_P_E"]
    gnns = sorted(summary["gnn"].unique())
    print()
    print(f"### Test accuracy on {args.dataset} (mean ± std over n seeds)")
    print()
    header = "| feature | " + " | ".join(gnns) + " |"
    sep = "|---|" + "|".join(["---"] * len(gnns)) + "|"
    print(header)
    print(sep)
    for feat in feature_order:
        cells = [feat]
        for gnn in gnns:
            sub = summary[(summary["feature"] == feat) & (summary["gnn"] == gnn)]
            if sub.empty:
                cells.append("—")
            else:
                m = float(sub["test_acc_mean"].iloc[0])
                s = float(sub["test_acc_std"].iloc[0])
                n = int(sub["n_seeds"].iloc[0])
                if n <= 1:
                    cells.append(f"{m:.4f} (n=1)")
                else:
                    cells.append(f"{m:.4f} ± {s:.4f}")
        print("| " + " | ".join(cells) + " |")

    # also print which seeds we have
    seeds_present = sorted(df["seed"].unique().tolist())
    print(f"\nSeeds present: {seeds_present}")
    if len(seeds_present) < 4:
        missing = [s for s in [0, 1, 2, 3] if s not in seeds_present]
        print(f"Run seeds {missing} next to fill the rest of the table.")


if __name__ == "__main__":
    main()
