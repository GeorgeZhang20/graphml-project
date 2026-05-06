"""Run one GNN cell via core.trainGNN (or core.trainEnsemble) and dump a
JSON result file at the path scripts/aggregate_results.py expects.

This is the same wrapper notebooks/e2_goodreads.ipynb does inline; we factor
it into a script so the shell pipelines (03_run_new_dataset.sh,
04_ablations.sh, 05_run_a5_ablation.sh) can produce aggregator-compatible
output without depending on Colab.

Output layout (consumed by aggregate_results.py):
  results/runs/<DATASET>/seed_<SEED>/<FEAT>_<GNN>.json     # one record
  results/runs/<DATASET>/seed_<SEED>/<FEAT>_<GNN>.log      # full stdout+stderr

Usage (run from project root):
  python scripts/_run_gnn_cell.py \
      --dataset goodreads_children --seed 0 --gnn GCN --feature TA
  python scripts/_run_gnn_cell.py \
      --dataset goodreads_children --seed 0 --gnn GCN --feature TA_P_E
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ENSEMBLE_PATTERN = re.compile(r"_")  # any feature_type with `_` is an ensemble
ACC_RX = re.compile(r"ValAcc:\s*([0-9.]+),\s*TestAcc:\s*([0-9.]+)")


def parse_last_acc(log: str):
    matches = ACC_RX.findall(log)
    if not matches:
        return None, None
    return float(matches[-1][0]), float(matches[-1][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gnn", required=True)
    ap.add_argument("--feature", required=True,
                    help="ogb / TA / P / E (single-feature) or "
                         "TA_P / TA_E / P_E / TA_P_E (ensemble)")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="default: results/runs/<dataset>/seed_<seed>/")
    ap.add_argument("--use_gpt", action="store_true",
                    help="extra arg used by core.trainEnsemble for cached LM "
                         "explanation .emb files; currently unused for trainGNN.")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra `key value` overrides forwarded to TAPE config "
                         "(e.g. --extra gnn.train.lr 0.002 gnn.train.dropout 0.5).")
    args = ap.parse_args()

    out_dir = args.out_dir or (
        Path("results") / "runs" / args.dataset / f"seed_{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / f"{args.feature}_{args.gnn}.json"
    log_path = record_path.with_suffix(".log")

    is_ensemble = "_" in args.feature
    entry = "core.trainEnsemble" if is_ensemble else "core.trainGNN"

    # All `python -m core.*` calls have to run from inside TAPE/ so the
    # `core` package resolves.
    cmd = [
        sys.executable, "-m", entry,
        "dataset", args.dataset,
        "gnn.model.name", args.gnn,
        "gnn.train.feature_type", args.feature,
        "seed", str(args.seed),
    ]
    cmd += list(args.extra)

    env = os.environ.copy()
    env.setdefault("WANDB_DISABLED", "True")
    env.setdefault("TOKENIZERS_PARALLELISM", "False")

    print("+", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="TAPE", env=env)
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    val_acc, test_acc = parse_last_acc(log)

    record = {
        "dataset": args.dataset,
        "seed": args.seed,
        "gnn": args.gnn,
        "feature": args.feature,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "wall_seconds": time.time() - t0,
        "returncode": proc.returncode,
    }
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    with open(log_path, "w") as f:
        f.write(log)

    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"  -> {status}: val={val_acc} test={test_acc} ({record['wall_seconds']:.0f}s)")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
