# scripts/

Headless utilities. Run from the repo root.

| script | role | input | output |
|---|---|---|---|
| `aggregate_results.py` | turn per-seed JSONs into a long/summary CSV | `results/runs/<dataset>/seed_*/*.json` | `results/<dataset>_long.csv` + `_summary.csv` |
| `make_plots.py` | render every figure used in the paper | `results/<dataset>_long.csv` (+ `data/tape_paper_table2.csv` for cross-dataset plots) | `results/figures/<dataset>/*.png` (or wherever `--out` points) |
| `plot_gnn_curves.py` | parse training stdout logs into val/loss curve grids | `results/runs/<dataset>/seed_*/*.log` | `results/figures/<subdir>/<dataset>_gnn_*_curves.png` |
| `00_setup_env.sh` ... `04_ablations.sh` | one-step shell helpers for the major runs | — | — |

## Typical pipeline

```sh
# 1. Run experiments (notebooks/e2_goodreads.ipynb or the scripts/04_*.sh helpers)
# 2. Aggregate results
python3 scripts/aggregate_results.py --dataset goodreads_children

# 3. Render paper figures into paper/figures/ (Overleaf-ready)
python3 scripts/make_plots.py --dataset goodreads_children --out paper/figures
```

## `make_plots.py` figures

Ten outputs (PNG, paper-quality styling):

- `<dataset>_strip_seeds`, `<dataset>_singletons`, `<dataset>_h3_singleton_gap`,
  `<dataset>_val_curves_shaded`, `<dataset>_pareto_compute_vs_acc`
- `cross_landscape`, `cross_lift_over_shallow`, `cross_lift_over_LM`,
  `h3_TA_minus_E_paper_vs_ours`
- `fig12_pipeline` (pure schematic; skipped if `paper/figures/fig12_pipeline.png`
  already exists, so hand-edited versions survive)

To re-render only the schematic:

```sh
python3 scripts/make_plots.py --pipeline-only --out paper/figures
```

## Paths

All defaults are relative to the repo root. Override with `--src`, `--out`,
`--paper_table` as needed; nothing reads absolute paths.
