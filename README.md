# router_nips

NeurIPS-deadline fork of `How2TrainARouter`. See `PIPELINE_NOTES.md` for hard-won
gotchas about the pipeline; **read it before running anything**.

## Quick start (fresh machine)

```bash
# 1. install uv if needed: https://docs.astral.sh/uv/
# 2. set up env, datasets symlink, and smoke-test imports
bash scripts/setup_env.sh --with-datasets-symlink /path/to/shared/datasets

# 3. regression test (must pass before you trust this fork)
bash scripts/regression_test.sh 0   # GPU index, default 0
# expected: PASS, test_routing_accuracy = 0.6903 ± 0.001
```

## Repo layout

```
router_nips/
├── main.py, main_util.py, trainer.py, evaluator.py    # core pipeline (preserved)
├── models/                                            # all router models (used + legacy)
├── data_processors/                                   # NB30k, NB270k, RouterBench, etc.
├── losses/                                            # BCE in trainer + PW (ranking_loss.py)
├── gating/                                            # required by some model imports
├── configs/
│   ├── training_config/                               # 3 canonical configs
│   ├── pipeline_config/                               # populated per-experiment
│   └── _legacy/                                       # don't use
├── scripts/
│   ├── dispatch.py                                    # clean job dispatcher (replaces gpu_chaser)
│   ├── aggregate_results.py                           # sweep_summary → CSV
│   ├── regression_test.sh                             # pipeline correctness check
│   ├── setup_env.sh                                   # fresh-machine setup
│   ├── sync_to_remote.sh                              # rsync to datalab2/Anvil
│   └── legacy/                                        # archived analysis scripts
├── PIPELINE_NOTES.md                                  # READ THIS FIRST
└── README.md (this file)
```

## Universal env vars (ALWAYS set when launching)

```bash
export TIER_AWARE_POOLING=1     # CLS for T2, last-token for T3, native for T1
export L2_NORM_INPUT=1          # L2-norm at MLP input (prevents BCE overflow)
export OPENBLAS_NUM_THREADS=8   # required for kmeans/knn (sklearn loky)
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8
```

The dispatcher (`scripts/dispatch.py`) sets these by default.

## Running an experiment

1. **Build configs** for each cell (one config per (method, encoder, lr, seed) combo).
   Place in `configs/pipeline_config/nips/<exp_name>/`.

2. **Build a manifest** JSON listing all cells:
   ```json
   [
     {
       "log_tag": "nb30k_kmeans_modernbert_b_lr1e-3_seed42",
       "pipeline": "configs/pipeline_config/nips/c1/nb30k_kmeans_mb_lr1e-3_s42.json",
       "training": "configs/training_config/softmax_bce_dft_fast.json"
     },
     ...
   ]
   ```

3. **Dispatch:**
   ```bash
   python scripts/dispatch.py \
       --manifest exp_planning_claude/nips_paper/manifests/c1_main.json \
       --log-dir logs/c1_main
   ```

4. **Aggregate** when done:
   ```bash
   python scripts/aggregate_results.py \
       --since 2026-04-23 \
       --best-routing-only \
       --out exp_planning_claude/nips_paper/results/c1_main.csv
   ```

## Sync to remote machines

```bash
# datalab2 (main portal)
bash scripts/sync_to_remote.sh ubuntu@datalab2.example.com:/home/ubuntu/router_nips

# Anvil (H100, for FT)
bash scripts/sync_to_remote.sh x-ylu27@anvil.rcac.purdue.edu:/anvil/scratch/.../router_nips
```

`sync_to_remote.sh` excludes `.venv`, `outputs`, `logs`, `datasets`, `.git`, etc.
**It does NOT use `--delete`** (per the rsync_delete gotcha in user memory).

After sync, run `bash scripts/setup_env.sh` on the remote.

## Methods that work and have been verified

- `mlp_cost_normalized` — ✅ regression-tested
- `mirt`, `knn`, `kmeans`, `equirouter` — ✅ run in overnight sweep
- `encoder_mlp_combined` — ✅ FT path (used in Phase 1A)
- Other methods import OK but are not on the planned matrix.

For methods being added (RouterDC parity, GraphRouter, HybridLLM, RouteLLM-MF,
CARROT, ELO), see `exp_planning_claude/nips_paper/tasks/methods_to_integrate.md`.
