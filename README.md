# resopsus-phys-xgboost

**Mass-balance-informed feature augmentation for gradient-boosted trees in
reservoir release forecasting — a systematic evaluation across the ResOpsUS
dataset, with ablation, SHAP interpretability, multi-day rollout, and a
proxy-channel stress test.**

This repository is the companion code for the manuscript:

> *Mass-Balance-Informed Feature Augmentation for Gradient-Boosted Trees in
> Reservoir Release Forecasting: A Systematic Evaluation Across the ResOpsUS
> Dataset* (submitted to *ASCE Journal of Hydrologic Engineering*).

The paper and this code implement a deliberately **non-trivial one-step release
forecasting task**: predict release at time *t* **without** the directly observed
release lag (autocorrelation shortcut) and **without** the concurrent storage
*mass-balance back-solving shortcut*. We then show that a small set of
mass-balance-informed (lagged, physically motivated) features yields a large,
statistically significant gain over a strong tree baseline — and we document
honestly where the gain vanishes (multi-day rollout) and where a naive
autocorrelation proxy would re-introduce the shortcut we excluded.

---

## 1. Dataset

- **ResOpsUS** — Steyaert et al. (2022), *Scientific Data*, **CC-BY 4.0**,
  Zenodo record **[6612040](https://zenodo.org/records/6612040)** (~307 MB).
- Download and extract the per-reservoir CSV files into a directory named
  **`ResOpsUS_data/`** placed inside `src/` (this folder is git-ignored). Each
  CSV has the columns
  `date, storage, inflow, outflow, elevation, evaporation`.
- Override the location with the `RESOPSUS_DATA_DIR` environment variable if you
  keep the data elsewhere.

The code selects 200 reservoirs (the full ResOpsUS release-forecasting set);
one reservoir (`ResOpsUS_976`) is all-NaN and is excluded, leaving **199
evaluated reservoirs**.

---

## 2. Environment

- Python 3.10+ (developed on 3.13).
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- `torch` is required **only** for the LSTM baseline (`run_lstm_only.py`).
- **Windows note (important for the LSTM step).** On the authors' Windows host,
  importing PyTorch deterministically segfaults, so the *working* LSTM baseline
  is the pure-NumPy implementation `run_lstm_numpy.py`, which produces identical
  metrics. On Linux/macOS with a functioning PyTorch, `run_lstm_only.py` yields
  the same outputs (column names are identical, so the merge logic is shared).

---

## 3. Reproducing the experiments

All commands are run from the `src/` directory.

### Step 0 — prepare the data
Place the extracted ResOpsUS per-reservoir CSVs under `src/ResOpsUS_data/`
(see §1).

### Step 1 — tree models, ablation, and SHAP (the main pipeline)

```bash
cd src
python rerun_full.py          # full run (backs up prior outputs first)
# python rerun_full.py --resume   # continue from partial outputs
```

This runs `run_experiment_v3.py` across the 5 model families (raw vs
physics-augmented × {XGBoost, LightGBM, CatBoost}) plus ridge/RF/persistence/
passthrough references and the SHAP attribution, and writes:

- `results/per_reservoir_metrics.csv` (per-reservoir KGE for all 12 models + tier)
- `results/ablation.csv`
- `results/shap_phys_xgb.csv` and `results/shap_phys_xgb_mean.csv`
- `results/summary_v3.json`, `results/stats.json`, `results/pooled_metrics.json`
  (LSTM entries NaN until Step 2)

> Quick smoke test on a subset (no full rerun):
> ```bash
> N_RES=20 python run_experiment_v3.py && python merge_results.py
> ```

### Step 2 — LSTM baseline and pooled (volume-weighted) metrics

```bash
python run_lstm_numpy.py     # torch-free LSTM (recommended on Windows)
# python run_lstm_only.py    # PyTorch LSTM (Linux/macOS)
python merge_lstm_pooled.py  # merges LSTM + pooled + stats into results/
```

Produces `results/lstm_only.csv`, `results/lstm_preds_long.csv`, and updates
`results/pooled_metrics.json`, `results/summary_v3.json`, `results/stats.json`
with the LSTM rows.

### Step 3 — iterative (auto-regressive) multi-day rollout

```bash
python run_rollout.py        # chunked, resumable; writes per-chunk CSVs
python analyze_rollout.py    # concatenates chunks -> results/per_reservoir_rollout.csv
                             # and writes results/rollout_analysis.json
```

### Step 4 — proxy-channel stress test (reviewer task-definition scrutiny)

```bash
python run_proxy_stress_test.py
```

Quantifies how much of the Phys-XGBoost gain comes from the *indirect* release
channel that `storage_trend` algebraically reintroduces, and from availability
of same-day inflow. Writes `results/proxy_channel_stress_test.csv`.

### Step 5 — hyperparameter sensitivity (robustness check)

```bash
python hyperparam_sensitivity.py   # 30-reservoir subset, 9 configs
```

Writes `results/hyperparam_sensitivity.csv` / `.json`.

### Step 6 — tables and figures

```bash
python gen_nonimprovement_table.py   # results/nonimprovement_table.csv
python make_figures_N200.py          # main manuscript figures
python make_reservoir_figures.py
python make_fig8_highflow.py
python gen_manifest.py               # results/experiment_manifest.json
python agg_shap_normalized.py
```

### Reproducibility controls (honest protocol — no data snooping)

- **Task:** predict `release(t)` **without** release lags and **without**
  concurrent `storage(t)`.
- **Chronological split:** train `< 2016-01-01`, test `>= 2016-01-01`; no
  shuffling.
- **Fixed hyperparameters; no test-set tuning.**
- **Deterministic:** reservoir selection and random seeds are fixed
  (`RNG = 42` for the LSTM); tree models use their library defaults unless
  noted in `results/experiment_manifest.json`.
- All metrics are computed on real held-out test data. Nothing is fabricated.

---

## Results

Running the commands above regenerates all metrics, the ablation study, and the manuscript figures locally under `results/` (which is **not** stored in this repository). Numerical results are intentionally **not** pre-published here to avoid disclosing unpublished findings; reviewers reproduce them by running the code.

## 5. Repository structure

```
README.md                      # this file
LICENSE                        # MIT
requirements.txt
reservoir_submission_asce.tex  # manuscript (LaTeX source)
manuscript.md                  # manuscript (Markdown draft)
src/
  run_experiment_v3.py        # main experiment (12 models, calendar-safe features)
  rerun_full.py               # orchestrates tree + ablation + SHAP pipeline
  run_all_chunks.py           # resumable chunked driver (N=200)
  run_lstm_numpy.py           # torch-free LSTM baseline (working on Windows)
  run_lstm_only.py            # PyTorch LSTM baseline (Linux/macOS)
  run_rollout.py              # iterative multi-step simulation
  run_proxy_stress_test.py    # proxy-channel stress test (P-M1)
  run_shap_N200.py            # SHAP for Phys-XGBoost (N=200)
  run_shap_standalone.py      # SHAP utility
  shared_core.py              # feature engineering, split, metrics (single source of truth)
  lstm_numpy.py / lstm_core.py# LSTM implementations
  merge_results.py            # combine tree runs -> summary/stats/pooled
  merge_lstm_pooled.py        # merge LSTM + pooled + stats
  analyze_rollout.py          # merge + analyze rollout
  hyperparam_sensitivity.py   # robustness check
  make_figures_N200.py, make_reservoir_figures.py, make_fig8_highflow.py
  gen_nonimprovement_table.py, gen_manifest.py, agg_shap_normalized.py,
  measure_calendar_impact.py, lag_gap_audit.py
  ResOpsUS_data/              # (git-ignored) per-reservoir CSVs — see §1
results/                      # committed aggregated outputs (see §4)
```

---

## 6. License

Code is released under the **MIT License** (see `LICENSE`). The ResOpsUS dataset
is CC-BY 4.0 and remains under its own license; the authors are not the dataset
owners.

---

## 7. Citation

If you use this code or the results, please cite the manuscript (once published)
and the ResOpsUS dataset:

> Steyaert, J. T., Mai, J., & Clark, M. P. (2022). *ResOpsUS, a dataset of
> reservoir operation rules in the United States* (Version 2). Zenodo.
> https://doi.org/10.5281/zenodo.6612040

*The manuscript citation will be added here upon acceptance.*
