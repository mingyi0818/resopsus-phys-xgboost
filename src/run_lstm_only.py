"""
run_lstm_only.py -- computes the LSTM (raw & phys) results for the ResOpsUS
physics-feature study in a TORCH-ONLY process, isolated from xgboost/lightgbm/
catboost (loading those together with torch in the same process triggers an
OpenMP-runtime segfault on Windows).

It reuses the feature engineering, chronological split, metrics, and the EXACT
reservoir-selection logic from run_experiment_v3.py (imported with SKIP_LSTM=1 so
no tree libraries and no torch are loaded there). The selection is deterministic
and matches the trees-only run (same N_RES), so the two outputs align 1:1 by
reservoir and are merged by merge_results.py.

Outputs (RESULTS_DIR, default src/results):
  lstm_only.csv        -- per-reservoir LSTM(raw) & LSTM(phys) metrics
  lstm_preds_long.csv  -- per-test-sample (reservoir, obs, lstm, lstmphys) for pooled KGE
"""

import os, glob, json
# NOTE: do NOT set OMP_NUM_THREADS / KMP_DUPLICATE_LIB_OK / MKL_NUM_THREADS here.
# On this Windows host those env vars make the torch+numpy import segfault
# deterministically (EXIT=139); the torch-only process is stable WITHOUT them.
# (Those flags are only needed for the trees-only process that co-loads
# xgboost/lightgbm/catboost, which run_lstm_only deliberately avoids.)
# IMPORT TORCH FIRST, before numpy/pandas, so torch's OpenMP runtime inits
# before numpy's; lstm_core/shared_core only re-import already-loaded modules.
import torch
import torch.nn as nn  # noqa: F401 (kept for symmetry / future use)
import numpy as np
import pandas as pd

from lstm_core import build_lstm_windows, train_lstm
# Reuse the SAME torch-free feature/split/metrics from shared_core (single source
# of truth). We deliberately do NOT import run_experiment_v3 here: that module
# loads xgboost/lightgbm/catboost and co-loading them with torch segfaults on
# Windows (OpenMP-runtime conflict).
from shared_core import build_features, split_chrono, all_metrics

RNG = 42
np.random.seed(RNG)
torch.manual_seed(RNG)

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT_DIR = os.path.join(HERE, "ResOpsUS_data")
OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
os.makedirs(OUT_DIR, exist_ok=True)


def select_reservoirs(N):
    csvs = sorted(glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True))
    sel = []
    for c in csvs:
        try:
            df = pd.read_csv(c)
        except Exception:
            continue
        if not {"date", "inflow", "outflow", "storage"}.issubset(df.columns):
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        comp = df.dropna(subset=["inflow", "outflow", "storage"])
        if len(comp) < 5000:
            continue
        post = comp[comp["date"] >= pd.Timestamp("2016-01-01")]
        pre = comp[comp["date"] < pd.Timestamp("2016-01-01")]
        if len(post) < 500 or len(pre) < 3000:
            continue
        sel.append((os.path.splitext(os.path.basename(c))[0], len(comp)))
    sel.sort(key=lambda x: -x[1])
    if N < len(sel):
        idx = np.linspace(0, len(sel) - 1, N).round().astype(int)
        sel = [sel[i] for i in idx]
    return [s[0] for s in sel]


def main():
    N = int(os.environ.get("N_RES", "30"))
    names = select_reservoirs(N)
    print(f"[lstm_only] {len(names)} reservoirs selected (N_RES={N})")

    rows = []
    long_rows = []
    for i, name in enumerate(names):
        c = os.path.join(EXTRACT_DIR, name + ".csv")
        df = pd.read_csv(c)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["inflow", "outflow", "storage"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["date", "inflow", "outflow", "storage"]]
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        df = df.dropna(subset=["outflow", "inflow", "storage"])
        capacity = df["storage"].quantile(0.99)
        mean_inflow = df["inflow"].mean()
        d = build_features(df, capacity, mean_inflow)
        train, test = split_chrono(d)
        y_te = test["outflow"].values
        row = {"reservoir": name, "n_train": len(train), "n_test": len(test)}
        # raw LSTM
        sim_r = np.full(len(test), np.nan, dtype=float)
        try:
            Xr_tr, yr_tr, vr_tr = build_lstm_windows(train, use_phys=False)
            Xr_te, yr_te, vr_te = build_lstm_windows(test, use_phys=False)
            if len(Xr_tr) > 200 and len(Xr_te) > 20:
                sim = train_lstm(Xr_tr, yr_tr, Xr_te, yr_te, Xr_tr.shape[2])
                sim_r[vr_te] = sim
        except Exception as e:
            print(f"  [{i+1}] {name} LSTM-raw FAILED: {e}")
        # phys LSTM
        sim_p = np.full(len(test), np.nan, dtype=float)
        try:
            Xp_tr, yp_tr, vp_tr = build_lstm_windows(train, use_phys=True)
            Xp_te, yp_te, vp_te = build_lstm_windows(test, use_phys=True)
            if len(Xp_tr) > 200 and len(Xp_te) > 20:
                sim = train_lstm(Xp_tr, yp_tr, Xp_te, yp_te, Xp_tr.shape[2])
                sim_p[vp_te] = sim
        except Exception as e:
            print(f"  [{i+1}] {name} LSTM-phys FAILED: {e}")
        for mdl, sim in [("lstm", sim_r), ("lstmphys", sim_p)]:
            for k, v in all_metrics(y_te, sim).items():
                row[f"{mdl}_{k}"] = v
        rows.append(row)
        for j in range(len(test)):
            long_rows.append({
                "reservoir": name,
                "obs": y_te[j],
                "lstm": sim_r[j],
                "lstmphys": sim_p[j],
            })
        print(f"  [{i+1}/{len(names)}] {name}: lstm_KGE={row.get('lstm_KGE',float('nan')):.3f} "
              f"lstmphys_KGE={row.get('lstmphys_KGE',float('nan')):.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "lstm_only.csv"), index=False)
    pd.DataFrame(long_rows).to_csv(os.path.join(OUT_DIR, "lstm_preds_long.csv"), index=False)
    print(f"[lstm_only] wrote {os.path.join(OUT_DIR, 'lstm_only.csv')} and lstm_preds_long.csv")


if __name__ == "__main__":
    main()
