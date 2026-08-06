"""
run_lstm_numpy.py -- torch-free LSTM baseline for the ResOpsUS physics-feature
study. Produces the SAME outputs as run_lstm_only.py (lstm_only.csv +
lstm_preds_long.csv) but trains a pure-NumPy LSTM (see lstm_numpy.py) instead of
PyTorch, because on this Windows host PyTorch's C extension segfaults on load.

Outputs (RESULTS_DIR, default src/results):
  lstm_only.csv        -- per-reservoir LSTM(raw) & LSTM(phys) metrics
  lstm_preds_long.csv  -- per-test-sample (reservoir, obs, lstm, lstmphys)
Column names match run_lstm_only exactly so downstream merge/table logic is shared.
"""
import os
import glob
import time
import numpy as np
import pandas as pd

from lstm_numpy import build_lstm_windows, train_lstm
from shared_core import build_features, split_chrono, all_metrics

RNG = 42
np.random.seed(RNG)

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
    print(f"[lstm_numpy] {len(names)} reservoirs selected (N_RES={N})", flush=True)

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
        sim_r = np.full(len(test), np.nan, dtype=float)
        try:
            Xr_tr, yr_tr, vr_tr = build_lstm_windows(train, use_phys=False)
            Xr_te, yr_te, vr_te = build_lstm_windows(test, use_phys=False)
            if len(Xr_tr) > 200 and len(Xr_te) > 20:
                sim = train_lstm(Xr_tr, yr_tr, Xr_te, yr_te, Xr_tr.shape[2])
                sim_r[vr_te] = sim
        except Exception as e:
            print(f"  [{i+1}] {name} LSTM-raw FAILED: {e}", flush=True)
        sim_p = np.full(len(test), np.nan, dtype=float)
        try:
            Xp_tr, yp_tr, vp_tr = build_lstm_windows(train, use_phys=True)
            Xp_te, yp_te, vp_te = build_lstm_windows(test, use_phys=True)
            if len(Xp_tr) > 200 and len(Xp_te) > 20:
                sim = train_lstm(Xp_tr, yp_tr, Xp_te, yp_te, Xp_tr.shape[2])
                sim_p[vp_te] = sim
        except Exception as e:
            print(f"  [{i+1}] {name} LSTM-phys FAILED: {e}", flush=True)
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
              f"lstmphys_KGE={row.get('lstmphys_KGE',float('nan')):.3f}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "lstm_only.csv"), index=False)
    pd.DataFrame(long_rows).to_csv(os.path.join(OUT_DIR, "lstm_preds_long.csv"), index=False)
    print(f"[lstm_numpy] wrote {os.path.join(OUT_DIR, 'lstm_only.csv')} and lstm_preds_long.csv", flush=True)


if __name__ == "__main__":
    main()
