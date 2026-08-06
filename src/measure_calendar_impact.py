# -*- coding: utf-8 -*-
"""Measure how many test rows survive under OLD (row-shift) vs NEW
(calendar-safe) feature construction, with strict vs tolerant rolling windows.
No model training -- just feature build + chronological split. Informs the
strict_windows decision for the P-M8 rerun.

Run: python measure_calendar_impact.py
"""
import os, glob, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_experiment_v3 as RE

OUT_DIR = RE.OUT_DIR
EXTRACT_DIR = r"D:\workbuddy\reservoir_experiment\ResOpsUS_data"
pm = pd.read_csv(os.path.join(OUT_DIR, "per_reservoir_metrics.csv"))
evaluated = [r for r in pm["reservoir"].tolist() if r != "ResOpsUS_976"]
csvmap = {os.path.splitext(os.path.basename(c))[0]: c
          for c in glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True)}


def load_cc(name):
    c = csvmap.get(name)
    if not c:
        return None
    df = pd.read_csv(c)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["inflow", "outflow", "storage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["date", "inflow", "outflow", "storage"]]
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["outflow", "inflow", "storage"])
    return df


def old_build(df, capacity, mean_inflow):
    """Replicate the PRE-FIX row-shift build_features for comparison."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["doy"] = d["date"].dt.dayofyear
    d["doy_sin"] = np.sin(2*np.pi*d["doy"]/365.25)
    d["doy_cos"] = np.cos(2*np.pi*d["doy"]/365.25)
    d["inflow_lag1"] = d["inflow"].shift(1)
    d["inflow_lag2"] = d["inflow"].shift(2)
    d["inflow_lag3"] = d["inflow"].shift(3)
    d["storage_lag1"] = d["storage"].shift(1)
    d["storage_lag2"] = d["storage"].shift(2)
    cap = capacity if capacity and capacity > 0 else d["storage"].quantile(0.99)
    d["storage_norm"] = d["storage_lag1"] / cap
    d["storage_deficit"] = d["storage_lag1"] - d["doy"] * 0  # placeholder; not used for count
    d["inflow_norm"] = d["inflow"] / mean_inflow
    d["inflow_ma7"]  = d["inflow"].rolling(7, min_periods=1).mean().shift(1)
    d["inflow_ma30"] = d["inflow"].rolling(30, min_periods=1).mean().shift(1)
    d["inflow_anom7"] = d["inflow"] - d["inflow"].rolling(7, min_periods=1).mean().shift(1)
    d["storage_trend"] = d["storage_lag1"] - d["storage_lag2"]
    d["throughput_ratio"] = d["inflow"] / (d["storage_lag1"].abs() + 1e-6)
    return d


rows = []
tot_old = tot_new_strict = tot_new_tol = 0
n_changed = 0
for i, name in enumerate(evaluated):
    df = load_cc(name)
    if df is None:
        continue
    cap = df["storage"].quantile(0.99)
    mi = df["inflow"].mean()
    # OLD
    d_old = old_build(df, cap, mi)
    tr_o, te_o = RE.split_chrono(d_old)
    # NEW strict
    d_ns = RE.build_features(df, cap, mi, strict_windows=True)
    tr_s, te_s = RE.split_chrono(d_ns)
    # NEW tolerant
    d_nt = RE.build_features(df, cap, mi, strict_windows=False)
    tr_t, te_t = RE.split_chrono(d_nt)
    rows.append({"reservoir": name, "n_test_old": len(te_o),
                 "n_test_new_strict": len(te_s), "n_test_new_tol": len(te_t)})
    tot_old += len(te_o); tot_new_strict += len(te_s); tot_new_tol += len(te_t)
    if len(te_o) != len(te_s):
        n_changed += 1
    if (i + 1) % 25 == 0:
        print(f"  [{i+1}/{len(evaluated)}] old={tot_old} strict={tot_new_strict} tol={tot_new_tol}", flush=True)

A = pd.DataFrame(rows)
A.to_csv(os.path.join(OUT_DIR, "calendar_impact.csv"), index=False)
print("\n===== CALENDAR-SAFE IMPACT (test rows) =====")
print(f"  reservoirs            : {len(A)}")
print(f"  OLD (row-shift) total : {tot_old}")
print(f"  NEW strict   total    : {tot_new_strict}  (drop {(1-tot_new_strict/tot_old)*100:.2f}%)")
print(f"  NEW tolerant total    : {tot_new_tol}  (drop {(1-tot_new_tol/tot_old)*100:.2f}%)")
print(f"  reservoirs whose test N changed (strict): {n_changed}")
chg = A[A["n_test_old"] != A["n_test_new_strict"]]
if len(chg):
    print("  largest drops (strict):")
    for _, r in chg.sort_values("n_test_old").head(8).iterrows():
        print(f"    {r['reservoir']:16s} old={r['n_test_old']:5d} strict={r['n_test_new_strict']:5d} "
              f"tol={r['n_test_new_tol']:5d}")
