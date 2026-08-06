# -*- coding: utf-8 -*-
"""P-M8 / P0 temporal-semantics audit: calendar-safe lag check.

Replicates EXACTLY the complete-case filter used by run_experiment_v3.main()
(lines 524-529):
    df = df[["date","inflow","outflow","storage"]]
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["outflow","inflow","storage"])
THEN build_features() applies .shift(1/2/3) on this already-filtered,
date-sorted frame. If two consecutive rows are NOT exactly 1 calendar day
apart, every lag (inflow_lag1..3, storage_lag1/2, storage_trend, and the
7/30-day rolling means) silently spans multiple days and the physical
meaning of "t-1" is violated.

This script measures how many target rows are affected, per reservoir and
for the whole evaluated set. Outputs:
  src/results/lag_gap_audit.csv  (reservoir, n_rows, n_gap_rows, max_gap_days,
                                  frac_gapped)
  prints a summary and a GO / NO-GO decision for the calendar-safe rerun.

Run:
  python lag_gap_audit.py
"""
import os, glob, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_experiment_v3 as RE

OUT_DIR = RE.OUT_DIR
EXTRACT_DIR = r"D:\workbuddy\reservoir_experiment\ResOpsUS_data"

# Evaluate EXACTLY the 199 reservoirs in the canonical per_reservoir_metrics.csv
pm = pd.read_csv(os.path.join(OUT_DIR, "per_reservoir_metrics.csv"))
evaluated = [r for r in pm["reservoir"].tolist() if r != "ResOpsUS_976"]
csvmap = {os.path.splitext(os.path.basename(c))[0]: c
          for c in glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True)}
print(f"[audit] {len(evaluated)} evaluated reservoirs; {len(csvmap)} raw CSVs", flush=True)


def load_complete_case(name):
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


def audit_one(df):
    """Return (n_rows, n_gap_rows, max_gap_days) for the complete-case series."""
    n_rows = len(df)
    if n_rows < 2:
        return n_rows, 0, 0
    d = df["date"].values
    # diff in days between consecutive rows (np.timedelta64 -> float days)
    diff_days = np.diff(d).astype("timedelta64[D]").astype(int)
    # A target row at position i uses S(t-1) = row i-1. It is mis-indexed for
    # lag1 if date[i] - date[i-1] != 1. For lag3 we would also need i-2, i-3;
    # we report the lag1 violation (strictest single-day requirement) and the
    # max gap, which bounds the worst-case multi-day lag.
    n_gap_rows = int((diff_days > 1).sum())
    max_gap = int(diff_days.max()) if len(diff_days) else 0
    return n_rows, n_gap_rows, max_gap


rows = []
total_rows = 0
total_gap = 0
reservoirs_with_gaps = []
for i, name in enumerate(evaluated):
    df = load_complete_case(name)
    if df is None:
        print("  MISSING", name, flush=True)
        continue
    n_rows, n_gap, max_gap = audit_one(df)
    frac = n_gap / n_rows if n_rows else 0.0
    rows.append({"reservoir": name, "n_rows": n_rows,
                 "n_gap_rows": n_gap, "max_gap_days": max_gap, "frac_gapped": round(frac, 6)})
    total_rows += n_rows
    total_gap += n_gap
    if n_gap > 0:
        reservoirs_with_gaps.append(name)
    if (i + 1) % 25 == 0 or (i + 1) == len(evaluated):
        print(f"  [{i+1}/{len(evaluated)}] scanned; {len(reservoirs_with_gaps)} res w/ gaps so far", flush=True)

A = pd.DataFrame(rows)
A.to_csv(os.path.join(OUT_DIR, "lag_gap_audit.csv"), index=False)

print("\n===== LAG-GAP AUDIT SUMMARY =====")
print(f"  reservoirs audited      : {len(A)}")
print(f"  total complete-case rows: {total_rows}")
print(f"  total mis-indexed rows  : {total_gap} ({100*total_gap/total_rows:.3f}% of all rows)")
print(f"  reservoirs with >=1 gap : {len(reservoirs_with_gaps)}")
if len(A):
    worst = A.sort_values("max_gap_days", ascending=False).head(5)
    print("  worst max-gap reservoirs:")
    for _, r in worst.iterrows():
        print(f"    {r['reservoir']:16s} n_rows={r['n_rows']:5d} n_gap={r['n_gap_rows']:4d} "
              f"max_gap={r['max_gap_days']}d frac={r['frac_gapped']:.4f}")
if total_gap == 0:
    print("\n  DECISION: NO-GO rerun. Zero mis-indexed rows -> row-shift lags are "
          "already calendar-exact for every evaluated reservoir. Calendar-safe "
          "reindex would reproduce identical results.")
else:
    print("\n  DECISION: GO rerun. Calendar-safe lag construction required; "
          "rerun main experiment + SHAP + rollout on date-safe features.")
print(f"\n[audit] saved to {os.path.join(OUT_DIR, 'lag_gap_audit.csv')}")
