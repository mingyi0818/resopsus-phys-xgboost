# -*- coding: utf-8 -*-
"""N=200 SHAP regeneration for Phys-XGBoost, torch-free (SKIP_LSTM=1).

Reuses build_features/split_chrono/run_shap from run_experiment_v3.py but
AVOIDS the torch import (which segfaults on this host) by setting SKIP_LSTM=1.
Computes SHAP over the exact 199 evaluated reservoirs (excludes ResOpsUS_976)
so the attribution matches the main-study set. Writes shap_phys_xgb.csv and
shap_phys_xgb_mean.csv into src/results (consumed by make_figures_N200.py).
"""
import os, glob, sys
os.environ["SKIP_LSTM"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_experiment_v3 as RE

# point at the real raw reservoir CSVs (EXTRACT_DIR default = src/ResOpsUS_data, which is empty)
RE.EXTRACT_DIR = r"D:\workbuddy\reservoir_experiment\ResOpsUS_data"

pm = pd.read_csv(os.path.join(RE.OUT_DIR, "per_reservoir_metrics.csv"))
evaluated = [r for r in pm["reservoir"].tolist() if r != "ResOpsUS_976"]
csvmap = {os.path.splitext(os.path.basename(c))[0]: c
          for c in glob.glob(os.path.join(RE.EXTRACT_DIR, "**", "*.csv"), recursive=True)}
print(f"[shap] {len(evaluated)} evaluated reservoirs; {len(csvmap)} raw CSVs found", flush=True)

frames = []
for i, name in enumerate(evaluated):
    c = csvmap.get(name)
    if not c:
        print("  MISSING", name, flush=True)
        continue
    try:
        df = pd.read_csv(c)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["inflow", "outflow", "storage"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = (df[["date", "inflow", "outflow", "storage"]]
              .dropna(subset=["date"]).sort_values("date").reset_index(drop=True))
        df = df.dropna(subset=["outflow", "inflow", "storage"])
        d = RE.build_features(df, df["storage"].quantile(0.99), df["inflow"].mean())
        tr, te = RE.split_chrono(d)
        sf = RE.run_shap(tr, te, tr["outflow"].values)
        sf["reservoir"] = name
        frames.append(sf)
        if (i + 1) % 20 == 0 or (i + 1) == len(evaluated):
            print(f"  [{i+1}/{len(evaluated)}] done (last={name})", flush=True)
    except Exception as e:
        print(f"  [{name}] FAILED: {repr(e)}", flush=True)

if frames:
    F = pd.concat(frames, ignore_index=True)
    mean_imp = (F.groupby("feature")["mean_abs_shap"].mean()
                .sort_values(ascending=False).reset_index())
    F.to_csv(os.path.join(RE.OUT_DIR, "shap_phys_xgb.csv"), index=False)
    mean_imp.to_csv(os.path.join(RE.OUT_DIR, "shap_phys_xgb_mean.csv"), index=False)
    print(f"\n[shap] mean |SHAP| over {F['reservoir'].nunique()} reservoirs:")
    for _, row in mean_imp.iterrows():
        print(f"    {row['feature']:18s} {row['mean_abs_shap']:.4f}")
    print("saved to", RE.OUT_DIR)
else:
    print("[shap] NO frames produced.")
