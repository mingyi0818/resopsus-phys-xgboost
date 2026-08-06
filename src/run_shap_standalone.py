# -*- coding: utf-8 -*-
"""Standalone SHAP runner for Reservoir v3 (Phys-XGBoost feature attribution).

The full experiment (run_experiment_v3.py) ran with an OLD in-memory copy of
run_shap that passed an empty eval_set to XGBoost 3.x and crashed. This script
imports the *fixed* run_shap (no eval_set, CPU-forced) and recomputes SHAP for
Phys-XGBoost across ALL 30 reservoirs, averaging the per-feature |SHAP| so the
attribution is robust rather than relying on a single reservoir.

Run:
  python run_shap_standalone.py
"""
import os, sys, glob, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_experiment_v3 as RE   # imports fixed run_shap, build_features, etc.

OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)


def select_reservoirs():
    csvs = sorted(glob.glob(os.path.join(RE.EXTRACT_DIR, "**", "*.csv"), recursive=True))
    selected = []
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
        selected.append((c, len(comp)))
    selected.sort(key=lambda x: -x[1])
    N = int(os.environ.get("N_RES", "30"))
    if N < len(selected):
        idx = np.linspace(0, len(selected) - 1, N).round().astype(int)
        selected = [selected[i] for i in idx]
    return selected


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    selected = select_reservoirs()
    print(f"[shap] {len(selected)} reservoirs selected")

    frames = []
    for i, (c, ncomp) in enumerate(selected):
        try:
            df = pd.read_csv(c)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for col in ["inflow", "outflow", "storage"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["date", "inflow", "outflow", "storage"]]
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            df = df.dropna(subset=["outflow", "inflow", "storage"])
            name = os.path.splitext(os.path.basename(c))[0]
            d = RE.build_features(df, df["storage"].quantile(0.99), df["inflow"].mean())
            tr, te = RE.split_chrono(d)
            sf = RE.run_shap(tr, te, tr["outflow"].values)
            sf["reservoir"] = name
            frames.append(sf)
            print(f"  [{i+1}/{len(selected)}] {name} SHAP ok (n_te={len(te)})", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(selected)}] {name} SHAP FAILED: {e}", flush=True)

    if not frames:
        print("[shap] NO SHAP frames produced."); return

    F = pd.concat(frames, ignore_index=True)
    mean_imp = (F.groupby("feature")["mean_abs_shap"].mean()
                  .sort_values(ascending=False).reset_index())
    mean_imp.to_csv(os.path.join(OUT_DIR, "shap_phys_xgb_mean.csv"), index=False)
    F.to_csv(os.path.join(OUT_DIR, "shap_phys_xgb.csv"), index=False)
    print("\n[shap] mean |SHAP| over 30 reservoirs (Phys-XGBoost):")
    for _, row in mean_imp.iterrows():
        print(f"    {row['feature']:22s} {row['mean_abs_shap']:.4f}")
    print(f"\n[shap] saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
