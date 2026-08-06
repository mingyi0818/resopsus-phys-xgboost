# -*- coding: utf-8 -*-
"""P-M1 / P0 #4 proxy-channel stress test (reviewer task-definition scrutiny).

The main task excludes directly-observed release lags and concurrent storage,
but `storage_trend = S_{t-1}-S_{t-2} = I_{t-1}-O_{t-1}` algebraically reintroduces
lagged release (an *indirect* channel). This script quantifies how much of the
Phys-XGBoost gain comes from that channel and from the availability of I_t, by
adding three control configurations and reporting per-reservoir KGE:

  1. raw_plus_outflow_lag1 : RAW features + O_{t-1} (release history made
     available). Upper bound on what release-history knowledge buys.
  2. phys_shifted_trend    : PHYS features but storage_trend (S_{t-1}-S_{t-2})
     replaced by shifted_trend = S_{t-2}-S_{t-3} (one step further back, so it
     no longer carries O_{t-1} directly). Isolates the indirect-channel effect.
  3. forecast_available    : ONLY predictors available at the issue time
     (lagged inflow/storage, calendar, derived storage-state features); the
     directly-observed same-day I_t and all I_t-derived moments are removed.
     This is the genuine forecast variant.

All configs use the SAME calendar-safe build_features (reviewer P-M8) and the
same XGBoost hyperparameters / chronological split as the main study.

Emits src/results/proxy_channel_stress_test.csv (per-reservoir KGE for each
config + the main phys / raw references) and prints median KGE + paired
median delta vs phys.

Run: python run_proxy_stress_test.py
"""
import os, glob, sys, time
os.environ["SKIP_LSTM"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_experiment_v3 as RE

OUT_DIR = RE.OUT_DIR
EXTRACT_DIR = os.environ.get("RESOPSUS_DATA_DIR", RE.EXTRACT_DIR)

pm = pd.read_csv(os.path.join(OUT_DIR, "per_reservoir_metrics.csv"))
evaluated = [r for r in pm["reservoir"].tolist() if r != "ResOpsUS_976"]
csvmap = {os.path.splitext(os.path.basename(c))[0]: c
          for c in glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True)}

RAW = RE.RAW_FEATS
PHYS = RE.PHYS_FEATS


def stress_features(d):
    """Add the control-specific features to a calendar-safe build_features frame."""
    dd = d.copy()
    dd["outflow_lag1"] = dd["outflow"].shift(1)
    dd["storage_lag3"] = dd["storage"].shift(3)
    dd["shifted_trend"] = dd["storage_lag2"] - dd["storage_lag3"]  # S(t-2)-S(t-3)
    return dd


CONFIGS = {
    "raw_plus_outflow_lag1": RAW + ["outflow_lag1"],
    "phys_shifted_trend":     [f for f in (RAW + PHYS) if f != "storage_trend"] + ["shifted_trend"],
    "forecast_available":     ["inflow_lag1", "inflow_lag2", "inflow_lag3",
                                "storage_lag1", "storage_lag2", "doy_sin", "doy_cos",
                                "storage_norm", "storage_deficit", "storage_trend"],
}


def main():
    rows = []
    for i, name in enumerate(evaluated):
        c = csvmap.get(name)
        if not c:
            print("  MISSING", name, flush=True); continue
        df = pd.read_csv(c)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["inflow", "outflow", "storage"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["date", "inflow", "outflow", "storage"]]
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        df = df.dropna(subset=["outflow", "inflow", "storage"])
        cap = df["storage"].quantile(0.99); mi = df["inflow"].mean()
        d = stress_features(RE.build_features(df, cap, mi))
        tr, te = RE.split_chrono(d)
        tr = tr.reset_index(drop=True)
        te = te.reset_index(drop=True)
        if len(tr) < 365*3 or len(te) < 200:
            print(f"  [{i+1}] {name} skipped (too short)"); continue
        y_tr, y_te = tr["outflow"].values, te["outflow"].values
        row = {"reservoir": name, "n_test": len(te),
               "phys_KGE": pm.loc[pm["reservoir"] == name, "phys_KGE"].iloc[0],
               "xgb_KGE": pm.loc[pm["reservoir"] == name, "xgb_KGE"].iloc[0]}
        for cfg, feats in CONFIGS.items():
            sub = te.dropna(subset=feats)
            if len(sub) < 50:
                row[f"{cfg}_KGE"] = np.nan; continue
            m = RE._xgb()
            m.fit(tr.dropna(subset=feats)[feats], y_tr, verbose=False)
            sim = m.predict(sub[feats])
            row[f"{cfg}_KGE"] = RE.kge(y_te[sub.index], sim)[0]
        rows.append(row)
        if (i + 1) % 20 == 0 or (i + 1) == len(evaluated):
            print(f"  [{i+1}/{len(evaluated)}] done (last={name})", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT_DIR, "proxy_channel_stress_test.csv"), index=False)
    print("\n===== PROXY-CHANNEL STRESS TEST (median KGE over evaluated reservoirs) =====")
    for cfg in ["phys", "xgb"] + list(CONFIGS.keys()):
        col = f"{cfg}_KGE"
        if col in R.columns:
            print(f"  {cfg:24s} median KGE = {R[col].median():.3f}")
    print("\n  paired median delta vs phys (Phys-XGBoost):")
    for cfg in list(CONFIGS.keys()):
        col = f"{cfg}_KGE"
        valid = R.dropna(subset=["phys_KGE", col])
        if len(valid) >= 3:
            diff = valid["phys_KGE"].values - valid[col].values
            rng = np.random.default_rng(42)
            boot = [float(np.median(diff[rng.integers(0, len(diff), len(diff))])) for _ in range(2000)]
            ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            print(f"    phys - {cfg:24s} = {np.median(diff):+.3f}  CI95=[{ci[0]:+.3f},{ci[1]:+.3f}]  "
                  f"(n={len(valid)})")
    print(f"\n[stress] saved to {os.path.join(OUT_DIR, 'proxy_channel_stress_test.csv')}")


if __name__ == "__main__":
    main()
