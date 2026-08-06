# -*- coding: utf-8 -*-
"""P0 #1 frozen experiment manifest -- the single source of truth.

Freezes, for the calendar-safe (reviewer P-M8) experiment:
  * data source + cohort (deterministic selection, 199 evaluated reservoirs)
  * model scope (199 independent per-reservoir local fits)
  * exact feature columns (RAW 8 + PHYS 10 = 18)
  * seeds + fixed hyperparameters (no test-set tuning)
  * temporal semantics (calendar reindex, strict windows, 2016-01-01 split)
  * SHA-256 hashes of every canonical output artifact

Emits src/results/experiment_manifest.json. Run AFTER all reruns (rerun_full,
run_lstm_numpy, run_rollout, stress test, non-improvement table) have finished.

Run: python gen_manifest.py
"""
import os, sys, json, hashlib, datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_experiment_v3 as RE

OUT_DIR = os.path.join(HERE, "results")
DATA_DIR = r"D:\workbuddy\reservoir_experiment\ResOpsUS_data"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    pm = pd.read_csv(os.path.join(OUT_DIR, "per_reservoir_metrics.csv"))
    evaluated = sorted([r for r in pm["reservoir"].tolist() if r != "ResOpsUS_976"])
    raw_csvs = sorted(glob_csv(DATA_DIR))

    manifest = {
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "study": "WB03 ResOpsUS Phys-XGBoost (calendar-safe revision)",
        "data_source": {
            "path": DATA_DIR,
            "n_raw_csv_files": len(raw_csvs),
            "format": "per-reservoir daily CSV: date,storage,inflow,outflow,evaporation,elevation",
            "selection": {
                "rule": "complete-case rows >=5000; test(>=2016-01-01)>=500; "
                        "train(<2016-01-01)>=3000; stratified linspace N=200; "
                        "ResOpsUS_976 excluded (all-NaN target)",
                "n_selected": 200,
                "n_evaluated": len(evaluated),
                "excluded": ["ResOpsUS_976"],
                "deterministic": True,
            },
            "evaluated_reservoirs": evaluated,
        },
        "model_scope": {
            "type": "199 independent per-reservoir LOCAL fits",
            "note": "one model per reservoir per family; reservoir_id is constant "
                    "within each fit, so no reservoir_id feature is used and no "
                    "global model is fitted (rules out the fabricated "
                    "reservoir_id_encoded SHAP attribution)",
            "families": RE.MODELS,
        },
        "features": {
            "raw_8": RE.RAW_FEATS,
            "phys_10": RE.PHYS_FEATS,
            "phys_classes": RE.PHYS_CLASSES,
            "total": len(RE.RAW_FEATS) + len(RE.PHYS_FEATS),
        },
        "seeds_and_hparams": {
            "RNG": 42,
            "numpy_seed": 42,
            "xgboost": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
                        "subsample": 0.8, "colsample_bytree": 0.8, "objective": "reg:squarederror"},
            "lightgbm": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
                         "subsample": 0.8, "colsample_bytree": 0.8},
            "catboost": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
                         "subsample": 0.8, "colsample_bylevel": 0.8},
            "lstm_numpy": {"hidden": 32, "epochs": 120, "lr": 0.005, "batch": 256,
                           "window": 30, "max_samples": 3000},
            "test_set_tuning": False,
        },
        "temporal_semantics": {
            "lag_construction": "calendar-safe: each reservoir reindexed to a "
                                "complete daily (freq='D') calendar BEFORE any "
                                "lag/window; .shift(k) is exactly k calendar days",
            "gap_handling": "rows whose required timestamps (t-1,t-2,t-3 and full "
                            "7/30-day windows under strict_windows=True) are missing "
                            "become NaN and are dropped by split_chrono",
            "strict_windows": True,
            "chronological_split": "train < 2016-01-01, test >= 2016-01-01, no shuffle",
            "audit": "lag_gap_audit.csv (pre-fix: 145/199 reservoirs, 16286 mis-indexed rows)",
        },
        "output_artifacts": {},
    }

    # hash every canonical output that exists
    targets = [
        "per_reservoir_metrics.csv", "ablation.csv", "shap_phys_xgb.csv",
        "shap_phys_xgb_mean.csv", "stats.json", "summary_v3.json",
        "pooled_metrics.json", "proxy_channel_stress_test.csv",
        "nonimprovement_table.csv", "per_reservoir_rollout_0_200.csv",
        "lstm_only.csv", "rollout_lead1_alignment.json",
    ]
    for t in targets:
        p = os.path.join(OUT_DIR, t)
        if os.path.exists(p):
            manifest["output_artifacts"][t] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
        else:
            manifest["output_artifacts"][t] = {"present": False}

    out = os.path.join(OUT_DIR, "experiment_manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    n_present = sum(1 for v in manifest["output_artifacts"].values() if "sha256" in v)
    print(f"[manifest] wrote {out}")
    print(f"  evaluated reservoirs : {len(evaluated)}")
    print(f"  features             : {manifest['features']['total']} "
          f"(RAW {len(RE.RAW_FEATS)} + PHYS {len(RE.PHYS_FEATS)})")
    print(f"  output artifacts hashed: {n_present}/{len(targets)}")


def glob_csv(d):
    import glob
    return glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)


if __name__ == "__main__":
    main()
