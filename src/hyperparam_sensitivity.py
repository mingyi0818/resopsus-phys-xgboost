# -*- coding: utf-8 -*-
"""
hyperparam_sensitivity.py -- reviewer P4 (robustness to hyperparameter choice).

Runs Phys-XGBoost (the headline model of this study) on a deterministic
30-reservoir subset spanning the difficulty spectrum, sweeping
    max_depth   in {4, 6, 8}
    n_estimators in {200, 400, 800}
(9 configurations), keeping every other hyperparameter identical to the main
run (_xgb defaults: learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
n_jobs=4, random_state=RNG, objective=reg:squarederror).

Reports the per-configuration MEDIAN test KGE across the 30 reservoirs, plus
the overall min/max spread, so the manuscript can state that the headline
conclusion (physics augmentation >> hyperparameter choice) is insensitive to
reasonable hyperparameter variation.

Traceability:
  * uses the SAME fixed build_features (training-only climatology) as the main
    rerun, so numbers are consistent with the leakage-fixed cohort;
  * reservoir selection = evenly-spaced 30 from the identical 200-cohort used
    by rerun_full.select(200);
  * all outputs written to results/hyperparam_sensitivity.csv + printed summary.

Run:  python hyperparam_sensitivity.py
"""
import os, glob, json, time
os.environ["SKIP_LSTM"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
import run_experiment_v3 as RE

OUT = RE.OUT_DIR
EXTRACT = RE.EXTRACT_DIR
RAW_FEATS = RE.RAW_FEATS
PHYS_FEATS = RE.PHYS_FEATS
RNG = RE.RNG
ALL = RAW_FEATS + PHYS_FEATS

DEPTHS = [4, 6, 8]
NESTS = [200, 400, 800]
N_SUBSET = 30


def select_subset():
    """Evenly-spaced 30 from the identical 200-cohort (rerun_full.select)."""
    csvs = sorted(glob.glob(os.path.join(EXTRACT, "**", "*.csv"), recursive=True))
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
    idx = np.linspace(0, len(sel) - 1, N_SUBSET).round().astype(int)
    names = [sel[i][0] for i in idx]
    # exclude the all-NaN reservoir (ResOpsUS_976) just like the main study
    return [n for n in names if n != "ResOpsUS_976"]


def load_res(name):
    csvmap = {os.path.splitext(os.path.basename(c))[0]: c
              for c in glob.glob(os.path.join(EXTRACT, "**", "*.csv"), recursive=True)}
    c = csvmap.get(name)
    df = pd.read_csv(c)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["inflow", "outflow", "storage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (df[["date", "inflow", "outflow", "storage"]]
          .dropna(subset=["date"]).sort_values("date").reset_index(drop=True))
    df = df.dropna(subset=["outflow", "inflow", "storage"])
    return df


def phys_xgb(depth, nest):
    return xgb.XGBRegressor(n_estimators=nest, max_depth=depth, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                            random_state=RNG, objective="reg:squarederror")


def main():
    names = select_subset()
    csvmap = {os.path.splitext(os.path.basename(c))[0]: c
              for c in glob.glob(os.path.join(EXTRACT, "**", "*.csv"), recursive=True)}
    print(f"[hyperparam] subset of {len(names)} reservoirs")
    rows = []
    configs = [(d, n) for d in DEPTHS for n in NESTS]
    per_cfg = {cfg: [] for cfg in configs}
    for i, name in enumerate(names):
        c = csvmap.get(name)
        if not c:
            print("  MISSING", name); continue
        df = load_res(name)
        capacity = df["storage"].quantile(0.99)
        mean_inflow = df["inflow"].mean()
        d = RE.build_features(df, capacity, mean_inflow)
        train, test = RE.split_chrono(d)
        if len(train) < 365 * 3 or len(test) < 200:
            print(f"  [{i+1}] {name} too small, skipped"); continue
        y_te = test["outflow"].values
        for (depth, nest) in configs:
            m = phys_xgb(depth, nest)
            y_tr = train["outflow"].values
            m.fit(train[ALL], y_tr, eval_set=[(test[ALL], y_te)], verbose=False)
            sim = m.predict(test[ALL])
            kge = RE.all_metrics(y_te, sim)["KGE"]
            per_cfg[(depth, nest)].append(kge)
            rows.append({"reservoir": name, "max_depth": depth,
                         "n_estimators": nest, "KGE": kge})
        print(f"  [{i+1}/{len(names)}] {name} done", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "hyperparam_sensitivity.csv"), index=False)

    print("\n[hyperparam] median test KGE by configuration (30-reservoir subset)")
    print(f"{'max_depth':>9} {'n_estimators':>12} {'median_KGE':>11} {'min':>7} {'max':>7}")
    medians = []
    for (depth, nest) in configs:
        arr = np.array(per_cfg[(depth, nest)])
        med = float(np.median(arr))
        medians.append(med)
        print(f"{depth:>9} {nest:>12} {med:>11.4f} {arr.min():>7.3f} {arr.max():>7.3f}")
    overall_med = float(np.median(medians))
    spread = max(medians) - min(medians)
    print(f"\n[summary] default (depth=6, nest=400) median KGE = "
          f"{per_cfg[(6,400)] and float(np.median(per_cfg[(6,400)])):.4f}")
    print(f"[summary] across all 9 configs: median-of-medians = {overall_med:.4f}, "
          f"spread (max-min) = {spread:.4f}")
    json.dump({"configs": {f"d{de}_n{ne}": float(np.median(per_cfg[(de, ne)]))
                           for (de, ne) in configs},
               "default_median": float(np.median(per_cfg[(6, 400)])),
               "spread": float(spread)},
              open(os.path.join(OUT, "hyperparam_sensitivity_summary.json"), "w"),
              indent=2)
    print("[hyperparam] COMPLETE")


if __name__ == "__main__":
    main()
