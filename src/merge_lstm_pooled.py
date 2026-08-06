# -*- coding: utf-8 -*-
"""P0 post-merge: fold the (calendar-safe, numpy) LSTM results back into the
main artifacts that rerun_full wrote WITHOUT LSTM (SKIP_LSTM=1).

- per_reservoir_metrics.csv : populate lstm_* / lstmphys_* columns per reservoir
- pooled_metrics.json        : compute volume-weighted pooled KGE/r/NSE/RMSE for
                               'lstm' and 'lstmphys' from lstm_preds_long.csv
- summary_v3.json (if present) : refresh lstm medians

Pure computation, no model training. Deterministic given the CSVs.
"""
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("SKIP_LSTM", "1")
import run_experiment_v3 as RE

OUT = RE.OUT_DIR


def main():
    pm = pd.read_csv(os.path.join(OUT, "per_reservoir_metrics.csv"))
    lo = pd.read_csv(os.path.join(OUT, "lstm_only.csv"))
    lp = pd.read_csv(os.path.join(OUT, "lstm_preds_long.csv"))

    # ---- 1. per-reservoir LSTM columns ------------------------------------
    lcols = ["reservoir", "lstm_KGE", "lstmphys_KGE", "lstm_r", "lstmphys_r",
             "lstm_alpha", "lstmphys_alpha", "lstm_beta", "lstmphys_beta",
             "lstm_NSE", "lstmphys_NSE", "lstm_logNSE", "lstmphys_logNSE",
             "lstm_RMSE", "lstmphys_RMSE"]
    lo_s = lo[lcols].copy()
    pm = pm.merge(lo_s, on="reservoir", how="left", suffixes=("", "_lstm"))
    # drop any stale duplicate *_lstm columns if merge produced them
    for c in [c for c in pm.columns if c.endswith("_lstm")]:
        base = c[:-5]
        if base in pm.columns:
            pm[base] = pm[c]
        pm = pm.drop(columns=[c])
    pm.to_csv(os.path.join(OUT, "per_reservoir_metrics.csv"), index=False)
    print(f"[merge] per_reservoir_metrics.csv lstm columns filled: "
          f"{int(pm['lstm_KGE'].notna().sum())}/{len(pm)} reservoirs")

    # ---- 2. pooled LSTM metrics -------------------------------------------
    pp = json.load(open(os.path.join(OUT, "pooled_metrics.json")))
    for mdl in ["lstm", "lstmphys"]:
        sub = lp[["obs", mdl]].dropna(subset=[mdl])
        obs = sub["obs"].values.astype(float)
        sim = sub[mdl].values.astype(float)
        metrics = RE.all_metrics(obs, sim)
        pp[mdl] = {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                   for k, v in metrics.items()}
        print(f"[merge] pooled {mdl}: KGE={metrics['KGE']:.4f} "
              f"N={len(obs)} r={metrics['r']:.4f} NSE={metrics['NSE']:.4f}")
    json.dump(pp, open(os.path.join(OUT, "pooled_metrics.json"), "w"), indent=2)

    # ---- 3. summary_v3.json refresh (if it has lstm keys) -----------------
    sp = os.path.join(OUT, "summary_v3.json")
    if os.path.exists(sp):
        summ = json.load(open(sp))
        changed = False
        for key in ["lstm_KGE_median", "lstmphys_KGE_median"]:
            if key in summ:
                base = key.replace("_median", "")
                summ[key] = float(pm[base].median())
                changed = True
        if changed:
            json.dump(summ, open(sp, "w"), indent=2)
            print(f"[merge] summary_v3.json lstm medians refreshed")

    print("[merge] done.")


if __name__ == "__main__":
    main()
