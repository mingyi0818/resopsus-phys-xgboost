"""
merge_results.py -- combine the trees-only run (run_experiment_v3.py with SKIP_LSTM=1)
and the torch-only LSTM run (run_lstm_only.py) into a single, consistent result set
under RESULTS_DIR.

The trees run already produced per_reservoir_metrics.csv (10 models + tier),
pooled_metrics.json, stats.json and summary_v3.json (LSTM columns NaN). This script:
  * joins the LSTM per-reservoir metrics (lstm_only.csv) by reservoir,
  * recomputes LSTM pooled KGE/NSE/RMSE from lstm_preds_long.csv,
  * adds the Phys-XGBoost vs LSTM Wilcoxon / bootstrap-CI / win-rate to stats.json,
  * adds LSTM rows to summary_v3.json,
and overwrites the per_reservoir / pooled / stats / summary files so the final
results_v4 is self-consistent across all 12 models.
"""

import os, json
import numpy as np
import pandas as pd
from scipy import stats as sps

from shared_core import all_metrics  # torch-free single source of truth

OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RNG = 42


def main():
    pr = pd.read_csv(os.path.join(OUT_DIR, "per_reservoir_metrics.csv"))
    lstm = pd.read_csv(os.path.join(OUT_DIR, "lstm_only.csv"))
    long = pd.read_csv(os.path.join(OUT_DIR, "lstm_preds_long.csv"))

    # 1) join LSTM per-reservoir metrics
    lstm_cols = [c for c in lstm.columns if c != "reservoir"]
    pr = pr.merge(lstm[["reservoir"] + lstm_cols], on="reservoir", how="left")
    pr.to_csv(os.path.join(OUT_DIR, "per_reservoir_metrics.csv"), index=False)
    print(f"[merge] per_reservoir: {pr.shape[0]} reservoirs, {pr.shape[1]} columns")

    # 2) pooled metrics: keep tree models from existing pooled_metrics.json, add LSTM
    with open(os.path.join(OUT_DIR, "pooled_metrics.json")) as f:
        pooled = json.load(f)
    for mdl in ["lstm", "lstmphys"]:
        sub = long.dropna(subset=[mdl])
        pooled[mdl] = all_metrics(sub[["obs"]].values.ravel(), sub[[mdl]].values.ravel())
    with open(os.path.join(OUT_DIR, "pooled_metrics.json"), "w") as f:
        json.dump(pooled, f, indent=2)
    print(f"[merge] pooled: added lstm (KGE={pooled['lstm']['KGE']:.3f}) "
          f"lstmphys (KGE={pooled['lstmphys']['KGE']:.3f})")

    # 3) stats: add Phys-XGBoost vs LSTM
    with open(os.path.join(OUT_DIR, "stats.json")) as f:
        stat = json.load(f)
    for other in ["lstm", "lstmphys"]:
        col = f"{other}_KGE"
        valid = pr.dropna(subset=["phys_KGE", col])
        if len(valid) < 3:
            continue
        diff = valid["phys_KGE"].values - valid[col].values
        try:
            w = sps.wilcoxon(valid["phys_KGE"], valid[col])
            stat[f"wilcoxon_{other}_p"] = float(w.pvalue)
        except Exception as e:
            stat[f"wilcoxon_{other}_error"] = str(e)
        stat[f"phys_wins_vs_{other}_n"] = int((diff > 0).sum())
        stat[f"phys_wins_vs_{other}_frac"] = float((diff > 0).mean())
        stat[f"median_dKGE_vs_{other}"] = float(np.median(diff))
        rng = np.random.default_rng(RNG)
        boot = [float(np.median(diff[rng.integers(0, len(diff), len(diff))])) for _ in range(2000)]
        stat[f"median_dKGE_vs_{other}_CI95"] = [float(np.percentile(boot, 2.5)),
                                                float(np.percentile(boot, 97.5))]
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stat, f, indent=2)
    print(f"[merge] stats: added phys vs lstm / lstmphys (wilcoxon p_lstm={stat.get('wilcoxon_lstm_p')})")

    # 4) summary: add LSTM rows
    with open(os.path.join(OUT_DIR, "summary_v3.json")) as f:
        summ = json.load(f)
    summ["n_reservoirs"] = int(len(pr))
    for mdl in ["lstm", "lstmphys"]:
        if f"{mdl}_KGE" not in pr.columns:
            continue
        summ[f"{mdl}_KGE_median"] = float(pr[f"{mdl}_KGE"].median())
        summ[f"{mdl}_NSE_median"] = float(pr[f"{mdl}_NSE"].median())
        summ[f"{mdl}_KGE_pooled"] = pooled[mdl]["KGE"]
    with open(os.path.join(OUT_DIR, "summary_v3.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(f"[merge] summary: n_reservoirs={summ['n_reservoirs']}, "
          f"phys_KGE_median={summ['phys_KGE_median']:.3f}, "
          f"lstmphys_KGE_median={summ.get('lstmphys_KGE_median', float('nan')):.3f}")
    print("[merge] done.")


if __name__ == "__main__":
    main()
