# -*- coding: utf-8 -*-
"""P-M5 / P0 #2 canonical non-improvement table (reviewer M5 reconciliation).

From the frozen per_reservoir_metrics.csv, compute delta = phys_KGE - xgb_KGE
(the raw-XGBoost baseline). The PRIMARY statistic is the win rate (181/199
reservoirs with delta > 0, reported in stats.json). This table DESCRIPTIVELY
reports the delta <= 0 subset (the reservoirs where physics augmentation did
not improve over raw), sorted worst-first, with a build-time assertion that the
count equals 199 - 181 = 18. It NEVER filters the primary statistic.

Categories (descriptive, reconciling the dossier/manuscript M5 accounts):
  both_gt_085 : xgb_KGE > 0.85 AND phys_KGE > 0.85   (near-saturated)
  both_lt_030 : xgb_KGE < 0.30 AND phys_KGE < 0.30   (both catastrophic)
  mixed       : otherwise

Emits src/results/nonimprovement_table.csv and prints the category breakdown
+ the assertion result.

Run: python gen_nonimprovement_table.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT_DIR = os.path.join(HERE, "results")


def main():
    pm = pd.read_csv(os.path.join(OUT_DIR, "per_reservoir_metrics.csv"))
    n_total = len(pm)
    pm = pm.copy()
    pm["delta"] = pm["phys_KGE"] - pm["xgb_KGE"]
    ni = pm[pm["delta"] <= 0].copy().sort_values("delta").reset_index(drop=True)

    def cat(r):
        if r["xgb_KGE"] > 0.85 and r["phys_KGE"] > 0.85:
            return "both_gt_085"
        if r["xgb_KGE"] < 0.30 and r["phys_KGE"] < 0.30:
            return "both_lt_030"
        return "mixed"
    ni["category"] = ni.apply(cat, axis=1)
    ni_out = ni[["reservoir", "xgb_KGE", "phys_KGE", "delta", "category", "n_test"]]
    ni_out.to_csv(os.path.join(OUT_DIR, "nonimprovement_table.csv"), index=False)

    n_wins = int((pm["delta"] > 0).sum())
    expected = n_total - n_wins
    ok = (len(ni) == expected)
    print("===== NON-IMPROVEMENT TABLE (delta = phys_KGE - xgb_KGE <= 0) =====")
    print(f"  total evaluated reservoirs : {n_total}")
    print(f"  phys wins (delta > 0)       : {n_wins}")
    print(f"  delta <= 0 (this table)    : {len(ni)}  (expected {expected})")
    print(f"  ASSERTION count=={expected}: {'PASS' if ok else 'FAIL <<<'}")
    print(f"  categories:")
    for c, g in ni.groupby("category"):
        print(f"    {c:14s} n={len(g):2d}  "
              f"delta range=[{g['delta'].min():.3f},{g['delta'].max():.3f}]  "
              f"worst={g.sort_values('delta').iloc[0]['reservoir']} ({g['delta'].min():.3f})")
    print(f"\n  saved to {os.path.join(OUT_DIR, 'nonimprovement_table.csv')}")
    if not ok:
        raise SystemExit(f"[ASSERT] non-improvement count {len(ni)} != expected {expected}")


if __name__ == "__main__":
    main()
