# -*- coding: utf-8 -*-
"""Aggregate per-reservoir raw |SHAP| into a scale-robust cross-reservoir
ranking.

The target (outflow) is on a heterogeneous absolute scale across reservoirs,
so an unweighted mean of raw |SHAP| is dominated by a few large-capacity
outliers and does NOT reflect feature importance. Instead, for each reservoir
we convert |SHAP| to a within-reservoir share (|SHAP| / sum of |SHAP| * 100),
then average those shares across all evaluated reservoirs. This yields a
ranking that is invariant to reservoir size.

Reads:  results/shap_phys_xgb.csv  (raw per-reservoir |SHAP|, 199 reservoirs)
Writes: results/shap_phys_xgb_mean.csv  (mean |SHAP| share %, averaged over reservoirs)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
F = pd.read_csv(os.path.join(RES, "shap_phys_xgb.csv"))

shares = []
for _, g in F.groupby("reservoir"):
    tot = g["mean_abs_shap"].sum()
    if tot > 0:
        shares.append(g.set_index("feature")["mean_abs_shap"] / tot * 100.0)
S = pd.DataFrame(shares)
mean_share = S.mean().sort_values(ascending=False).reset_index()
mean_share.columns = ["feature", "mean_abs_shap"]  # value = mean |SHAP| share (%)
mean_share.to_csv(os.path.join(RES, "shap_phys_xgb_mean.csv"), index=False)

print(f"[agg] {F['reservoir'].nunique()} reservoirs; mean |SHAP| share (%) over reservoirs:")
for _, row in mean_share.iterrows():
    print(f"    {row['feature']:18s} {row['mean_abs_shap']:6.2f}%")
