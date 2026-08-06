"""Analyze the N=200 iterative-simulation (rollout) results and emit numbers
for manuscript Section 6.10.

Reads results/per_reservoir_rollout_0_200.csv (produced by run_rollout.py,
6 models + persistence, leads 1..7) and writes results/rollout_analysis.json
plus a printable median-KGE-by-lead table.

Median is taken PER RESERVOIR across the evaluated set (reservoirs with
non-finite KGE at a given lead are excluded from that lead's median), matching
the per-reservoir-median convention used everywhere else in the paper.
"""
import json
import numpy as np
import pandas as pd

CSV = "results/per_reservoir_rollout_0_200.csv"
MODELS = ["persist", "xgb", "lgbm", "cb", "phys", "lgbmphys", "cbphys"]
LABELS = {
    "persist": "Persistence-L", "xgb": "XGBoost (raw)", "lgbm": "LightGBM (raw)",
    "cb": "CatBoost (raw)", "phys": "Phys-XGBoost", "lgbmphys": "LightGBM-phys",
    "cbphys": "CatBoost-phys",
}
LEADS = [1, 2, 3, 4, 5, 6, 7]

d = pd.read_csv(CSV)
n_total = len(d)

# median KGE per model x lead (exclude non-finite)
med = {}
n_finite = {}
for m in MODELS:
    med[m] = {}
    n_finite[m] = {}
    for L in LEADS:
        col = f"{m}_KGE_L{L}"
        vals = pd.to_numeric(d[col], errors="coerce").dropna().values
        med[m][L] = float(np.median(vals)) if len(vals) else float("nan")
        n_finite[m][L] = int(len(vals))

# fraction of reservoirs where phys > raw xgb, per lead
frac_phys_gt_xgb = {}
gain_phys_minus_xgb = {}
for L in LEADS:
    pv = pd.to_numeric(d[f"phys_KGE_L{L}"], errors="coerce")
    xv = pd.to_numeric(d[f"xgb_KGE_L{L}"], errors="coerce")
    both = pd.concat([pv, xv], axis=1).dropna()
    both.columns = ["p", "x"]
    frac_phys_gt_xgb[L] = float((both["p"] > both["x"]).mean()) if len(both) else float("nan")
    gain_phys_minus_xgb[L] = float(np.median(both["p"] - both["x"])) if len(both) else float("nan")

# retention of one-step skill at lead 7 (one-step phys median KGE = 0.877)
ONE_STEP_PHYS = 0.877
retention_l7 = med["phys"][7] / ONE_STEP_PHYS if med["phys"][7] == med["phys"][7] else float("nan")

out = {
    "n_total": n_total,
    "n_finite_phys_lead1": n_finite["phys"][1],
    "median_by_model_lead": med,
    "n_finite_by_model_lead": n_finite,
    "frac_phys_gt_xgb_by_lead": frac_phys_gt_xgb,
    "median_gain_phys_minus_xgb_by_lead": gain_phys_minus_xgb,
    "phys_lead1": med["phys"][1], "phys_lead7": med["phys"][7],
    "xgb_lead1": med["xgb"][1], "xgb_lead7": med["xgb"][7],
    "persist_lead1": med["persist"][1], "persist_lead7": med["persist"][7],
    "retention_l7_vs_onestep": retention_l7,
}
with open("results/rollout_analysis.json", "w") as f:
    json.dump(out, f, indent=2)

# ---- printable table ----
hdr = "Lead | " + " | ".join(LABELS[m] for m in MODELS) + " | frac(phys>xgb)"
print(hdr)
for L in LEADS:
    row = f"{L}"
    for m in MODELS:
        row += f" | {med[m][L]:.3f}"
    row += f" | {frac_phys_gt_xgb[L]:.2f}"
    print(row)
print()
print(f"n_total={n_total}  n_finite_phys_L1={n_finite['phys'][1]}")
print(f"phys L1={med['phys'][1]:.3f}  L7={med['phys'][7]:.3f}  "
      f"retention(L7/one-step 0.877)={retention_l7:.2f}")
print(f"xgb(raw) L1={med['xgb'][1]:.3f}  L7={med['xgb'][7]:.3f}")
print(f"persist L1={med['persist'][1]:.3f}  L7={med['persist'][7]:.3f}")
print(f"median gain phys-xgb by lead: " +
      ", ".join(f"L{L}:{gain_phys_minus_xgb[L]:.3f}" for L in LEADS))
