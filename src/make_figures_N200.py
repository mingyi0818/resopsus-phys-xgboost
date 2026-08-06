# -*- coding: utf-8 -*-
"""Regenerate ALL 7 manuscript figures from the N=200 results in src/results.

Reads:
  per_reservoir_metrics.csv  (N=200, 199 evaluated; 10 tree/ref models + tier)
  lstm_only.csv              (N=200 LSTM baseline; merged in by reservoir)
  ablation.csv               (per-reservoir ablation medians)
  all_numbers_N200.json      (KGE component decomposition)
  shap_phys_xgb_mean.csv     (mean |SHAP| over reservoirs; Fig 7, N=200 if regenerated)
Writes PNG (+PDF) into ../manuscript/figures/ with the exact names the tex
references: fig1_kge_boxplot, fig2_nse_boxplot, fig3_scatter_improvement,
fig4_ablation, fig5_tiers, fig6_decomposition, fig7_shap.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "manuscript", "figures")
os.makedirs(FIG, exist_ok=True)

MODELS = ["passthrough", "persist", "ridge", "rf", "xgb", "phys",
          "lgbm", "lgbmphys", "cb", "cbphys", "lstm", "lstmphys"]
DISPLAY = {
    "passthrough": "Pass-through (I=O)", "persist": "Persistence (O$_{t-1}$)",
    "ridge": "Ridge", "rf": "Random Forest",
    "xgb": "Raw-XGBoost", "phys": "Phys-XGBoost",
    "lgbm": "Raw-LightGBM", "lgbmphys": "Phys-LightGBM",
    "cb": "Raw-CatBoost", "cbphys": "Phys-CatBoost",
    "lstm": "LSTM (raw)", "lstmphys": "LSTM (phys)",
}

def color_for(m):
    if m in ("passthrough", "persist"):
        return "#9aa0a6"
    if m in ("lstm", "lstmphys"):
        return "#d9534f"
    if m == "phys":
        return "#1a7f37"
    if m in ("lgbmphys", "cbphys"):
        return "#3a8ed0"
    return "#6c8ebf"

plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

EXCL = "ResOpsUS_976"
R = pd.read_csv(os.path.join(RES, "per_reservoir_metrics.csv"))
R = R.set_index("reservoir")
# merge N=200 LSTM baseline
L = pd.read_csv(os.path.join(RES, "lstm_only.csv"))
L = L[~L["reservoir"].str.contains(EXCL, na=False)]
for _, row in L.iterrows():
    res = row["reservoir"]
    if res in R.index:
        R.loc[res, "lstm_KGE"] = row["lstm_KGE"]
        R.loc[res, "lstmphys_KGE"] = row["lstmphys_KGE"]
        R.loc[res, "lstm_NSE"] = row["lstm_NSE"]
        R.loc[res, "lstmphys_NSE"] = row["lstmphys_NSE"]
R = R.reset_index()
# evaluation set = reservoirs with a valid phys_KGE (199)
Rv = R[R["phys_KGE"].notna()].copy()
print(f"[fig] base rows={len(R)} evaluated={len(Rv)} (excluded {EXCL})")

def col_values(model, metric):
    c = f"{model}_{metric}"
    if c not in Rv.columns:
        return np.array([])
    return Rv[c].dropna().values

# ---- Fig 1: boxplot of per-reservoir KGE across 12 models ----
fig, ax = plt.subplots(figsize=(9, 5.5))
data, labels, cols = [], [], []
for m in MODELS:
    s = col_values(m, "KGE")
    if len(s) > 0:
        data.append(s); labels.append(DISPLAY[m]); cols.append(color_for(m))
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=1.2))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.axhline(0.5, color="black", ls="--", lw=1, label="KGE = 0.5 (skilful)")
ax.set_ylabel("Per-reservoir KGE")
ax.set_title("Distribution of per-reservoir KGE across 12 models (N=200; box = IQR, line = median)")
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_kge_boxplot.png"))
plt.savefig(os.path.join(FIG, "fig1_kge_boxplot.pdf"))
plt.close(fig)
print("fig1_kge_boxplot written")

# ---- Fig 2: boxplot of per-reservoir NSE across 12 models ----
fig, ax = plt.subplots(figsize=(9, 5.5))
data, labels, cols = [], [], []
for m in MODELS:
    s = col_values(m, "NSE")
    if len(s) > 0:
        data.append(s); labels.append(DISPLAY[m]); cols.append(color_for(m))
if data:
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.axhline(0.5, color="black", ls="--", lw=1, label="NSE = 0.5")
    ax.set_ylabel("Per-reservoir NSE")
    ax.set_title("Distribution of per-reservoir NSE across 12 models (N=200; box = IQR, line = median)")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2_nse_boxplot.png"))
    plt.savefig(os.path.join(FIG, "fig2_nse_boxplot.pdf"))
    plt.close(fig)
    print("fig2_nse_boxplot written")

# ---- Fig 3: scatter phys vs xgb per-reservoir KGE (improvement) ----
fig, ax = plt.subplots(figsize=(6.5, 6.0))
xs = Rv["xgb_KGE"].values; ys = Rv["phys_KGE"].values
mask = ~np.isnan(xs) & ~np.isnan(ys)
xs, ys = xs[mask], ys[mask]
ax.scatter(xs, ys, s=18, c="#1a7f37", alpha=0.6, edgecolor="none")
lim = [min(xs.min(), ys.min()) - 0.05, max(xs.max(), ys.max()) + 0.05]
ax.plot(lim, lim, "k--", lw=1, label="y = x")
ax.axhline(0.5, color="grey", ls=":", lw=1); ax.axvline(0.5, color="grey", ls=":", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Raw XGBoost KGE (per reservoir)")
ax.set_ylabel("Phys-XGBoost KGE (per reservoir)")
ax.set_title("Phys-XGBoost vs Raw XGBoost (N=200)")
# count improvements
imp = int(np.sum(ys > xs))
ax.text(0.03, 0.97, f"Phys-XGBoost better on {imp}/{len(xs)} reservoirs",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="grey"))
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig3_scatter_improvement.png"))
plt.savefig(os.path.join(FIG, "fig3_scatter_improvement.pdf"))
plt.close(fig)
print("fig3_scatter_improvement written")

# ---- Fig 4: ablation (median KGE across reservoirs) ----
ab = pd.read_csv(os.path.join(RES, "ablation.csv"))
ab_mean = ab[["full", "raw_only", "no_mass_balance", "no_storage_state", "no_flow_anomaly"]].median()
fig, ax = plt.subplots(figsize=(8, 4.5))
nice = {"full": "Full (all phys)", "raw_only": "Raw only (no phys)",
        "no_mass_balance": "− mass-balance", "no_storage_state": "− storage state",
        "no_flow_anomaly": "− flow anomaly"}
keys = list(ab_mean.index)
disp = [nice.get(k, k) for k in keys]
vals = ab_mean.values
bars = ax.barh(disp, vals, color="#1a7f37", edgecolor="black", linewidth=0.5)
for b, v in zip(bars, vals):
    ax.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=9)
ax.axvline(0.5, color="black", ls="--", lw=1)
ax.set_xlabel("Median KGE (over 199 evaluated reservoirs)")
ax.set_title("Ablation study — contribution of each physics-derived feature group (N=200)")
ax.set_xlim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig4_ablation.png"))
plt.savefig(os.path.join(FIG, "fig4_ablation.pdf"))
plt.close(fig)
print("fig4_ablation written")

# ---- Fig 5: tier-stratified median KGE ----
# Tiers are TRAINING-period persistence-KGE terciles (a priori stratification,
# independent of the test set) -- matches the figure caption and Section 6.4.
# per_reservoir_metrics.csv only carries the test-period `tier`; load the
# training-period tier from the companion file.
_Wt = pd.read_csv(os.path.join(RES, "per_reservoir_metrics_with_train_tier.csv"))
_tier_map = _Wt.set_index("reservoir")["train_tier"]
R["train_tier"] = R["reservoir"].map(_tier_map)
Rv["train_tier"] = Rv["reservoir"].map(_tier_map)
key_models = ["xgb", "phys", "lgbmphys", "cbphys", "lstm", "lstmphys"]
tiers = ["hard", "medium", "easy"]
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(key_models)); w = 0.26
for i, tier in enumerate(tiers):
    sub = Rv[Rv["train_tier"] == tier]
    if len(sub) == 0:
        continue
    ys = [sub[f"{m}_KGE"].median() if f"{m}_KGE" in sub.columns else np.nan for m in key_models]
    ax.bar(x + (i-1)*w, ys, w, label=f"{tier} (n={len(sub)})")
ax.axhline(0.5, color="black", ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in key_models], rotation=30, ha="right")
ax.set_ylabel("Median KGE")
ax.set_title("Median KGE by difficulty tier (N=200)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig5_tiers.png"))
plt.savefig(os.path.join(FIG, "fig5_tiers.pdf"))
plt.close(fig)
print("fig5_tiers written")

# ---- Fig 6: KGE component decomposition (XGBoost raw vs Phys-XGBoost) ----
decomp = json.load(open(os.path.join(RES, "all_numbers_N200.json")))["decomp"]
fig, ax = plt.subplots(figsize=(7, 5))
comps = ["r", "alpha", "beta"]
labels6 = ["r (correlation)", r"$\alpha$ (variability)", r"$\beta$ (bias)"]
x6 = np.arange(len(comps)); w6 = 0.35
xb = [decomp["xgb"][c] for c in comps]
pb = [decomp["phys"][c] for c in comps]
ax.bar(x6 - w6/2, xb, w6, label="XGBoost (raw)", color="#6c8ebf", edgecolor="black", linewidth=0.5)
ax.bar(x6 + w6/2, pb, w6, label="Phys-XGBoost", color="#1a7f37", edgecolor="black", linewidth=0.5)
ax.axhline(1.0, color="black", ls="--", lw=1, label="ideal (=1)")
ax.set_xticks(x6); ax.set_xticklabels(labels6)
ax.set_ylim(0.7, 1.05)
ax.set_ylabel("KGE component value")
ax.set_title("KGE component decomposition: XGBoost vs Phys-XGBoost (median, N=200)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig6_decomposition.png"))
plt.savefig(os.path.join(FIG, "fig6_decomposition.pdf"))
plt.close(fig)
print("fig6_decomposition written")

# ---- Fig 7: SHAP feature attribution (mean |SHAP| over reservoirs) ----
shap_mean = os.path.join(RES, "shap_phys_xgb_mean.csv")
if os.path.exists(shap_mean):
    sdf = pd.read_csv(shap_mean).sort_values("mean_abs_shap", ascending=True)
    nres = pd.read_csv(shap_mean).get("reservoir", pd.Series([None])).nunique()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(sdf["feature"], sdf["mean_abs_shap"], color="#3a8ed0", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP| share (%) over reservoirs (Phys-XGBoost)")
    ax.set_title("SHAP feature attribution for Phys-XGBoost (N=200)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig7_shap.png"))
    plt.savefig(os.path.join(FIG, "fig7_shap.pdf"))
    plt.close(fig)
    print(f"fig7_shap written (from {shap_mean}, n_res={nres})")
else:
    print("shap_phys_xgb_mean.csv NOT FOUND; run SHAP regeneration first (fig7 skipped)")

print("DONE. Figures in", FIG)
print(sorted(os.listdir(FIG)))
