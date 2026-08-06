# -*- coding: utf-8 -*-
"""Generate figures for the Reservoir JHE manuscript (v3, 12 models).

Reads results from results_v3/ and writes PNGs into submission_JHE/figures/.
No raw predictions are required (uses median/pooled/per-reservoir KGE).
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_v3")
FIG = os.path.join(HERE, "submission_JHE", "figures")
os.makedirs(FIG, exist_ok=True)

# MODELS order (must match run_experiment_v3.py)
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
# colour: tree models blue, phys variants orange/teal, DL red, refs grey
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

summary = json.load(open(os.path.join(RES, "summary_v3.json")))
pooled = json.load(open(os.path.join(RES, "pooled_metrics.json")))
R = pd.read_csv(os.path.join(RES, "per_reservoir_metrics.csv"))


def kge_med(m):
    return summary.get(f"{m}_KGE_median", np.nan)

def kge_pooled(m):
    return pooled.get(m, {}).get("KGE", np.nan)


# ---- Fig 1: median KGE bar chart across 12 models ----
fig, ax = plt.subplots(figsize=(9, 5))
order = sorted(MODELS, key=lambda m: kge_med(m))
vals = [kge_med(m) for m in order]
colors = [color_for(m) for m in order]
bars = ax.bar([DISPLAY[m] for m in order], vals, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(0.5, color="black", ls="--", lw=1, label="KGE = 0.5 (skilful)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + (0.01 if v >= 0 else -0.03),
            f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
ax.set_ylabel("Median KGE (over reservoirs)")
ax.set_title("Reservoir release prediction — median KGE by model (12 models)")
ax.set_ylim(min(-3, min(vals) - 0.2), 1.05)
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_kge_median_bar.png"))
plt.close(fig)

# ---- Fig 2: pooled KGE bar chart ----
fig, ax = plt.subplots(figsize=(9, 5))
order2 = sorted(MODELS, key=lambda m: kge_pooled(m))
vals2 = [kge_pooled(m) for m in order2]
colors2 = [color_for(m) for m in order2]
bars = ax.bar([DISPLAY[m] for m in order2], vals2, color=colors2, edgecolor="black", linewidth=0.5)
ax.axhline(0.5, color="black", ls="--", lw=1, label="KGE = 0.5")
for b, v in zip(bars, vals2):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Pooled KGE (all test days)")
ax.set_title("Reservoir release prediction — pooled KGE by model (12 models)")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_kge_pooled_bar.png"))
plt.close(fig)

# ---- Fig 3: boxplot of per-reservoir KGE distribution ----
fig, ax = plt.subplots(figsize=(9, 5.5))
data, labels, cols = [], [], []
for m in MODELS:
    col = f"{m}_KGE"
    if col in R.columns:
        s = R[col].dropna().values
        if len(s) > 0:
            data.append(s); labels.append(DISPLAY[m]); cols.append(color_for(m))
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=1.2))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.axhline(0.5, color="black", ls="--", lw=1, label="KGE = 0.5")
ax.set_ylabel("Per-reservoir KGE")
ax.set_title("Distribution of per-reservoir KGE across 12 models (box = IQR, line = median)")
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_kge_boxplot.png"))   # manuscript Fig. 1
plt.savefig(os.path.join(FIG, "fig1_kge_boxplot.pdf"))
plt.close(fig)

# ---- Fig 2 (manuscript): boxplot of per-reservoir NSE distribution ----
fig, ax = plt.subplots(figsize=(9, 5.5))
data, labels, cols = [], [], []
for m in MODELS:
    col = f"{m}_NSE"
    if col in R.columns:
        s = R[col].dropna().values
        if len(s) > 0:
            data.append(s); labels.append(DISPLAY[m]); cols.append(color_for(m))
if data:
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.axhline(0.5, color="black", ls="--", lw=1, label="NSE = 0.5")
    ax.set_ylabel("Per-reservoir NSE")
    ax.set_title("Distribution of per-reservoir NSE across 12 models (box = IQR, line = median)")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig2_nse_boxplot.png"))   # manuscript Fig. 2
    plt.savefig(os.path.join(FIG, "fig2_nse_boxplot.pdf"))
    plt.close(fig)

# ---- Fig 4: tier-stratified median KGE (Phys vs XGB vs LSTM) ----
fig, ax = plt.subplots(figsize=(8, 5))
key_models = ["xgb", "phys", "lgbmphys", "cbphys", "lstm", "lstmphys"]
tiers = ["hard", "medium", "easy"]
x = np.arange(len(key_models)); w = 0.26
for i, tier in enumerate(tiers):
    sub = R[R["tier"] == tier]
    if len(sub) == 0:
        continue
    ys = [sub[f"{m}_KGE"].median() if f"{m}_KGE" in sub.columns else np.nan for m in key_models]
    ax.bar(x + (i-1)*w, ys, w, label=f"{tier} (n={len(sub)})")
ax.axhline(0.5, color="black", ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels([DISPLAY[m] for m in key_models], rotation=30, ha="right")
ax.set_ylabel("Median KGE")
ax.set_title("Median KGE by difficulty tier")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig4_tier_strat.png"))
plt.savefig(os.path.join(FIG, "fig5_tiers.png"))   # manuscript Fig. 5 (alias)
plt.close(fig)

# ---- Fig 5: ablation (median KGE across reservoirs) ----
ab = summary.get("ablation_median", {})
if ab:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = list(ab.keys())
    nice = {"full": "Full (all phys)", "raw_only": "Raw only (no phys)",
            "no_mass_balance": "− mass-balance", "no_storage_state": "− storage state",
            "no_flow_anomaly": "− flow anomaly", "no_climate": "− climate index"}
    disp = [nice.get(k, k) for k in keys]
    vals = [ab[k] for k in keys]
    bars = ax.barh(disp, vals, color="#1a7f37", edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_xlabel("Median KGE")
    ax.set_title("Ablation study — contribution of each physics-derived feature group")
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig5_ablation.png"))
    plt.savefig(os.path.join(FIG, "fig4_ablation.png"))   # manuscript Fig. 4 (alias)
    plt.close(fig)

# ---- Fig 6: KGE component decomposition (XGBoost raw vs Phys-XGBoost) ----
fig, ax = plt.subplots(figsize=(7, 5))
comps = ["r", "alpha", "beta"]
labels6 = ["r (correlation)", r"\alpha (variability)", r"\beta (bias)"]
x6 = np.arange(len(comps)); w6 = 0.35
xb = [pooled["xgb"][c] for c in comps]
pb = [pooled["phys"][c] for c in comps]
ax.bar(x6 - w6/2, xb, w6, label="XGBoost (raw)", color="#6c8ebf", edgecolor="black", linewidth=0.5)
ax.bar(x6 + w6/2, pb, w6, label="Phys-XGBoost", color="#1a7f37", edgecolor="black", linewidth=0.5)
ax.axhline(1.0, color="black", ls="--", lw=1, label="ideal (=1)")
ax.set_xticks(x6); ax.set_xticklabels(labels6)
ax.set_ylim(0.7, 1.05)
ax.set_ylabel("KGE component value")
ax.set_title("KGE component decomposition: XGBoost vs Phys-XGBoost (pooled)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig6_decomposition.png"))
plt.savefig(os.path.join(FIG, "fig6_decomposition.pdf"))
plt.close(fig)
print("fig6_decomposition written")

# ---- Fig 7: SHAP feature attribution (mean |SHAP| over 30 reservoirs) ----
shap_mean = os.path.join(RES, "shap_phys_xgb_mean.csv")
if os.path.exists(shap_mean):
    sdf = pd.read_csv(shap_mean).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(sdf["feature"], sdf["mean_abs_shap"], color="#3a8ed0", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP| value (Phys-XGBoost, averaged over 30 reservoirs)")
    ax.set_title("SHAP feature attribution for Phys-XGBoost")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig7_shap.png"))
    plt.savefig(os.path.join(FIG, "fig7_shap.pdf"))
    plt.close(fig)
    print("fig7_shap written")
else:
    print("shap_phys_xgb_mean.csv not found; skipping fig7")

print("Figures written to", FIG)
print("Files:", sorted(os.listdir(FIG)))
