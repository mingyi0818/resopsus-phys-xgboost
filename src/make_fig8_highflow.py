# -*- coding: utf-8 -*-
"""极端事件 top-10% inflow KGE 箱线图（fig8_highflow_kge）
使用已有 per_reservoir_metrics.csv 中的 *_KGE_hi 列，无需重跑实验。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"D:\workbuddy\WB03_ResOpsUS_PhysXGBoost"
METRICS = os.path.join(BASE, "src", "results", "per_reservoir_metrics.csv")
FIG_DIR = os.path.join(BASE, "manuscript", "figures")

R = pd.read_csv(METRICS)
models_hi = ["xgb_KGE_hi", "phys_KGE_hi", "lgbm_KGE_hi", "lgbmphys_KGE_hi",
             "cb_KGE_hi", "cbphys_KGE_hi"]
labels = ["XGBoost\n(raw)", "Phys-XGBoost\n(this work)", "LightGBM\n(raw)",
          "LightGBM\n(phys)", "CatBoost\n(raw)", "CatBoost\n(phys)"]
colors = ["#d62728","#2ca02c","#ff7f0e","#ff7f0e","#1f77b4","#1f77b4"]

data = []
for m in models_hi:
    vals = R[m].dropna().values
    data.append(vals)

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"linewidth": 1}, capprops={"linewidth": 1})
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("KGE (top-10% inflow events)", fontsize=11)
ax.set_title("Extreme-value segment performance", fontsize=12)
ax.axhline(0, color="gray", ls="--", lw=0.8)
# 标注中位数
for i, d in enumerate(data):
    med = np.median(d)
    ax.text(i+1, med, f"{med:.2f}", ha="center", va="bottom", fontsize=7,
            fontweight="bold" if i==1 else "normal")

plt.tight_layout()
for ext in [".png", ".pdf"]:
    fp = os.path.join(FIG_DIR, f"fig8_highflow_kge{ext}")
    plt.savefig(fp, dpi=200, bbox_inches="tight")
    print(f"[fig8] {fp}")
print("[fig8 完成]")
