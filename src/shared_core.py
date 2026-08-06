"""
shared_core.py -- metrics, feature engineering, and the chronological split used by
BOTH run_experiment_v3.py (tree/gradient-boosting comparison) and run_lstm_only.py
(LSTM comparison). This module depends ONLY on numpy / pandas / scipy so it can be
imported safely in a torch-only process (run_lstm_only) WITHOUT pulling in
xgboost/lightgbm/catboost, whose OpenMP runtimes segfault when co-loaded with torch.

Single source of truth: keep feature definitions in sync here only.
"""

import numpy as np
import pandas as pd


# ---------------- metrics ----------------
def _clean(obs, sim):
    obs = np.asarray(obs, float); sim = np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]


def nse(obs, sim):
    obs, sim = _clean(obs, sim)
    if len(obs) < 2: return np.nan
    den = np.sum((obs - obs.mean())**2)
    return np.nan if den == 0 else 1 - np.sum((obs - sim)**2) / den


def log_nse(obs, sim):
    obs, sim = _clean(obs, sim)
    if len(obs) < 2: return np.nan
    o = np.log(obs - obs.min() + 1.0); s = np.log(sim - obs.min() + 1.0)
    den = np.sum((o - o.mean())**2)
    return np.nan if den == 0 else 1 - np.sum((o - s)**2) / den


def rmse(obs, sim):
    obs, sim = _clean(obs, sim)
    return np.nan if len(obs) < 1 else np.sqrt(np.mean((obs - sim)**2))


def kge(obs, sim):
    obs, sim = _clean(obs, sim)
    if len(obs) < 2: return (np.nan, np.nan, np.nan, np.nan)
    so, ss = obs.std(ddof=1), sim.std(ddof=1)
    mo, ms = obs.mean(), sim.mean()
    r = np.corrcoef(obs, sim)[0, 1] if so > 0 and ss > 0 else np.nan
    alpha = ss / so if so > 0 else np.nan
    beta = ms / mo if mo != 0 else np.nan
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return (np.nan, r, alpha, beta)
    return (1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2), r, alpha, beta)


def all_metrics(obs, sim):
    k, r, a, b = kge(obs, sim)
    return {"KGE": k, "r": r, "alpha": a, "beta": b,
            "NSE": nse(obs, sim), "logNSE": log_nse(obs, sim),
            "RMSE": rmse(obs, sim)}


# ---------------- features ----------------
RAW_FEATS = [
    "inflow", "inflow_lag1", "inflow_lag2", "inflow_lag3",
    "storage_lag1", "storage_lag2",
    "doy_sin", "doy_cos",
]
PHYS_CLASSES = {
    "mass_balance":  ["storage_trend", "throughput_ratio"],
    "storage_state": ["storage_norm", "storage_deficit"],
    "flow_anomaly":  ["inflow_norm", "inflow_ma7", "inflow_ma30", "inflow_anom7"],
}
PHYS_FEATS = sum(PHYS_CLASSES.values(), [])


def build_features(df, capacity=None, mean_inflow=None, train_end="2016-01-01",
                   strict_windows=True):
    """Calendar-safe feature construction (reviewer P-M8 / P0 temporal semantics).

    See run_experiment_v3.build_features for the full rationale. Reindexes each
    reservoir to a complete daily calendar before any lag/window so .shift(k) is
    exactly k calendar days; rows whose required timestamps are missing become
    NaN and drop out in split_chrono. Kept in sync with run_experiment_v3 as the
    single source of truth for feature engineering (shared by the LSTM path).
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
    full_idx = pd.date_range(d.index.min(), d.index.max(), freq="D")
    d = d.reindex(full_idx)
    d["doy"] = d.index.dayofyear
    d["doy_sin"] = np.sin(2*np.pi*d["doy"]/365.25)
    d["doy_cos"] = np.cos(2*np.pi*d["doy"]/365.25)
    d["inflow_lag1"] = d["inflow"].shift(1)
    d["inflow_lag2"] = d["inflow"].shift(2)
    d["inflow_lag3"] = d["inflow"].shift(3)
    d["storage_lag1"] = d["storage"].shift(1)
    d["storage_lag2"] = d["storage"].shift(2)
    cap = capacity if capacity and capacity > 0 else d["storage"].quantile(0.99)
    d["storage_norm"] = d["storage_lag1"] / cap
    # FIX (reviewer P3, climatology leakage): seasonal mean from TRAINING period
    # only (matches split_chrono's 2016-01-01 cutoff).
    _tr = d.index < pd.Timestamp(train_end)
    _doy_mean = d.loc[_tr].groupby(d.loc[_tr].index.dayofyear)["storage"].mean()
    _fallback = d.loc[_tr, "storage"].mean()
    d["storage_deficit"] = d["storage_lag1"] - d["doy"].map(_doy_mean).fillna(_fallback)
    mi = mean_inflow if mean_inflow and mean_inflow > 0 else d["inflow"].mean()
    d["inflow_norm"] = d["inflow"] / mi
    mp7 = 7 if strict_windows else 1
    mp30 = 30 if strict_windows else 1
    d["inflow_ma7"]  = d["inflow"].rolling(7, min_periods=mp7).mean().shift(1)
    d["inflow_ma30"] = d["inflow"].rolling(30, min_periods=mp30).mean().shift(1)
    d["inflow_anom7"] = d["inflow"] - d["inflow"].rolling(7, min_periods=mp7).mean().shift(1)
    d["storage_trend"] = d["storage_lag1"] - d["storage_lag2"]
    d["throughput_ratio"] = d["inflow"] / (d["storage_lag1"].abs() + 1e-6)
    d = d.reset_index().rename(columns={"index": "date"})
    return d


def split_chrono(d, test_start="2016-01-01"):
    d = d.dropna(subset=RAW_FEATS + PHYS_FEATS + ["outflow"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    return d[d["date"] < test_start], d[d["date"] >= test_start]
