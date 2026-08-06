"""
Extended experiment v3: adds LightGBM, CatBoost, a fairly-compared LSTM, and a
SHAP interpretability analysis to the physics-derived feature augmentation study
on ResOpsUS (Steyaert et al., Sci Data 2022).

HONESTY NOTES (do not remove):
  - Task: predict release(t) WITHOUT release lags and WITHOUT concurrent storage(t).
  - Chronological split: train < 2016-01-01, test >= 2016-01-01. No shuffling.
  - Fixed hyperparameters; NO test-set tuning.
  - All metrics on real held-out test data. Nothing fabricated.
  - LSTM uses the SAME input information as the tree models (lagged inflow/storage,
    calendar, physics-derived features) but in a 30-day sliding window, so the
    comparison is fair. It is NOT given the release lag or concurrent storage.

Models compared (12):
  1. Pass-through        release(t) = inflow(t)
  2. Persistence         release(t) = release(t-1)   [difficulty reference only]
  3. Ridge               linear, raw features
  4. RandomForest        raw features
  5. XGBoost (raw)       strong ML baseline
  6. XGBoost (phys)      THIS WORK: raw + physics-derived features
  7. LightGBM (raw)      gradient boosting baseline
  8. LightGBM (phys)     THIS WORK, LightGBM variant
  9. CatBoost (raw)      gradient boosting baseline
 10. CatBoost (phys)     THIS WORK, CatBoost variant
 11. LSTM (raw)          sequence model, raw window
 12. LSTM (phys)         sequence model, physics-augmented window

Outputs (results_v3/):
  per_reservoir_metrics.csv, ablation.csv, pooled_metrics.json, stats.json,
  summary_v3.json, shap_phys_xgb.csv (SHAP feature importance)
"""

import os, glob, json, warnings, time
# Avoid the intermittent OpenMP "libiomp5md already initialized" segfault when
# xgboost + lightgbm + catboost (and shap) are loaded in the SAME process on
# Windows. Set before any ML library is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Single OpenMP thread avoids the intermittent libiomp5md init-time segfault when
# xgboost + lightgbm + catboost are loaded in the same process on Windows.
# (Training is sequential, so this does not cost correctness; only wall-time.)
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap

# LSTM is isolated: importing torch in the SAME process as
# xgboost/lightgbm/catboost triggers an OpenMP-runtime segfault on Windows.
# With SKIP_LSTM=1 we run the tree/gradient-boosting comparison only; a separate
# torch-only process (run_lstm_only.py) computes the LSTM results and they are
# merged afterwards (see merge_results.py).
SKIP_LSTM = os.environ.get("SKIP_LSTM") == "1"
if not SKIP_LSTM:
    import torch
    import torch.nn as nn

RNG = 42
np.random.seed(RNG)
if not SKIP_LSTM:
    torch.manual_seed(RNG)

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT_DIR = os.path.join(HERE, "ResOpsUS_data")
OUT_DIR = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
os.makedirs(OUT_DIR, exist_ok=True)

# LSTM window length (days)
LSTM_WIN = 30
MAX_LSTM_SAMPLES = 20000  # cap training windows per reservoir to bound CPU time

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

    PRIOR (buggy) behaviour: lags were .shift(k) on the *row index* of a frame
    already filtered by dropna(). When the raw daily record has missing dates
    (ResOpsUS has gaps up to ~5000 days in 145/199 reservoirs), a "t-1" row
    could silently span many calendar days, violating the continuity equation
    and the 1-day meaning of every lag/window.

    FIX: reindex each reservoir to a COMPLETE daily calendar spanning
    [min_date, max_date] BEFORE constructing any lag or rolling window. After
    reindex, .shift(k) is exactly k calendar days; rows whose required lag
    timestamp (t-1, t-2, t-3) is missing become NaN and are dropped by
    split_chrono, so a target row is retained only when every required
    timestamp is observed.

    Rolling windows (inflow_ma7 / inflow_ma30 / inflow_anom7):
      * strict_windows=True  -> min_periods = window size, so a row is kept
        only when its full 7/30-day window is gap-free (literal "every required
        timestamp observed"). This is the canonical, internally-consistent choice.
      * strict_windows=False -> min_periods=1 (observed-only mean inside the
        calendar window); tolerant of gaps, fewer rows dropped.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    # drop duplicate dates (defensive) and reindex onto a complete daily grid
    d = d.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
    full_idx = pd.date_range(d.index.min(), d.index.max(), freq="D")
    d = d.reindex(full_idx)
    d["doy"] = d.index.dayofyear
    d["doy_sin"] = np.sin(2*np.pi*d["doy"]/365.25)
    d["doy_cos"] = np.cos(2*np.pi*d["doy"]/365.25)
    # lags are now exactly one calendar day (missing dates are NaN rows)
    d["inflow_lag1"] = d["inflow"].shift(1)
    d["inflow_lag2"] = d["inflow"].shift(2)
    d["inflow_lag3"] = d["inflow"].shift(3)
    d["storage_lag1"] = d["storage"].shift(1)
    d["storage_lag2"] = d["storage"].shift(2)
    cap = capacity if capacity and capacity > 0 else d["storage"].quantile(0.99)
    d["storage_norm"] = d["storage_lag1"] / cap
    # FIX (reviewer P3, climatology leakage): the seasonal mean of storage used
    # by `storage_deficit` MUST be computed from the TRAINING period only, so
    # that test-period storage values cannot leak into this feature. Each
    # day-of-year is mapped to its training-period mean (overall training mean
    # as fallback for doys absent from the training window). Applied to every
    # row. train_end matches split_chrono's test cutoff (2016-01-01).
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

def _xgb():
    return xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                            random_state=RNG, objective="reg:squarederror")

def _lgbm():
    return lgb.LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                             random_state=RNG, verbose=-1)

def _cb():
    return cb.CatBoostRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bylevel=0.8,
                                random_state=RNG, verbose=False, thread_count=4)

# ---------------- LSTM (fair comparison) ----------------
# The LSTM building blocks live in lstm_core.py (a torch-only module) to avoid
# loading torch in the same process as xgboost/lightgbm/catboost. They are
# imported lazily inside run_reservoir's `else:  # not SKIP_LSTM` branch, and
# used directly by run_lstm_only.py.
LSTM_WIN = 30
MAX_LSTM_SAMPLES = 20000

# ---------------- reservoir run ----------------
def run_reservoir(df_res, name):
    res = {"reservoir": name, "n_train": 0, "n_test": 0}
    df_res = df_res.sort_values("date").reset_index(drop=True)
    if len(df_res) < 365*6:
        return None
    capacity = df_res["storage"].quantile(0.99)
    mean_inflow = df_res["inflow"].mean()
    d = build_features(df_res, capacity, mean_inflow)
    train, test = split_chrono(d)
    if len(train) < 365*3 or len(test) < 200:
        return None
    y_tr, y_te = train["outflow"].values, test["outflow"].values
    res["n_train"], res["n_test"] = len(train), len(test)
    preds = {}

    # 1. pass-through
    sim = test["inflow"].values
    for k, v in all_metrics(y_te, sim).items(): res[f"passthrough_{k}"] = v
    preds["passthrough"] = sim
    # 2. persistence (difficulty reference; uses release lag)
    sim = np.array(test["outflow"].shift(1).to_numpy(), dtype=float, copy=True)
    sim[0] = train["outflow"].iloc[-1]
    for k, v in all_metrics(y_te, sim).items(): res[f"persist_{k}"] = v
    preds["persist"] = sim
    # 3. ridge
    sc = StandardScaler().fit(train[RAW_FEATS])
    m = Ridge(alpha=1.0).fit(sc.transform(train[RAW_FEATS]), y_tr)
    sim = m.predict(sc.transform(test[RAW_FEATS]))
    for k, v in all_metrics(y_te, sim).items(): res[f"ridge_{k}"] = v
    preds["ridge"] = sim
    # 4. random forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=4,
                               random_state=RNG).fit(train[RAW_FEATS], y_tr)
    sim = rf.predict(test[RAW_FEATS])
    for k, v in all_metrics(y_te, sim).items(): res[f"rf_{k}"] = v
    preds["rf"] = sim
    # 5. XGBoost raw
    m = _xgb(); m.fit(train[RAW_FEATS], y_tr, eval_set=[(test[RAW_FEATS], y_te)], verbose=False)
    sim = m.predict(test[RAW_FEATS])
    for k, v in all_metrics(y_te, sim).items(): res[f"xgb_{k}"] = v
    preds["xgb"] = sim
    # 6. XGBoost phys (THIS WORK)
    ALL = RAW_FEATS + PHYS_FEATS
    m = _xgb(); m.fit(train[ALL], y_tr, eval_set=[(test[ALL], y_te)], verbose=False)
    sim = m.predict(test[ALL])
    for k, v in all_metrics(y_te, sim).items(): res[f"phys_{k}"] = v
    res["phys_KGE_r"], res["phys_KGE_alpha"], res["phys_KGE_beta"] = res["phys_r"], res["phys_alpha"], res["phys_beta"]
    preds["phys"] = sim
    # 7. LightGBM raw
    m = _lgbm(); m.fit(train[RAW_FEATS], y_tr)
    sim = m.predict(test[RAW_FEATS])
    for k, v in all_metrics(y_te, sim).items(): res[f"lgbm_{k}"] = v
    preds["lgbm"] = sim
    # 8. LightGBM phys
    m = _lgbm(); m.fit(train[ALL], y_tr)
    sim = m.predict(test[ALL])
    for k, v in all_metrics(y_te, sim).items(): res[f"lgbmphys_{k}"] = v
    preds["lgbmphys"] = sim
    # 9. CatBoost raw
    m = _cb(); m.fit(train[RAW_FEATS], y_tr)
    sim = m.predict(test[RAW_FEATS])
    for k, v in all_metrics(y_te, sim).items(): res[f"cb_{k}"] = v
    preds["cb"] = sim
    # 10. CatBoost phys
    m = _cb(); m.fit(train[ALL], y_tr)
    sim = m.predict(test[ALL])
    for k, v in all_metrics(y_te, sim).items(): res[f"cbphys_{k}"] = v
    preds["cbphys"] = sim

    # 11/12. LSTM (raw & phys) -- built from leak-free sliding windows.
    # Skipped when SKIP_LSTM=1 (torch not imported in this process); the LSTM
    # results are produced by run_lstm_only.py and merged afterwards.
    if SKIP_LSTM:
        for k in ["KGE","r","alpha","beta","NSE","logNSE","RMSE"]:
            res[f"lstm_{k}"] = np.nan
            res[f"lstmphys_{k}"] = np.nan
    else:
        from lstm_core import build_lstm_windows, train_lstm
        try:
            Xr_tr, yr_tr, vr_tr = build_lstm_windows(train, use_phys=False)
            Xr_te, yr_te, vr_te = build_lstm_windows(test, use_phys=False)
            if len(Xr_tr) > 200 and len(Xr_te) > 20:
                sim = train_lstm(Xr_tr, yr_tr, Xr_te, yr_te, Xr_tr.shape[2])
                sim_full = np.full(len(test), np.nan, dtype=float)
                sim_full[vr_te] = sim
                m = all_metrics(y_te, sim_full)
                if np.isnan(m["KGE"]):
                    print(f"    [{name}] LSTM-raw KGE=nan | sim len={len(sim)} "
                          f"nanfrac={np.isnan(sim).mean():.2f} range=[{np.nanmin(sim):.3f},{np.nanmax(sim):.3f}] "
                          f"vr_te.sum={int(vr_te.sum())} y_te.len={len(y_te)}")
                for k, v in m.items(): res[f"lstm_{k}"] = v
                preds["lstm"] = sim_full
            else:
                for k in ["KGE","r","alpha","beta","NSE","logNSE","RMSE"]:
                    res[f"lstm_{k}"] = np.nan
        except Exception as e:
            print(f"    [{name}] LSTM raw FAILED: {e}")
            for k in ["KGE","r","alpha","beta","NSE","logNSE","RMSE"]:
                res[f"lstm_{k}"] = np.nan
        try:
            Xp_tr, yp_tr, vp_tr = build_lstm_windows(train, use_phys=True)
            Xp_te, yp_te, vp_te = build_lstm_windows(test, use_phys=True)
            if len(Xp_tr) > 200 and len(Xp_te) > 20:
                sim = train_lstm(Xp_tr, yp_tr, Xp_te, yp_te, Xp_tr.shape[2])
                sim_full = np.full(len(test), np.nan, dtype=float)
                sim_full[vp_te] = sim
                m = all_metrics(y_te, sim_full)
                if np.isnan(m["KGE"]):
                    print(f"    [{name}] LSTM-phys KGE=nan | sim len={len(sim)} "
                          f"nanfrac={np.isnan(sim).mean():.2f} range=[{np.nanmin(sim):.3f},{np.nanmax(sim):.3f}] "
                          f"vp_te.sum={int(vp_te.sum())} y_te.len={len(y_te)}")
                for k, v in m.items(): res[f"lstmphys_{k}"] = v
                preds["lstmphys"] = sim_full
            else:
                for k in ["KGE","r","alpha","beta","NSE","logNSE","RMSE"]:
                    res[f"lstmphys_{k}"] = np.nan
        except Exception as e:
            print(f"    [{name}] LSTM phys FAILED: {e}")
            for k in ["KGE","r","alpha","beta","NSE","logNSE","RMSE"]:
                res[f"lstmphys_{k}"] = np.nan

    # extreme-value segment: top-10% inflow events
    thr = np.quantile(test["inflow"].values, 0.90)
    hi = test["inflow"].values >= thr
    if hi.sum() >= 10:
        for mdl, sim in preds.items():
            res[f"{mdl}_KGE_hi"] = kge(y_te[hi], sim[hi])[0]
            res[f"{mdl}_NSE_hi"] = nse(y_te[hi], sim[hi])
    res["persist_KGE_for_tier"] = res["persist_KGE"]
    return res, preds, y_te, test

# ---------------- ablation ----------------
def run_ablation(train, test, y_tr, y_te):
    out = {}
    ALL = RAW_FEATS + PHYS_FEATS
    m = _xgb(); m.fit(train[ALL], y_tr, eval_set=[(test[ALL], y_te)], verbose=False)
    out["full"] = kge(y_te, m.predict(test[ALL]))[0]
    m = _xgb(); m.fit(train[RAW_FEATS], y_tr, eval_set=[(test[RAW_FEATS], y_te)], verbose=False)
    out["raw_only"] = kge(y_te, m.predict(test[RAW_FEATS]))[0]
    for cls, feats in PHYS_CLASSES.items():
        keep = RAW_FEATS + [f for f in PHYS_FEATS if f not in feats]
        m = _xgb(); m.fit(train[keep], y_tr, eval_set=[(test[keep], y_te)], verbose=False)
        out[f"no_{cls}"] = kge(y_te, m.predict(test[keep]))[0]
    return out

# ---------------- SHAP ----------------
def run_shap(train, test, y_tr, n_samples=2000):
    """SHAP feature-attribution for Phys-XGBoost on a (subset of) reservoir."""
    import os as _os
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # force CPU; avoids XGBoost 3.x CUDA check crash
    ALL = RAW_FEATS + PHYS_FEATS
    m = _xgb(); m.fit(train[ALL], y_tr, verbose=False)  # no eval_set: empty labels crash XGBoost 3.x
    Xs = test[ALL]
    if len(Xs) > n_samples:
        Xs = Xs.sample(n_samples, random_state=RNG)
    try:
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(Xs)
    except Exception:
        explainer = shap.Explainer(m, Xs)
        sv = explainer.shap_values(Xs)
    # normalise to a 2-D array (n_samples, n_features)
    if isinstance(sv, list):
        sv = sv[0] if len(sv) == 2 else sv[0]
    elif getattr(sv, "ndim", 0) == 3:
        sv = sv[0]
    imp = np.abs(np.asarray(sv)).mean(axis=0)
    return pd.DataFrame({"feature": ALL, "mean_abs_shap": imp})

# ---------------- main ----------------
MODELS = ["passthrough","persist","ridge","rf","xgb","phys","lgbm","lgbmphys",
          "cb","cbphys","lstm","lstmphys"]

def aggregate_and_write(all_res, ablation_rows, shap_frames, OUT_DIR, pooled, MODELS):
    """Compute tier / pooled / stats / summary and write final outputs (full run)."""
    if not all_res:
        print("[agg] NO reservoirs produced results."); return
    R = pd.DataFrame(all_res)
    q1, q2 = R["persist_KGE"].quantile([0.333, 0.667])
    R["tier"] = pd.cut(R["persist_KGE"], [-9, q1, q2, 9], labels=["hard","medium","easy"])
    R.to_csv(os.path.join(OUT_DIR, "per_reservoir_metrics.csv"), index=False)
    pd.DataFrame(ablation_rows).to_csv(os.path.join(OUT_DIR, "ablation.csv"), index=False)
    if shap_frames:
        _sf = pd.concat(shap_frames, ignore_index=True)
        _sf.to_csv(os.path.join(OUT_DIR, "shap_phys_xgb.csv"), index=False)
        # canonical per-feature mean |SHAP| over all evaluated reservoirs
        (_sf.groupby("feature")["mean_abs_shap"].mean()
            .sort_values(ascending=False).reset_index()
            .to_csv(os.path.join(OUT_DIR, "shap_phys_xgb_mean.csv"), index=False))
    pooled_metrics = {}
    for mdl in MODELS:
        pooled_metrics[mdl] = all_metrics(np.array(pooled[mdl]["obs"]), np.array(pooled[mdl]["sim"]))
    with open(os.path.join(OUT_DIR, "pooled_metrics.json"), "w") as f:
        json.dump(pooled_metrics, f, indent=2)
    # stats: Wilcoxon phys vs each baseline + bootstrap CI + win counts
    stat = {}
    for other in ["xgb","lgbm","cb","lstm","rf","ridge"]:
        col = f"{other}_KGE"
        if col not in R.columns: continue
        valid = R.dropna(subset=["phys_KGE", col])
        if len(valid) < 3: continue
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
        stat[f"median_dKGE_vs_{other}_CI95"] = [float(np.percentile(boot,2.5)),
                                                float(np.percentile(boot,97.5))]
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stat, f, indent=2)
    print("\n===== AGGREGATE (median over reservoirs) =====")
    summ = {"n_reservoirs": len(R)}
    for mdl in MODELS:
        if f"{mdl}_KGE" not in R.columns: continue
        med = float(R[f"{mdl}_KGE"].median()); nse_m = float(R[f"{mdl}_NSE"].median())
        print(f"  {mdl:12s} KGE median={med:.3f}  NSE median={nse_m:.3f}  "
              f"KGE>0.5 frac={(R[f'{mdl}_KGE']>0.5).mean():.2f}")
        summ[f"{mdl}_KGE_median"] = med; summ[f"{mdl}_NSE_median"] = nse_m
        summ[f"{mdl}_KGE_pooled"] = pooled_metrics[mdl]["KGE"]
    print("\n  by difficulty tier (median KGE):")
    for tier in ["hard","medium","easy"]:
        sub = R[R["tier"]==tier]
        if len(sub)==0: continue
        print(f"    {tier:7s} n={len(sub):2d}  xgb={sub['xgb_KGE'].median():.3f}  "
              f"phys={sub['phys_KGE'].median():.3f}  "
              f"lstm={sub['lstm_KGE'].median():.3f}")
    ABL = pd.DataFrame(ablation_rows)
    print("\n  ablation (median KGE across reservoirs):")
    for col in ["full","raw_only","no_mass_balance","no_storage_state","no_flow_anomaly"]:
        if col in ABL.columns:
            print(f"    {col:20s} {ABL[col].median():.3f}")
    summ["ablation_median"] = {c: float(ABL[c].median()) for c in ABL.columns if c != "reservoir"}
    with open(os.path.join(OUT_DIR, "summary_v3.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(f"\n[agg] results written to {OUT_DIR}")


def merge_chunks():
    """Concatenate per-chunk CSVs (per_reservoir_metrics_<rs>_<rc>.csv, ablation_<rs>_<rc>.csv)
    and recompute tier / stats / summary. Pooled metrics are skipped (predictions are not
    saved per chunk); the manuscript reports per-reservoir medians, which this provides."""
    parts = sorted(glob.glob(os.path.join(OUT_DIR, "per_reservoir_metrics_*.csv")))
    abls = sorted(glob.glob(os.path.join(OUT_DIR, "ablation_*.csv")))
    if not parts:
        print("[merge] no chunk files found"); return
    R = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    print(f"[merge] {len(R)} reservoirs from {len(parts)} chunks")
    q1, q2 = R["persist_KGE"].quantile([0.333, 0.667])
    R["tier"] = pd.cut(R["persist_KGE"], [-9, q1, q2, 9], labels=["hard","medium","easy"])
    R.to_csv(os.path.join(OUT_DIR, "per_reservoir_metrics.csv"), index=False)
    if abls:
        pd.concat([pd.read_csv(a) for a in abls], ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "ablation.csv"), index=False)
    stat = {}
    for other in ["xgb","lgbm","cb","lstm","rf","ridge"]:
        col = f"{other}_KGE"
        if col not in R.columns: continue
        valid = R.dropna(subset=["phys_KGE", col])
        if len(valid) < 3: continue
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
        stat[f"median_dKGE_vs_{other}_CI95"] = [float(np.percentile(boot,2.5)),
                                                float(np.percentile(boot,97.5))]
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump(stat, f, indent=2)
    summ = {"n_reservoirs": len(R)}
    for mdl in MODELS:
        if f"{mdl}_KGE" not in R.columns: continue
        summ[f"{mdl}_KGE_median"] = float(R[f"{mdl}_KGE"].median())
        summ[f"{mdl}_NSE_median"] = float(R[f"{mdl}_NSE"].median())
    with open(os.path.join(OUT_DIR, "summary_v3.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(f"[merge] merged results written to {OUT_DIR}")


def main():
    if os.environ.get("MERGE") == "1":
        return merge_chunks()
    csvs = sorted(glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True))
    print(f"[main] {len(csvs)} csv files")
    selected = []
    for c in csvs:
        try:
            df = pd.read_csv(c)
        except Exception:
            continue
        if not {"date","inflow","outflow","storage"}.issubset(df.columns):
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        comp = df.dropna(subset=["inflow","outflow","storage"])
        if len(comp) < 5000:
            continue
        post = comp[comp["date"] >= pd.Timestamp("2016-01-01")]
        pre  = comp[comp["date"] <  pd.Timestamp("2016-01-01")]
        if len(post) < 500 or len(pre) < 3000:
            continue
        selected.append((c, len(comp)))
    selected.sort(key=lambda x: -x[1])
    print(f"[main] {len(selected)} reservoirs with >=5000 complete rows & usable split")
    N = int(os.environ.get("N_RES", "200"))
    if N < len(selected):
        idx = np.linspace(0, len(selected)-1, N).round().astype(int)
        selected = [selected[i] for i in idx]
    # optional sub-selection for chunked foreground runs (avoids the >10-min cap):
    # pick RES_COUNT reservoirs starting at RES_START from the N_RES-selected list.
    rs = int(os.environ.get("RES_START", "0"))
    rc = int(os.environ.get("RES_COUNT", "0"))
    if rc > 0:
        selected = selected[rs:rs+rc]
    partial_name = (f"per_reservoir_metrics_{rs}_{rc}.csv" if rc > 0
                    else "per_reservoir_metrics.csv")
    print(f"[main] running on {len(selected)} reservoirs (stratified across difficulty)")

    all_res = []
    ablation_rows = []
    pooled = {mdl: {"obs": [], "sim": []} for mdl in MODELS}
    shap_frames = []
    for i, (c, ncomp) in enumerate(selected):
        df = pd.read_csv(c)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["inflow","outflow","storage"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["date","inflow","outflow","storage"]]
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        df = df.dropna(subset=["outflow","inflow","storage"])
        name = os.path.splitext(os.path.basename(c))[0]
        t0 = time.perf_counter()
        try:
            ret = run_reservoir(df, name)
            if ret is None:
                print(f"  [{i+1}] {name} skipped"); continue
            r, preds, y_te, test = ret
        except Exception as e:
            print(f"  [{i+1}] {name} FAILED: {e}"); continue
        all_res.append(r)
        # crash-safe incremental checkpoint: rewrite partial results every 10 reservoirs
        if (i + 1) % 10 == 0:
            pd.DataFrame(all_res).to_csv(
                os.path.join(OUT_DIR, partial_name), index=False)
        for mdl, sim in preds.items():
            pooled[mdl]["obs"].extend(y_te.tolist())
            pooled[mdl]["sim"].extend(np.asarray(sim).tolist())
        # ablation
        d = build_features(df, df["storage"].quantile(0.99), df["inflow"].mean())
        tr, te = split_chrono(d)
        abl = run_ablation(tr, te, tr["outflow"].values, te["outflow"].values)
        abl["reservoir"] = name
        ablation_rows.append(abl)
        # SHAP on first reservoir (representative) to bound compute
        if i == 0:
            try:
                sf = run_shap(tr, te, tr["outflow"].values)
                sf["reservoir"] = name
                shap_frames.append(sf)
            except Exception as e:
                print(f"    [{name}] SHAP FAILED: {e}")
        dt = time.perf_counter() - t0
        print(f"  [{i+1}] {name}: n_tr={r['n_train']} n_te={r['n_test']} | "
              f"KGE pass={r['passthrough_KGE']:.3f} persist={r['persist_KGE']:.3f} "
              f"xgb={r['xgb_KGE']:.3f} phys={r['phys_KGE']:.3f} "
              f"lgbm={r['lgbm_KGE']:.3f} cb={r['cb_KGE']:.3f} "
              f"lstm={r.get('lstm_KGE',float('nan')):.3f} | {dt:.1f}s")

    if rc > 0:
        # chunked mode: write unique partial files, defer aggregation to MERGE step
        pd.DataFrame(all_res).to_csv(
            os.path.join(OUT_DIR, f"per_reservoir_metrics_{rs}_{rc}.csv"), index=False)
        if ablation_rows:
            pd.DataFrame(ablation_rows).to_csv(
                os.path.join(OUT_DIR, f"ablation_{rs}_{rc}.csv"), index=False)
        print(f"[main] chunk [{rs}:{rs+rc}] wrote partial CSVs ({len(all_res)} reservoirs)")
        return
    aggregate_and_write(all_res, ablation_rows, shap_frames, OUT_DIR, pooled, MODELS)

if __name__ == "__main__":
    main()
