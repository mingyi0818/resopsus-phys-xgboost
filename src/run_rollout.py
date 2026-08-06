"""
Iterative (auto-regressive) multi-step release simulation -- Tier 2D.

For each reservoir we train the three physics-augmented tree models on the SAME
chronological split as the main experiment, then evaluate them in an
auto-regressive "iterative simulation" setting that is directly comparable to
the recurrent release models of Tran et al. (2025, conditioned LSTM) and
Zhang et al. (2025, Mamba/S4D), which roll their own predictions forward.

Setting (honest, stated in the manuscript):
  * Future INFLOW is taken from the true record (perfect-forcing / perfect
    meteorological prognosis). This isolates the error propagation due to
    release-prediction mistakes through the storage-accounting state.
  * FUTURE STORAGE is ROLLED FORWARD with the model's OWN predicted releases
    via mass balance:  S(t) = S(t-1) + I(t) - O_hat(t-1).
  * At lead L we predict O(tau+L) by rolling the state L steps and compare to
    the true O(tau+L). This yields a KGE-vs-lead-time skill curve (1..7 days).

A persistence-L baseline O_hat(tau+L) = O(tau) (repeat last known release) is
reported as the difficulty reference.

Outputs (results/):
  per_reservoir_rollout_<rs>_<rc>.csv   (per-reservoir KGE by model x lead)
  rollout_summary.json                  (median KGE by model x lead, written by MERGE)
Chunked + resumable exactly like run_experiment_v3 (skip if CSV exists).
"""
import os
# Keep torch OUT of this process (OpenMP segfault with xgb/lgbm/cb on Windows).
# This module runs tree-model rollout only.
os.environ["SKIP_LSTM"] = "1"
import glob
import time
import json
import numpy as np
import pandas as pd

import run_experiment_v3 as R

EXTRACT_DIR = R.EXTRACT_DIR
OUT_DIR = R.OUT_DIR
RAW_FEATS = R.RAW_FEATS
PHYS_FEATS = R.PHYS_FEATS
ALL_FEATS = RAW_FEATS + PHYS_FEATS
MODELS = ["phys", "lgbmphys", "cbphys", "xgb", "lgbm", "cb"]
MODEL_FEATS = {
    "phys": ALL_FEATS, "lgbmphys": ALL_FEATS, "cbphys": ALL_FEATS,
    "xgb": RAW_FEATS, "lgbm": RAW_FEATS, "cb": RAW_FEATS,
}
LEADS = [1, 2, 3, 4, 5, 6, 7]


def kge1(obs, sim):
    return R.kge(obs, sim)[0]


def select(N):
    csvs = sorted(glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True))
    selected = []
    for c in csvs:
        try:
            df = pd.read_csv(c)
        except Exception:
            continue
        if not {"date", "inflow", "outflow", "storage"}.issubset(df.columns):
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        comp = df.dropna(subset=["inflow", "outflow", "storage"])
        if len(comp) < 5000:
            continue
        post = comp[comp["date"] >= pd.Timestamp("2016-01-01")]
        pre = comp[comp["date"] < pd.Timestamp("2016-01-01")]
        if len(post) < 500 or len(pre) < 3000:
            continue
        selected.append((c, len(comp)))
    selected.sort(key=lambda x: -x[1])
    if N < len(selected):
        idx = np.linspace(0, len(selected) - 1, N).round().astype(int)
        selected = [selected[i] for i in idx]
    return selected


def build_state_arrays(df_res):
    """Precompute true-feature arrays for the full reservoir (train+test)."""
    df = df_res.sort_values("date").reset_index(drop=True)
    if len(df) < 365 * 6:
        return None
    capacity = df["storage"].quantile(0.99)
    mean_inflow = df["inflow"].mean()
    d = R.build_features(df, capacity, mean_inflow)
    train, test = R.split_chrono(d)
    if len(train) < 365 * 3 or len(test) < 200:
        return None
    # Keep ONLY complete-case rows, exactly like the main one-step experiment
    # (run_experiment_v3.evaluate drops rows with any NaN feature). This is
    # essential for the lead-1 == direct-one-step invariant: after dropping the
    # calendar gap rows, consecutive test rows have storage_lag1[tau] ==
    # storage_lag2[tau+1], so the rollout's lead-1 reproduces the one-step model
    # exactly. Without this, gap-boundary rows violate the invariant.
    test = test.dropna(subset=RAW_FEATS + PHYS_FEATS).reset_index(drop=True)
    if len(test) < 200:
        return None
    n = len(test)
    inflow = test["inflow"].values.astype(float)
    # TRUE end-of-day storage at each test row (its calendar day). The rollout
    # state is re-initialised from THIS (not from storage_lag1): storage(tau) at
    # issue time tau. storage_lag1 only equals storage(tau) when rows are
    # calendar-adjacent, so using it broke the lead-1 == direct invariant
    # across date gaps (P-M8 / P-M6).
    storage_true = test["storage"].values.astype(float)
    outflow_true = test["outflow"].values.astype(float)
    date = pd.to_datetime(test["date"]).values
    doy = pd.to_datetime(test["date"]).dt.dayofyear.values
    # precompute inflow-based rolling features (time-invariant under perfect inflow)
    inflow_lag1 = test["inflow_lag1"].values.astype(float)
    inflow_lag2 = test["inflow_lag2"].values.astype(float)
    inflow_lag3 = test["inflow_lag3"].values.astype(float)
    inflow_norm = test["inflow_norm"].values.astype(float)
    inflow_ma7 = test["inflow_ma7"].values.astype(float)
    inflow_ma30 = test["inflow_ma30"].values.astype(float)
    inflow_anom7 = test["inflow_anom7"].values.astype(float)
    storage_deficit_base = test["storage_deficit"].values.astype(float)  # = storage_lag1 - doy_mean[doy]
    # Recover the per-row seasonal (doy) mean so that feat_row can reconstruct
    # storage_deficit = storage_lag1 - doy_mean at ANY rolled state:
    #   doy_mean = storage_lag1 - (storage_lag1 - doy_mean)
    seasonal_means = (test["storage_lag1"].values.astype(float) - storage_deficit_base)  # = doy_mean per row
    doy_sin = test["doy_sin"].values.astype(float)
    doy_cos = test["doy_cos"].values.astype(float)
    return dict(n=n, inflow=inflow, storage_true=storage_true, outflow_true=outflow_true,
                date=date, doy=doy, inflow_lag1=inflow_lag1, inflow_lag2=inflow_lag2,
                inflow_lag3=inflow_lag3, inflow_norm=inflow_norm, inflow_ma7=inflow_ma7,
                inflow_ma30=inflow_ma30, inflow_anom7=inflow_anom7, seasonal_means=seasonal_means,
                doy_sin=doy_sin, doy_cos=doy_cos, capacity=capacity, mean_inflow=mean_inflow,
                train=train, test=test)


def feat_row(S, idx, s_prev, s_prev2):
    """One feature row at test index `idx` with ROLLED storage state."""
    storage_norm = s_prev / S["capacity"]
    storage_deficit = s_prev - S["seasonal_means"][idx]
    storage_trend = s_prev - s_prev2
    throughput = S["inflow"][idx] / (abs(s_prev) + 1e-6)
    return {
        "inflow": S["inflow"][idx],
        "inflow_lag1": S["inflow_lag1"][idx],
        "inflow_lag2": S["inflow_lag2"][idx],
        "inflow_lag3": S["inflow_lag3"][idx],
        "storage_lag1": s_prev,
        "storage_lag2": s_prev2,
        "doy_sin": S["doy_sin"][idx],
        "doy_cos": S["doy_cos"][idx],
        "storage_norm": storage_norm,
        "storage_deficit": storage_deficit,
        "inflow_norm": S["inflow_norm"][idx],
        "inflow_ma7": S["inflow_ma7"][idx],
        "inflow_ma30": S["inflow_ma30"][idx],
        "inflow_anom7": S["inflow_anom7"][idx],
        "storage_trend": storage_trend,
        "throughput_ratio": throughput,
    }


def rollout_reservoir(S, models, model_feats):
    """Run lead-1..7 AR rollout for each model; return (per-model KGE by lead,
    raw prediction arrays by lead).

    Optimized: for each issue time tau we roll ONE Lmax=7-step trajectory
    (re-initialised from the true state at tau, exactly as the iterative
    simulators of Tran/Zhang do), recording the prediction at every step and
    assigning it to its lead. Single-row numpy predict (no per-row DataFrame)
    keeps the N=200 run tractable.

    The raw prediction arrays (results[m][L], length n, NaN where invalid) are
    returned alongside the KGE so the lead-1 pass can be validated against the
    direct one-step prediction (see run_one / P-M6).
    """
    n = S["n"]
    Lmax = max(LEADS)  # 7
    pred_arrays = {m: {L: np.full(n, np.nan, dtype=float) for L in LEADS} for m in models}
    kge_results = {m: {L: np.nan for L in LEADS} for m in models}
    date = S["date"]
    # window_ok[tau]: rows tau-1 .. tau+Lmax are CONSECUTIVE calendar days (no
    # gaps). Only within such a window is the autoregressive rollout
    # calendar-valid: every lag/window then points at the true previous day,
    # and lead-1 exactly reproduces the direct one-step prediction. Issue times
    # that straddle a date gap are skipped (their predictions stay NaN and are
    # excluded from KGE and from the lead-1 == direct validation).
    window_ok = np.zeros(n, dtype=bool)
    for tau in range(0, n - Lmax):
        if tau - 1 < 0:
            continue
        gap_days = int((date[tau + Lmax] - date[tau - 1]) / np.timedelta64(1, "D"))
        if gap_days == Lmax + 1:
            window_ok[tau] = True
    for mname, model in models.items():
        feats = model_feats[mname]
        for tau in range(0, n - Lmax):
            if not window_ok[tau]:
                continue
            # re-initialise the rolled state from the TRUE storage at issue time
            # tau (storage(tau)), so lead-1 == direct one-step prediction.
            cur_s = S["storage_true"][tau]
            cur_s2 = (S["storage_true"][tau - 1]
                      if not np.isnan(S["storage_true"][tau - 1]) else cur_s)
            for s in range(1, Lmax + 1):
                idx = tau + s
                row = feat_row(S, idx, cur_s, cur_s2)
                X = np.array([[row[f] for f in feats]], dtype=float)
                o = float(model.predict(X)[0])
                pred_arrays[mname][s][tau + s] = o
                # mass-balance update of rolled end-of-day storage
                new_s = cur_s + S["inflow"][idx] - o
                cur_s2, cur_s = cur_s, new_s
        for L in LEADS:
            preds = pred_arrays[mname][L]
            valid = ~np.isnan(preds)
            kge_results[mname][L] = (kge1(S["outflow_true"][valid], preds[valid])
                                    if valid.sum() > 10 else np.nan)
    return kge_results, pred_arrays, window_ok


def persistence_L_baseline(S):
    n = S["n"]
    out = {}
    for L in LEADS:
        preds = np.full(n, np.nan, dtype=float)
        for tau in range(0, n - L):
            preds[tau + L] = S["outflow_true"][tau]  # repeat last known release
        valid = ~np.isnan(preds)
        out[L] = kge1(S["outflow_true"][valid], preds[valid]) if valid.sum() > 10 else np.nan
    return out


def run_one(c, name):
    df = pd.read_csv(c)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["inflow", "outflow", "storage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["date", "inflow", "outflow", "storage"]]
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["outflow", "inflow", "storage"])
    S = build_state_arrays(df)
    if S is None:
        return None
    train, test = S["train"], S["test"]
    y = train["outflow"].values
    models = {
        "phys": R._xgb().fit(train[ALL_FEATS], y),
        "lgbmphys": R._lgbm().fit(train[ALL_FEATS], y),
        "cbphys": R._cb().fit(train[ALL_FEATS], y),
        "xgb": R._xgb().fit(train[RAW_FEATS], y),
        "lgbm": R._lgbm().fit(train[RAW_FEATS], y),
        "cb": R._cb().fit(train[RAW_FEATS], y),
    }
    res, pred_arrays, window_ok = rollout_reservoir(S, models, MODEL_FEATS)
    persist = persistence_L_baseline(S)
    row = {"reservoir": name, "n_test": S["n"]}
    for m in MODELS:
        for L in LEADS:
            row[f"{m}_KGE_L{L}"] = res[m].get(L, np.nan)
    for L in LEADS:
        row[f"persist_KGE_L{L}"] = persist.get(L, np.nan)
    # ---- P-M6 / P0 temporal-semantics validation -------------------------
    # Lead-1 rollout must equal the DIRECT one-step prediction at the SAME
    # issue time and TRUE state. This is asserted ONLY at gap-free issue times
    # (window_ok), where the rollout is calendar-valid and the two must agree
    # to floating point. Issue times that straddle a date gap are excluded.
    n = S["n"]
    Lmax = max(LEADS)
    align = {"reservoir": name, "n_test": n, "n_window_ok": int(window_ok.sum()),
             "max_abs_lead1_diff": {}, "pass": True}
    for mname, model in models.items():
        feats = MODEL_FEATS[mname]
        y_hat_direct = np.asarray(model.predict(test[feats].values), dtype=float)
        lead1 = pred_arrays[mname][1]  # rollout lead-1 prediction array (length n)
        # lead1[k] is set only when issue tau=k-1 is window_ok
        win_lead1 = np.zeros(n, dtype=bool)
        win_lead1[1:] = window_ok[:n - 1]
        valid = win_lead1 & (~np.isnan(lead1)) & (~np.isnan(y_hat_direct))
        if valid.sum() == 0:
            align["max_abs_lead1_diff"][mname] = None
            continue
        diff = np.abs(lead1[valid] - y_hat_direct[valid])
        mad = float(diff.max())
        align["max_abs_lead1_diff"][mname] = mad
        if mad > 1e-9:
            align["pass"] = False
    return row, align


def main():
    N = int(os.environ.get("N_RES", "200"))
    CHUNK = 30
    sel = select(N)
    rs = int(os.environ.get("RES_START", "0"))
    rc = int(os.environ.get("RES_COUNT", "0"))
    if rc > 0:
        sel = sel[rs:rs + rc]
    partial = (f"per_reservoir_rollout_{rs}_{rc}.csv" if rc > 0
               else "per_reservoir_rollout.csv")
    print(f"[rollout] running on {len(sel)} reservoirs (N={N}, rs={rs}, rc={rc})", flush=True)
    rows = []
    aligns = []
    for i, (c, ncomp) in enumerate(sel):
        name = os.path.splitext(os.path.basename(c))[0]
        t0 = time.perf_counter()
        try:
            out = run_one(c, name)
        except Exception as e:
            print(f"  [{i + 1}] {name} FAILED: {e}", flush=True)
            continue
        if out is None:
            print(f"  [{i + 1}] {name} skipped", flush=True)
            continue
        row, align = out
        rows.append(row)
        aligns.append(align)
        if (i + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, partial), index=False)
        dt = time.perf_counter() - t0
        flag = "" if align["pass"] else "  <<< LEAD1 MISMATCH"
        print(f"  [{i + 1}] {name}: "
              f"phys L1={row.get('phys_KGE_L1', float('nan')):.3f} "
              f"L3={row.get('phys_KGE_L3', float('nan')):.3f} "
              f"L7={row.get('phys_KGE_L7', float('nan')):.3f} | {dt:.1f}s{flag}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, partial), index=False)
    # lead-1 alignment report (P-M6)
    import json as _json
    worst = {}
    overall_pass = True
    for a in aligns:
        if not a["pass"]:
            overall_pass = False
        for m, v in a["max_abs_lead1_diff"].items():
            if v is None:
                continue
            worst[m] = max(worst.get(m, 0.0), v)
    report = {
        "n_reservoirs": len(aligns),
        "tolerance": 1e-9,
        "overall_pass": overall_pass,
        "max_abs_lead1_diff_by_model": worst,
    }
    with open(os.path.join(OUT_DIR, "rollout_lead1_alignment.json"), "w") as f:
        _json.dump(report, f, indent=2)
    print(f"[rollout] wrote {partial} ({len(rows)} reservoirs)")
    print(f"[rollout] LEAD-1 DIRECT==ROLLOUT validation: "
          f"{'PASS' if overall_pass else 'FAIL'} (tol=1e-9); "
          f"worst max|diff| by model: {worst}")


if __name__ == "__main__":
    main()
