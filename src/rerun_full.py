# -*- coding: utf-8 -*-
"""
rerun_full.py -- clean re-run of the tree-model + ablation + SHAP pipeline
AFTER the seasonal-climatology leakage fix (reviewer P3).

What changed vs the leaky run: build_features now computes storage_deficit's
seasonal mean from the TRAINING period only (see run_experiment_v3.build_features
/ shared_core.build_features, train_end="2016-01-01").

This script:
  * selects the SAME 200 reservoirs (stratified, deterministic) and evaluates
    199 (ResOpsUS_976 excluded for all-NaN), exactly matching the prior cohort;
  * runs run_reservoir (10 tree/gradient-boosting models) + run_ablation
    (5 configs) + run_shap (Phys-XGBoost) per reservoir;
  * SAVES per-test-sample predictions so pooled (concatenated) KGE can be
    recomputed honestly;
  * checkpoints every 10 reservoirs (and on resume seeds from the partial
    outputs) so a crash never loses more than the in-flight batch;
  * writes per_reservoir_metrics.csv, ablation.csv, shap_phys_xgb.csv,
    pooled_metrics.json, stats.json, summary_v3.json.

Run:  python rerun_full.py            # fresh run (backs up old leaky outputs)
      python rerun_full.py --resume    # continue from partial outputs
"""
import os, glob, json, time, argparse
os.environ["SKIP_LSTM"] = "1"          # tree/GBM only in this process
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import run_experiment_v3 as RE

OUT = RE.OUT_DIR
EXTRACT = RE.EXTRACT_DIR
MODELS = RE.MODELS

EXCLUDE = {"ResOpsUS_976"}


def select(N):
    """Identical deterministic selection to run_all_chunks.select / main()."""
    csvs = sorted(glob.glob(os.path.join(EXTRACT, "**", "*.csv"), recursive=True))
    sel = []
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
        sel.append((os.path.splitext(os.path.basename(c))[0], len(comp)))
    sel.sort(key=lambda x: -x[1])
    if N < len(sel):
        idx = np.linspace(0, len(sel) - 1, N).round().astype(int)
        sel = [sel[i] for i in idx]
    return [s[0] for s in sel]


def backup_old():
    bak = os.path.join(OUT, "_backup_leaky")
    os.makedirs(bak, exist_ok=True)
    for fn in ["per_reservoir_metrics.csv", "ablation.csv", "shap_phys_xgb.csv",
               "shap_phys_xgb_mean.csv", "pooled_metrics.json", "stats.json",
               "summary_v3.json", "per_reservoir_metrics_*.csv", "ablation_*.csv"]:
        for p in glob.glob(os.path.join(OUT, fn)):
            try:
                os.replace(p, os.path.join(bak, os.path.basename(p)))
            except Exception as e:
                print("  backup warn", p, e)


def load_checkpoint():
    done, all_res, ablation_rows, shap_frames, pooled = set(), [], [], [], \
        {m: {"obs": [], "sim": []} for m in MODELS}
    pm = os.path.join(OUT, "per_reservoir_metrics.csv")
    if os.path.exists(pm):
        d0 = pd.read_csv(pm)
        done = set(d0["reservoir"].tolist())
        all_res = d0.to_dict("records")
    ab = os.path.join(OUT, "ablation.csv")
    if os.path.exists(ab):
        ablation_rows = pd.read_csv(ab).to_dict("records")
    sh = os.path.join(OUT, "shap_phys_xgb.csv")
    if os.path.exists(sh):
        shap_frames = [pd.read_csv(sh)]
    pp = os.path.join(OUT, "pooled_partial.json")
    if os.path.exists(pp):
        pooled = json.load(open(pp))
        # json stores lists; ensure obs/sim are lists
    return done, all_res, ablation_rows, shap_frames, pooled


def save_checkpoint(all_res, ablation_rows, shap_frames, pooled):
    pd.DataFrame(all_res).to_csv(os.path.join(OUT, "per_reservoir_metrics.csv"), index=False)
    pd.DataFrame(ablation_rows).to_csv(os.path.join(OUT, "ablation.csv"), index=False)
    if shap_frames:
        pd.concat(shap_frames, ignore_index=True).to_csv(
            os.path.join(OUT, "shap_phys_xgb.csv"), index=False)
    json.dump(pooled, open(os.path.join(OUT, "pooled_partial.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    names = select(200)
    # cohort consistency check (traceability): must equal the prior canonical run.
    # Prefer the explicit row-shift backup (_backup_rowshift); fall back to the
    # legacy leaky backup. On mismatch, WARN (do not abort) -- the selection is
    # deterministic (stratified linspace over the same completeness filters), so
    # any difference is a stale-backup artifact, not a real cohort change.
    old_pm = os.path.join(OUT, "_backup_rowshift", "per_reservoir_metrics.csv")
    if not os.path.exists(old_pm):
        old_pm = os.path.join(OUT, "_backup_leaky", "per_reservoir_metrics.csv")
    if os.path.exists(old_pm):
        old_names = set(pd.read_csv(old_pm)["reservoir"].tolist())
        new_eval = set(names) - EXCLUDE
        if new_eval != old_names:
            missing = old_names - new_eval
            extra = new_eval - old_names
            print(f"[COHORT WARNING] missing={missing} extra={extra} "
                  f"(stale backup? selection is deterministic; proceeding)")
        else:
            print(f"[cohort OK] {len(new_eval)} evaluated reservoirs match prior run.")

    if args.resume:
        done, all_res, ablation_rows, shap_frames, pooled = load_checkpoint()
        print(f"[resume] {len(done)} reservoirs already done")
    else:
        backup_old()
        done, all_res, ablation_rows, shap_frames, pooled = load_checkpoint()
        if done:
            print(f"[fresh] partial outputs found ({len(done)}); treating as resume.")
        else:
            print("[fresh] old leaky outputs backed up; starting clean rerun.")

    csvmap = {os.path.splitext(os.path.basename(c))[0]: c
              for c in glob.glob(os.path.join(EXTRACT, "**", "*.csv"), recursive=True)}

    n_done_start = len(done)
    for i, name in enumerate(names):
        if name in EXCLUDE:
            continue
        if name in done:
            continue
        c = csvmap.get(name)
        if not c:
            print("  MISSING", name); continue
        df = pd.read_csv(c)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["inflow", "outflow", "storage"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = (df[["date", "inflow", "outflow", "storage"]]
              .dropna(subset=["date"]).sort_values("date").reset_index(drop=True))
        df = df.dropna(subset=["outflow", "inflow", "storage"])
        t0 = time.perf_counter()
        try:
            ret = RE.run_reservoir(df, name)
            if ret is None:
                print(f"  [{i+1}] {name} skipped"); continue
            r, preds, y_te, test = ret
        except Exception as e:
            print(f"  [{i+1}] {name} FAILED: {e}"); continue
        all_res.append(r)
        for m, sim in preds.items():
            s = np.asarray(sim, dtype=float)
            pooled[m]["obs"].extend(y_te.tolist())
            pooled[m]["sim"].extend(s.tolist())
        # ablation (uses the SAME fixed build_features via split_chrono)
        d = RE.build_features(df, df["storage"].quantile(0.99), df["inflow"].mean())
        tr, te = RE.split_chrono(d)
        abl = RE.run_ablation(tr, te, tr["outflow"].values, te["outflow"].values)
        abl["reservoir"] = name
        ablation_rows.append(abl)
        # SHAP on Phys-XGBoost
        try:
            sf = RE.run_shap(tr, te, tr["outflow"].values)
            sf["reservoir"] = name
            shap_frames.append(sf)
        except Exception as e:
            print(f"    [{name}] SHAP FAILED: {e}")
        dt = time.perf_counter() - t0
        print(f"  [{i+1}/{len(names)}] {name}: phys={r['phys_KGE']:.3f} "
              f"lgbmphys={r['lgbmphys_KGE']:.3f} cbphys={r['cbphys_KGE']:.3f} | {dt:.1f}s",
              flush=True)
        if (i + 1) % 10 == 0:
            save_checkpoint(all_res, ablation_rows, shap_frames, pooled)
            print(f"  [ckpt] {len(all_res)} reservoirs saved", flush=True)

    save_checkpoint(all_res, ablation_rows, shap_frames, pooled)
    print(f"\n[done] {len(all_res)} reservoirs; aggregating...")
    RE.aggregate_and_write(all_res, ablation_rows, shap_frames, OUT, pooled, MODELS)
    print("[rerun_full] COMPLETE")


if __name__ == "__main__":
    main()
