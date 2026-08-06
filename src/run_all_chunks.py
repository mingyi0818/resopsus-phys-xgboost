"""
Resumable driver for the N=200 (default) tree-model experiment.

Runs run_experiment_v3.main() one chunk at a time in a SINGLE Python process
(SKIP_LSTM=1 -> tree/gradient-boosting models only; LSTM is produced by
run_lstm_only.py and merged separately). Each chunk writes a unique
per_reservoir_metrics_<rs>_<rc>.csv; already-completed chunks are skipped, so
the driver is safe to relaunch after an interruption -- no wasted work.

Run:
    python run_all_chunks.py            # N=200, chunk size 30
    N_RES=60 python run_all_chunks.py   # override reservoir count
"""
import os
# Keep torch OUT of this process: importing torch alongside xgboost/lightgbm/
# catboost triggers an OpenMP (libiomp5md) segfault on Windows. This module only
# runs the tree/gradient-boosting comparison, so LSTM is skipped here.
os.environ["SKIP_LSTM"] = "1"
import glob
import time
import json
import numpy as np
import pandas as pd

import run_experiment_v3 as R

EXTRACT_DIR = R.EXTRACT_DIR
OUT_DIR = R.OUT_DIR
PROGRESS = os.path.join(OUT_DIR, "chunks_progress.json")


def select(N):
    """Identical selection logic to run_experiment_v3.main() so the N_RES subset
    is byte-for-byte the same across invocations (deterministic)."""
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


def main():
    N = int(os.environ.get("N_RES", "200"))
    CHUNK = 30
    sel = select(N)
    os.environ["N_RES"] = str(N)
    os.environ["SKIP_LSTM"] = "1"
    done_log = []
    if os.path.exists(PROGRESS):
        try:
            done_log = json.load(open(PROGRESS))
        except Exception:
            done_log = []
    for rs in range(0, len(sel), CHUNK):
        rc = min(CHUNK, len(sel) - rs)
        out = os.path.join(OUT_DIR, f"per_reservoir_metrics_{rs}_{rc}.csv")
        if os.path.exists(out):
            print(f"[driver] skip rs={rs} rc={rc} (exists)", flush=True)
            continue
        t0 = time.perf_counter()
        print(f"[driver] === chunk start rs={rs} rc={rc} ===", flush=True)
        os.environ["RES_START"] = str(rs)
        os.environ["RES_COUNT"] = str(rc)
        try:
            R.main()
        except Exception as e:
            print(f"[driver] chunk rs={rs} FAILED: {e}", flush=True)
            continue
        dt = time.perf_counter() - t0
        print(f"[driver] === chunk end rs={rs} rc={rc} ({dt:.1f}s) ===", flush=True)
        done_log.append({"rs": rs, "rc": rc, "t": time.time()})
        json.dump(done_log, open(PROGRESS, "w"))
    print("[driver] ALL CHUNKS DONE", flush=True)


if __name__ == "__main__":
    main()
