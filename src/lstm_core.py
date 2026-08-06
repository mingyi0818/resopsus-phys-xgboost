"""
lstm_core.py -- self-contained LSTM building blocks for the ResOpsUS physics-feature
study. Kept in a SEPARATE module that imports torch at the top so it can run in a
torch-only process, isolated from xgboost/lightgbm/catboost (loading those together
with torch in the same process triggers an OpenMP-runtime segfault on Windows).

run_lstm_only.py imports this module directly. run_experiment_v3.py imports it only
inside its `else:  # not SKIP_LSTM` branch (which is never taken in the trees-only run).
"""

import numpy as np
import torch
import torch.nn as nn

# LSTM window length (days)
LSTM_WIN = 30
MAX_LSTM_SAMPLES = 20000  # cap training windows per reservoir to bound CPU time

# Feature bases (must match run_experiment_v3.RAW_FEATS / PHYS_FEATS column names)
LSTM_RAW_BASE = ["inflow", "storage_lag1", "doy_sin", "doy_cos"]
LSTM_PHYS_BASE = ["storage_trend", "throughput_ratio", "storage_norm",
                  "storage_deficit", "inflow_norm", "inflow_ma7", "inflow_anom7"]


def build_lstm_windows(df, use_phys, win=LSTM_WIN):
    """Return (X, y) where X is (n_samples, win, n_feats), y is (n_samples,)."""
    feats = LSTM_RAW_BASE + (LSTM_PHYS_BASE if use_phys else [])
    d = df.copy()
    req = feats
    cols = []
    for f in req:
        for k in range(0, win):
            cols.append(d[f].shift(k).values)
    M = np.column_stack(cols)  # (T, win*len(req))  [feature-major]
    y = d["outflow"].values
    valid = np.isfinite(M).all(axis=1) & np.isfinite(y)
    n_feats = len(req)
    X = M[valid].reshape(-1, n_feats, win).transpose(0, 2, 1)  # (n, win, n_feats)
    yv = y[valid]
    return X.astype(np.float32), yv.astype(np.float32), valid


class LSTMModel(nn.Module):
    def __init__(self, n_feats, hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feats, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm(X_tr, y_tr, X_te, y_te, n_feats, epochs=50, batch=256, lr=0.001,
               rng=42):
    # cap training windows per reservoir for compute (documented preprocessing choice)
    if len(X_tr) > MAX_LSTM_SAMPLES:
        rs = np.random.RandomState(rng)
        idx = rs.choice(len(X_tr), MAX_LSTM_SAMPLES, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    # per-reservoir-style standardization of the TARGET using training stats
    ym = float(np.nanmean(y_tr)); ys = float(np.nanstd(y_tr)) + 1e-9
    yt = (y_tr - ym) / ys
    # standardize features globally (z-score) for stability
    mu = np.nanmean(X_tr, axis=(0, 1)); sd = np.nanstd(X_tr, axis=(0, 1)) + 1e-9
    Xtr = (np.nan_to_num(X_tr, nan=0.0) - mu) / sd
    Xte = (np.nan_to_num(X_te, nan=0.0) - mu) / sd
    Xtr = np.nan_to_num(Xtr, nan=0.0); Xte = np.nan_to_num(Xte, nan=0.0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = LSTMModel(n_feats).to(dev)
    print(f"[train_lstm] device={dev.type} "
          f"gpu={torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'n/a'}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=8, factor=0.5)
    crit = nn.MSELoss()
    xtr = torch.FloatTensor(Xtr).to(dev); ytr = torch.FloatTensor(yt).to(dev)
    xte = torch.FloatTensor(Xte).to(dev)
    n_val = int(len(xtr) * 0.1)
    perm = np.random.permutation(len(xtr))
    xv, yv = xtr[perm[:n_val]], ytr[perm[:n_val]]
    xtr, ytr = xtr[perm[n_val:]], ytr[perm[n_val:]]
    best, wait, bstate = np.inf, 0, None
    for ep in range(epochs):
        model.train()
        for b in range(0, len(xtr), batch):
            xb = xtr[b:b+batch]; yb = ytr[b:b+batch]
            opt.zero_grad(); loss = crit(model(xb).squeeze(-1), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(xv).squeeze(-1), yv).item()
        sch.step(vl)
        if vl < best:
            best = vl; bstate = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= 10: break
    if bstate is not None:
        model.load_state_dict(bstate)
    model.eval()
    with torch.no_grad():
        pred = model(xte).squeeze(-1).cpu().numpy()
    pred = pred * ys + ym
    return pred
