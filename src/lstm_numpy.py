"""
lstm_numpy.py -- a self-contained, torch-free LSTM implementation in NumPy.

Motivation: on this Windows host (Python 3.13 + Xeon W7-2595X) the PyTorch C
extension (torch._C) segfaults on load in any non-trivial process, so the
original torch-based LSTM (lstm_core.train_lstm) cannot run. This module
provides a faithful recurrent-LSTM baseline that depends only on NumPy, so the
paper's deep-learning comparison (vs the physics-augmented tree models, and in
the context of recurrent SOTA such as Tran et al. 2025 and Zhang et al. 2025)
remains reproducible.

Interface deliberately mirrors lstm_core:
  * build_lstm_windows(df, use_phys) -> (X, y, valid) with X (n, win, n_feats)
  * train_lstm(X_tr, y_tr, X_te, y_te, n_feats, ...) -> predictions (n_te,)
    in the ORIGINAL (un-standardised) target scale, exactly like the torch
    version, so run_lstm_numpy.py can reuse the same downstream merge logic.

A numeric gradient check (grad_check) is included so correctness can be
verified before any large run.
"""
import numpy as np

LSTM_WIN = 30
MAX_LSTM_SAMPLES = 3000  # cap training windows/reservoir to bound CPU time

LSTM_RAW_BASE = ["inflow", "storage_lag1", "doy_sin", "doy_cos"]
LSTM_PHYS_BASE = ["storage_trend", "throughput_ratio", "storage_norm",
                  "storage_deficit", "inflow_norm", "inflow_ma7", "inflow_anom7"]


def build_lstm_windows(df, use_phys, win=LSTM_WIN):
    """Return (X, y, valid) where X is (n_samples, win, n_feats), y (n_samples,)."""
    feats = LSTM_RAW_BASE + (LSTM_PHYS_BASE if use_phys else [])
    d = df.copy()
    cols = []
    for f in feats:
        for k in range(0, win):
            cols.append(d[f].shift(k).values)
    M = np.column_stack(cols)
    y = d["outflow"].values
    valid = np.isfinite(M).all(axis=1) & np.isfinite(y)
    n_feats = len(feats)
    X = M[valid].reshape(-1, n_feats, win).transpose(0, 2, 1)
    yv = y[valid]
    return X.astype(np.float64), yv.astype(np.float64), valid


# ----------------------------------------------------------------------------
# NumPy LSTM (1 layer, minibatched, Adam, BPTT)
# ----------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _init_params(n_feats, hidden, rng):
    lim_i = np.sqrt(1.0 / n_feats)
    lim_r = np.sqrt(1.0 / hidden)
    p = {}
    for g in ("i", "f", "g", "o"):
        p["W" + g] = rng.uniform(-lim_i, lim_i, (hidden, n_feats))
        p["U" + g] = rng.uniform(-lim_r, lim_r, (hidden, hidden))
        p["b" + g] = np.zeros(hidden)
    p["Wout"] = rng.uniform(-np.sqrt(1.0 / hidden), np.sqrt(1.0 / hidden), (hidden,))
    p["bout"] = np.zeros(())
    return p


def _forward(X, p):
    """X (N,T,F) -> pred (N,); cache holds per-timestep tensors + last hidden."""
    N, T, F = X.shape
    h = np.zeros((N, p["Wi"].shape[0]))
    c = np.zeros((N, p["Wi"].shape[0]))
    cache = {"x": [], "hprev": [], "cprev": [], "i": [], "f": [], "g": [],
             "o": [], "c": [], "tanhc": []}
    for t in range(T):
        x = X[:, t, :]
        h_prev_state = h.copy()          # h_{t-1} (state entering this step)
        c_prev_state = c.copy()          # c_{t-1}
        ai = x @ p["Wi"].T + h @ p["Ui"].T + p["bi"]
        af = x @ p["Wf"].T + h @ p["Uf"].T + p["bf"]
        ag = x @ p["Wg"].T + h @ p["Ug"].T + p["bg"]
        ao = x @ p["Wo"].T + h @ p["Uo"].T + p["bo"]
        i = _sigmoid(ai); f = _sigmoid(af); g = np.tanh(ag); o = _sigmoid(ao)
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        cache["x"].append(x)
        cache["hprev"].append(h_prev_state)   # h_{t-1}
        cache["cprev"].append(c_prev_state)   # c_{t-1}
        cache["i"].append(i); cache["f"].append(f); cache["g"].append(g)
        cache["o"].append(o); cache["c"].append(c_new); cache["tanhc"].append(np.tanh(c_new))
        h, c = h_new, c_new
    pred = h @ p["Wout"] + p["bout"]
    cache["h_final"] = h
    return pred, cache


def _backward(X, y, pred, p, cache):
    N = X.shape[0]
    dLdpred = (pred - y) / N
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    h_last = cache["h_final"]
    grads["Wout"] = h_last.T @ dLdpred
    grads["bout"] = np.sum(dLdpred)
    dh = dLdpred[:, None] * p["Wout"]
    dc = np.zeros_like(dh)
    T = len(cache["i"])
    for t in range(T - 1, -1, -1):
        i = cache["i"][t]; f = cache["f"][t]; g = cache["g"][t]
        o = cache["o"][t]; c = cache["c"][t]; tanhc = cache["tanhc"][t]
        h_prev = cache["hprev"][t]; x = cache["x"][t]
        c_prev = cache["cprev"][t]
        do = dh * tanhc
        dc_from_h = dh * o * (1.0 - tanhc ** 2)
        dc_total = dc + dc_from_h
        di = dc_total * g
        dg = dc_total * i
        df = dc_total * c_prev
        dc_prev = dc_total * f
        dai = di * i * (1.0 - i)
        daf = df * f * (1.0 - f)
        dag = dg * (1.0 - g ** 2)
        dao = do * o * (1.0 - o)
        grads["Wi"] += dai.T @ x
        grads["Ui"] += dai.T @ h_prev
        grads["bi"] += np.sum(dai, axis=0)
        grads["Wf"] += daf.T @ x
        grads["Uf"] += daf.T @ h_prev
        grads["bf"] += np.sum(daf, axis=0)
        grads["Wg"] += dag.T @ x
        grads["Ug"] += dag.T @ h_prev
        grads["bg"] += np.sum(dag, axis=0)
        grads["Wo"] += dao.T @ x
        grads["Uo"] += dao.T @ h_prev
        grads["bo"] += np.sum(dao, axis=0)
        dh = dai @ p["Ui"] + daf @ p["Uf"] + dag @ p["Ug"] + dao @ p["Uo"]
        dc = dc_prev
    return grads


def _clip_grads(grads, clip=1.0):
    norm = np.sqrt(sum(np.sum(v ** 2) for v in grads.values()))
    if norm > clip:
        for k in grads:
            grads[k] = grads[k] * (clip / norm)
    return grads


def _adam_step(p, grads, m, v, t, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    for k in p:
        m[k] = b1 * m[k] + (1 - b1) * grads[k]
        v[k] = b2 * v[k] + (1 - b2) * (grads[k] ** 2)
        mhat = m[k] / (1 - b1 ** t)
        vhat = v[k] / (1 - b2 ** t)
        p[k] = p[k] - lr * mhat / (np.sqrt(vhat) + eps)


def train_lstm(X_tr, y_tr, X_te, y_te, n_feats, epochs=120, batch=256, lr=0.005,
               hidden=32, rng=42, verbose=False):
    rng = np.random.RandomState(rng)
    if len(X_tr) > MAX_LSTM_SAMPLES:
        idx = rng.choice(len(X_tr), MAX_LSTM_SAMPLES, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    ym = float(np.mean(y_tr)); ys = float(np.std(y_tr)) + 1e-9
    yt = (y_tr - ym) / ys
    mu = np.mean(X_tr, axis=(0, 1)); sd = np.std(X_tr, axis=(0, 1)) + 1e-9
    Xtr = np.nan_to_num((np.nan_to_num(X_tr, nan=0.0) - mu) / sd)
    Xte = np.nan_to_num((np.nan_to_num(X_te, nan=0.0) - mu) / sd)

    p = _init_params(n_feats, hidden, rng)
    # initialise best-state to the random init so a fully-diverged run still
    # returns finite predictions instead of NaN/Inf
    bstate = {k: val.copy() for k, val in p.items()}
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    n_val = max(1, int(len(Xtr) * 0.1))
    perm = rng.permutation(len(Xtr))
    Xv, yv = Xtr[perm[:n_val]], yt[perm[:n_val]]
    Xtr, ytr = Xtr[perm[n_val:]], yt[perm[n_val:]]
    best, wait = np.inf, 0
    t = 0
    for ep in range(epochs):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), batch):
            bidx = order[s:s + batch]
            pred, cache = _forward(Xtr[bidx], p)
            grads = _backward(Xtr[bidx], ytr[bidx], pred, p, cache)
            grads = _clip_grads(grads, 1.0)
            t += 1
            _adam_step(p, grads, m, v, t, lr=lr)
        vp, _ = _forward(Xv, p)
        vl = float(np.mean((vp - yv) ** 2))
        if not np.isfinite(vl):
            # divergence: stop and fall back to best-so-far
            break
        if vl < best:
            best = vl
            bstate = {k: val.copy() for k, val in p.items()}
            wait = 0
        else:
            wait += 1
            if wait >= 15:
                break
    p = bstate
    pred_te, _ = _forward(Xte, p)
    return pred_te * ys + ym


def grad_check(n_feats, hidden=8, rng=0, eps=1e-6, tol=1e-4):
    """Finite-difference gradient check for the LSTM backward pass."""
    rng = np.random.RandomState(rng)
    p = _init_params(n_feats, hidden, rng)
    N = 5
    Xs = np.random.RandomState(1).randn(N, LSTM_WIN, n_feats)
    ys = np.random.RandomState(2).randn(N)
    pred, cache = _forward(Xs, p)
    grads = _backward(Xs, ys, pred, p, cache)
    ok = True
    for k in p:
        gnum = np.zeros_like(p[k])
        it = np.nditer(p[k], flags=["multi_index"])
        while not it.finished:
            ix = it.multi_index
            old = p[k][ix]
            p[k][ix] = old + eps
            fp, _ = _forward(Xs, p)
            p[k][ix] = old - eps
            fm, _ = _forward(Xs, p)
            p[k][ix] = old
            gnum[ix] = (0.5 * np.mean((fp - ys) ** 2) - 0.5 * np.mean((fm - ys) ** 2)) / (2 * eps)
            it.iternext()
        rel = np.max(np.abs(gnum - grads[k]) / (np.abs(gnum) + np.abs(grads[k]) + 1e-12))
        if rel > tol:
            ok = False
            print(f"  grad_check FAIL {k}: max rel err = {rel:.2e}")
        else:
            print(f"  grad_check OK   {k}: max rel err = {rel:.2e}")
    return ok


if __name__ == "__main__":
    print("Running LSTM gradient check (numpy)...")
    grad_check(4)
