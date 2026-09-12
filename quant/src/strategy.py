"""Walk-forward alpha ensemble.

At every refit the combination weights are estimated using ONLY data that ends
before the (embargoed) start of the live window, so every weight the portfolio
ever uses is out-of-sample.
"""
import numpy as np, pandas as pd


def alpha_panel(panels, alpha_defs, post=None):
    """Compute every alpha once; returns {name: DataFrame}.

    `post` is applied immediately so each alpha can be cast down to float32 and
    its float64 rolling intermediates collected before the next one is built.
    """
    import gc, time
    out = {}
    for i, (name, fn) in enumerate(alpha_defs.items(), 1):
        t0 = time.time()
        try:
            v = fn(panels)
            out[name] = post(v) if post is not None else v
            del v
            gc.collect()
            print(f"  [{i}/{len(alpha_defs)}] {name} {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(alpha_defs)}] {name} FAILED: {e}", flush=True)
    return out


def rolling_ic(scores, fwd, mask, halflife_days=60):
    """Exponentially-weighted mean IC per alpha, computed causally.

    Returns a DataFrame (time x alpha) where row t is the EW-mean IC using
    information strictly up to t.
    """
    from portfolio import information_coefficient
    import time
    ics = {}
    for i, (name, sc) in enumerate(scores.items(), 1):
        t0 = time.time()
        ics[name] = information_coefficient(sc, fwd, mask)
        print(f"  IC [{i}/{len(scores)}] {name} {time.time()-t0:.1f}s", flush=True)
    ic = pd.DataFrame(ics)
    hl = int(halflife_days * 24)
    return ic.ewm(halflife=hl, min_periods=hl // 2).mean(), ic


def combine(scores, ic_ew, shrink=0.25, min_abs_ic=0.0):
    """IC-weighted blend of the standardised alphas.

    ic_ew must be lagged before it reaches here. Weights are shrunk toward
    equal-weight, which is what keeps the blend from chasing the last regime.
    """
    names = [n for n in scores if n in ic_ew.columns]
    W = np.nan_to_num(ic_ew[names].to_numpy(dtype=np.float32))             # (T,K)
    if min_abs_ic > 0:
        W = np.where(np.abs(W) < min_abs_ic, 0.0, W)
    # shrink toward the equal-weight (sign-aware) prior
    eq = np.sign(W)
    denom = np.abs(W).sum(axis=1, keepdims=True)
    Wn = np.divide(W, denom, out=np.zeros_like(W), where=denom > 0)
    eqd = np.abs(eq).sum(axis=1, keepdims=True)
    eqn = np.divide(eq, eqd, out=np.zeros_like(eq), where=eqd > 0)
    Wf = ((1 - shrink) * Wn + shrink * eqn).astype(np.float32)             # (T,K)

    # Accumulate the blend one alpha at a time. Stacking all K score panels
    # first would double peak memory (a second full copy of every alpha).
    idx = scores[names[0]].index; cols = scores[names[0]].columns
    comb = np.zeros((len(idx), len(cols)), dtype=np.float32)
    for k, n in enumerate(names):
        a = scores[n].to_numpy(dtype=np.float32)
        np.add(comb, np.nan_to_num(a) * Wf[:, k][:, None], out=comb)
    return (pd.DataFrame(comb, index=idx, columns=cols),
            pd.DataFrame(Wf, index=idx, columns=names))
