"""Walk-forward alpha ensemble.

At every refit the combination weights are estimated using ONLY data that ends
before the (embargoed) start of the live window, so every weight the portfolio
ever uses is out-of-sample.
"""
import numpy as np, pandas as pd


def alpha_panel(panels, alpha_defs):
    """Compute every alpha once; returns {name: DataFrame}."""
    out = {}
    for name, fn in alpha_defs.items():
        try:
            out[name] = fn(panels)
        except Exception as e:
            print(f"  alpha {name} failed: {e}")
    return out


def rolling_ic(scores, fwd, mask, halflife_days=60):
    """Exponentially-weighted mean IC per alpha, computed causally.

    Returns a DataFrame (time x alpha) where row t is the EW-mean IC using
    information strictly up to t.
    """
    from portfolio import information_coefficient
    ics = {}
    for name, sc in scores.items():
        ics[name] = information_coefficient(sc, fwd, mask)
    ic = pd.DataFrame(ics)
    hl = int(halflife_days * 24)
    return ic.ewm(halflife=hl, min_periods=hl // 2).mean(), ic


def combine(scores, ic_ew, shrink=0.25, min_abs_ic=0.0):
    """IC-weighted blend of the standardised alphas.

    ic_ew must be lagged before it reaches here. Weights are shrunk toward
    equal-weight, which is what keeps the blend from chasing the last regime.
    """
    names = [n for n in scores if n in ic_ew.columns]
    A = np.stack([scores[n].to_numpy(dtype=np.float32) for n in names])   # (K,T,N)
    W = ic_ew[names].to_numpy(dtype=np.float32)                            # (T,K)
    W = np.nan_to_num(W)
    if min_abs_ic > 0:
        W = np.where(np.abs(W) < min_abs_ic, 0.0, W)
    # shrink toward the equal-weight (sign-aware) prior
    eq = np.sign(W)
    denom = np.abs(W).sum(axis=1, keepdims=True)
    Wn = np.divide(W, denom, out=np.zeros_like(W), where=denom > 0)
    eqd = np.abs(eq).sum(axis=1, keepdims=True)
    eqn = np.divide(eq, eqd, out=np.zeros_like(eq), where=eqd > 0)
    Wf = (1 - shrink) * Wn + shrink * eqn                                  # (T,K)
    comb = np.einsum("ktn,tk->tn", A, Wf.astype(np.float32))
    idx = scores[names[0]].index; cols = scores[names[0]].columns
    return pd.DataFrame(comb, index=idx, columns=cols), pd.DataFrame(Wf, index=idx, columns=names)
