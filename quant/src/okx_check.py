"""Cross-exchange check: do the same alphas pay on OKX?

Different venue, matching engine, fee schedule, listing set and flow. Measured
in arithmetic P&L (the log-return version of this comparison is misleading, as
established earlier). OKX candles carry no taker-side split, so order-flow
alphas are excluded rather than faked.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alphas, portfolio, dataset, okx_dataset, run_wf
from v3 import alpha_book

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
Y = 24 * 365
SKIP = {"ofi_6", "ofi_24"}          # need Binance's taker split
H = 24


def book_sharpe(panels, mask, name, H):
    sc = alphas.zscore_xs(alphas.ALPHAS[name](panels).where(mask)).astype(np.float32)
    w = alpha_book(sc, mask, H, 0.10)
    cl = panels["close"]
    r = (cl / cl.shift(1) - 1.0)
    pr = (w.shift(1) * r).sum(axis=1)
    pr = pr[w.abs().sum(axis=1) > 0]
    if len(pr) < 300 or pr.std() == 0:
        return np.nan
    return pr.mean() / pr.std() * np.sqrt(Y)


if __name__ == "__main__":
    okx = okx_dataset.load()
    oidx = okx["close"].index
    print(f"OKX {okx['close'].shape}  {oidx.min().date()} -> {oidx.max().date()}", flush=True)
    omask = portfolio.tradable_mask(okx, min_dollar_vol=1e6)
    print(f"OKX tradable/bar {omask.sum(axis=1).mean():.1f}", flush=True)

    bnc = dataset.load()
    bnc = {k: v.loc[(v.index >= oidx.min()) & (v.index <= oidx.max())] for k, v in bnc.items()}
    bnc = run_wf.screen_universe(bnc, floor_dv=2e6)
    bmask = portfolio.tradable_mask(bnc, min_dollar_vol=3e6)
    print(f"Binance (same window) tradable/bar {bmask.sum(axis=1).mean():.1f}", flush=True)

    rows = []
    for name in alphas.ALPHAS:
        if name in SKIP:
            continue
        b = book_sharpe(bnc, bmask, name, H)
        o = book_sharpe(okx, omask, name, H)
        if np.isfinite(b) and np.isfinite(o):
            rows.append(dict(alpha=name, binance_sr=b, okx_sr=o,
                             same_sign=np.sign(b) == np.sign(o)))
            print(f"  {name:14s} Binance SR={b:+5.2f}   OKX SR={o:+5.2f}   "
                  f"{'agree' if np.sign(b)==np.sign(o) else 'DISAGREE'}", flush=True)
    df = pd.DataFrame(rows).set_index("alpha")
    df.to_csv(os.path.join(RES, "cross_exchange.csv"))
    print(f"\nsign agreement : {df.same_sign.mean():.0%} of {len(df)} alphas")
    print(f"rank corr      : {df.binance_sr.corr(df.okx_sr, method='spearman'):+.2f}")
    print(f"pearson corr   : {df.binance_sr.corr(df.okx_sr):+.2f}")
