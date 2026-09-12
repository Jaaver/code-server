"""Log-return IC vs arithmetic (tradable) P&L.

Every IC number so far was computed on LOG returns. A portfolio's realised P&L
is the weighted sum of ARITHMETIC returns. For a long/short book that shorts the
most volatile names, the two differ enormously: a coin that doubles costs a short
100%, but only 0.69 in log terms. Summing log returns quietly credits the short
book with money it never made.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, run_wf

panels = run_wf.screen_universe(dataset.load(), floor_dv=2e6)
mask = portfolio.tradable_mask(panels, min_dollar_vol=3e6)
cl = panels["close"]; lp = np.log(cl)
Y = 24 * 365

for name in ("low_vol", "xs_mom_168", "xs_rev_72", "illiq"):
    sc = alphas.zscore_xs(alphas.ALPHAS[name](panels).where(mask)).astype(np.float32)
    q = sc.rank(axis=1, pct=True)
    long = (q >= 0.9) & mask; short = (q <= 0.1) & mask
    nl = long.sum(axis=1).replace(0, np.nan); ns = short.sum(axis=1).replace(0, np.nan)
    W = (long.div(nl, axis=0) * 0.5).fillna(0) - (short.div(ns, axis=0) * 0.5).fillna(0)
    W = W.where(nl.notna() & ns.notna() & (nl >= 3) & (ns >= 3), 0.0)
    live = W.abs().sum(axis=1) > 0

    r_log = (lp.shift(-1) - lp)
    r_ari = (cl.shift(-1) / cl - 1.0)
    pl = (W * r_log).sum(axis=1)[live]
    pa = (W * r_ari).sum(axis=1)[live]
    # split the arithmetic P&L by side to show where it goes
    long_a = (W.clip(lower=0) * r_ari).sum(axis=1)[live]
    short_a = (W.clip(upper=0) * r_ari).sum(axis=1)[live]

    print(f"{name:11s} LOG  {pl.mean()*1e4:+7.3f}bp/bar SR={pl.mean()/pl.std()*np.sqrt(Y):+6.2f}  | "
          f"ARITH {pa.mean()*1e4:+7.3f}bp/bar SR={pa.mean()/pa.std()*np.sqrt(Y):+6.2f}  | "
          f"long {long_a.mean()*1e4:+6.2f}bp  short {short_a.mean()*1e4:+6.2f}bp", flush=True)
