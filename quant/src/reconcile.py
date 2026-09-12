"""Reconcile the engine against an analytic portfolio return for identical weights.

If sum_i w[t,i]*r[t,i] and the zero-cost engine equity disagree, the engine (or
the weight lag) is wrong. If they agree, the earlier standalone decile figure was
measuring something the portfolio cannot actually hold.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, run_wf
from engine import Backtester, stats

panels = run_wf.screen_universe(dataset.load(), floor_dv=2e6)
mask = portfolio.tradable_mask(panels, min_dollar_vol=3e6)
cl = panels["close"]
lp = np.log(cl)
sc = alphas.zscore_xs(alphas.ALPHAS["low_vol"](panels).where(mask)).astype(np.float32)

# --- fixed, simple weights: equal-weight decile long/short, gross 1, no extras ---
q = sc.rank(axis=1, pct=True)
long = (q >= 0.9) & mask
short = (q <= 0.1) & mask
nl = long.sum(axis=1).replace(0, np.nan)
ns = short.sum(axis=1).replace(0, np.nan)
W = (long.div(nl, axis=0) * 0.5).fillna(0) - (short.div(ns, axis=0) * 0.5).fillna(0)
W = W.where(nl.notna() & ns.notna() & (nl >= 3) & (ns >= 3), 0.0)

# 1) analytic: weight decided at t, earns the close-to-close return over t+1
ret_cc = (lp.shift(-1) - lp)                     # return over bar t+1
an = (W * ret_cc).sum(axis=1)
an = an[W.abs().sum(axis=1) > 0]
print(f"analytic  close-to-close : mean/bar={an.mean()*1e4:+.3f}bp  "
      f"SR={an.mean()/an.std()*np.sqrt(24*365):+.2f}  n={len(an)}")

# 2) analytic but OPEN-TO-OPEN, which is what an order actually gets
lo = np.log(panels["open"])
ret_oo = (lo.shift(-2) - lo.shift(-1))           # enter open(t+1), exit open(t+2)
ao = (W * ret_oo).sum(axis=1)
ao = ao[W.abs().sum(axis=1) > 0]
print(f"analytic  open-to-open   : mean/bar={ao.mean()*1e4:+.3f}bp  "
      f"SR={ao.mean()/ao.std()*np.sqrt(24*365):+.2f}  n={len(ao)}")

# 3) engine, zero cost, same weights (engine holds target_w[t] during bar t)
a = {k: panels[k].to_numpy(dtype=np.float64) for k in
     ("open","high","low","close","quote_volume","funding")}
z = np.zeros_like(a["close"])
bt = Backtester(a["open"], a["high"], a["low"], a["close"], a["quote_volume"], z,
                capital=1e6, max_gross_leverage=3.0, fee=0.0, impact_coef=0.0,
                half_spread=z, max_participation=1e9)
out = bt.run(W.shift(1).fillna(0.0).to_numpy(dtype=np.float64))
r = np.diff(np.log(np.maximum(out["equity"], 1e-9)))
r = r[np.isfinite(r) & (r != 0)]
print(f"engine    zero cost      : mean/bar={r.mean()*1e4:+.3f}bp  "
      f"SR={r.mean()/r.std()*np.sqrt(24*365):+.2f}  n={len(r)}")
print(f"engine turnover = {out['turnover'].sum()/1e6/stats(out['equity'])['years']:.1f}x equity/yr")
print(f"avg names long={nl.mean():.1f} short={ns.mean():.1f}  avg gross={W.abs().sum(axis=1).mean():.2f}")
