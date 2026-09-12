"""Ablate portfolio construction one component at a time.

low_vol has a pure decile Sharpe above 3 but survives build_weights at ~0.2 even
with costs off, so one of the construction steps is destroying it. Turn them on
one at a time and watch where the signal goes.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, run_wf
from engine import Backtester, stats

VOL_FLOOR = portfolio.VOL_FLOOR


def weights(score, mask, panels, neutral, inv_vol, smooth, voltarget,
            top_frac=None, max_pos=0.08, target_vol=0.30, gross_cap=3.0):
    s = score.where(mask)
    if top_frac is not None:
        q = s.rank(axis=1, pct=True)
        s = pd.DataFrame(np.where(q >= 1 - top_frac, 1.0, np.where(q <= top_frac, -1.0, 0.0)),
                         index=s.index, columns=s.columns).where(s.notna())
    if neutral:
        s = s.sub(s.mean(axis=1), axis=0).where(s.notna())
    if inv_vol:
        v = np.log(panels["close"]).diff().rolling(24*14, min_periods=112).std()
        s = s / v.where(mask).clip(lower=VOL_FLOOR)
    gross = s.abs().sum(axis=1).replace(0, np.nan)
    w = s.div(gross, axis=0).fillna(0.0).clip(-max_pos, max_pos)
    if smooth > 1:
        w = w.ewm(span=smooth, min_periods=1).mean()
    if voltarget:
        ret = np.log(panels["close"]).diff()
        pr = (w.shift(1) * ret).sum(axis=1)
        rv = pr.rolling(24*30, min_periods=240).std() * np.sqrt(24*365)
        w = w.mul((target_vol / rv.replace(0, np.nan)).clip(upper=gross_cap).fillna(0.0), axis=0)
    else:
        w = w * gross_cap          # comparable gross so Sharpe is comparable
    g = w.abs().sum(axis=1); over = g > gross_cap
    w.loc[over] = w.loc[over].div(g[over], axis=0) * gross_cap
    return w.shift(1).fillna(0.0)


_BT = {}


def make_engines(panels, capital=1e6):
    """Build both engines once. Each Backtester holds ~1.4GB of float64 panels,
    so re-creating them per case OOMs the box."""
    a = {k: panels[k].to_numpy(dtype=np.float64) for k in
         ("open","high","low","close","quote_volume","funding")}
    zeros = np.zeros_like(a["close"])
    _BT["free"] = Backtester(a["open"], a["high"], a["low"], a["close"], a["quote_volume"],
                             zeros, capital=capital, max_gross_leverage=3.0,
                             fee=0.0, impact_coef=0.0, half_spread=zeros)
    _BT["cost"] = Backtester(a["open"], a["high"], a["low"], a["close"], a["quote_volume"],
                             a["funding"], capital=capital, max_gross_leverage=3.0,
                             half_spread=_BT["free"].hs * 0 + _abdi(a))
    return _BT


def _abdi(a):
    from engine import abdi_ranaldo_half_spread
    return abdi_ranaldo_half_spread(a["high"], a["low"], a["close"])


def run(panels, w, free=True, capital=1e6):
    return _BT["free" if free else "cost"].run(w.to_numpy(dtype=np.float64))


if __name__ == "__main__":
    panels = run_wf.screen_universe(dataset.load(), floor_dv=2e6)
    mask = portfolio.tradable_mask(panels, min_dollar_vol=3e6)
    print(f"tradable/bar {mask.sum(axis=1).mean():.1f}", flush=True)
    make_engines(panels)
    print("engines built", flush=True)

    cases = [
        ("raw score",                dict(neutral=False, inv_vol=False, smooth=1,  voltarget=False)),
        ("+neutral",                 dict(neutral=True,  inv_vol=False, smooth=1,  voltarget=False)),
        ("+neutral +invvol",         dict(neutral=True,  inv_vol=True,  smooth=1,  voltarget=False)),
        ("+neutral +smooth24",       dict(neutral=True,  inv_vol=False, smooth=24, voltarget=False)),
        ("+neutral +voltgt",         dict(neutral=True,  inv_vol=False, smooth=1,  voltarget=True)),
        ("ALL (v1/v2 config)",       dict(neutral=True,  inv_vol=True,  smooth=24, voltarget=True)),
        ("decile EW",                dict(neutral=True,  inv_vol=False, smooth=1,  voltarget=False, top_frac=0.1)),
        ("decile EW +smooth24",      dict(neutral=True,  inv_vol=False, smooth=24, voltarget=False, top_frac=0.1)),
    ]
    for name in ("low_vol", "xs_mom_168"):
        sc = alphas.zscore_xs(alphas.ALPHAS[name](panels).where(mask)).astype(np.float32)
        print(f"--- {name} ---", flush=True)
        for lbl, kw in cases:
            w = weights(sc, mask, panels, **kw)
            of = run(panels, w, free=True)
            oc = run(panels, w, free=False)
            sf, scst = stats(of["equity"]), stats(oc["equity"])
            print(f"  {lbl:22s} FREE SR={sf['sharpe']:+6.2f} CAGR={sf['cagr']:+8.1%} | "
                  f"NET SR={scst['sharpe']:+6.2f} CAGR={scst['cagr']:+8.1%} | "
                  f"turn={oc['turnover'].sum()/1e6/max(scst['years'],1e-9):5.1f}x  "
                  f"gross={np.nanmean(of['gross'][of['gross']>0]):.2f}", flush=True)
