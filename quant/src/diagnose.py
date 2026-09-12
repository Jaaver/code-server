"""Decisive test: does a single high-IC alpha make money with costs switched off?

If yes, the problem is cost/turnover design. If no, the problem is in the
portfolio construction or the timing convention, and no amount of signal work
will fix it.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, run_wf
from engine import Backtester, stats

def bt_run(panels, w, capital=1e6, gross=3.0, free=False):
    a = {k: panels[k].to_numpy(dtype=np.float64) for k in
         ("open","high","low","close","quote_volume","funding")}
    kw = {}
    if free:
        kw = dict(fee=0.0, impact_coef=0.0,
                  half_spread=np.zeros_like(a["close"]))
    b = Backtester(a["open"], a["high"], a["low"], a["close"], a["quote_volume"],
                   np.zeros_like(a["close"]) if free else a["funding"],
                   capital=capital, max_gross_leverage=gross, **kw)
    return b.run(w.to_numpy(dtype=np.float64))

def main():
    panels = run_wf.screen_universe(dataset.load(), floor_dv=2e6)
    mask = portfolio.tradable_mask(panels, min_dollar_vol=3e6)
    idx = panels["close"].index
    lp = np.log(panels["close"])
    print(f"tradable/bar {mask.sum(axis=1).mean():.1f}", flush=True)

    for name in ["low_vol", "xs_rev_72", "xs_mom_168", "illiq", "xs_rev_6"]:
        sc = alphas.zscore_xs(alphas.ALPHAS[name](panels).where(mask)).astype(np.float32)
        ic1 = portfolio.information_coefficient(sc, lp.shift(-1)-lp, mask).mean()
        ic24 = portfolio.information_coefficient(sc, lp.shift(-24)-lp, mask).mean()

        # --- A. pure signal return, no portfolio machinery at all ---
        # equal-weight long top decile / short bottom decile, held 1 bar,
        # measured on close-to-close returns. This is the textbook IC-to-PnL link.
        r = lp.diff().shift(-1)                      # return over the NEXT bar
        q = sc.rank(axis=1, pct=True)
        long = (q >= 0.9); short = (q <= 0.1)
        nl = long.sum(axis=1).replace(0, np.nan); ns = short.sum(axis=1).replace(0, np.nan)
        pure = (r.where(long).sum(axis=1)/nl - r.where(short).sum(axis=1)/ns).dropna()
        sr_pure = pure.mean()/pure.std()*np.sqrt(24*365)

        for smooth, lbl in ((1, "smooth1"), (24, "smooth24")):
            w = portfolio.build_weights(sc, mask, panels, target_vol=0.30,
                                        gross_cap=3.0, smooth=smooth, max_pos=0.08)
            of = bt_run(panels, w, free=True)
            oc = bt_run(panels, w, free=False)
            sf, sc_ = stats(of["equity"]), stats(oc["equity"])
            print(f"{name:11s} IC1={ic1:+.4f} IC24={ic24:+.4f} pureSR={sr_pure:+5.2f} | "
                  f"{lbl:9s} FREE SR={sf['sharpe']:+6.2f} CAGR={sf['cagr']:+8.1%} | "
                  f"COST SR={sc_['sharpe']:+6.2f} CAGR={sc_['cagr']:+8.1%} "
                  f"turn={oc['turnover'].sum()/1e6/sc_['years']:.0f}x", flush=True)

if __name__ == "__main__":
    main()
