"""Fast comparison of allocation rules, without the engine.

Gross P&L, turnover and an analytic net estimate for several ways of blending
the alpha books. The engine is the arbiter, but it takes minutes per run; this
narrows the field first.
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, run_wf
from v3 import alpha_book, no_trade_band

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
Y = 24 * 365


def evaluate(W, r_ari, cost_bp, label):
    """W[t] is decided at t; it earns bar t+1. Returns gross/net summary."""
    pr = (W.shift(1) * r_ari).sum(axis=1)
    live = W.abs().sum(axis=1) > 0
    pr = pr[live]
    if len(pr) < 500 or pr.std() == 0:
        return None
    turn_bar = W.diff().abs().sum(axis=1)[live].mean()
    gross_sr = pr.mean() / pr.std() * np.sqrt(Y)
    cost_bar = turn_bar * cost_bp / 1e4
    net_sr = (pr.mean() - cost_bar) / pr.std() * np.sqrt(Y)
    return dict(label=label, gross_sr=gross_sr, net_sr=net_sr,
                turn_yr=turn_bar * Y, bp_bar=pr.mean() * 1e4,
                cost_bar_bp=cost_bar * 1e4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=72)
    ap.add_argument("--min_dv", type=float, default=3e6)
    ap.add_argument("--screen_dv", type=float, default=2e6)
    ap.add_argument("--cost_bp", type=float, default=15.0)
    ap.add_argument("--max_pos", type=float, default=0.10)
    args = ap.parse_args()

    panels = run_wf.screen_universe(dataset.load(), floor_dv=args.screen_dv)
    mask = portfolio.tradable_mask(panels, min_dollar_vol=args.min_dv)
    cl = panels["close"]
    r_ari = (cl / cl.shift(1) - 1.0)
    print(f"tradable/bar {mask.sum(axis=1).mean():.1f}", flush=True)

    scores = strategy.alpha_panel(
        panels, alphas.ALPHAS,
        post=lambda v: alphas.zscore_xs(v.where(mask)).astype(np.float32))
    names = list(scores)

    # pass 1: per-alpha return series (books are rebuilt later, not stored)
    rets = {}
    for k in names:
        w = alpha_book(scores[k], mask, args.H, args.max_pos)
        rets[k] = (w.shift(1) * r_ari).sum(axis=1)
        del w
    R = pd.DataFrame(rets)

    lb = 180 * 24
    mu = R.rolling(lb, min_periods=lb // 3).mean()
    sd = R.rolling(lb, min_periods=lb // 3).std().replace(0, np.nan)
    tr_sr = (mu / sd).shift(1)                      # lagged trailing Sharpe

    rules = {}
    rules["EW"] = pd.DataFrame(1.0, index=R.index, columns=names)
    rules["sign(trailing)"] = np.sign(tr_sr).fillna(0.0)
    rules["trailing SR"] = tr_sr.fillna(0.0)

    out = []
    for rule_name, A in rules.items():
        An = A.div(A.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        W = np.zeros(cl.shape, dtype=np.float32)
        for k in names:
            w = alpha_book(scores[k], mask, args.H, args.max_pos)
            np.add(W, w.to_numpy(dtype=np.float32) * An[k].to_numpy(dtype=np.float32)[:, None], out=W)
            del w
        Wd = pd.DataFrame(W, index=cl.index, columns=cl.columns).where(mask, 0.0)
        g = Wd.abs().sum(axis=1).replace(0, np.nan)
        Wd = Wd.div(g, axis=0).fillna(0.0)
        for every in (1, 8, 24, 72, 168):
            Ws = portfolio.rebalance_schedule(Wd, every)
            for band in (0.0, 0.005, 0.015):
                Wb = no_trade_band(Ws, band) if band > 0 else Ws
                r = evaluate(Wb, r_ari, args.cost_bp, f"{rule_name}|rebal={every}h|band={band}")
                if r:
                    out.append(r)
                    print(f"  {r['label']:40s} gross SR={r['gross_sr']:+5.2f} "
                          f"net SR={r['net_sr']:+5.2f} turn={r['turn_yr']:6.0f}x "
                          f"gross={r['bp_bar']:+.3f}bp cost={r['cost_bar_bp']:.3f}bp", flush=True)
    df = pd.DataFrame(out).sort_values("net_sr", ascending=False)
    df.to_csv(os.path.join(RES, "combo_rules.csv"), index=False)
    print("\n=== BEST ===")
    print(df.head(10).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
