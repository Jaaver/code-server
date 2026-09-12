"""Rank every alpha by the P&L it actually produces, not by rank IC.

Rank IC is invariant to the log/arithmetic choice, but P&L is not: a signal can
have a strongly positive IC and still lose money, because a handful of explosive
names dominate the arithmetic sum. xs_rev_72 does exactly that (IC +0.056,
arithmetic Sharpe -0.57). So we score alphas by the realised arithmetic return of
the portfolio they imply, at several holding horizons.

Holding for H bars is modelled as overlapping portfolios: the weight actually
held is the rolling mean of the last H targets, which is what a desk running H
staggered tranches would hold, and it cuts turnover roughly like 1/H.
"""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, run_wf

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
Y = 24 * 365
HORIZONS = [1, 4, 12, 24, 72, 168]


def signal_weights(sc, mask, top_frac=0.2):
    """Equal-weight long/short tails, gross 1, dollar-neutral."""
    q = sc.rank(axis=1, pct=True)
    long = (q >= 1 - top_frac) & mask
    short = (q <= top_frac) & mask
    nl = long.sum(axis=1).replace(0, np.nan)
    ns = short.sum(axis=1).replace(0, np.nan)
    W = (long.div(nl, axis=0) * 0.5).fillna(0) - (short.div(ns, axis=0) * 0.5).fillna(0)
    return W.where(nl.notna() & ns.notna() & (nl >= 3) & (ns >= 3), 0.0)


def hold(W, H, mask):
    """Overlapping portfolios: hold each target H bars, then re-mask and renormalise
    so we never carry weight in a name the universe filter has dropped."""
    if H > 1:
        W = W.rolling(H, min_periods=1).mean()
    W = W.where(mask, 0.0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    return W.div(g, axis=0).fillna(0.0)


def perf(W, r_ari):
    p = (W * r_ari).sum(axis=1)
    p = p[W.abs().sum(axis=1) > 0]
    if len(p) < 500 or p.std() == 0:
        return np.nan, np.nan, np.nan
    turn = W.diff().abs().sum(axis=1).mean() * Y      # gross traded per year
    return p.mean() * 1e4, p.mean() / p.std() * np.sqrt(Y), turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_dv", type=float, default=3e6)
    ap.add_argument("--screen_dv", type=float, default=2e6)
    ap.add_argument("--top_frac", type=float, default=0.2)
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()

    panels = run_wf.screen_universe(dataset.load(), floor_dv=args.screen_dv)
    mask = portfolio.tradable_mask(panels, min_dollar_vol=args.min_dv)
    cl = panels["close"]
    r_ari = (cl.shift(-1) / cl - 1.0)                  # arithmetic return over bar t+1
    print(f"tradable/bar {mask.sum(axis=1).mean():.1f}", flush=True)

    scores = strategy.alpha_panel(
        panels, alphas.ALPHAS,
        post=lambda v: alphas.zscore_xs(v.where(mask)).astype(np.float32))

    rows = []
    for name, sc in scores.items():
        W0 = signal_weights(sc, mask, args.top_frac)
        rec = {"alpha": name}
        for H in HORIZONS:
            bp, sr, tn = perf(hold(W0, H, mask), r_ari)
            rec[f"sr_h{H}"] = sr
            rec[f"bp_h{H}"] = bp
            rec[f"turn_h{H}"] = tn
        rows.append(rec)
        print(f"  {name:14s} " + "  ".join(f"h{H}:SR={rec[f'sr_h{H}']:+5.2f}/t={rec[f'turn_h{H}']:5.0f}"
                                           for H in HORIZONS), flush=True)

    df = pd.DataFrame(rows).set_index("alpha")
    df.to_csv(os.path.join(RES, f"alpha_arith_{args.tag}.csv"))
    print("\n=== ARITHMETIC SHARPE BY HOLDING HORIZON (gross of costs) ===")
    print(df[[f"sr_h{H}" for H in HORIZONS]].to_string(float_format=lambda x: f"{x:+.2f}"))
    print("\n=== ANNUAL TURNOVER (x gross) ===")
    print(df[[f"turn_h{H}" for H in HORIZONS]].to_string(float_format=lambda x: f"{x:.0f}"))

    # net-of-cost estimate: subtract a flat round-trip cost per unit traded
    print("\n=== EST. NET SHARPE at 10bp per unit traded ===")
    est = {}
    for H in HORIZONS:
        gross_bp = df[f"bp_h{H}"]                     # bp per bar
        cost_bp = df[f"turn_h{H}"] / Y * 10.0         # 10bp on each unit of gross traded
        vol_bp = gross_bp / df[f"sr_h{H}"].replace(0, np.nan) * np.sqrt(Y) / np.sqrt(Y)
        est[f"net_h{H}"] = (gross_bp - cost_bp) / vol_bp.abs() * np.sqrt(Y)
    e = pd.DataFrame(est)
    print(e.to_string(float_format=lambda x: f"{x:+.2f}"))
    e.to_csv(os.path.join(RES, f"alpha_net_{args.tag}.csv"))


if __name__ == "__main__":
    main()
