"""Where does the alpha actually live?

Two diagnostics that decide the whole design:
  1. IC decay: IC of each signal against returns h bars ahead, for a range of h.
     This says how fast the edge decays, hence how fast we must trade.
  2. A zero-cost, unsmoothed single-alpha backtest, to verify that sign and
     timing conventions survive the portfolio layer before costs enter.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, run_wf
from engine import Backtester, stats

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
HORIZONS = [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 168]


def ic_at(score, lp, mask, h):
    """IC of score_t against the log return from t to t+h."""
    fwd = (lp.shift(-h) - lp)
    return portfolio.information_coefficient(score, fwd, mask).mean()


def main(min_dv=3e6):
    panels = run_wf.screen_universe(dataset.load())
    mask = portfolio.tradable_mask(panels, min_dollar_vol=min_dv)
    print(f"avg tradable/bar: {mask.sum(axis=1).mean():.1f}", flush=True)
    lp = np.log(panels["close"])

    scores = strategy.alpha_panel(
        panels, alphas.ALPHAS,
        post=lambda v: alphas.zscore_xs(v.where(mask)).astype(np.float32))

    rows = {}
    for name, sc in scores.items():
        rows[name] = {f"h{h}": ic_at(sc, lp, mask, h) for h in HORIZONS}
        print(f"  {name:14s} " + " ".join(f"{v:+.4f}" for v in rows[name].values()), flush=True)
    df = pd.DataFrame(rows).T
    df.to_csv(os.path.join(RES, "ic_decay.csv"))
    print("\n=== IC DECAY (columns = forward horizon in hours) ===")
    print(df.to_string(float_format=lambda x: f"{x:+.4f}"))

    # cumulative IC tells us the best holding period: IC_h / sqrt(h) is the
    # per-unit-time information rate of holding for h bars
    rate = df.copy()
    for h in HORIZONS:
        rate[f"h{h}"] = df[f"h{h}"] / np.sqrt(h)
    print("\n=== INFORMATION RATE  IC_h / sqrt(h)  (best holding period per alpha) ===")
    print(rate.to_string(float_format=lambda x: f"{x:+.4f}"))
    rate.to_csv(os.path.join(RES, "ic_rate.csv"))
    best = rate.abs().idxmax(axis=1)
    print("\nbest holding horizon per alpha:")
    for k, v in best.items():
        print(f"  {k:14s} {v}")
    return df, scores, panels, mask


if __name__ == "__main__":
    main()
