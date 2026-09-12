"""Horizon-matched ensemble.

The v1 failure was a design error, not a bug: weights were set by the 1-bar IC,
which is dominated by fast reversal signals whose edge decays within hours. Those
demand hourly trading, and at ~10bp a side the cost exceeds the edge - the book
paid the spread to chase an effect that had already reverted.

v2 picks a holding horizon H, scores alphas by their IC *at that horizon*, and
holds each target for H bars (overlapping portfolios), so turnover falls roughly
like 1/H while the slow, high-IC signals keep their edge.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, validation, run_wf
from engine import Backtester, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
YEAR = 24 * 365


def horizon_ic(scores, lp, mask, H, halflife_days):
    """EW-mean IC of each alpha against the H-bar-ahead return, lagged so that
    the weight used at bar t cannot depend on any return realised after t."""
    fwd = lp.shift(-H) - lp
    ics = {}
    for name, sc in scores.items():
        ics[name] = portfolio.information_coefficient(sc, fwd, mask)
    ic = pd.DataFrame(ics)
    ew = ic.ewm(halflife=int(halflife_days * 24), min_periods=int(halflife_days * 12)).mean()
    # IC at t peeks H bars ahead, so a weight may only use ICs from t-H-1 back.
    return ew.shift(H + 1), ic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=24, help="holding horizon in bars")
    ap.add_argument("--min_dv", type=float, default=3e6)
    ap.add_argument("--ic_halflife", type=float, default=90)
    ap.add_argument("--shrink", type=float, default=0.3)
    ap.add_argument("--target_vol", type=float, default=0.30)
    ap.add_argument("--gross_cap", type=float, default=3.0)
    ap.add_argument("--max_pos", type=float, default=0.08)
    ap.add_argument("--capital", type=float, default=1e6)
    ap.add_argument("--maker_frac", type=float, default=0.0)
    ap.add_argument("--max_part", type=float, default=0.05)
    ap.add_argument("--screen_dv", type=float, default=2e6)
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    panels = run_wf.screen_universe(dataset.load(), floor_dv=args.screen_dv)
    idx = panels["close"].index
    mask = portfolio.tradable_mask(panels, min_dollar_vol=args.min_dv)
    print(f"panel {panels['close'].shape}  avg tradable/bar {mask.sum(axis=1).mean():.1f}", flush=True)

    scores = strategy.alpha_panel(
        panels, alphas.ALPHAS,
        post=lambda v: alphas.zscore_xs(v.where(mask)).astype(np.float32))

    lp = np.log(panels["close"])
    ic_lag, ic_raw = horizon_ic(scores, lp, mask, args.H, args.ic_halflife)
    comb, wts = strategy.combine(scores, ic_lag, shrink=args.shrink)

    # hold each target for H bars: turnover falls ~1/H, matching the decay rate
    w = portfolio.build_weights(comb, mask, panels, target_vol=args.target_vol,
                                gross_cap=args.gross_cap, smooth=args.H,
                                max_pos=args.max_pos)

    a = {k: panels[k].to_numpy(dtype=np.float64) for k in
         ("open", "high", "low", "close", "quote_volume", "funding")}
    bt = Backtester(a["open"], a["high"], a["low"], a["close"], a["quote_volume"],
                    a["funding"], capital=args.capital, max_gross_leverage=args.gross_cap,
                    maker_frac=args.maker_frac, max_participation=args.max_part)
    out = bt.run(w.to_numpy(dtype=np.float64))
    st = stats(out["equity"])
    m = validation.monthly_returns(out["equity"], idx)

    print(f"\n===== H={args.H}h  min_dv=${args.min_dv:,.0f}  tv={args.target_vol:.0%} =====")
    print(f"  CAGR {st['cagr']:+.1%}  monthly {st['monthly_equiv']:+.2%}  vol {st['ann_vol']:.1%}")
    print(f"  Sharpe {st['sharpe']:+.2f}  Sortino {st['sortino']:+.2f}  maxDD {st['max_dd']:.1%}")
    print(f"  months>0 {(m>0).mean():.0%} of {len(m)}   median {m.median():+.2%}")
    print(f"  ann turnover {out['turnover'].sum()/args.capital/st['years']:.1f}x   "
          f"costs {out['costs'].sum()/args.capital:.1%}   funding {out['funding'].sum()/args.capital:+.1%}")
    print(f"  avg gross {np.nanmean(out['gross'][out['gross']>0]):.2f}  liq={out['liquidated_at']}")

    pd.Series(out["equity"], index=idx).to_csv(os.path.join(RES, f"equity_{args.tag}.csv"))
    m.to_csv(os.path.join(RES, f"monthly_{args.tag}.csv"))
    json.dump({k: float(v) for k, v in st.items()},
              open(os.path.join(RES, f"stats_{args.tag}.json"), "w"), indent=1)
    return st


if __name__ == "__main__":
    main()
