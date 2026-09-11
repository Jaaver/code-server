"""Main walk-forward experiment: build the ensemble, trade it, and measure it honestly."""
import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, validation
from engine import Backtester, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
YEAR = 24 * 365


def build(panels, args):
    mask = portfolio.tradable_mask(panels, min_dollar_vol=args.min_dv)
    fwd = np.log(panels["close"]).diff().shift(-1)
    scores = strategy.alpha_panel(panels, alphas.ALPHAS)
    # standardise every alpha cross-sectionally so the blend is scale-free
    scores = {k: alphas.zscore_xs(v.where(mask)) for k, v in scores.items()}

    ic_ew, ic_raw = strategy.rolling_ic(scores, fwd, mask, halflife_days=args.ic_halflife)
    # LAG the IC weights: weights used for bar t may only know ICs through t-1.
    ic_lag = ic_ew.shift(args.ic_lag)
    comb, wts = strategy.combine(scores, ic_lag, shrink=args.shrink)
    return mask, scores, comb, wts, ic_raw, ic_ew


def run(panels, comb, mask, args, gross_cap=None, target_vol=None):
    w = portfolio.build_weights(
        comb, mask, panels,
        target_vol=target_vol if target_vol is not None else args.target_vol,
        gross_cap=gross_cap if gross_cap is not None else args.gross_cap,
        smooth=args.smooth, max_pos=args.max_pos, top_frac=args.top_frac)
    a = {k: panels[k].to_numpy(dtype=np.float64) for k in
         ("open", "high", "low", "close", "quote_volume", "funding")}
    bt = Backtester(a["open"], a["high"], a["low"], a["close"], a["quote_volume"],
                    a["funding"], capital=args.capital,
                    max_gross_leverage=(gross_cap if gross_cap is not None else args.gross_cap),
                    maker_frac=args.maker_frac, max_participation=args.max_part)
    out = bt.run(w.to_numpy(dtype=np.float64))
    return w, out


def report(out, index, label, capital):
    eq = out["equity"]
    st = stats(eq)
    m = validation.monthly_returns(eq, index)
    print(f"\n===== {label} =====")
    print(f"  years           : {st['years']:.2f}")
    print(f"  total return    : {st['total_return']:+.1%}")
    print(f"  CAGR            : {st['cagr']:+.1%}   (monthly equiv {st['monthly_equiv']:+.2%})")
    print(f"  ann vol         : {st['ann_vol']:.1%}")
    print(f"  Sharpe          : {st['sharpe']:.2f}")
    print(f"  Sortino         : {st['sortino']:.2f}")
    print(f"  max drawdown    : {st['max_dd']:.1%}")
    print(f"  Calmar          : {st['calmar']:.2f}")
    print(f"  months >0       : {(m>0).mean():.1%} of {len(m)}")
    print(f"  median month    : {m.median():+.2%}   best {m.max():+.2%}  worst {m.min():+.2%}")
    print(f"  months >= 33%   : {(m>=0.33).sum()} of {len(m)}")
    print(f"  ann turnover    : {out['turnover'].sum()/capital/max(st['years'],1e-9):.0f}x equity")
    print(f"  total costs     : {out['costs'].sum()/capital:.1%} of initial capital")
    print(f"  total funding   : {out['funding'].sum()/capital:+.1%} of initial capital")
    print(f"  avg gross lev   : {np.nanmean(out['gross'][out['gross']>0]):.2f}")
    print(f"  liquidated      : {out['liquidated_at']}")
    return st, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_dv", type=float, default=3e6)
    ap.add_argument("--ic_halflife", type=float, default=60)
    ap.add_argument("--ic_lag", type=int, default=24)
    ap.add_argument("--shrink", type=float, default=0.3)
    ap.add_argument("--target_vol", type=float, default=0.30)
    ap.add_argument("--gross_cap", type=float, default=3.0)
    ap.add_argument("--smooth", type=int, default=12)
    ap.add_argument("--max_pos", type=float, default=0.08)
    ap.add_argument("--top_frac", type=float, default=None)
    ap.add_argument("--maker_frac", type=float, default=0.0)
    ap.add_argument("--max_part", type=float, default=0.05)
    ap.add_argument("--capital", type=float, default=1e6)
    ap.add_argument("--tag", type=str, default="base")
    args = ap.parse_args()

    panels = dataset.load()
    idx = panels["close"].index
    print(f"panel {panels['close'].shape}  {idx.min()} -> {idx.max()}")

    mask, scores, comb, wts, ic_raw, ic_ew = build(panels, args)
    print(f"avg tradable/bar: {mask.sum(axis=1).mean():.1f}")

    print("\n--- individual alpha IC (full sample, for reference only) ---")
    rows = []
    for n in ic_raw.columns:
        s = ic_raw[n].dropna()
        t = s.mean() / (s.std() / np.sqrt(len(s))) if s.std() > 0 else np.nan
        rows.append({"alpha": n, "IC": s.mean(), "t": t, "n": len(s)})
        print(f"  {n:14s} IC={s.mean():+.5f}  t={t:+6.1f}")
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"ic_{args.tag}.csv"), index=False)

    w, out = run(panels, comb, mask, args)
    st, m = report(out, idx, f"ENSEMBLE {args.tag}", args.capital)

    pd.Series(out["equity"], index=idx).to_csv(os.path.join(RES, f"equity_{args.tag}.csv"))
    m.to_csv(os.path.join(RES, f"monthly_{args.tag}.csv"))
    json.dump({k: float(v) for k, v in st.items()},
              open(os.path.join(RES, f"stats_{args.tag}.json"), "w"), indent=1)
    return st


if __name__ == "__main__":
    main()
