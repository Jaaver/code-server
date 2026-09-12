"""v4: (alpha x horizon) sleeves, allocated walk-forward on realised P&L.

Each alpha decays at its own rate - order flow is worth holding for hours,
trend for days - so forcing one holding horizon on all of them throws away
edge. v4 makes every (alpha, horizon) pair its own sleeve and lets the
allocator choose among them out-of-sample, from trailing realised arithmetic
return only. Nothing about sign, horizon or sleeve selection is chosen in
sample.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, validation, run_wf
from engine import Backtester, stats
from v3 import alpha_book, no_trade_band

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
Y = 24 * 365



CACHE = os.path.join(ROOT, "data", "scores")


def cached_scores(panels, mask, args):
    """Alpha scores keyed by the universe settings that produced them.

    Recomputing 25 panels costs minutes and they are identical across every
    allocation experiment, so cache them to disk.
    """
    os.makedirs(CACHE, exist_ok=True)
    key = f"dv{int(args.min_dv)}_sc{int(args.screen_dv)}"
    path = os.path.join(CACHE, f"{key}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        names = sorted({c.split("||")[0] for c in df.columns})
        out = {}
        for n in names:
            cols = [c for c in df.columns if c.startswith(n + "||")]
            sub = df[cols]
            sub.columns = [c.split("||")[1] for c in cols]
            out[n] = sub
        print(f"loaded {len(out)} cached alpha scores from {os.path.basename(path)}", flush=True)
        return out
    sc = strategy.alpha_panel(
        panels, alphas.ALPHAS,
        post=lambda v: alphas.zscore_xs(v.where(mask)).astype(np.float32))
    flat = pd.concat({k: v for k, v in sc.items()}, axis=1)
    flat.columns = [f"{a}||{b}" for a, b in flat.columns]
    flat.to_parquet(path, compression="zstd")
    print(f"cached {len(sc)} alpha scores -> {os.path.basename(path)}", flush=True)
    return sc


def build_sleeves(scores, mask, horizons, max_pos):
    """Yield (key, weight_frame) for every (alpha, horizon) pair, one at a time."""
    for k, sc in scores.items():
        for H in horizons:
            yield f"{k}@{H}", alpha_book(sc, mask, H, max_pos)


def allocation(R, lookback_days, shrink, cap_frac, min_sr):
    """Weight each sleeve by its lagged trailing Sharpe.

    Lagged by one bar, shrunk toward equal weight over the sleeves that pass a
    minimum trailing Sharpe, and capped so no sleeve dominates.
    """
    lb = int(lookback_days * 24)
    mu = R.rolling(lb, min_periods=lb // 3).mean()
    sd = R.rolling(lb, min_periods=lb // 3).std().replace(0, np.nan)
    sr = (mu / sd).shift(1)                       # per-bar Sharpe, causal
    thresh = min_sr / np.sqrt(Y)
    keep = sr.abs() >= thresh
    A = (sr * keep).fillna(0.0)
    An = A.div(A.abs().sum(axis=1).replace(0, np.nan), axis=0)
    sg = np.sign(A)
    eq = sg.div(sg.abs().sum(axis=1).replace(0, np.nan), axis=0)
    W = ((1 - shrink) * An + shrink * eq).fillna(0.0)
    return W.clip(-cap_frac, cap_frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="4,12,24,72")
    ap.add_argument("--min_dv", type=float, default=3e6)
    ap.add_argument("--screen_dv", type=float, default=2e6)
    ap.add_argument("--lookback_days", type=float, default=180)
    ap.add_argument("--shrink", type=float, default=0.4)
    ap.add_argument("--cap_frac", type=float, default=0.08)
    ap.add_argument("--min_sr", type=float, default=0.3)
    ap.add_argument("--rebal", type=int, default=8)
    ap.add_argument("--band", type=float, default=0.005)
    ap.add_argument("--pos_band", type=float, default=0.0)
    ap.add_argument("--target_vol", type=float, default=0.30)
    ap.add_argument("--gross_cap", type=float, default=3.0)
    ap.add_argument("--max_pos", type=float, default=0.10)
    ap.add_argument("--capital", type=float, default=1e6)
    ap.add_argument("--maker_frac", type=float, default=0.0)
    ap.add_argument("--max_part", type=float, default=0.05)
    ap.add_argument("--vol_lookback", type=int, default=24 * 30)
    ap.add_argument("--tag", default="v4")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]

    panels = run_wf.screen_universe(dataset.load(), floor_dv=args.screen_dv)
    idx = panels["close"].index
    mask = portfolio.tradable_mask(panels, min_dollar_vol=args.min_dv)
    cl = panels["close"]
    r_ari = (cl / cl.shift(1) - 1.0)
    print(f"panel {cl.shape}  tradable/bar {mask.sum(axis=1).mean():.1f}", flush=True)

    scores = cached_scores(panels, mask, args)

    # pass 1: sleeve return series only (books are not kept)
    rets = {}
    for key, w in build_sleeves(scores, mask, horizons, args.max_pos):
        rets[key] = (w.shift(1) * r_ari).sum(axis=1)
        del w
    R = pd.DataFrame(rets)
    print(f"sleeves: {R.shape[1]}", flush=True)

    A = allocation(R, args.lookback_days, args.shrink, args.cap_frac, args.min_sr)

    # pass 2: rebuild and accumulate
    W = np.zeros(cl.shape, dtype=np.float32)
    for key, w in build_sleeves(scores, mask, horizons, args.max_pos):
        np.add(W, w.to_numpy(dtype=np.float32) * A[key].to_numpy(dtype=np.float32)[:, None], out=W)
        del w
    Wd = pd.DataFrame(W, index=cl.index, columns=cl.columns).where(mask, 0.0)
    g = Wd.abs().sum(axis=1).replace(0, np.nan)
    Wd = Wd.div(g, axis=0).fillna(0.0)

    pr = (Wd.shift(1) * r_ari).sum(axis=1)
    rv = pr.rolling(args.vol_lookback, min_periods=args.vol_lookback // 3).std() * np.sqrt(Y)
    scale = (args.target_vol / rv.replace(0, np.nan)).clip(upper=args.gross_cap).fillna(0.0)
    Wd = Wd.mul(scale, axis=0)
    gg = Wd.abs().sum(axis=1); over = gg > args.gross_cap
    Wd.loc[over] = Wd.loc[over].div(gg[over], axis=0) * args.gross_cap

    Wd = portfolio.rebalance_schedule(Wd, args.rebal)
    Wd = no_trade_band(Wd, args.band).shift(1).fillna(0.0)

    a = {k: panels[k].to_numpy(dtype=np.float64) for k in
         ("open", "high", "low", "close", "quote_volume", "funding")}
    bt = Backtester(a["open"], a["high"], a["low"], a["close"], a["quote_volume"],
                    a["funding"], capital=args.capital, max_gross_leverage=args.gross_cap,
                    maker_frac=args.maker_frac, max_participation=args.max_part,
                    rebalance_band=args.pos_band)
    out = bt.run(Wd.to_numpy(dtype=np.float64))
    st = stats(out["equity"]); m = validation.monthly_returns(out["equity"], idx)

    print(f"\n===== v4 {args.tag}  H={horizons} rebal={args.rebal}h band={args.band} "
          f"tv={args.target_vol:.0%} maker={args.maker_frac:.0%} =====")
    print(f"  CAGR {st['cagr']:+.1%}   monthly-equiv {st['monthly_equiv']:+.2%}   vol {st['ann_vol']:.1%}")
    print(f"  Sharpe {st['sharpe']:+.2f}  Sortino {st['sortino']:+.2f}  maxDD {st['max_dd']:.1%}  Calmar {st['calmar']:+.2f}")
    print(f"  months>0 {(m>0).mean():.0%} of {len(m)}  median {m.median():+.2%}  best {m.max():+.2%}  worst {m.min():+.2%}")
    print(f"  months>=33%: {(m>=0.33).sum()}")
    print(f"  turnover {out['turnover'].sum()/args.capital/st['years']:.1f}x/yr  "
          f"costs {out['costs'].sum()/args.capital:.1%}  funding {out['funding'].sum()/args.capital:+.1%}")
    print(f"  avg gross {np.nanmean(out['gross'][out['gross']>0]):.2f}  liq={out['liquidated_at']}")

    pd.Series(out["equity"], index=idx).to_csv(os.path.join(RES, f"equity_{args.tag}.csv"))
    m.to_csv(os.path.join(RES, f"monthly_{args.tag}.csv"))
    json.dump({k: float(v) for k, v in st.items()},
              open(os.path.join(RES, f"stats_{args.tag}.json"), "w"), indent=1)
    R.to_csv(os.path.join(RES, f"sleeve_returns_{args.tag}.csv"))
    A.iloc[::24].to_csv(os.path.join(RES, f"alloc_{args.tag}.csv"))
    return st


if __name__ == "__main__":
    main()
