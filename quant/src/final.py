"""Final model: ensemble over the hyperparameter grid, then the full engine.

The out-of-sample study found rank correlation -0.66 between train and test
Sharpe across 36 hyperparameter settings: choosing settings by backtest
performance was actively worse than not choosing. So the final model does not
choose. It averages the book over the whole grid, which needs no hindsight and
whose expected performance is the grid mean (+0.57 net Sharpe out of sample)
rather than the tuned figure (+1.57 in sample, -0.13 out).

Everything below is causal: sleeve allocation uses only trailing realised P&L,
and the grid average is a fixed rule, not a fitted one.
"""
import os, sys, json, argparse, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, validation, run_wf
from engine import Backtester, stats
from v3 import alpha_book, no_trade_band
from v4 import cached_scores, allocation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
Y = 24 * 365

CAPS = (0.05, 0.10)
MINSR = (0.3, 0.6)
REBALS = (4, 8, 16)
BANDS = (0.03, 0.08, 0.12)


def build_ensemble_W(scores, mask, cl, horizons, max_pos, R, vol_lookback,
                     target_vol, gross_cap, r_ari):
    """Average the book over every hyperparameter setting in the grid."""
    acc = np.zeros(cl.shape, dtype=np.float32)
    n = 0
    for cap, min_sr in itertools.product(CAPS, MINSR):
        A = allocation(R, 180.0, 0.4, cap, min_sr)
        W = np.zeros(cl.shape, dtype=np.float32)
        for k, sc in scores.items():
            for H in horizons:
                w = alpha_book(sc, mask, H, max_pos)
                np.add(W, w.to_numpy(dtype=np.float32) *
                       A[f"{k}@{H}"].to_numpy(dtype=np.float32)[:, None], out=W)
                del w
        Wb = pd.DataFrame(W, index=cl.index, columns=cl.columns).where(mask, 0.0)
        g = Wb.abs().sum(axis=1).replace(0, np.nan)
        Wb = Wb.div(g, axis=0).fillna(0.0)
        del W
        for rebal, band in itertools.product(REBALS, BANDS):
            Wd = no_trade_band(portfolio.rebalance_schedule(Wb, rebal), band)
            acc += Wd.to_numpy(dtype=np.float32)
            n += 1
            del Wd
        del Wb
    Wavg = pd.DataFrame(acc / n, index=cl.index, columns=cl.columns)
    g = Wavg.abs().sum(axis=1).replace(0, np.nan)
    Wavg = Wavg.div(g, axis=0).fillna(0.0)

    # volatility targeting on the ensemble book, causal
    pr = (Wavg.shift(1) * r_ari).sum(axis=1)
    rv = pr.rolling(vol_lookback, min_periods=vol_lookback // 3).std() * np.sqrt(Y)
    scale = (target_vol / rv.replace(0, np.nan)).clip(upper=gross_cap).fillna(0.0)
    Wavg = Wavg.mul(scale, axis=0)
    gg = Wavg.abs().sum(axis=1); over = gg > gross_cap
    Wavg.loc[over] = Wavg.loc[over].div(gg[over], axis=0) * gross_cap
    print(f"ensemble over {n} hyperparameter settings", flush=True)
    return Wavg


def run_engine(panels, W, args, target_vol=None, gross_cap=None, capital=None,
               fee_mult=1.0, impact_mult=1.0, maker_frac=None):
    a = {k: panels[k].to_numpy(dtype=np.float64) for k in
         ("open", "high", "low", "close", "quote_volume", "funding")}
    from engine import TAKER_FEE
    bt = Backtester(a["open"], a["high"], a["low"], a["close"], a["quote_volume"],
                    a["funding"],
                    fee=TAKER_FEE * fee_mult,
                    maker_frac=args.maker_frac if maker_frac is None else maker_frac,
                    impact_coef=0.5 * impact_mult,
                    capital=args.capital if capital is None else capital,
                    max_gross_leverage=args.gross_cap if gross_cap is None else gross_cap,
                    max_participation=args.max_part,
                    rebalance_band=args.pos_band)
    return bt.run(W.to_numpy(dtype=np.float64))


def summarise(out, idx, capital, label):
    st = stats(out["equity"]); m = validation.monthly_returns(out["equity"], idx)
    print(f"\n===== {label} =====")
    print(f"  CAGR {st['cagr']:+.1%}  monthly-equiv {st['monthly_equiv']:+.2%}  vol {st['ann_vol']:.1%}")
    print(f"  Sharpe {st['sharpe']:+.2f}  Sortino {st['sortino']:+.2f}  "
          f"maxDD {st['max_dd']:.1%}  Calmar {st['calmar']:+.2f}")
    print(f"  months>0 {(m>0).mean():.0%} of {len(m)}  median {m.median():+.2%}  "
          f"best {m.max():+.2%}  worst {m.min():+.2%}   months>=33%: {(m>=0.33).sum()}")
    print(f"  turnover {out['turnover'].sum()/capital/max(st['years'],1e-9):.1f}x/yr  "
          f"costs {out['costs'].sum()/capital:.1%}  funding {out['funding'].sum()/capital:+.1%}")
    return st, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="4,12,24,72")
    ap.add_argument("--min_dv", type=float, default=3e6)
    ap.add_argument("--screen_dv", type=float, default=2e6)
    ap.add_argument("--max_pos", type=float, default=0.10)
    ap.add_argument("--target_vol", type=float, default=0.30)
    ap.add_argument("--gross_cap", type=float, default=3.0)
    ap.add_argument("--capital", type=float, default=1e6)
    ap.add_argument("--maker_frac", type=float, default=0.7)
    ap.add_argument("--max_part", type=float, default=0.05)
    ap.add_argument("--pos_band", type=float, default=0.01)
    ap.add_argument("--vol_lookback", type=int, default=24 * 30)
    ap.add_argument("--split", default="2023-09-01")
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]

    panels = run_wf.screen_universe(dataset.load(), floor_dv=args.screen_dv)
    mask = portfolio.tradable_mask(panels, min_dollar_vol=args.min_dv)
    cl = panels["close"]; idx = cl.index
    r_ari = (cl / cl.shift(1) - 1.0)
    scores = cached_scores(panels, mask, args)

    rets = {}
    for k, sc in scores.items():
        for H in horizons:
            w = alpha_book(sc, mask, H, args.max_pos)
            rets[f"{k}@{H}"] = (w.shift(1) * r_ari).sum(axis=1)
            del w
    R = pd.DataFrame(rets)

    W = build_ensemble_W(scores, mask, cl, horizons, args.max_pos, R,
                         args.vol_lookback, args.target_vol, args.gross_cap, r_ari)
    W = W.shift(1).fillna(0.0)
    W.astype(np.float32).to_parquet(os.path.join(RES, f"weights_{args.tag}.parquet"),
                                    compression="zstd")

    out = run_engine(panels, W, args)
    st, m = summarise(out, idx, args.capital, f"FULL SAMPLE ({args.tag})")
    pd.Series(out["equity"], index=idx).to_csv(os.path.join(RES, f"equity_{args.tag}.csv"))
    m.to_csv(os.path.join(RES, f"monthly_{args.tag}.csv"))

    cut = idx.get_indexer([pd.Timestamp(args.split, tz="UTC")], method="nearest")[0]
    eq = pd.Series(out["equity"], index=idx)
    eq_te = eq.iloc[cut:] / eq.iloc[cut]
    st_te = stats(eq_te.to_numpy())
    m_te = validation.monthly_returns(eq_te.to_numpy(), eq_te.index)
    print(f"\n===== TEST WINDOW ONLY {idx[cut].date()} -> {idx[-1].date()} =====")
    print(f"  CAGR {st_te['cagr']:+.1%}  monthly-equiv {st_te['monthly_equiv']:+.2%}  "
          f"vol {st_te['ann_vol']:.1%}  Sharpe {st_te['sharpe']:+.2f}  maxDD {st_te['max_dd']:.1%}")
    print(f"  months>0 {(m_te>0).mean():.0%} of {len(m_te)}  median {m_te.median():+.2%}  "
          f"months>=33%: {(m_te>=0.33).sum()}")

    json.dump({"full": {k: float(v) for k, v in st.items()},
               "test": {k: float(v) for k, v in st_te.items()}},
              open(os.path.join(RES, f"stats_{args.tag}.json"), "w"), indent=1)
    R.to_csv(os.path.join(RES, f"sleeve_returns_{args.tag}.csv"))


if __name__ == "__main__":
    main()
