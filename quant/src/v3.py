"""v3: ensemble allocated on realised arithmetic P&L, walk-forward.

What changed from v1/v2, and why:
  * Objective. v1/v2 weighted alphas by rank IC on LOG returns. Rank IC is
    sign-blind to the log/arithmetic choice but does not determine P&L: several
    signals here have positive IC and negative arithmetic P&L (xs_rev_72:
    IC +0.056, arithmetic Sharpe -0.57). v3 allocates on each alpha's own
    realised arithmetic return, which is what the book actually earns, and which
    also fixes each signal's sign without an in-sample choice.
  * Smoothing. v1/v2 smoothed the WEIGHT matrix, which left slowly decaying
    residual positions in names the universe filter had dropped - turnover went
    UP with smoothing (113x -> 566x). v3 smooths the SIGNAL, then re-applies the
    mask and renormalises, so weight only ever sits in tradable names.
  * Turnover. A no-trade band holds the existing weight until the target moves
    far enough to be worth the spread.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, validation, run_wf
from engine import Backtester, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
Y = 24 * 365


def alpha_book(sc, mask, H, max_pos=0.10):
    """Standalone dollar-neutral book for one alpha, gross 1.

    Signal is smoothed over the holding horizon FIRST, then masked and
    renormalised - never the other way round.
    """
    s = sc.where(mask)
    if H > 1:
        s = s.rolling(H, min_periods=max(2, H // 4)).mean()
    s = s.where(mask)
    s = s.sub(s.mean(axis=1), axis=0).where(s.notna())
    g = s.abs().sum(axis=1).replace(0, np.nan)
    w = s.div(g, axis=0).fillna(0.0).clip(-max_pos, max_pos)
    g2 = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(g2, axis=0).fillna(0.0)


def no_trade_band(W, band):
    """Hold the current weight until the target drifts more than `band` away.

    Cuts turnover without materially delaying the signal. Implemented per bar
    across the whole weight matrix.
    """
    if band <= 0:
        return W
    A = W.to_numpy(dtype=np.float32)
    out = np.empty_like(A)
    cur = np.zeros(A.shape[1], dtype=np.float32)
    for t in range(A.shape[0]):
        tgt = A[t]
        move = np.abs(tgt - cur) > band
        cur = np.where(move, tgt, cur)
        cur = np.where(np.abs(tgt) < 1e-12, tgt, cur)   # always honour an exit
        out[t] = cur
    return pd.DataFrame(out, index=W.index, columns=W.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=72)
    ap.add_argument("--band", type=float, default=0.002)
    ap.add_argument("--lookback_days", type=float, default=180)
    ap.add_argument("--min_dv", type=float, default=3e6)
    ap.add_argument("--screen_dv", type=float, default=2e6)
    ap.add_argument("--target_vol", type=float, default=0.30)
    ap.add_argument("--gross_cap", type=float, default=3.0)
    ap.add_argument("--max_pos", type=float, default=0.10)
    ap.add_argument("--capital", type=float, default=1e6)
    ap.add_argument("--maker_frac", type=float, default=0.0)
    ap.add_argument("--max_part", type=float, default=0.05)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--vol_lookback", type=int, default=24 * 30)
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args()

    panels = run_wf.screen_universe(dataset.load(), floor_dv=args.screen_dv)
    idx = panels["close"].index
    mask = portfolio.tradable_mask(panels, min_dollar_vol=args.min_dv)
    cl = panels["close"]
    r_ari = (cl / cl.shift(1) - 1.0)                 # return realised over bar t
    print(f"panel {cl.shape}  tradable/bar {mask.sum(axis=1).mean():.1f}", flush=True)

    scores = strategy.alpha_panel(
        panels, alphas.ALPHAS,
        post=lambda v: alphas.zscore_xs(v.where(mask)).astype(np.float32))

    names = list(scores)
    # Pass 1: each book's own return series. Books are ~226MB each, so we build
    # one, take its return series, and drop it rather than holding all 25.
    rets = {}
    for k in names:
        w = alpha_book(scores[k], mask, args.H, args.max_pos)
        rets[k] = (w.shift(1) * r_ari).sum(axis=1)     # causal: w from t-1 earns bar t
        del w
    R = pd.DataFrame(rets)

    # --- walk-forward allocation on trailing realised P&L (sign included) ---
    lb = int(args.lookback_days * 24)
    mu = R.rolling(lb, min_periods=lb // 3).mean()
    sd = R.rolling(lb, min_periods=lb // 3).std().replace(0, np.nan)
    score = (mu / sd)                                  # trailing Sharpe per alpha
    A = score.shift(1)                                 # lag: no same-bar information
    A = A.clip(-0.02, 0.02)                            # cap any single alpha's pull
    denom = A.abs().sum(axis=1).replace(0, np.nan)
    An = A.div(denom, axis=0)
    eq = np.sign(A).div(np.sign(A).abs().sum(axis=1).replace(0, np.nan), axis=0)
    Aw = ((1 - args.shrink) * An + args.shrink * eq).fillna(0.0)
    print(f"alphas allocated: {len(names)}   mean |weight| spread ok", flush=True)

    # --- Pass 2: rebuild each book and accumulate it into the blend ---
    W = np.zeros(cl.shape, dtype=np.float32)
    for k in names:
        w = alpha_book(scores[k], mask, args.H, args.max_pos)
        np.add(W, w.to_numpy(dtype=np.float32) * Aw[k].to_numpy(dtype=np.float32)[:, None],
               out=W)
        del w
    W = pd.DataFrame(W, index=cl.index, columns=cl.columns).where(mask, 0.0)
    g = W.abs().sum(axis=1).replace(0, np.nan)
    W = W.div(g, axis=0).fillna(0.0)                   # gross 1

    # --- volatility targeting on the combined book, causal ---
    pr = (W.shift(1) * r_ari).sum(axis=1)
    rv = pr.rolling(args.vol_lookback, min_periods=args.vol_lookback // 3).std() * np.sqrt(Y)
    scale = (args.target_vol / rv.replace(0, np.nan)).clip(upper=args.gross_cap).fillna(0.0)
    W = W.mul(scale, axis=0)
    gg = W.abs().sum(axis=1); over = gg > args.gross_cap
    W.loc[over] = W.loc[over].div(gg[over], axis=0) * args.gross_cap

    W = no_trade_band(W, args.band).shift(1).fillna(0.0)

    a = {k: panels[k].to_numpy(dtype=np.float64) for k in
         ("open", "high", "low", "close", "quote_volume", "funding")}
    bt = Backtester(a["open"], a["high"], a["low"], a["close"], a["quote_volume"],
                    a["funding"], capital=args.capital, max_gross_leverage=args.gross_cap,
                    maker_frac=args.maker_frac, max_participation=args.max_part)
    out = bt.run(W.to_numpy(dtype=np.float64))
    st = stats(out["equity"]); m = validation.monthly_returns(out["equity"], idx)

    print(f"\n===== v3  H={args.H} band={args.band} tv={args.target_vol:.0%} "
          f"maker={args.maker_frac:.0%} =====")
    print(f"  CAGR {st['cagr']:+.1%}   monthly-equiv {st['monthly_equiv']:+.2%}   vol {st['ann_vol']:.1%}")
    print(f"  Sharpe {st['sharpe']:+.2f}  Sortino {st['sortino']:+.2f}  maxDD {st['max_dd']:.1%}  Calmar {st['calmar']:+.2f}")
    print(f"  months>0 {(m>0).mean():.0%} of {len(m)}   median {m.median():+.2%}  best {m.max():+.2%}  worst {m.min():+.2%}")
    print(f"  months>=33% : {(m>=0.33).sum()}")
    print(f"  ann turnover {out['turnover'].sum()/args.capital/st['years']:.1f}x   "
          f"costs {out['costs'].sum()/args.capital:.1%}   funding {out['funding'].sum()/args.capital:+.1%}")
    print(f"  avg gross {np.nanmean(out['gross'][out['gross']>0]):.2f}   liq={out['liquidated_at']}")

    pd.Series(out["equity"], index=idx).to_csv(os.path.join(RES, f"equity_{args.tag}.csv"))
    m.to_csv(os.path.join(RES, f"monthly_{args.tag}.csv"))
    json.dump({k: float(v) for k, v in st.items()},
              open(os.path.join(RES, f"stats_{args.tag}.json"), "w"), indent=1)
    R.to_csv(os.path.join(RES, f"alpha_returns_{args.tag}.csv"))
    return st


if __name__ == "__main__":
    main()
