"""Full validation suite on the final ensemble book.

Cost sensitivity, capacity, leverage frontier, regime breakdown, deflated Sharpe,
bootstrap confidence intervals and risk of ruin - all on the same weight matrix,
so differences are attributable to the stress being applied and nothing else.
"""
import os, sys, json, argparse, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, validation, run_wf
from engine import Backtester, stats, TAKER_FEE
from v3 import alpha_book, no_trade_band
from v4 import cached_scores, allocation
import final as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
Y = 24 * 365


def get_weights(panels, mask, args):
    path = os.path.join(RES, "weights_final.parquet")
    if os.path.exists(path):
        W = pd.read_parquet(path)
        print(f"loaded cached ensemble weights {W.shape}", flush=True)
        return W
    cl = panels["close"]
    r_ari = (cl / cl.shift(1) - 1.0)
    scores = cached_scores(panels, mask, args)
    horizons = [int(x) for x in args.horizons.split(",")]
    rets = {}
    for k, sc in scores.items():
        for H in horizons:
            w = alpha_book(sc, mask, H, args.max_pos)
            rets[f"{k}@{H}"] = (w.shift(1) * r_ari).sum(axis=1)
            del w
    R = pd.DataFrame(rets)
    W = F.build_ensemble_W(scores, mask, cl, horizons, args.max_pos, R,
                           args.vol_lookback, args.target_vol, args.gross_cap, r_ari)
    W = W.shift(1).fillna(0.0).astype(np.float32)
    W.to_parquet(path, compression="zstd")
    return W


def engine(panels, W, capital, gross_cap, maker_frac, pos_band, max_part,
           fee_mult=1.0, impact_mult=1.0, spread_mult=1.0, scale=1.0):
    a = {k: panels[k].to_numpy(dtype=np.float64) for k in
         ("open", "high", "low", "close", "quote_volume", "funding")}
    from engine import measure_ticks, tick_half_spread
    hs = tick_half_spread(a["close"], measure_ticks(a["close"])) * spread_mult
    bt = Backtester(a["open"], a["high"], a["low"], a["close"], a["quote_volume"],
                    a["funding"], fee=TAKER_FEE * fee_mult, maker_frac=maker_frac,
                    impact_coef=0.5 * impact_mult, capital=capital,
                    max_gross_leverage=gross_cap, max_participation=max_part,
                    half_spread=hs, rebalance_band=pos_band)
    Wx = W if scale == 1.0 else W * scale
    return bt.run(Wx.to_numpy(dtype=np.float64))


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
    args = ap.parse_args()

    panels = run_wf.screen_universe(dataset.load(), floor_dv=args.screen_dv)
    mask = portfolio.tradable_mask(panels, min_dollar_vol=args.min_dv)
    idx = panels["close"].index
    W = get_weights(panels, mask, args)
    rep = {}

    base = engine(panels, W, args.capital, args.gross_cap, args.maker_frac,
                  args.pos_band, args.max_part)
    st = stats(base["equity"])
    print(f"\nbaseline: Sharpe {st['sharpe']:+.2f} CAGR {st['cagr']:+.1%} "
          f"vol {st['ann_vol']:.1%} maxDD {st['max_dd']:.1%}", flush=True)

    # ---------- cost sensitivity ----------
    print("\n=== COST SENSITIVITY ===", flush=True)
    rows = []
    for lbl, fm, im, sm, mk in [("free", 0, 0, 0, 0.0), ("baseline (70% maker)", 1, 1, 1, args.maker_frac),
                                ("all taker", 1, 1, 1, 0.0), ("1.5x costs", 1.5, 1.5, 1.5, args.maker_frac),
                                ("2x costs", 2, 2, 2, args.maker_frac),
                                ("3x costs", 3, 3, 3, args.maker_frac)]:
        o = engine(panels, W, args.capital, args.gross_cap, mk, args.pos_band,
                   args.max_part, fee_mult=fm, impact_mult=im, spread_mult=sm)
        s = stats(o["equity"])
        rows.append(dict(case=lbl, sharpe=s["sharpe"], cagr=s["cagr"],
                         monthly=s["monthly_equiv"], maxdd=s["max_dd"]))
        print(f"  {lbl:22s} SR={s['sharpe']:+5.2f} CAGR={s['cagr']:+8.1%} "
              f"mo={s['monthly_equiv']:+6.2%} DD={s['max_dd']:7.1%}", flush=True)
    rep["costs"] = rows

    # ---------- capacity ----------
    print("\n=== CAPACITY ===", flush=True)
    rows = []
    for cap in (1e5, 1e6, 5e6, 2e7, 1e8, 5e8):
        o = engine(panels, W, cap, args.gross_cap, args.maker_frac, args.pos_band, args.max_part)
        s = stats(o["equity"])
        rows.append(dict(capital=cap, sharpe=s["sharpe"], cagr=s["cagr"],
                         monthly=s["monthly_equiv"], maxdd=s["max_dd"]))
        print(f"  ${cap:>12,.0f} SR={s['sharpe']:+5.2f} CAGR={s['cagr']:+8.1%} "
              f"mo={s['monthly_equiv']:+6.2%} DD={s['max_dd']:7.1%}", flush=True)
    rep["capacity"] = rows

    # ---------- leverage frontier: the path to 33%/month ----------
    print("\n=== LEVERAGE SWEEP (is 33%/month reachable?) ===", flush=True)
    rows = []
    for mult, gc in [(0.5, 3), (1, 3), (2, 6), (3, 9), (4, 12), (6, 18), (8, 24), (12, 36)]:
        o = engine(panels, W, args.capital, gc, args.maker_frac, args.pos_band,
                   args.max_part, scale=mult)
        s = stats(o["equity"])
        m = validation.monthly_returns(o["equity"], idx)
        rows.append(dict(leverage_mult=mult, gross_cap=gc, sharpe=s["sharpe"],
                         vol=s["ann_vol"], cagr=s["cagr"], monthly=s["monthly_equiv"],
                         maxdd=s["max_dd"], liquidated=o["liquidated_at"] is not None,
                         pct_months_ge33=float((m >= 0.33).mean())))
        print(f"  x{mult:<4} vol={s['ann_vol']:7.1%} CAGR={s['cagr']:+10.1%} "
              f"mo={s['monthly_equiv']:+7.2%} DD={s['max_dd']:7.1%} "
              f"liq={o['liquidated_at'] is not None}", flush=True)
    rep["leverage"] = rows

    # ---------- regimes ----------
    eq = pd.Series(base["equity"], index=idx)
    yr = eq.resample("YE").last()
    yr0 = pd.concat([pd.Series([eq.iloc[0]], index=[eq.index[0]]), yr]).pct_change().dropna()
    print("\n=== YEAR BY YEAR ===")
    for d, v in yr0.items():
        print(f"  {d.year}: {v:+8.1%}")
    rep["yearly"] = {str(d.year): float(v) for d, v in yr0.items()}

    # ---------- statistics ----------
    r = np.diff(np.log(np.maximum(base["equity"], 1e-9)))
    r = r[np.isfinite(r)]
    sr_bar = st["sharpe"] / np.sqrt(Y)
    sk = float(pd.Series(r).skew()); ku = float(pd.Series(r).kurt() + 3)
    psr = validation.probabilistic_sharpe(sr_bar, len(r), sk, ku, 0.0)
    grid = pd.read_csv(os.path.join(RES, "oos_grid.csv"))
    sr_var = float(np.var(grid["train_sr"].dropna() / np.sqrt(Y), ddof=1))
    n_trials = 25 * 4 + len(grid)      # alphas x horizons, plus the hyperparameter grid
    dsr = validation.deflated_sharpe(sr_bar, len(r), n_trials, sr_var, sk, ku)
    boot = validation.stationary_bootstrap(r, n_boot=1000, mean_block=168)
    ror = validation.risk_of_ruin(r, threshold=-0.5, horizon=Y, n_sim=2000)
    print("\n=== STATISTICAL VALIDATION ===")
    print(f"  annual Sharpe        : {st['sharpe']:.2f}")
    print(f"  skew / kurtosis      : {sk:+.2f} / {ku:.1f}")
    print(f"  PSR P(SR>0)          : {psr:.4f}")
    print(f"  trials counted       : {n_trials}")
    print(f"  DEFLATED Sharpe prob : {dsr:.4f}")
    if len(boot):
        print(f"  bootstrap SR 95% CI  : [{np.percentile(boot,2.5):+.2f}, {np.percentile(boot,97.5):+.2f}]")
        print(f"  P(SR<=0)             : {(boot<=0).mean():.4f}")
        rep["boot_ci"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        rep["p_sr_le_0"] = float((boot <= 0).mean())
    print(f"  P(>50% drawdown in 1y at baseline leverage) = {ror:.3f}")
    rep.update(dict(sharpe=st["sharpe"], cagr=st["cagr"], vol=st["ann_vol"],
                    maxdd=st["max_dd"], psr=psr, dsr=dsr, skew=sk, kurt=ku,
                    n_trials=n_trials, risk_of_ruin_50=float(ror)))
    json.dump(rep, open(os.path.join(RES, "validation.json"), "w"), indent=1, default=float)
    print("\nwrote results/validation.json")


if __name__ == "__main__":
    main()
