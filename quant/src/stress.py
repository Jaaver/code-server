"""Stress tests: does the edge survive costs, leverage, capacity, regimes, and multiple testing?"""
import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, validation, run_wf
from engine import Backtester, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
YEAR = 24 * 365


def bt_with(panels, w, capital, gross_cap, fee_mult=1.0, impact_mult=1.0,
            spread_mult=1.0, max_part=0.05, maker_frac=0.0):
    a = {k: panels[k].to_numpy(dtype=np.float64) for k in
         ("open", "high", "low", "close", "quote_volume", "funding")}
    from engine import abdi_ranaldo_half_spread, TAKER_FEE
    hs = abdi_ranaldo_half_spread(a["high"], a["low"], a["close"]) * spread_mult
    bt = Backtester(a["open"], a["high"], a["low"], a["close"], a["quote_volume"],
                    a["funding"], fee=TAKER_FEE * fee_mult, maker_frac=maker_frac,
                    impact_coef=1.0 * impact_mult, capital=capital,
                    max_gross_leverage=gross_cap, max_participation=max_part,
                    half_spread=hs)
    return bt.run(w)


def main():
    ap = argparse.ArgumentParser()
    for k, v in dict(min_dv=3e6, ic_halflife=60.0, shrink=0.3, target_vol=0.30,
                     gross_cap=3.0, max_pos=0.08, max_part=0.05, capital=1e6).items():
        ap.add_argument(f"--{k}", type=float, default=v)
    ap.add_argument("--ic_lag", type=int, default=24)
    ap.add_argument("--smooth", type=int, default=12)
    ap.add_argument("--top_frac", type=float, default=None)
    ap.add_argument("--maker_frac", type=float, default=0.0)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--n_trials", type=int, default=60)
    args = ap.parse_args()

    panels = dataset.load()
    idx = panels["close"].index
    mask, scores, comb, wts, ic_raw, ic_ew = run_wf.build(panels, args)

    base_w = portfolio.build_weights(comb, mask, panels, target_vol=args.target_vol,
                                     gross_cap=args.gross_cap, smooth=args.smooth,
                                     max_pos=args.max_pos, top_frac=args.top_frac)
    W = base_w.to_numpy(dtype=np.float64)

    report = {}

    # ---------- 1. cost sensitivity ----------
    print("\n=== COST SENSITIVITY (fee x, spread x, impact x) ===")
    rows = []
    for fm, sm, im, lbl in [(0, 0, 0, "zero cost"), (1, 1, 1, "baseline"),
                            (1.5, 1.5, 1.5, "1.5x all"), (2, 2, 2, "2x all"),
                            (3, 3, 3, "3x all"), (1, 3, 1, "3x spread"),
                            (1, 1, 3, "3x impact")]:
        o = bt_with(panels, W, args.capital, args.gross_cap, fm, im, sm, args.max_part)
        s = stats(o["equity"])
        rows.append({"case": lbl, "sharpe": s["sharpe"], "cagr": s["cagr"],
                     "monthly": s["monthly_equiv"], "maxdd": s["max_dd"]})
        print(f"  {lbl:12s} SR={s['sharpe']:+5.2f} CAGR={s['cagr']:+8.1%} "
              f"mo={s['monthly_equiv']:+6.2%} DD={s['max_dd']:6.1%}")
    report["costs"] = rows
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"stress_costs_{args.tag}.csv"), index=False)

    # ---------- 2. capacity ----------
    print("\n=== CAPACITY (does the edge survive size?) ===")
    rows = []
    for cap in (1e5, 1e6, 5e6, 2e7, 1e8, 5e8):
        o = bt_with(panels, W, cap, args.gross_cap, max_part=args.max_part)
        s = stats(o["equity"])
        rows.append({"capital": cap, "sharpe": s["sharpe"], "cagr": s["cagr"],
                     "monthly": s["monthly_equiv"], "maxdd": s["max_dd"]})
        print(f"  ${cap:>12,.0f}  SR={s['sharpe']:+5.2f} CAGR={s['cagr']:+8.1%} "
              f"mo={s['monthly_equiv']:+6.2%} DD={s['max_dd']:6.1%}")
    report["capacity"] = rows
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"stress_capacity_{args.tag}.csv"), index=False)

    # ---------- 3. leverage / target-vol sweep ----------
    print("\n=== LEVERAGE SWEEP (path to 33%/month) ===")
    rows = []
    for tv, gc in [(0.15, 2), (0.30, 3), (0.50, 5), (0.75, 8), (1.0, 10),
                   (1.5, 15), (2.0, 20), (2.5, 25), (3.0, 30)]:
        w = portfolio.build_weights(comb, mask, panels, target_vol=tv, gross_cap=gc,
                                    smooth=args.smooth, max_pos=max(args.max_pos, tv/10),
                                    top_frac=args.top_frac)
        o = bt_with(panels, w.to_numpy(dtype=np.float64), args.capital, gc,
                    max_part=args.max_part)
        s = stats(o["equity"])
        m = validation.monthly_returns(o["equity"], idx)
        rows.append({"target_vol": tv, "gross_cap": gc, "sharpe": s["sharpe"],
                     "realised_vol": s["ann_vol"], "cagr": s["cagr"],
                     "monthly": s["monthly_equiv"], "maxdd": s["max_dd"],
                     "median_month": m.median(), "pct_months_ge_33": (m >= .33).mean(),
                     "liquidated": o["liquidated_at"] is not None})
        print(f"  tv={tv:4.0%} cap={gc:4.1f}x  SR={s['sharpe']:+5.2f} "
              f"vol={s['ann_vol']:6.1%} CAGR={s['cagr']:+9.1%} mo={s['monthly_equiv']:+7.2%} "
              f"DD={s['max_dd']:7.1%} liq={o['liquidated_at'] is not None}")
    report["leverage"] = rows
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"stress_leverage_{args.tag}.csv"), index=False)

    # ---------- 4. regime breakdown + statistics on the baseline ----------
    o = bt_with(panels, W, args.capital, args.gross_cap, max_part=args.max_part)
    eq = o["equity"]
    r = np.diff(np.log(np.maximum(eq, 1e-9)))
    s = stats(eq)
    print("\n=== YEAR BY YEAR (baseline) ===")
    ser = pd.Series(eq, index=idx)
    yr = ser.resample("YE").last()
    yr0 = pd.concat([pd.Series([ser.iloc[0]], index=[ser.index[0]]), yr]).pct_change().dropna()
    for d, v in yr0.items():
        print(f"  {d.year}: {v:+8.1%}")
    report["yearly"] = {str(d.year): float(v) for d, v in yr0.items()}

    print("\n=== STATISTICAL VALIDATION (baseline) ===")
    sr_ann = s["sharpe"]
    sr_bar = sr_ann / np.sqrt(YEAR)
    n = len(r)
    sk = float(pd.Series(r).skew()); ku = float(pd.Series(r).kurt() + 3)
    psr = validation.probabilistic_sharpe(sr_bar, n, sk, ku, 0.0)
    # variance of Sharpe across the alphas we tried = the multiple-testing penalty
    trial_srs = []
    for name in list(alphas.ALPHAS)[:]:
        wt = portfolio.build_weights(scores[name], mask, panels, target_vol=args.target_vol,
                                     gross_cap=args.gross_cap, smooth=args.smooth,
                                     max_pos=args.max_pos)
        ot = bt_with(panels, wt.to_numpy(dtype=np.float64), args.capital, args.gross_cap,
                     max_part=args.max_part)
        trial_srs.append(stats(ot["equity"])["sharpe"] / np.sqrt(YEAR))
    sr_var = float(np.var(trial_srs, ddof=1))
    dsr = validation.deflated_sharpe(sr_bar, n, max(args.n_trials, len(trial_srs)), sr_var, sk, ku)
    print(f"  annual Sharpe        : {sr_ann:.2f}")
    print(f"  skew / kurtosis      : {sk:+.2f} / {ku:.1f}")
    print(f"  PSR  (SR>0)          : {psr:.4f}")
    print(f"  trials counted       : {max(args.n_trials, len(trial_srs))}  var(SR)={sr_var:.3e}")
    print(f"  DEFLATED Sharpe prob : {dsr:.4f}")
    boot = validation.stationary_bootstrap(r, n_boot=1000, mean_block=168)
    if len(boot):
        print(f"  bootstrap SR 95% CI  : [{np.percentile(boot,2.5):.2f}, {np.percentile(boot,97.5):.2f}]")
        print(f"  P(SR<=0) bootstrap   : {(boot<=0).mean():.4f}")
        report["boot_ci"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    report.update({"sharpe": sr_ann, "psr": psr, "dsr": dsr, "sr_var_trials": sr_var,
                   "skew": sk, "kurt": ku})
    json.dump(report, open(os.path.join(RES, f"stress_{args.tag}.json"), "w"), indent=1, default=float)
    print(f"\nwrote results/stress_{args.tag}.json")


if __name__ == "__main__":
    main()
