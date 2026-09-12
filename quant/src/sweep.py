"""Build the sleeves once, then sweep allocation and turnover settings.

Sleeve construction dominates runtime, so it happens once; each configuration is
then scored analytically, and only the best few are re-run through the full
engine (which is the arbiter, since it alone models margin, funding, liquidity
caps and compounding).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="4,12,24,72")
    ap.add_argument("--min_dv", type=float, default=3e6)
    ap.add_argument("--screen_dv", type=float, default=2e6)
    ap.add_argument("--max_pos", type=float, default=0.10)
    ap.add_argument("--cost_bp", type=float, default=7.0)
    ap.add_argument("--tag", default="sweep2")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]

    panels = run_wf.screen_universe(dataset.load(), floor_dv=args.screen_dv)
    mask = portfolio.tradable_mask(panels, min_dollar_vol=args.min_dv)
    cl = panels["close"]
    r_ari = (cl / cl.shift(1) - 1.0)
    scores = cached_scores(panels, mask, args)

    # Pass 1: sleeve return series only. Caching every sleeve's weight matrix
    # would need ~11GB, so weights are rebuilt per allocation instead.
    keys, rets = [], {}
    for k, sc in scores.items():
        for H in horizons:
            key = f"{k}@{H}"
            w = alpha_book(sc, mask, H, args.max_pos)
            rets[key] = (w.shift(1) * r_ari).sum(axis=1)
            keys.append(key)
            del w
    R = pd.DataFrame(rets)
    print(f"sleeves {len(keys)}", flush=True)

    results = []
    # The combined book depends only on (cap, min_sr); rebal and band are cheap
    # post-processing, so only the outer pair needs a pass over the sleeves.
    for cap, min_sr in itertools.product((0.05,), (0.3,)):
        A = allocation(R, 180.0, 0.4, cap, min_sr)
        W = np.zeros(cl.shape, dtype=np.float32)
        for k, sc in scores.items():
            for H in horizons:
                key = f"{k}@{H}"
                w = alpha_book(sc, mask, H, args.max_pos)
                np.add(W, w.to_numpy(dtype=np.float32) * A[key].to_numpy(dtype=np.float32)[:, None], out=W)
                del w
        Wbase = pd.DataFrame(W, index=cl.index, columns=cl.columns).where(mask, 0.0)
        g = Wbase.abs().sum(axis=1).replace(0, np.nan)
        Wbase = Wbase.div(g, axis=0).fillna(0.0)
        del W

        for rebal, band in itertools.product((1, 4, 8, 16), (0.03, 0.05, 0.08, 0.12, 0.20)):
            Wd = no_trade_band(portfolio.rebalance_schedule(Wbase, rebal), band)
            pr = (Wd.shift(1) * r_ari).sum(axis=1)
            live = Wd.abs().sum(axis=1) > 0
            pr = pr[live]
            if len(pr) < 1000 or pr.std() == 0:
                continue
            turn_bar = Wd.diff().abs().sum(axis=1)[live].mean()
            gross_sr = pr.mean() / pr.std() * np.sqrt(Y)
            net_mu = pr.mean() - turn_bar * args.cost_bp / 1e4
            net_sr = net_mu / pr.std() * np.sqrt(Y)
            results.append(dict(cap=cap, min_sr=min_sr, rebal=rebal, band=band,
                                gross_sr=gross_sr, net_sr=net_sr,
                                turn_yr=turn_bar * Y, net_bp=net_mu * 1e4))
            print(f"  cap={cap} minsr={min_sr} rebal={rebal}h band={band}: "
                  f"gross={gross_sr:+5.2f} net={net_sr:+5.2f} turn={turn_bar*Y:6.0f}x", flush=True)
        del Wbase

    df = pd.DataFrame(results).sort_values("net_sr", ascending=False)
    df.to_csv(os.path.join(RES, f"{args.tag}.csv"), index=False)
    print("\n=== TOP CONFIGS ===")
    print(df.head(12).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
