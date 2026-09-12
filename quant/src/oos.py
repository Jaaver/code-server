"""Strict out-of-sample test.

Everything up to now chose four hyperparameters (rebalance cadence, weight band,
allocation cap, minimum sleeve Sharpe) by looking at full-sample results. The
alpha allocation was already walk-forward, but those four were not, so the
headline number is contaminated by selection.

This splits the sample: hyperparameters are chosen on the TRAIN window only,
then applied unchanged to a TEST window the selection never saw.
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


def build_W(scores, mask, cl, horizons, max_pos, R, cap, min_sr, rebal, band):
    A = allocation(R, 180.0, 0.4, cap, min_sr)
    W = np.zeros(cl.shape, dtype=np.float32)
    for k, sc in scores.items():
        for H in horizons:
            w = alpha_book(sc, mask, H, max_pos)
            np.add(W, w.to_numpy(dtype=np.float32) * A[f"{k}@{H}"].to_numpy(dtype=np.float32)[:, None], out=W)
            del w
    Wd = pd.DataFrame(W, index=cl.index, columns=cl.columns).where(mask, 0.0)
    g = Wd.abs().sum(axis=1).replace(0, np.nan)
    Wd = Wd.div(g, axis=0).fillna(0.0)
    return no_trade_band(portfolio.rebalance_schedule(Wd, rebal), band)


def score_window(W, r_ari, sl, cost_bp):
    Ww = W.iloc[sl]
    pr = (Ww.shift(1) * r_ari.iloc[sl]).sum(axis=1)
    live = Ww.abs().sum(axis=1) > 0
    pr = pr[live]
    if len(pr) < 500 or pr.std() == 0:
        return np.nan, np.nan
    turn = Ww.diff().abs().sum(axis=1)[live].mean()
    net = (pr.mean() - turn * cost_bp / 1e4) / pr.std() * np.sqrt(Y)
    return net, turn * Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="4,12,24,72")
    ap.add_argument("--min_dv", type=float, default=3e6)
    ap.add_argument("--screen_dv", type=float, default=2e6)
    ap.add_argument("--max_pos", type=float, default=0.10)
    ap.add_argument("--cost_bp", type=float, default=7.0)
    ap.add_argument("--split", default="2023-09-01")
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

    cut = idx.get_indexer([pd.Timestamp(args.split, tz="UTC")], method="nearest")[0]
    tr, te = slice(0, cut), slice(cut, len(idx))
    print(f"train {idx[0].date()}..{idx[cut-1].date()}   test {idx[cut].date()}..{idx[-1].date()}", flush=True)

    rows = []
    # The blend depends only on (cap, min_sr); rebal and band are post-processing,
    # so the expensive sleeve pass runs once per allocation, not per config.
    for cap, min_sr in itertools.product((0.05, 0.10), (0.3, 0.6)):
        A = allocation(R, 180.0, 0.4, cap, min_sr)
        Wacc = np.zeros(cl.shape, dtype=np.float32)
        for k, sc in scores.items():
            for H in horizons:
                w = alpha_book(sc, mask, H, args.max_pos)
                np.add(Wacc, w.to_numpy(dtype=np.float32) * A[f"{k}@{H}"].to_numpy(dtype=np.float32)[:, None], out=Wacc)
                del w
        Wb = pd.DataFrame(Wacc, index=cl.index, columns=cl.columns).where(mask, 0.0)
        g = Wb.abs().sum(axis=1).replace(0, np.nan)
        Wb = Wb.div(g, axis=0).fillna(0.0)
        del Wacc
        for rebal, band in itertools.product((4, 8, 16), (0.03, 0.08, 0.12)):
            W = no_trade_band(portfolio.rebalance_schedule(Wb, rebal), band)
            s_tr, t_tr = score_window(W, r_ari, tr, args.cost_bp)
            s_te, t_te = score_window(W, r_ari, te, args.cost_bp)
            rows.append(dict(cap=cap, min_sr=min_sr, rebal=rebal, band=band,
                             train_sr=s_tr, test_sr=s_te, train_turn=t_tr, test_turn=t_te))
            print(f"  cap={cap} minsr={min_sr} rebal={rebal} band={band}: "
                  f"TRAIN={s_tr:+.2f}  TEST={s_te:+.2f}", flush=True)
            del W
        del Wb

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "oos_grid.csv"), index=False)

    best = df.loc[df["train_sr"].idxmax()]
    print("\n=== SELECTION MADE ON TRAIN ONLY ===")
    print(f"  chosen: cap={best.cap} min_sr={best.min_sr} rebal={best.rebal} band={best.band}")
    print(f"  train net Sharpe = {best.train_sr:+.2f}")
    print(f"  TEST  net Sharpe = {best.test_sr:+.2f}   <- honest out-of-sample")
    print(f"\n  mean TEST Sharpe across all {len(df)} configs = {df.test_sr.mean():+.2f}")
    print(f"  best  TEST Sharpe (hindsight, NOT achievable) = {df.test_sr.max():+.2f}")
    print(f"  rank correlation train vs test = {df.train_sr.corr(df.test_sr, method='spearman'):+.2f}")
    json.dump(dict(chosen=best.to_dict(), mean_test=float(df.test_sr.mean()),
                   best_test=float(df.test_sr.max()),
                   rank_corr=float(df.train_sr.corr(df.test_sr, method="spearman"))),
              open(os.path.join(RES, "oos_summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
