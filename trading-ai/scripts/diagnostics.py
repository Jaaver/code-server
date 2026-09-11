#!/usr/bin/env python3
"""Signal diagnostics for a saved out-of-sample score series.

Reports how fast the signal decays, how it behaves per year and per volatility
regime, the gross and net spread between score deciles, and how net Sharpe trades
off against turnover.  These are the numbers that tell you whether a backtest's
equity curve rests on a real, slowly-decaying signal or on a handful of lucky bars.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.config import COST_SCENARIOS, RESULTS_DIR, StrategyConfig, UniverseConfig
from tai.evaluation import metrics as M
from tai.features.labels import forward_return
from tai.research import pipeline as P

log = logging.getLogger("diag")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tag", default="diagnostics")
    ap.add_argument("--max-date", default=None)
    ap.add_argument("--min-date", default=None)
    ap.add_argument("--top-n", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--rebalance", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    score = pd.read_parquet(args.scores)["score"]
    ucfg = UniverseConfig(top_n=args.top_n)
    scfg = StrategyConfig(label_horizon=args.horizon, rebalance_every=args.rebalance)
    ds = P.build_context(ucfg, scfg, max_date=args.max_date, min_date=args.min_date)
    out = {}

    # ---- signal decay: IC against each forward horizon ------------------- #
    decay = {}
    for h in (1, 2, 4, 8, 12, 24, 48, 96):
        fwd = forward_return(ds.panel["open"], h).where(ds.mask)
        resid = fwd.sub(ds.aux["beta"].mul(fwd.median(axis=1), axis=0))
        flat = resid.stack(future_stack=True)
        flat.index.names = ["ts", "symbol"]
        ic = M.information_coefficient(score, flat.reindex(score.index))
        decay[h] = ic
        log.info("h=%-3d IC=%.4f  IC-IR=%.1f  pos=%.2f", h, ic.get("ic_mean", np.nan),
                 ic.get("ic_ir", np.nan), ic.get("ic_positive_frac", np.nan))
    out["signal_decay"] = decay

    # ---- IC stability by year and by market-vol regime ------------------- #
    fwd_h = forward_return(ds.panel["open"], args.horizon).where(ds.mask)
    resid_h = fwd_h.sub(ds.aux["beta"].mul(fwd_h.median(axis=1), axis=0))
    flat_h = resid_h.stack(future_stack=True)
    flat_h.index.names = ["ts", "symbol"]
    aligned = flat_h.reindex(score.index)
    df = pd.concat([score.rename("s"), aligned.rename("a")], axis=1).dropna()
    ts = df.index.get_level_values(0)
    by_year = {}
    for y in sorted(set(ts.year)):
        sub = df[ts.year == y]
        by_year[str(y)] = M.information_coefficient(sub["s"], sub["a"])
        log.info("year %s IC=%.4f n=%d", y, by_year[str(y)].get("ic_mean", np.nan),
                 by_year[str(y)].get("n_periods", 0))
    out["ic_by_year"] = by_year

    mkt_vol = ds.aux["mkt_ret"].ewm(span=72, min_periods=24).std() * np.sqrt(24 * 365)
    q = mkt_vol.rank(pct=True)
    regime = pd.cut(q.reindex(ts).to_numpy(), [0, 1 / 3, 2 / 3, 1.01],
                    labels=["low_vol", "mid_vol", "high_vol"])
    by_reg = {}
    for r in ["low_vol", "mid_vol", "high_vol"]:
        sub = df[np.asarray(regime) == r]
        if len(sub) > 1000:
            by_reg[r] = M.information_coefficient(sub["s"], sub["a"])
            log.info("regime %-9s IC=%.4f n=%d", r, by_reg[r]["ic_mean"], by_reg[r]["n_periods"])
    out["ic_by_vol_regime"] = by_reg

    # ---- decile spread (gross, before portfolio construction) ------------ #
    dec = df.groupby(level=0)["s"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop"))
    sp = df.assign(d=dec).groupby("d")["a"].mean()
    out["decile_mean_residual_fwd_return"] = {int(k): float(v) for k, v in sp.items()}
    if 9 in sp.index and 0 in sp.index:
        bars_year = 24 * 365 / args.horizon
        out["top_minus_bottom_decile_bps"] = float((sp.loc[9] - sp.loc[0]) * 1e4)
        out["top_minus_bottom_ann_gross"] = float((sp.loc[9] - sp.loc[0]) * bars_year)
        log.info("decile 10 - decile 1 = %.1f bps per %dh holding period",
                 out["top_minus_bottom_decile_bps"], args.horizon)

    # ---- turnover / net-Sharpe frontier ---------------------------------- #
    frontier = []
    # The development sweep already searched this space in detail; this is the
    # condensed version kept in the report for the shape of the trade-off.
    for fm in (False, True):
        for reb in (4, 12, 24):
            for sm in (0.0, 3.0):
                for mw in (0.06,):
                    sc2 = StrategyConfig(label_horizon=args.horizon, rebalance_every=reb,
                                         max_weight=mw)
                    res, _ = P.backtest_scores(ds, score, sc2, "base", leverage=1.0,
                                               factor_model=fm, smooth_halflife=sm,
                                       estimated_spread=True,
                                       exec_overrides={"no_trade_band": 0.008,
                                                       "cost_penalty": 1.0})
                    res = P.trim_to_oos(res, score)
                    days = max((res.equity.index[-1] - res.equity.index[0]).days, 1)
                    rec = {
                        "factor_model": fm, "rebalance": reb, "smooth_halflife": sm,
                        "max_weight": mw,
                        "sharpe": M.sharpe(res.returns, ds.bars_per_day),
                        "geom_monthly": float(np.expm1(
                            np.log1p(M.monthly_returns(res.equity)).mean())),
                        "daily_turnover": float(res.turnover.sum() / days),
                        "ann_vol": float(res.returns.std() * np.sqrt(24 * 365)),
                        "avg_gross": float(res.gross.mean()),
                        "max_dd": M.max_drawdown(res.equity),
                        "cost_drag_frac": float(res.diagnostics["total_costs"] / 1e6)}
                    # volatility per unit of gross notional decides how much gross is
                    # needed to reach a given volatility target, i.e. the growth ceiling
                    rec["vol_per_unit_gross"] = (rec["ann_vol"] / rec["avg_gross"]
                                                 if rec["avg_gross"] > 0 else float("nan"))
                    frontier.append(rec)
                    log.info("fm=%d reb=%-3d sm=%-3.1f mw=%.2f sharpe=%5.2f turn/day=%5.2f "
                             "vol/gross=%.3f", fm, reb, sm, mw, rec["sharpe"],
                             rec["daily_turnover"], rec["vol_per_unit_gross"])
    out["turnover_frontier"] = frontier
    best = max(frontier, key=lambda r: (r["sharpe"] if np.isfinite(r["sharpe"]) else -9))
    out["best_construction"] = best
    log.info("best: %s", best)

    P.save_json(out, RESULTS_DIR / args.tag / "diagnostics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
