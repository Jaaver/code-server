#!/usr/bin/env python3
"""Decompose the frozen strategy's out-of-sample record by calendar year.

A full-sample Sharpe ratio is an average, and an average hides a trend. If an edge
is being competed away, the year-by-year table is where it shows -- and it is the
difference between "this strategy works" and "this strategy worked".
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

from tai.config import RESULTS_DIR, StrategyConfig, UniverseConfig
from tai.evaluation import metrics as M
from tai.research import pipeline as P

log = logging.getLogger("byyear")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tag", default="by_year")
    ap.add_argument("--top-n", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--rebalance", type=int, default=12)
    ap.add_argument("--no-trade-band", type=float, default=0.008)
    ap.add_argument("--cost-penalty", type=float, default=1.0)
    ap.add_argument("--costs", default="maker_only,passive,base")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    score = pd.read_parquet(args.scores)["score"]
    scfg = StrategyConfig(label_horizon=args.horizon, rebalance_every=args.rebalance)
    ds = P.build_context(UniverseConfig(top_n=args.top_n), scfg)
    out: dict = {}
    for cost in args.costs.split(","):
        res, _ = P.backtest_scores(
            ds, score, scfg, cost, leverage=1.0, factor_model=True,
            estimated_spread=True,
            exec_overrides={"no_trade_band": args.no_trade_band,
                            "cost_penalty": args.cost_penalty})
        res = P.trim_to_oos(res, score)
        r, eq = res.returns, res.equity
        log.info("cost=%s full-sample sharpe=%.2f", cost, M.sharpe(r, ds.bars_per_day))
        rows = {}
        for y, g in eq.groupby(eq.index.year):
            if len(g) < 500:
                continue
            ry = r[r.index.year == y]
            ty = res.turnover[res.turnover.index.year == y]
            days = max((g.index[-1] - g.index[0]).days, 1)
            mr = M.monthly_returns(g)
            rows[str(int(y))] = {
                "sharpe": M.sharpe(ry, ds.bars_per_day),
                "geom_monthly": float(np.expm1(np.log1p(mr).mean())) if len(mr) else float("nan"),
                "total_return": float(g.iloc[-1] / g.iloc[0] - 1),
                "ann_vol": float(ry.std() * np.sqrt(ds.bars_per_day * 365)),
                "max_drawdown": M.max_drawdown(g),
                "daily_turnover": float(ty.sum() / days),
                "months": int(len(mr)),
            }
            v = rows[str(int(y))]
            log.info("  %s sharpe=%6.2f monthly=%7.2f%% ret=%8.1f%% dd=%6.1f%% turn=%.2f",
                     y, v["sharpe"], 100 * v["geom_monthly"], 100 * v["total_return"],
                     100 * v["max_drawdown"], v["daily_turnover"])
        out[cost] = {"full_sample_sharpe": M.sharpe(r, ds.bars_per_day), "years": rows}
    P.save_json(out, RESULTS_DIR / args.tag / "by_year.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
