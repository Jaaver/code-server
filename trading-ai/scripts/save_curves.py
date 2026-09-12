#!/usr/bin/env python3
"""Export the unit-risk equity curve for a window, for the report's charts."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.config import RESULTS_DIR, StrategyConfig, UniverseConfig
from tai.evaluation import metrics as M
from tai.research import pipeline as P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cost", default="passive")
    ap.add_argument("--max-date", default=None)
    ap.add_argument("--min-date", default=None)
    ap.add_argument("--eval-from", default=None)
    ap.add_argument("--top-n", type=int, default=120)
    ap.add_argument("--rebalance", type=int, default=12)
    ap.add_argument("--no-trade-band", type=float, default=0.008)
    ap.add_argument("--cost-penalty", type=float, default=1.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    score = pd.read_parquet(args.scores)["score"]
    scfg = StrategyConfig(label_horizon=4, rebalance_every=args.rebalance)
    ds = P.build_context(UniverseConfig(top_n=args.top_n), scfg,
                         max_date=args.max_date, min_date=args.min_date)
    res, _ = P.backtest_scores(ds, score, scfg, args.cost, leverage=1.0,
                               factor_model=True, estimated_spread=True,
                               exec_overrides={"no_trade_band": args.no_trade_band,
                                               "cost_penalty": args.cost_penalty})
    res = P.trim_to_oos(res, score, args.eval_from or args.min_date)
    d = RESULTS_DIR / args.tag
    d.mkdir(parents=True, exist_ok=True)
    res.equity.to_csv(d / "equity_unit_risk.csv")
    M.monthly_returns(res.equity).to_csv(d / "monthly_unit_risk.csv")
    logging.info("%s: sharpe=%.2f monthly=%.2f%% saved to %s", args.tag,
                 M.sharpe(res.returns, 24),
                 100 * (M.monthly_stats(res.equity).get("geom_monthly") or float("nan")), d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
