#!/usr/bin/env python3
"""Walk-forward research driver.

Typical use::

    # development phase: data after --max-date is not even loaded
    python scripts/run_research.py --tag dev --max-date 2025-07-01

    # sealed holdout, run once, with the configuration frozen by the dev phase
    python scripts/run_research.py --tag holdout --min-date 2024-01-01
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.config import COST_SCENARIOS, RESULTS_DIR, StrategyConfig, UniverseConfig
from tai.evaluation import metrics as M
from tai.models.ensemble import LGB_VARIANTS
from tai.research import pipeline as P

log = logging.getLogger("research")


def build(args):
    ucfg = UniverseConfig(top_n=args.top_n, min_adv_usd=args.min_adv,
                          adv_lookback_days=args.adv_days,
                          listing_burn_bars=args.listing_burn)
    scfg = StrategyConfig(interval=args.interval, rebalance_every=args.rebalance,
                          label_horizon=args.horizon, embargo_bars=args.embargo,
                          min_train_bars=args.min_train_bars, test_months=args.test_months,
                          train_months=args.train_months,
                          target_ann_vol=args.target_vol, max_gross=args.max_gross,
                          max_weight=args.max_weight)
    ds = P.build_dataset(ucfg, scfg, max_date=args.max_date, min_date=args.min_date,
                         label=args.label, use_metrics=args.use_metrics)
    return ds, ucfg, scfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--max-date", default=None)
    ap.add_argument("--min-date", default=None)
    ap.add_argument("--top-n", type=int, default=120)
    ap.add_argument("--min-adv", type=float, default=3e6)
    ap.add_argument("--adv-days", type=int, default=30)
    ap.add_argument("--listing-burn", type=int, default=24 * 14)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--rebalance", type=int, default=4)
    ap.add_argument("--embargo", type=int, default=48)
    ap.add_argument("--min-train-bars", type=int, default=24 * 365)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--train-months", type=int, default=0)
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--halflife-days", type=float, default=0.0)
    ap.add_argument("--label", default="y_cs")
    ap.add_argument("--target-vol", type=float, default=0.20)
    ap.add_argument("--max-gross", type=float, default=4.0)
    ap.add_argument("--max-weight", type=float, default=0.06)
    ap.add_argument("--leverages", default="1,2,4,6,8")
    ap.add_argument("--costs", default="base,conservative,optimistic,brutal")
    ap.add_argument("--save-scores", action="store_true")
    ap.add_argument("--use-metrics", action="store_true",
                    help="include open-interest / positioning features")
    ap.add_argument("--factor-model", action="store_true",
                    help="size and neutralise with a trailing PCA factor model")
    ap.add_argument("--n-factors", type=int, default=5)
    ap.add_argument("--smooth-halflife", type=float, default=0.0,
                    help="EWMA halflife (in decision bars) applied to scores")
    ap.add_argument("--horizons", default="",
                    help="comma-separated extra label horizons to ensemble over")
    ap.add_argument("--frozen", default=None,
                    help="path to a frozen config JSON; its values override the CLI")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.frozen:
        from tai.freeze import load_frozen
        frozen = load_frozen(Path(args.frozen))
        for k, v in frozen.items():
            if hasattr(args, k):
                setattr(args, k, v)
        log.info("loaded frozen config from %s: %s", args.frozen, frozen)
    t0 = time.time()
    ds, ucfg, scfg = build(args)

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    if horizons:
        from tai.models.ensemble import cs_standardise
        parts = []
        for h in horizons:
            sc_h = StrategyConfig(**{**vars(scfg), "label_horizon": h})
            yw = ds.labels_for_horizon(h, args.label)
            ds.labels["y"] = yw
            (s_h, _folds) = P.run_walkforward(ds, sc_h, train_stride=args.train_stride,
                                              sample_halflife_days=args.halflife_days)
            if isinstance(s_h, tuple):
                s_h = s_h[0]
            log.info("horizon %d: %d OOS rows", h, len(s_h))
            parts.append(cs_standardise(s_h))
        common = parts[0].index
        for p_ in parts[1:]:
            common = common.intersection(p_.index)
        score = cs_standardise(sum(p_.reindex(common) for p_ in parts) / len(parts))
        folds = _folds
    else:
        (score, folds) = P.run_walkforward(ds, scfg, train_stride=args.train_stride,
                                           sample_halflife_days=args.halflife_days)
        if isinstance(score, tuple):
            score = score[0]
    log.info("walk-forward done in %.1f min; %d OOS rows", (time.time() - t0) / 60, len(score))

    out_dir = RESULTS_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_scores:
        score.to_frame("score").to_parquet(out_dir / "oos_scores.parquet")

    # ---- signal quality ------------------------------------------------- #
    fwd = ds.labels["fwd"].stack(future_stack=True)
    fwd.index.names = ["ts", "symbol"]
    resid = ds.labels["resid"].stack(future_stack=True)
    resid.index.names = ["ts", "symbol"]
    ic_raw = M.information_coefficient(score, fwd.reindex(score.index))
    ic_res = M.information_coefficient(score, resid.reindex(score.index))
    log.info("IC(raw fwd) %s", {k: round(v, 4) for k, v in ic_raw.items()})
    log.info("IC(residual) %s", {k: round(v, 4) for k, v in ic_res.items()})

    report = {"args": vars(args), "ic_raw": ic_raw, "ic_residual": ic_res,
              "n_oos_rows": int(len(score)),
              "oos_start": str(score.index.get_level_values(0).min()),
              "oos_end": str(score.index.get_level_values(0).max()),
              "mean_universe": float(ds.mask.sum(axis=1).mean()),
              "n_features": int(ds.X.shape[1]),
              "runs": {}}

    levs = [float(x) for x in args.leverages.split(",")]
    costs = args.costs.split(",")
    curves = {}
    for cname in costs:
        for lev in levs:
            res, cfg = P.backtest_scores(ds, score, scfg, cname, leverage=lev,
                                         factor_model=args.factor_model,
                                         n_factors=args.n_factors,
                                         smooth_halflife=args.smooth_halflife)
            res = P.trim_to_oos(res, score)
            s = M.summarise(res, ds.bars_per_day, n_trials=max(len(levs) * len(costs), 10))
            key = f"{cname}_lev{lev:g}"
            report["runs"][key] = s
            curves[key] = res.equity
            log.info("%-22s sharpe=%.2f  cagr=%.1f%%  geo_monthly=%.2f%%  maxDD=%.1f%%  "
                     "gross=%.2f  turn/day=%.2f  blown=%s", key, s["sharpe"],
                     100 * s["cagr"], 100 * s.get("geom_monthly", float("nan")),
                     100 * s["max_drawdown"], s["avg_gross"], s["daily_turnover"],
                     s["blown_up"])
            if cname == "base" and lev == levs[0]:
                mr = M.monthly_returns(res.equity)
                mr.to_csv(out_dir / "monthly_returns_base_lev1.csv")
                res.equity.to_csv(out_dir / "equity_base_lev1.csv")

    pd.DataFrame(curves).to_parquet(out_dir / "equity_curves.parquet")
    P.save_json(report, out_dir / "report.json")
    log.info("total %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
