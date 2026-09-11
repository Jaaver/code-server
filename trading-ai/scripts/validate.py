#!/usr/bin/env python3
"""Full validation battery for a saved out-of-sample score series.

Produces: cost sensitivity, leverage calibration to a monthly-return target,
block-bootstrap confidence intervals, risk of ruin, per-year and per-regime
breakdowns, capacity analysis by AUM, and CSCV probability of backtest overfitting
across the portfolio-construction configurations that were searched.
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
from tai.research import pipeline as P

log = logging.getLogger("validate")


def monthly_geo(equity: pd.Series) -> float:
    mr = M.monthly_returns(equity)
    return float(np.expm1(np.log1p(mr).mean())) if len(mr) else float("nan")


def solve_leverage(ds, score, scfg, cost_name, target_monthly, *, lo=0.5, hi=25.0,
                   iters=9, exec_overrides=None):
    """Bisect on leverage for the lowest multiple reaching ``target_monthly``."""
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        res, _ = P.backtest_scores(ds, score, scfg, cost_name, leverage=mid,
                                   exec_overrides=exec_overrides)
        res = P.trim_to_oos(res, score)
        g = monthly_geo(res.equity)
        blown = res.blown_up
        rec = {"leverage": mid, "geom_monthly": g, "blown_up": blown,
               "sharpe": M.sharpe(res.returns, ds.bars_per_day),
               "max_drawdown": M.max_drawdown(res.equity),
               "ann_vol": float(res.returns.std() * np.sqrt(24 * 365)),
               "avg_gross": float(res.gross.mean())}
        log.info("  lev=%.2f -> monthly=%.2f%% dd=%.1f%% sharpe=%.2f blown=%s",
                 mid, 100 * g, 100 * rec["max_drawdown"], rec["sharpe"], blown)
        if blown or not np.isfinite(g) or g < target_monthly:
            lo = mid
        else:
            hi = mid
            best = (rec, res)
        if hi - lo < 0.1:
            break
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tag", default="validation")
    ap.add_argument("--max-date", default=None)
    ap.add_argument("--min-date", default=None)
    ap.add_argument("--top-n", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--rebalance", type=int, default=4)
    ap.add_argument("--target-vol", type=float, default=0.20)
    ap.add_argument("--max-gross", type=float, default=4.0)
    ap.add_argument("--max-weight", type=float, default=0.06)
    ap.add_argument("--target-monthly", type=float, default=0.33)
    ap.add_argument("--aums", default="1e6,1e7,5e7,2e8")
    ap.add_argument("--n-trials", type=int, default=40,
                    help="number of configurations searched, for the deflated Sharpe")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    score = pd.read_parquet(args.scores)["score"]
    ucfg = UniverseConfig(top_n=args.top_n)
    scfg = StrategyConfig(label_horizon=args.horizon, rebalance_every=args.rebalance,
                          target_ann_vol=args.target_vol, max_gross=args.max_gross,
                          max_weight=args.max_weight)
    ds = P.build_context(ucfg, scfg, max_date=args.max_date, min_date=args.min_date)
    log.info("context: %d bars, mean universe %.1f", len(ds.mask), ds.mask.sum(axis=1).mean())

    out = {"scores": args.scores, "target_monthly": args.target_monthly}
    bpd = ds.bars_per_day

    # ---- 1. unit-risk book under every cost scenario --------------------- #
    base_curves = {}
    out["cost_sensitivity"] = {}
    for cname in COST_SCENARIOS:
        res, _ = P.backtest_scores(ds, score, scfg, cname, leverage=1.0)
        res = P.trim_to_oos(res, score)
        s = M.summarise(res, bpd, n_trials=args.n_trials)
        out["cost_sensitivity"][cname] = s
        base_curves[cname] = res.returns
        log.info("cost=%-13s sharpe=%.2f monthly=%.2f%% dd=%.1f%% turn/day=%.2f cost_drag=%.1f%%",
                 cname, s["sharpe"], 100 * s["geom_monthly"], 100 * s["max_drawdown"],
                 s["daily_turnover"], 100 * s["total_cost_frac_of_initial"])

    # ---- 2. statistical confidence on the base scenario ------------------ #
    r_base = base_curves["base"]
    out["bootstrap_sharpe"] = M.block_bootstrap_sharpe(r_base, n_boot=2000, block=24 * 7,
                                                       bars_per_day=bpd)
    out["bootstrap_monthly_unit_risk"] = M.bootstrap_monthly(r_base, n_boot=2000,
                                                             block=24 * 7, bars_per_day=bpd)
    log.info("bootstrap sharpe %s", {k: round(v, 3) for k, v in out["bootstrap_sharpe"].items()})

    # ---- 3. leverage calibration to the monthly target ------------------- #
    out["leverage_calibration"] = {}
    for cname in ("optimistic", "base", "conservative", "brutal"):
        log.info("solving leverage for cost=%s", cname)
        sol = solve_leverage(ds, score, scfg, cname, args.target_monthly)
        if sol is None:
            out["leverage_calibration"][cname] = {"reached": False}
            log.info("  target unreachable under cost=%s within leverage bound", cname)
            continue
        rec, res = sol
        full = M.summarise(res, bpd, n_trials=args.n_trials)
        rec["summary"] = full
        rec["bootstrap_monthly"] = M.bootstrap_monthly(res.returns, n_boot=2000,
                                                       block=24 * 7, bars_per_day=bpd)
        rec["risk_of_ruin_50pct_1y"] = M.risk_of_ruin(res.returns, -0.50, 12,
                                                      bars_per_day=bpd)
        rec["risk_of_ruin_30pct_1y"] = M.risk_of_ruin(res.returns, -0.30, 12,
                                                      bars_per_day=bpd)
        rec["reached"] = True
        out["leverage_calibration"][cname] = rec
        if cname == "base":
            M.monthly_returns(res.equity).to_csv(
                RESULTS_DIR / args.tag / "monthly_returns_target.csv")
            res.equity.to_csv(RESULTS_DIR / args.tag / "equity_target.csv")
            out["per_year_target"] = {
                str(y): {"ret": float(g.iloc[-1] / g.iloc[0] - 1),
                         "sharpe": M.sharpe(res.returns[res.returns.index.year == y], bpd),
                         "max_dd": M.max_drawdown(g)}
                for y, g in res.equity.groupby(res.equity.index.year) if len(g) > 200}

    # ---- 4. capacity: participation truncation vs AUM -------------------- #
    out["capacity"] = {}
    lev_base = out["leverage_calibration"].get("base", {}).get("leverage", 1.0)
    for aum in [float(x) for x in args.aums.split(",")]:
        res, _ = P.backtest_scores(ds, score, scfg, "base", leverage=lev_base,
                                   exec_overrides={"init_equity": aum})
        res = P.trim_to_oos(res, score)
        s = M.summarise(res, bpd, n_trials=args.n_trials)
        out["capacity"][f"{aum:.0f}"] = {
            "geom_monthly": s["geom_monthly"], "sharpe": s["sharpe"],
            "truncation_frac": s["capacity_truncation_frac"],
            "max_drawdown": s["max_drawdown"]}
        log.info("aum=%.0e monthly=%.2f%% sharpe=%.2f truncated=%.1f%%", aum,
                 100 * s["geom_monthly"], s["sharpe"],
                 100 * s["capacity_truncation_frac"])

    # ---- 5. PBO across the construction configurations searched ---------- #
    grid = []
    for hz_reb in (2, 4, 6, 8):
        for mw in (0.04, 0.06, 0.10):
            for nb in (0.0, 0.0015, 0.004):
                grid.append({"rebalance": hz_reb, "max_weight": mw, "no_trade_band": nb})
    rets = []
    labels = []
    for g in grid:
        sc2 = StrategyConfig(label_horizon=args.horizon, rebalance_every=g["rebalance"],
                             target_ann_vol=args.target_vol, max_gross=args.max_gross,
                             max_weight=g["max_weight"])
        res, _ = P.backtest_scores(ds, score, sc2, "base", leverage=1.0,
                                   exec_overrides={"no_trade_band": g["no_trade_band"]})
        res = P.trim_to_oos(res, score)
        rets.append(res.returns.to_numpy())
        labels.append(g)
    n = min(len(r) for r in rets)
    mat = np.column_stack([r[-n:] for r in rets])
    out["pbo"] = {"value": M.pbo_cscv(mat, n_splits=10), "n_configs": len(grid)}
    sr = [M.sharpe(pd.Series(r, index=r_base.index[-n:]), bpd) for r in mat.T]
    out["pbo"]["config_sharpes"] = [{**labels[i], "sharpe": sr[i]} for i in range(len(grid))]
    log.info("PBO = %.3f over %d configs; sharpe range %.2f..%.2f", out["pbo"]["value"],
             len(grid), min(sr), max(sr))

    P.save_json(out, RESULTS_DIR / args.tag / "validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
