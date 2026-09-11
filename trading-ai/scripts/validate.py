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
from tai.evaluation import growth as G
from tai.evaluation import metrics as M
from tai.research import pipeline as P

log = logging.getLogger("validate")


def monthly_geo(equity: pd.Series) -> float:
    mr = M.monthly_returns(equity)
    return float(np.expm1(np.log1p(mr).mean())) if len(mr) else float("nan")


def solve_leverage(ds, score, scfg, cost_name, target_monthly, *, lo=0.5, hi=40.0,
                   iters=11, exec_overrides=None, bt_kwargs=None, eval_from=None):
    """Bisect on leverage for the lowest multiple reaching ``target_monthly``.

    The gross-notional cap rises with the leverage multiple, because the binding
    constraint on a diversified market-neutral book is the exchange's leverage
    limit rather than the volatility target: a unit-gross book of ~100 perps has an
    annualised volatility of only single-digit percent, so reaching a high
    volatility target *requires* gross notional well above 1x equity.  The realised
    average and maximum gross are reported so the requirement is explicit.
    """
    best = None
    bt_kwargs = bt_kwargs or {}
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ov = dict(exec_overrides or {})
        ov.setdefault("max_gross", float(min(25.0, max(4.0, 3.0 * mid))))
        res, _ = P.backtest_scores(ds, score, scfg, cost_name, leverage=mid,
                                   exec_overrides=ov, **bt_kwargs)
        res = P.trim_to_oos(res, score, eval_from)
        g = monthly_geo(res.equity)
        blown = res.blown_up
        rec = {"leverage": mid, "geom_monthly": g, "blown_up": blown,
               "sharpe": M.sharpe(res.returns, ds.bars_per_day),
               "max_drawdown": M.max_drawdown(res.equity),
               "ann_vol": float(res.returns.std() * np.sqrt(24 * 365)),
               "avg_gross": float(res.gross.mean()),
               "max_gross_realised": float(res.gross.max()),
               "gross_cap": ov["max_gross"]}
        log.info("  lev=%.2f -> monthly=%.2f%% vol=%.0f%% dd=%.1f%% sharpe=%.2f "
                 "gross=%.1f blown=%s", mid, 100 * g, 100 * rec["ann_vol"],
                 100 * rec["max_drawdown"], rec["sharpe"], rec["avg_gross"], blown)
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
    ap.add_argument("--min-date", default=None,
                    help="first bar loaded into the panel (includes warm-up)")
    ap.add_argument("--eval-from", default=None,
                    help="first bar counted in the statistics; defaults to --min-date")
    ap.add_argument("--top-n", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--rebalance", type=int, default=4)
    ap.add_argument("--target-vol", type=float, default=0.20)
    ap.add_argument("--max-gross", type=float, default=4.0)
    ap.add_argument("--max-weight", type=float, default=0.06)
    ap.add_argument("--target-monthly", type=float, default=0.33)
    ap.add_argument("--aums", default="1e6,1e7,5e7,2e8")
    ap.add_argument("--factor-model", action="store_true")
    ap.add_argument("--n-factors", type=int, default=5)
    ap.add_argument("--smooth-halflife", type=float, default=0.0)
    ap.add_argument("--no-trade-band", type=float, default=0.0015)
    ap.add_argument("--cost-penalty", type=float, default=0.0)
    ap.add_argument("--headline-cost", default="passive",
                    help="cost scenario used for the headline statistics")
    ap.add_argument("--estimated-spread", action="store_true",
                    help="charge the measured per-symbol spread instead of a constant")
    ap.add_argument("--spread-method", default="measured", choices=("measured", "highlow"))
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
    EVAL_FROM = args.eval_from or args.min_date
    log.info("context: %d bars, mean universe %.1f", len(ds.mask), ds.mask.sum(axis=1).mean())
    BT = {"factor_model": args.factor_model, "n_factors": args.n_factors,
          "smooth_halflife": args.smooth_halflife,
          "estimated_spread": args.estimated_spread,
          "spread_method": args.spread_method}
    BAND = {"no_trade_band": args.no_trade_band, "cost_penalty": args.cost_penalty}

    out = {"scores": args.scores, "target_monthly": args.target_monthly}
    bpd = ds.bars_per_day

    # ---- 1. unit-risk book under every cost scenario --------------------- #
    base_curves = {}
    out["cost_sensitivity"] = {}
    for cname in COST_SCENARIOS:
        res, _ = P.backtest_scores(ds, score, scfg, cname, leverage=1.0,
                                   exec_overrides=dict(BAND), **BT)
        res = P.trim_to_oos(res, score, EVAL_FROM)
        s = M.summarise(res, bpd, n_trials=args.n_trials)
        out["cost_sensitivity"][cname] = s
        base_curves[cname] = res.returns
        log.info("cost=%-13s sharpe=%.2f monthly=%.2f%% dd=%.1f%% turn/day=%.2f cost_drag=%.1f%%",
                 cname, s["sharpe"], 100 * s["geom_monthly"], 100 * s["max_drawdown"],
                 s["daily_turnover"], 100 * s["total_cost_frac_of_initial"])

    # ---- 1b. is the monthly target reachable at all, given this Sharpe? --- #
    # Leverage rescales an edge, it does not create one: expected log growth is
    # S*s - s^2/2, so the target sets a hard floor on the Sharpe ratio before the
    # question of leverage even arises.  Record the verdict per cost scenario.
    out["growth_verdict"] = {}
    for cname, srec in out["cost_sensitivity"].items():
        vpg = (srec["ann_vol"] / srec["avg_gross"]) if srec["avg_gross"] > 0 else float("nan")
        v = G.verdict(srec["sharpe"], args.target_monthly, vpg)
        v["vol_per_unit_gross"] = vpg
        out["growth_verdict"][cname] = v
        log.info("verdict %-13s sharpe=%.2f need=%.2f -> %s (max %.1f%%/mo)%s", cname,
                 v["sharpe"], v["required_sharpe"],
                 "reachable" if v["reachable"] else "UNREACHABLE",
                 100 * v["max_monthly_at_growth_optimal_leverage"],
                 ("  needs vol %.0f%%, gross %.1fx" %
                  (100 * v["required_ann_vol"], v["required_gross"]))
                 if v["reachable"] else "")

    # ---- 2. statistical confidence on the base scenario ------------------ #
    r_base = base_curves.get(args.headline_cost, base_curves["base"])
    out["bootstrap_sharpe"] = M.block_bootstrap_sharpe(r_base, n_boot=2000, block=24 * 7,
                                                       bars_per_day=bpd)
    out["bootstrap_monthly_unit_risk"] = M.bootstrap_monthly(r_base, n_boot=2000,
                                                             block=24 * 7, bars_per_day=bpd)
    log.info("bootstrap sharpe %s", {k: round(v, 3) for k, v in out["bootstrap_sharpe"].items()})

    # ---- 3. leverage calibration to the monthly target ------------------- #
    out["leverage_calibration"] = {}
    for cname in ("maker_only", "passive", "base", "conservative", "brutal"):
        if cname not in COST_SCENARIOS:
            continue
        log.info("solving leverage for cost=%s", cname)
        sol = solve_leverage(ds, score, scfg, cname, args.target_monthly,
                             exec_overrides=dict(BAND), bt_kwargs=BT,
                             eval_from=EVAL_FROM)
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
        if cname == args.headline_cost:
            (RESULTS_DIR / args.tag).mkdir(parents=True, exist_ok=True)
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
    lev_base = (out["leverage_calibration"].get(args.headline_cost, {})
                .get("leverage", 1.0))
    for aum in [float(x) for x in args.aums.split(",")]:
        res, _ = P.backtest_scores(
            ds, score, scfg, args.headline_cost, leverage=lev_base,
            exec_overrides={**BAND, "init_equity": aum,
                            "max_gross": float(min(25.0, max(4.0, 3.0 * lev_base)))}, **BT)
        res = P.trim_to_oos(res, score, EVAL_FROM)
        s = M.summarise(res, bpd, n_trials=args.n_trials)
        out["capacity"][f"{aum:.0f}"] = {
            "geom_monthly": s["geom_monthly"], "sharpe": s["sharpe"],
            "truncation_frac": s["capacity_truncation_frac"],
            "max_drawdown": s["max_drawdown"]}
        log.info("aum=%.0e monthly=%.2f%% sharpe=%.2f truncated=%.1f%%", aum,
                 100 * s["geom_monthly"], s["sharpe"],
                 100 * s["capacity_truncation_frac"])

    # ---- 4b. growth frontier: what gross notional buys what monthly return - #
    out["growth_frontier"] = []
    for gross_cap in (2, 4, 6, 8, 10, 12, 15, 20):
        res, _ = P.backtest_scores(ds, score, scfg, args.headline_cost, leverage=50.0,
                                   exec_overrides={**BAND, "max_gross": float(gross_cap),
                                                   "vol_scalar_bounds": (0.25, 50.0)},
                                   **BT)
        res = P.trim_to_oos(res, score, EVAL_FROM)
        rec = {"gross_cap": gross_cap, "avg_gross": float(res.gross.mean()),
               "ann_vol": float(res.returns.std() * np.sqrt(24 * 365)),
               "geom_monthly": monthly_geo(res.equity),
               "sharpe": M.sharpe(res.returns, bpd),
               "max_drawdown": M.max_drawdown(res.equity),
               "blown_up": bool(res.blown_up)}
        out["growth_frontier"].append(rec)
        log.info("gross<=%-3d vol=%3.0f%% monthly=%6.2f%% dd=%5.1f%% sharpe=%.2f blown=%s",
                 gross_cap, 100 * rec["ann_vol"], 100 * rec["geom_monthly"],
                 100 * rec["max_drawdown"], rec["sharpe"], rec["blown_up"])

    # ---- 4c. the honest ceiling: best monthly return without liquidation --- #
    ok = [r for r in out["growth_frontier"] if not r["blown_up"]]
    if ok:
        best = max(ok, key=lambda r: r["geom_monthly"])
        out["max_sustainable"] = best
        deep = [r for r in ok if r["max_drawdown"] > -0.50]
        out["max_sustainable_dd50"] = (max(deep, key=lambda r: r["geom_monthly"])
                                       if deep else None)
        log.info("best monthly without liquidation: %.2f%% at gross %.1fx (dd %.0f%%)",
                 100 * best["geom_monthly"], best["avg_gross"],
                 100 * best["max_drawdown"])
        if out["max_sustainable_dd50"]:
            b2 = out["max_sustainable_dd50"]
            log.info("best monthly with drawdown under 50%%: %.2f%% at gross %.1fx",
                     100 * b2["geom_monthly"], b2["avg_gross"])

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
                                   exec_overrides={"no_trade_band": g["no_trade_band"]},
                                   **BT)
        res = P.trim_to_oos(res, score, EVAL_FROM)
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
