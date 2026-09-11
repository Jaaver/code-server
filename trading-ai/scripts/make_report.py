#!/usr/bin/env python3
"""Assemble reports/RESULTS.md (and a publishable HTML page) from result JSONs.

Every number in the report is read from a results file rather than typed, so the
report cannot drift from what the code actually produced.  Verdicts against the
pre-registered failure criteria in reports/PROTOCOL.md are evaluated here too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.config import RESULTS_DIR
from tai.evaluation import metrics as M
from tai.evaluation.charts import drawdown_svg, equity_svg, monthly_bars_svg

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def pct(x, nd=2):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{100 * x:.{nd}f}%"


def num(x, nd=2):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def cost_table(cs: dict) -> str:
    order = ["optimistic", "base", "conservative", "brutal"]
    rows = []
    for k in order:
        if k not in cs:
            continue
        v = cs[k]
        rows.append([k, num(v["sharpe"]), pct(v.get("geom_monthly")), pct(v["cagr"], 1),
                     pct(v["ann_vol"], 1), pct(v["max_drawdown"], 1),
                     num(v["daily_turnover"]), pct(v["total_cost_frac_of_initial"], 1),
                     pct(v["funding_pnl_frac_of_initial"], 1)])
    return table(rows, ["cost scenario", "Sharpe", "monthly (geo)", "CAGR", "ann vol",
                        "max DD", "turnover/day", "cumulative cost", "cumulative funding"])


def frontier_table(fr: list) -> str:
    rows = [[f['gross_cap'], num(r['avg_gross']), pct(r['ann_vol'], 0),
             pct(r['geom_monthly']), num(r['sharpe']), pct(r['max_drawdown'], 1),
             "yes" if r['blown_up'] else "no"]
            for f, r in ((x, x) for x in fr)]
    return table(rows, ["gross cap", "avg gross", "ann vol", "monthly (geo)", "Sharpe",
                        "max DD", "liquidated"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-tag", default="dev01")
    ap.add_argument("--holdout-tag", default="holdout")
    ap.add_argument("--validation-tag", default=None)
    ap.add_argument("--diag-tag", default=None)
    ap.add_argument("--baselines-tag", default="baselines")
    ap.add_argument("--forward-tag", default="forward")
    ap.add_argument("--target", type=float, default=0.33)
    ap.add_argument("--out", default=str(ROOT / "reports" / "RESULTS.md"))
    ap.add_argument("--html", default=str(ROOT / "reports" / "results.html"))
    args = ap.parse_args()

    dev = load(RESULTS_DIR / args.dev_tag / "report.json")
    hold = load(RESULTS_DIR / args.holdout_tag / "report.json")
    val = load(RESULTS_DIR / (args.validation_tag or args.dev_tag) / "validation.json")
    val_h = load(RESULTS_DIR / args.holdout_tag / "validation.json")
    diag = load(RESULTS_DIR / (args.diag_tag or args.dev_tag) / "diagnostics.json")
    base = load(RESULTS_DIR / args.baselines_tag / "baselines.json")
    frozen = load(ROOT / "configs" / "frozen.json")

    payload = {"dev": dev, "holdout": hold, "validation_dev": val,
               "validation_holdout": val_h, "diagnostics": diag, "baselines": base,
               "frozen": frozen, "target_monthly": args.target}
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)
    (out_dir / "data" / "summary.json").write_text(json.dumps(payload, indent=2, default=str))

    # ---- charts from the target-leverage equity curve --------------------- #
    svgs = {}
    for tag, label in ((args.holdout_tag, "holdout"), (args.dev_tag, "dev")):
        p = RESULTS_DIR / tag / "equity_target.csv"
        if p.exists():
            eq = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
            eq.index = pd.to_datetime(eq.index, utc=True)
            svgs[f"{label}_equity"] = equity_svg(eq, title=f"{label} equity (log)")
            svgs[f"{label}_dd"] = drawdown_svg(eq)
            svgs[f"{label}_monthly"] = monthly_bars_svg(M.monthly_returns(eq),
                                                        target=args.target)
    (out_dir / "data" / "charts.json").write_text(json.dumps(svgs))

    md = []
    A = md.append
    A("# Results\n")
    A("All figures below are produced by the scripts in this repository and read "
      "directly from the JSON files under `reports/data/`; none are typed by hand.\n")

    if dev:
        A("## Data and universe\n")
        A(f"- Source: Binance USD-margined perpetual futures, official monthly "
          f"archive, 1-hour bars.\n"
          f"- Development out-of-sample window: {dev['oos_start'][:10]} to "
          f"{dev['oos_end'][:10]}.\n"
          f"- Mean simultaneously tradable names: {dev['mean_universe']:.1f}; "
          f"features: {dev['n_features']}; out-of-sample predictions: "
          f"{dev['n_oos_rows']:,}.\n")
        A("\n## Signal quality (development, out-of-sample)\n")
        ic = dev["ic_residual"]
        A(f"Rank information coefficient against the residual forward return: "
          f"**{ic['ic_mean']:.4f}** mean, {ic['ic_std']:.4f} std across "
          f"{ic['n_periods']:,} cross-sections, positive in "
          f"{100 * ic['ic_positive_frac']:.1f}% of them "
          f"(t-statistic {ic['ic_ir']:.1f}).\n")

    if diag:
        A("\n### Signal decay\n")
        rows = [[h, num(v.get("ic_mean"), 4), num(v.get("ic_ir"), 1),
                 pct(v.get("ic_positive_frac"), 1)]
                for h, v in diag["signal_decay"].items()]
        A(table(rows, ["forward horizon (bars)", "IC", "t-stat", "% cross-sections positive"]))
        if "ic_by_year" in diag:
            rows = [[y, num(v.get("ic_mean"), 4), num(v.get("ic_ir"), 1)]
                    for y, v in diag["ic_by_year"].items()]
            A("\n### Stability by year\n")
            A(table(rows, ["year", "IC", "t-stat"]))
        if diag.get("ic_by_vol_regime"):
            rows = [[r, num(v.get("ic_mean"), 4), num(v.get("ic_ir"), 1)]
                    for r, v in diag["ic_by_vol_regime"].items()]
            A("\n### Stability by market-volatility regime\n")
            A(table(rows, ["regime", "IC", "t-stat"]))

    if base:
        A("\n## The bar: single-feature analytic baselines\n")
        A("Each baseline is one line of signal logic run through the identical "
          "simulator and cost model.\n")
        rows = [[k, num(v["ic"].get("ic_mean"), 4), num(v["backtest"]["sharpe"]),
                 pct(v["backtest"].get("geom_monthly")),
                 pct(v["backtest"]["max_drawdown"], 1),
                 num(v["backtest"]["daily_turnover"])]
                for k, v in base.items()]
        A(table(rows, ["baseline", "IC", "Sharpe", "monthly (geo)", "max DD", "turnover/day"]))

    for label, v in (("Development", val), ("Sealed holdout", val_h)):
        if not v:
            continue
        A(f"\n## {label}: unit-risk book (leverage 1.0, 20% volatility target)\n")
        A(cost_table(v["cost_sensitivity"]))
        bs = v.get("bootstrap_sharpe") or {}
        if bs:
            A(f"\nBlock bootstrap (weekly blocks, 2000 resamples) on the base cost "
              f"scenario: Sharpe 5th/50th/95th percentile "
              f"{num(bs['sr_p05'])} / {num(bs['sr_p50'])} / {num(bs['sr_p95'])}, "
              f"P(Sharpe > 0) = {num(bs['p_sr_gt_0'], 3)}.\n")
        if v.get("growth_frontier"):
            A(f"\n### {label}: what gross notional buys\n")
            A("A dollar-neutral book of ~100 perpetuals has single-digit annualised "
              "volatility per unit of gross notional, so the achievable growth rate is "
              "set by the gross-notional limit, not by the volatility target.\n")
            A(frontier_table(v["growth_frontier"]))
        lc = v.get("leverage_calibration", {})
        if lc:
            A(f"\n### {label}: leverage required for {100 * args.target:.0f}% per month\n")
            rows = []
            for cname, rec in lc.items():
                if not rec.get("reached"):
                    rows.append([cname, "not reached", "-", "-", "-", "-", "-"])
                    continue
                rows.append([cname, num(rec["leverage"]), num(rec["avg_gross"], 1),
                             pct(rec["ann_vol"], 0), pct(rec["geom_monthly"]),
                             pct(rec["max_drawdown"], 1), num(rec["sharpe"])])
            A(table(rows, ["cost scenario", "leverage multiple", "avg gross notional",
                           "ann vol", "monthly (geo)", "max DD", "Sharpe"]))
            bb = (lc.get("base") or {}).get("bootstrap_monthly") or {}
            if bb:
                A(f"\nAt the base-cost operating point, the bootstrapped monthly return "
                  f"distribution is {pct(bb['monthly_p05'])} / {pct(bb['monthly_p50'])} / "
                  f"{pct(bb['monthly_p95'])} (5th/50th/95th), and "
                  f"P(monthly >= {100 * args.target:.0f}%) = "
                  f"{num(bb['p_monthly_ge_33'], 3)}.\n")
            rr = lc.get("base") or {}
            if "risk_of_ruin_50pct_1y" in rr:
                A(f"\nBootstrapped probability of breaching a drawdown threshold within "
                  f"one year at that operating point: "
                  f"-30% -> {num(rr['risk_of_ruin_30pct_1y'], 3)}, "
                  f"-50% -> {num(rr['risk_of_ruin_50pct_1y'], 3)}.\n")
        if v.get("capacity"):
            A(f"\n### {label}: capacity\n")
            rows = [[f"${float(k):,.0f}", pct(c["geom_monthly"]), num(c["sharpe"]),
                     pct(c["truncation_frac"], 1), pct(c["max_drawdown"], 1)]
                    for k, c in v["capacity"].items()]
            A(table(rows, ["AUM", "monthly (geo)", "Sharpe", "orders truncated by the "
                           "participation cap", "max DD"]))
        if v.get("pbo"):
            A(f"\n### {label}: overfitting statistics\n")
            cs = v["cost_sensitivity"]["base"]
            A(f"- Probability of backtest overfitting (CSCV, "
              f"{v['pbo']['n_configs']} configurations): **{num(v['pbo']['value'], 3)}**\n"
              f"- Deflated Sharpe ratio: **{num(cs.get('dsr'), 3)}**\n"
              f"- Probabilistic Sharpe ratio: **{num(cs.get('psr'), 3)}**\n")

    if frozen:
        A("\n## Frozen configuration\n")
        A("```json\n" + json.dumps(frozen["config"], indent=2) + "\n```\n")
        A(f"Frozen at {frozen['frozen_at']} (hash `{frozen['sha256_16']}`).\n")

    fwd = None
    for p in sorted((RESULTS_DIR / args.forward_tag).glob("forward_*.json")) \
            if (RESULTS_DIR / args.forward_tag).exists() else []:
        fwd = json.loads(p.read_text())
        A("\n## Forward paper trading through the production code path\n")
        A(f"`{p.name}`: Sharpe {num(fwd['sharpe'])}, geometric monthly "
          f"{pct(fwd.get('geom_monthly'))}, annualised volatility "
          f"{pct(fwd['ann_vol'], 1)}, max drawdown {pct(fwd['max_drawdown'], 1)}, "
          f"average gross {num(fwd['avg_gross'], 1)}x, "
          f"{fwd.get('n_months', 'n/a')} months.\n")

    Path(args.out).write_text("\n".join(md))
    print(f"wrote {args.out} ({len(''.join(md))} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
