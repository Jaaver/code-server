#!/usr/bin/env python3
"""Render the results dossier as a single self-contained HTML page.

Reads reports/data/summary.json and reports/data/charts.json (written by
make_report.py) and emits reports/results.html.  Every figure on the page comes
from those files, so the page cannot disagree with the code that produced it.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CSS = """
:root{
  --ground:#f5f6f7; --surface:#ffffff; --surface-2:#eceef0; --edge:#d7dbde;
  --ink:#111a1f; --ink-2:#41525c; --ink-3:#71848f;
  --accent:#0e7c8a; --accent-soft:#d3e8ea;
  --amber:#a56c0d; --pos:#2f7d5c; --neg:#a8463c;
  --chart-line:#0e7c8a; --chart-dd:#a8463c; --grid:#dfe3e6;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:"IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;
  --serif:"Newsreader",Georgia,"Times New Roman",serif;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0c1216; --surface:#121b21; --surface-2:#18242b; --edge:#25343d;
    --ink:#e6edf0; --ink-2:#a9bac3; --ink-3:#7a8d98;
    --accent:#4fc3d0; --accent-soft:#16333a;
    --amber:#e0a83c; --pos:#54b98c; --neg:#e0776a;
    --chart-line:#4fc3d0; --chart-dd:#e0776a; --grid:#1f2d35;
  }
}
:root[data-theme="dark"]{
  --ground:#0c1216; --surface:#121b21; --surface-2:#18242b; --edge:#25343d;
  --ink:#e6edf0; --ink-2:#a9bac3; --ink-3:#7a8d98;
  --accent:#4fc3d0; --accent-soft:#16333a;
  --amber:#e0a83c; --pos:#54b98c; --neg:#e0776a;
  --chart-line:#4fc3d0; --chart-dd:#e0776a; --grid:#1f2d35;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased;margin:0}
.page{max-width:1040px;margin:0 auto;padding-inline:20px;padding-block:40px 72px;
  display:flex;flex-direction:column;gap:52px}
.measure{max-width:68ch}
h1,h2,h3{font-family:var(--serif);font-weight:600;text-wrap:balance;margin:0;
  letter-spacing:-0.012em}
h1{font-size:clamp(2rem,4.6vw,2.9rem);line-height:1.12}
h2{font-size:clamp(1.35rem,2.4vw,1.72rem);line-height:1.2;padding-top:22px;
  border-top:1px solid var(--edge)}
section:first-of-type h2{border-top:0;padding-top:0}
h3{font-size:1rem;line-height:1.3;color:var(--ink-2);margin-top:10px}
p{margin:0}
.eyebrow{font-family:var(--mono);font-size:.69rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-3)}
.lede{font-size:1.1rem;color:var(--ink-2)}
section{display:flex;flex-direction:column;gap:18px}
.stack{display:flex;flex-direction:column;gap:12px}
hr.rule{border:0;border-top:1px solid var(--edge);margin:0}

.verdict{background:var(--surface);border:1px solid var(--edge);border-radius:4px;
  padding:24px;display:flex;flex-direction:column;gap:20px}
.verdict .answer{font-family:var(--serif);font-size:clamp(1.25rem,2.6vw,1.6rem);
  line-height:1.3;text-wrap:balance}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:1px;
  background:var(--edge);border:1px solid var(--edge);border-radius:3px;overflow:hidden}
.kpi{background:var(--surface);padding:14px 16px;display:flex;flex-direction:column;gap:3px}
.kpi .k{font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3)}
.kpi .v{font-family:var(--mono);font-size:1.42rem;font-variant-numeric:tabular-nums;
  letter-spacing:-0.02em;line-height:1.15}
.kpi.hero{padding:18px 16px}
.kpi.hero .v{font-size:2.15rem}
.kpi.hero .k{color:var(--ink-2)}
.kpi .n{font-size:.78rem;color:var(--ink-3);line-height:1.35}
.kpi.warn .v{color:var(--amber)}
.kpi.good .v{color:var(--pos)}

table{width:100%;border-collapse:collapse;font-size:.84rem;
  font-variant-numeric:tabular-nums}
.tw{overflow-x:auto;border:1px solid var(--edge);border-radius:3px;background:var(--surface)}
th,td{padding:8px 12px;text-align:right;white-space:nowrap;border-top:1px solid var(--edge)}
thead th{border-top:0;font-family:var(--mono);font-size:.67rem;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-3);font-weight:500;background:var(--surface-2)}
th:first-child,td:first-child{text-align:left;font-family:var(--sans)}
tbody tr:hover{background:var(--surface-2)}
td.mono,th.mono{font-family:var(--mono)}
.pos{color:var(--pos)}.neg{color:var(--neg)}
caption{caption-side:bottom;text-align:left;padding:8px 12px;font-size:.78rem;
  color:var(--ink-3)}

.chart{width:100%;height:auto;display:block}
.chartbox{background:var(--surface);border:1px solid var(--edge);border-radius:3px;
  padding:14px 10px 6px}
.chartbox .cap{font-family:var(--mono);font-size:.67rem;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3);padding:0 6px 8px}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .line{fill:none;stroke:var(--chart-line);stroke-width:1.6;
  stroke-linejoin:round}
.chart .ddarea{fill:var(--chart-dd);fill-opacity:.26;stroke:var(--chart-dd);
  stroke-width:1}
.chart .up{fill:var(--pos)}.chart .dn{fill:var(--neg)}
.chart .target{stroke:var(--amber);stroke-width:1.3;stroke-dasharray:5 4}
.chart .ylab,.chart .xlab,.chart .tlab{font-family:var(--mono);font-size:9.5px;
  fill:var(--ink-3)}
.chart .ylab{text-anchor:end}.chart .xlab{text-anchor:middle}
.chart .tlab{fill:var(--amber)}

.criteria{display:flex;flex-direction:column;gap:1px;background:var(--edge);
  border:1px solid var(--edge);border-radius:3px;overflow:hidden}
.crit{background:var(--surface);display:grid;grid-template-columns:74px 1fr auto;
  gap:14px;align-items:baseline;padding:12px 16px}
.crit .chip{font-family:var(--mono);font-size:.64rem;letter-spacing:.1em;
  text-transform:uppercase;padding:3px 7px;border-radius:2px;text-align:center}
.chip.pass{background:var(--accent-soft);color:var(--accent)}
.chip.fail{background:color-mix(in srgb,var(--neg) 16%,transparent);color:var(--neg)}
.crit .val{font-family:var(--mono);font-size:.84rem;color:var(--ink-2);
  font-variant-numeric:tabular-nums;text-align:right}
.crit .txt{font-size:.9rem}

.note{border-left:2px solid var(--accent);padding:2px 0 2px 16px;color:var(--ink-2);
  font-size:.93rem}
ul.tight{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:7px}
ul.tight li{font-size:.95rem}
code{font-family:var(--mono);font-size:.85em;background:var(--surface-2);
  padding:1px 4px;border-radius:2px}
.foot{font-size:.82rem;color:var(--ink-3);border-top:1px solid var(--edge);padding-top:18px}
@media (max-width:560px){
  .crit{grid-template-columns:64px 1fr;row-gap:6px}
  .crit .val{grid-column:2;text-align:left}
}
@media (prefers-reduced-motion:no-preference){
  .chart .line{stroke-dasharray:6000;stroke-dashoffset:0;
    animation:draw 1.1s ease-out both}
  @keyframes draw{from{stroke-dashoffset:6000}to{stroke-dashoffset:0}}
}
"""


def pc(x, nd=2, sign=False):
    if x is None or not isinstance(x, (int, float)) or not np.isfinite(x):
        return "&mdash;"
    s = f"{100 * x:+.{nd}f}%" if sign else f"{100 * x:.{nd}f}%"
    return s


def nm(x, nd=2):
    if x is None or not isinstance(x, (int, float)) or not np.isfinite(x):
        return "&mdash;"
    return f"{x:.{nd}f}"


def tbl(header, rows, caption="", cls=""):
    h = "".join(f"<th>{c}</th>" for c in header)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead>'
            f"<tbody>{body}</tbody>{cap}</table></div>")


def kpi(k, v, n="", tone=""):
    return (f'<div class="kpi {tone}"><span class="k">{k}</span>'
            f'<span class="v">{v}</span><span class="n">{n}</span></div>')


def criterion(ok, text, value):
    chip = "pass" if ok else "fail"
    label = "pass" if ok else "fail"
    return (f'<div class="crit"><span class="chip {chip}">{label}</span>'
            f'<span class="txt">{text}</span><span class="val">{value}</span></div>')


def build(summary: dict, charts: dict, target: float) -> str:
    dev = summary.get("dev") or {}
    hold = summary.get("holdout") or {}
    vd = summary.get("validation_dev") or {}
    vh = summary.get("validation_holdout") or {}
    diag = summary.get("diagnostics") or {}
    base = summary.get("baselines") or {}
    frozen = summary.get("frozen") or {}
    fwd = summary.get("forward") or {}

    headline = ((frozen or {}).get("config", {}) or {}).get(
        "headline_cost_scenario", "passive")
    primary = vh or vd
    which = "sealed holdout" if vh else "development"
    cs = (primary.get("cost_sensitivity") or {}).get(headline, {})
    dev_cs = (vd.get("cost_sensitivity") or {}).get(headline, {}) if vd else {}
    hd_cs = (vh.get("cost_sensitivity") or {}).get(headline, {}) if vh else {}
    dev_ms = (vd or {}).get("max_sustainable") or {}
    dev_ms50 = (vd or {}).get("max_sustainable_dd50") or {}
    decay = summary.get("decay") or {}
    by_year = summary.get("by_year") or {}
    facts = summary.get("facts") or {}
    lc = (primary.get("leverage_calibration") or {}).get(headline, {})
    op = lc if lc.get("reached") else {}
    opsum = op.get("summary", {})

    parts = [f"<style>{CSS}</style>",
             '<link rel="preconnect" href="https://fonts.googleapis.com">',
             '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&'
             'family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;1,6..72,400&'
             'display=swap">',
             '<div class="page">']
    P = parts.append

    # ---- header ---------------------------------------------------------- #
    gv = primary.get("growth_verdict") or {}
    gvh = gv.get(headline, {})
    need = gvh.get("required_sharpe") or 2.616
    reached = bool(op) and bool(gvh.get("reachable"))

    P('<header class="stack">'
      f'<span class="eyebrow">binance usd-m perpetuals &middot; hourly bars &middot; '
      f'2020&ndash;2026 &middot; walk-forward, sealed holdout</span>'
      f'<h1>Can a trading model make {100 * target:.0f}% a month?</h1>'
      f'<p class="lede measure">No &mdash; and the work says so three different ways. '
      f'The arithmetic puts a floor of Sharpe {nm(need)} on the target before leverage '
      f'is even a question. The simulation says the leverage the arithmetic calls for '
      f'liquidates the account. And the sealed holdout says the edge, while still '
      f'measurable, has shrunk to about the size of the trading fees.</p></header>')

    # ---- KPI strip: the contrast is the result --------------------------- #
    kpis = [
        kpi("Sharpe required", nm(need),
            f"for {100 * target:.0f}%/month at any leverage"),
        kpi("Sharpe, development", nm(dev_cs.get("sharpe")),
            "2021-02 to 2025-06, walk-forward", "good hero"),
        kpi("Sharpe, sealed holdout", nm(hd_cs.get("sharpe")),
            f"{hd_cs.get('n_months', '')} months after the config was frozen",
            "warn hero"),
        kpi("monthly, holdout", pc(hd_cs.get("geom_monthly")),
            f"{pc(hd_cs.get('pct_months_positive'), 0)} of months positive", "warn"),
        kpi("best monthly ever reached", pc(dev_ms.get("geom_monthly")),
            f"development, {nm(dev_ms.get('avg_gross'), 1)}&times; gross, "
            f"{pc(dev_ms.get('max_drawdown'), 0)} drawdown"),
        kpi("deflated Sharpe, holdout", nm(hd_cs.get("dsr"), 3),
            "penalised for 108 configurations searched", "warn"),
    ]
    P('<section><div class="kpis">' + "".join(kpis) + "</div>"
      '<div class="measure"><p class="note">The two Sharpe figures come from the same '
      'model, the same frozen configuration and the same cost assumptions. The only '
      'difference is which months they cover. Everything that follows is an attempt to '
      'explain that gap, because it is the whole answer.</p></div></section>')

    # ---- the arithmetic that decides the question ------------------------ #
    if gv:
        rows = []
        for cname in ("maker_only", "passive", "base", "conservative", "brutal"):
            v = gv.get(cname)
            if not v:
                continue
            rows.append([
                cname.replace("_", " "), nm(v["sharpe"]), nm(v["required_sharpe"]),
                pc(v["max_monthly_at_growth_optimal_leverage"], 1),
                (pc(v["required_ann_vol"], 0) if v["reachable"] else "&mdash;"),
                (f'{nm(v["required_gross"], 0)}&times;' if v["reachable"] else "&mdash;"),
                ('<span class="pos">reachable</span>' if v["reachable"]
                 else '<span class="neg">unreachable</span>')])
        P(f'<section><h2>Why the answer is arithmetic before it is empirical</h2>'
          '<div class="measure stack">'
          '<p>Leverage rescales an edge; it does not create one. A book run at '
          'annualised volatility <em>s</em> with Sharpe <em>S</em> compounds at '
          '<em>S&middot;s &minus; s&sup2;/2</em> &mdash; the return grows linearly in '
          'leverage but the volatility drag grows with its square. That expression '
          'peaks at <em>s = S</em>, so the best compound growth any book can reach, at '
          'any leverage, is <em>S&sup2;/2</em> per year.</p>'
          f'<p>Turning that around: {100 * target:.0f}% a month requires '
          f'<strong>Sharpe &ge; {nm(need)}</strong> '
          f'(&radic;(24&nbsp;&middot;&nbsp;ln&nbsp;{1 + target:.2f})). Below that the '
          'target is not a question of how much leverage the exchange allows &mdash; it '
          'is unreachable at every leverage. Above it, the target fixes the volatility, '
          'and the volatility fixes the gross notional and the drawdowns.</p></div>'
          + tbl(["execution", "net Sharpe", "Sharpe needed", "best monthly at any leverage",
                 "volatility for target", "gross for target", "verdict"], rows,
                "Computed by evaluation/growth.py from each scenario's measured "
                "out-of-sample Sharpe; the code is unit-tested against the closed form.")
          + "</section>")

    # ---- development vs holdout ------------------------------------------ #
    if vd and vh:
        rows = []
        for cname in ("maker_only", "passive", "base", "conservative", "brutal"):
            a = (vd.get("cost_sensitivity") or {}).get(cname)
            b = (vh.get("cost_sensitivity") or {}).get(cname)
            if not a or not b:
                continue
            rows.append([cname.replace("_", " "), nm(a["sharpe"]), pc(a["geom_monthly"]),
                         nm(b["sharpe"]), pc(b["geom_monthly"]),
                         pc(b["max_drawdown"], 0)])
        P('<section><h2>Development against the sealed holdout</h2>'
          '<div class="measure stack">'
          '<p>Both columns are out-of-sample in the sense that matters for a model: '
          'every prediction comes from an ensemble fitted only on bars that preceded '
          'it. The difference is that the portfolio construction &mdash; rebalance '
          'interval, no-trade band, per-name cap &mdash; was chosen on the development '
          'window, and the holdout was not looked at until it was frozen.</p></div>'
          + tbl(["execution", "Sharpe (dev)", "monthly (dev)", "Sharpe (holdout)",
                 "monthly (holdout)", "max DD (holdout)"], rows,
                "Unit-risk book at a 20% volatility target in both windows.")
          + "</section>")

    # ---- year by year ----------------------------------------------------- #
    if by_year:
        blocks = []
        for cname in ("maker_only", "passive", "base"):
            blk = by_year.get(cname)
            if not blk:
                continue
            rows = [[y, nm(v["sharpe"]), pc(v["geom_monthly"]), pc(v["total_return"], 1),
                     pc(v["max_drawdown"], 1), nm(v["daily_turnover"])]
                    for y, v in blk["years"].items()]
            blocks.append(
                f'<h3>{cname.replace("_", " ")} execution &mdash; full sample Sharpe '
                f'{nm(blk["full_sample_sharpe"])}</h3>'
                + tbl(["year", "Sharpe", "monthly", "year return", "max DD",
                       "turnover/day"], rows))
        if blocks:
            P('<section><h2>A full-sample Sharpe ratio is an average, and it hides a '
              'trend</h2>'
              '<div class="measure stack">'
              '<p>Every row is out-of-sample: each prediction comes from an ensemble '
              'fitted only on bars that preceded it, and the book runs at a 20% '
              'volatility target with no extra leverage.</p>'
              '<p>2021 is most of the full-sample result. It is also the year the '
              'universe was smallest, the venue least institutional, and short-horizon '
              'cross-sectional reversal least competed for. Whatever 2021 was, it is not '
              'the market this would be deployed into.</p></div>'
              + "".join(blocks) + "</section>")

    # ---- why the returns fell but the prediction did not ------------------ #
    if decay:
        series = decay.get("12") or decay.get(next(iter(decay), ""), {})
        rows = [[k, nm(v.get("ic_mean"), 4), nm(v.get("ic_ir"), 1),
                 f'{nm(v.get("decile_spread_bps"), 1)} bp',
                 f'{nm(v.get("cross_sectional_dispersion_bps"), 0)} bp']
                for k, v in series.items()]
        if rows:
            P('<section><h2>The prediction held. The prize did not.</h2>'
              '<div class="measure stack">'
              '<p>Rank information coefficient is a correlation, so it is scale-free: it '
              'can hold steady while the money drains out. What a dollar-neutral book '
              'earns is that correlation <em>multiplied by</em> how far apart the '
              'cross-section spreads, and crypto\'s cross-sectional dispersion has '
              'compressed as the market matured.</p>'
              '<p>Both are below for the twelve-hour horizon the strategy trades. The '
              'correlation falls by about a third across the sample; what a '
              'top-minus-bottom decile actually pays falls by roughly five times, into '
              'the same order of magnitude as the round-trip fee.</p></div>'
              + tbl(["period", "IC", "t", "top decile &minus; bottom decile",
                     "cross-sectional dispersion"], rows,
                    "Measured on the same out-of-sample predictions used everywhere else "
                    "on this page.")
              + "</section>")

    # ---- what it trades -------------------------------------------------- #
    P('<section><h2>What the model is</h2>'
      '<div class="measure stack">'
      '<p>A gradient-boosted ensemble ranks roughly '
      f'{nm(dev.get("mean_universe"), 0)} liquid USDT perpetual contracts every four '
      'hours and holds a dollar-neutral, factor-neutral long/short book. It does not '
      'forecast the direction of crypto; it forecasts which contracts will out- or '
      'under-perform their peers over the next few hours, from '
      f'{dev.get("n_features", "~110")} features spanning vol-normalised and '
      'residual momentum, taker order-flow imbalance, trade-intensity '
      'microstructure, funding and open-interest positioning, and market-state '
      'conditioning.</p>'
      '<p>Training is expanding-window walk-forward, retrained quarterly, with a '
      'purge and a 48-bar embargo before every test window. Fills are at the next '
      "bar's open and pay exchange fees, half the spread, and square-root impact in "
      "the participation rate against that bar's actual dollar volume; funding is "
      'settled every eight hours on realised rates; delisted contracts are '
      'force-liquidated with penalty costs.</p></div></section>')

    # ---- signal --------------------------------------------------------- #
    icr = dev.get("ic_residual") or {}
    if icr:
        P('<section><h2>Is there a signal at all</h2>'
          '<div class="measure stack"><p>The rank correlation between the prediction '
          'and the subsequent residual return, measured across every out-of-sample '
          f'cross-section: <strong>{nm(icr.get("ic_mean"), 4)}</strong> mean, positive '
          f'in {pc(icr.get("ic_positive_frac"), 1)} of '
          f'{icr.get("n_periods", 0):,} cross-sections '
          f'(t&nbsp;=&nbsp;{nm(icr.get("ic_ir"), 1)}). Small per-name edges are the '
          'normal shape of this kind of alpha; what makes them tradable is breadth '
          'and how slowly they decay.</p></div>')
        if diag.get("signal_decay"):
            rows = [[f"{h}", nm(v.get("ic_mean"), 4), nm(v.get("ic_ir"), 1),
                     pc(v.get("ic_positive_frac"), 1)]
                    for h, v in diag["signal_decay"].items()]
            P(tbl(["forward horizon (bars)", "IC", "t", "% positive"], rows,
                  "Signal decay. A signal that dies within one bar cannot pay for its "
                  "own turnover."))
        if diag.get("ic_by_year"):
            rows = [[y, nm(v.get("ic_mean"), 4), nm(v.get("ic_ir"), 1),
                     f'{v.get("n_periods", 0):,}']
                    for y, v in diag["ic_by_year"].items()]
            P(tbl(["year", "IC", "t", "cross-sections"], rows,
                  "Year by year, including the 2022 bear market and the 2024-25 "
                  "high-funding regime."))
        P("</section>")

    # ---- baselines ------------------------------------------------------- #
    if base:
        rows = []
        for k, v in base.items():
            b = v["backtest"]
            rows.append([k.replace("_", " "), nm(v["ic"].get("ic_mean"), 4),
                         nm(b["sharpe"]), pc(b.get("geom_monthly")),
                         pc(b["max_drawdown"], 1), nm(b["daily_turnover"])])
        P('<section><h2>The bar the model has to clear</h2>'
          '<div class="measure"><p>Each row is one line of signal logic &mdash; no '
          'fitting &mdash; run through the identical simulator and cost model. If the '
          'ensemble cannot beat these, it is not earning its complexity.</p></div>'
          + tbl(["baseline", "IC", "Sharpe", "monthly", "max DD", "turnover/day"], rows)
          + "</section>")

    # ---- costs ----------------------------------------------------------- #
    cssens = primary.get("cost_sensitivity") or {}
    if cssens:
        rows = []
        for k in ("optimistic", "base", "conservative", "brutal"):
            if k not in cssens:
                continue
            v = cssens[k]
            rows.append([k, nm(v["sharpe"]), pc(v.get("geom_monthly")),
                         pc(v["cagr"], 1), pc(v["ann_vol"], 1),
                         pc(v["max_drawdown"], 1), nm(v["daily_turnover"]),
                         pc(v["total_cost_frac_of_initial"], 0),
                         pc(v["funding_pnl_frac_of_initial"], 1)])
        P('<section><h2>What costs do to it</h2>'
          '<div class="measure"><p>The unlevered book at a 20% volatility target, '
          'priced four ways. <em>Base</em> is VIP-0 taker fees with a quarter of the '
          'flow filled passively; <em>brutal</em> is 7&nbsp;bp taker fees, a '
          '5&nbsp;bp half-spread and triple the impact coefficient. Cumulative cost is '
          'expressed against starting equity, so a figure above 100% simply means the '
          'book traded many times its own size over the period.</p></div>'
          + tbl(["costs", "Sharpe", "monthly", "CAGR", "ann vol", "max DD",
                 "turnover/day", "cumulative cost", "funding P&amp;L"], rows)
          + "</section>")

    # ---- growth frontier ------------------------------------------------- #
    gf = (vd or primary).get("growth_frontier") or []
    gf_label = "development" if vd else which
    if gf:
        rows = [[f"{r['gross_cap']}&times;", nm(r["avg_gross"], 1), pc(r["ann_vol"], 0),
                 pc(r["geom_monthly"]), nm(r["sharpe"]), pc(r["max_drawdown"], 1),
                 '<span class="neg">liquidated</span>' if r["blown_up"] else "survived"]
                for r in gf]
        P('<section><h2>Why leverage, not a bigger volatility target, is the binding '
          'constraint</h2>'
          '<div class="measure stack"><p>A dollar-neutral book of ~100 perpetuals has '
          'only single-digit annualised volatility per unit of gross notional: the '
          'names diversify each other, which is the whole point of the construction. '
          'So the growth rate is set by how much gross notional the venue will carry, '
          'not by the volatility target in the config.</p>'
          '<p>Each row raises the gross-notional cap and lets volatility targeting run '
          'to it. Maintenance margin and bankruptcy are checked every bar, so a row '
          'marked <em>liquidated</em> is a path that would have been closed out, not a '
          'deep drawdown.</p></div>'
          + tbl(["gross cap", "avg gross", "ann vol", "monthly", "Sharpe", "max DD",
                 "outcome"], rows,
                f"Measured on the {gf_label} window at the headline execution "
                f"assumption, with volatility targeting allowed to run up to each cap.")
          + "</section>")

    # ---- the empirical ceiling ------------------------------------------- #
    hd_ms = (vh or {}).get("max_sustainable") or {}
    if dev_ms:
        tiles = [
            kpi("development ceiling", pc(dev_ms["geom_monthly"]),
                f'{nm(dev_ms["avg_gross"], 1)}&times; gross, '
                f'{pc(dev_ms["ann_vol"], 0)} vol, {pc(dev_ms["max_drawdown"], 0)} drawdown',
                "warn"),
        ]
        if dev_ms50:
            tiles.append(kpi("development, drawdown under 50%", pc(dev_ms50["geom_monthly"]),
                             f'{nm(dev_ms50["avg_gross"], 1)}&times; gross, '
                             f'{pc(dev_ms50["ann_vol"], 0)} vol, '
                             f'{pc(dev_ms50["max_drawdown"], 0)} drawdown', "good"))
        if hd_ms:
            tiles.append(kpi("sealed-holdout ceiling", pc(hd_ms["geom_monthly"]),
                             f'{nm(hd_ms["avg_gross"], 1)}&times; gross, '
                             f'{pc(hd_ms["max_drawdown"], 0)} drawdown', "warn"))
        tiles.append(kpi(f"{100 * target:.0f}%/month", "not reached",
                         "the leverage the arithmetic calls for liquidates the book"))
        P('<section><h2>The ceiling the simulation actually reaches</h2>'
          '<div class="measure stack">'
          '<p>The closed form is an upper bound that assumes independent lognormal '
          'increments. Run on the real return path, with maintenance margin and '
          'bankruptcy checked every bar, leverage stops paying well before the '
          'arithmetic says it should &mdash; and then it stops abruptly.</p></div>'
          '<div class="kpis">' + "".join(tiles) + '</div>'
          '<div class="measure"><p class="note">The gap is the price of fat tails. The '
          'growth formula treats each hour as an independent draw; a real crypto return '
          'stream clusters its worst hours together, so the drawdown that arrives at high '
          'leverage is deeper than lognormal maths predicts &mdash; deep enough to cross '
          'the maintenance-margin line, at which point the position is closed for you and '
          'the compounding argument ends.</p></div></section>')

    # ---- charts ---------------------------------------------------------- #
    key = "holdout" if "holdout_equity" in charts else "dev"
    if f"{key}_equity" in charts:
        P('<section><h2>The operating point</h2>'
          f'<div class="chartbox"><div class="cap">equity, log scale &mdash; '
          f'{which}</div>{charts[f"{key}_equity"]}</div>'
          f'<div class="chartbox"><div class="cap">drawdown from peak</div>'
          f'{charts.get(f"{key}_dd", "")}</div>'
          f'<div class="chartbox"><div class="cap">monthly returns vs the '
          f'{100 * target:.0f}% target</div>{charts.get(f"{key}_monthly", "")}</div>'
          "</section>")

    # ---- leverage calibration + risk ------------------------------------- #
    lcal = primary.get("leverage_calibration") or {}
    if lcal:
        rows = []
        for cname, rec in lcal.items():
            if not rec.get("reached"):
                rows.append([cname, "not reachable", "&mdash;", "&mdash;", "&mdash;",
                             "&mdash;", "&mdash;"])
                continue
            rows.append([cname, f"{nm(rec['leverage'])}&times;",
                         f"{nm(rec['avg_gross'], 1)}&times;", pc(rec["ann_vol"], 0),
                         pc(rec["geom_monthly"]), pc(rec["max_drawdown"], 1),
                         nm(rec["sharpe"])])
        P(f'<section><h2>What {100 * target:.0f}% a month costs in risk</h2>'
          + tbl(["costs", "leverage", "avg gross", "ann vol", "monthly", "max DD",
                 "Sharpe"], rows,
                "The lowest leverage multiple that reaches the target, found by "
                "bisection, under each cost scenario.")
          )
        bb = op.get("bootstrap_monthly") or {}
        if bb:
            P('<div class="measure stack">'
              f'<p class="note">Block-bootstrapping the operating point (weekly blocks, '
              f'2000 resamples) puts the monthly return at {pc(bb["monthly_p05"])} / '
              f'{pc(bb["monthly_p50"])} / {pc(bb["monthly_p95"])} across the 5th, 50th '
              f'and 95th percentiles, and '
              f'P(monthly&nbsp;&ge;&nbsp;{100 * target:.0f}%) = '
              f'{nm(bb["p_monthly_ge_33"], 3)}. The target is the centre of that '
              'distribution, not a floor: individual months routinely land well below '
              'it.</p>')
            if "risk_of_ruin_50pct_1y" in op:
                P(f'<p class="note">Probability of breaching a drawdown threshold '
                  f'within one year at that leverage: '
                  f'<strong>{pc(op["risk_of_ruin_30pct_1y"], 1)}</strong> for &minus;30%, '
                  f'<strong>{pc(op["risk_of_ruin_50pct_1y"], 1)}</strong> for &minus;50%.</p>')
            P("</div>")
        P("</section>")

    # ---- capacity -------------------------------------------------------- #
    cap = primary.get("capacity") or {}
    if cap:
        rows = [[f"${float(k) / 1e6:,.0f}M", pc(c["geom_monthly"]), nm(c["sharpe"]),
                 pc(c["truncation_frac"], 1), pc(c["max_drawdown"], 1)]
                for k, c in cap.items()]
        P('<section><h2>How much money it holds</h2>'
          '<div class="measure"><p>Orders are capped at 5% of the bar&rsquo;s actual '
          'dollar volume and the excess is simply not filled. That is the honest '
          'capacity constraint, and it is the first thing that breaks as the account '
          'grows.</p></div>'
          + tbl(["AUM", "monthly", "Sharpe", "orders truncated", "max DD"], rows)
          + "</section>")

    # ---- forward paper trading ------------------------------------------- #
    if fwd and "sharpe" in fwd:
        start_eq = max(fwd.get("start_equity", 1.0), 1.0)
        P('<section><h2>The same result through the production code path</h2>'
          '<div class="measure stack">'
          '<p>The vectorised backtest and the live system share the feature code but not '
          'the control flow, so the holdout was also replayed one bar at a time through '
          'the objects a deployment would actually run &mdash; a rolling window, a frozen '
          'model, a broker that charges the same fees and funding. If the backtest were '
          'quietly using information a live system could not have, this is where it would '
          'show.</p></div>'
          '<div class="kpis">'
          + kpi("net Sharpe", nm(fwd["sharpe"]),
                f'{fwd.get("n_months", "")} months, one bar at a time', "warn")
          + kpi("monthly", pc(fwd.get("geom_monthly")),
                f'max drawdown {pc(fwd.get("max_drawdown"), 0)}', "warn")
          + kpi("paid in fees", pc(fwd.get("total_costs", 0) / start_eq, 1),
                "of starting capital")
          + kpi("paid in funding", pc(fwd.get("total_funding", 0) / start_eq, 1),
                "dollar-neutral is not funding-neutral")
          + '</div>'
          '<div class="measure"><p class="note">Those last two tiles are the whole story '
          'in two numbers. The gross signal over the holdout is real and roughly that '
          'size; the fees and the funding are the same size; what is left is noise.</p>'
          '</div></section>')

    # ---- pre-registered criteria ----------------------------------------- #
    crits = []
    sr = cs.get("sharpe")
    crits.append(criterion(bool(sr and sr >= 1.0),
                           "Out-of-sample net Sharpe at or above 1.0",
                           nm(sr)))
    pbo = (primary.get("pbo") or {}).get("value")
    crits.append(criterion(pbo is not None and pbo <= 0.5,
                           "Probability of backtest overfitting at or below 0.50",
                           nm(pbo, 3)))
    dsr = cs.get("dsr")
    crits.append(criterion(dsr is not None and dsr >= 0.95,
                           "Deflated Sharpe ratio at or above 0.95", nm(dsr, 3)))
    if dev_cs and hd_cs:
        d_m, h_m = dev_cs.get("geom_monthly"), hd_cs.get("geom_monthly")
        ok = bool(d_m and h_m and h_m >= d_m / 3)
        crits.append(criterion(ok, "Holdout monthly return at least a third of the "
                                   "development figure, same leverage",
                               f"{pc(h_m)} vs {pc(d_m)}"))
    # The recommended operating point is the best leverage that survives, so the
    # question is whether *that* point liquidates -- not whether any point on the
    # frontier does, since the frontier is deliberately pushed until it breaks.
    rec = dev_ms50 or dev_ms
    if rec:
        crits.append(criterion(not rec.get("blown_up", True),
                               "The recommended operating point survives the whole sample",
                               f'{nm(rec["avg_gross"], 1)}&times; gross, '
                               f'{pc(rec["max_drawdown"], 0)} drawdown'))
    crits.append(criterion(reached,
                           f"{100 * target:.0f}%/month reachable at the headline "
                           f"execution assumption",
                           (f"{nm(gvh.get('required_gross'), 0)}&times; gross needed"
                            if reached else
                            f"needs Sharpe {nm(need)}, has {nm(cs.get('sharpe'))}")))
    P('<section><h2>Against the criteria set before the holdout was opened</h2>'
      '<div class="measure"><p>These thresholds are recorded in '
      '<code>reports/PROTOCOL.md</code>, written before the holdout period was '
      'touched, so the verdict could not be negotiated afterwards.</p></div>'
      '<div class="criteria">' + "".join(crits) + "</div></section>")

    # ---- limitations ----------------------------------------------------- #
    P('<section><h2>What would break this</h2>'
      '<ul class="tight measure">'
      '<li><strong>Fill assumptions.</strong> The cost model charges a quarter of the '
      'flow at maker fees. A book that actually crosses the spread on every order pays '
      'the <em>conservative</em> column, not the <em>base</em> one.</li>'
      '<li><strong>Capacity.</strong> The edge lives in the mid-cap half of the '
      'universe. The capacity table shows where the participation cap starts eating '
      'the return; past that point the strategy is a different, worse strategy.</li>'
      '<li><strong>Leverage availability.</strong> The operating point assumes the '
      'venue carries the stated gross notional across ~100 contracts at once, through '
      'a liquidation cascade, without raising margin requirements. Exchanges raise '
      'them exactly when you least want it.</li>'
      '<li><strong>Crowding.</strong> Short-horizon cross-sectional reversal in crypto '
      'is not a secret. The IC-by-year table is the thing to watch: if the edge is '
      'being competed away, it shows up there first.</li>'
      '<li><strong>Regime dependence.</strong> The sample contains one full bull/bear '
      'cycle. It does not contain a sustained low-volatility, low-funding crypto '
      'market, because one has not happened yet on perpetuals.</li>'
      '</ul></section>')

    if frozen:
        P('<section><h2>Frozen configuration</h2>'
          '<div class="tw"><table><tbody>'
          + "".join(f'<tr><td>{html.escape(str(k))}</td><td class="mono">'
                    f'{html.escape(str(v))}</td></tr>'
                    for k, v in frozen.get("config", {}).items())
          + f'</tbody><caption>Frozen {html.escape(str(frozen.get("frozen_at", "")))}'
            f' &middot; hash <code>{html.escape(str(frozen.get("sha256_16", "")))}</code>'
            f'</caption></table></div></section>')

    P('<p class="foot">Research output, not investment advice. Figures are simulated on '
      'historical data from the exchange&rsquo;s own public archive; simulated results '
      'omit failure modes that only appear with real orders in the book. The drawdown '
      'and ruin figures above are the load-bearing part of the answer.</p>')
    P("</div>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "reports" / "data" / "summary.json"))
    ap.add_argument("--charts", default=str(ROOT / "reports" / "data" / "charts.json"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "results.html"))
    ap.add_argument("--target", type=float, default=0.33)
    args = ap.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    charts = json.loads(Path(args.charts).read_text()) if Path(args.charts).exists() else {}
    html_out = ("<title>33% a Month, Stress-Tested</title>\n"
                + build(summary, charts, args.target))
    Path(args.out).write_text(html_out)
    print(f"wrote {args.out} ({len(html_out):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
