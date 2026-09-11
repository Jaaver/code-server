#!/usr/bin/env python3
"""Measure the real effective spread from the exchange's trade archive.

The single most important assumption in a high-turnover backtest is what it costs
to cross the spread, and the Corwin-Schultz high-low estimator is badly upward
biased at hourly frequency on assets this volatile.  So measure it instead.

Binance's aggTrades archive flags every print with ``is_buyer_maker``, i.e. whether
the aggressor was a seller (hitting the bid) or a buyer (lifting the offer).  The
mean price of buyer-initiated prints minus the mean price of seller-initiated
prints inside a short window is a direct estimate of the effective spread, with no
quote data required.  Sampling a spread of symbols and dates gives the relationship
between spread and liquidity, which is then applied to the whole panel.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.config import BINANCE_VISION, CACHE_DIR, PANEL_DIR
from tai.data.binance_vision import _get, _read_zip_csv

log = logging.getLogger("spread")

AGG_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
            "transact_time", "is_buyer_maker"]


def effective_spread_for_day(symbol: str, day: str, window: str = "1min") -> dict | None:
    """Mean effective spread (bps) and traded notional for one symbol-day."""
    url = (f"{BINANCE_VISION}/data/futures/um/daily/aggTrades/{symbol}/"
           f"{symbol}-aggTrades-{day}.zip")
    body = _get(url, retries=3, timeout=120)
    if body is None:
        return None
    df = _read_zip_csv(body, AGG_COLS)
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    need = ("price", "quantity", "transact_time", "is_buyer_maker")
    if not all(n in cols for n in need):
        return None
    px = pd.to_numeric(df[cols["price"]], errors="coerce")
    qty = pd.to_numeric(df[cols["quantity"]], errors="coerce")
    ts = pd.to_numeric(df[cols["transact_time"]], errors="coerce")
    ibm = df[cols["is_buyer_maker"]]
    if ibm.dtype == object:
        ibm = ibm.astype(str).str.lower().isin(["true", "1"])
    else:
        ibm = ibm.astype(bool)
    good = px.notna() & qty.notna() & ts.notna() & (px > 0) & (qty > 0)
    px, qty, ts, ibm = px[good], qty[good], ts[good], ibm[good]
    if len(ts) < 500:
        return None
    scale = np.where(ts.to_numpy() > 1e15, 1_000, 1)
    idx = pd.to_datetime((ts.to_numpy() // scale).astype("int64"), unit="ms", utc=True)
    d = pd.DataFrame({"px": px.to_numpy(), "qty": qty.to_numpy(),
                      # is_buyer_maker True  -> the aggressor SOLD into the bid
                      "aggressor_buy": (~ibm).to_numpy()}, index=idx)
    if d.empty:
        return None
    g = d.groupby([pd.Grouper(freq=window), "aggressor_buy"])
    agg = g.apply(lambda x: np.average(x["px"], weights=x["qty"]) if x["qty"].sum() > 0
                  else np.nan, include_groups=False)
    wide = agg.unstack("aggressor_buy")
    if wide.shape[1] < 2:
        return None
    buy, sell = wide[True], wide[False]
    mid = (buy + sell) / 2.0
    spread_bps = ((buy - sell) / mid * 1e4).replace([np.inf, -np.inf], np.nan).dropna()
    spread_bps = spread_bps[(spread_bps > -50) & (spread_bps < 500)]
    if len(spread_bps) < 60:
        return None
    notional = float((d["px"] * d["qty"]).sum())
    return {"symbol": symbol, "day": day,
            "eff_spread_bps_mean": float(spread_bps.mean()),
            "eff_spread_bps_median": float(spread_bps.median()),
            "eff_spread_bps_p75": float(np.percentile(spread_bps, 75)),
            "half_spread_bps_median": float(spread_bps.median() / 2.0),
            "daily_notional_usd": notional, "n_windows": int(len(spread_bps)),
            "n_trades": int(len(d))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=18)
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=str(CACHE_DIR / "spread_calibration.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    qv = pd.read_parquet(PANEL_DIR / "1h" / "quote_volume.parquet")
    recent = qv.loc[qv.index >= qv.index[-1] - pd.Timedelta(days=400)]
    adv = (recent.mean() * 24).dropna().sort_values(ascending=False)
    # sample across the liquidity spectrum, not just the top of the book
    ranks = np.linspace(0, min(len(adv), 220) - 1, args.symbols).astype(int)
    syms = [adv.index[r] for r in ranks]
    days = [d.strftime("%Y-%m-%d") for d in
            pd.date_range(qv.index[-1].normalize() - pd.Timedelta(days=340),
                          qv.index[-1].normalize() - pd.Timedelta(days=2),
                          periods=args.days)]
    log.info("sampling %d symbols x %d days: %s", len(syms), len(days), syms)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(effective_spread_for_day, s, d): (s, d)
                for s in syms for d in days}
        for fut in as_completed(futs):
            s, d = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                log.warning("%s %s: %s", s, d, exc)
                r = None
            if r:
                rows.append(r)
                log.info("%-14s %s  half-spread %5.2f bps  ADV $%.1fM", s, d,
                         r["half_spread_bps_median"], r["daily_notional_usd"] / 1e6)
    if not rows:
        log.error("no samples")
        return 1
    df = pd.DataFrame(rows)
    df["adv_musd"] = df["daily_notional_usd"] / 1e6
    # fit half-spread (bps) = a + b / sqrt(ADV in $m), the usual liquidity shape
    x = 1.0 / np.sqrt(df["adv_musd"].clip(lower=0.5))
    y = df["half_spread_bps_median"]
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    pred = A @ coef
    ss = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    out = {"n_samples": int(len(df)),
           "half_spread_bps": {"mean": float(y.mean()), "median": float(y.median()),
                               "p10": float(np.percentile(y, 10)),
                               "p90": float(np.percentile(y, 90))},
           "model": {"form": "half_spread_bps = a + b / sqrt(adv_musd)",
                     "a": float(coef[0]), "b": float(coef[1]), "r2": float(ss)},
           "by_symbol": df.groupby("symbol").agg(
               half_spread_bps=("half_spread_bps_median", "median"),
               adv_musd=("adv_musd", "median")).sort_values("adv_musd",
                                                            ascending=False).reset_index()
               .to_dict("records"),
           "samples": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    log.info("half-spread: median %.2f bps, p10 %.2f, p90 %.2f; model a=%.2f b=%.2f R2=%.2f",
             y.median(), np.percentile(y, 10), np.percentile(y, 90),
             coef[0], coef[1], ss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
