#!/usr/bin/env python3
"""Download the daily open-interest / positioning metrics archive."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.config import CACHE_DIR, PANEL_DIR
from tai.data.metrics_archive import cache_metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    qv = pd.read_parquet(PANEL_DIR / args.interval / "quote_volume.parquet")
    # rank symbols by the dollar volume they actually carried over their life
    score = qv.sum(axis=0).sort_values(ascending=False)
    syms = list(score.index[: args.top])
    first = {s: qv[s].first_valid_index() for s in syms}
    last = {s: qv[s].last_valid_index() for s in syms}
    start = pd.Timestamp(args.start, tz="UTC")
    days = {}
    for s in syms:
        lo = max(first[s], start) if first[s] is not None else start
        hi = last[s] if last[s] is not None else qv.index[-1]
        days[s] = [d.strftime("%Y-%m-%d") for d in pd.date_range(lo.normalize(),
                                                                 hi.normalize(), freq="D")]
    total = sum(len(v) for v in days.values())
    logging.info("%d symbols, %d symbol-days", len(syms), total)
    res = cache_metrics(syms, days, args.interval, workers=args.workers)
    got = {k: v for k, v in res.items() if v and v > 0}
    logging.info("cached metrics for %d symbols (%d already present)", len(got),
                 sum(1 for v in res.values() if v == -1))
    (CACHE_DIR / "metrics_symbols.json").write_text(json.dumps(sorted(res)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
