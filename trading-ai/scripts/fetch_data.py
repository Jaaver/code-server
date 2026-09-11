#!/usr/bin/env python3
"""Download and cache Binance USD-M perpetual history (klines + funding)."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.config import CACHE_DIR
from tai.data import binance_vision as bv

log = logging.getLogger("fetch")


def discover(interval: str) -> dict[str, list[str]]:
    """Map symbol -> available months, cached on disk."""
    cache = CACHE_DIR / f"symbol_months_{interval}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    syms = [s for s in bv.list_perp_symbols() if re.fullmatch(r"[0-9A-Z]+USDT", s)]
    out: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=24) as pool:
        futs = {pool.submit(bv.list_symbol_months, s, interval): s for s in syms}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                out[s] = fut.result()
            except Exception:
                out[s] = []
    cache.write_text(json.dumps(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--min-months", type=int, default=12)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    months_map = discover(args.interval)
    sel = {s: m for s, m in months_map.items() if len(m) >= args.min_months}
    syms = sorted(sel, key=lambda s: -len(sel[s]))
    if args.limit:
        syms = syms[: args.limit]
    log.info("downloading %d symbols, %d files", len(syms), sum(len(sel[s]) for s in syms))

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(bv.cache_symbol, s, args.interval, sel[s], refresh=args.refresh): s
                for s in syms}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                fut.result()
            except Exception as exc:
                log.warning("%s failed: %s", s, exc)
            done += 1
            if done % 20 == 0:
                log.info("progress %d/%d", done, len(syms))
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
