#!/usr/bin/env python3
"""Assemble cached per-symbol files into aligned wide panels on disk."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.config import UniverseConfig
from tai.data import panel as pm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    syms = pm.available_symbols(args.interval)
    logging.info("%d cached symbols", len(syms))
    p = pm.build_panel(args.interval, UniverseConfig(), symbols=syms, save=True)
    c = p["close"]
    logging.info("panel %s bars x %s symbols, %s .. %s", len(c), c.shape[1], c.index[0], c.index[-1])
    nz = c.notna().sum(axis=1)
    logging.info("symbols live: start=%d mid=%d end=%d", nz.iloc[0], nz.iloc[len(nz)//2], nz.iloc[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
