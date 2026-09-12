#!/usr/bin/env python3
"""Record the dataset's shape and the measured cost inputs, for the report header."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tai.config import CACHE_DIR, PANEL_DIR, RAW_DIR, RESULTS_DIR


def main() -> int:
    close = pd.read_parquet(PANEL_DIR / "1h" / "close.parquet")
    cal = json.loads((CACHE_DIR / "spread_calibration.json").read_text())
    scores = pd.read_parquet(RESULTS_DIR / "final" / "oos_scores.parquet")["score"]
    ts = scores.index.get_level_values(0)
    facts = {
        "venue": "Binance USD-margined perpetual futures",
        "source": "data.binance.vision (the exchange's public archive)",
        "bar": "1 hour",
        "symbols_cached": len(glob.glob(str(RAW_DIR / "klines" / "1h" / "*.parquet"))),
        "symbols_in_panel": int(close.shape[1]),
        "bars": int(close.shape[0]),
        "panel_start": str(close.index[0]), "panel_end": str(close.index[-1]),
        "metrics_symbols": len(glob.glob(str(RAW_DIR / "metrics" / "1h" / "*.parquet"))),
        "oos_predictions": int(len(scores)),
        "oos_start": str(ts.min()), "oos_end": str(ts.max()),
        "measured_half_spread_bps_median": cal["half_spread_bps"]["median"],
        "measured_half_spread_bps_p10": cal["half_spread_bps"]["p10"],
        "measured_half_spread_bps_p90": cal["half_spread_bps"]["p90"],
        "spread_samples": cal["n_samples"],
    }
    out = ROOT / "reports" / "data" / "dataset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
