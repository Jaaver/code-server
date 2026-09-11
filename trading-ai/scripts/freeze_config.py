#!/usr/bin/env python3
"""Freeze the configuration chosen during development."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.freeze import freeze

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON object of chosen settings")
    ap.add_argument("--dev-metrics", default="{}")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=str(ROOT / "configs" / "frozen.json"))
    args = ap.parse_args()
    payload = freeze(json.loads(args.config), Path(args.out),
                     dev_metrics=json.loads(args.dev_metrics), note=args.note)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
