#!/usr/bin/env python3
"""Second research pass: more breadth, two horizons, positioning features.

The baseline pipeline establishes a defensible result with one horizon and 120
names.  Sharpe is the binding constraint on the monthly-return question, so this
pass spends compute on the three things most likely to raise it:

  * a wider universe (breadth raises the information ratio roughly as sqrt(N),
    and the cost-aware sizing keeps the extra, less liquid names from being
    over-weighted);
  * an ensemble over two label horizons, so the model is trained on the horizon it
    is actually traded at as well as a faster one;
  * open-interest and long/short positioning features, where the archive covers
    the symbol.

It writes to its own tags so the baseline result stays intact for comparison.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tai.config import CACHE_DIR, RESULTS_DIR

log = logging.getLogger("enhanced")
PY = sys.executable
CUTOFF = "2025-07-01"
TARGET = 0.33


def run(name: str, args: list[str]) -> int:
    logfile = CACHE_DIR / f"{name}.log"
    log.info("=== %s", name)
    t0 = time.time()
    with open(logfile, "w") as fh:
        p = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ))
    log.info("=== %s rc=%d in %.1f min", name, p.returncode, (time.time() - t0) / 60)
    if p.returncode != 0:
        log.error("\n".join(logfile.read_text().splitlines()[-20:]))
    return p.returncode


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    frozen = json.loads((ROOT / "configs" / "frozen.json").read_text())["config"]
    top_n = int(os.environ.get("TAI_ENH_TOPN", "150"))
    horizons = os.environ.get("TAI_ENH_HORIZONS", "4,12")
    use_metrics = os.environ.get("TAI_ENH_METRICS", "1") == "1"
    # The development window's own IC-by-period table shows the edge decaying
    # (0.129 in 2021H1 to 0.071 in 2025H1), so weighting recent bars more heavily is
    # justified by development evidence alone, not by anything seen in the holdout.
    halflife = os.environ.get("TAI_ENH_HALFLIFE", "540")

    scores = RESULTS_DIR / "enh" / "oos_scores.parquet"
    if not scores.exists():
        args = [PY, "-u", str(ROOT / "scripts" / "run_research.py"),
                "--tag", "enh", "--top-n", str(top_n),
                "--horizon", str(frozen["horizon"]), "--horizons", horizons,
                "--rebalance", str(frozen["rebalance"]),
                "--train-stride", os.environ.get("TAI_ENH_STRIDE", "3"),
                "--leverages", "1", "--costs", "passive", "--save-scores",
                "--factor-model", "--estimated-spread",
                "--no-trade-band", str(frozen["no_trade_band"]),
                "--cost-penalty", str(frozen.get("cost_penalty", 1.0)),
                "--halflife-days", halflife]
        if use_metrics:
            args.append("--use-metrics")
        if run("enh_wf", args) != 0:
            return 1

    common = ["--scores", str(scores), "--top-n", str(top_n),
              "--horizon", str(frozen["horizon"]),
              "--rebalance", str(frozen["rebalance"]),
              "--target-vol", str(frozen["target_vol"]),
              "--max-weight", str(frozen["max_weight"]),
              "--smooth-halflife", str(frozen["smooth_halflife"]),
              "--no-trade-band", str(frozen["no_trade_band"]),
              "--cost-penalty", str(frozen.get("cost_penalty", 1.0)),
              "--factor-model", "--estimated-spread",
              "--headline-cost", frozen["headline_cost_scenario"],
              "--target-monthly", str(TARGET),
              "--n-trials", str(frozen.get("n_configs_searched", 144))]
    run("enh_validate_dev", [PY, "-u", str(ROOT / "scripts" / "validate.py"),
                             "--tag", "enh_dev", "--max-date", CUTOFF] + common)
    run("enh_validate_holdout", [PY, "-u", str(ROOT / "scripts" / "validate.py"),
                                 "--tag", "enh_holdout", "--min-date", "2025-01-01",
                                 "--eval-from", CUTOFF] + common)
    log.info("enhanced pass complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
