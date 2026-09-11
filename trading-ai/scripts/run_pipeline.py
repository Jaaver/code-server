#!/usr/bin/env python3
"""Run the remaining research pipeline end to end, unattended.

Stages, in order, each skipped if its output already exists:

  1. wait for the development construction sweep to finish
  2. freeze the best construction found on development data
  3. rebuild the panel over the complete symbol universe
  4. walk-forward over the full history (models only ever see the past)
  5. validate on the development window and, separately, on the sealed holdout
  6. fit the production model at the cutoff and forward paper-trade the holdout
  7. assemble the report

Each stage logs to its own file under the cache directory.
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
from tai.freeze import freeze

log = logging.getLogger("pipeline")
PY = sys.executable
CUTOFF = "2025-07-01"
TARGET = 0.33


def run(name: str, args: list[str], *, env: dict | None = None) -> int:
    logfile = CACHE_DIR / f"{name}.log"
    log.info("=== %s: %s", name, " ".join(args[1:]))
    t0 = time.time()
    with open(logfile, "w") as fh:
        p = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT,
                           env={**os.environ, **(env or {})})
    log.info("=== %s finished rc=%d in %.1f min", name, p.returncode,
             (time.time() - t0) / 60)
    if p.returncode != 0:
        log.error("--- tail of %s ---", logfile)
        log.error("\n".join(logfile.read_text().splitlines()[-25:]))
    return p.returncode


def pick_best(sweep_path: Path, scenario: str) -> dict:
    rows = json.loads(sweep_path.read_text())
    cand = [r for r in rows if r["cost"] == scenario and isinstance(r["sharpe"], (int, float))
            and r["sharpe"] == r["sharpe"]]
    if not cand:
        cand = [r for r in rows if isinstance(r["sharpe"], (int, float))]
    best = max(cand, key=lambda r: r["sharpe"])
    return best


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    sweep = RESULTS_DIR / "dev01" / "sweep.json"

    # ---- 1. wait for the sweep ------------------------------------------ #
    waited = 0
    while not sweep.exists() and waited < 7200:
        time.sleep(30)
        waited += 30
    if not sweep.exists():
        log.error("sweep never produced %s", sweep)
        return 1

    # ---- 2. freeze the construction chosen on development data ---------- #
    frozen_path = ROOT / "configs" / "frozen.json"
    best = pick_best(sweep, "passive")
    n_configs = len(json.loads(sweep.read_text()))
    cfg = {"interval": "1h", "top_n": 120, "horizon": 4,
           "rebalance": int(best["reb"]), "smooth_halflife": float(best["smooth"]),
           "no_trade_band": float(best["band"]), "factor_model": True, "n_factors": 5,
           "target_vol": 0.20, "max_weight": 0.06, "embargo": 48, "test_months": 3,
           "train_stride": 2, "estimated_spread": True, "spread_method": "measured",
           "cost_penalty": 1.0,
           "headline_cost_scenario": "passive", "cutoff": CUTOFF,
           "n_configs_searched": n_configs}
    freeze(cfg, frozen_path,
           dev_metrics={k: best[k] for k in ("sharpe", "monthly", "vol", "dd", "gross",
                                             "turn", "cost")
                        if k in best},
           note=("construction chosen on the development window only "
                 "(data before %s), from %d configurations" % (CUTOFF, n_configs)))
    log.info("frozen config: %s", cfg)

    # ---- 3. rebuild the panel over every cached symbol ------------------- #
    if not (CACHE_DIR / "panel_full.done").exists():
        rc = run("panel_full", [PY, "-u", str(ROOT / "scripts" / "build_panel.py"),
                                "--interval", "1h"])
        if rc != 0:
            return rc
        (CACHE_DIR / "panel_full.done").write_text("ok")

    # ---- 4. full-history walk-forward ----------------------------------- #
    final_scores = RESULTS_DIR / "final" / "oos_scores.parquet"
    if not final_scores.exists():
        rc = run("final_wf", [PY, "-u", str(ROOT / "scripts" / "run_research.py"),
                              "--tag", "final", "--top-n", str(cfg["top_n"]),
                              "--horizon", str(cfg["horizon"]),
                              "--rebalance", str(cfg["rebalance"]),
                              "--train-stride", str(cfg["train_stride"]),
                              "--leverages", "1", "--costs", "passive",
                              "--save-scores", "--per-model"])
        if rc != 0:
            return rc

    # ---- 5. validation on development and on the sealed holdout ---------- #
    common = ["--scores", str(final_scores), "--top-n", str(cfg["top_n"]),
              "--horizon", str(cfg["horizon"]), "--rebalance", str(cfg["rebalance"]),
              "--target-vol", str(cfg["target_vol"]),
              "--max-weight", str(cfg["max_weight"]),
              "--smooth-halflife", str(cfg["smooth_halflife"]),
              "--no-trade-band", str(cfg["no_trade_band"]),
              "--cost-penalty", str(cfg["cost_penalty"]),
              "--factor-model", "--estimated-spread",
              "--headline-cost", cfg["headline_cost_scenario"],
              "--target-monthly", str(TARGET),
              "--n-trials", str(n_configs)]
    if not (RESULTS_DIR / "dev_final" / "validation.json").exists():
        run("validate_dev", [PY, "-u", str(ROOT / "scripts" / "validate.py"),
                             "--tag", "dev_final", "--max-date", CUTOFF] + common)
    if not (RESULTS_DIR / "holdout" / "validation.json").exists():
        run("validate_holdout", [PY, "-u", str(ROOT / "scripts" / "validate.py"),
                                 "--tag", "holdout", "--min-date", CUTOFF] + common)

    # ---- 6. production model + forward paper trading --------------------- #
    model_name = "prod_2025H1"
    if not (RESULTS_DIR / "models" / f"{model_name}.pkl").exists():
        run("train_final", [PY, "-u", str(ROOT / "scripts" / "train_final.py"),
                            "--cutoff", CUTOFF, "--name", model_name,
                            "--top-n", str(cfg["top_n"]),
                            "--horizon", str(cfg["horizon"]), "--train-stride", "2"])
    if (RESULTS_DIR / "models" / f"{model_name}.pkl").exists():
        run("paper_forward", [PY, "-u", str(ROOT / "scripts" / "paper_forward.py"),
                              "--model", model_name, "--start", CUTOFF,
                              "--rebalance", str(cfg["rebalance"]),
                              "--cost", "passive", "--tag", "forward",
                              "--target-vol", str(cfg["target_vol"])])

    # ---- 7. baselines, diagnostics and the report ------------------------ #
    if not (RESULTS_DIR / "baselines" / "baselines.json").exists():
        run("baselines", [PY, "-u", str(ROOT / "scripts" / "baselines.py"),
                          "--tag", "baselines", "--max-date", CUTOFF,
                          "--top-n", str(cfg["top_n"]),
                          "--horizon", str(cfg["horizon"]),
                          "--rebalance", str(cfg["rebalance"]), "--factor-model"])
    run("diagnostics", [PY, "-u", str(ROOT / "scripts" / "diagnostics.py"),
                        "--scores", str(final_scores), "--tag", "dev_final",
                        "--max-date", CUTOFF, "--top-n", str(cfg["top_n"]),
                        "--horizon", str(cfg["horizon"]),
                        "--rebalance", str(cfg["rebalance"])])
    run("report", [PY, "-u", str(ROOT / "scripts" / "make_report.py"),
                   "--dev-tag", "dev_final", "--holdout-tag", "holdout",
                   "--diag-tag", "dev_final", "--target", str(TARGET)])
    log.info("pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
