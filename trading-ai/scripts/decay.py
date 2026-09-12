#!/usr/bin/env python3
"""How the edge has changed over time, in correlation and in basis points.

Rank information coefficient is scale-free, so it can hold steady while the money
drains out of a strategy: profit is IC times cross-sectional dispersion, and in a
maturing market the dispersion falls. Reporting both per period is what separates
"the model stopped working" from "the opportunity shrank to the size of the fees".
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.config import RESULTS_DIR, StrategyConfig, UniverseConfig
from tai.evaluation import metrics as M
from tai.features.labels import forward_return
from tai.research import pipeline as P

log = logging.getLogger("decay")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tag", default="decay")
    ap.add_argument("--top-n", type=int, default=120)
    ap.add_argument("--horizons", default="4,12")
    ap.add_argument("--freq", default="H", choices=("H", "Y"),
                    help="H = half-year buckets, Y = calendar years")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    score = pd.read_parquet(args.scores)["score"]
    ds = P.build_context(UniverseConfig(top_n=args.top_n), StrategyConfig(label_horizon=4))
    out: dict = {}
    for h in [int(x) for x in args.horizons.split(",")]:
        fwd = forward_return(ds.panel["open"], h).where(ds.mask)
        resid = fwd.sub(ds.aux["beta"].mul(fwd.median(axis=1), axis=0))
        flat = resid.stack(future_stack=True)
        flat.index.names = ["ts", "symbol"]
        df = pd.concat([score.rename("s"), flat.reindex(score.index).rename("a")],
                       axis=1).dropna()
        ts = df.index.get_level_values(0)
        key = (ts.year.astype(str) if args.freq == "Y" else
               ts.year.astype(str) + "H" + ((ts.month > 6).astype(int) + 1).astype(str))
        rows = {}
        log.info("horizon %dh", h)
        for k in sorted(set(key)):
            sub = df[key == k]
            if len(sub) < 20_000:
                continue
            ic = M.information_coefficient(sub["s"], sub["a"])
            dec = sub.groupby(level=0)["s"].transform(
                lambda x: pd.qcut(x.rank(method="first"), 10, labels=False,
                                  duplicates="drop"))
            sp = sub.assign(d=dec).groupby("d")["a"].mean()
            spread = float((sp.loc[9] - sp.loc[0]) * 1e4) if {0, 9} <= set(sp.index) \
                else float("nan")
            disp = float(sub.groupby(level=0)["a"].std().mean() * 1e4)
            rows[k] = {**ic, "decile_spread_bps": spread,
                       "cross_sectional_dispersion_bps": disp}
            log.info("  %-7s IC=%+.4f t=%6.1f pos=%4.1f%% d10-d1=%7.1fbps disp=%7.1fbps",
                     k, ic["ic_mean"], ic["ic_ir"], 100 * ic["ic_positive_frac"],
                     spread, disp)
        out[str(h)] = rows
    P.save_json(out, RESULTS_DIR / args.tag / "decay.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
