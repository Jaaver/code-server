#!/usr/bin/env python3
"""Fit the production ensemble on every bar strictly before a cutoff and save it."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.config import StrategyConfig, UniverseConfig
from tai.models.ensemble import fit_ensemble
from tai.models.store import save_model
from tai.research import pipeline as P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True, help="train on bars strictly before this date")
    ap.add_argument("--name", default=None)
    ap.add_argument("--top-n", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--embargo", type=int, default=48)
    ap.add_argument("--halflife-days", type=float, default=0.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ucfg = UniverseConfig(top_n=args.top_n)
    scfg = StrategyConfig(label_horizon=args.horizon, embargo_bars=args.embargo)
    # Purge the final horizon+embargo bars so no label overlaps the cutoff.
    purge = pd.Timedelta(hours=args.embargo + args.horizon + 1)
    ds = P.build_dataset(ucfg, scfg, max_date=args.cutoff)
    ts = ds.X.index.get_level_values(0)
    keep = ts <= (pd.Timestamp(args.cutoff, tz="UTC") - purge)
    X, y = ds.X[keep], ds.labels["y"][keep]
    ok = np.isfinite(y.to_numpy())
    X, y = X[ok], y[ok]
    if args.train_stride > 1:
        codes = pd.factorize(X.index.get_level_values(0), sort=True)[0]
        sel = (codes % args.train_stride) == 0
        X, y = X[sel], y[sel]
    sw = None
    if args.halflife_days > 0:
        age = (pd.Timestamp(args.cutoff, tz="UTC") -
               X.index.get_level_values(0)).total_seconds() / 86400.0
        sw = np.exp(-np.log(2) * age / args.halflife_days)
    logging.info("fitting on %d rows x %d features up to %s", len(X), X.shape[1],
                 X.index.get_level_values(0).max())
    ens = fit_ensemble(X, y, sample_weight=sw)
    name = args.name or f"ens_{args.cutoff.replace('-', '')}"
    p = save_model({"ensemble": ens, "ucfg": ucfg, "scfg": scfg,
                    "cutoff": args.cutoff, "features": list(X.columns)}, name)
    logging.info("saved %s", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
