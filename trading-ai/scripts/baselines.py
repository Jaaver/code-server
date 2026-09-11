#!/usr/bin/env python3
"""Single-feature analytic baselines, run through the identical simulator.

If the machine-learning model cannot beat a one-line reversal rule after costs, it
is not earning its complexity.  These baselines are also the cleanest evidence that
the *data* contains exploitable structure independent of any model fitting.
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
from tai.features.core import _cs_rank, _zs
from tai.features.labels import forward_return
from tai.research import pipeline as P

log = logging.getLogger("baselines")
EPS = 1e-12


def signals(ds) -> dict[str, pd.DataFrame]:
    p, mask = ds.panel, ds.mask
    c, o = p["close"], p["open"]
    qv, tbq = p["quote_volume"], p["taker_buy_quote_volume"]
    vol = ds.aux["vol_ref"]
    beta = ds.aux["beta"]
    ret1 = np.log(c / c.shift(1))
    mkt = ret1.where(mask).median(axis=1)
    out = {}
    for k in (1, 4, 24, 72):
        r = np.log(c / c.shift(k))
        resid = r.sub(beta.mul(mkt.rolling(k, min_periods=k).sum(), axis=0))
        out[f"reversal_{k}h"] = -_cs_rank((resid / (vol * np.sqrt(k) + EPS)), mask)
    ofi = (2 * tbq - qv) / (qv + EPS)
    out["ofi_4h_reversal"] = -_cs_rank(ofi.rolling(4, min_periods=2).mean(), mask)
    out["ofi_4h_momentum"] = _cs_rank(ofi.rolling(4, min_periods=2).mean(), mask)
    fr = p["funding_last"]
    out["funding_carry"] = -_cs_rank(_zs(fr, 24 * 30), mask)
    out["volume_shock_reversal"] = -_cs_rank(
        _zs(np.log1p(qv), 24 * 14) * np.sign(ret1), mask)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baselines")
    ap.add_argument("--max-date", default=None)
    ap.add_argument("--min-date", default="2021-02-01")
    ap.add_argument("--top-n", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--rebalance", type=int, default=4)
    ap.add_argument("--smooth-halflife", type=float, default=0.0)
    ap.add_argument("--factor-model", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ucfg = UniverseConfig(top_n=args.top_n)
    scfg = StrategyConfig(label_horizon=args.horizon, rebalance_every=args.rebalance)
    ds = P.build_context(ucfg, scfg, max_date=args.max_date, min_date=args.min_date)

    fwd = forward_return(ds.panel["open"], args.horizon).where(ds.mask)
    resid = fwd.sub(ds.aux["beta"].mul(fwd.median(axis=1), axis=0))
    flat = resid.stack(future_stack=True)
    flat.index.names = ["ts", "symbol"]

    out = {}
    for name, sig in signals(ds).items():
        s_long = sig.stack(future_stack=True)
        s_long.index.names = ["ts", "symbol"]
        s_long = s_long.dropna()
        ic = M.information_coefficient(s_long, flat.reindex(s_long.index))
        res, _ = P.backtest_scores(ds, s_long, scfg, "base", leverage=1.0,
                                   factor_model=args.factor_model,
                                   smooth_halflife=args.smooth_halflife)
        res = P.trim_to_oos(res, s_long)
        st = M.summarise(res, ds.bars_per_day, n_trials=10)
        out[name] = {"ic": ic, "backtest": st}
        log.info("%-22s IC=%+.4f  sharpe=%5.2f  monthly=%+6.2f%%  turn/day=%5.2f  dd=%.1f%%",
                 name, ic.get("ic_mean", np.nan), st["sharpe"],
                 100 * st.get("geom_monthly", np.nan), st["daily_turnover"],
                 100 * st["max_drawdown"])
    P.save_json(out, RESULTS_DIR / args.tag / "baselines.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
