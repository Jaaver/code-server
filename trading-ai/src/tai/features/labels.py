"""Label construction.

The trading convention is fixed and shared by labels and backtest:

    decision at close of bar t  ->  fill at open of bar t+1
    holding period              ->  open of t+1 .. open of t+1+h

so the realisable forward return for a decision at ``t`` is

    fwd[t] = open[t + 1 + h] / open[t + 1] - 1

Labels are (a) market-residualised with a *trailing* beta, and (b) scaled by
trailing volatility so a single model can learn across assets and regimes, and
(c) cross-sectionally standardised, which is exactly the quantity a
dollar-neutral book needs to rank.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def forward_return(open_px: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Tradable forward simple return for a decision taken at each bar's close."""
    entry = open_px.shift(-1)
    exit_ = open_px.shift(-(1 + horizon))
    return (exit_ / entry - 1.0).astype("float32")


def build_labels(panel: dict[str, pd.DataFrame], mask: pd.DataFrame, horizon: int,
                 beta: pd.DataFrame, vol_ref: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return raw, residual, vol-scaled and cross-sectional-z label panels."""
    o = panel["open"]
    fwd = forward_return(o, horizon)
    fwd = fwd.where(mask)

    # market = cross-sectional median forward return over tradable names
    mkt = fwd.median(axis=1)
    resid = fwd.sub(beta.mul(mkt, axis=0))

    scale = (vol_ref * np.sqrt(horizon)).replace(0, np.nan)
    y_vol = (resid / (scale + EPS)).clip(-8, 8).astype("float32")

    mu = y_vol.mean(axis=1)
    sd = y_vol.std(axis=1)
    y_cs = (y_vol.sub(mu, axis=0).div(sd + EPS, axis=0)).clip(-4, 4).astype("float32")

    # rank label in [-1, 1]: fully outlier-robust ranking target
    y_rank = ((fwd.rank(axis=1, pct=True) - 0.5) * 2).astype("float32")

    return {"fwd": fwd.astype("float32"), "resid": resid.astype("float32"),
            "y_vol": y_vol, "y_cs": y_cs, "y_rank": y_rank,
            "mkt_fwd": mkt.astype("float32")}
