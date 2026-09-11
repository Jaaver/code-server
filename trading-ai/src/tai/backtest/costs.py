"""Data-driven transaction-cost inputs.

A cross-sectional reversal signal is exactly the kind of thing that can be an
artefact of bid-ask bounce rather than a tradable edge, so charging a single
constant half-spread to every contract is not good enough: the spread has to come
from the data, per symbol and per period.

:func:`corwin_schultz_spread` implements the Corwin & Schultz (2012) high-low
estimator, which recovers the effective spread from two consecutive bars' high-low
ranges without needing quote data:

    beta  = E[(ln(H_t/L_t))^2 + (ln(H_t+1/L_t+1))^2]
    gamma = (ln(H_{t,t+1}/L_{t,t+1}))^2
    alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
    S     = 2 * (exp(alpha) - 1) / (1 + exp(alpha))

Negative estimates (which the estimator produces when volatility dominates) are
floored at zero and the series is smoothed, following the original paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
K = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz_spread(high: pd.DataFrame, low: pd.DataFrame, *,
                          smooth: int = 24 * 5, floor_bps: float = 0.5,
                          cap_bps: float = 120.0) -> pd.DataFrame:
    """Rolling effective *full* spread as a fraction of price, per symbol and bar.

    The result is strictly trailing: the value at bar ``t`` uses bars ``t-1`` and
    ``t`` and is then shifted one bar, so a decision made at ``t`` is priced with a
    spread estimate that was already observable.
    """
    h = high.where(high > 0)
    l = low.where(low > 0)
    hl = np.log(h / l) ** 2
    beta = (hl + hl.shift(1))
    # two-bar high and low (elementwise, not a groupby: the concat form costs
    # gigabytes on a multi-year, multi-hundred-symbol panel)
    h2 = np.maximum(h, h.shift(1))
    l2 = np.minimum(l, l.shift(1))
    gamma = np.log(h2 / l2) ** 2

    sqrt_beta = np.sqrt(beta.clip(lower=0))
    alpha = (np.sqrt(2.0) * sqrt_beta - sqrt_beta) / K - np.sqrt((gamma / K).clip(lower=0))
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = s.clip(lower=0.0)
    s = s.rolling(smooth, min_periods=max(6, smooth // 6)).median()
    s = s.shift(1)
    bps = (s * 1e4).clip(lower=floor_bps, upper=cap_bps)
    return bps.astype("float32")


def half_spread_panel(panel: dict[str, pd.DataFrame], *, smooth: int = 24 * 5,
                      floor_bps: float = 0.5, cap_bps: float = 120.0) -> pd.DataFrame:
    """Per-side (half) spread in basis points, ready for the simulator."""
    full = corwin_schultz_spread(panel["high"], panel["low"], smooth=smooth,
                                 floor_bps=floor_bps, cap_bps=cap_bps)
    return (full * 0.5).astype("float32")


def spread_summary(half_bps: pd.DataFrame, mask: pd.DataFrame) -> dict:
    """Distribution of the estimated half-spread across the tradable universe."""
    x = half_bps.where(mask).to_numpy(dtype="float32").ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {"mean_half_spread_bps": float(x.mean()),
            "median_half_spread_bps": float(np.median(x)),
            "p25_half_spread_bps": float(np.percentile(x, 25)),
            "p75_half_spread_bps": float(np.percentile(x, 75)),
            "p95_half_spread_bps": float(np.percentile(x, 95)),
            "n_obs": int(x.size)}


# --------------------------------------------------------------------------- #
# measured (rather than inferred) spreads
# --------------------------------------------------------------------------- #
def liquidity_half_spread_panel(panel: dict[str, pd.DataFrame], *, a: float, b: float,
                                bars_per_day: int = 24, lookback_days: int = 30,
                                floor_bps: float = 0.15, cap_bps: float = 25.0
                                ) -> pd.DataFrame:
    """Per-symbol, per-bar half spread from a liquidity model fitted to real trades.

    ``scripts/calibrate_spread.py`` measures the effective spread directly from the
    exchange's aggregated-trade archive (the aggressor side of every print is
    published) and fits

        half_spread_bps = a + b / sqrt(average daily notional in $m)

    Applying that fit to each symbol's own trailing dollar volume gives a spread
    panel that is grounded in measurement rather than in a high-low proxy, which is
    badly upward biased on assets this volatile.  The trailing window and the shift
    keep it usable by a decision taken at the bar in question.
    """
    qv = panel["quote_volume"]
    win = lookback_days * bars_per_day
    adv_musd = (qv.rolling(win, min_periods=win // 4).mean() * bars_per_day / 1e6).shift(1)
    hs = a + b / np.sqrt(adv_musd.clip(lower=0.25))
    return hs.clip(lower=floor_bps, upper=cap_bps).astype("float32")


def load_calibration(path) -> dict | None:
    """Read the coefficients produced by scripts/calibrate_spread.py."""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("model")
