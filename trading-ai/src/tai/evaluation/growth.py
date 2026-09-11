"""The arithmetic that connects a Sharpe ratio to a compound monthly return.

Leverage does not create edge, it rescales it, and the rescaling is not linear:
the volatility drag grows with the square of leverage while the expected return
grows only linearly.  For a book run at annualised volatility ``s`` with
annualised Sharpe ``S``, the expected *log* growth rate is

    g(s) = S * s - s^2 / 2

which is maximised at ``s = S`` with ``g = S^2 / 2``.  Two consequences decide
whether a monthly-return target is reachable at all:

* a target monthly growth ``m`` needs ``S >= sqrt(24 * ln(1 + m))`` -- no amount
  of leverage substitutes for Sharpe;
* once ``S`` clears that bar, the target is hit at the *lower* of the two
  volatilities solving ``g(s) = 12 ln(1 + m)``, and the gross notional that
  volatility implies is what the venue actually has to carry.

These are properties of the growth rate, not of any particular strategy, so the
verdict on a monthly-return target is computed here rather than asserted in prose.
"""
from __future__ import annotations

import math

MONTHS = 12.0


def max_log_growth(sharpe: float) -> float:
    """Highest achievable expected log growth per year, over all leverages."""
    return sharpe * sharpe / 2.0


def max_monthly_return(sharpe: float) -> float:
    """Highest achievable compound monthly return, at the growth-optimal leverage."""
    return math.expm1(max_log_growth(sharpe) / MONTHS)


def required_sharpe(monthly_target: float) -> float:
    """Minimum annualised Sharpe for a monthly target to be reachable at any leverage."""
    return math.sqrt(2.0 * MONTHS * math.log1p(monthly_target))


def vol_for_monthly_target(sharpe: float, monthly_target: float) -> float | None:
    """Annualised volatility that compounds to ``monthly_target``; None if unreachable.

    Returns the smaller of the two roots -- the same growth rate is available at a
    higher volatility, but only with strictly more risk, so it is never the right
    operating point.
    """
    g = MONTHS * math.log1p(monthly_target)
    disc = sharpe * sharpe - 2.0 * g
    if disc < -1e-12:
        return None
    return sharpe - math.sqrt(max(disc, 0.0))


def gross_for_vol(target_vol: float, vol_per_unit_gross: float) -> float:
    """Gross notional (as a multiple of equity) needed to reach a volatility."""
    if vol_per_unit_gross <= 0:
        return float("nan")
    return target_vol / vol_per_unit_gross


def drawdown_exceedance_probability(sharpe: float, vol: float, depth: float) -> float:
    """P(ever drawing down by ``depth``) for a geometric Brownian book.

    For drift ``mu = S*s`` and volatility ``s``, the probability of the equity
    curve ever falling to a fraction ``(1 - depth)`` of its running maximum is
    ``(1 - depth) ** (2 * mu / s^2)``.  Real strategies have fatter tails than
    this, so treat it as a floor on the risk, not an estimate of it.
    """
    if vol <= 0 or not 0 < depth < 1:
        return float("nan")
    exponent = 2.0 * (sharpe * vol) / (vol * vol)
    return (1.0 - depth) ** exponent


def verdict(sharpe: float, monthly_target: float, vol_per_unit_gross: float) -> dict:
    """Everything needed to say whether a target is reachable, and at what cost."""
    need = required_sharpe(monthly_target)
    best = max_monthly_return(sharpe)
    vol = vol_for_monthly_target(sharpe, monthly_target)
    out = {
        "sharpe": sharpe,
        "monthly_target": monthly_target,
        "required_sharpe": need,
        "sharpe_shortfall": need - sharpe,
        "max_monthly_at_growth_optimal_leverage": best,
        "growth_optimal_vol": sharpe,
        "reachable": vol is not None,
    }
    if vol is not None:
        out.update({
            "required_ann_vol": vol,
            "required_gross": gross_for_vol(vol, vol_per_unit_gross),
            "fraction_of_growth_optimal_leverage": vol / sharpe if sharpe else float("nan"),
            "gbm_p_drawdown_30": drawdown_exceedance_probability(sharpe, vol, 0.30),
            "gbm_p_drawdown_50": drawdown_exceedance_probability(sharpe, vol, 0.50),
            "gbm_p_drawdown_80": drawdown_exceedance_probability(sharpe, vol, 0.80),
        })
    return out
