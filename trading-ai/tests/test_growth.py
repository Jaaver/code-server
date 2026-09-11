"""The leverage arithmetic behind the verdict."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.evaluation.growth import (drawdown_exceedance_probability, max_monthly_return,
                                   required_sharpe, verdict, vol_for_monthly_target)


def test_required_sharpe_matches_the_growth_bound():
    for target in (0.05, 0.10, 0.33, 0.50):
        s = required_sharpe(target)
        # at exactly the required Sharpe the target is reachable only at s = S
        assert abs(max_monthly_return(s) - target) < 1e-9
        assert abs(vol_for_monthly_target(s, target) - s) < 1e-6


def test_33_percent_needs_sharpe_above_2_6():
    s = required_sharpe(0.33)
    assert 2.60 < s < 2.63
    assert vol_for_monthly_target(2.0, 0.33) is None       # unreachable at Sharpe 2
    assert vol_for_monthly_target(3.0, 0.33) is not None    # reachable at Sharpe 3


def test_lower_root_is_returned():
    s, m = 3.0, 0.33
    v = vol_for_monthly_target(s, m)
    g = 12 * math.log1p(m)
    assert abs(s * v - v * v / 2 - g) < 1e-9
    other = s + math.sqrt(s * s - 2 * g)
    assert v < other


def test_verdict_reports_the_shortfall_when_unreachable():
    out = verdict(2.0, 0.33, vol_per_unit_gross=0.15)
    assert out["reachable"] is False
    assert out["sharpe_shortfall"] > 0.6
    assert out["max_monthly_at_growth_optimal_leverage"] < 0.33


def test_drawdown_probability_is_monotone():
    p30 = drawdown_exceedance_probability(3.0, 1.5, 0.30)
    p50 = drawdown_exceedance_probability(3.0, 1.5, 0.50)
    p80 = drawdown_exceedance_probability(3.0, 1.5, 0.80)
    assert p30 > p50 > p80
    # more leverage at the same Sharpe means deeper drawdowns are likelier
    assert (drawdown_exceedance_probability(3.0, 2.5, 0.50)
            > drawdown_exceedance_probability(3.0, 1.0, 0.50))
