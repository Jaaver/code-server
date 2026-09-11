"""What Sharpe does a 33%/month target actually require?

For log-returns, a strategy with Sharpe S run at annualised volatility s earns
    mu_log = S*s - s^2/2
Maximising over s gives s* = S and mu_log_max = S^2/2 (the full-Kelly point).
Leverage beyond s* REDUCES compound growth: that is the ceiling no amount of
leverage can beat, so it tells us the minimum Sharpe the target implies.
"""
import numpy as np

TARGET_MONTHLY = 0.33
TARGET_LOG_ANNUAL = 12 * np.log(1 + TARGET_MONTHLY)


def growth(S, s):
    return S * s - 0.5 * s ** 2


def min_sharpe_for(target_log=TARGET_LOG_ANNUAL, kelly_frac=1.0):
    """Minimum Sharpe needed when running at kelly_frac * full-Kelly leverage."""
    # s = f*S  ->  mu = f*S^2 - f^2 S^2/2 = S^2 (f - f^2/2)
    k = kelly_frac - 0.5 * kelly_frac ** 2
    return np.sqrt(target_log / k)


def required_vol(S, target_log=TARGET_LOG_ANNUAL):
    """Volatility levels at which Sharpe S achieves the target (if any)."""
    disc = S ** 2 - 2 * target_log
    if disc < 0:
        return None
    return S - np.sqrt(disc), S + np.sqrt(disc)


def expected_max_dd(s, years=1.0, mu_log=None):
    """Rough expected max drawdown for a GBM with vol s (Magdon-Ismail approx)."""
    if mu_log is None:
        mu_log = 0.0
    # For driftless GBM the expected max DD over T grows ~ s*sqrt(T)*1.25
    return 1 - np.exp(-1.25 * s * np.sqrt(years))


if __name__ == "__main__":
    print(f"target: {TARGET_MONTHLY:.0%}/month = {np.exp(TARGET_LOG_ANNUAL)-1:.1%}/year "
          f"= {TARGET_LOG_ANNUAL:.3f} log/year\n")
    print("Minimum net Sharpe required, by Kelly fraction actually run:")
    for f in (1.0, 0.75, 0.5, 0.33, 0.25):
        S = min_sharpe_for(kelly_frac=f)
        print(f"  {f:>4.0%} Kelly : Sharpe >= {S:5.2f}   (run at vol {f*S:6.1%}, "
              f"1yr expected maxDD ~ {expected_max_dd(f*S):.0%})")
    print("\nWhat a given Sharpe can compound to at its OWN optimum (full Kelly):")
    for S in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        g = S ** 2 / 2
        print(f"  Sharpe {S:4.1f} -> max {np.exp(g)-1:>10.1%}/yr = "
              f"{np.exp(g/12)-1:>7.2%}/month   at vol {S:.0%}")
