"""Statistical honesty tools: multiple-testing correction, bootstrap CIs, walk-forward splits."""
import numpy as np
from scipy import stats as sps


def probabilistic_sharpe(sr, n, skew=0.0, kurt=3.0, sr_benchmark=0.0):
    """Bailey & Lopez de Prado PSR: P(true SR > benchmark) given non-normal returns."""
    if n < 10:
        return np.nan
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2, 1e-12))
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(sps.norm.cdf(z))


def deflated_sharpe(sr, n, n_trials, sr_variance, skew=0.0, kurt=3.0):
    """DSR: PSR against the SR you'd expect from the BEST of n_trials random strategies.

    sr, n are per-observation Sharpe and sample size (same frequency).
    sr_variance is the variance of the Sharpe estimates across the trials tried.
    """
    if n_trials < 2 or sr_variance <= 0:
        return probabilistic_sharpe(sr, n, skew, kurt, 0.0)
    e = np.euler_gamma
    z1 = sps.norm.ppf(1 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(sr_variance) * ((1 - e) * z1 + e * z2)
    return probabilistic_sharpe(sr, n, skew, kurt, sr0)


def stationary_bootstrap(returns, n_boot=2000, mean_block=168, seed=0, stat=None):
    """Politis-Romano stationary bootstrap: preserves autocorrelation / vol clustering."""
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 50:
        return np.array([])
    p = 1.0 / mean_block
    if stat is None:
        stat = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(24 * 365) if x.std() > 0 else 0.0
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for k in range(n):
            idx[k] = i
            if rng.random() < p:
                i = rng.integers(n)
            else:
                i = (i + 1) % n
        out[b] = stat(r[idx])
    return out


def walk_forward_splits(n, train=24 * 365, test=24 * 90, embargo=24 * 7, min_train=24 * 240):
    """Anchored-then-rolling walk-forward with an embargo gap between train and test.

    Yields (train_slice, test_slice). Training never sees a bar within `embargo`
    of the test window, which kills leakage through overlapping feature windows.
    """
    splits = []
    start = 0
    tr_end = max(train, min_train)
    while tr_end + embargo + test <= n:
        te_start = tr_end + embargo
        te_end = min(te_start + test, n)
        splits.append((slice(start, tr_end), slice(te_start, te_end)))
        tr_end = te_end
    return splits


def max_dd(equity):
    eq = np.asarray(equity, float)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def monthly_returns(equity, index):
    """Calendar-month returns from an hourly equity curve."""
    import pandas as pd
    s = pd.Series(equity, index=index)
    m = s.resample("ME").last()
    m0 = pd.concat([pd.Series([s.iloc[0]], index=[s.index[0]]), m])
    return m0.pct_change().dropna()


def risk_of_ruin(returns, threshold=-0.5, horizon=24 * 365, n_sim=5000, seed=0, mean_block=168):
    """Bootstrapped probability of drawing down past `threshold` within `horizon`."""
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 100:
        return np.nan
    n = len(r); p = 1.0 / mean_block
    hits = 0
    for _ in range(n_sim):
        idx = np.empty(horizon, dtype=int)
        i = rng.integers(n)
        for k in range(horizon):
            idx[k] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        eq = np.exp(np.cumsum(r[idx]))
        if (eq / np.maximum.accumulate(eq) - 1).min() <= threshold:
            hits += 1
    return hits / n_sim
