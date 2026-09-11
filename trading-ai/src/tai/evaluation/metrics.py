"""Performance and overfitting statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12


def ann_factor(bars_per_day: int) -> float:
    return bars_per_day * 365.0


def sharpe(returns: pd.Series, bars_per_day: int = 24) -> float:
    r = returns.dropna()
    r = r[r != 0] if (r == 0).mean() > 0.98 else r
    if len(r) < 10 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(ann_factor(bars_per_day)))


def sortino(returns: pd.Series, bars_per_day: int = 24) -> float:
    r = returns.dropna()
    dn = r[r < 0]
    if len(dn) < 5 or dn.std() == 0:
        return float("nan")
    return float(r.mean() / dn.std() * np.sqrt(ann_factor(bars_per_day)))


def max_drawdown(equity: pd.Series) -> float:
    e = equity.dropna()
    if e.empty:
        return float("nan")
    peak = e.cummax()
    return float((e / peak - 1.0).min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def cagr(equity: pd.Series) -> float:
    e = equity.dropna()
    if len(e) < 2 or e.iloc[0] <= 0 or e.iloc[-1] <= 0:
        return float("nan")
    years = (e.index[-1] - e.index[0]).total_seconds() / (365.25 * 86400)
    return float((e.iloc[-1] / e.iloc[0]) ** (1 / max(years, EPS)) - 1.0)


def monthly_returns(equity: pd.Series) -> pd.Series:
    """Calendar-month returns from an equity curve.

    The first partial month is seeded with the curve's opening value so the first
    month is a real return rather than a spurious 0%.
    """
    e = equity.dropna()
    if len(e) < 2:
        return pd.Series(dtype="float64")
    m = e.resample("ME").last()
    seed_idx = e.index[0].normalize() - pd.Timedelta(days=1)
    seed = pd.Series([e.iloc[0]], index=pd.DatetimeIndex([seed_idx], tz=e.index.tz))
    m = pd.concat([seed, m])
    m = m[~m.index.duplicated(keep="last")].sort_index()
    return (m / m.shift(1) - 1.0).dropna()


def monthly_stats(equity: pd.Series) -> dict:
    mr = monthly_returns(equity)
    if mr.empty:
        return {}
    geo = float(np.expm1(np.log1p(mr).mean()))
    return {
        "n_months": int(len(mr)),
        "mean_monthly": float(mr.mean()),
        "median_monthly": float(mr.median()),
        "geom_monthly": geo,
        "std_monthly": float(mr.std()),
        "pct_months_positive": float((mr > 0).mean()),
        "pct_months_ge_33": float((mr >= 0.33).mean()),
        "worst_month": float(mr.min()),
        "best_month": float(mr.max()),
    }


def deflated_sharpe(observed_sr: float, n_trials: int, n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0,
                    sr_variance_trials: float | None = None) -> float:
    """Bailey & Lopez de Prado deflated Sharpe ratio (probability the true SR > 0).

    ``observed_sr`` and the result are per-observation / non-annualised inputs are
    handled by passing the *annualised* SR together with ``n_obs`` in years-equivalent
    observations; we follow the original formulation using per-observation SR.
    """
    if not np.isfinite(observed_sr) or n_obs < 10:
        return float("nan")
    if sr_variance_trials is None:
        sr_variance_trials = 1.0 / max(n_obs - 1, 1)
    gamma = 0.5772156649
    e = np.sqrt(max(sr_variance_trials, EPS))
    z1 = stats.norm.ppf(1 - 1.0 / max(n_trials, 2))
    z2 = stats.norm.ppf(1 - 1.0 / (max(n_trials, 2) * np.e))
    sr0 = e * ((1 - gamma) * z1 + gamma * z2)
    num = (observed_sr - sr0) * np.sqrt(max(n_obs - 1, 1))
    den = np.sqrt(max(1 - skew * observed_sr + (kurt - 1) / 4.0 * observed_sr ** 2, EPS))
    return float(stats.norm.cdf(num / den))


def dsr_from_annual(sr_ann: float, n_trials: int, returns: pd.Series,
                    bars_per_day: int = 24) -> float:
    r = returns.dropna()
    n = len(r)
    if n < 30:
        return float("nan")
    sr_obs = sr_ann / np.sqrt(ann_factor(bars_per_day))
    return deflated_sharpe(sr_obs, n_trials, n, float(stats.skew(r)),
                           float(stats.kurtosis(r, fisher=False)))


def probabilistic_sharpe(sr_ann: float, returns: pd.Series, benchmark_ann: float = 0.0,
                         bars_per_day: int = 24) -> float:
    """P(true SR > benchmark) accounting for skew and kurtosis."""
    r = returns.dropna()
    n = len(r)
    if n < 30 or not np.isfinite(sr_ann):
        return float("nan")
    af = np.sqrt(ann_factor(bars_per_day))
    sr, b = sr_ann / af, benchmark_ann / af
    sk, ku = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    den = np.sqrt(max(1 - sk * sr + (ku - 1) / 4.0 * sr ** 2, EPS))
    return float(stats.norm.cdf((sr - b) * np.sqrt(n - 1) / den))


def block_bootstrap_sharpe(returns: pd.Series, n_boot: int = 2000, block: int = 24 * 7,
                           bars_per_day: int = 24, seed: int = 0) -> dict:
    """Stationary block bootstrap CI for the Sharpe ratio."""
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < block * 3:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    srs = np.empty(n_boot)
    af = np.sqrt(ann_factor(bars_per_day))
    for i in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        samp = np.concatenate([r[s:s + block] for s in starts])[:n]
        sd = samp.std()
        srs[i] = samp.mean() / sd * af if sd > 0 else 0.0
    return {"sr_mean": float(srs.mean()), "sr_p05": float(np.percentile(srs, 5)),
            "sr_p50": float(np.percentile(srs, 50)), "sr_p95": float(np.percentile(srs, 95)),
            "p_sr_gt_0": float((srs > 0).mean())}


def bootstrap_monthly(returns: pd.Series, n_boot: int = 2000, block: int = 24 * 7,
                      bars_per_day: int = 24, seed: int = 0) -> dict:
    """Block-bootstrap distribution of the geometric monthly return."""
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < block * 3:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    bars_month = bars_per_day * 30.44
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        samp = np.concatenate([r[s:s + block] for s in starts])[:n]
        lg = np.log1p(np.clip(samp, -0.95, None)).mean()
        out[i] = np.expm1(lg * bars_month)
    return {"monthly_mean": float(out.mean()), "monthly_p05": float(np.percentile(out, 5)),
            "monthly_p50": float(np.percentile(out, 50)),
            "monthly_p95": float(np.percentile(out, 95)),
            "p_monthly_ge_33": float((out >= 0.33).mean())}


def pbo_cscv(perf_matrix: np.ndarray, n_splits: int = 10) -> float:
    """Probability of Backtest Overfitting via combinatorially-symmetric CV.

    ``perf_matrix`` is ``(n_observations, n_configurations)`` of per-bar returns for
    each candidate configuration.  Returns the fraction of IS/OOS splits in which
    the IS-best configuration underperforms the OOS median.
    """
    from itertools import combinations

    T, Ncfg = perf_matrix.shape
    if Ncfg < 2:
        return float("nan")
    n_splits = max(4, min(n_splits, 12))
    if n_splits % 2:
        n_splits += 1
    bounds = np.linspace(0, T, n_splits + 1).astype(int)
    blocks = [perf_matrix[bounds[i]:bounds[i + 1]] for i in range(n_splits)]
    half = n_splits // 2
    losses = []
    for combo in combinations(range(n_splits), half):
        is_idx = list(combo)
        oos_idx = [i for i in range(n_splits) if i not in combo]
        is_r = np.concatenate([blocks[i] for i in is_idx])
        oos_r = np.concatenate([blocks[i] for i in oos_idx])
        with np.errstate(invalid="ignore", divide="ignore"):
            is_sr = np.nan_to_num(is_r.mean(0) / (is_r.std(0) + EPS))
            oos_sr = np.nan_to_num(oos_r.mean(0) / (oos_r.std(0) + EPS))
        best = int(np.argmax(is_sr))
        rank = stats.rankdata(oos_sr)[best] / Ncfg
        losses.append(rank < 0.5)
    return float(np.mean(losses))


def risk_of_ruin(returns: pd.Series, threshold: float = -0.5, horizon_months: int = 12,
                 n_paths: int = 5000, block: int = 24 * 7, bars_per_day: int = 24,
                 seed: int = 0) -> float:
    """Block-bootstrapped probability of ever breaching a drawdown threshold."""
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < block * 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    horizon = int(bars_per_day * 30.44 * horizon_months)
    n_blocks = int(np.ceil(horizon / block))
    hits = 0
    for _ in range(n_paths):
        starts = rng.integers(0, n - block, size=n_blocks)
        samp = np.concatenate([r[s:s + block] for s in starts])[:horizon]
        eq = np.cumprod(1.0 + np.clip(samp, -0.999, None))
        dd = eq / np.maximum.accumulate(eq) - 1.0
        if dd.min() <= threshold:
            hits += 1
    return hits / n_paths


def information_coefficient(pred: pd.Series, actual: pd.Series) -> dict:
    """Cross-sectional IC statistics for a (ts, symbol)-indexed prediction."""
    df = pd.concat([pred.rename("p"), actual.rename("a")], axis=1).dropna()
    if df.empty:
        return {}
    g = df.groupby(level=0)
    ic = g.apply(lambda x: x["p"].corr(x["a"], method="spearman") if len(x) > 5 else np.nan).dropna()
    pic = g.apply(lambda x: x["p"].corr(x["a"]) if len(x) > 5 else np.nan).dropna()
    return {"ic_mean": float(ic.mean()), "ic_std": float(ic.std()),
            "ic_ir": float(ic.mean() / (ic.std() + EPS) * np.sqrt(len(ic))),
            "ic_positive_frac": float((ic > 0).mean()),
            "pearson_ic_mean": float(pic.mean()), "n_periods": int(len(ic))}


def summarise(result, bars_per_day: int = 24, n_trials: int = 1) -> dict:
    eq, r = result.equity, result.returns
    md = max_drawdown(eq)
    sr = sharpe(r, bars_per_day)
    out = {
        "start": str(eq.index[0]), "end": str(eq.index[-1]),
        "final_equity_mult": float(eq.iloc[-1] / eq.iloc[0]) if eq.iloc[0] else float("nan"),
        "cagr": cagr(eq), "ann_vol": float(r.std() * np.sqrt(ann_factor(bars_per_day))),
        "sharpe": sr, "sortino": sortino(r, bars_per_day),
        "max_drawdown": md, "calmar": (cagr(eq) / abs(md)) if md and md < 0 else float("nan"),
        "avg_gross": float(result.gross.mean()), "max_gross": float(result.gross.max()),
        "avg_net_exposure": float(result.net_exposure.abs().mean()),
        "daily_turnover": float(result.turnover.sum() /
                                max((eq.index[-1] - eq.index[0]).days, 1)),
        "avg_positions": float(result.n_positions.mean()),
        "total_cost_frac_of_initial": float(result.diagnostics.get("total_costs", np.nan) /
                                           max(eq.iloc[0], EPS)),
        "funding_pnl_frac_of_initial": float(result.diagnostics.get("total_funding", np.nan) /
                                             max(eq.iloc[0], EPS)),
        "capacity_truncation_frac": float(result.diagnostics.get("truncation_frac", np.nan)),
        "blown_up": bool(result.blown_up),
        "psr": probabilistic_sharpe(sr, r, 0.0, bars_per_day),
        "dsr": dsr_from_annual(sr, n_trials, r, bars_per_day),
    }
    out.update(monthly_stats(eq))
    return out
