"""Trailing statistical risk model used for neutralisation and ex-ante sizing.

A principal-component factor model fitted on a *trailing* window of returns
subsumes market beta and the de-facto sector structure of crypto (L1s, DeFi,
memecoins move together), which a single-beta hedge leaves exposed.  Loadings are
refreshed every ``step`` bars and held constant in between, mirroring what a live
system would do.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
EPS = 1e-12


def rolling_factor_model(ret: pd.DataFrame, mask: pd.DataFrame, *, n_factors: int = 5,
                         window: int = 24 * 45, step: int = 24 * 7,
                         min_obs: int = 24 * 10):
    """Fit PCA loadings on trailing returns at every ``step``-th bar.

    Returns ``(loadings, factor_var, idio_vol, step_of_bar)`` where

    * ``loadings``   ``[n_steps, n_symbols, n_factors]``
    * ``factor_var`` ``[n_steps, n_factors]`` per-bar factor variances
    * ``idio_vol``   ``[n_steps, n_symbols]`` per-bar residual vols
    * ``step_of_bar````[n_bars]`` index into the first three, always pointing at a
      fit that used only data strictly before the bar.
    """
    T, N = ret.shape
    starts = list(range(0, T, step))
    n_steps = len(starts)
    loadings = np.zeros((n_steps, N, n_factors), dtype="float32")
    factor_var = np.zeros((n_steps, n_factors), dtype="float32")
    idio = np.full((n_steps, N), np.nan, dtype="float32")
    R = ret.to_numpy(dtype="float32")
    Mk = mask.to_numpy()

    for si, t in enumerate(starts):
        lo = max(0, t - window)
        if t - lo < min_obs:
            continue
        block = R[lo:t]
        live = Mk[max(lo, t - step):t].any(axis=0) & (np.isfinite(block).sum(axis=0) > min_obs // 2)
        if live.sum() < n_factors * 4:
            continue
        sub = block[:, live]
        sub = np.where(np.isfinite(sub), sub, 0.0)
        sd = sub.std(axis=0)
        sd = np.where(sd > EPS, sd, 1.0)
        Z = sub / sd
        Z = Z - Z.mean(axis=0, keepdims=True)
        try:
            # economical SVD of the standardised return block
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        k = min(n_factors, Vt.shape[0])
        V = Vt[:k].T                           # (n_live, k) unit-norm loadings
        f = Z @ V                              # factor scores in units of sd
        fv = f.var(axis=0)
        resid = Z - f @ V.T
        rv = resid.var(axis=0)
        L = np.zeros((N, n_factors), dtype="float32")
        L[live, :k] = (V * sd[:, None]).astype("float32")
        loadings[si] = L
        factor_var[si, :k] = fv.astype("float32")
        idv = np.full(N, np.nan, dtype="float32")
        idv[live] = np.sqrt(np.maximum(rv, 0.0)) * sd
        idio[si] = idv

    step_of_bar = np.maximum(np.searchsorted(np.asarray(starts), np.arange(T), side="right") - 1, 0)
    return loadings, factor_var, idio, step_of_bar.astype("int32")


def neutralise(w: np.ndarray, extra_basis: np.ndarray | None, live: np.ndarray,
               dollar_neutral: bool = True) -> np.ndarray:
    """Project ``w`` off the constant vector and the columns of ``extra_basis``."""
    raw = []
    if dollar_neutral:
        raw.append(np.where(live, 1.0, 0.0))
    if extra_basis is not None:
        for j in range(extra_basis.shape[1]):
            v = np.where(live, np.nan_to_num(extra_basis[:, j], nan=0.0), 0.0)
            if np.abs(v).sum() > 0:
                raw.append(v)
    basis: list[np.ndarray] = []
    for v in raw:
        for u in basis:
            v = v - (v @ u) * u
        nv = float(np.sqrt(v @ v))
        if nv > 1e-9:
            basis.append(v / nv)
    for u in basis:
        w = w - (w @ u) * u
    return np.where(live, w, 0.0)


def factor_portfolio_vol(w: np.ndarray, L: np.ndarray, fvar: np.ndarray,
                         idio: np.ndarray, bars_per_year: float) -> float:
    """Annualised ex-ante vol under the factor model."""
    wl = w @ np.nan_to_num(L, nan=0.0)
    fv = float(np.sum(fvar * wl ** 2))
    iv = float(np.sum((w * np.nan_to_num(idio, nan=0.0)) ** 2))
    return float(np.sqrt(max(fv + iv, 0.0) * bars_per_year))


def smooth_scores(scores_wide: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """EWMA a decision-bar score panel across decision bars.

    Smoothing trades a little signal freshness for materially lower turnover, which
    is usually net-positive once costs are charged.
    """
    if halflife <= 0:
        return scores_wide
    dec = scores_wide.dropna(how="all")
    sm = dec.ewm(halflife=halflife, min_periods=1, ignore_na=True).mean()
    out = scores_wide.copy()
    out.loc[sm.index] = sm
    return out
