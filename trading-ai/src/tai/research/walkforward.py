"""Purged, embargoed walk-forward training and out-of-sample prediction.

Every prediction returned by :func:`walk_forward` is produced by a model that
never saw (a) any bar at or after the test window's start, nor (b) any bar whose
label window overlaps the test window.  The purge removes ``horizon + 1`` bars and
the embargo a further ``embargo_bars`` from the end of each training set.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models.ensemble import blend, fit_ensemble

log = logging.getLogger(__name__)


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(index: pd.DatetimeIndex, *, min_train_bars: int, test_months: int,
               embargo_bars: int, horizon: int, train_months: int = 0) -> list[Fold]:
    """Expanding (or rolling) folds stepping forward ``test_months`` at a time."""
    idx = index.sort_values()
    first_test = idx[min(min_train_bars + embargo_bars + horizon, len(idx) - 1)]
    starts = pd.date_range(first_test.normalize(), idx[-1], freq=f"{test_months}MS", tz="UTC")
    folds = []
    for s in starts:
        e = s + pd.DateOffset(months=test_months)
        if s >= idx[-1]:
            break
        purge = pd.Timedelta(hours=embargo_bars + horizon + 1)
        tr_end = s - purge
        tr_start = idx[0] if train_months <= 0 else tr_end - pd.DateOffset(months=train_months)
        if (tr_end - tr_start) < pd.Timedelta(days=180):
            continue
        folds.append(Fold(max(tr_start, idx[0]), tr_end, s, min(e, idx[-1] + pd.Timedelta(hours=1))))
    return folds


def walk_forward(X: pd.DataFrame, y: pd.Series, folds: list[Fold], *,
                 train_stride: int = 1, variants: dict | None = None,
                 sample_halflife_days: float = 0.0, blend_weights: dict | None = None,
                 return_per_model: bool = False):
    """Fit per fold, predict the fold's test window, and concatenate OOS scores."""
    ts = X.index.get_level_values(0)
    preds = []
    per_model = []
    for i, f in enumerate(folds):
        tr = (ts >= f.train_start) & (ts <= f.train_end)
        te = (ts >= f.test_start) & (ts < f.test_end)
        if tr.sum() < 50_000 or te.sum() == 0:
            log.info("fold %d skipped (train=%d test=%d)", i, tr.sum(), te.sum())
            continue
        Xtr, ytr = X[tr], y[tr]
        if train_stride > 1:
            keep = np.zeros(len(Xtr), dtype=bool)
            uts = Xtr.index.get_level_values(0)
            codes = pd.factorize(uts, sort=True)[0]
            keep = (codes % train_stride) == 0
            Xtr, ytr = Xtr[keep], ytr[keep]
        ok = np.isfinite(ytr.to_numpy())
        Xtr, ytr = Xtr[ok], ytr[ok]
        sw = None
        if sample_halflife_days > 0:
            age_days = (f.train_end - Xtr.index.get_level_values(0)).total_seconds() / 86400.0
            sw = np.exp(-np.log(2) * age_days / sample_halflife_days).astype("float64")
        log.info("fold %d: train %s..%s (%d rows) -> test %s..%s (%d rows)", i,
                 f.train_start.date(), f.train_end.date(), len(Xtr),
                 f.test_start.date(), f.test_end.date(), int(te.sum()))
        ens = fit_ensemble(Xtr, ytr, sample_weight=sw, variants=variants)
        pm = ens.predict(X[te])
        preds.append(blend(pm, blend_weights))
        if return_per_model:
            per_model.append(pm)
        del Xtr, ytr, ens
        gc.collect()
    if not preds:
        raise RuntimeError("no usable folds")
    score = pd.concat(preds).sort_index()
    score = score[~score.index.duplicated(keep="last")]
    if return_per_model:
        pmdf = pd.concat(per_model).sort_index()
        pmdf = pmdf[~pmdf.index.duplicated(keep="last")]
        return score, pmdf
    return score


def scores_to_wide(score: pd.Series, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """Long (ts, symbol) -> wide [time, symbol] score panel."""
    w = score.unstack(level=1)
    return w.reindex(index=index, columns=columns)


def apply_rebalance_schedule(scores_wide: pd.DataFrame, every: int, phase: int = 0) -> pd.DataFrame:
    """Blank out scores on bars that are not decision bars."""
    if every <= 1:
        return scores_wide
    pos = np.arange(len(scores_wide))
    keep = (pos % every) == (phase % every)
    out = scores_wide.copy()
    out.loc[~keep] = np.nan
    return out
