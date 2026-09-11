"""Fold construction must purge and embargo correctly.

A training set that reaches within ``horizon + embargo`` bars of a test window
leaks, because the last training labels are drawn from the test window's own
returns.  This is the single most common way a walk-forward backtest is wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.research.walkforward import apply_rebalance_schedule, make_folds, scores_to_wide


def test_folds_purge_and_embargo():
    idx = pd.date_range("2020-01-01", periods=24 * 365 * 4, freq="1h", tz="UTC")
    horizon, embargo = 4, 48
    folds = make_folds(idx, min_train_bars=24 * 365, test_months=3,
                       embargo_bars=embargo, horizon=horizon)
    assert len(folds) >= 8
    for f in folds:
        gap = (f.test_start - f.train_end).total_seconds() / 3600
        assert gap >= embargo + horizon + 1, f"gap {gap}h too small"
        assert f.train_end < f.test_start
        assert f.test_start < f.test_end
    # test windows must tile forward without overlapping
    for a, b in zip(folds, folds[1:]):
        assert b.test_start >= a.test_end - pd.Timedelta(hours=1)
        assert b.train_end >= a.train_end


def test_expanding_vs_rolling_windows():
    idx = pd.date_range("2020-01-01", periods=24 * 365 * 4, freq="1h", tz="UTC")
    exp = make_folds(idx, min_train_bars=24 * 365, test_months=3, embargo_bars=48, horizon=4)
    roll = make_folds(idx, min_train_bars=24 * 365, test_months=3, embargo_bars=48,
                      horizon=4, train_months=12)
    assert all(f.train_start == idx[0] for f in exp)
    assert all(f.train_start > idx[0] for f in roll[1:])
    for f in roll:
        span_days = (f.train_end - f.train_start).days
        assert 330 <= span_days <= 400


def test_rebalance_schedule_blanks_non_decision_bars():
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    wide = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
    out = apply_rebalance_schedule(wide, every=4, phase=0)
    assert out.iloc[0].notna().all() and out.iloc[4].notna().all()
    assert out.iloc[1].isna().all() and out.iloc[3].isna().all()
    assert out.notna().all(axis=1).sum() == 25


def test_scores_to_wide_roundtrip():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    syms = ["A", "B", "C"]
    mi = pd.MultiIndex.from_product([idx[:5], syms], names=["ts", "symbol"])
    s = pd.Series(np.arange(15.0), index=mi)
    w = scores_to_wide(s, idx, syms)
    assert w.shape == (10, 3)
    assert float(w.loc[idx[2], "B"]) == 7.0
    assert w.iloc[5:].isna().all().all()
