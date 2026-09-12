"""Positioning features must have a chunk-invariant schema.

The open-interest archive starts later than the price history, so a chunk covering
2020 sees no metrics data at all.  If that chunk were allowed to define a narrower
feature schema than later chunks, every positioning feature would be silently
dropped from the design matrix -- which is exactly what happened once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.config import UniverseConfig
from tai.data.panel import pit_universe
from tai.features.core import build_feature_panels, build_features_chunked
from tests.test_no_lookahead import _synth_panel

METRIC_NAMES = ["sum_open_interest", "sum_open_interest_value",
                "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]


def _metrics(panel, start_frac: float):
    """Metrics panels that only carry data after ``start_frac`` of the sample."""
    idx, cols = panel["close"].index, panel["close"].columns
    rng = np.random.default_rng(3)
    cut = int(len(idx) * start_frac)
    out = {}
    for j, name in enumerate(METRIC_NAMES):
        vals = np.full((len(idx), len(cols)), np.nan, dtype="float32")
        tail = rng.lognormal(10 if j < 2 else 0.0, 0.4,
                             (len(idx) - cut, len(cols))).astype("float32")
        vals[cut:] = tail
        out[name] = pd.DataFrame(vals, index=idx, columns=cols)
    return out


@pytest.fixture(scope="module")
def setup():
    panel = _synth_panel(n_bars=6000, n_sym=8, seed=21)
    cfg = UniverseConfig(top_n=8, min_adv_usd=0.0, adv_lookback_days=10,
                         listing_burn_bars=24)
    return panel, pit_universe(panel, cfg, 24)


def test_metrics_add_features(setup):
    panel, mask = setup
    base, _ = build_feature_panels(panel, mask, 24)
    with_m, _ = build_feature_panels(panel, mask, 24, metrics=_metrics(panel, 0.0))
    added = set(with_m) - set(base)
    assert len(added) >= 20, f"only {len(added)} positioning features added"
    assert any(n.startswith("oi_chg") for n in added)
    assert any(n.startswith("cs_oi") for n in added)
    assert any("ls_" in n for n in added)


def test_schema_is_the_same_before_and_after_the_archive_starts(setup):
    """A slice with no metrics data must still produce the full column set."""
    panel, mask = setup
    m = _metrics(panel, 0.5)
    early = {k: v.iloc[:1000] for k, v in m.items()}
    late = {k: v.iloc[-1000:] for k, v in m.items()}
    f_early, _ = build_feature_panels({k: v.iloc[:1000] for k, v in panel.items()},
                                      mask.iloc[:1000], 24, metrics=early)
    f_late, _ = build_feature_panels({k: v.iloc[-1000:] for k, v in panel.items()},
                                     mask.iloc[-1000:], 24, metrics=late)
    assert list(f_early.keys()) == list(f_late.keys())
    # the early slice's positioning features exist but are empty
    assert f_early["oi_chg_24"].notna().to_numpy().sum() == 0
    assert f_late["oi_chg_24"].notna().to_numpy().sum() > 0


def test_chunked_build_keeps_positioning_features(setup):
    panel, mask = setup
    m = _metrics(panel, 0.5)
    X, _ = build_features_chunked(panel, mask, 24, chunk=1500, warmup=2600,
                                  log_progress=False, metrics=m)
    base, _ = build_feature_panels(panel, mask, 24)
    assert len(X.columns) > len(base), "positioning features were dropped"
    oi_cols = [c for c in X.columns if c.startswith("oi_") or "ls_" in c]
    assert len(oi_cols) >= 15
    # and they carry real values in the half of the sample the archive covers
    assert np.isfinite(X["oi_chg_24"].to_numpy()).sum() > 0
