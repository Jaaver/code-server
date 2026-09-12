"""The chunked feature build must reproduce the single-pass build exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.config import UniverseConfig
from tai.data.panel import pit_universe
from tai.features.core import (aux_panels, build_feature_panels, build_features_chunked,
                               stack_features)
from tests.test_no_lookahead import _synth_panel  # noqa: E402


def test_chunked_matches_single_pass():
    panel = _synth_panel(n_bars=6000, n_sym=10, seed=4)
    cfg = UniverseConfig(top_n=10, min_adv_usd=0.0, adv_lookback_days=10, listing_burn_bars=24)
    mask = pit_universe(panel, cfg, 24)
    F, _ = build_feature_panels(panel, mask, 24)
    X_full = stack_features(F, mask)
    X_chunk, aux = build_features_chunked(panel, mask, 24, chunk=1500, warmup=2600,
                                          log_progress=False)
    common = X_full.index.intersection(X_chunk.index)
    # the first chunk cannot be warmed up, so compare from the first warmed bar on
    first_warm = mask.index[2600]
    sel = common[common.get_level_values(0) >= first_warm]
    assert len(sel) > 10_000
    a = X_full.loc[sel].to_numpy(dtype="float64")
    b = X_chunk.loc[sel].to_numpy(dtype="float64")
    both = np.isfinite(a) & np.isfinite(b)
    assert both.mean() > 0.9
    diff = np.abs(a[both] - b[both])
    scale = np.maximum(np.abs(a[both]), 1.0)
    assert float(np.max(diff / scale)) < 2e-3, f"max rel diff {np.max(diff / scale):.2e}"
    assert (np.isfinite(a) == np.isfinite(b)).mean() > 0.999


def test_aux_panels_match_feature_builder():
    panel = _synth_panel(n_bars=3000, n_sym=8, seed=6)
    cfg = UniverseConfig(top_n=8, min_adv_usd=0.0, adv_lookback_days=10, listing_burn_bars=24)
    mask = pit_universe(panel, cfg, 24)
    _, aux_ref = build_feature_panels(panel, mask, 24)
    aux = aux_panels(panel, mask, 24)
    for k in ("vol_ref", "vol_ann", "beta"):
        a = aux_ref[k].to_numpy(dtype="float64")
        b = aux[k].to_numpy(dtype="float64")
        both = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[both], b[both], rtol=1e-5, atol=1e-8), k


def test_chunked_aux_has_the_same_keys_as_aux_panels():
    """Downstream stages read aux['ret'] and aux['mkt_ret']; a narrower dict here
    surfaces as a KeyError two stages later, after an hour of training."""
    panel = _synth_panel(n_bars=4000, n_sym=8, seed=9)
    cfg = UniverseConfig(top_n=8, min_adv_usd=0.0, adv_lookback_days=10,
                         listing_burn_bars=24)
    mask = pit_universe(panel, cfg, 24)
    _, aux_chunk = build_features_chunked(panel, mask, 24, chunk=1500, warmup=2600,
                                         log_progress=False)
    aux_ref = aux_panels(panel, mask, 24)
    assert set(aux_ref).issubset(set(aux_chunk)), (
        f"chunked aux is missing {set(aux_ref) - set(aux_chunk)}")
    for k in ("vol_ann", "beta", "ret"):
        assert len(aux_chunk[k]) == len(mask)
    assert len(aux_chunk["mkt_ret"]) == len(mask)
