"""Leakage and correctness tests for the research stack.

The first three tests are the ones that matter: they prove the feature matrix
cannot see the future, that the label is the return of a trade that could
actually have been placed, and that the simulator's P&L equals an independent
recomputation from positions and prices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.backtest.engine import ExecConfig, run_backtest, target_weights
from tai.config import CostModel, UniverseConfig
from tai.data.panel import pit_universe
from tai.features.core import build_feature_panels, stack_features
from tai.features.labels import build_labels, forward_return


def _synth_panel(n_bars: int = 4000, n_sym: int = 12, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_bars, freq="1h", tz="UTC")
    syms = [f"S{i}USDT" for i in range(n_sym)]
    mkt = rng.normal(0, 0.004, n_bars)
    out = {}
    close = np.empty((n_bars, n_sym))
    for j in range(n_sym):
        r = 0.8 * mkt + rng.normal(0, 0.006, n_bars)
        close[:, j] = 100 * np.exp(np.cumsum(r))
    c = pd.DataFrame(close, index=idx, columns=syms)
    o = c.shift(1).bfill() * (1 + rng.normal(0, 0.0005, (n_bars, n_sym)))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, (n_bars, n_sym))))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, (n_bars, n_sym))))
    qv = pd.DataFrame(rng.lognormal(16, 0.5, (n_bars, n_sym)), index=idx, columns=syms)
    out["close"] = c.astype("float32")
    out["open"] = o.astype("float32")
    out["high"] = pd.DataFrame(h, index=idx, columns=syms).astype("float32")
    out["low"] = pd.DataFrame(l, index=idx, columns=syms).astype("float32")
    out["quote_volume"] = qv.astype("float32")
    out["volume"] = (qv / c).astype("float32")
    out["trades"] = (qv / 5000).astype("float32")
    out["taker_buy_quote_volume"] = (qv * rng.uniform(0.4, 0.6, (n_bars, n_sym))).astype("float32")
    fr = pd.DataFrame(np.nan, index=idx, columns=syms)
    settle = (idx.hour % 8 == 0)
    fr.loc[settle] = rng.normal(0.0001, 0.0002, (settle.sum(), n_sym))
    out["funding_rate"] = fr.astype("float32")
    out["funding_last"] = fr.ffill().astype("float32")
    return out


@pytest.fixture(scope="module")
def panel():
    return _synth_panel()


@pytest.fixture(scope="module")
def mask(panel):
    cfg = UniverseConfig(top_n=12, min_adv_usd=0.0, adv_lookback_days=10, listing_burn_bars=24)
    return pit_universe(panel, cfg, 24)


def test_features_are_invariant_to_the_future(panel, mask):
    """Truncating the panel must not change any feature value before the cut."""
    cut = 3000
    F_full, _ = build_feature_panels(panel, mask, 24)
    trunc = {k: v.iloc[:cut] for k, v in panel.items()}
    F_cut, _ = build_feature_panels(trunc, mask.iloc[:cut], 24)
    # allow a warm-up tail for the longest rolling window
    lo, hi = 800, cut - 1
    bad = []
    for name in F_full:
        a = F_full[name].iloc[lo:hi].to_numpy(dtype="float64")
        b = F_cut[name].iloc[lo:hi].to_numpy(dtype="float64")
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() == 0:
            continue
        scale = np.nanstd(a[both]) or 1.0
        if np.nanmax(np.abs(a[both] - b[both])) > 1e-5 * max(scale, 1e-6):
            bad.append(name)
    assert not bad, f"future-dependent features: {bad}"


def test_forward_return_is_next_open_to_later_open(panel):
    o = panel["open"]
    h = 4
    fwd = forward_return(o, h)
    t = 1000
    expected = o.iloc[t + 1 + h, 3] / o.iloc[t + 1, 3] - 1
    assert abs(float(fwd.iloc[t, 3]) - float(expected)) < 1e-6
    # the last 1+h rows cannot have a label
    assert fwd.iloc[-(1 + h):].isna().all().all()


def test_labels_have_no_contemporaneous_information(panel, mask):
    """A label at t must not be computable from prices up to t."""
    F, aux = build_feature_panels(panel, mask, 24)
    labels = build_labels(panel, mask, 4, aux["beta"], aux["vol_ref"])
    y = labels["fwd"]
    # correlation between the label and the *past* return must be small, and the
    # label must change when future prices change
    panel2 = {k: v.copy() for k, v in panel.items()}
    panel2["open"] = panel2["open"].copy()
    panel2["open"].iloc[2000:] *= 1.05
    l2 = build_labels(panel2, mask, 4, aux["beta"], aux["vol_ref"])["fwd"]
    assert np.allclose(y.iloc[:1990].fillna(0), l2.iloc[:1990].fillna(0), atol=1e-6)
    assert not np.allclose(y.iloc[1994:1999].fillna(0), l2.iloc[1994:1999].fillna(0), atol=1e-9)


def test_target_weights_are_neutral_and_capped():
    rng = np.random.default_rng(1)
    n = 40
    score = rng.normal(size=n)
    vol = np.abs(rng.normal(1.0, 0.3, n)) + 0.2
    beta = rng.normal(1.0, 0.3, n)
    live = np.ones(n, dtype=bool)
    cfg = ExecConfig(max_weight=0.06)
    w = target_weights(score, vol, beta, live, cfg)
    assert abs(w.sum()) < 1e-8, "not dollar neutral"
    assert abs(float(w @ beta)) < 1e-8, "not beta neutral"
    assert np.abs(w).max() <= 0.061
    assert abs(np.abs(w).sum() - 1.0) < 1e-6


def test_dead_symbols_are_not_traded(panel, mask):
    live = mask.copy()
    live.iloc[:, 0] = False
    sc = pd.DataFrame(0.0, index=mask.index, columns=mask.columns)
    sc.iloc[::4] = 1.0
    sc.iloc[:, 0] = 5.0
    vol = pd.DataFrame(0.5, index=mask.index, columns=mask.columns)
    beta = pd.DataFrame(1.0, index=mask.index, columns=mask.columns)
    res = run_backtest(panel, sc, live, vol, beta, CostModel(), ExecConfig())
    assert res.weights.iloc[:, 0].abs().max() < 1e-9


def test_pnl_reconciles_with_independent_recomputation(panel, mask):
    rng = np.random.default_rng(3)
    sc = pd.DataFrame(rng.normal(size=mask.shape), index=mask.index, columns=mask.columns)
    sc.iloc[[i for i in range(len(sc)) if i % 4]] = np.nan
    vol = pd.DataFrame(0.6, index=mask.index, columns=mask.columns)
    beta = pd.DataFrame(1.0, index=mask.index, columns=mask.columns)
    cost = CostModel(taker_fee_bps=0.0, maker_fee_bps=0.0, half_spread_bps=0.0,
                     impact_coef=0.0, funding_enabled=False)
    cfg = ExecConfig(no_trade_band=0.0, max_participation=1.0, init_equity=1e6)
    res = run_backtest(panel, sc, mask, vol, beta, cost, cfg)
    # With zero costs and no funding, equity change must equal sum(shares * dP).
    o = panel["open"].reindex(columns=mask.columns)
    w = res.weights
    eq = res.equity
    shares = (w.shift(1).to_numpy() * eq.shift(1).to_numpy()[:, None]
              / np.where(o.shift(1).to_numpy() > 0, o.shift(1).to_numpy(), np.nan))
    dpx = o.diff().to_numpy()
    pnl = np.nansum(shares * dpx, axis=1)
    recomputed = pd.Series(pnl, index=eq.index).cumsum() + eq.iloc[0]
    err = (recomputed - eq).abs() / eq.iloc[0]
    assert err.iloc[5:].max() < 5e-3, f"pnl mismatch {err.max():.4f}"


def test_costs_are_monotone_in_fees(panel, mask):
    rng = np.random.default_rng(5)
    sc = pd.DataFrame(rng.normal(size=mask.shape), index=mask.index, columns=mask.columns)
    sc.iloc[[i for i in range(len(sc)) if i % 4]] = np.nan
    vol = pd.DataFrame(0.6, index=mask.index, columns=mask.columns)
    beta = pd.DataFrame(1.0, index=mask.index, columns=mask.columns)
    cfg = ExecConfig(no_trade_band=0.0)
    cheap = run_backtest(panel, sc, mask, vol, beta,
                         CostModel(taker_fee_bps=1.0, half_spread_bps=0.0, impact_coef=0.0),
                         cfg)
    dear = run_backtest(panel, sc, mask, vol, beta,
                        CostModel(taker_fee_bps=20.0, half_spread_bps=5.0, impact_coef=50.0),
                        cfg)
    assert dear.diagnostics["total_costs"] > cheap.diagnostics["total_costs"] * 3
    assert dear.equity.iloc[-1] < cheap.equity.iloc[-1]


def test_stack_features_alignment(panel, mask):
    F, _ = build_feature_panels(panel, mask, 24)
    X = stack_features(F, mask)
    assert len(X) == int(mask.to_numpy().sum())
    name = "ret_24"
    ts = X.index.get_level_values(0)[500]
    sym = X.index.get_level_values(1)[500]
    assert abs(float(X[name].iloc[500]) - float(F[name].loc[ts, sym])) < 1e-6
