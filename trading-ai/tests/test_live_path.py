"""The live code path must agree with the research path.

A live system sees a rolling window; research sees the whole history.  If those two
disagree, every backtest number is meaningless, so this is asserted rather than
assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.backtest.engine import ExecConfig
from tai.config import CostModel, UniverseConfig
from tai.data.panel import pit_universe
from tai.features.core import build_feature_panels, stack_features
from tai.live.paper_trader import LiveStrategy, PaperBroker, VolTargeter
from tests.test_no_lookahead import _synth_panel


class _LinearModel:
    """Deterministic stand-in for the fitted ensemble."""

    def __init__(self, features):
        self.features = list(features)
        rng = np.random.default_rng(0)
        self.w = rng.normal(size=len(self.features)).astype("float32")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = np.nan_to_num(X[self.features].to_numpy(dtype="float32"), nan=0.0)
        return pd.DataFrame({"lin": arr @ self.w}, index=X.index)


def test_rolling_window_scores_match_full_history():
    panel = _synth_panel(n_bars=5200, n_sym=10, seed=11)
    ucfg = UniverseConfig(top_n=10, min_adv_usd=0.0, adv_lookback_days=10, listing_burn_bars=24)
    mask = pit_universe(panel, ucfg, 24)
    F, _ = build_feature_panels(panel, mask, 24)
    X_full = stack_features(F, mask)
    model = _LinearModel(X_full.columns)

    strat = LiveStrategy(model=model, ucfg=ucfg, exec_cfg=ExecConfig(), bars_per_day=24)
    lookback = 2880
    for i in (4000, 4600, 5100):
        ts = mask.index[i]
        window = {k: v.iloc[i - lookback + 1:i + 1] for k, v in panel.items()}
        sc_live, mask_w, aux_w = strat.score_window(window)
        sel = X_full.index.get_level_values(0) == ts
        pred_full = model.predict(X_full[sel])
        from tai.models.ensemble import blend
        sc_full = blend(pred_full).droplevel(0)
        common = sc_live.index.intersection(sc_full.index)
        assert len(common) >= 8
        # cross-sectional standardisation makes the comparison scale-free
        a = sc_live.reindex(common).to_numpy()
        b = sc_full.reindex(common).to_numpy()
        assert np.corrcoef(a, b)[0, 1] > 0.995, f"live/research divergence at {ts}"
        assert np.max(np.abs(a - b)) < 0.05


def test_paper_broker_matches_explicit_arithmetic():
    cfg = ExecConfig(no_trade_band=0.0, max_participation=1.0)
    cost = CostModel(taker_fee_bps=5.0, maker_fee_bps=5.0, maker_ratio=0.0,
                     half_spread_bps=0.0, impact_coef=0.0)
    b = PaperBroker(cost=cost, cfg=cfg, equity=1_000_000.0)
    prices = {"A": 100.0, "B": 50.0}
    dv = {"A": 1e9, "B": 1e9}
    b.rebalance(pd.Timestamp("2024-01-01", tz="UTC"), {"A": 0.5, "B": -0.5}, prices, dv)
    # 1,000,000 of notional traded at 5 bps = 500 in fees
    assert abs(b.total_costs - 500.0) < 1e-6
    assert abs(b.equity - 999_500.0) < 1e-6
    assert abs(b.shares["A"] - 5000.0) < 1e-6
    assert abs(b.shares["B"] + 10000.0) < 1e-6
    # a 1% rise in A and 1% fall in B pays both legs
    pnl = b.mark({"A": 101.0, "B": 49.5}, prices)
    assert abs(pnl - (5000 * 1.0 + (-10000) * (-0.5))) < 1e-6
    # funding: long A pays when the rate is positive, short B receives
    eq0 = b.equity
    f = b.settle_funding({"A": 101.0, "B": 49.5}, {"A": 0.0001, "B": 0.0001})
    assert abs(f - (-(5000 * 101.0 * 1e-4) - (-10000 * 49.5 * 1e-4))) < 1e-6
    assert abs(b.equity - (eq0 + f)) < 1e-9


def test_vol_targeter_is_scale_invariant():
    vt = VolTargeter(0.20, 24 * 30, 24)
    rng = np.random.default_rng(2)
    unit = rng.normal(0, 0.40 / np.sqrt(24 * 365), 2000)
    vt.last_scale = 3.0
    for r in unit * 3.0:
        vt.observe(r)
    # realised unit vol is 40%, target 20% -> scalar 0.5
    assert abs(vt.scalar() - 0.5) < 0.06


def test_forward_returns_reconcile_with_the_equity_curve():
    """A forward run's return series must compound to its own equity curve.

    The first version computed the bar return before charging the rebalance's costs,
    so the reported Sharpe ratio was gross of cost while the equity curve compounded
    the net one -- a 15-percentage-point discrepancy over six months.
    """
    from tai.live.paper_trader import run_forward

    idx = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")
    syms = ["A", "B", "C", "D"]
    rng = np.random.default_rng(5)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.004, (len(idx), 4)), axis=0)),
                      index=idx, columns=syms)
    dv = {s: 1e12 for s in syms}

    class _Provider:
        def __call__(self, ts):
            i = idx.get_loc(ts)
            if i + 1 >= len(idx):
                return None
            nxt = idx[i + 1]
            return {"window": None,
                    "next_open": {s: float(px.loc[nxt, s]) for s in syms},
                    "next_volume": dv, "funding": {}}

    class _Strat:
        exec_cfg = ExecConfig()

        def score_window(self, window):
            # alternate the sign so the book actually trades every rebalance
            _Strat.flip = not getattr(_Strat, "flip", False)
            sgn = 1.0 if _Strat.flip else -1.0
            return (pd.Series([sgn, -sgn, sgn, -sgn], index=syms), None, None)

        def target_weights(self, scores, aux, mask, vol_scalar, leverage):
            return {s: float(v) * 0.25 for s, v in scores.items()}

    cost = CostModel(taker_fee_bps=10.0, maker_fee_bps=10.0, maker_ratio=0.0,
                     half_spread_bps=0.0, impact_coef=0.0)
    cfg = ExecConfig(no_trade_band=0.0, max_participation=1.0)
    broker = PaperBroker(cost=cost, cfg=cfg, equity=1_000_000.0)
    vt = VolTargeter(0.20, 24 * 30, 24)
    df = run_forward(_Provider(), _Strat(), broker, list(idx[:-1]),
                     rebalance_every=4, leverage=1.0, vol_targeter=vt, log_every=0)

    assert broker.total_costs > 0, "the test must actually incur costs"
    implied = float((1.0 + df["ret"]).prod())
    actual = float(df["equity"].iloc[-1] / 1_000_000.0)
    assert abs(implied - actual) < 2e-3, (
        f"return series compounds to {implied:.5f} but equity says {actual:.5f}")
