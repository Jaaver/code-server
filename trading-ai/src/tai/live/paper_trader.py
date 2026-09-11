"""Forward paper-trading loop.

This is the production code path.  At every decision time it holds only a rolling
window of history in memory, recomputes features with the *same* functions the
research stack uses, asks the frozen model for scores, and hands target weights to
a broker that fills at the next bar's open with the same cost model as the
backtest.  Running it over a period the model was never fitted on is the strongest
test available short of sending real orders.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import ExecConfig, target_weights
from ..config import CostModel, UniverseConfig
from ..data.panel import pit_universe
from ..features.core import build_feature_panels, stack_features
from ..models.ensemble import blend

log = logging.getLogger(__name__)
EPS = 1e-12


@dataclass
class Position:
    shares: float = 0.0


@dataclass
class PaperBroker:
    """Mark-to-market book with explicit fills, fees, impact and funding."""

    cost: CostModel
    cfg: ExecConfig
    equity: float = 1_000_000.0
    shares: dict = field(default_factory=dict)
    ledger: list = field(default_factory=list)
    total_costs: float = 0.0
    total_funding: float = 0.0

    def mark(self, prices: dict[str, float], prev_prices: dict[str, float]) -> float:
        pnl = 0.0
        for s, sh in self.shares.items():
            p0, p1 = prev_prices.get(s), prices.get(s)
            if sh and p0 and p1 and np.isfinite(p0) and np.isfinite(p1):
                pnl += sh * (p1 - p0)
        self.equity += pnl
        return pnl

    def settle_funding(self, prices: dict[str, float], rates: dict[str, float]) -> float:
        f = 0.0
        if not self.cost.funding_enabled:
            return 0.0
        for s, sh in self.shares.items():
            r, p = rates.get(s), prices.get(s)
            if sh and r and p and np.isfinite(r) and np.isfinite(p):
                f -= sh * p * r
        self.equity += f
        self.total_funding += f
        return f

    def rebalance(self, ts, target_w: dict[str, float], prices: dict[str, float],
                  bar_dollar_volume: dict[str, float]) -> dict:
        """Trade toward ``target_w`` (fractions of equity) at ``prices``."""
        if self.equity <= 0:
            return {"ts": ts, "blown": True}
        traded = 0.0
        cost = 0.0
        orders = []
        names = set(target_w) | {s for s, sh in self.shares.items() if sh}
        for s in sorted(names):
            px = prices.get(s)
            if not px or not np.isfinite(px) or px <= 0:
                continue
            cur_w = self.shares.get(s, 0.0) * px / self.equity
            tgt = target_w.get(s, 0.0)
            dw = tgt - cur_w
            if abs(dw) < self.cfg.no_trade_band:
                continue
            notional = dw * self.equity
            dv = bar_dollar_volume.get(s, 0.0) or 0.0
            cap = self.cfg.max_participation * dv
            if cap > 0 and abs(notional) > cap:
                notional = np.sign(notional) * cap
            part = abs(notional) / dv if dv > 0 else 0.0
            bps = self.cost.linear_bps + self.cost.impact_coef * np.sqrt(min(part, 1.0))
            c = abs(notional) * bps / 1e4
            cost += c
            traded += abs(notional)
            self.shares[s] = self.shares.get(s, 0.0) + notional / px
            orders.append({"symbol": s, "notional": round(notional, 2),
                           "bps": round(bps, 2)})
        self.equity -= cost
        self.total_costs += cost
        rec = {"ts": str(ts), "equity": self.equity, "traded_notional": traded,
               "cost": cost, "n_orders": len(orders), "orders": orders}
        self.ledger.append(rec)
        return rec

    def gross(self, prices: dict[str, float]) -> float:
        g = sum(abs(sh) * prices.get(s, 0.0) for s, sh in self.shares.items()
                if np.isfinite(prices.get(s, np.nan)))
        return g / max(self.equity, EPS)


@dataclass
class LiveStrategy:
    """Stateless scorer: window of bars in, target weights out."""

    model: object
    ucfg: UniverseConfig
    exec_cfg: ExecConfig
    bars_per_day: int = 24
    blend_weights: dict | None = None

    def score_window(self, window: dict[str, pd.DataFrame]) -> tuple[pd.Series, pd.DataFrame, dict]:
        """Score the *last* bar of ``window``; returns (scores, mask, aux)."""
        mask = pit_universe(window, self.ucfg, self.bars_per_day)
        F, aux = build_feature_panels(window, mask, self.bars_per_day)
        X = stack_features(F, mask)
        ts_last = mask.index[-1]
        sel = X.index.get_level_values(0) == ts_last
        Xl = X[sel]
        if Xl.empty:
            return pd.Series(dtype="float64"), mask, aux
        preds = self.model.predict(Xl)
        sc = blend(preds, self.blend_weights)
        return sc.droplevel(0), mask, aux

    def target_weights(self, scores: pd.Series, aux: dict, mask: pd.DataFrame,
                       vol_scalar: float, leverage: float) -> dict[str, float]:
        ts_last = mask.index[-1]
        syms = list(mask.columns)
        sc = scores.reindex(syms).to_numpy(dtype="float64")
        vol = aux["vol_ann"].loc[ts_last].reindex(syms).to_numpy(dtype="float64")
        beta = aux["beta"].loc[ts_last].reindex(syms).to_numpy(dtype="float64")
        live = mask.loc[ts_last].reindex(syms).fillna(False).to_numpy() & np.isfinite(sc)
        w = target_weights(sc, vol, beta, live, self.exec_cfg)
        w = w * vol_scalar * leverage
        g = np.abs(w).sum()
        if g > self.exec_cfg.max_gross:
            w = w * (self.exec_cfg.max_gross / g)
        return {s: float(x) for s, x in zip(syms, w) if abs(x) > 1e-6}


class VolTargeter:
    """Trailing realised-volatility scaler for the unit-risk book."""

    def __init__(self, target_ann_vol: float, window: int, bars_per_day: int,
                 bounds=(0.25, 3.0)):
        self.target = target_ann_vol
        self.window = window
        self.ann = np.sqrt(bars_per_day * 365.0)
        self.bounds = bounds
        self.unit_returns: list[float] = []
        self.last_scale = 1.0

    def observe(self, ret: float):
        self.unit_returns.append(ret / max(self.last_scale, 1e-6))
        if len(self.unit_returns) > self.window:
            self.unit_returns = self.unit_returns[-self.window:]

    def scalar(self) -> float:
        r = np.asarray([x for x in self.unit_returns if x != 0.0])
        if r.size < 30:
            return 1.0
        vol = float(r.std() * self.ann)
        if vol <= 1e-4:
            return 1.0
        return float(np.clip(self.target / vol, *self.bounds))


def run_forward(window_provider, strategy: LiveStrategy, broker: PaperBroker,
                timestamps, *, rebalance_every: int, leverage: float,
                vol_targeter: VolTargeter, log_every: int = 200) -> pd.DataFrame:
    """Step through ``timestamps`` sequentially.

    ``window_provider(ts)`` must return a dict of wide panels containing bars with
    close time at or before ``ts`` only.  Execution happens at the open of the bar
    following ``ts``, which the provider exposes as ``next_open``/``next_volume``.
    """
    rows = []
    prev_prices: dict[str, float] = {}
    prev_equity = broker.equity
    for i, ts in enumerate(timestamps):
        ctx = window_provider(ts)
        if ctx is None:
            continue
        prices = ctx["next_open"]
        broker.mark(prices, prev_prices or prices)
        broker.settle_funding(prices, ctx.get("funding", {}))
        ret = broker.equity / max(prev_equity, EPS) - 1.0
        vol_targeter.observe(ret)
        if i % rebalance_every == 0 and broker.equity > 0:
            sc, mask, aux = strategy.score_window(ctx["window"])
            if not sc.empty:
                vs = vol_targeter.scalar()
                vol_targeter.last_scale = vs * leverage
                tw = strategy.target_weights(sc, aux, mask, vs, leverage)
                broker.rebalance(ts, tw, prices, ctx.get("next_volume", {}))
        rows.append({"ts": ts, "equity": broker.equity, "gross": broker.gross(prices),
                     "ret": ret})
        prev_prices = {k: v for k, v in prices.items() if np.isfinite(v)}
        prev_equity = broker.equity
        if log_every and i % log_every == 0:
            log.info("%s equity=%.0f gross=%.2f", ts, broker.equity, broker.gross(prices))
        if broker.equity <= 0:
            log.warning("account wiped out at %s", ts)
            break
    return pd.DataFrame(rows).set_index("ts")
