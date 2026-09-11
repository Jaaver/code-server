"""Bar-level perpetual-futures portfolio simulator.

Realism features
----------------
* decisions at bar close, fills at the **next bar's open** (no close-on-close fill);
* per-side exchange fees, half-spread, and square-root market impact driven by
  the trade's participation in that bar's actual dollar volume;
* a hard participation cap, so capacity limits truncate trades instead of
  silently assuming infinite liquidity;
* 8-hourly funding settled on the *realised* historical funding rate;
* share-based positions, so weights drift with price between rebalances;
* forced liquidation of delisted symbols at the last printed open, with
  penalty costs (removes delisting survivorship bias);
* maintenance-margin and bankruptcy checks at every bar;
* volatility targeting driven only by *trailing* realised strategy volatility.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CostModel
from .risk import factor_portfolio_vol, neutralise

log = logging.getLogger(__name__)
EPS = 1e-12


@dataclass
class ExecConfig:
    target_ann_vol: float = 0.20
    leverage: float = 1.0            # applied on top of vol targeting
    max_gross: float = 4.0           # gross notional / equity cap
    max_weight: float = 0.06         # per-name |weight| cap (fraction of equity)
    max_participation: float = 0.05  # max fraction of a bar's dollar volume we take
    bars_per_day: int = 24
    vol_target_window_days: int = 30
    vol_scalar_bounds: tuple = (0.25, 3.0)
    maintenance_margin: float = 0.0075
    dollar_neutral: bool = True
    beta_neutral: bool = True
    no_trade_band: float = 0.0015     # skip trades smaller than this in weight terms
    partial_adjust: float = 1.0       # 1.0 = go fully to target
    init_equity: float = 1_000_000.0
    delist_cost_mult: float = 3.0
    allow_bankruptcy_stop: bool = True


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    gross: pd.Series
    net_exposure: pd.Series
    turnover: pd.Series
    costs: pd.Series
    funding: pd.Series
    n_positions: pd.Series
    weights: pd.DataFrame
    vol_scalar: pd.Series
    trades_notional: pd.Series
    blown_up: bool = False
    blowup_time: object = None
    diagnostics: dict = field(default_factory=dict)


def _project_neutral(w: np.ndarray, beta: np.ndarray, live: np.ndarray,
                     dollar_neutral: bool, beta_neutral: bool) -> np.ndarray:
    """Orthogonalise the weight vector against the constant and beta vectors."""
    if not live.any():
        return w
    raw = []
    if dollar_neutral:
        raw.append(np.where(live, 1.0, 0.0))
    if beta_neutral:
        raw.append(np.where(live, np.nan_to_num(beta, nan=1.0), 0.0))
    # Gram-Schmidt: the constraint vectors are not mutually orthogonal, so a
    # sequential projection would re-introduce the component it had just removed.
    basis = []
    for v in raw:
        for u in basis:
            v = v - (v @ u) * u
        nv = float(np.sqrt(v @ v))
        if nv > 1e-9:
            basis.append(v / nv)
    for u in basis:
        w = w - (w @ u) * u
    return np.where(live, w, 0.0)


def target_weights(score: np.ndarray, vol: np.ndarray, beta: np.ndarray, live: np.ndarray,
                   cfg: ExecConfig) -> np.ndarray:
    """Map cross-sectional scores to risk-scaled, neutralised, capped weights."""
    s = np.where(live, np.nan_to_num(score, nan=0.0), 0.0)
    if not live.any() or np.all(s == 0):
        return np.zeros_like(s)
    n = live.sum()
    s = s - s[live].mean()
    sd = s[live].std()
    if sd > EPS:
        s = s / sd
    v = np.where(live & (vol > 0), vol, np.nan)
    med_v = np.nanmedian(v) if np.isfinite(v).any() else 1.0
    v = np.where(np.isfinite(v), v, med_v)
    v = np.clip(v, 0.25 * med_v, 6.0 * med_v)

    w = np.where(live, s / v, 0.0)
    w = _project_neutral(w, beta, live, cfg.dollar_neutral, cfg.beta_neutral)
    gross = np.abs(w).sum()
    if gross <= EPS:
        return np.zeros_like(w)
    w = w / gross                                     # unit gross
    cap = max(cfg.max_weight, 2.0 / max(n, 1))
    for _ in range(6):                                # iterate cap + renormalise
        w = np.clip(w, -cap, cap)
        w = _project_neutral(w, beta, live, cfg.dollar_neutral, cfg.beta_neutral)
        g = np.abs(w).sum()
        if g <= EPS:
            return np.zeros_like(w)
        w = w / g
        if np.abs(w).max() <= cap + 1e-9:
            break
    return w


def target_weights_factor(score: np.ndarray, idio_vol: np.ndarray, L: np.ndarray,
                          live: np.ndarray, cfg: ExecConfig) -> np.ndarray:
    """Factor-neutral, idiosyncratic-risk-scaled unit-gross weights.

    With a factor model in hand the correct denominator is idiosyncratic rather
    than total volatility, and the correct neutrality constraint is the whole
    loading matrix rather than a single market beta.
    """
    s = np.where(live, np.nan_to_num(score, nan=0.0), 0.0)
    if not live.any() or np.all(s == 0):
        return np.zeros_like(s)
    n = int(live.sum())
    s = s - s[live].mean()
    sd = s[live].std()
    if sd > EPS:
        s = s / sd
    v = np.where(live & (idio_vol > 0), idio_vol, np.nan)
    med = np.nanmedian(v) if np.isfinite(v).any() else 1.0
    v = np.where(np.isfinite(v), v, med)
    v = np.clip(v, 0.25 * med, 6.0 * med)
    w = np.where(live, s / v, 0.0)
    w = neutralise(w, L, live, cfg.dollar_neutral)
    g = np.abs(w).sum()
    if g <= EPS:
        return np.zeros_like(w)
    w = w / g
    cap = max(cfg.max_weight, 2.0 / max(n, 1))
    for _ in range(6):
        w = np.clip(w, -cap, cap)
        w = neutralise(w, L, live, cfg.dollar_neutral)
        g = np.abs(w).sum()
        if g <= EPS:
            return np.zeros_like(w)
        w = w / g
        if np.abs(w).max() <= cap + 1e-9:
            break
    return w


def _portfolio_vol(w: np.ndarray, vol_ann: np.ndarray, rho: float) -> float:
    """Ex-ante annualised vol under an equicorrelated model.

    Var = (1 - rho) * sum(w_i^2 s_i^2) + rho * (sum(w_i s_i))^2
    """
    wv = w * np.nan_to_num(vol_ann, nan=0.0)
    var = (1.0 - rho) * float(np.sum(wv ** 2)) + rho * float(np.sum(wv)) ** 2
    return float(np.sqrt(max(var, 0.0)))


def run_backtest(panel: dict[str, pd.DataFrame], scores: pd.DataFrame, mask: pd.DataFrame,
                 vol_ann: pd.DataFrame, beta: pd.DataFrame, cost: CostModel,
                 cfg: ExecConfig, *, rho: float = 0.30,
                 factor_model: tuple | None = None) -> BacktestResult:
    """Simulate the book.  ``scores`` is NaN at bars where no decision is made."""
    syms = list(mask.columns)
    idx = mask.index
    T, N = len(idx), len(syms)

    op = panel["open"].reindex(index=idx, columns=syms).to_numpy(dtype="float64")
    qv = panel["quote_volume"].reindex(index=idx, columns=syms).to_numpy(dtype="float64")
    fr = panel["funding_rate"].reindex(index=idx, columns=syms).to_numpy(dtype="float64")
    sc = scores.reindex(index=idx, columns=syms).to_numpy(dtype="float64")
    mk = mask.reindex(index=idx, columns=syms).to_numpy()
    vl = vol_ann.reindex(index=idx, columns=syms).to_numpy(dtype="float64")
    bt = beta.reindex(index=idx, columns=syms).to_numpy(dtype="float64")

    # last printed open, used to liquidate symbols whose data ends
    last_px = np.full(N, np.nan)
    shares = np.zeros(N)
    equity = cfg.init_equity

    eq_arr = np.full(T, np.nan)
    ret_arr = np.zeros(T)
    gross_arr = np.zeros(T)
    net_arr = np.zeros(T)
    to_arr = np.zeros(T)
    cost_arr = np.zeros(T)
    fund_arr = np.zeros(T)
    npos_arr = np.zeros(T)
    vs_arr = np.full(T, np.nan)
    notional_arr = np.zeros(T)
    w_hist = np.zeros((T, N), dtype="float32")

    vol_win = cfg.vol_target_window_days * cfg.bars_per_day
    ann = np.sqrt(cfg.bars_per_day * 365.0)
    ret_hist = np.zeros(T)
    unit_ret = np.zeros(T)
    scale_hist = np.ones(T)

    fm_L = fm_fvar = fm_idio = fm_step = None
    if factor_model is not None:
        fm_L, fm_fvar, fm_idio, fm_step = factor_model
        bars_year = cfg.bars_per_day * 365.0

    blown = False
    blow_t = None
    truncated_notional = 0.0
    requested_notional = 0.0
    delist_events = 0
    w_prev = np.zeros(N)

    for t in range(T):
        px = op[t]
        valid_px = np.isfinite(px) & (px > 0)
        last_px = np.where(valid_px, px, last_px)

        # ---- 1. mark-to-market over [open[t-1], open[t]] ------------------ #
        if t > 0:
            prev_px = np.where(np.isfinite(op[t - 1]) & (op[t - 1] > 0), op[t - 1], np.nan)
            cur = np.where(valid_px, px, prev_px)
            dpx = np.nan_to_num(cur - prev_px, nan=0.0)
            pnl = float(np.sum(shares * dpx))
            equity += pnl

            # ---- 2. funding settled inside the bar ------------------------ #
            rate = np.nan_to_num(fr[t], nan=0.0)
            if cost.funding_enabled and np.any(rate != 0.0):
                notional = shares * np.nan_to_num(cur, nan=0.0)
                f_pnl = -float(np.sum(notional * rate))
                equity += f_pnl
                fund_arr[t] = f_pnl

        if equity <= 0:
            blown, blow_t = True, idx[t]
            eq_arr[t:] = 0.0
            if cfg.allow_bankruptcy_stop:
                break

        # ---- 3. force-close positions in symbols that stopped printing ---- #
        stale = (shares != 0) & (~valid_px)
        if stale.any():
            liq_px = np.nan_to_num(last_px, nan=0.0)
            notion = np.abs(shares * liq_px)[stale].sum()
            c = notion * (cost.linear_bps * cfg.delist_cost_mult) / 1e4
            equity -= c
            cost_arr[t] += c
            notional_arr[t] += notion
            shares = np.where(stale, 0.0, shares)
            delist_events += int(stale.sum())

        # ---- 4. rebalance if a decision was made at bar t-1 --------------- #
        have_score = t > 0 and np.isfinite(sc[t - 1]).any()
        if have_score and equity > 0:
            live = mk[t - 1] & valid_px & np.isfinite(sc[t - 1])
            if fm_L is not None:
                si = int(fm_step[t - 1])
                L_t = fm_L[si]
                idio_t = fm_idio[si] * np.sqrt(bars_year)
                live = live & np.isfinite(idio_t) & (np.abs(L_t).sum(axis=1) > 0)
                w_t = target_weights_factor(sc[t - 1], idio_t, L_t, live, cfg)
            else:
                w_t = target_weights(sc[t - 1], vl[t - 1], bt[t - 1], live, cfg)

            # volatility targeting from trailing realised strategy vol only
            if t > vol_win // 4:
                lo = max(0, t - vol_win)
                rv = unit_ret[lo:t]
                rv = rv[rv != 0.0]
                realised = float(np.std(rv) * ann) if rv.size > 20 else np.nan
            else:
                realised = np.nan
            if fm_L is not None:
                ex_ante = factor_portfolio_vol(w_t, fm_L[int(fm_step[t - 1])],
                                               fm_fvar[int(fm_step[t - 1])],
                                               fm_idio[int(fm_step[t - 1])], bars_year)
            else:
                ex_ante = _portfolio_vol(w_t, vl[t - 1], rho)
            base_vol = realised if np.isfinite(realised) and realised > 1e-4 else ex_ante
            if not np.isfinite(base_vol) or base_vol <= 1e-4:
                vs = 1.0
            else:
                vs = cfg.target_ann_vol / base_vol
            vs = float(np.clip(vs, *cfg.vol_scalar_bounds))
            vs_arr[t] = vs

            scale = vs * cfg.leverage
            w_t = w_t * scale
            applied_scale = scale
            g = np.abs(w_t).sum()
            if g > cfg.max_gross:
                shrink = cfg.max_gross / g
                w_t = w_t * shrink
                applied_scale *= shrink
            scale_hist[t] = max(applied_scale, 1e-6)

            if cfg.partial_adjust < 1.0:
                w_t = w_prev + cfg.partial_adjust * (w_t - w_prev)

            # current weights from held shares
            cur_px = np.where(valid_px, px, 0.0)
            w_cur = shares * cur_px / max(equity, EPS)
            dw = w_t - w_cur
            if cfg.no_trade_band > 0:
                dw = np.where(np.abs(dw) < cfg.no_trade_band, 0.0, dw)

            trade_notional = dw * equity
            requested_notional += float(np.abs(trade_notional).sum())

            # participation cap -> truncate trades we cannot realistically do
            bar_dv = np.nan_to_num(qv[t], nan=0.0)
            cap_notional = cfg.max_participation * bar_dv
            over = np.abs(trade_notional) > cap_notional
            if over.any():
                truncated_notional += float((np.abs(trade_notional) - cap_notional)[over].sum())
                trade_notional = np.where(over, np.sign(trade_notional) * cap_notional,
                                          trade_notional)

            part = np.where(bar_dv > 0, np.abs(trade_notional) / bar_dv, 0.0)
            bps = cost.linear_bps + cost.impact_coef * np.sqrt(np.clip(part, 0, 1))
            c = float(np.sum(np.abs(trade_notional) * bps) / 1e4)
            equity -= c
            cost_arr[t] += c
            notional_arr[t] += float(np.abs(trade_notional).sum())

            d_shares = np.where(cur_px > 0, trade_notional / np.where(cur_px > 0, cur_px, 1.0), 0.0)
            shares = shares + d_shares

        # ---- 5. bookkeeping ---------------------------------------------- #
        if t > 0 and scale_hist[t] == 1.0 and not have_score:
            scale_hist[t] = scale_hist[t - 1]
        cur_px = np.where(valid_px, px, np.nan_to_num(last_px, nan=0.0))
        notion = shares * cur_px
        gross = float(np.abs(notion).sum())
        gross_arr[t] = gross / max(equity, EPS)
        net_arr[t] = float(notion.sum()) / max(equity, EPS)
        npos_arr[t] = int((np.abs(shares) > 0).sum())
        w_prev = notion / max(equity, EPS)
        w_hist[t] = w_prev.astype("float32")
        to_arr[t] = notional_arr[t] / max(equity, EPS)
        eq_arr[t] = equity
        if t > 0 and np.isfinite(eq_arr[t - 1]) and eq_arr[t - 1] > 0:
            ret_arr[t] = equity / eq_arr[t - 1] - 1.0
            ret_hist[t] = ret_arr[t]
            # de-lever by the scale that was in force, so vol targeting measures
            # the unit-risk book rather than chasing its own leverage
            s_eff = scale_hist[t] if scale_hist[t] > 1e-6 else scale_hist[max(t - 1, 0)]
            unit_ret[t] = ret_arr[t] / max(s_eff, 1e-6)

        # ---- 6. margin check --------------------------------------------- #
        if gross * cfg.maintenance_margin > equity:
            blown, blow_t = True, idx[t]
            eq_arr[t:] = 0.0
            if cfg.allow_bankruptcy_stop:
                break

    eq = pd.Series(eq_arr, index=idx).ffill()
    rets = pd.Series(ret_arr, index=idx)
    res = BacktestResult(
        equity=eq,
        returns=rets,
        gross=pd.Series(gross_arr, index=idx),
        net_exposure=pd.Series(net_arr, index=idx),
        turnover=pd.Series(to_arr, index=idx),
        costs=pd.Series(cost_arr, index=idx),
        funding=pd.Series(fund_arr, index=idx),
        n_positions=pd.Series(npos_arr, index=idx),
        weights=pd.DataFrame(w_hist, index=idx, columns=syms),
        vol_scalar=pd.Series(vs_arr, index=idx),
        trades_notional=pd.Series(notional_arr, index=idx),
        blown_up=blown,
        blowup_time=blow_t,
        diagnostics={
            "requested_notional": requested_notional,
            "truncated_notional": truncated_notional,
            "truncation_frac": truncated_notional / max(requested_notional, EPS),
            "delist_liquidations": delist_events,
            "total_costs": float(np.nansum(cost_arr)),
            "total_funding": float(np.nansum(fund_arr)),
        },
    )
    return res
