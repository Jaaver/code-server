"""Leak-free feature construction for the cross-sectional perp model.

Contract
--------
Every feature at bar ``t`` is a function of data whose *close* is at or before
``t``.  Decisions are taken at the close of ``t`` and executed at the open of
``t+1``; labels start at the open of ``t+1``.  No feature uses ``shift(-k)``.

``build_features`` returns a long-format frame with a ``MultiIndex`` of
``(ts, symbol)`` restricted to the point-in-time tradable mask.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

EPS = 1e-12


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _zs(df: pd.DataFrame, win: int, minp: int | None = None) -> pd.DataFrame:
    """Trailing z-score of each column."""
    minp = minp or max(10, win // 4)
    m = df.rolling(win, min_periods=minp).mean()
    s = df.rolling(win, min_periods=minp).std()
    return ((df - m) / (s + EPS)).astype("float32")


def _ewm_vol(ret: pd.DataFrame, span: int) -> pd.DataFrame:
    return np.sqrt((ret ** 2).ewm(span=span, min_periods=span // 3).mean()).astype("float32")


def _cs_rank(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank in [-1, 1] computed only over tradable names."""
    x = df.where(mask)
    r = x.rank(axis=1, pct=True)
    return ((r - 0.5) * 2.0).astype("float32")


def _cs_demean(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    x = df.where(mask)
    return (x.sub(x.mean(axis=1), axis=0)).astype("float32")


def _cs_z(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    x = df.where(mask)
    mu = x.mean(axis=1)
    sd = x.std(axis=1)
    return (x.sub(mu, axis=0).div(sd + EPS, axis=0)).clip(-5, 5).astype("float32")


def _winsor(df: pd.DataFrame, lo: float = 0.005, hi: float = 0.995) -> pd.DataFrame:
    ql = df.quantile(lo, axis=1)
    qh = df.quantile(hi, axis=1)
    return df.clip(lower=ql, upper=qh, axis=0)


# --------------------------------------------------------------------------- #
# main builder
# --------------------------------------------------------------------------- #
def build_feature_panels(panel: dict[str, pd.DataFrame], mask: pd.DataFrame,
                         bars_per_day: int = 24) -> tuple[dict[str, pd.DataFrame], dict]:
    """Compute every feature as a wide ``[time, symbol]`` panel.

    Returns ``(features, aux)`` where ``aux`` carries the trailing volatility and
    beta estimates that label construction and position sizing reuse, so the three
    stages cannot drift apart.
    """
    o, h, l, c = panel["open"], panel["high"], panel["low"], panel["close"]
    qv = panel["quote_volume"].astype("float32")
    trades = panel["trades"].astype("float32")
    tbq = panel["taker_buy_quote_volume"].astype("float32")
    fund_last = panel["funding_last"]

    d = bars_per_day
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).astype("float32")
    F: dict[str, pd.DataFrame] = {}

    # ---- volatility scales (trailing only) ------------------------------- #
    vol_fast = _ewm_vol(ret, 24)
    vol_med = _ewm_vol(ret, 72)
    vol_slow = _ewm_vol(ret, 24 * 14)
    vol_ref = vol_med.where(vol_med > 0).ffill(limit=6)

    # ---- multi-horizon vol-normalised momentum / reversal ---------------- #
    for k in (1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 120, 168, 336, 720):
        r = np.log(c / c.shift(k)).astype("float32")
        F[f"ret_{k}"] = (r / (vol_ref * np.sqrt(k) + EPS)).clip(-10, 10).astype("float32")

    # residual (BTC-neutral) returns: the market factor is the cross-sectional
    # median return of the tradable universe, which is observable at t.
    mkt = ret.where(mask).median(axis=1).astype("float32")
    mkt_cum = {k: mkt.rolling(k, min_periods=k).sum() for k in (4, 12, 24, 72, 168)}
    beta_win = 24 * 30
    cov = ret.mul(mkt, axis=0).rolling(beta_win, min_periods=beta_win // 3).mean() - \
        ret.rolling(beta_win, min_periods=beta_win // 3).mean().mul(
            mkt.rolling(beta_win, min_periods=beta_win // 3).mean(), axis=0)
    var_m = mkt.rolling(beta_win, min_periods=beta_win // 3).var()
    beta = cov.div(var_m + EPS, axis=0).clip(-3, 4).astype("float32")
    F["beta"] = beta
    for k in (4, 12, 24, 72, 168):
        r = np.log(c / c.shift(k)).astype("float32")
        resid = r.sub(beta.mul(mkt_cum[k], axis=0))
        F[f"resid_{k}"] = (resid / (vol_ref * np.sqrt(k) + EPS)).clip(-10, 10).astype("float32")

    # ---- volatility state ------------------------------------------------ #
    F["vol_fast"] = (vol_fast * np.sqrt(d * 365)).clip(0, 20).astype("float32")
    F["vol_ratio_fast_slow"] = (vol_fast / (vol_slow + EPS)).clip(0, 8).astype("float32")
    F["vol_ratio_med_slow"] = (vol_med / (vol_slow + EPS)).clip(0, 8).astype("float32")
    F["vol_z"] = _zs(vol_fast, 24 * 30)

    # Parkinson range vol and its relation to close-to-close vol (trend vs chop)
    hl = np.log((h / l).replace([np.inf, -np.inf], np.nan)).astype("float32")
    park = np.sqrt((hl ** 2).rolling(24, min_periods=8).mean() / (4 * np.log(2)))
    F["park_over_cc"] = (park / (vol_fast + EPS)).clip(0, 8).astype("float32")
    F["range_rel"] = (hl / (hl.rolling(24 * 7, min_periods=24).mean() + EPS)).clip(0, 10).astype("float32")

    # ---- intrabar position / pressure ------------------------------------ #
    rng = (h - l).replace(0, np.nan)
    F["close_loc"] = (((c - l) / (rng + EPS)) * 2 - 1).clip(-1, 1).astype("float32")
    F["close_loc_4"] = F["close_loc"].rolling(4, min_periods=2).mean().astype("float32")
    F["close_loc_24"] = F["close_loc"].rolling(24, min_periods=8).mean().astype("float32")
    F["gap"] = ((o - c.shift(1)) / (c.shift(1) * vol_ref + EPS)).clip(-10, 10).astype("float32")

    # ---- order-flow imbalance (taker buy vs taker sell notional) --------- #
    ofi = ((2 * tbq - qv) / (qv + EPS)).clip(-1, 1).astype("float32")
    F["ofi_1"] = ofi
    for k in (4, 12, 24, 72, 168):
        F[f"ofi_{k}"] = ofi.rolling(k, min_periods=max(2, k // 3)).mean().astype("float32")
    # notional-weighted imbalance: signed flow relative to trailing dollar volume
    signed = (2 * tbq - qv).astype("float32")
    adv_bar = qv.rolling(24 * 7, min_periods=24).mean()
    for k in (4, 24, 72):
        F[f"flow_{k}"] = (signed.rolling(k, min_periods=max(2, k // 3)).sum()
                          / (adv_bar * k + EPS)).clip(-3, 3).astype("float32")
    F["ofi_z"] = _zs(F["ofi_24"], 24 * 30)

    # ---- volume / trade-intensity microstructure ------------------------- #
    lqv = np.log1p(qv)
    F["qv_z"] = _zs(lqv, 24 * 14)
    F["qv_z_long"] = _zs(lqv, 24 * 60)
    F["trades_z"] = _zs(np.log1p(trades), 24 * 14)
    avg_trade = (qv / (trades + 1.0)).astype("float32")
    F["avg_trade_z"] = _zs(np.log1p(avg_trade), 24 * 14)
    F["avg_trade_chg"] = (np.log1p(avg_trade) -
                          np.log1p(avg_trade).rolling(24, min_periods=8).mean()).astype("float32")
    # Amihud illiquidity: |return| per unit of dollar volume
    amihud = (ret.abs() / (qv + 1.0)).astype("float32")
    F["amihud_z"] = _zs(np.log(amihud + 1e-15), 24 * 30)
    F["turnover_accel"] = (qv.rolling(4, min_periods=2).mean()
                           / (qv.rolling(24 * 7, min_periods=24).mean() + EPS)
                           ).clip(0, 20).astype("float32")

    # ---- funding / perp basis -------------------------------------------- #
    fr = fund_last.astype("float32")
    F["funding"] = (fr * 1e4).clip(-100, 100).astype("float32")
    F["funding_z"] = _zs(fr, 24 * 30)
    for k in (24, 72, 168, 720):
        F[f"funding_cum_{k}"] = (fr.rolling(k, min_periods=max(3, k // 4)).mean()
                                 * 1e4).clip(-100, 100).astype("float32")
    F["funding_minus_mkt"] = (F["funding"] - F["funding"].where(mask).median(axis=1).values[:, None]
                              ).astype("float32")
    # carry/return interaction: crowded longs (high funding) after a rally
    F["funding_x_ret24"] = (F["funding_z"] * F["ret_24"]).clip(-20, 20).astype("float32")

    # ---- distance from reference levels ---------------------------------- #
    vwap_num = (c * qv)
    for k in (24, 72, 168):
        vwap = (vwap_num.rolling(k, min_periods=max(4, k // 4)).sum()
                / (qv.rolling(k, min_periods=max(4, k // 4)).sum() + EPS))
        F[f"dist_vwap_{k}"] = (np.log(c / (vwap + EPS)) / (vol_ref * np.sqrt(k) + EPS)
                               ).clip(-10, 10).astype("float32")
    for k in (24, 168, 720):
        mx = h.rolling(k, min_periods=max(4, k // 4)).max()
        mn = l.rolling(k, min_periods=max(4, k // 4)).min()
        F[f"pos_range_{k}"] = (((c - mn) / ((mx - mn) + EPS)) * 2 - 1).clip(-1, 1).astype("float32")
        F[f"drawdown_{k}"] = (np.log(c / (mx + EPS)) / (vol_ref * np.sqrt(k) + EPS)
                              ).clip(-10, 0).astype("float32")

    # ---- short-horizon autocorrelation / chop ---------------------------- #
    sgn = np.sign(ret)
    F["sign_persist_24"] = sgn.rolling(24, min_periods=8).mean().astype("float32")
    F["abs_ret_ratio"] = (ret.abs().rolling(4, min_periods=2).mean()
                          / (ret.abs().rolling(72, min_periods=24).mean() + EPS)
                          ).clip(0, 10).astype("float32")
    up = (ret > 0).astype("float32")
    F["up_frac_72"] = up.rolling(72, min_periods=24).mean().astype("float32")

    # ---- seasonality ----------------------------------------------------- #
    idx = c.index
    hour = idx.hour.values.astype("float32")
    dow = idx.dayofweek.values.astype("float32")
    ones = pd.DataFrame(1.0, index=idx, columns=c.columns, dtype="float32")
    F["hour_sin"] = ones.mul(np.sin(2 * np.pi * hour / 24), axis=0).astype("float32")
    F["hour_cos"] = ones.mul(np.cos(2 * np.pi * hour / 24), axis=0).astype("float32")
    F["dow_sin"] = ones.mul(np.sin(2 * np.pi * dow / 7), axis=0).astype("float32")
    F["dow_cos"] = ones.mul(np.cos(2 * np.pi * dow / 7), axis=0).astype("float32")
    # hours until the next 8h funding settlement (00/08/16 UTC)
    F["bars_to_funding"] = ones.mul(((8 - (hour % 8)) % 8).astype("float32"), axis=0)

    # ---- market-state (identical across symbols, lets the model condition) #
    n_live = mask.sum(axis=1).replace(0, np.nan)
    breadth = (ret.where(mask) > 0).sum(axis=1) / n_live
    disp = ret.where(mask).std(axis=1)
    F["mkt_ret_24"] = ones.mul((mkt.rolling(24, min_periods=8).sum()
                                / (disp.rolling(24 * 7, min_periods=24).mean() * np.sqrt(24) + EPS)
                                ).clip(-10, 10), axis=0).astype("float32")
    F["mkt_ret_168"] = ones.mul((mkt.rolling(168, min_periods=48).sum()
                                 / (disp.rolling(24 * 7, min_periods=24).mean() * np.sqrt(168) + EPS)
                                 ).clip(-10, 10), axis=0).astype("float32")
    F["mkt_vol"] = ones.mul((mkt.ewm(span=72, min_periods=24).std() * np.sqrt(d * 365)
                             ).clip(0, 10), axis=0).astype("float32")
    F["mkt_breadth"] = ones.mul((breadth.rolling(24, min_periods=8).mean() - 0.5) * 2, axis=0).astype("float32")
    F["mkt_disp"] = ones.mul((disp / (disp.rolling(24 * 30, min_periods=24 * 5).mean() + EPS)
                              ).clip(0, 8), axis=0).astype("float32")
    F["mkt_funding"] = ones.mul((fr.where(mask).median(axis=1) * 1e4).clip(-100, 100), axis=0).astype("float32")
    F["n_live"] = ones.mul(n_live.astype("float32") / 100.0, axis=0).astype("float32")

    # ---- cross-sectional transforms of the strongest raw signals --------- #
    cs_src = ["ret_1", "ret_4", "ret_12", "ret_24", "ret_72", "ret_168",
              "resid_4", "resid_12", "resid_24", "resid_72", "resid_168",
              "ofi_4", "ofi_24", "ofi_72", "flow_24", "qv_z", "trades_z",
              "avg_trade_z", "funding", "funding_z", "vol_fast", "vol_ratio_fast_slow",
              "dist_vwap_24", "dist_vwap_168", "pos_range_168", "amihud_z",
              "turnover_accel", "close_loc_24"]
    for k in cs_src:
        F[f"cs_{k}"] = _cs_rank(F[k], mask)

    aux = {
        "vol_ref": vol_ref,
        "vol_ann": (vol_ref * np.sqrt(d * 365)).astype("float32"),
        "beta": beta,
        "mkt_ret": mkt,
        "ret": ret,
    }
    return F, aux


def stack_features(F: dict[str, pd.DataFrame], mask: pd.DataFrame) -> pd.DataFrame:
    """Long-format float32 feature matrix restricted to tradable cells."""
    names = list(F.keys())
    m = mask.to_numpy()
    flat_mask = m.ravel()
    n_rows = int(flat_mask.sum())
    ts_idx = np.repeat(mask.index.to_numpy(), mask.shape[1])[flat_mask]
    sym_idx = np.tile(np.asarray(mask.columns, dtype=object), mask.shape[0])[flat_mask]
    data = np.empty((n_rows, len(names)), dtype="float32")
    for j, n in enumerate(names):
        arr = F[n].reindex(index=mask.index, columns=mask.columns).to_numpy(dtype="float32")
        data[:, j] = arr.ravel()[flat_mask]
    idx = pd.MultiIndex.from_arrays([ts_idx, sym_idx], names=["ts", "symbol"])
    out = pd.DataFrame(data, index=idx, columns=names)
    return out
