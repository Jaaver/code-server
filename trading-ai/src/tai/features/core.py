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

import gc
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
                         bars_per_day: int = 24, metrics: dict | None = None
                         ) -> tuple[dict[str, pd.DataFrame], dict]:
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

    if metrics:
        F = add_metrics_features(F, metrics, panel, mask,
                                 {"ret": ret, "vol_ref": vol_ref})

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


# --------------------------------------------------------------------------- #
# chunked / streaming construction
# --------------------------------------------------------------------------- #
def build_features_chunked(panel: dict[str, pd.DataFrame], mask: pd.DataFrame,
                           bars_per_day: int = 24, *, chunk: int = 4000,
                           warmup: int = 2600, log_progress: bool = True,
                           metrics: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Long-format feature matrix built in overlapping time chunks.

    Holding ~130 wide panels for a 6-year, 800-symbol history costs more memory
    than the machine has.  Chunking bounds peak memory at roughly
    ``n_features * n_symbols * (chunk + warmup)`` floats while producing values
    identical to the unchunked build, provided ``warmup`` covers the longest
    rolling window in :func:`build_feature_panels` (the longest chain is a 720-bar
    statistic z-scored over 720 bars, so ~1500 bars; 2600 leaves a wide margin).
    ``tests/test_chunking.py`` asserts the equivalence.

    The per-chunk results are written straight into one preallocated array rather
    than collected and concatenated: concatenating a list of chunk frames needs the
    parts and the result in memory at once, which doubles the peak exactly at the
    end of the build and is what makes this step run out of memory.
    """
    n = len(mask)
    mask_np = mask.to_numpy()
    n_rows_total = int(mask_np.sum())
    starts = list(range(0, n, chunk))

    data: np.ndarray | None = None
    names: list[str] = []
    ts_out = np.empty(n_rows_total, dtype="datetime64[ns]")
    sym_out = np.empty(n_rows_total, dtype=object)
    aux_parts: dict[str, list] = {"vol_ann": [], "beta": [], "vol_ref": []}
    filled = 0

    for ci, i0 in enumerate(starts):
        i1 = min(i0 + chunk, n)
        lo = max(0, i0 - warmup)
        sub = {k: v.iloc[lo:i1] for k, v in panel.items()}
        sub_mask = mask.iloc[lo:i1]
        sub_met = ({k: v.iloc[lo:i1] for k, v in metrics.items()} if metrics else None)
        F, aux = build_feature_panels(sub, sub_mask, bars_per_day, metrics=sub_met)
        head = i0 - lo
        keep_mask = sub_mask.iloc[head:]
        F = {k: v.iloc[head:] for k, v in F.items()}

        if data is None:
            names = list(F.keys())
            data = np.empty((n_rows_total, len(names)), dtype="float32")
        elif list(F.keys()) != names:
            # Silently reindexing here once dropped every open-interest feature,
            # because the first chunk predates the metrics archive and so defined a
            # narrower schema than later chunks.  Fail loudly instead.
            missing = set(F) - set(names)
            extra = set(names) - set(F)
            raise RuntimeError(
                f"feature schema changed at chunk {ci + 1}: +{sorted(missing)} "
                f"-{sorted(extra)}; every chunk must produce the same columns")
        block = stack_features(F, keep_mask)
        k = len(block)
        if k:
            data[filled:filled + k] = block.to_numpy(dtype="float32")
            # store UTC instants as naive datetime64 and re-attach UTC at the end;
            # .to_numpy() on a tz-aware index warns about exactly this conversion
            ts_out[filled:filled + k] = (block.index.get_level_values(0)
                                         .tz_convert(None).to_numpy())
            sym_out[filled:filled + k] = np.asarray(block.index.get_level_values(1),
                                                    dtype=object)
            filled += k
        for key in aux_parts:
            aux_parts[key].append(aux[key].iloc[head:])
        del F, aux, sub, sub_mask, sub_met, block
        gc.collect()
        if log_progress:
            log.info("features chunk %d/%d (%s..%s) rows=%d/%d", ci + 1, len(starts),
                     mask.index[i0].date(), mask.index[i1 - 1].date(), filled,
                     n_rows_total)

    if data is None:
        raise RuntimeError("no feature chunks produced")
    if filled != n_rows_total:
        data = data[:filled]
        ts_out = ts_out[:filled]
        sym_out = sym_out[:filled]
    idx = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(ts_out, tz="UTC"), sym_out], names=["ts", "symbol"])
    X = pd.DataFrame(data, index=idx, columns=names, copy=False)
    aux = {k: pd.concat(v) for k, v in aux_parts.items()}
    return X, aux


def aux_panels(panel: dict[str, pd.DataFrame], mask: pd.DataFrame,
               bars_per_day: int = 24) -> dict[str, pd.DataFrame]:
    """Just the trailing volatility and beta estimates used for sizing and labels.

    Computed with exactly the same definitions as :func:`build_feature_panels`, so
    evaluation code can rebuild sizing inputs cheaply without the full matrix.
    """
    c = panel["close"]
    d = bars_per_day
    ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan).astype("float32")
    vol_med = _ewm_vol(ret, 72)
    vol_ref = vol_med.where(vol_med > 0).ffill(limit=6)
    mkt = ret.where(mask).median(axis=1).astype("float32")
    bw = 24 * 30
    cov = ret.mul(mkt, axis=0).rolling(bw, min_periods=bw // 3).mean() - \
        ret.rolling(bw, min_periods=bw // 3).mean().mul(
            mkt.rolling(bw, min_periods=bw // 3).mean(), axis=0)
    var_m = mkt.rolling(bw, min_periods=bw // 3).var()
    beta = cov.div(var_m + EPS, axis=0).clip(-3, 4).astype("float32")
    return {"vol_ref": vol_ref, "vol_ann": (vol_ref * np.sqrt(d * 365)).astype("float32"),
            "beta": beta, "mkt_ret": mkt, "ret": ret}


# --------------------------------------------------------------------------- #
# open-interest / positioning features (optional; requires the metrics archive)
# --------------------------------------------------------------------------- #
def add_metrics_features(F: dict[str, pd.DataFrame], metrics: dict[str, pd.DataFrame],
                         panel: dict[str, pd.DataFrame], mask: pd.DataFrame,
                         aux: dict) -> dict[str, pd.DataFrame]:
    """Derive positioning features from the exchange's open-interest snapshots.

    Open interest tells you whether a move was driven by new leveraged positions
    being opened or by existing ones being closed, which price and volume cannot
    distinguish.  The long/short ratios separate the crowd from the large accounts.
    All values are snapshot-at-or-before-bar-close, so the transforms below are
    trailing by construction.
    """
    if not metrics:
        return F
    cols = mask.columns
    idx = mask.index
    ret = aux["ret"]
    vol_ref = aux["vol_ref"]

    def al(df):
        return df.reindex(index=idx, columns=cols).astype("float32")

    oi = al(metrics.get("sum_open_interest", pd.DataFrame()))
    oiv = al(metrics.get("sum_open_interest_value", pd.DataFrame()))
    qv = panel["quote_volume"].reindex(index=idx, columns=cols).astype("float32")

    loi = np.log(oi.where(oi > 0))
    for k in (1, 4, 12, 24, 72, 168):
        d = (loi - loi.shift(k)).astype("float32")
        F[f"oi_chg_{k}"] = (d / (vol_ref * np.sqrt(k) + EPS)).clip(-15, 15).astype("float32")
    F["oi_chg_z"] = _zs(loi.diff(24), 24 * 30)
    # positioning direction: OI rising with price = new longs, a crowding signal
    r24 = np.log(panel["close"] / panel["close"].shift(24)).reindex(
        index=idx, columns=cols).astype("float32")
    F["oi_x_ret_24"] = (F["oi_chg_24"] * np.sign(r24)).clip(-15, 15).astype("float32")
    F["oi_x_ret_4"] = (F["oi_chg_4"] * np.sign(
        np.log(panel["close"] / panel["close"].shift(4)).reindex(
            index=idx, columns=cols))).clip(-15, 15).astype("float32")

    # leverage intensity: open interest relative to the flow that has to unwind it
    ratio = (oiv / (qv.rolling(24, min_periods=8).mean() * 24 + EPS)).astype("float32")
    F["oi_over_adv"] = np.log1p(ratio.clip(0, 500)).astype("float32")
    F["oi_over_adv_z"] = _zs(F["oi_over_adv"], 24 * 30)

    ls_retail = al(metrics.get("count_long_short_ratio", pd.DataFrame()))
    ls_top_acct = al(metrics.get("count_toptrader_long_short_ratio", pd.DataFrame()))
    ls_top_pos = al(metrics.get("sum_toptrader_long_short_ratio", pd.DataFrame()))
    taker_ls = al(metrics.get("sum_taker_long_short_vol_ratio", pd.DataFrame()))

    for name, df in (("ls_retail", ls_retail), ("ls_top_acct", ls_top_acct),
                     ("ls_top_pos", ls_top_pos), ("taker_ls", taker_ls)):
        lg = np.log(df.where(df > 0)).astype("float32")
        F[name] = lg.clip(-3, 3)
        F[f"{name}_z"] = _zs(lg, 24 * 30)
        F[f"{name}_chg_24"] = (lg - lg.shift(24)).clip(-3, 3).astype("float32")

    # large accounts positioned against the crowd is the classic setup
    F["ls_top_minus_retail"] = (np.log(ls_top_pos.where(ls_top_pos > 0))
                                - np.log(ls_retail.where(ls_retail > 0))
                                ).clip(-3, 3).astype("float32")
    F["ls_top_minus_retail_z"] = _zs(F["ls_top_minus_retail"], 24 * 30)

    for k in ("oi_chg_24", "oi_chg_4", "oi_over_adv", "ls_retail_z",
              "ls_top_minus_retail", "taker_ls_z", "oi_x_ret_24"):
        F[f"cs_{k}"] = _cs_rank(F[k], mask)
    return F
