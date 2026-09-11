"""Alpha signal library for hourly crypto perp panels.

Every function returns a (T x N) DataFrame whose row t uses ONLY information
available at or before the close of bar t. The portfolio layer applies the
one-bar shift that makes it tradable, so nothing here may peek forward.
"""
import numpy as np, pandas as pd

H = 1  # one bar = 1 hour


# ---------- helpers ----------
def zscore_xs(df, clip=3.0):
    """Cross-sectional z-score across symbols within each bar."""
    m = df.mean(axis=1)
    s = df.std(axis=1).replace(0, np.nan)
    return df.sub(m, axis=0).div(s, axis=0).clip(-clip, clip)


def rank_xs(df):
    """Cross-sectional rank mapped to [-1, 1], NaN-safe."""
    r = df.rank(axis=1, pct=True)
    return (r - 0.5) * 2


def ts_z(df, window):
    m = df.rolling(window, min_periods=window // 2).mean()
    s = df.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return (df - m) / s


def realized_vol(close, window=24 * 7):
    r = np.log(close).diff()
    return r.rolling(window, min_periods=window // 3).std()


# ---------- alpha families ----------
def ts_momentum(close, lookback):
    """Time-series momentum: own past return, vol-normalised."""
    r = np.log(close).diff(lookback)
    v = realized_vol(close, max(lookback, 48)) * np.sqrt(lookback)
    return (r / v.replace(0, np.nan)).clip(-4, 4)


def xs_momentum(close, lookback, skip=0):
    """Cross-sectional momentum, optionally skipping the most recent bars
    to avoid contamination by short-term reversal."""
    lp = np.log(close)
    r = lp.shift(skip) - lp.shift(lookback + skip)
    return rank_xs(r)


def xs_reversal(close, lookback):
    """Short-horizon cross-sectional reversal: fade recent relative winners."""
    lp = np.log(close)
    r = lp - lp.shift(lookback)
    # de-market: reversal is about idiosyncratic moves, not beta
    r = r.sub(r.mean(axis=1), axis=0)
    return -rank_xs(r)


def funding_carry(funding, window=24 * 3):
    """Perp carry: shorts collect when funding is persistently positive."""
    f = funding.ffill(limit=8).fillna(0.0)
    avg = f.rolling(window, min_periods=3).mean()
    return -rank_xs(avg)


def order_flow_imbalance(taker_buy_quote, quote_volume, window=24):
    """Aggressor imbalance: (buy - sell) / total taker volume, smoothed."""
    tot = quote_volume.replace(0, np.nan)
    imb = (2 * taker_buy_quote - quote_volume) / tot
    return rank_xs(imb.rolling(window, min_periods=window // 2).mean())


def volume_shock(quote_volume, window=24 * 7):
    """Abnormal volume: crowding / attention proxy."""
    lv = np.log1p(quote_volume)
    return rank_xs(ts_z(lv, window))


def low_vol(close, window=24 * 14):
    """Low-volatility premium: prefer the calmer names."""
    return -rank_xs(realized_vol(close, window))


def range_position(close, high, low, window=24 * 7):
    """Where price sits inside its recent range (breakout / stretch)."""
    hh = high.rolling(window, min_periods=window // 3).max()
    ll = low.rolling(window, min_periods=window // 3).min()
    pos = (close - ll) / (hh - ll).replace(0, np.nan)
    return rank_xs(pos - 0.5)


def intraday_reversal(close, open_, window=6):
    """Fade the recent intrabar drift (overnight/intraday reversal analogue)."""
    r = (np.log(close) - np.log(open_)).rolling(window, min_periods=2).sum()
    r = r.sub(r.mean(axis=1), axis=0)
    return -rank_xs(r)


def illiq_amihud(close, quote_volume, window=24 * 7):
    """Amihud illiquidity: |return| per dollar of volume."""
    r = np.log(close).diff().abs()
    il = (r / quote_volume.replace(0, np.nan)).rolling(window, min_periods=window // 3).mean()
    return -rank_xs(np.log1p(il * 1e9))


def beta_to_market(close, window=24 * 14):
    """Beta against the equal-weight crypto market."""
    r = np.log(close).diff()
    mkt = r.mean(axis=1)
    cov = r.mul(mkt, axis=0).rolling(window, min_periods=window // 3).mean() \
          - r.rolling(window, min_periods=window // 3).mean().mul(
              mkt.rolling(window, min_periods=window // 3).mean(), axis=0)
    var = mkt.rolling(window, min_periods=window // 3).var()
    return cov.div(var, axis=0)


def idio_momentum(close, window=24 * 14, lookback=24 * 7):
    """Momentum of the market-residual return (beta-neutral momentum)."""
    r = np.log(close).diff()
    mkt = r.mean(axis=1)
    b = beta_to_market(close, window)
    resid = r.sub(b.mul(mkt, axis=0))
    return rank_xs(resid.rolling(lookback, min_periods=lookback // 3).sum())


def basis_momentum(funding, window=24 * 7):
    """Change in funding: a proxy for shifts in leveraged positioning."""
    f = funding.ffill(limit=8).fillna(0.0)
    avg = f.rolling(window, min_periods=3).mean()
    return -rank_xs(avg.diff(window))


ALPHAS = {
    "ts_mom_24":   lambda p: ts_momentum(p["close"], 24),
    "ts_mom_168":  lambda p: ts_momentum(p["close"], 168),
    "ts_mom_720":  lambda p: ts_momentum(p["close"], 720),
    "xs_mom_168":  lambda p: xs_momentum(p["close"], 168, skip=6),
    "xs_mom_720":  lambda p: xs_momentum(p["close"], 720, skip=24),
    "xs_rev_6":    lambda p: xs_reversal(p["close"], 6),
    "xs_rev_24":   lambda p: xs_reversal(p["close"], 24),
    "xs_rev_72":   lambda p: xs_reversal(p["close"], 72),
    "carry":       lambda p: funding_carry(p["funding"], 72),
    "carry_chg":   lambda p: basis_momentum(p["funding"], 168),
    "ofi_24":      lambda p: order_flow_imbalance(p["taker_buy_quote"], p["quote_volume"], 24),
    "ofi_6":       lambda p: order_flow_imbalance(p["taker_buy_quote"], p["quote_volume"], 6),
    "vol_shock":   lambda p: volume_shock(p["quote_volume"], 168),
    "low_vol":     lambda p: low_vol(p["close"], 336),
    "range_pos":   lambda p: range_position(p["close"], p["high"], p["low"], 168),
    "intraday_rev": lambda p: intraday_reversal(p["close"], p["open"], 6),
    "illiq":       lambda p: illiq_amihud(p["close"], p["quote_volume"], 168),
    "idio_mom":    lambda p: idio_momentum(p["close"], 336, 168),
}


def wick_reversal(open_, high, low, close, window=4):
    """Liquidation-cascade fade: a long lower wick is forced selling, and the
    price that printed it tends to bounce. Upper wick is the mirror image."""
    rng = (high - low).replace(0, np.nan)
    upper = (high - np.maximum(open_, close)) / rng
    lower = (np.minimum(open_, close) - low) / rng
    sig = (lower - upper).rolling(window, min_periods=1).mean()
    return rank_xs(sig)


def funding_extreme(funding, window=24 * 14):
    """Crowding: fade positioning when funding is at a local extreme."""
    f = funding.ffill(limit=8).fillna(0.0)
    z = ts_z(f.rolling(8, min_periods=1).mean(), window)
    return -rank_xs(z.clip(-4, 4))


def vol_of_vol(close, window=24 * 7, long_window=24 * 30):
    """Prefer names whose volatility is itself stable (regime-stability premium)."""
    v = realized_vol(close, window)
    return -rank_xs(v.rolling(long_window, min_periods=long_window // 3).std() /
                    v.rolling(long_window, min_periods=long_window // 3).mean())


def mom_accel(close, fast=72, slow=336):
    """Momentum acceleration: fast trend relative to slow trend."""
    lp = np.log(close)
    f = lp - lp.shift(fast)
    sl = (lp - lp.shift(slow)) * (fast / slow)
    return rank_xs(f - sl)


def dollar_vol_trend(quote_volume, window=24 * 3, base=24 * 21):
    """Liquidity trend: names gaining traded value attract flow."""
    a = quote_volume.rolling(window, min_periods=window // 2).mean()
    b = quote_volume.rolling(base, min_periods=base // 3).mean()
    return rank_xs(np.log1p(a) - np.log1p(b))


def resid_reversal(close, window=24 * 14, lookback=24):
    """Reversal on the market-residual return (beta-neutral short-term reversal)."""
    r = np.log(close).diff()
    mkt = r.mean(axis=1)
    b = beta_to_market(close, window)
    resid = r.sub(b.mul(mkt, axis=0))
    return -rank_xs(resid.rolling(lookback, min_periods=lookback // 2).sum())


ALPHAS.update({
    "wick_rev":    lambda p: wick_reversal(p["open"], p["high"], p["low"], p["close"], 4),
    "fund_extreme": lambda p: funding_extreme(p["funding"], 336),
    "vol_of_vol":  lambda p: vol_of_vol(p["close"], 168, 720),
    "mom_accel":   lambda p: mom_accel(p["close"], 72, 336),
    "dv_trend":    lambda p: dollar_vol_trend(p["quote_volume"], 72, 504),
    "resid_rev_24": lambda p: resid_reversal(p["close"], 336, 24),
    "resid_rev_72": lambda p: resid_reversal(p["close"], 336, 72),
})
