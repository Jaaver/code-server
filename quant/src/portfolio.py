"""Turn alpha scores into tradable target weights.

The single most important line in this file is the .shift(1): a score computed
from the close of bar t-1 becomes the weight held during bar t. Everything the
engine sees is already lagged.
"""
import numpy as np, pandas as pd

# hourly log-return vol floor (~5% annualised): below any real perp
VOL_FLOOR = 1e-4


def tradable_mask(panels, min_dollar_vol=2e6, vol_window=24, min_history=24 * 30):
    """Liquidity + listing filter, computed only from past data."""
    dv = panels["quote_volume"]
    adv = dv.rolling(vol_window, min_periods=vol_window // 2).mean()
    liquid = adv >= min_dollar_vol
    seasoned = panels["close"].notna().cumsum() >= min_history
    priced = panels["close"].notna() & panels["open"].notna() & (dv > 0)
    return liquid & seasoned & priced


def build_weights(score, mask, panels,
                  neutral=True, inv_vol=True, vol_window=24 * 14,
                  target_vol=0.20, gross_cap=3.0, max_pos=0.10,
                  smooth=1, top_frac=None, bars_per_year=24 * 365,
                  vol_lookback=24 * 30):
    """score: (T x N) alpha, row t usable at close of bar t.

    Returns target weights where row t is what we want to HOLD during bar t,
    i.e. already shifted by one bar relative to the score.
    """
    s = score.where(mask)

    if top_frac is not None:
        # keep only the strongest tail on each side, equal-weight within it
        r = s.rank(axis=1, pct=True)
        n = mask.sum(axis=1)
        lo, hi = top_frac, 1 - top_frac
        s = pd.DataFrame(np.where(r <= lo, -1.0, np.where(r >= hi, 1.0, 0.0)),
                         index=s.index, columns=s.columns).where(s.notna())
        s = s.where(n.gt(10), np.nan)

    if neutral:                       # dollar-neutral: strip the market bet
        s = s.sub(s.mean(axis=1), axis=0).where(s.notna())

    if inv_vol:                       # equalise risk contribution across names
        v = np.log(panels["close"]).diff().rolling(
            vol_window, min_periods=vol_window // 3).std()
        # Floor is a fixed constant, NOT a sample quantile: a quantile taken over
        # the whole panel would leak future information into every past weight.
        s = s / v.where(mask).clip(lower=VOL_FLOOR)

    gross = s.abs().sum(axis=1).replace(0, np.nan)
    w = s.div(gross, axis=0).fillna(0.0)          # gross = 1 before sizing
    w = w.clip(-max_pos, max_pos)

    if smooth > 1:                                 # EWMA the target to cut turnover
        w = w.ewm(span=smooth, min_periods=1).mean()

    # --- volatility targeting on the realised portfolio return (past only) ---
    ret = np.log(panels["close"]).diff()
    port_r = (w.shift(1) * ret).sum(axis=1)
    rv = port_r.rolling(vol_lookback, min_periods=vol_lookback // 3).std() * np.sqrt(bars_per_year)
    scale = (target_vol / rv.replace(0, np.nan)).clip(upper=gross_cap).fillna(0.0)
    w = w.mul(scale, axis=0)

    g = w.abs().sum(axis=1)
    over = g > gross_cap
    w.loc[over] = w.loc[over].div(g[over], axis=0) * gross_cap

    return w.shift(1).fillna(0.0)                  # <-- the no-look-ahead lag


def information_coefficient(score, fwd_ret, mask):
    """Spearman IC per bar between the score and the next-bar return."""
    s = score.where(mask)
    f = fwd_ret.where(mask)
    sr = s.rank(axis=1)
    fr = f.rank(axis=1)
    sr = sr.sub(sr.mean(axis=1), axis=0)
    fr = fr.sub(fr.mean(axis=1), axis=0)
    num = (sr * fr).sum(axis=1)
    den = np.sqrt((sr ** 2).sum(axis=1) * (fr ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    return ic.where(mask.sum(axis=1) >= 10)
