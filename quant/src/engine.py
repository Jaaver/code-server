"""Realistic portfolio backtester for USD-M perpetual futures.

Timing convention (no look-ahead anywhere):
    signal for bar t is built from data up to and including the CLOSE of bar t-1
    the resulting trade is executed at the OPEN of bar t
    the position held through bar t bears the gap risk from close(t-1) -> open(t)

Costs modelled:
    * exchange fees (taker/maker)
    * bid-ask half-spread, estimated per symbol per bar from Abdi-Ranaldo (2017)
    * square-root market impact, scaled by participation in that bar's volume
    * perpetual funding paid/received on notional at each 8h funding stamp
    * margin: gross leverage cap, maintenance margin, and hard liquidation
"""
import numpy as np

# --- Binance USD-M fee schedule (VIP 0, no BNB discount) ---
TAKER_FEE = 4.5e-4      # 0.045%
MAKER_FEE = 2.0e-4      # 0.020%

# --- microstructure defaults ---
MIN_HALF_SPREAD = 0.5e-4   # 0.5 bp floor: not even BTC is tighter round-trip
MAX_HALF_SPREAD = 50e-4    # 50 bp cap on the estimator's tail
IMPACT_ETA = 0.5           # Almgren temporary-impact coefficient
IMPACT_EXP = 0.6           # Almgren exponent on participation


def abdi_ranaldo_half_spread(high, low, close, window=48):
    """Rolling Abdi-Ranaldo (2017) effective half-spread estimate, in fraction of price.

    S^2 = 4 * E[(c_t - eta_t) * (c_t - eta_{t+1})], eta = midpoint of log(high),log(low).
    Negative estimates (noise) are floored; result is clipped to a sane band.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.log(close)
        eta = 0.5 * (np.log(high) + np.log(low))
    prod = (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    prod = np.vstack([np.full((1, close.shape[1]), np.nan), prod])

    # rolling mean ignoring NaN
    out = np.full_like(prod, np.nan, dtype=np.float64)
    cs = np.nancumsum(np.nan_to_num(prod), axis=0)
    cnt = np.cumsum(~np.isnan(prod), axis=0)
    out[window:] = (cs[window:] - cs[:-window]) / np.maximum(cnt[window:] - cnt[:-window], 1)
    s2 = np.maximum(4.0 * out, 0.0)
    half = 0.5 * np.sqrt(s2)
    half = np.clip(np.nan_to_num(half, nan=5e-4), MIN_HALF_SPREAD, MAX_HALF_SPREAD)
    return half


def tick_half_spread(close, ticks, spread_in_ticks=2.0, floor_bp=0.1, cap_bp=40.0):
    """Half-spread from each symbol's measured tick size.

    Abdi-Ranaldo on hourly bars conflates volatility with spread: it puts BTC's
    mean half-spread at ~4.7bp when the instrument's whole tick is 0.15bp. The
    exchange tick is directly measurable from the price series and bounds the
    spread from below, so we model spread as a small multiple of it (2 ticks is
    conservative for a liquid perp) and let the impact term carry size effects.
    """
    t = np.asarray(ticks, dtype=np.float64)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        hs = 0.5 * spread_in_ticks * t / close
    hs = np.nan_to_num(hs, nan=floor_bp * 1e-4, posinf=cap_bp * 1e-4)
    return np.clip(hs, floor_bp * 1e-4, cap_bp * 1e-4)


def measure_ticks(close):
    """Smallest genuine price increment per symbol, from observed closes."""
    out = np.full(close.shape[1], np.nan)
    for j in range(close.shape[1]):
        c = close[:, j]
        c = c[np.isfinite(c)]
        if len(c) < 500:
            continue
        d = np.abs(np.diff(c))
        d = d[d > 0]
        if len(d) < 100:
            continue
        out[j] = np.percentile(d, 0.5)
    med = np.nanmedian(out)
    return np.where(np.isfinite(out), out, med if np.isfinite(med) else 1e-6)


class Backtester:
    def __init__(self, op, hi, lo, cl, dv, funding,
                 fee=TAKER_FEE, maker_frac=0.0,
                 impact_coef=IMPACT_ETA, capital=1_000_000.0,
                 max_gross_leverage=3.0, maint_margin=0.005,
                 max_participation=0.05, half_spread=None, rebalance_band=0.0):
        """All panels are (T x N) float arrays aligned on the same hourly grid.

        dv        : quote (USD) volume traded in that bar
        funding   : funding rate stamped at its settlement bar, NaN elsewhere
        maker_frac: fraction of flow assumed to earn the maker fee instead of taker
        max_participation: cap on our share of a bar's volume (a liquidity limit,
                    enforced by truncating the trade, not by ignoring it)
        """
        self.op, self.hi, self.lo, self.cl = op, hi, lo, cl
        self.dv = np.nan_to_num(dv, nan=0.0)
        self.fund = np.nan_to_num(funding, nan=0.0)
        self.fee = fee * (1 - maker_frac) + MAKER_FEE * maker_frac
        self.impact_coef = impact_coef
        self.capital = capital
        self.max_gross = max_gross_leverage
        self.maint = maint_margin
        self.max_part = max_participation
        # Do not trade a name until its weight has drifted more than this
        # fraction of equity from target. Without it the book is forced to
        # re-true every position every bar purely because prices moved, which
        # is turnover a real desk would never pay.
        self.rebalance_band = rebalance_band
        self.T, self.N = cl.shape

        if half_spread is None:
            # tick-grounded, not Abdi-Ranaldo: see tick_half_spread for why
            self.ticks = measure_ticks(cl)
            self.hs = tick_half_spread(cl, self.ticks)
        else:
            self.hs = half_spread
        # per-bar volatility, used for the impact term
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.diff(np.log(cl), axis=0)
        r = np.vstack([np.full((1, self.N), np.nan), r])
        self.bar_vol = _rolling_std(r, 168)
        # A bar with no volume means we cannot TRADE the symbol, not that we must
        # exit it. Conflating the two force-closes and reopens every position in
        # any quiet hour, which manufactures enormous turnover and destroys the
        # holding-period return. Keep the two concepts separate:
        self.priced = np.isfinite(op) & np.isfinite(cl)      # can hold and mark
        self.tradable = self.priced & (self.dv > 0)          # can adjust
        self.valid = self.priced

    def run(self, target_w, verbose=False):
        """target_w[t] = desired weight (fraction of equity) for the position held
        during bar t; it must be computable from information up to close(t-1)."""
        T, N = self.T, self.N
        op, cl = np.nan_to_num(self.op), np.nan_to_num(self.cl)

        equity = np.empty(T); equity[:] = np.nan
        eq = self.capital
        units = np.zeros(N)              # signed position size in base units
        prev_cl = np.where(self.valid[0], cl[0], np.nan)

        costs_t = np.zeros(T); fund_t = np.zeros(T); turn_t = np.zeros(T)
        gross_t = np.zeros(T); npos_t = np.zeros(T)
        liquidated_at = None

        for t in range(1, T):
            o, c = op[t], cl[t]
            ok = self.priced[t]
            can_trade = self.tradable[t]

            # --- 1. PnL from close(t-1) to open(t) on the OLD position (gap risk) ---
            gap = np.where(np.isfinite(prev_cl) & ok, o - np.nan_to_num(prev_cl), 0.0)
            eq += float(units @ gap)

            if eq <= 0:
                liquidated_at = t; equity[t:] = 0.0; break

            # --- 2. Rebalance at open(t), only where the symbol actually trades ---
            w = np.nan_to_num(target_w[t]) * can_trade
            gross = np.abs(w).sum()
            if gross > self.max_gross:                    # enforce leverage cap
                w *= self.max_gross / gross
            # where we cannot trade, we simply keep what we already hold
            tgt_units = np.where(can_trade & (o > 0), w * eq / np.where(o > 0, o, 1.0), units)
            d_units = np.where(can_trade, tgt_units - units, 0.0)

            if self.rebalance_band > 0:
                drift = np.abs(d_units) * o / max(eq, 1e-9)
                # always allow a full exit, otherwise require meaningful drift
                keep = (drift < self.rebalance_band) & (np.abs(tgt_units) > 0)
                d_units = np.where(keep, 0.0, d_units)

            # liquidity cap: cannot trade more than max_part of the bar's volume
            cap_units = np.where(o > 0, self.max_part * self.dv[t] / np.where(o > 0, o, 1.0), 0.0)
            d_units = np.clip(d_units, -cap_units, cap_units)
            units = units + d_units

            trade_notional = np.abs(d_units) * o
            tot_trade = trade_notional.sum()
            if tot_trade > 0:
                # Almgren temporary impact on DAILY-equivalent quantities:
                #   impact = eta * sigma_daily * (Q / ADV_daily)^0.6
                # The plain per-bar square-root form charged ~8bp for trading
                # 0.03% of a day's volume, which is an order that in practice
                # crosses the touch and pays little beyond the spread.
                adv_daily = self.dv[t] * 24.0
                part = np.divide(trade_notional, adv_daily,
                                 out=np.zeros(N), where=adv_daily > 0)
                vol_daily = np.nan_to_num(self.bar_vol[t], nan=0.02) * np.sqrt(24.0)
                slip = self.hs[t] + self.impact_coef * vol_daily * np.power(
                    np.clip(part, 0, 1), IMPACT_EXP)
                cost = float((trade_notional * (self.fee + slip)).sum())
                eq -= cost
                costs_t[t] = cost
                turn_t[t] = tot_trade

            # --- 3. Funding on the notional we hold through this bar ---
            fr = self.fund[t]
            if fr.any():
                pay = float((units * o * fr).sum())       # long pays a positive rate
                eq -= pay
                fund_t[t] = pay

            # --- 4. PnL from open(t) to close(t) on the NEW position ---
            eq += float(units @ np.where(ok, c - o, 0.0))

            # --- 5. Margin check ---
            notional = float(np.abs(units * np.where(ok, c, 0.0)).sum())
            gross_t[t] = notional / eq if eq > 0 else np.inf
            npos_t[t] = int((np.abs(units) > 0).sum())
            if eq <= self.maint * notional or eq <= 0:
                liquidated_at = t
                equity[t] = max(eq, 0.0); equity[t + 1:] = max(eq, 0.0)
                break

            # only a symbol that stops being PRICED at all (delisting) is force-closed,
            # at its last valid price, paying the exit cost like any other trade
            dead = (~ok) & (units != 0)
            if dead.any():
                exit_notional = np.abs(units * np.nan_to_num(prev_cl)) * dead
                exit_cost = float((exit_notional * (self.fee + self.hs[t])).sum())
                eq -= exit_cost
                costs_t[t] += exit_cost
                turn_t[t] += float(exit_notional.sum())
                units = np.where(dead, 0.0, units)

            equity[t] = eq
            units = np.where(ok, units, 0.0)
            prev_cl = np.where(ok, c, np.nan)

        equity[0] = self.capital
        equity = _ffill(equity)
        return {
            "equity": equity, "costs": costs_t, "funding": fund_t,
            "turnover": turn_t, "gross": gross_t, "n_pos": npos_t,
            "liquidated_at": liquidated_at,
        }


def _rolling_std(x, w):
    out = np.full_like(x, np.nan, dtype=np.float64)
    xn = np.nan_to_num(x)
    m = (~np.isnan(x)).astype(np.float64)
    cs, cs2, cm = np.cumsum(xn, 0), np.cumsum(xn ** 2, 0), np.cumsum(m, 0)
    n = cm[w:] - cm[:-w]
    s = cs[w:] - cs[:-w]
    s2 = cs2[w:] - cs2[:-w]
    n = np.maximum(n, 1)
    var = s2 / n - (s / n) ** 2
    out[w:] = np.sqrt(np.maximum(var, 0))
    return out


def _ffill(a):
    idx = np.where(np.isfinite(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def stats(equity, bars_per_year=24 * 365, capital=None):
    eq = np.asarray(equity, dtype=float)
    eq = np.where(eq <= 0, 1e-9, eq)
    r = np.diff(np.log(eq))
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return {}
    ann_ret = np.exp(r.mean() * bars_per_year) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(bars_per_year)
    sharpe = (r.mean() / r.std(ddof=1)) * np.sqrt(bars_per_year) if r.std() > 0 else 0.0
    dn = r[r < 0]
    sortino = (r.mean() / dn.std(ddof=1)) * np.sqrt(bars_per_year) if len(dn) > 2 and dn.std() > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    maxdd = dd.min()
    years = len(r) / bars_per_year
    total = eq[-1] / eq[0] - 1
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else -1
    monthly = (1 + cagr) ** (1 / 12) - 1
    return {
        "total_return": total, "cagr": cagr, "monthly_equiv": monthly,
        "ann_vol": ann_vol, "sharpe": sharpe, "sortino": sortino,
        "max_dd": maxdd, "calmar": cagr / abs(maxdd) if maxdd < 0 else np.inf,
        "years": years, "final_equity": eq[-1],
    }
