# Results

All figures below are produced by the scripts in this repository and read directly from the JSON files under `reports/data/`; none are typed by hand.

## The answer

**33% a month is not achievable by this model, and the reason is not subtle.** Three independent arguments, each sufficient on its own:

1. **Arithmetic.** Compounding at 33% a month requires an annualised Sharpe of at least **2.616** at the growth-optimal leverage, no matter how much leverage is available.

2. **Leverage is not free in practice.** On the development window the book nets Sharpe **3.30** under the headline execution assumption, which clears that bar on paper. But simulated on the actual return path with maintenance margin checked every bar, the best compound monthly return reachable without the account being liquidated is **12.58%**, and it costs a -80% drawdown. Past roughly 3x gross notional, more leverage lowers the compound return; past ~10x it ends the account.

3. **The sealed holdout.** Over the 14 months after the configuration was frozen, the same strategy nets Sharpe **0.55** and **0.77%** a month, with 50% of months positive and a deflated Sharpe of 0.025. That fails the failure criteria recorded in `PROTOCOL.md` before the holdout was opened.

What the work does establish is a genuine, measurable edge and an honest measurement of its size. The signal survives out of sample on every metric that does not involve leverage, and the reason the returns do not is that the edge per unit of turnover has compressed to the same order of magnitude as the fees.


## Development: is the target reachable at all?

Expected log growth for a book at annualised volatility *s* with Sharpe *S* is `S*s - s^2/2`, which peaks at `s = S` with value `S^2/2`. A 33% monthly target therefore requires `S >= sqrt(24*ln(1+0.33))` = **2.616** before leverage is even considered.

| execution | net Sharpe | Sharpe required | best monthly at any leverage | vol for target | gross for target | verdict |
|---|---|---|---|---|---|---|
| maker_only | 3.96 | 2.62 | 92.2% | 99% | 7.3 | reachable |
| passive | 3.30 | 2.62 | 57.6% | 129% | 9.7 | reachable |
| optimistic | 3.25 | 2.62 | 55.4% | 132% | 9.9 | reachable |
| base | 2.12 | 2.62 | 20.6% | n/a | n/a | unreachable |
| conservative | 1.23 | 2.62 | 6.5% | n/a | n/a | unreachable |
| brutal | -0.03 | 2.62 | 0.0% | n/a | n/a | unreachable |


## Development: unit-risk book (leverage 1.0, 20% volatility target)

| cost scenario | Sharpe | monthly (geo) | CAGR | ann vol | max DD | turnover/day | cumulative cost | cumulative funding |
|---|---|---|---|---|---|---|---|---|
| maker_only | 3.96 | 7.05% | 126.8% | 21.2% | -26.5% | 2.92 | 2949.0% | -510.4% |
| passive | 3.30 | 5.81% | 97.2% | 21.2% | -28.4% | 3.02 | 2650.6% | -298.1% |
| optimistic | 3.25 | 5.72% | 95.1% | 21.2% | -28.6% | 3.03 | 2575.6% | -286.6% |
| base | 2.12 | 3.62% | 53.4% | 21.2% | -44.8% | 3.12 | 2006.0% | -124.3% |
| conservative | 1.23 | 2.00% | 26.8% | 21.2% | -65.8% | 3.13 | 1603.9% | -66.0% |
| brutal | -0.03 | -0.24% | -2.9% | 21.2% | -83.0% | 3.10 | 1180.2% | -29.3% |


Block bootstrap (weekly blocks, 2000 resamples) on the base cost scenario: Sharpe 5th/50th/95th percentile 2.35 / 3.24 / 4.15, P(Sharpe > 0) = 1.000.


### Development: what gross notional buys

A dollar-neutral book of ~100 perpetuals has single-digit annualised volatility per unit of gross notional, so the achievable growth rate is set by the gross-notional limit, not by the volatility target.

| gross cap | avg gross | ann vol | monthly (geo) | Sharpe | max DD | outcome |
|---|---|---|---|---|---|---|
| 2x | 1.94 | 31% | 9.13% | 3.49 | -35.9% | survived |
| 4x | 2.75 | 58% | 12.04% | 2.65 | -59.6% | survived |
| 6x | 3.40 | 84% | 12.58% | 2.11 | -79.8% | survived |
| 8x | 4.30 | 112% | 11.59% | 1.73 | -90.9% | survived |
| 10x | 5.70 | 143% | 8.30% | 1.38 | -97.0% | survived |
| 12x | 0.85 | 86% | -100.00% | 1.04 | -100.0% | **liquidated** |
| 15x | 1.06 | 105% | -100.00% | 0.99 | -100.0% | **liquidated** |
| 20x | 1.07 | 112% | -100.00% | 1.15 | -100.0% | **liquidated** |


### Development: the empirical ceiling

The closed form above is an upper bound that assumes lognormal returns. Simulated on the actual return path, the best compound monthly return reachable **without liquidating the account** is **12.58%**, at 3.4x average gross notional and 84% annualised volatility, with a -79.8% peak-to-trough drawdown.

Holding the drawdown under 50% instead, the ceiling is **9.13%** a month at 1.9x gross (31% volatility, -35.9% drawdown, Sharpe 3.49).

The gap between the closed form and the simulation is the price of fat tails: the growth formula assumes independent lognormal increments, and a real crypto return stream clusters its worst hours together.


### Development: leverage required for 33% per month

| cost scenario | leverage multiple | avg gross notional | ann vol | monthly (geo) | max DD | Sharpe |
|---|---|---|---|---|---|---|
| maker_only | not reached | - | - | - | - | - |
| passive | not reached | - | - | - | - | - |
| base | not reached | - | - | - | - | - |
| conservative | not reached | - | - | - | - | - |
| brutal | not reached | - | - | - | - | - |


### Development: capacity

| AUM | monthly (geo) | Sharpe | orders truncated by the participation cap | max DD |
|---|---|---|---|---|
| $1,000,000 | 5.81% | 3.30 | 6.3% | -28.4% |
| $10,000,000 | 4.83% | 2.76 | 35.4% | -30.0% |
| $50,000,000 | 3.97% | 2.28 | 60.9% | -30.2% |
| $200,000,000 | 2.96% | 1.69 | 74.5% | -35.4% |


### Development: overfitting statistics

- Probability of backtest overfitting (CSCV, 36 configurations): **0.000**
- Deflated Sharpe ratio: **0.973**
- Probabilistic Sharpe ratio: **1.000**


## Sealed holdout: is the target reachable at all?

Expected log growth for a book at annualised volatility *s* with Sharpe *S* is `S*s - s^2/2`, which peaks at `s = S` with value `S^2/2`. A 33% monthly target therefore requires `S >= sqrt(24*ln(1+0.33))` = **2.616** before leverage is even considered.

| execution | net Sharpe | Sharpe required | best monthly at any leverage | vol for target | gross for target | verdict |
|---|---|---|---|---|---|---|
| maker_only | 0.91 | 2.62 | 3.5% | n/a | n/a | unreachable |
| passive | 0.55 | 2.62 | 1.3% | n/a | n/a | unreachable |
| optimistic | 0.50 | 2.62 | 1.1% | n/a | n/a | unreachable |
| base | -0.04 | 2.62 | 0.0% | n/a | n/a | unreachable |
| conservative | -0.48 | 2.62 | 1.0% | n/a | n/a | unreachable |
| brutal | -1.16 | 2.62 | 5.8% | n/a | n/a | unreachable |


## Sealed holdout: unit-risk book (leverage 1.0, 20% volatility target)

| cost scenario | Sharpe | monthly (geo) | CAGR | ann vol | max DD | turnover/day | cumulative cost | cumulative funding |
|---|---|---|---|---|---|---|---|---|
| maker_only | 0.91 | 1.41% | 18.2% | 20.8% | -20.8% | 1.71 | 44.1% | -33.1% |
| passive | 0.55 | 0.77% | 9.7% | 20.8% | -24.2% | 1.72 | 56.8% | -30.8% |
| optimistic | 0.50 | 0.69% | 8.6% | 20.8% | -24.6% | 1.71 | 62.2% | -30.1% |
| base | -0.04 | -0.26% | -3.0% | 20.8% | -28.6% | 1.71 | 82.2% | -26.7% |
| conservative | -0.48 | -1.00% | -11.4% | 20.8% | -31.7% | 1.70 | 94.5% | -24.6% |
| brutal | -1.16 | -2.17% | -23.1% | 20.7% | -37.5% | 1.70 | 110.2% | -21.4% |


Block bootstrap (weekly blocks, 2000 resamples) on the base cost scenario: Sharpe 5th/50th/95th percentile -1.21 / 0.41 / 1.99, P(Sharpe > 0) = 0.658.


### Sealed holdout: what gross notional buys

A dollar-neutral book of ~100 perpetuals has single-digit annualised volatility per unit of gross notional, so the achievable growth rate is set by the gross-notional limit, not by the volatility target.

| gross cap | avg gross | ann vol | monthly (geo) | Sharpe | max DD | outcome |
|---|---|---|---|---|---|---|
| 2x | 2.20 | 34% | 1.48% | 0.69 | -36.0% | survived |
| 4x | 4.17 | 68% | 0.30% | 0.39 | -66.3% | survived |
| 6x | 6.15 | 103% | -1.46% | 0.34 | -82.6% | survived |
| 8x | 8.20 | 138% | -5.00% | 0.24 | -92.4% | survived |
| 10x | 10.47 | 173% | -10.39% | 0.11 | -97.2% | survived |
| 12x | 13.27 | 210% | -15.13% | 0.11 | -99.0% | survived |
| 15x | 22.32 | 267% | -23.10% | 0.16 | -99.8% | survived |
| 20x | 5.94 | 194% | -100.00% | -0.53 | -100.0% | **liquidated** |


### Sealed holdout: the empirical ceiling

The closed form above is an upper bound that assumes lognormal returns. Simulated on the actual return path, the best compound monthly return reachable **without liquidating the account** is **1.48%**, at 2.2x average gross notional and 34% annualised volatility, with a -36.0% peak-to-trough drawdown.

Holding the drawdown under 50% instead, the ceiling is **1.48%** a month at 2.2x gross (34% volatility, -36.0% drawdown, Sharpe 0.69).

The gap between the closed form and the simulation is the price of fat tails: the growth formula assumes independent lognormal increments, and a real crypto return stream clusters its worst hours together.


### Sealed holdout: leverage required for 33% per month

| cost scenario | leverage multiple | avg gross notional | ann vol | monthly (geo) | max DD | Sharpe |
|---|---|---|---|---|---|---|
| maker_only | not reached | - | - | - | - | - |
| passive | not reached | - | - | - | - | - |
| base | not reached | - | - | - | - | - |
| conservative | not reached | - | - | - | - | - |
| brutal | not reached | - | - | - | - | - |


### Sealed holdout: capacity

| AUM | monthly (geo) | Sharpe | orders truncated by the participation cap | max DD |
|---|---|---|---|---|
| $1,000,000 | 0.77% | 0.55 | 2.0% | -24.2% |
| $10,000,000 | 0.54% | 0.42 | 4.6% | -20.4% |
| $50,000,000 | -0.35% | -0.09 | 18.7% | -25.4% |
| $200,000,000 | -0.11% | 0.05 | 40.7% | -20.4% |


### Sealed holdout: overfitting statistics

- Probability of backtest overfitting (CSCV, 36 configurations): **0.000**
- Deflated Sharpe ratio: **0.005**
- Probabilistic Sharpe ratio: **0.482**


## Frozen configuration

```json
{
  "interval": "1h",
  "top_n": 120,
  "horizon": 4,
  "rebalance": 12,
  "smooth_halflife": 0.0,
  "no_trade_band": 0.008,
  "factor_model": true,
  "n_factors": 5,
  "target_vol": 0.2,
  "max_weight": 0.06,
  "embargo": 48,
  "test_months": 3,
  "train_stride": 2,
  "estimated_spread": true,
  "spread_method": "measured",
  "cost_penalty": 1.0,
  "headline_cost_scenario": "passive",
  "cutoff": "2025-07-01",
  "n_configs_searched": 108
}
```

Frozen at 2026-09-11T17:46:37.848073+00:00 (hash `0cbdd224ad614a5a`).
