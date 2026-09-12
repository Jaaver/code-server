# Results

All figures below are produced by the scripts in this repository and read directly from the JSON files under `reports/data/`; none are typed by hand.

## The answer

**33% a month is not achievable by this model, and the reason is not subtle.** Three independent arguments, each sufficient on its own:

1. **Arithmetic.** Compounding at 33% a month requires an annualised Sharpe of at least **2.616** at the growth-optimal leverage, no matter how much leverage is available.

2. **Leverage is not free in practice.** On the development window the book nets Sharpe **3.30** under the headline execution assumption, which clears that bar on paper. But simulated on the actual return path with maintenance margin checked every bar, the best compound monthly return reachable without the account being liquidated is **12.58%**, and it costs a -80% drawdown. Beyond that point more leverage *lowers* the compound return, and beyond a gross-notional cap of about 10x it liquidates the account outright.

3. **The sealed holdout.** Over the 14 months after the configuration was frozen, the same strategy nets Sharpe **0.55** and **0.77%** a month, with 50% of months positive and a deflated Sharpe of 0.025. That fails the failure criteria recorded in `PROTOCOL.md` before the holdout was opened.

What the work does establish is a genuine, measurable edge and an honest measurement of its size. The signal survives out of sample on every metric that does not involve leverage, and the reason the returns do not is that the edge per unit of turnover has compressed to the same order of magnitude as the fees.


## Data and method

- Binance USD-margined perpetual futures, 1 hour bars from data.binance.vision (the exchange's public archive).
- 829 symbols, 58,440 bars, 2020-01-01 to 2026-08-31; 130 of them also carry the open-interest archive.
- **5,741,401 out-of-sample predictions** from 2021-02-01 to 2026-08-31, each produced by an ensemble retrained quarterly on bars that preceded it, with a purge and a 48-bar embargo before every test window.
- Transaction costs are not assumed: the effective spread was measured from the exchange's aggregated-trade archive across 42 symbol-days spanning the liquidity spectrum. Median half-spread **1.08 bp** (0.16 bp to 2.79 bp, 10th to 90th percentile). Fees, not spread, dominate: the exchange charges 5 bp taker and 2 bp maker.


### Signal decay

| forward horizon (bars) | IC | t-stat | % cross-sections positive |
|---|---|---|---|
| 1 | 0.0733 | 105.7 | 70.7% |
| 2 | 0.0825 | 116.9 | 72.6% |
| 4 | 0.0880 | 123.3 | 73.9% |
| 8 | 0.0902 | 126.5 | 74.1% |
| 12 | 0.0903 | 126.5 | 74.3% |
| 24 | 0.0839 | 119.3 | 73.3% |
| 48 | 0.0738 | 105.7 | 70.8% |
| 96 | 0.0710 | 104.6 | 70.5% |


### Stability by year

| year | IC | t-stat |
|---|---|---|
| 2021 | 0.1288 | 80.3 |
| 2022 | 0.0944 | 67.6 |
| 2023 | 0.0754 | 54.6 |
| 2024 | 0.0653 | 44.7 |
| 2025 | 0.0708 | 28.8 |


### Stability by market-volatility regime

| regime | IC | t-stat |
|---|---|---|
| low_vol | 0.0779 | 65.6 |
| mid_vol | 0.0893 | 73.0 |
| high_vol | 0.0961 | 75.0 |


## The full-sample Sharpe ratio is an average, and it hides a trend

Every row below is out-of-sample: each prediction comes from an ensemble fitted only on bars that preceded it. The book is run at a 20% volatility target with no extra leverage.


**maker_only execution** (full sample Sharpe 3.24)

| year | Sharpe | monthly (geo) | year return | max DD | turnover/day |
|---|---|---|---|---|---|
| 2021 | 9.68 | 19.28% | 595.4% | -14.0% | 2.17 |
| 2022 | 3.71 | 6.59% | 115.0% | -13.8% | 3.23 |
| 2023 | 1.66 | 2.79% | 39.1% | -8.6% | 3.61 |
| 2024 | 1.42 | 2.33% | 31.9% | -26.5% | 2.90 |
| 2025 | 2.30 | 3.88% | 57.9% | -15.4% | 2.08 |
| 2026 | -0.28 | -0.68% | -5.3% | -20.6% | 0.90 |


**passive execution** (full sample Sharpe 2.66)

| year | Sharpe | monthly (geo) | year return | max DD | turnover/day |
|---|---|---|---|---|---|
| 2021 | 9.31 | 18.47% | 545.2% | -14.1% | 2.17 |
| 2022 | 2.95 | 5.15% | 82.8% | -15.7% | 3.29 |
| 2023 | 0.57 | 0.83% | 10.5% | -12.1% | 3.79 |
| 2024 | 0.67 | 1.00% | 12.7% | -28.4% | 3.04 |
| 2025 | 2.29 | 3.84% | 57.3% | -16.0% | 2.22 |
| 2026 | -0.77 | -1.49% | -11.3% | -22.0% | 1.13 |


**base execution** (full sample Sharpe 1.64)

| year | Sharpe | monthly (geo) | year return | max DD | turnover/day |
|---|---|---|---|---|---|
| 2021 | 8.41 | 16.52% | 437.4% | -14.3% | 2.16 |
| 2022 | 1.22 | 1.98% | 26.5% | -23.6% | 3.39 |
| 2023 | -0.86 | -1.68% | -18.4% | -28.3% | 4.01 |
| 2024 | -0.19 | -0.51% | -5.9% | -33.7% | 3.10 |
| 2025 | 1.67 | 2.74% | 38.3% | -16.6% | 2.34 |
| 2026 | -1.10 | -2.04% | -15.2% | -25.0% | 1.39 |


The 2021 column is most of the full-sample result. It is also the year the universe was smallest, the market least institutional, and short-horizon cross-sectional reversal least competed-for. Whatever 2021 was, it is not the market the strategy would be deployed into.


## Why the returns decayed while the prediction did not

Rank information coefficient is a correlation, so it is scale-free: it can hold steady while the money drains out. What a dollar-neutral book actually earns is information coefficient multiplied by cross-sectional dispersion, and in a maturing market the dispersion falls. Both are below, per half-year, for the 12-hour horizon the strategy trades.

| period | IC | t | top-minus-bottom decile | cross-sectional dispersion |
|---|---|---|---|---|
| 2021H1 | 0.1246 | 48.5 | 134.0 bp | 453 bp |
| 2021H2 | 0.1177 | 56.8 | 57.2 bp | 355 bp |
| 2022H1 | 0.0936 | 44.6 | 17.6 bp | 289 bp |
| 2022H2 | 0.0882 | 46.0 | 30.0 bp | 224 bp |
| 2023H1 | 0.0952 | 47.7 | 18.0 bp | 229 bp |
| 2023H2 | 0.0706 | 36.0 | 10.3 bp | 258 bp |
| 2024H1 | 0.0761 | 34.4 | 16.8 bp | 304 bp |
| 2024H2 | 0.0640 | 32.6 | 9.6 bp | 297 bp |
| 2025H1 | 0.0891 | 37.9 | 39.5 bp | 359 bp |
| 2025H2 | 0.1099 | 49.3 | 13.9 bp | 459 bp |
| 2026H1 | 0.0760 | 34.5 | 19.5 bp | 488 bp |
| 2026H2 | 0.0863 | 21.3 | 96.4 bp | 478 bp |

Against a round-trip cost of roughly 6 bp in the headline scenario, a decile spread of 70+ bp is a business and a spread in the low teens is not. The prediction is still there; the prize is not.


## The bar: single-feature analytic baselines

Each baseline is one line of signal logic run through the identical simulator and cost model.

| baseline | IC | Sharpe | monthly (geo) | max DD | turnover/day |
|---|---|---|---|---|---|
| reversal_1h | 0.0320 | -3.64 | -6.92% | -97.9% | 5.42 |
| reversal_4h | 0.0392 | -4.32 | -8.20% | -99.1% | 5.32 |
| reversal_24h | 0.0377 | -2.52 | -4.89% | -94.9% | 3.59 |
| reversal_72h | 0.0317 | -1.47 | -2.91% | -85.4% | 2.16 |
| ofi_4h_reversal | 0.0123 | -6.64 | -10.86% | -99.8% | 6.01 |
| ofi_4h_momentum | -0.0123 | -5.67 | -9.31% | -99.5% | 6.05 |
| funding_carry | -0.0047 | -2.26 | -4.11% | -92.4% | 2.89 |
| volume_shock_reversal | 0.0094 | -3.80 | -6.77% | -97.8% | 5.42 |


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


## Against the criteria recorded before the holdout was opened

| criterion | result | value |
|---|---|---|
| Out-of-sample net Sharpe at or above 1.0 | **fail** | 0.55 |
| Probability of backtest overfitting at or below 0.50 | **pass** | 0.000 |
| Deflated Sharpe at or above 0.95 | **fail** | 0.025 |
| Holdout monthly at least a third of development, same leverage | **fail** | 0.77% vs 5.81% |
| 33%/month reachable at the headline assumption | **fail** | needs 2.62, has 0.55 |


4 of 5 failed. The criteria were written down in `reports/PROTOCOL.md` before the holdout period was evaluated, precisely so this verdict could not be renegotiated afterwards.


## A second attempt to raise the Sharpe ratio

Because Sharpe is the binding constraint, a second research pass spent compute on the three things most likely to raise it: a wider universe (150 names instead of 120, since breadth raises the information ratio roughly as its square root), an ensemble over two label horizons rather than one, and an 18-month half-life on the training weights, justified by the decay visible in the development window alone.

| execution | Sharpe dev (base model) | Sharpe dev (enhanced) | Sharpe holdout (base model) | Sharpe holdout (enhanced) |
|---|---|---|---|---|
| maker_only | 3.96 | 4.36 | 0.91 | 0.79 |
| passive | 3.30 | 3.87 | 0.55 | 0.81 |
| base | 2.12 | 2.90 | -0.04 | 0.33 |

This is a second look at the holdout and therefore weaker evidence than the first; it is reported separately for that reason rather than folded into the headline.


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


## Forward paper trading through the production code path

`scripts/paper_forward.py` replays the holdout one bar at a time through the same `LiveStrategy` and broker objects a deployment would use, handing the strategy only a rolling window of history. It is the strongest test available short of sending real orders, and it is the one that would catch a backtest quietly using information a live system could not have.

- Window: 6 months from the cutoff, 999,508 of starting capital.
- Net Sharpe **-0.41**, geometric monthly **-0.89%**, CAGR -10.1%, annualised volatility 20.8%, max drawdown -20.1%, average gross 0.98x.
- Of the starting capital, **14.1%** went to trading costs and **-7.5%** to funding. That is the whole story of the holdout in two numbers: the gross signal is real and roughly that size.

- Note: recomputed from the equity curve; the first run reported a Sharpe ratio computed on a return series that excluded rebalance costs.
