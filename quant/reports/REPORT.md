# Can a trading model produce 33% monthly returns?

**Verdict: no — not by a wide margin, and the gap is structural rather than a
matter of finding a better signal.**

I built a complete systematic trading system for crypto perpetual futures,
validated it out-of-sample and across a second exchange, and measured what it
can actually compound at. The honest answer:

| | Achievable | Target |
|---|---|---|
| Best monthly growth at **any** leverage | **+1.91%** | +33% |
| Net Sharpe (full sample 2020-2026) | **0.60** | ≥2.62 required |
| Net Sharpe (held-out test window) | **0.87** | ≥2.62 required |

33%/month is ~17x the maximum growth rate this strategy can compound at, and the
shortfall cannot be closed with leverage — past a certain point leverage *reduces*
compound growth and then liquidates the account.

---

## 1. Why 33%/month is a Sharpe problem, not a leverage problem

33% monthly compounds to 2,963%/year, or 3.42 in log terms. For a strategy with
Sharpe `S` run at annual volatility `s`, log growth is

```
g = S·s − s²/2
```

which is maximised at `s* = S`, giving `g_max = S²/2`. That is the ceiling **no
amount of leverage can beat** — beyond `s*`, volatility drag exceeds the added
drift. Setting `S²/2 ≥ 3.42`:

| Fraction of Kelly actually run | Minimum Sharpe needed | Volatility that implies | 1-yr expected max drawdown |
|---|---|---|---|
| 100% (full Kelly) | 2.62 | 262% | ~96% |
| 50% | 3.02 | 151% | ~85% |
| 25% | 3.96 | 99% | ~71% |

So the target requires a **sustained net Sharpe of 2.6–4.0**. For reference, a
Sharpe above 2 net of costs, at capacity, is elite-tier for a systematic book.
The measured figure here is 0.60–0.87.

The leverage sweep below confirms the theory empirically.

---

## 2. What was built

**Data** — survivorship-bias-free. Every USD-M perpetual that ever listed on
Binance (860 symbols including delisted contracts), hourly OHLCV plus 8-hour
funding rates, 2020-01-01 → 2026-08-31: 58,440 hourly bars × 686 symbols after
filtering, 1.1 GB. 24 contracts in the panel are delisted, and they are retained.

The data audit passes clean: no irregular timestamps, no OHLC violations, no
non-positive prices, funding settling at the expected hours and within exchange
caps.

**Execution engine** — signal at `close(t−1)` → fill at `open(t)`, so the
close-to-open gap is borne by the *old* position. Models exchange fees
(maker/taker), bid-ask spread, square-root market impact, funding paid on
notional, gross-leverage caps, maintenance margin and liquidation, a
participation cap that truncates trades too large for the bar, and exit costs
on delisting.

The engine is verified against analytic P&L on a synthetic fixture (65.82 vs
65.97 bp/bar), returns exactly flat equity on zero weights, and is symmetric
under sign inversion of a foresight signal.

**Signals** — 25 alphas across momentum, reversal, carry, order flow, liquidity
and volatility families, each computed causally, crossed with 4 holding horizons
(4h/12h/24h/72h) to give 100 sleeves.

**Allocation** — each sleeve is weighted by its own *trailing realised
arithmetic P&L*, lagged one bar, shrunk toward equal weight. Nothing about a
signal's sign, horizon or inclusion is chosen in sample.

---

## 3. Four findings that changed the result

These cost most of the research time and each one inverted a conclusion.

**(a) Rank IC on log returns is the wrong objective.** Rank IC is invariant to
the log/arithmetic choice, but P&L is not. In a cross-section with this much
volatility dispersion, a handful of explosive names dominate the arithmetic sum,
and several signals have a strongly positive IC while *losing money*:

| alpha | IC (log) | Arithmetic Sharpe |
|---|---|---|
| `low_vol` | +3.36 | **+0.60** |
| `illiq` | +3.10 | **+0.83** |
| `xs_rev_72` | +0.93 | **−0.57** — sign flips |
| `xs_mom_168` | −0.37 | **+0.99** — sign flips |

The first ensemble was allocated on log-return IC and posted a Sharpe of **−5.06**:
it was loading on signals whose true tradable sign was the opposite. Rebuilding
the objective around realised arithmetic P&L fixed it.

**(b) "Cannot trade" is not "must exit".** The engine's validity mask required
non-zero volume every bar, so any symbol with a quiet hour was force-closed and
reopened. Low-volatility names have quiet hours constantly; turnover hit 379x/yr
and a genuine +3.20 bp/bar signal was ground down to +0.17. Separating *priced*
(can hold and mark) from *tradable* (can adjust) was essential.

**(c) The cost model was 4x too pessimistic.** The Abdi-Ranaldo spread estimator,
applied to hourly bars, conflates volatility with spread — it put BTC's mean
half-spread at 4.7bp when the instrument's entire tick is 0.147bp. Measuring tick
sizes directly from the price series and modelling spread as ~2 ticks gives a
median half-spread of 0.54bp, which matches real perp markets. The square-root
impact law was likewise miscalibrated for small orders, charging 8bp to trade
0.03% of a day's volume; the Almgren form on daily-equivalent quantities gives
~1.8bp.

**(d) Tuning hyperparameters on the backtest was worse than not tuning.** Across
36 settings of (rebalance cadence, no-trade band, allocation cap, sleeve
threshold), the rank correlation between train-window and test-window Sharpe was
**−0.66**:

- Config chosen by best training Sharpe: train **+1.57** → test **−0.13**
- Average config: test **+0.57**
- The `band=0.08` family had the best training numbers and ~zero test Sharpe, uniformly

The final model therefore **does not select hyperparameters at all** — it averages
the book over the entire grid, a rule that needs no hindsight.

---

## 4. Results

Final model, $1M, 70% maker fills, 30% target volatility, dollar-neutral.

| | Full sample 2020-2026 | Held-out test 2023-09 → 2026-08 |
|---|---|---|
| CAGR | +21.8% | +32.2% |
| Monthly equivalent | +1.66% | +2.35% |
| Annual volatility | 32.8% | 32.3% |
| **Sharpe** | **0.60** | **0.87** |
| Max drawdown | −75.1% | −43.4% |
| Months ≥ 33% | **0 of 80** | **0 of 36** |

Monthly distribution: mean +2.29%, median +1.77%, std 11.33%; 56% positive;
best month +27.0%, worst −24.2%. **Longest time underwater: 58 months.**

**Year by year:** +70.9% (2020), +145.6% (2021), −35.1% (2022), −32.4% (2023),
−19.1% (2024), +50.8% (2025), +66.1% (2026). Three consecutive losing years.

### The leverage frontier — the direct answer

| Leverage | Volatility | CAGR | Monthly | Max DD | Liquidated |
|---|---|---|---|---|---|
| ×0.5 | 16.3% | +14.2% | +1.11% | −39.7% | no |
| ×1 | 32.8% | +21.8% | +1.66% | −75.1% | no |
| **×2** | **64.8%** | **+25.5%** | **+1.91%** | **−96.0%** | no |
| ×3 | 97.4% | +18.6% | +1.43% | −99.5% | no |
| ×4 | 130.1% | +1.2% | +0.10% | −100.0% | no |
| ×6 | 196.5% | −40.4% | −4.23% | −100.0% | no |
| ×12 | 1185.7% | −99.4% | −35.04% | −100.0% | **yes** |

Growth peaks at **1.91%/month** and that peak already carries a 96% drawdown.
This matches theory: with S = 0.60, predicted max growth is S²/2 = 19.7%/yr =
1.51%/month. **More leverage cannot get to 33%/month; it destroys the account.**

### Robustness

- **Costs.** Gross (zero-cost) Sharpe 0.90 → 0.60 at baseline → **+0.21 at 3x
  costs**. Degrades gracefully; the edge is not a cost artifact.
- **Capacity.** Sharpe 0.70 at $100k, 0.60 at $1M, 0.45 at $20M, 0.21 at $500M.
  Impact scales as size^0.6, so this is a small-capital strategy.
- **Cross-exchange.** The same alphas run on OKX perpetuals (independent venue,
  fees, listings and flow): **78% sign agreement across 23 alphas, rank
  correlation +0.71**. The trend family replicates closely — `idio_mom` +3.33
  Binance / +3.05 OKX, `range_pos` +3.10 / +2.66, `xs_mom_168` +3.01 / +2.44.
  This is the strongest evidence the edge is a real market phenomenon.
- **Statistics.** PSR (P(Sharpe > 0)) = 0.94. Stationary-bootstrap 95% CI on
  Sharpe: **[−0.21, +1.47]**, P(Sharpe ≤ 0) = 7.1%. **Deflated Sharpe
  probability = 0.31** after correcting for 136 trials — below the conventional
  0.95 bar. The full-sample result is *not* statistically decisive; the
  out-of-sample window and the cross-exchange replication are the stronger
  evidence.
- **Risk of ruin.** P(>50% drawdown within a year) = 3.3% at ×1 leverage.

---

## 5. What would actually be required

To reach 33%/month one of these would have to be true, and none is:

1. **Sharpe ≥ 2.6 net of costs, sustained.** Measured: 0.60–0.87. Closing this
   needs roughly a 3–4x improvement in risk-adjusted edge, not a parameter tweak.
2. **Run at 260%+ volatility and accept ~96% expected drawdowns** — mathematically
   the full-Kelly point, practically indistinguishable from ruin, and the sweep
   above shows returns *collapse* before reaching that leverage because real
   costs and margin constraints bite first.
3. **A different opportunity set** — latency-sensitive market making, exchange
   arbitrage at co-located speed, or capacity of a few hundred thousand dollars
   in structurally inefficient venues. Those can post higher Sharpes, but they
   are infrastructure businesses, not a model, and their capacity is tiny.

Published claims of 33%/month sustained are, in my assessment, either
short-lucky-streak selection, unaccounted leverage risk, survivorship reporting,
or fraud. A strategy that genuinely compounded at 33%/month would turn $10k into
$1.6bn in five years.

## 6. What is defensible

A realistic deployment of this system:

- **1.1–1.9% per month** (14–25%/yr), at 16–65% volatility
- Drawdowns of **40–75%**, with underwater periods measured in **years**
- Capacity a few million dollars before decay becomes severe
- Requires continued research: three consecutive losing years (2022–2024) is
  well within this strategy's normal behaviour

That is a real, validated edge. It is roughly 1/17th of the target.

---

## Reproducing

```
python3 src/manifest.py        # enumerate the archive (survivorship-bias-free)
python3 src/download.py 32     # klines + funding for all 860 symbols
python3 src/dataset.py         # aligned panels
python3 src/audit.py           # data integrity
python3 src/alpha_eval.py      # rank alphas by arithmetic P&L
python3 src/oos.py             # strict train/test hyperparameter study
python3 src/final.py           # the ensemble model
python3 src/validate.py        # costs, capacity, leverage, statistics
python3 src/okx_check.py       # cross-exchange replication
```
