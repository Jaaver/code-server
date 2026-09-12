# trading-ai — cross-sectional machine-learning alpha on crypto perpetual futures

A complete research-to-execution stack for a market-neutral, cross-sectional
machine-learning trading model on Binance USD-margined perpetual futures, built to
answer one question honestly:

> **Can a trading model deliver 33% per month under realistic conditions — and at
> what risk?**

**The answer is no, and the evidence is in [`reports/RESULTS.md`](reports/RESULTS.md).**
Three independent arguments, each sufficient on its own:

1. **Arithmetic.** 33%/month requires an annualised Sharpe of at least **2.616** at the
   growth-optimal leverage, whatever leverage the venue allows.
2. **Leverage is not free.** On the development window the book nets Sharpe 3.30, which
   clears that bar on paper. Simulated on the actual return path with maintenance margin
   checked every bar, the best compound monthly return reachable without liquidation is
   **12.6%**, at an 80% drawdown; beyond ~10x gross notional the account is closed out.
3. **The sealed holdout.** Over the 14 months after the configuration was frozen, the
   same strategy nets Sharpe **0.55** and **0.77%/month** — 4 of the 6 failure criteria
   recorded in [`reports/PROTOCOL.md`](reports/PROTOCOL.md) before the holdout was opened.

What the work does establish is a genuine, measured edge and an honest measurement of its
size: rank IC **0.086** across 48,907 out-of-sample cross-sections, positive in 73% of
them, still intact in the holdout. What decayed is not the prediction but the prize — the
top-minus-bottom decile spread fell from ~77 bp in 2021 to ~13 bp, which is the same order
of magnitude as the round-trip fee.

The question turns out to be arithmetic before it is empirical. A book run at
annualised volatility *s* with annualised Sharpe *S* compounds at

    g(s) = S*s - s^2 / 2

because expected return grows linearly in leverage while volatility drag grows
with its square. That peaks at *s = S*, so the best compound growth **any** book
can reach, at **any** leverage, is *S^2 / 2* per year. Turning it around: 33% a
month requires

    S >= sqrt(24 * ln(1.33)) = 2.616

before the question of how much leverage the exchange permits even arises. Below
that Sharpe the target is unreachable at every leverage; above it, the target
fixes the volatility, and the volatility fixes the gross notional and the
drawdowns. `src/tai/evaluation/growth.py` computes this, `tests/test_growth.py`
checks it against the closed form, and the validation step evaluates the verdict
separately for each execution-cost assumption rather than asserting one in prose.

So the work divides in two: measure the net Sharpe honestly, then report exactly
what the target would cost in leverage and risk at that Sharpe.

---

## What the model trades and why it should work

The strategy is **cross-sectional and market-neutral**: at each rebalance it ranks
~100 liquid USDT perpetuals and holds a dollar-neutral, factor-neutral long/short
book. It does not try to forecast the direction of crypto; it tries to forecast
which names will out- or under-perform their peers over the next few hours.

The economic content of the signal comes from four families of effects that are
well documented in intraday crypto markets:

| Family | Features | Mechanism |
|---|---|---|
| Short-horizon reversal | vol-normalised returns and residual returns over 1–720 bars, distance from rolling VWAP, position in rolling range | liquidity provision: leveraged liquidation cascades push price away from fair value and it reverts |
| Order-flow imbalance | signed taker notional (`2·taker_buy − volume`) over 1–168 bars, flow relative to trailing ADV | aggressive flow is informative about very short horizons and over-extends at longer ones |
| Trade-intensity microstructure | trade counts, average trade size and its change, Amihud illiquidity, turnover acceleration | composition of flow (retail vs institutional) and the price of immediacy |
| Positioning and carry | funding rate level / z-score / cumulative, funding × recent return | funding is the observable price of leveraged positioning; crowded carry unwinds |

On top of those, market-state features (realised market vol, breadth, cross-sectional
dispersion, aggregate funding, number of live names) let the model condition the same
features differently across regimes, and cyclical hour/day-of-week features capture
the session and funding-settlement seasonality that crypto genuinely has.

## Architecture

```
src/tai/
  config.py              universe, strategy and cost-model configuration
  data/
    binance_vision.py    official monthly archive ingestion (klines, funding rates)
    panel.py             aligned wide panels + point-in-time universe screen
  features/
    core.py              ~110 leak-free features; chunked streaming construction
    labels.py            next-open-to-open forward returns, beta- and vol-residualised
  models/
    ensemble.py          LightGBM (3 configurations) + ridge, cross-sectionally blended
    store.py             model persistence
  research/
    walkforward.py       purged, embargoed expanding-window walk-forward
    pipeline.py          end-to-end orchestration
  backtest/
    engine.py            bar-level perpetual simulator
    risk.py              trailing PCA factor model, neutralisation, score smoothing
    costs.py             effective-spread estimators and the fitted liquidity model
  evaluation/
    metrics.py           Sharpe/Sortino/Calmar, PSR, deflated Sharpe, CSCV PBO,
                         block bootstrap, risk of ruin, information coefficient
    growth.py            required Sharpe, growth-optimal leverage, reachability
    charts.py            dependency-free SVG for the report
  freeze.py              hash-checked configuration freezing for the holdout
  live/
    adapters.py          exchange REST adapter + official T+1 archive adapter
    paper_trader.py      sequential forward paper-trading loop (production code path)
scripts/
  fetch_data.py          download and cache the kline archive
  fetch_metrics.py       download the open-interest / positioning archive
  calibrate_spread.py    measure the real effective spread from the trade archive
  build_panel.py         assemble panels
  run_research.py        walk-forward + backtest sweep
  baselines.py           single-feature analytic strategies, same simulator
  diagnostics.py         signal decay, IC stability, construction frontier
  train_final.py         fit the production model up to a cutoff
  paper_forward.py       forward paper-trade a period the model never saw
  validate.py            full validation battery + the reachability verdict
  freeze_config.py       freeze the configuration chosen on development data
  run_pipeline.py        run the whole thing unattended
  run_enhanced.py        second pass: more breadth, two horizons, positioning
  make_report.py         assemble reports/RESULTS.md from the result JSONs
  make_artifact.py       render the same figures as a publishable page
tests/                   leakage, timing, neutrality, P&L reconciliation, chunk equivalence
```

## How the backtest avoids the usual ways of lying to yourself

Every one of these is enforced in code and, where testable, asserted in `tests/`:

1. **Fill timing.** Decisions use information up to a bar's close; fills happen at
   the *next* bar's open. Labels are `open[t+1+h]/open[t+1] − 1` — the return of a
   trade that could actually have been placed.
2. **No future in the features.** `tests/test_no_lookahead.py::test_features_are_invariant_to_the_future`
   recomputes every feature on a panel truncated at an arbitrary bar and asserts
   that no value before the cut changes.
3. **Point-in-time universe.** Tradability at bar `t` uses only trailing 30-day
   median dollar volume, a minimum listing age, and a top-N liquidity rank — no
   knowledge of which coins later became liquid, and no knowledge of whether the
   next bar prints.
4. **Delisting is priced, not ignored.** Symbols whose data ends are force-liquidated
   at the last printed open with penalty costs, so the backtest cannot harvest
   returns of coins that quietly vanished.
5. **Purge and embargo.** Training data stops `horizon + embargo` bars before each
   test window, so no training label overlaps the test period.
6. **Costs that scale with size, and a spread that was measured rather than
   assumed.** Each side pays exchange fees, a half-spread, and square-root impact in
   the participation rate computed against that bar's *actual* dollar volume; trades
   above a participation cap are truncated rather than assumed fillable. The spread
   is not a guess: `scripts/calibrate_spread.py` recovers it from the exchange's
   aggregated-trade archive, which publishes the aggressor side of every print, so
   the notional-weighted buy price minus sell price inside a minute *is* the
   effective spread. Cross-checked against the tick size and the Roll estimator.
   (The Corwin-Schultz high-low proxy, tried first, reads hourly crypto volatility
   as spread and overstates it by more than an order of magnitude; it is kept only
   as a deliberately pessimistic stress case.)
7. **Funding is paid.** 8-hourly funding is settled on the realised historical rate
   for every open position — a cost the naive crypto backtest omits and which
   dominates carry-style signals.
8. **Leverage is not free.** Maintenance margin and bankruptcy are checked every bar;
   a path that would have been liquidated is reported as blown up, not as a drawdown.
9. **Volatility targeting uses only the past.** The scaler is computed from trailing
   realised volatility of the *de-levered* book, so it cannot chase its own leverage.
10. **A sealed holdout.** During development, data after the cutoff is not merely
    unused — `--max-date` truncates the panels before a single feature is computed.
11. **Overfitting is measured, not assumed away.** Deflated Sharpe (penalised by the
    number of configurations searched), probabilistic Sharpe, CSCV probability of
    backtest overfitting, and block-bootstrap confidence intervals are all reported.

## Reproducing

```bash
pip install -r requirements.txt

python scripts/fetch_data.py   --interval 1h --min-months 12      # ~1 GB of archive
python scripts/build_panel.py  --interval 1h

# development: everything from 2025-07 onward is invisible to this command
python scripts/run_research.py --tag dev --max-date 2025-07-01 --save-scores \
    --factor-model --smooth-halflife 2

python scripts/validate.py --scores ~/tai-data/results/dev/oos_scores.parquet \
    --tag dev --max-date 2025-07-01

# sealed holdout, run once with the configuration frozen above
python scripts/run_research.py --tag holdout --save-scores --factor-model \
    --smooth-halflife 2

# forward paper trading through the production code path
python scripts/train_final.py  --cutoff 2025-07-01 --name prod_2025H1
python scripts/paper_forward.py --model prod_2025H1 --start 2025-07-01 --leverage 1
```

Running the tests:

```bash
python -m pytest tests/ -q
```

## Operating it live

`scripts/paper_forward.py` runs the same `LiveStrategy` / `PaperBroker` objects a
live deployment would, driven by a rolling window rather than the full history, so
the production path is exercised rather than merely described. To trade real money
you replace the data adapter with `BinanceFuturesREST` (or any exchange adapter
exposing `klines`/`funding`) and route `PaperBroker.rebalance`'s order list to an
execution client. The cost model assumes the orders are worked — split across the
bar, predominantly passive where the maker ratio says so — not sent as one market
order.

**This repository is research code. Nothing in it is investment advice, and the
risk section of `reports/RESULTS.md` is not boilerplate: the leverage required to
reach the headline target carries a materially non-zero probability of a 30–50%
drawdown.**
