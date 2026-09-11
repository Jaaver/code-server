# Validation protocol

Written **before** the holdout was touched, and followed as written. The point of
recording it is that the order of operations is the only thing separating an
out-of-sample result from a curve-fitted one, and the order of operations is not
visible in a results table.

## Phase 1 — development (data strictly before 2025-07-01)

`scripts/run_research.py --max-date 2025-07-01` truncates every panel before a
single feature is computed, so bars from 2025-07 onward are not in the process.
Within that window:

* walk-forward is expanding, retrained every 3 months, with a 48-bar embargo plus
  a `horizon + 1` purge before each test window;
* all configuration search — label horizon, rebalance interval, score smoothing,
  factor-model usage, per-name cap, universe size, cost scenario — happens here;
* every configuration evaluated is counted and fed to the deflated Sharpe ratio and
  to the CSCV probability-of-backtest-overfitting estimate, so the search itself is
  priced into the statistics.

## Phase 2 — freeze

The chosen configuration is written to `configs/frozen.json` with a timestamp and
the dev-phase metrics that justified it. Nothing in it changes afterwards.

## Phase 3 — sealed holdout (2025-07-01 onward)

The same walk-forward is extended over the full history. The models that predict
holdout bars are trained only on bars preceding them, exactly as in phase 1, and
the holdout segment is evaluated **once**, with the frozen configuration. If the
holdout disagrees with the development period, the holdout number is the one that
gets reported — there is no second attempt, and no re-freezing.

## Phase 4 — forward paper trading

`scripts/paper_forward.py` replays the holdout period through the production code
path: a rolling window, the same feature functions, a frozen model fitted at the
cutoff, and a broker that charges the same fees, spread, impact and funding. This
catches the class of bug where the vectorised backtest quietly uses information the
live system would not have. Agreement between phase 3 and phase 4 is reported; a
disagreement would be reported as a failure of the backtest, not of the model.

## What would count as failure

Stated in advance so the conclusion cannot be negotiated after the fact:

* holdout net Sharpe below 1.0, or holdout geometric monthly return below a third
  of the development figure at the same leverage;
* CSCV probability of backtest overfitting above 0.5;
* deflated Sharpe below 0.95;
* the target monthly return requiring gross notional beyond what the venue's
  leverage tiers and maintenance-margin schedule permit;
* any blown-up path in the simulation at the reported operating leverage.

Each of these is reported in `reports/RESULTS.md` whether it passed or failed.
