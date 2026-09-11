"""End-to-end research pipeline: data -> features -> walk-forward -> backtest."""
from __future__ import annotations

import gc
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import ExecConfig, run_backtest
from ..backtest.costs import (half_spread_panel, liquidity_half_spread_panel,
                             load_calibration, spread_summary)
from ..backtest.risk import rolling_factor_model, smooth_scores
from ..config import BARS_PER_DAY, COST_SCENARIOS, CACHE_DIR, RESULTS_DIR, StrategyConfig, UniverseConfig
from ..data import panel as panel_mod
from ..evaluation import metrics as M
from ..features.core import (aux_panels, build_feature_panels,
                             build_features_chunked, stack_features)
from ..features.labels import build_labels
from .walkforward import apply_rebalance_schedule, make_folds, scores_to_wide, walk_forward

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    panel: dict
    mask: pd.DataFrame
    X: pd.DataFrame
    labels: dict
    aux: dict
    interval: str

    @property
    def bars_per_day(self) -> int:
        return BARS_PER_DAY[self.interval]

    def labels_for_horizon(self, horizon: int, label: str = "y_cs") -> pd.Series:
        """Rebuild the label for a different forward horizon, aligned to ``self.X``."""
        lab = build_labels(self.panel, self.mask, horizon, self.aux["beta"],
                           self.aux["vol_ref"])
        flat = lab[label].to_numpy(dtype="float32").ravel()[self.mask.to_numpy().ravel()]
        return pd.Series(flat, index=self.X.index, name="y")


def build_dataset(ucfg: UniverseConfig, scfg: StrategyConfig, *, symbols=None,
                  rebuild_panel: bool = False, label: str = "y_cs",
                  max_date: str | None = None, min_date: str | None = None,
                  use_metrics: bool = False) -> Dataset:
    """Build the design matrix.

    ``max_date`` hard-truncates every panel *before* any feature is computed, which
    is how the sealed holdout is enforced: during development the later data is not
    merely unused, it is not present in the process at all.
    """
    interval = scfg.interval
    bpd = BARS_PER_DAY[interval]
    try:
        if rebuild_panel:
            raise FileNotFoundError
        panel = panel_mod.load_panel(interval)
        log.info("loaded cached panel: %d bars x %d symbols",
                 len(panel["close"]), panel["close"].shape[1])
    except FileNotFoundError:
        panel = panel_mod.build_panel(interval, ucfg, symbols=symbols)

    if min_date or max_date:
        lo = pd.Timestamp(min_date, tz="UTC") if min_date else None
        hi = pd.Timestamp(max_date, tz="UTC") if max_date else None
        for k in list(panel):
            df = panel[k]
            if lo is not None:
                df = df[df.index >= lo]
            if hi is not None:
                df = df[df.index < hi]
            panel[k] = df
        log.info("panel truncated to %s .. %s (%d bars)", panel["close"].index[0],
                 panel["close"].index[-1], len(panel["close"]))

    mask = panel_mod.pit_universe(panel, ucfg, bpd)
    log.info("universe: mean tradable names = %.1f", mask.sum(axis=1).mean())

    metrics = None
    if use_metrics:
        from ..data.metrics_archive import build_metrics_panels
        metrics = build_metrics_panels(list(panel["close"].columns), panel["close"].index, interval)
        cover = {k: float(v.notna().mean().mean()) for k, v in metrics.items()}
        log.info("metrics panels: %s", {k: round(v, 3) for k, v in cover.items()})

    X, aux = build_features_chunked(panel, mask, bpd, metrics=metrics)
    log.info("design matrix: %s rows x %s features", f"{len(X):,}", X.shape[1])
    gc.collect()
    labels = build_labels(panel, mask, scfg.label_horizon, aux["beta"], aux["vol_ref"])

    y_wide = labels[label]
    yv = y_wide.to_numpy(dtype="float32").ravel()[mask.to_numpy().ravel()]
    y = pd.Series(yv, index=X.index, name="y")
    return Dataset(panel=panel, mask=mask, X=X, labels=labels | {"y": y}, aux=aux,
                   interval=interval)


def build_context(ucfg: UniverseConfig, scfg: StrategyConfig, *, max_date: str | None = None,
                  min_date: str | None = None) -> Dataset:
    """Panels, mask, labels and aux only -- no design matrix.

    Used by evaluation code that needs to simulate a saved score series without
    paying the cost of rebuilding the (multi-gigabyte) feature matrix.
    """
    interval = scfg.interval
    bpd = BARS_PER_DAY[interval]
    panel = panel_mod.load_panel(interval)
    if min_date or max_date:
        lo = pd.Timestamp(min_date, tz="UTC") if min_date else None
        hi = pd.Timestamp(max_date, tz="UTC") if max_date else None
        for k in list(panel):
            df = panel[k]
            if lo is not None:
                df = df[df.index >= lo]
            if hi is not None:
                df = df[df.index < hi]
            panel[k] = df
    mask = panel_mod.pit_universe(panel, ucfg, bpd)
    aux = aux_panels(panel, mask, bpd)
    labels = build_labels(panel, mask, scfg.label_horizon, aux["beta"], aux["vol_ref"])
    return Dataset(panel=panel, mask=mask, X=pd.DataFrame(), labels=labels, aux=aux,
                   interval=interval)


def run_walkforward(ds: Dataset, scfg: StrategyConfig, *, train_stride: int = 1,
                    variants=None, sample_halflife_days: float = 0.0,
                    return_per_model: bool = False):
    idx = ds.mask.index
    folds = make_folds(idx, min_train_bars=scfg.min_train_bars,
                       test_months=scfg.test_months, embargo_bars=scfg.embargo_bars,
                       horizon=scfg.label_horizon, train_months=scfg.train_months)
    log.info("%d walk-forward folds: %s .. %s", len(folds),
             folds[0].test_start.date(), folds[-1].test_end.date())
    return walk_forward(ds.X, ds.labels["y"], folds, train_stride=train_stride,
                        variants=variants, sample_halflife_days=sample_halflife_days,
                        return_per_model=return_per_model), folds


def get_factor_model(ds: Dataset, n_factors: int = 5, window_days: int = 45,
                     step_days: int = 7):
    """Trailing PCA factor model, cached on the Dataset."""
    key = (n_factors, window_days, step_days)
    cache = getattr(ds, "_fm_cache", None)
    if cache is None:
        cache = {}
        object.__setattr__(ds, "_fm_cache", cache)
    if key not in cache:
        bpd = ds.bars_per_day
        cache[key] = rolling_factor_model(ds.aux["ret"], ds.mask, n_factors=n_factors,
                                          window=window_days * bpd, step=step_days * bpd)
    return cache[key]


def get_spread_panel(ds: Dataset, method: str = "measured") -> pd.DataFrame:
    """Per-symbol half-spread panel, cached on the Dataset.

    ``measured`` applies the liquidity model fitted to the exchange's own trade
    archive (see scripts/calibrate_spread.py) and is the default, because the
    ``highlow`` alternative -- the Corwin-Schultz estimator -- reads most of
    crypto's hourly volatility as spread and overstates it by an order of
    magnitude.  ``highlow`` is kept as a deliberately pessimistic stress case.
    """
    key = f"_spread_{method}"
    sp = getattr(ds, key, None)
    if sp is None:
        if method == "highlow":
            sp = half_spread_panel(ds.panel)
        else:
            cal = load_calibration(CACHE_DIR / "spread_calibration.json")
            a, b = (cal["a"], cal["b"]) if cal else (0.35, 1.1)
            if cal is None:
                log.warning("no spread calibration on disk; using default a=%.2f b=%.2f", a, b)
            sp = liquidity_half_spread_panel(ds.panel, a=a, b=b,
                                             bars_per_day=ds.bars_per_day)
        object.__setattr__(ds, key, sp)
        log.info("half spread (bps, %s): %s", method,
                 {k: round(v, 2) for k, v in spread_summary(sp, ds.mask).items()
                  if k != "n_obs"})
    return sp


def backtest_scores(ds: Dataset, score: pd.Series, scfg: StrategyConfig,
                    cost_name: str = "base", *, leverage: float = 1.0,
                    exec_overrides: dict | None = None, phase: int = 0,
                    factor_model: bool = False, n_factors: int = 5,
                    smooth_halflife: float = 0.0,
                    estimated_spread: bool = False, spread_method: str = "measured"):
    wide = scores_to_wide(score, ds.mask.index, ds.mask.columns)
    wide = apply_rebalance_schedule(wide, scfg.rebalance_every, phase=phase)
    if smooth_halflife > 0:
        wide = smooth_scores(wide, smooth_halflife)
    cfg = ExecConfig(target_ann_vol=scfg.target_ann_vol, leverage=leverage,
                     max_gross=scfg.max_gross, max_weight=scfg.max_weight,
                     bars_per_day=ds.bars_per_day, dollar_neutral=scfg.dollar_neutral,
                     beta_neutral=scfg.beta_neutral)
    if exec_overrides:
        for k, v in exec_overrides.items():
            setattr(cfg, k, v)
    cost = COST_SCENARIOS[cost_name]
    fm = get_factor_model(ds, n_factors=n_factors) if factor_model else None
    hs = get_spread_panel(ds, spread_method) if estimated_spread else None
    res = run_backtest(ds.panel, wide, ds.mask, ds.aux["vol_ann"], ds.aux["beta"],
                       cost, cfg, factor_model=fm, half_spread_bps=hs)
    return res, cfg


def trim_to_oos(res, score: pd.Series):
    """Restrict an equity curve to the out-of-sample window."""
    start = score.index.get_level_values(0).min()
    keep = res.equity.index >= start
    res.equity = res.equity[keep]
    res.returns = res.returns[keep]
    for attr in ("gross", "net_exposure", "turnover", "costs", "funding",
                 "n_positions", "vol_scalar", "trades_notional"):
        setattr(res, attr, getattr(res, attr)[keep])
    res.weights = res.weights[keep]
    if len(res.equity):
        res.equity = res.equity / res.equity.iloc[0] * 1_000_000.0
    return res


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    log.info("wrote %s", path)
