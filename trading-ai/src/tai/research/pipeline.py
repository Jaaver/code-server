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
from ..config import BARS_PER_DAY, COST_SCENARIOS, CACHE_DIR, RESULTS_DIR, StrategyConfig, UniverseConfig
from ..data import panel as panel_mod
from ..evaluation import metrics as M
from ..features.core import build_feature_panels, stack_features
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


def build_dataset(ucfg: UniverseConfig, scfg: StrategyConfig, *, symbols=None,
                  rebuild_panel: bool = False, label: str = "y_cs",
                  max_date: str | None = None, min_date: str | None = None) -> Dataset:
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

    F, aux = build_feature_panels(panel, mask, bpd)
    log.info("features: %d", len(F))
    labels = build_labels(panel, mask, scfg.label_horizon, aux["beta"], aux["vol_ref"])

    X = stack_features(F, mask)
    del F
    gc.collect()
    y_wide = labels[label]
    yv = y_wide.to_numpy(dtype="float32").ravel()[mask.to_numpy().ravel()]
    y = pd.Series(yv, index=X.index, name="y")
    log.info("design matrix: %s rows x %s features", f"{len(X):,}", X.shape[1])
    return Dataset(panel=panel, mask=mask, X=X, labels=labels | {"y": y}, aux=aux,
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


def backtest_scores(ds: Dataset, score: pd.Series, scfg: StrategyConfig,
                    cost_name: str = "base", *, leverage: float = 1.0,
                    exec_overrides: dict | None = None, phase: int = 0):
    wide = scores_to_wide(score, ds.mask.index, ds.mask.columns)
    wide = apply_rebalance_schedule(wide, scfg.rebalance_every, phase=phase)
    cfg = ExecConfig(target_ann_vol=scfg.target_ann_vol, leverage=leverage,
                     max_gross=scfg.max_gross, max_weight=scfg.max_weight,
                     bars_per_day=ds.bars_per_day, dollar_neutral=scfg.dollar_neutral,
                     beta_neutral=scfg.beta_neutral)
    if exec_overrides:
        for k, v in exec_overrides.items():
            setattr(cfg, k, v)
    cost = COST_SCENARIOS[cost_name]
    res = run_backtest(ds.panel, wide, ds.mask, ds.aux["vol_ann"], ds.aux["beta"],
                       cost, cfg)
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
