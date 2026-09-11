"""Model zoo and ensembling for cross-sectional alpha prediction."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import QuantileTransformer

log = logging.getLogger(__name__)
EPS = 1e-12

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:  # pragma: no cover
    HAS_LGB = False


LGB_BASE = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=300,
    feature_fraction=0.55,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=5.0,
    max_bin=127,
    verbosity=-1,
    n_jobs=4,
)

LGB_VARIANTS = {
    "gbm_reg": dict(LGB_BASE, n_estimators=400, seed=11),
    "gbm_huber": dict(LGB_BASE, objective="huber", alpha=1.0, n_estimators=400,
                      num_leaves=31, learning_rate=0.04, feature_fraction=0.45, seed=23),
    "gbm_deep": dict(LGB_BASE, n_estimators=300, num_leaves=127, learning_rate=0.025,
                     min_child_samples=600, feature_fraction=0.4, lambda_l2=20.0, seed=37),
}


@dataclass
class FittedEnsemble:
    models: dict = field(default_factory=dict)
    ridge: object = None
    ridge_cols: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    medians: np.ndarray | None = None

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-model predictions (columns) aligned to ``X.index``."""
        Xv = X[self.feature_names]
        arr = Xv.to_numpy(dtype="float32")
        if self.medians is not None:
            bad = ~np.isfinite(arr)
            if bad.any():
                arr = np.where(bad, np.broadcast_to(self.medians, arr.shape), arr)
        out = {}
        for name, m in self.models.items():
            out[name] = m.predict(arr)
        if self.ridge is not None:
            cols = [self.feature_names.index(c) for c in self.ridge_cols]
            out["ridge"] = self.ridge.predict(np.clip(arr[:, cols], -5, 5))
        return pd.DataFrame(out, index=X.index)


def fit_ensemble(X: pd.DataFrame, y: pd.Series, *, sample_weight: np.ndarray | None = None,
                 variants: dict | None = None, use_ridge: bool = True) -> FittedEnsemble:
    variants = variants if variants is not None else LGB_VARIANTS
    feats = list(X.columns)
    arr = X.to_numpy(dtype="float32")
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype("float32")
    bad = ~np.isfinite(arr)
    if bad.any():
        arr = np.where(bad, np.broadcast_to(med, arr.shape), arr)
    yv = y.to_numpy(dtype="float32")

    models = {}
    if HAS_LGB:
        for name, params in variants.items():
            m = lgb.LGBMRegressor(**params)
            m.fit(arr, yv, sample_weight=sample_weight,
                  feature_name=feats, callbacks=[lgb.log_evaluation(0)])
            models[name] = m
    else:  # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingRegressor
        models["hgb"] = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                      max_leaf_nodes=63).fit(arr, yv)

    ridge = None
    ridge_cols: list[str] = []
    if use_ridge:
        ridge_cols = [c for c in feats if c.startswith("cs_")] or feats
        cols = [feats.index(c) for c in ridge_cols]
        ridge = Ridge(alpha=200.0, fit_intercept=True)
        ridge.fit(np.clip(arr[:, cols], -5, 5), yv, sample_weight=sample_weight)

    return FittedEnsemble(models=models, ridge=ridge, ridge_cols=ridge_cols,
                          feature_names=feats, medians=med)


def cs_standardise(pred: pd.Series) -> pd.Series:
    """Cross-sectionally demean and unit-scale a (ts, symbol)-indexed prediction."""
    g = pred.groupby(level=0)
    return ((pred - g.transform("mean")) / (g.transform("std") + EPS)).clip(-4, 4)


def blend(preds: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Blend per-model predictions after cross-sectional standardisation."""
    cols = list(preds.columns)
    w = weights or {c: 1.0 / len(cols) for c in cols}
    acc = None
    for c in cols:
        z = cs_standardise(preds[c])
        contrib = z * w.get(c, 0.0)
        acc = contrib if acc is None else acc + contrib
    return cs_standardise(acc.rename("score"))
