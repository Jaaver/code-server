"""Persistence for fitted ensembles."""
from __future__ import annotations

import pickle
from pathlib import Path

from ..config import RESULTS_DIR


def save_model(obj, name: str, directory: Path | None = None) -> Path:
    d = directory or (RESULTS_DIR / "models")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.pkl"
    with open(p, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_model(name: str, directory: Path | None = None):
    d = directory or (RESULTS_DIR / "models")
    with open(d / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)
