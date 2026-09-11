"""A frozen configuration must not be loadable after being edited."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.freeze import freeze, load_frozen


def test_roundtrip_and_tamper_detection(tmp_path):
    p = tmp_path / "frozen.json"
    cfg = {"horizon": 4, "rebalance": 8, "smooth_halflife": 2.0}
    freeze(cfg, p, dev_metrics={"sharpe": 2.1}, note="dev01")
    assert load_frozen(p) == cfg
    payload = json.loads(p.read_text())
    payload["config"]["rebalance"] = 1
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="edited since freezing"):
        load_frozen(p)


def test_refreezing_the_same_config_keeps_the_original_timestamp(tmp_path):
    p = tmp_path / "frozen.json"
    cfg = {"horizon": 4, "rebalance": 12}
    first = freeze(cfg, p, note="dev")
    second = freeze(cfg, p, note="dev rerun")
    assert second["frozen_at"] == first["frozen_at"]
    assert second["note"] == "dev"
    # a genuinely different configuration does get a new stamp
    third = freeze({**cfg, "rebalance": 24}, p, note="changed")
    assert third["frozen_at"] != first["frozen_at"]
    assert third["sha256_16"] != first["sha256_16"]
