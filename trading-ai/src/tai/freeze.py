"""Configuration freezing.

The sealed-holdout protocol only means something if the configuration used on the
holdout is provably the one chosen before the holdout was seen.  Freezing writes
the configuration plus a content hash and the justifying development metrics;
loading refuses to silently accept a modified file.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _digest(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def freeze(config: dict, path: Path, *, dev_metrics: dict | None = None,
           note: str = "") -> dict:
    payload = {"config": config, "dev_metrics": dev_metrics or {}, "note": note,
               "frozen_at": datetime.now(timezone.utc).isoformat()}
    payload["sha256_16"] = _digest(payload["config"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def load_frozen(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    want = payload.get("sha256_16")
    got = _digest(payload["config"])
    if want != got:
        raise ValueError(
            f"frozen config at {path} has been edited since freezing "
            f"(hash {got} != recorded {want}); re-running the holdout with a changed "
            f"configuration would invalidate it")
    return payload["config"]
