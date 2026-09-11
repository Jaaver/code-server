"""Market-data adapters for forward/live operation.

The research stack is trained on Binance USD-M perpetual bars.  In production the
same bars arrive from the exchange's REST/WebSocket API; the adapters below expose
one interface so the *identical* feature code runs in research and in production.

``BinanceFuturesREST``   the real thing (needs an unrestricted egress / API host).
``BinanceVisionDaily``   the official T+1 daily archive; used for forward paper
                         trading where live egress to the futures API is blocked.
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from typing import Protocol

import numpy as np
import pandas as pd
import requests

from ..config import BINANCE_VISION
from ..data.binance_vision import KLINE_COLS, _read_zip_csv

log = logging.getLogger(__name__)


class DataAdapter(Protocol):
    def klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame: ...
    def funding(self, symbol: str, limit: int) -> pd.Series: ...


class BinanceFuturesREST:
    """Live adapter against the USD-M futures REST API."""

    def __init__(self, base: str = "https://fapi.binance.com", timeout: int = 20):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()

    def _get(self, path: str, params: dict):
        for attempt in range(4):
            try:
                r = self.s.get(f"{self.base}{path}", params=params, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                log.warning("%s -> %s %s", path, r.status_code, r.text[:160])
            except requests.RequestException as exc:
                log.warning("%s -> %s", path, exc)
            time.sleep(2 ** attempt)
        raise RuntimeError(f"request failed: {path}")

    def exchange_symbols(self) -> list[str]:
        info = self._get("/fapi/v1/exchangeInfo", {})
        return [s["symbol"] for s in info["symbols"]
                if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING"
                and s.get("quoteAsset") == "USDT"]

    def klines(self, symbol: str, interval: str, limit: int = 1500) -> pd.DataFrame:
        rows = self._get("/fapi/v1/klines", {"symbol": symbol, "interval": interval,
                                             "limit": min(limit, 1500)})
        df = pd.DataFrame(rows, columns=KLINE_COLS)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        return df.set_index("ts").drop(columns=["open_time", "close_time", "ignore"])

    def funding(self, symbol: str, limit: int = 1000) -> pd.Series:
        rows = self._get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": min(limit, 1000)})
        if not rows:
            return pd.Series(dtype="float64")
        df = pd.DataFrame(rows)
        idx = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
        return pd.Series(pd.to_numeric(df["fundingRate"]).to_numpy(), index=idx).sort_index()

    def mark_prices(self) -> pd.Series:
        rows = self._get("/fapi/v1/premiumIndex", {})
        return pd.Series({r["symbol"]: float(r["markPrice"]) for r in rows})


class BinanceVisionDaily:
    """Adapter over the official daily archive (available T+1).

    Used for forward paper trading in environments where the futures REST host is
    unreachable.  It is the same data the exchange serves, one day later.
    """

    def __init__(self, base: str = BINANCE_VISION):
        self.base = base.rstrip("/")
        self.s = requests.Session()
        self._cache: dict[tuple, pd.DataFrame] = {}

    def _day(self, symbol: str, interval: str, day: str) -> pd.DataFrame:
        key = (symbol, interval, day)
        if key in self._cache:
            return self._cache[key]
        url = (f"{self.base}/data/futures/um/daily/klines/{symbol}/{interval}/"
               f"{symbol}-{interval}-{day}.zip")
        try:
            r = self.s.get(url, timeout=40)
        except requests.RequestException:
            return pd.DataFrame()
        if r.status_code != 200:
            return pd.DataFrame()
        df = _read_zip_csv(r.content, KLINE_COLS)
        if df is None or df.empty:
            return pd.DataFrame()
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        ot = df["open_time"].astype("int64")
        scale = np.where(ot > 1e15, 1_000, 1)
        df["ts"] = pd.to_datetime(ot // scale, unit="ms", utc=True)
        df = df.set_index("ts").drop(columns=[c for c in ("open_time", "close_time", "ignore")
                                              if c in df.columns])
        self._cache[key] = df
        return df

    def klines(self, symbol: str, interval: str, limit: int = 1500,
               end: pd.Timestamp | None = None) -> pd.DataFrame:
        end = end or pd.Timestamp.utcnow().tz_localize("UTC")
        bars_per_day = {"1h": 24, "4h": 6, "15m": 96}[interval]
        n_days = int(np.ceil(limit / bars_per_day)) + 1
        days = pd.date_range(end.normalize() - pd.Timedelta(days=n_days), end.normalize(), freq="D")
        frames = [self._day(symbol, interval, d.strftime("%Y-%m-%d")) for d in days]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        return out[~out.index.duplicated(keep="last")].iloc[-limit:]

    def funding(self, symbol: str, limit: int = 1000) -> pd.Series:
        return pd.Series(dtype="float64")
