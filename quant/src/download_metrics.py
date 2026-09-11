"""Download Binance USD-M `metrics` (open interest + long/short positioning), daily files.

Columns: sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio,
sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio.
Sampled every 5 min; we resample to the hourly grid (last value in each hour).
"""
import io, os, json, sys, zipfile, time, urllib.request, urllib.error
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT = os.path.join(DATA, "metrics")
os.makedirs(OUT, exist_ok=True)

COLS = ["sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]


def _get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.3 ** i)
        except Exception:
            time.sleep(1.3 ** i)
    return None


def list_days(sym):
    days, marker = [], ""
    pre = f"data/futures/um/daily/metrics/{sym}/"
    while True:
        url = f"{S3}?prefix={pre}&max-keys=1000" + (f"&marker={marker}" if marker else "")
        b = _get(url)
        if b is None:
            return days
        root = ET.fromstring(b)
        got = [c.findtext(f"{NS}Key") for c in root.findall(f"{NS}Contents")]
        days += [k[-14:-4] for k in got if k and k.endswith(".zip")]
        if root.findtext(f"{NS}IsTruncated") == "true" and got:
            marker = got[-1]
        else:
            return days


def do_symbol(sym):
    path = os.path.join(OUT, f"{sym}.parquet")
    if os.path.exists(path):
        return f"{sym}: cached"
    days = list_days(sym)
    if not days:
        return f"{sym}: none"
    frames = []
    for d in days:
        b = _get(f"{BASE}/{sym}/{sym}-metrics-{d}.zip")
        if b is None:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                with z.open(z.namelist()[0]) as f:
                    frames.append(pd.read_csv(io.BytesIO(f.read())))
        except Exception:
            continue
    if not frames:
        return f"{sym}: empty"
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["create_time"], utc=True, errors="coerce")
    for c in COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    hourly = df[COLS].resample("1h").last()          # last observation in the hour
    hourly.astype(np.float32).to_parquet(path, compression="zstd")
    return f"{sym}: {len(hourly)} hours from {hourly.index.min().date()}"


if __name__ == "__main__":
    man = json.load(open(os.path.join(DATA, "manifest.json")))
    # only symbols with a meaningful amount of price history
    syms = sorted([s for s, v in man.items() if len(v["klines"]) >= 12])
    print(f"{len(syms)} symbols with >=12 months of klines", flush=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(do_symbol, s): s for s in syms}
        for f in as_completed(futs):
            done += 1
            try:
                r = f.result()
            except Exception as e:
                r = f"{futs[f]}: ERR {e}"
            if done % 25 == 0:
                print(f"[{done}/{len(syms)}] {r}", flush=True)
    print("METRICS COMPLETE", flush=True)
