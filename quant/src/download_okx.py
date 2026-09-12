"""Download OKX USDT perpetual swaps: an INDEPENDENT venue for cross-exchange validation.

If an edge found on Binance also shows up on OKX - different matching engine,
different fee schedule, different listing set, different flow - it is far less
likely to be an artifact of one exchange's data.
"""
import os, json, time, sys, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

API = "https://www.okx.com/api/v5"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "okx")
os.makedirs(os.path.join(DATA, "klines"), exist_ok=True)
os.makedirs(os.path.join(DATA, "funding"), exist_ok=True)

_lock_sleep = 0.02   # proxy latency already paces us well inside OKX limits


def get(path, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(API + path, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            if d.get("code") == "0":
                return d.get("data", [])
            time.sleep(0.5 * (i + 1))
        except Exception:
            time.sleep(0.8 * (i + 1))
    return None


def instruments():
    d = get("/public/instruments?instType=SWAP") or []
    return [x["instId"] for x in d if x.get("settleCcy") == "USDT" and x.get("state") == "live"]


def klines(inst, start_ms, bar="1H"):
    """Page backwards from now to start_ms using the `after` cursor."""
    rows, after = [], ""
    while True:
        path = f"/market/history-candles?instId={inst}&bar={bar}&limit=100" + (f"&after={after}" if after else "")
        d = get(path)
        time.sleep(_lock_sleep)
        if not d:
            break
        rows += d
        oldest = int(d[-1][0])
        if oldest <= start_ms or len(d) < 100:
            break
        after = str(oldest)
        if len(rows) > 80000:
            break
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                     "volume", "volCcy", "volCcyQuote", "confirm"])
    for c in ("open", "high", "low", "close", "volCcyQuote"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms", utc=True)
    df = df.rename(columns={"volCcyQuote": "quote_volume"})
    df = (df[df["confirm"] == "1"].dropna(subset=["ts", "close"])
            .drop_duplicates("ts").sort_values("ts"))
    return df[["ts", "open", "high", "low", "close", "quote_volume"]]


def funding(inst, start_ms):
    rows, after = [], ""
    while True:
        path = f"/public/funding-rate-history?instId={inst}&limit=100" + (f"&after={after}" if after else "")
        d = get(path)
        time.sleep(_lock_sleep)
        if not d:
            break
        rows += d
        oldest = int(d[-1]["fundingTime"])
        if oldest <= start_ms or len(d) < 100:
            break
        after = str(oldest)
        if len(rows) > 20000:
            break
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(pd.to_numeric(df["fundingTime"]), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["realizedRate"], errors="coerce")
    return df.dropna(subset=["ts", "funding_rate"]).drop_duplicates("ts").sort_values("ts")[["ts", "funding_rate"]]


def do(inst, start_ms):
    kp = os.path.join(DATA, "klines", f"{inst}.parquet")
    fp = os.path.join(DATA, "funding", f"{inst}.parquet")
    note = []
    if not os.path.exists(kp):
        k = klines(inst, start_ms)
        if k is None or len(k) < 24 * 120:
            return f"{inst}: thin"
        k.to_parquet(kp, index=False, compression="zstd"); note.append(f"k={len(k)}")
    if not os.path.exists(fp):
        f = funding(inst, start_ms)
        if f is not None:
            f.to_parquet(fp, index=False, compression="zstd"); note.append(f"f={len(f)}")
    return f"{inst}: " + (", ".join(note) or "cached")


if __name__ == "__main__":
    years = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    start_ms = int((pd.Timestamp.now('UTC') - pd.Timedelta(days=365 * years)).timestamp() * 1000)
    inst = instruments()
    # Restrict to bases that also trade as Binance perps. OKX lists equity and
    # commodity swaps (CL, SNDK, SPCX...) that would make the cross-exchange
    # comparison meaningless, and it keeps the two universes comparable.
    import glob
    bnc = {os.path.basename(f)[:-8].replace("USDT", "")
           for f in glob.glob(os.path.join(os.path.dirname(DATA), "klines", "*.parquet"))}
    inst = [i for i in inst if i.split("-")[0] in bnc or i.split("-")[0].lstrip("1234567890") in bnc]
    tick = get("/market/tickers?instType=SWAP") or []
    vol = {t["instId"]: float(t.get("volCcy24h") or 0) * float(t.get("last") or 0) for t in tick}
    inst = sorted(inst, key=lambda s: -vol.get(s, 0))[:nmax]
    print(f"{len(inst)} OKX USDT swaps, {years}y history", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(do, s, start_ms): s for s in inst}
        for f in as_completed(futs):
            done += 1
            try:
                r = f.result()
            except Exception as e:
                r = f"{futs[f]}: ERR {e}"
            if done % 10 == 0:
                print(f"[{done}/{len(inst)}] {r}", flush=True)
    print("OKX COMPLETE", flush=True)
