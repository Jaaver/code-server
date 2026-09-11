"""Download the full Binance USD-M perp history (1h klines + funding) using the manifest.

Survivorship-bias-free: every symbol that ever traded in the archive is fetched,
including delisted ones, with its true first-listed month preserved.
"""
import io, os, json, zipfile, time, urllib.request, urllib.error, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

BASE = "https://data.binance.vision/data/futures/um/monthly"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
for d in ("klines", "funding"):
    os.makedirs(os.path.join(DATA, d), exist_ok=True)

KCOLS = ["open_time","open","high","low","close","volume","close_time",
         "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]

def fetch(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 ** i)
        except Exception:
            time.sleep(1.5 ** i)
    return None

def read_zip_csv(blob, **kw):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(io.BytesIO(f.read()), **kw)

def norm_ms(series):
    """Binance moved some timestamps from ms to microseconds; normalise to ms."""
    t = pd.to_numeric(series, errors="coerce")
    return t.where(t < 1e15, t / 1000.0)

def do_symbol(sym, man):
    kpath = os.path.join(DATA, "klines", f"{sym}.parquet")
    fpath = os.path.join(DATA, "funding", f"{sym}.parquet")
    note = []

    if not os.path.exists(kpath):
        frames = []
        for m, _sz in man["klines"]:
            blob = fetch(f"{BASE}/klines/{sym}/1h/{sym}-1h-{m}.zip")
            if blob is None:
                continue
            try:
                df = read_zip_csv(blob, header=None, names=KCOLS)
                if str(df.iloc[0, 0]).lower().startswith("open"):
                    df = df.iloc[1:]
                frames.append(df)
            except Exception:
                continue
        if not frames:
            return f"{sym}: no klines"
        df = pd.concat(frames, ignore_index=True)
        for c in ("open","high","low","close","volume","quote_volume","trades","taker_buy_quote"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts"] = pd.to_datetime(norm_ms(df["open_time"]), unit="ms", utc=True)
        df = (df.dropna(subset=["ts","close"]).drop_duplicates("ts").sort_values("ts")
                [["ts","open","high","low","close","volume","quote_volume","trades","taker_buy_quote"]])
        for c in df.columns:
            if c != "ts":
                df[c] = df[c].astype(np.float64)
        df.to_parquet(kpath, index=False, compression="zstd")
        note.append(f"k={len(df)}@{df.ts.min().date()}")

    if not os.path.exists(fpath) and man.get("funding"):
        frames = []
        for m, _sz in man["funding"]:
            blob = fetch(f"{BASE}/fundingRate/{sym}/{sym}-fundingRate-{m}.zip")
            if blob is None:
                continue
            try:
                frames.append(read_zip_csv(blob))
            except Exception:
                continue
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df.columns = [str(c).strip().lower() for c in df.columns]
            tcol = next((c for c in df.columns if "time" in c), None)
            rcol = next((c for c in df.columns if "rate" in c), None)
            if tcol and rcol:
                df["ts"] = pd.to_datetime(norm_ms(df[tcol]), unit="ms", utc=True)
                df["funding_rate"] = pd.to_numeric(df[rcol], errors="coerce")
                df = (df.dropna(subset=["ts","funding_rate"]).drop_duplicates("ts")
                        .sort_values("ts")[["ts","funding_rate"]])
                df.to_parquet(fpath, index=False, compression="zstd")
                note.append(f"f={len(df)}")
    return f"{sym}: " + (", ".join(note) if note else "cached")

if __name__ == "__main__":
    man = json.load(open(os.path.join(DATA, "manifest.json")))
    syms = sorted(man)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(do_symbol, s, man[s]): s for s in syms}
        for f in as_completed(futs):
            done += 1
            try:
                r = f.result()
            except Exception as e:
                r = f"{futs[f]}: ERR {e}"
            if done % 50 == 0 or "ERR" in r or "no klines" in r:
                print(f"[{done}/{len(syms)}] {r}", flush=True)
    print("DOWNLOAD COMPLETE", flush=True)
