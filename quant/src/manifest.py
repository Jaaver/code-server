"""List which monthly archives exist per symbol (1 S3 request/symbol, no wasted 404s).

Uses a real XML parser: S3 varies the element order/содержимое of <Contents>
(newer objects carry ChecksumAlgorithm/ChecksumType), which silently breaks
regex-based scraping.
"""
import re, json, os, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _get(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception:
            time.sleep(1.5 ** attempt)
    return None


def list_prefix(prefix):
    """Return [(key, size)] for every object under prefix, following pagination."""
    keys, marker = [], ""
    while True:
        url = (f"{BASE}?prefix={urllib.parse.quote(prefix)}&max-keys=1000"
               + (f"&marker={urllib.parse.quote(marker)}" if marker else ""))
        blob = _get(url)
        if blob is None:
            return keys
        root = ET.fromstring(blob)
        got = []
        for c in root.findall(f"{NS}Contents"):
            k = c.findtext(f"{NS}Key")
            s = c.findtext(f"{NS}Size")
            if k is not None:
                got.append((k, int(s or 0)))
        keys += got
        if root.findtext(f"{NS}IsTruncated") == "true" and got:
            marker = got[-1][0]
        else:
            return keys


def list_symbols():
    """Every USD-M symbol that ever appears in the archive (delisted included)."""
    syms, marker = [], ""
    pre = "data/futures/um/monthly/klines/"
    while True:
        url = (f"{BASE}?delimiter=/&prefix={pre}"
               + (f"&marker={urllib.parse.quote(marker)}" if marker else ""))
        blob = _get(url)
        if blob is None:
            break
        root = ET.fromstring(blob)
        got = [p.findtext(f"{NS}Prefix")[len(pre):].rstrip("/")
               for p in root.findall(f"{NS}CommonPrefixes")]
        syms += got
        if root.findtext(f"{NS}IsTruncated") == "true" and got:
            marker = pre + got[-1] + "/"
        else:
            break
    return sorted(set(syms))


def symbol_months(sym):
    res = {}
    for kind, pre in (("klines", f"data/futures/um/monthly/klines/{sym}/1h/"),
                      ("funding", f"data/futures/um/monthly/fundingRate/{sym}/")):
        got = []
        for key, size in list_prefix(pre):
            if not key.endswith(".zip"):
                continue
            m = re.search(r"(\d{4}-\d{2})\.zip$", key)
            if m:
                got.append((m.group(1), size))
        res[kind] = sorted(got)
    return sym, res


if __name__ == "__main__":
    allsym = list_symbols()
    usdt = [s for s in allsym if s.endswith("USDT") and "_" not in s]
    json.dump({"all": allsym, "usdt_perp": usdt},
              open(os.path.join(DATA, "all_symbols.json"), "w"), indent=1)
    print(f"archive symbols: {len(allsym)}  USDT perps: {len(usdt)}")

    out, done = {}, 0
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = [ex.submit(symbol_months, s) for s in usdt]
        for f in as_completed(futs):
            try:
                s, r = f.result(); out[s] = r
            except Exception:
                pass
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(usdt)}", flush=True)
    json.dump(out, open(os.path.join(DATA, "manifest.json"), "w"))
    nk = sum(len(v["klines"]) for v in out.values())
    tot = sum(sz for v in out.values() for _, sz in v["klines"])
    last = max((m for v in out.values() for m, _ in v["klines"]), default="?")
    print(f"symbols={len(out)} kline-months={nk} bytes={tot/1e9:.2f}GB latest_month={last}")
