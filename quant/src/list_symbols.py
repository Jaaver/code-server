import re, urllib.request, json, os
BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PREFIX = "data/futures/um/monthly/klines/"
syms, marker = [], ""
while True:
    url = f"{BASE}?delimiter=/&prefix={PREFIX}" + (f"&marker={urllib.parse.quote(marker)}" if marker else "")
    with urllib.request.urlopen(url, timeout=90) as r:
        xml = r.read().decode()
    found = re.findall(r"<Prefix>" + re.escape(PREFIX) + r"([^<]+)/</Prefix>", xml)
    syms += found
    if "<IsTruncated>true</IsTruncated>" in xml and found:
        marker = PREFIX + found[-1] + "/"
    else:
        break
syms = sorted(set(syms))
usdt = [s for s in syms if s.endswith("USDT") and "_" not in s]
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "all_symbols.json")
json.dump({"all": syms, "usdt_perp": usdt}, open(out, "w"), indent=1)
print(f"total symbols in archive: {len(syms)}   USDT perps: {len(usdt)}")
print("sample:", usdt[:15])
