"""Assemble per-symbol parquet files into aligned (time x symbol) panels.

Everything downstream reads these panels. NaN means "not listed / no data" at
that timestamp, which is how delisted and not-yet-listed symbols are handled.
"""
import os, glob, json
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PANEL = os.path.join(DATA, "panel")
os.makedirs(PANEL, exist_ok=True)

FIELDS = ["open", "high", "low", "close", "quote_volume", "trades", "taker_buy_quote"]

def build(freq="1h", min_bars=24 * 120):
    files = sorted(glob.glob(os.path.join(DATA, "klines", "*.parquet")))
    print(f"found {len(files)} symbol files")
    series, funding, kept = {f: {} for f in FIELDS}, {}, []

    for fp in files:
        sym = os.path.basename(fp)[:-8]
        df = pd.read_parquet(fp)
        if len(df) < min_bars:
            continue
        df = df.set_index("ts")
        # Drop bars that never traded - they are exchange padding, not real prices.
        traded = df["quote_volume"].fillna(0) > 0
        if traded.sum() < min_bars:
            continue
        for f in FIELDS:
            series[f][sym] = df[f]
        fp_f = os.path.join(DATA, "funding", f"{sym}.parquet")
        if os.path.exists(fp_f):
            fd = pd.read_parquet(fp_f)
            # Settlement stamps carry ms jitter (16:00:00.001); ~46% of them miss
            # an exact hourly grid, so floor before aligning or they get dropped.
            fd["ts"] = fd["ts"].dt.floor("h")
            fd = fd.drop_duplicates("ts", keep="last").set_index("ts")["funding_rate"]
            funding[sym] = fd
        kept.append(sym)

    print(f"kept {len(kept)} symbols with >= {min_bars} bars")
    panels = {}
    for f in FIELDS:
        panels[f] = pd.DataFrame(series[f]).sort_index()

    idx = panels["close"].index
    # Funding is stamped every 8h; align onto the hourly grid (NaN elsewhere).
    fund = pd.DataFrame(funding).sort_index() if funding else pd.DataFrame(index=idx)
    fund = fund.reindex(columns=panels["close"].columns)
    fund = fund[~fund.index.duplicated()].reindex(idx.union(fund.index)).loc[idx]

    for name, p in list(panels.items()) + [("funding", fund)]:
        p = p.astype(np.float32)
        p.to_parquet(os.path.join(PANEL, f"{name}.parquet"), compression="zstd")
        print(f"  {name}: {p.shape}")
    json.dump(kept, open(os.path.join(PANEL, "symbols.json"), "w"))
    print(f"date range: {idx.min()} -> {idx.max()}  ({len(idx)} hourly bars)")
    return panels

def load():
    out = {}
    for f in FIELDS + ["funding"]:
        out[f] = pd.read_parquet(os.path.join(PANEL, f"{f}.parquet"))
    return out

if __name__ == "__main__":
    build()
