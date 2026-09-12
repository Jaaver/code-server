"""Assemble OKX panels for cross-exchange validation (mirrors dataset.py)."""
import os, glob, json
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "okx")
PANEL = os.path.join(DATA, "panel")
os.makedirs(PANEL, exist_ok=True)
FIELDS = ["open", "high", "low", "close", "quote_volume"]


def build(min_bars=24 * 90):
    files = sorted(glob.glob(os.path.join(DATA, "klines", "*.parquet")))
    series, funding, kept = {f: {} for f in FIELDS}, {}, []
    for fp in files:
        sym = os.path.basename(fp)[:-8]
        df = pd.read_parquet(fp)
        if len(df) < min_bars:
            continue
        df = df.set_index("ts")
        for f in FIELDS:
            series[f][sym] = df[f]
        ff = os.path.join(DATA, "funding", f"{sym}.parquet")
        if os.path.exists(ff):
            fd = pd.read_parquet(ff)
            fd["ts"] = fd["ts"].dt.floor("h")
            funding[sym] = fd.drop_duplicates("ts", keep="last").set_index("ts")["funding_rate"]
        kept.append(sym)

    panels = {f: pd.DataFrame(series[f]).sort_index() for f in FIELDS}
    idx = pd.date_range(panels["close"].index.min(), panels["close"].index.max(),
                        freq="1h", tz="UTC")
    panels = {k: v.reindex(idx) for k, v in panels.items()}
    # OKX gives no taker-side split; use a neutral placeholder so shared alpha
    # code runs, and simply exclude order-flow alphas from the OKX comparison.
    panels["taker_buy_quote"] = panels["quote_volume"] * 0.5
    panels["trades"] = panels["quote_volume"] * 0.0
    fund = pd.DataFrame(funding).sort_index() if funding else pd.DataFrame(index=idx)
    panels["funding"] = fund.reindex(columns=panels["close"].columns).reindex(idx)

    for name, p in panels.items():
        p.astype(np.float32).to_parquet(os.path.join(PANEL, f"{name}.parquet"), compression="zstd")
    json.dump(kept, open(os.path.join(PANEL, "symbols.json"), "w"))
    print(f"OKX panel: {panels['close'].shape}  {idx.min()} -> {idx.max()}  symbols={len(kept)}")
    return panels


def load():
    return {f: pd.read_parquet(os.path.join(PANEL, f"{f}.parquet"))
            for f in FIELDS + ["taker_buy_quote", "trades", "funding"]}


if __name__ == "__main__":
    build()
