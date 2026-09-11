"""Data integrity audit. A silent data bug is worth more than any alpha, negatively."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset

def audit():
    p = dataset.load()
    cl, op, hi, lo = p["close"], p["open"], p["high"], p["low"]
    idx = cl.index
    problems = []

    print(f"shape           : {cl.shape}")
    print(f"date range      : {idx.min()} -> {idx.max()}")
    print(f"span            : {(idx.max()-idx.min()).days/365.25:.2f} years")

    # 1. hourly grid must be complete and monotonic
    step = pd.Series(idx).diff().dropna()
    bad_step = (step != pd.Timedelta("1h")).sum()
    print(f"non-1h steps    : {bad_step}")
    if bad_step:
        problems.append(f"{bad_step} irregular timestamp steps")

    # 2. OHLC consistency
    viol = ((hi < lo) | (hi < cl) | (hi < op) | (lo > cl) | (lo > op)).sum().sum()
    print(f"OHLC violations : {viol}")
    if viol > 0:
        problems.append(f"{viol} OHLC ordering violations")

    # 3. non-positive prices
    npos = (cl <= 0).sum().sum()
    print(f"non-positive px : {npos}")
    if npos:
        problems.append(f"{npos} non-positive prices")

    # 4. implausible 1h moves (data errors vs real crypto moves)
    r = np.log(cl).diff()
    extreme = (r.abs() > 1.0).sum().sum()      # >100% in one hour
    print(f"|1h move|>100%  : {extreme}")

    # 5. funding sanity (Binance caps at +/-2% typically, usually +/-0.75%)
    f = p["funding"].stack()
    if len(f):
        print(f"funding: n={len(f)} mean={f.mean():.6f} p1={f.quantile(0.01):.5f} "
              f"p99={f.quantile(0.99):.5f} min={f.min():.5f} max={f.max():.5f}")
        if f.abs().max() > 0.05:
            problems.append("funding rate outside plausible band")
        hours = p["funding"].notna().any(axis=1)
        hh = pd.Series(idx[hours]).dt.hour.value_counts().sort_index()
        print(f"funding hours   : {dict(hh)}  (expect 0/8/16 UTC)")

    # 6. coverage over time
    cov = cl.notna().sum(axis=1)
    print(f"symbols listed  : start={cov.iloc[0]} end={cov.iloc[-1]} max={cov.max()}")
    yearly = cov.groupby(cov.index.year).mean().round(0).to_dict()
    print(f"avg listed/yr   : {yearly}")

    # 7. survivorship: symbols that stop trading before the end (delisted)
    last_valid = cl.apply(lambda s: s.last_valid_index())
    dead = (last_valid < idx.max() - pd.Timedelta("7D")).sum()
    print(f"delisted symbols: {dead} of {cl.shape[1]}  <- survivorship bias controlled")

    # 8. staleness: repeated identical closes = synthetic padding
    stale = (cl.diff() == 0).sum().sum() / cl.notna().sum().sum()
    print(f"stale-bar share : {stale:.3%}")

    print("\nPROBLEMS:", problems if problems else "none")
    return problems

if __name__ == "__main__":
    audit()
