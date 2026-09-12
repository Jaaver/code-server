"""Run the SAME alphas on OKX data and compare sign/strength with Binance.

This is the strongest generalisation test available here: a different exchange,
different fee schedule, different flow, different listing set. Alphas that only
work on one venue's data are usually artifacts.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alphas, portfolio, okx_dataset, dataset

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# order-flow alphas need Binance's taker split, which OKX candles do not provide
SKIP = {"ofi_6", "ofi_24"}


def ic_table(panels, min_dv, label):
    mask = portfolio.tradable_mask(panels, min_dollar_vol=min_dv)
    fwd = np.log(panels["close"]).diff().shift(-1)
    rows = {}
    for name, fn in alphas.ALPHAS.items():
        if name in SKIP:
            continue
        try:
            sc = alphas.zscore_xs(fn(panels).where(mask))
        except Exception as e:
            print(f"  {label} {name}: failed {e}")
            continue
        ic = portfolio.information_coefficient(sc, fwd, mask).dropna()
        if len(ic) < 200:
            continue
        rows[name] = {"IC": ic.mean(), "t": ic.mean() / (ic.std() / np.sqrt(len(ic))), "n": len(ic)}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    okx = okx_dataset.load()
    idx = okx["close"].index
    print(f"OKX panel {okx['close'].shape}  {idx.min()} -> {idx.max()}")
    o = ic_table(okx, 1e6, "okx")

    bnc = dataset.load()
    # restrict Binance to the same calendar window for a fair comparison
    b_all = bnc
    win = {k: v.loc[(v.index >= idx.min()) & (v.index <= idx.max())] for k, v in b_all.items()}
    b = ic_table(win, 3e6, "bnc")

    both = b.join(o, lsuffix="_bnc", rsuffix="_okx", how="inner")
    both["same_sign"] = np.sign(both["IC_bnc"]) == np.sign(both["IC_okx"])
    both = both.sort_values("IC_bnc", key=abs, ascending=False)
    pd.set_option("display.width", 200)
    print("\n=== CROSS-EXCHANGE IC COMPARISON (same window) ===")
    print(both[["IC_bnc", "t_bnc", "IC_okx", "t_okx", "same_sign"]].to_string(
        float_format=lambda x: f"{x:+.5f}"))
    agree = both["same_sign"].mean()
    print(f"\nsign agreement: {agree:.1%} of {len(both)} alphas")
    corr = both["IC_bnc"].corr(both["IC_okx"])
    print(f"cross-exchange IC correlation: {corr:+.3f}")
    both.to_csv(os.path.join(RES, "cross_exchange_ic.csv"))
