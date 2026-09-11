"""Evaluate each alpha standalone: information coefficient + net-of-cost backtest."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio
from engine import Backtester, stats

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RES, exist_ok=True)


def arrays(panels):
    return {k: panels[k].to_numpy(dtype=np.float64) for k in
            ("open", "high", "low", "close", "quote_volume", "funding")}


def make_bt(panels, **kw):
    a = arrays(panels)
    return Backtester(a["open"], a["high"], a["low"], a["close"],
                      a["quote_volume"], a["funding"], **kw)


def main(target_vol=0.25, gross_cap=3.0, smooth=12):
    panels = dataset.load()
    print(f"panel: {panels['close'].shape}  {panels['close'].index.min()} -> {panels['close'].index.max()}")
    mask = portfolio.tradable_mask(panels)
    print(f"avg tradable symbols/bar: {mask.sum(axis=1).mean():.1f}  max: {mask.sum(axis=1).max()}")

    fwd = np.log(panels["close"]).diff().shift(-1)
    bt = make_bt(panels, max_gross_leverage=gross_cap)
    rows = []
    for name, fn in alphas.ALPHAS.items():
        try:
            sc = fn(panels)
        except Exception as e:
            print(f"{name}: FAILED {e}"); continue
        ic = portfolio.information_coefficient(sc, fwd, mask)
        w = portfolio.build_weights(sc, mask, panels, target_vol=target_vol,
                                    gross_cap=gross_cap, smooth=smooth)
        out = bt.run(w.to_numpy(dtype=np.float64))
        st = stats(out["equity"])
        icm = ic.mean(); ics = ic.std()
        rows.append({
            "alpha": name,
            "IC": icm,
            "IC_t": icm / (ics / np.sqrt(ic.notna().sum())) if ics > 0 else np.nan,
            "sharpe": st.get("sharpe", np.nan),
            "cagr": st.get("cagr", np.nan),
            "monthly": st.get("monthly_equiv", np.nan),
            "maxdd": st.get("max_dd", np.nan),
            "ann_turnover": out["turnover"].sum() / 1e6 / st.get("years", 1),
            "cost_drag": out["costs"].sum() / 1e6 / max(st.get("years", 1), 1e-9),
        })
        print(f"{name:14s} IC={icm:+.5f} t={rows[-1]['IC_t']:+6.1f} "
              f"SR={st.get('sharpe',0):+5.2f} CAGR={st.get('cagr',0):+7.1%} "
              f"DD={st.get('max_dd',0):6.1%}", flush=True)

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv(os.path.join(RES, "alpha_scan.csv"), index=False)
    print("\n" + df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
