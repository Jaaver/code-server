"""IC measured against EXECUTABLE returns.

A signal built from closing prices can show a large one-bar close-to-close IC
that is pure bid-ask bounce: the "reversal" is the print alternating between bid
and ask, and you pay exactly that spread trying to capture it.

The strategy enters at open(t) and exits at open(t+h), so open-to-open returns
are what it can actually realise. Comparing the two ICs separates real edge from
microstructure illusion.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset, alphas, portfolio, strategy, run_wf

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
HORIZONS = [1, 2, 4, 8, 24, 72, 168]


def main(min_dv=3e6):
    panels = run_wf.screen_universe(dataset.load())
    mask = portfolio.tradable_mask(panels, min_dollar_vol=min_dv)
    print(f"avg tradable/bar: {mask.sum(axis=1).mean():.1f}", flush=True)
    lc = np.log(panels["close"])
    lo = np.log(panels["open"])

    scores = strategy.alpha_panel(
        panels, alphas.ALPHAS,
        post=lambda v: alphas.zscore_xs(v.where(mask)).astype(np.float32))

    rows = []
    for name, sc in scores.items():
        r = {"alpha": name}
        for h in HORIZONS:
            # close-to-close, the naive measure
            cc = lc.shift(-h) - lc
            # open-to-open: signal known at close(t), enter open(t+1), exit open(t+1+h)
            oo = lo.shift(-(h + 1)) - lo.shift(-1)
            r[f"cc_h{h}"] = portfolio.information_coefficient(sc, cc, mask).mean()
            r[f"oo_h{h}"] = portfolio.information_coefficient(sc, oo, mask).mean()
        rows.append(r)
        print(f"  {name:14s} " + " ".join(
            f"h{h}: cc={r[f'cc_h{h}']:+.4f} oo={r[f'oo_h{h}']:+.4f}" for h in (1, 4, 24)), flush=True)

    df = pd.DataFrame(rows).set_index("alpha")
    df.to_csv(os.path.join(RES, "tradable_ic.csv"))
    print("\n=== CLOSE-TO-CLOSE vs EXECUTABLE (OPEN-TO-OPEN) IC ===")
    cols = [c for h in HORIZONS for c in (f"cc_h{h}", f"oo_h{h}")]
    print(df[cols].to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\n=== how much of the 1-bar IC survives execution? ===")
    surv = (df["oo_h1"] / df["cc_h1"].replace(0, np.nan)).sort_values()
    for k, v in surv.items():
        print(f"  {k:14s} cc={df.loc[k,'cc_h1']:+.4f} -> oo={df.loc[k,'oo_h1']:+.4f}  ratio={v:+.2f}")
    return df


if __name__ == "__main__":
    main()
