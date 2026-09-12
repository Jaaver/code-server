"""Assemble every result file into the final written report."""
import os, json, glob
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
REP = os.path.join(ROOT, "reports")
os.makedirs(REP, exist_ok=True)


def load(name, default=None):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return default
    if name.endswith(".json"):
        return json.load(open(p))
    return pd.read_csv(p, index_col=0)


def fmt_pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x*100:.{d}f}%"


def table(df, cols=None, floatfmt="{:+.3f}"):
    if df is None or len(df) == 0:
        return "_(not available)_"
    d = df[cols] if cols else df
    return d.to_markdown(floatfmt=".4f")


if __name__ == "__main__":
    print("report helpers ready")
