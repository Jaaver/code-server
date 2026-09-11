"""Minimal dependency-free SVG charts for the results report.

Matplotlib is not a dependency of this project, and the report has to be readable
as a static page, so the few charts it needs are emitted as hand-built inline SVG
that inherits the page's colours through CSS custom properties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _path(xs: np.ndarray, ys: np.ndarray) -> str:
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def _downsample(s: pd.Series, n: int = 900) -> pd.Series:
    if len(s) <= n:
        return s
    step = int(np.ceil(len(s) / n))
    return s.iloc[::step]


def equity_svg(equity: pd.Series, *, width: int = 760, height: int = 260,
               log_scale: bool = True, title: str = "") -> str:
    s = _downsample(equity.dropna())
    if s.empty:
        return ""
    pad_l, pad_r, pad_t, pad_b = 56, 12, 18, 26
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    v = s.to_numpy(dtype="float64")
    v = np.maximum(v, v[v > 0].min() * 1e-3) if (v > 0).any() else v
    y = np.log(v) if log_scale else v
    ylo, yhi = float(np.min(y)), float(np.max(y))
    if yhi - ylo < 1e-9:
        yhi = ylo + 1e-9
    xs = pad_l + np.linspace(0, w, len(s))
    ys = pad_t + h - (y - ylo) / (yhi - ylo) * h

    ticks = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = pad_t + h - frac * h
        val = np.exp(ylo + frac * (yhi - ylo)) if log_scale else ylo + frac * (yhi - ylo)
        mult = val / v[0]
        ticks.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l + w}" y2="{yy:.1f}" '
            f'class="grid"/><text x="{pad_l - 6}" y="{yy + 4:.1f}" class="ylab">'
            f'{mult:,.3g}x</text>')

    n_lab = 5
    xlabs = []
    for i in range(n_lab):
        k = int(i * (len(s) - 1) / (n_lab - 1))
        xlabs.append(f'<text x="{xs[k]:.1f}" y="{height - 8}" class="xlab">'
                     f'{s.index[k].strftime("%Y-%m")}</text>')

    return (f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
            f'aria-label="{title or "equity curve"}">'
            + "".join(ticks)
            + f'<path d="{_path(xs, ys)}" class="line"/>'
            + "".join(xlabs) + "</svg>")


def drawdown_svg(equity: pd.Series, *, width: int = 760, height: int = 150) -> str:
    s = _downsample(equity.dropna())
    if s.empty:
        return ""
    dd = (s / s.cummax() - 1.0).to_numpy(dtype="float64")
    pad_l, pad_r, pad_t, pad_b = 56, 12, 12, 26
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    lo = float(min(dd.min(), -1e-6))
    xs = pad_l + np.linspace(0, w, len(s))
    ys = pad_t + (dd / lo) * h
    area = (f'M{xs[0]:.2f},{pad_t:.2f} '
            + " ".join(f"L{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
            + f" L{xs[-1]:.2f},{pad_t:.2f} Z")
    ticks = "".join(
        f'<line x1="{pad_l}" y1="{pad_t + f * h:.1f}" x2="{pad_l + w}" '
        f'y2="{pad_t + f * h:.1f}" class="grid"/>'
        f'<text x="{pad_l - 6}" y="{pad_t + f * h + 4:.1f}" class="ylab">'
        f'{100 * lo * f:.0f}%</text>' for f in (0.0, 0.5, 1.0))
    return (f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
            f'aria-label="drawdown">{ticks}'
            f'<path d="{area}" class="ddarea"/></svg>')


def monthly_bars_svg(monthly: pd.Series, *, width: int = 760, height: int = 190,
                     target: float | None = None) -> str:
    m = monthly.dropna()
    if m.empty:
        return ""
    pad_l, pad_r, pad_t, pad_b = 56, 12, 14, 34
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    vals = m.to_numpy(dtype="float64")
    hi = float(max(vals.max(), target or 0, 0.01)) * 1.1
    lo = float(min(vals.min(), -0.01)) * 1.1
    zero = pad_t + h * (hi / (hi - lo))
    bw = max(1.6, w / len(vals) * 0.8)
    bars = []
    for i, v in enumerate(vals):
        x = pad_l + (i + 0.1) * (w / len(vals))
        y = pad_t + h * ((hi - v) / (hi - lo))
        bars.append(f'<rect x="{x:.2f}" y="{min(y, zero):.2f}" width="{bw:.2f}" '
                    f'height="{abs(y - zero):.2f}" class="{"up" if v >= 0 else "dn"}"/>')
    tgt = ""
    if target is not None:
        yt = pad_t + h * ((hi - target) / (hi - lo))
        tgt = (f'<line x1="{pad_l}" y1="{yt:.1f}" x2="{pad_l + w}" y2="{yt:.1f}" '
               f'class="target"/><text x="{pad_l + w}" y="{yt - 5:.1f}" '
               f'class="tlab" text-anchor="end">target {100 * target:.0f}%</text>')
    ticks = "".join(
        f'<line x1="{pad_l}" y1="{pad_t + h * ((hi - t) / (hi - lo)):.1f}" '
        f'x2="{pad_l + w}" y2="{pad_t + h * ((hi - t) / (hi - lo)):.1f}" class="grid"/>'
        f'<text x="{pad_l - 6}" y="{pad_t + h * ((hi - t) / (hi - lo)) + 4:.1f}" '
        f'class="ylab">{100 * t:.0f}%</text>'
        for t in np.linspace(lo, hi, 5))
    n_lab = min(8, len(m))
    xlabs = "".join(
        f'<text x="{pad_l + (int(i * (len(m) - 1) / max(n_lab - 1, 1)) + 0.5) * (w / len(m)):.1f}" '
        f'y="{height - 10}" class="xlab">'
        f'{m.index[int(i * (len(m) - 1) / max(n_lab - 1, 1))].strftime("%Y-%m")}</text>'
        for i in range(n_lab))
    return (f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
            f'aria-label="monthly returns">{ticks}{"".join(bars)}{tgt}{xlabs}</svg>')
