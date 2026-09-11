#!/usr/bin/env python3
"""Sequential forward paper-trading run over a period the model never saw.

Unlike the vectorised backtest, this walks the clock one bar at a time and hands
the strategy only a rolling window of history, exercising the same code path that
would run in production.  Agreement between this and the backtest is evidence the
backtest is not quietly using information the live system would not have.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tai.backtest.engine import ExecConfig
from tai.config import COST_SCENARIOS, RESULTS_DIR, StrategyConfig, UniverseConfig
from tai.data import panel as pm
from tai.evaluation import metrics as M
from tai.live.paper_trader import LiveStrategy, PaperBroker, VolTargeter, run_forward
from tai.models.store import load_model

log = logging.getLogger("paper")
FIELDS = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
          "taker_buy_quote_volume", "funding_rate", "funding_last"]


class PanelWindowProvider:
    """Serves a strictly trailing window plus the next bar's open/volume."""

    def __init__(self, panel: dict, lookback: int):
        self.panel = panel
        self.lookback = lookback
        self.index = panel["close"].index
        self.pos = {ts: i for i, ts in enumerate(self.index)}

    def __call__(self, ts):
        i = self.pos.get(ts)
        if i is None or i + 1 >= len(self.index):
            return None
        lo = max(0, i - self.lookback + 1)
        window = {k: self.panel[k].iloc[lo:i + 1] for k in FIELDS if k in self.panel}
        nxt = self.index[i + 1]
        nxt_open = self.panel["open"].loc[nxt]
        nxt_vol = self.panel["quote_volume"].loc[nxt]
        fr = self.panel["funding_rate"].loc[nxt]
        return {"window": window,
                "next_open": {k: float(v) for k, v in nxt_open.items() if np.isfinite(v)},
                "next_volume": {k: float(v) for k, v in nxt_vol.items() if np.isfinite(v)},
                "funding": {k: float(v) for k, v in fr.items() if np.isfinite(v)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--lookback", type=int, default=2880)
    ap.add_argument("--rebalance", type=int, default=4)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--cost", default="base")
    ap.add_argument("--target-vol", type=float, default=0.20)
    ap.add_argument("--max-gross", type=float, default=4.0)
    ap.add_argument("--tag", default="forward")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bundle = load_model(args.model)
    ens, ucfg = bundle["ensemble"], bundle["ucfg"]
    cutoff = pd.Timestamp(bundle["cutoff"], tz="UTC")
    start = pd.Timestamp(args.start, tz="UTC")
    if start < cutoff:
        raise SystemExit(f"forward start {start} precedes the model cutoff {cutoff}")

    panel = pm.load_panel("1h")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else panel["close"].index[-1]
    idx = panel["close"].index
    ts_list = [t for t in idx if start <= t <= end]
    log.info("forward window %s .. %s (%d bars)", ts_list[0], ts_list[-1], len(ts_list))

    exec_cfg = ExecConfig(target_ann_vol=args.target_vol, leverage=args.leverage,
                          max_gross=args.max_gross, bars_per_day=24)
    strat = LiveStrategy(model=ens, ucfg=ucfg, exec_cfg=exec_cfg, bars_per_day=24)
    broker = PaperBroker(cost=COST_SCENARIOS[args.cost], cfg=exec_cfg)
    vt = VolTargeter(args.target_vol, 24 * 30, 24)
    provider = PanelWindowProvider(panel, args.lookback)

    df = run_forward(provider, strat, broker, ts_list, rebalance_every=args.rebalance,
                     leverage=args.leverage, vol_targeter=vt)

    out = RESULTS_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"forward_{args.cost}_lev{args.leverage:g}.csv")
    stats = {
        "sharpe": M.sharpe(df["ret"], 24), "cagr": M.cagr(df["equity"]),
        "max_drawdown": M.max_drawdown(df["equity"]),
        "ann_vol": float(df["ret"].std() * np.sqrt(24 * 365)),
        "avg_gross": float(df["gross"].mean()),
        "total_costs": broker.total_costs, "total_funding": broker.total_funding,
        "final_equity": float(df["equity"].iloc[-1]),
        **M.monthly_stats(df["equity"]),
    }
    (out / f"forward_{args.cost}_lev{args.leverage:g}.json").write_text(
        json.dumps(stats, indent=2, default=str))
    log.info("forward: %s", {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in stats.items()})
    json.dump(broker.ledger[-50:], open(out / "recent_orders.json", "w"), indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
