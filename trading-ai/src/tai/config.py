"""Global configuration for the trading-AI research stack."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DATA_ROOT = Path(os.environ.get("TAI_DATA_ROOT", "/home/user/tai-data"))
RAW_DIR = DATA_ROOT / "raw"
PANEL_DIR = DATA_ROOT / "panel"
CACHE_DIR = DATA_ROOT / "cache"
RESULTS_DIR = DATA_ROOT / "results"
for _d in (RAW_DIR, PANEL_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BINANCE_VISION = "https://data.binance.vision"
BINANCE_S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

BARS_PER_DAY = {"1h": 24, "2h": 12, "4h": 6, "15m": 96, "5m": 288, "30m": 48, "1d": 1}
HOURS_PER_YEAR = 24 * 365


@dataclass
class CostModel:
    """Per-side transaction cost model for USD-margined perpetual futures.

    Costs are expressed in basis points of notional traded.

    ``taker_fee_bps``   exchange fee paid when crossing the spread.
    ``maker_fee_bps``   exchange fee paid when resting (can be negative for rebates).
    ``maker_ratio``     fraction of volume expected to be filled passively.
    ``half_spread_bps`` effective half-spread paid on the taker portion.
    ``impact_coef``     square-root market-impact coefficient; impact in bps is
                        ``impact_coef * sqrt(participation)`` where participation is
                        the traded notional divided by the bar's dollar volume.
    """

    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    maker_ratio: float = 0.0
    half_spread_bps: float = 1.5
    impact_coef: float = 25.0
    funding_enabled: bool = True

    @property
    def fee_bps(self) -> float:
        return self.maker_ratio * self.maker_fee_bps + (1.0 - self.maker_ratio) * self.taker_fee_bps

    @property
    def spread_bps(self) -> float:
        # Passive fills earn (do not pay) the half spread; be conservative and credit nothing.
        return (1.0 - self.maker_ratio) * self.half_spread_bps

    @property
    def linear_bps(self) -> float:
        """Participation-independent cost per side, in bps."""
        return self.fee_bps + self.spread_bps


# Cost scenarios used for sensitivity analysis. "base" is the honest default:
# VIP-0 taker fees with BNB discount, full spread crossing, sqrt impact.
COST_SCENARIOS = {
    # A book with a multi-hour horizon does not have to cross the spread: it can
    # work orders passively across the bar and be filled by someone else's urgency.
    # This scenario assumes 80% of the flow rests and earns the maker fee tier.
    # Orders worked entirely passively: the fee tier is the whole cost and the
    # spread is earned rather than paid.  Achievable at a multi-hour horizon, but
    # it assumes away adverse selection on the fills, so it is the optimistic end.
    "maker_only": CostModel(taker_fee_bps=5.0, maker_fee_bps=2.0, maker_ratio=1.0,
                            half_spread_bps=2.0, impact_coef=15.0),
    "passive": CostModel(taker_fee_bps=5.0, maker_fee_bps=2.0, maker_ratio=0.8,
                         half_spread_bps=2.0, impact_coef=20.0),
    "optimistic": CostModel(taker_fee_bps=4.0, maker_fee_bps=1.8, maker_ratio=0.5,
                            half_spread_bps=1.0, impact_coef=15.0),
    "base": CostModel(taker_fee_bps=5.0, maker_fee_bps=2.0, maker_ratio=0.25,
                      half_spread_bps=1.5, impact_coef=25.0),
    "conservative": CostModel(taker_fee_bps=5.0, maker_fee_bps=2.0, maker_ratio=0.0,
                              half_spread_bps=3.0, impact_coef=40.0),
    "brutal": CostModel(taker_fee_bps=7.0, maker_fee_bps=3.0, maker_ratio=0.0,
                        half_spread_bps=5.0, impact_coef=60.0),
}


@dataclass
class UniverseConfig:
    interval: str = "1h"
    start: str = "2020-01"
    end: str = "2025-08"
    min_months: int = 12
    # Point-in-time liquidity screen
    top_n: int = 120
    adv_lookback_days: int = 30
    min_adv_usd: float = 3_000_000.0
    # Exclude the first N bars after a symbol's listing (listing-day chaos / no history)
    listing_burn_bars: int = 24 * 14
    exclude_quote: tuple = ("BUSD", "USDC")
    exclude_symbols: tuple = ()


@dataclass
class StrategyConfig:
    interval: str = "1h"
    rebalance_every: int = 4          # bars between rebalances
    label_horizon: int = 4            # bars ahead the model predicts
    embargo_bars: int = 48            # purge/embargo around train/test split
    train_months: int = 0             # 0 => expanding window
    min_train_bars: int = 24 * 365    # minimum history before first prediction
    test_months: int = 3              # walk-forward step size
    target_ann_vol: float = 0.20      # unlevered target vol of the alpha book
    max_gross: float = 3.0            # gross leverage cap at unit risk scaling
    max_weight: float = 0.06          # per-name cap as fraction of gross
    beta_neutral: bool = True
    dollar_neutral: bool = True
    cost_aware_shrink: float = 1.0    # strength of turnover penalty
    seed: int = 7
