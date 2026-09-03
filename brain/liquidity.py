"""Liquidity pools (equal highs/lows, where stop orders likely cluster) and
liquidity sweeps (price pierces one of those levels, then closes back on
the other side -- the classic "stop hunt then reverse" pattern).

Same data ceiling as everything else in this project's indicator layer:
there's no real order-book/order-flow data available anywhere (robin_stocks
and yfinance both only give OHLCV), so "liquidity" here means the OHLCV
proxy for it -- clustered swing highs/lows -- not actual resting order
data. Flagged plainly; this is a reasonable proxy, not the real thing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd

from brain.market_structure import SwingPoint

DEFAULT_EQUAL_LEVEL_TOLERANCE_PCT = 0.002  # swing points within 0.2% of each other count as "equal"
DEFAULT_MIN_TOUCHES = 2


@dataclass(frozen=True)
class LiquidityPool:
    kind: Literal["buy_side", "sell_side"]  # buy_side = resting above equal highs; sell_side = resting below equal lows
    price: float
    touches: int


def find_liquidity_pools(
    swing_points: List[SwingPoint],
    tolerance_pct: float = DEFAULT_EQUAL_LEVEL_TOLERANCE_PCT,
    min_touches: int = DEFAULT_MIN_TOUCHES,
) -> List[LiquidityPool]:
    pools: List[LiquidityPool] = []
    for kind, swing_kind, pool_kind in (
        ("high", "high", "buy_side"), ("low", "low", "sell_side"),
    ):
        prices = sorted(p.price for p in swing_points if p.kind == swing_kind)
        cluster: List[float] = []
        for price in prices:
            if cluster and (price - cluster[-1]) / cluster[-1] > tolerance_pct:
                if len(cluster) >= min_touches:
                    pools.append(LiquidityPool(pool_kind, sum(cluster) / len(cluster), len(cluster)))
                cluster = []
            cluster.append(price)
        if len(cluster) >= min_touches:
            pools.append(LiquidityPool(pool_kind, sum(cluster) / len(cluster), len(cluster)))
    return pools


@dataclass(frozen=True)
class LiquiditySweep:
    direction: Literal["bullish", "bearish"]  # bullish = sell-side liquidity swept then reversed up
    level: float
    index: int


def detect_liquidity_sweep(bars: pd.DataFrame, pools: List[LiquidityPool]) -> Optional[LiquiditySweep]:
    """Checks only the most recent bar (same "trade the event" convention
    as fvg_indicators/market_structure): did this bar's wick pierce a
    known liquidity pool, but its CLOSE come back on the other side? That's
    the signature of stops getting run and then price reversing -- treated
    as bullish if sell-side (below-lows) liquidity got swept and price
    closed back above it, bearish if buy-side (above-highs) liquidity got
    swept and price closed back below it.
    """
    if bars.empty:
        return None
    latest = bars.iloc[-1]
    latest_index = len(bars) - 1

    for pool in pools:
        if pool.kind == "sell_side" and latest["low"] < pool.price < latest["close"]:
            return LiquiditySweep("bullish", pool.price, latest_index)
        if pool.kind == "buy_side" and latest["high"] > pool.price > latest["close"]:
            return LiquiditySweep("bearish", pool.price, latest_index)
    return None
