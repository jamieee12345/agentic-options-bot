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
from datetime import datetime
from typing import List, Literal, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from brain.market_structure import SwingPoint, find_swing_points

MARKET_TZ = ZoneInfo("America/New_York")

DEFAULT_EQUAL_LEVEL_TOLERANCE_PCT = 0.002  # swing points within 0.2% of each other count as "equal"
DEFAULT_MIN_TOUCHES = 2


@dataclass(frozen=True)
class LiquidityPool:
    kind: Literal["buy_side", "sell_side"]  # buy_side = resting above equal highs; sell_side = resting below equal lows
    price: float
    touches: int
    # Where the resting orders are assumed to come from -- decides the
    # sweep model's sizing tier (brain/sweep_strategy.py): "major" pools
    # are the levels everyone watches (previous-day high/low, the opening
    # range, equal highs/lows), "minor" are single swing points.
    tier: Literal["major", "minor"] = "major"
    label: str = "equal_levels"


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
    # The wick extreme of the sweep bar (low for a bullish sweep, high for a
    # bearish one) -- the sweep model's INVALIDATION price: a close beyond
    # it means the sweep failed. Defined at detection, never moved.
    extreme: float = 0.0
    pool: Optional[LiquidityPool] = None


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


# ----------------------------------------------------------------------------- session pools + windowed sweep
# The sweep MODEL (brain/sweep_strategy.py) needs two things the original
# helpers above don't provide: pools that include the levels traders
# actually engineer around (previous-day high/low, the opening range) and
# not just equal swing highs/lows; and sweep detection over a WINDOW of
# recent bars rather than only the latest bar -- the live routine samples
# once an hour and would otherwise see the sweep bar 1 time in 12.

def find_session_pools(
    bars: pd.DataFrame,
    daily_bars: Optional[pd.DataFrame],
    now: datetime,
    opening_range_minutes: int = 15,
    swing_window: int = 5,
    tolerance_pct: float = DEFAULT_EQUAL_LEVEL_TOLERANCE_PCT,
) -> List[LiquidityPool]:
    """Every pool in play right now, tiered:
      major -- previous-day high/low (from `daily_bars`, the last bar
               strictly before today's ET date), today's opening-range
               high/low (first `opening_range_minutes` of regular session
               bars), equal highs/lows (find_liquidity_pools).
      minor -- each remaining single swing high/low in `bars`.
    `bars` are regular-session intraday bars (UTC index)."""
    pools: List[LiquidityPool] = []
    today = now.astimezone(MARKET_TZ).date()

    if daily_bars is not None and not daily_bars.empty:
        mask = [ts.astimezone(MARKET_TZ).date() < today for ts in daily_bars.index]
        prior = daily_bars[mask]
        if not prior.empty:
            pd_bar = prior.iloc[-1]
            pools.append(LiquidityPool("buy_side", float(pd_bar["high"]), 1, "major", "previous_day_high"))
            pools.append(LiquidityPool("sell_side", float(pd_bar["low"]), 1, "major", "previous_day_low"))

    todays = bars[[ts.astimezone(MARKET_TZ).date() == today for ts in bars.index]]
    if not todays.empty:
        first_ts = todays.index[0]
        opening = todays[todays.index < first_ts + pd.Timedelta(minutes=opening_range_minutes)]
        # Only a COMPLETED opening range is a level -- while it's still
        # forming there's nothing to sweep yet.
        if not opening.empty and len(todays) > len(opening):
            pools.append(LiquidityPool("buy_side", float(opening["high"].max()), 1, "major", "opening_range_high"))
            pools.append(LiquidityPool("sell_side", float(opening["low"].min()), 1, "major", "opening_range_low"))

    swings = find_swing_points(bars, swing_window)
    equal = find_liquidity_pools(swings, tolerance_pct)
    pools.extend(LiquidityPool(p.kind, p.price, p.touches, "major", "equal_levels") for p in equal)
    equal_prices = [p.price for p in equal]
    for sp in swings:
        if any(abs(sp.price - ep) / ep <= tolerance_pct for ep in equal_prices):
            continue  # already represented by an equal-levels pool
        pools.append(LiquidityPool(
            "buy_side" if sp.kind == "high" else "sell_side", sp.price, 1, "minor",
            "swing_high" if sp.kind == "high" else "swing_low",
        ))
    return pools


def detect_sweep_in_window(bars: pd.DataFrame, pools: List[LiquidityPool], lookback_bars: int) -> Optional[LiquiditySweep]:
    """The most recent sweep within the last `lookback_bars` bars: a bar
    whose wick traded through a pool but whose close came back on the
    other side. Among sweeps on the same bar, a major pool wins over a
    minor one. Returns None if nothing was swept in the window."""
    if bars.empty or not pools:
        return None
    n = len(bars)
    start = max(0, n - lookback_bars)
    for i in range(n - 1, start - 1, -1):
        bar = bars.iloc[i]
        best: Optional[LiquiditySweep] = None
        for pool in pools:
            if pool.kind == "sell_side" and bar["low"] < pool.price < bar["close"]:
                cand = LiquiditySweep("bullish", pool.price, i, float(bar["low"]), pool)
            elif pool.kind == "buy_side" and bar["high"] > pool.price > bar["close"]:
                cand = LiquiditySweep("bearish", pool.price, i, float(bar["high"]), pool)
            else:
                continue
            if best is None or (best.pool.tier == "minor" and pool.tier == "major"):
                best = cand
        if best is not None:
            return best
    return None
