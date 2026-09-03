"""Swing points, trend structure, and break-of-structure (BOS) detection.

All of this is fractal-based (a swing high is a bar whose high is the
highest within a window on both sides; swing low is the mirror) since
that's the only market-structure definition computable from OHLCV bars
alone -- there's no order-flow/order-book data available anywhere in this
project's data layer to do anything fancier. Flagged as an interpretation
call, same as everywhere else in this codebase: "market structure" has no
single universally agreed definition, this is the swing-high/swing-low one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd

DEFAULT_SWING_WINDOW = 5  # bars on each side that must be lower/higher for a point to count as a swing


@dataclass(frozen=True)
class SwingPoint:
    kind: Literal["high", "low"]
    index: int
    price: float


def find_swing_points(bars: pd.DataFrame, swing_window: int = DEFAULT_SWING_WINDOW) -> List[SwingPoint]:
    highs, lows = bars["high"], bars["low"]
    points: List[SwingPoint] = []
    for i in range(swing_window, len(bars) - swing_window):
        window_highs = highs.iloc[i - swing_window: i + swing_window + 1]
        window_lows = lows.iloc[i - swing_window: i + swing_window + 1]
        if highs.iloc[i] == window_highs.max() and (window_highs == window_highs.max()).sum() == 1:
            points.append(SwingPoint("high", i, float(highs.iloc[i])))
        if lows.iloc[i] == window_lows.min() and (window_lows == window_lows.min()).sum() == 1:
            points.append(SwingPoint("low", i, float(lows.iloc[i])))
    return sorted(points, key=lambda p: p.index)


def classify_structure(swing_points: List[SwingPoint]) -> Literal["uptrend", "downtrend", "ranging"]:
    """Uptrend: the last two swing highs are rising AND the last two swing
    lows are rising (higher highs + higher lows). Downtrend is the mirror.
    Anything else (not enough swings yet, or highs/lows disagree) is
    "ranging" -- deliberately the conservative default, not a guess.
    """
    recent_highs = [p for p in swing_points if p.kind == "high"][-2:]
    recent_lows = [p for p in swing_points if p.kind == "low"][-2:]
    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return "ranging"

    higher_highs = recent_highs[-1].price > recent_highs[-2].price
    higher_lows = recent_lows[-1].price > recent_lows[-2].price
    lower_highs = recent_highs[-1].price < recent_highs[-2].price
    lower_lows = recent_lows[-1].price < recent_lows[-2].price

    if higher_highs and higher_lows:
        return "uptrend"
    if lower_highs and lower_lows:
        return "downtrend"
    return "ranging"


@dataclass(frozen=True)
class BreakOfStructure:
    direction: Literal["bullish", "bearish"]
    level: float
    index: int


def detect_break_of_structure(bars: pd.DataFrame, swing_points: List[SwingPoint]) -> Optional[BreakOfStructure]:
    """Bullish BOS: the most recent bar's close trades above the most
    recent prior swing high (confirms upside structure has broken to a new
    level, i.e. buyers are in control past where sellers previously
    defended). Bearish is the mirror. Only checks the LATEST bar, matching
    fvg_indicators.py's "trade the event, not stale history" convention.
    """
    if bars.empty:
        return None
    latest_close = float(bars["close"].iloc[-1])
    latest_index = len(bars) - 1

    prior_highs = [p for p in swing_points if p.kind == "high" and p.index < latest_index]
    prior_lows = [p for p in swing_points if p.kind == "low" and p.index < latest_index]

    if prior_highs and latest_close > prior_highs[-1].price:
        return BreakOfStructure("bullish", prior_highs[-1].price, latest_index)
    if prior_lows and latest_close < prior_lows[-1].price:
        return BreakOfStructure("bearish", prior_lows[-1].price, latest_index)
    return None
