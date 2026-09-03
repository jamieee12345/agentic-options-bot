"""Supply/demand zones: the base candle(s) immediately before a strong
displacement move.

Deliberately reuses the same "displacement body is significantly larger
than its trailing average" test as fvg_indicators.py's FVG detection --
both concepts are describing the same underlying phenomenon (a strong,
committed move), just naming different parts of it: an FVG is the PRICE
GAP a displacement move leaves behind; a supply/demand zone is the BASE
CANDLE the displacement move launched FROM. Keeping the same significance
test means a "demand zone" and a "bullish FVG" found on the same
displacement candle agree with each other by construction, rather than
disagreeing because two different thresholds were used to describe the
same event.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import pandas as pd

from brain.fvg_indicators import DEFAULT_BODY_MULTIPLIER, DEFAULT_LOOKBACK_PERIOD


@dataclass(frozen=True)
class Zone:
    kind: Literal["demand", "supply"]
    low: float
    high: float
    index: int    # index of the BASE candle (not the displacement candle that followed it)


def detect_zones(
    bars: pd.DataFrame,
    lookback_period: int = DEFAULT_LOOKBACK_PERIOD,
    body_multiplier: float = DEFAULT_BODY_MULTIPLIER,
) -> List[Zone]:
    """A demand zone forms when candle i's body is a significant
    displacement (per the same avg-body-size test as FVG detection) AND
    it's bullish (close > open) -- the zone is candle i-1's full range (the
    "base" the move left from). Supply is the mirror (bearish displacement).
    Scans the whole DataFrame (unlike the FVG/BOS detectors, which only
    look at the latest bar) since zones are used as reference levels a
    later bar's price action gets checked against, not as a per-bar trigger
    themselves.
    """
    zones: List[Zone] = []
    for i in range(1, len(bars)):
        window_start = max(0, i - lookback_period)
        prev_bodies = (bars["close"].iloc[window_start:i] - bars["open"].iloc[window_start:i]).abs()
        avg_body_size = prev_bodies.mean()
        avg_body_size = avg_body_size if avg_body_size > 0 else 0.001

        body = bars["close"].iloc[i] - bars["open"].iloc[i]
        if abs(body) <= avg_body_size * body_multiplier:
            continue

        base = bars.iloc[i - 1]
        if body > 0:
            zones.append(Zone("demand", float(base["low"]), float(base["high"]), i - 1))
        else:
            zones.append(Zone("supply", float(base["low"]), float(base["high"]), i - 1))
    return zones


def price_in_zone(price: float, zone: Zone) -> bool:
    return zone.low <= price <= zone.high


def nearest_zone(price: float, zones: List[Zone], kind: str) -> "Zone | None":
    candidates = [z for z in zones if z.kind == kind]
    if not candidates:
        return None
    return min(candidates, key=lambda z: abs((z.low + z.high) / 2 - price))
