"""200-period SMA trend filter -- "the past 200 days" as a long-term trend
gate, as requested. Deliberately the simplest module in the indicator
layer: one well-defined, unambiguous calculation, no interpretation calls
needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

DEFAULT_SMA_PERIOD = 200


@dataclass(frozen=True)
class TrendResult:
    direction: Optional[Literal["bullish", "bearish"]]  # None if there isn't enough history yet
    price: float
    sma: Optional[float]


def sma_trend(bars: pd.DataFrame, period: int = DEFAULT_SMA_PERIOD) -> TrendResult:
    price = float(bars["close"].iloc[-1])
    if len(bars) < period:
        return TrendResult(None, price, None)

    sma = float(bars["close"].iloc[-period:].mean())
    direction = "bullish" if price > sma else "bearish"
    return TrendResult(direction, price, sma)
