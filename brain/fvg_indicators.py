"""Fair Value Gap (FVG) detection + volume confirmation.

`detect_fair_value_gaps()` below is the user-supplied `detect_fvg()`
function, translated to this project's lowercase OHLCV column convention
(`data/fetchers.py` already normalizes everything to
open/high/low/close/volume) and returning a typed `FairValueGap` instead of
a bare tuple, but the actual 3-candle detection logic -- the displacement-
body-vs-average-body check included -- is unchanged. If you tweak the
math, tweak it here so there's one definition instead of two drifting apart.

Volume confirmation is a SEPARATE, additive check, not fused into gap
detection itself: a gap either geometrically exists or it doesn't (per the
original function's own criteria), and `volume_confirmed` records whether
the displacement candle *also* traded on above-average volume. Keeping
these independent means the strategy layer (options_strategy.py) can choose
to require both (which is what was asked: "fair value gaps and volume where
momentum is displayed") without baking that requirement into the detector
itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

DEFAULT_LOOKBACK_PERIOD = 10
DEFAULT_BODY_MULTIPLIER = 1.5
DEFAULT_VOLUME_MULTIPLIER = 1.5


@dataclass(frozen=True)
class FairValueGap:
    kind: str          # "bullish" or "bearish"
    gap_low: float
    gap_high: float
    index: int          # positional index into the bars DataFrame of the third (confirming) candle
    volume_confirmed: bool


def detect_fair_value_gaps(
    bars: pd.DataFrame,
    lookback_period: int = DEFAULT_LOOKBACK_PERIOD,
    body_multiplier: float = DEFAULT_BODY_MULTIPLIER,
    volume_multiplier: float = DEFAULT_VOLUME_MULTIPLIER,
) -> List[Optional[FairValueGap]]:
    """Same three-candle logic as the original detect_fvg(): candle i-2
    ("first"), i-1 ("middle"/displacement), i ("third"/confirmation).
    Bullish: third candle's low sits above the first candle's high (a true
    price gap), AND the middle candle's body is significantly larger than
    its own recent average (real displacement, not noise). Bearish is the
    mirror. Returns a list the same length as `bars`, with `None` in every
    position that isn't a qualifying gap (including the first two, which
    have no candle i-2).
    """
    required = ("open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing required column(s): {missing}")

    result: List[Optional[FairValueGap]] = [None, None]

    for i in range(2, len(bars)):
        first_high = bars["high"].iloc[i - 2]
        first_low = bars["low"].iloc[i - 2]
        middle_open = bars["open"].iloc[i - 1]
        middle_close = bars["close"].iloc[i - 1]
        middle_volume = bars["volume"].iloc[i - 1]
        third_low = bars["low"].iloc[i]
        third_high = bars["high"].iloc[i]

        window_start = max(0, i - 1 - lookback_period)
        prev_bodies = (bars["close"].iloc[window_start:i - 1] - bars["open"].iloc[window_start:i - 1]).abs()
        avg_body_size = prev_bodies.mean()
        avg_body_size = avg_body_size if avg_body_size > 0 else 0.001

        prev_volumes = bars["volume"].iloc[window_start:i - 1]
        avg_volume = prev_volumes.mean()
        avg_volume = avg_volume if avg_volume > 0 else 0.001

        middle_body = abs(middle_close - middle_open)
        volume_confirmed = middle_volume > avg_volume * volume_multiplier

        if third_low > first_high and middle_body > avg_body_size * body_multiplier:
            result.append(FairValueGap("bullish", first_high, third_low, i, volume_confirmed))
        elif third_high < first_low and middle_body > avg_body_size * body_multiplier:
            result.append(FairValueGap("bearish", first_low, third_high, i, volume_confirmed))
        else:
            result.append(None)

    return result
