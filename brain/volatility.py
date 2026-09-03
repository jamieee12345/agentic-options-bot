"""Average True Range (ATR) volatility-expansion check -- a 7th soft
confluence check, added on request to broaden the vote. The idea: a real
momentum move should show up as an ABOVE-average trading RANGE, not just
an above-average close-to-close body (brain/fvg_indicators.py's own
body_multiplier check already covers body size) -- range captures
wicks/intrabar volatility that body size alone misses, so this isn't
redundant with the FVG's own displacement check, just a related signal
read a different way.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

DEFAULT_ATR_PERIOD = 14
# 1.0x, not stricter -- this is one soft check among several feeding a
# score, not a standalone volatility filter, so it doesn't need its own
# wide margin.
DEFAULT_EXPANSION_MULTIPLIER = 1.0


def compute_atr(bars: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    prev_close = bars["close"].shift(1)
    true_range = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(true_range.iloc[-period:].mean())


def volatility_expansion_check(
    bars: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD, expansion_multiplier: float = DEFAULT_EXPANSION_MULTIPLIER,
) -> str:
    """"pass" if the latest bar's own range is at or above the recent ATR
    average -- a real expansion, not a quiet bar drifting into a technical
    gap. "fail" if the latest bar's range is below-average (low
    conviction). "n/a" if there isn't enough history yet to compute ATR.
    """
    atr = compute_atr(bars, period)
    if atr is None or atr <= 0:
        return "n/a"
    latest_range = float(bars["high"].iloc[-1] - bars["low"].iloc[-1])
    return "pass" if latest_range >= atr * expansion_multiplier else "fail"
