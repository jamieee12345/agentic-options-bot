"""RSI(14) momentum check -- a 6th soft confluence check, added on request
to broaden the vote beyond pure price-structure indicators (market
structure, S/R, supply/demand, liquidity, volume profile all read the
SAME price series; RSI reads momentum, a genuinely different signal).
Standard Wilder RSI (EWM smoothing), not a naive rolling-mean variant --
matches what every chart platform actually shows.
"""
from __future__ import annotations

from typing import Literal, Optional

import pandas as pd

DEFAULT_RSI_PERIOD = 14
OVERBOUGHT = 80.0
OVERSOLD = 20.0


def compute_rsi(bars: pd.DataFrame, period: int = DEFAULT_RSI_PERIOD) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    delta = bars["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    latest_gain = float(avg_gain.iloc[-1])
    latest_loss = float(avg_loss.iloc[-1])
    if latest_loss == 0:
        return 100.0
    rs = latest_gain / latest_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_momentum_check(bars: pd.DataFrame, direction: Literal["bullish", "bearish"], period: int = DEFAULT_RSI_PERIOD) -> str:
    """"pass" if momentum genuinely supports the direction -- above the 50
    midline for bullish, below it for bearish -- without already sitting at
    an exhaustion extreme (>=OVERBOUGHT for bullish, <=OVERSOLD for
    bearish): entering right as RSI peaks is chasing the move, not catching
    it. "fail" if momentum disagrees or is already exhausted. "n/a" if
    there isn't enough history yet to compute RSI.
    """
    rsi = compute_rsi(bars, period)
    if rsi is None:
        return "n/a"
    if direction == "bullish":
        return "pass" if 50.0 < rsi < OVERBOUGHT else "fail"
    return "pass" if OVERSOLD < rsi < 50.0 else "fail"
