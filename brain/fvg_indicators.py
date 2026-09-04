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

MINIMUM GAP SIZE (relative to ATR) -- added after a real diagnostic
finding, not a guess: a 20-symbol/58-day intraday backtest showed
`fvg_invalidated` (price closing back through the ENTIRE gap) was the
single largest loss bucket by trade count, 42% of all trades, 0% win rate
by construction. lookback_period/body_multiplier above were never actually
calibrated for the 5-minute bars this strategy runs live on -- they're
just the defaults from the originally supplied reference detect_fvg()
(see settings.yaml's comment). On a 5-minute chart, a geometrically valid
3-candle gap can be trivially small -- well within normal noise for that
symbol's volatility -- while still passing the body/volume multiplier
tests, since those only compare the DISPLACEMENT CANDLE's body/volume to
its own recent average, never the GAP ITSELF to anything. A gap that's
tiny relative to how much this symbol typically moves in a single bar
(ATR) isn't a real imbalance, just noise that happens to satisfy the
letter of the geometric test -- and noise gaps get closed immediately,
which is exactly the failure mode the diagnostic found. This is a
structural fix to an unvalidated default, not a threshold tuned against
that backtest's outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from brain.volatility import DEFAULT_ATR_PERIOD

DEFAULT_LOOKBACK_PERIOD = 10
DEFAULT_BODY_MULTIPLIER = 1.5
DEFAULT_VOLUME_MULTIPLIER = 1.5
# Require the gap itself to span at least half of this symbol's own
# average true range -- i.e., at least half of what a normal single bar's
# high-low range already covers. Chosen as a moderate, principled floor
# (not zero, not a full ATR) rather than searched for against backtest
# results -- see this module's docstring.
DEFAULT_MIN_GAP_ATR_MULTIPLIER = 0.5


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
    min_gap_atr_multiplier: float = DEFAULT_MIN_GAP_ATR_MULTIPLIER,
) -> List[Optional[FairValueGap]]:
    """Same three-candle logic as the original detect_fvg(): candle i-2
    ("first"), i-1 ("middle"/displacement), i ("third"/confirmation).
    Bullish: third candle's low sits above the first candle's high (a true
    price gap), AND the middle candle's body is significantly larger than
    its own recent average (real displacement, not noise), AND the gap
    itself is at least `min_gap_atr_multiplier` times this symbol's own
    ATR (see module docstring for why -- the body/volume checks constrain
    the displacement candle, never the gap's own size). Bearish is the
    mirror. Returns a list the same length as `bars`, with `None` in every
    position that isn't a qualifying gap (including the first two, which
    have no candle i-2).
    """
    required = ("open", "high", "low", "close", "volume")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars is missing required column(s): {missing}")

    # Vectorized true-range/ATR, computed ONCE for the whole series with
    # pandas' own rolling mean, rather than re-sliced and rebuilt from
    # scratch inside the loop below on every single bar (that was the
    # first version of this filter -- correct, but ~4x slower per symbol
    # in backtesting purely from repeated small-Series construction
    # overhead, not from the underlying math). This is also more
    # CORRECT, not just faster: slicing a fresh window per bar meant each
    # window's first true-range value had no real prior close to diff
    # against (shift(1) inside an isolated slice starts at NaN), quietly
    # averaging over one fewer sample than intended at every single bar.
    # Using the real, continuous close series avoids that entirely.
    prev_close = bars["close"].shift(1)
    true_range = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = true_range.rolling(DEFAULT_ATR_PERIOD).mean()

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

        atr = atr_series.iloc[i]
        min_gap_size = (atr * min_gap_atr_multiplier) if (pd.notna(atr) and atr > 0) else 0.0

        if third_low > first_high and middle_body > avg_body_size * body_multiplier and (third_low - first_high) >= min_gap_size:
            result.append(FairValueGap("bullish", first_high, third_low, i, volume_confirmed))
        elif third_high < first_low and middle_body > avg_body_size * body_multiplier and (first_low - third_high) >= min_gap_size:
            result.append(FairValueGap("bearish", first_low, third_high, i, volume_confirmed))
        else:
            result.append(None)

    return result
