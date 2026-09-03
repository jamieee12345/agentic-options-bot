"""Elliott Wave: a simplified, rules-only heuristic -- NOT a full Elliott
Wave count.

Stated plainly, because this is more important to be honest about here
than almost anywhere else in this codebase: genuine Elliott Wave analysis
is famously subjective (professional analysts routinely disagree on the
same chart's wave count) and doesn't reduce cleanly to an algorithm. What
IS objective and codifiable are its three hard validity rules, so that's
all this implements -- given the last 6 alternating swing points
(market_structure.find_swing_points), check whether they satisfy:

    Rule 1: wave 2 never retraces more than 100% of wave 1
    Rule 2: wave 3 is never the shortest of waves 1, 3, and 5
    Rule 3: wave 4 never enters wave 1's price territory (no overlap)

If all three hold, this is a VALID (not necessarily "correct" -- there is
no ground truth) 5-wave impulse count. Because this only recognizes an
impulse using swing points that already exist, it only ever detects an
impulse that has ALREADY COMPLETED -- this is not a live "we're currently
in wave 3" momentum tool, it's a "a 5-wave move just finished, so classic
theory says a corrective move is next" tool. Used that way in
options_strategy.py: a confirmed impulse in one direction is treated as
confluence FOR a trade in the OPPOSITE direction (the expected correction),
not for continuation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from brain.market_structure import SwingPoint


@dataclass(frozen=True)
class ElliottWaveResult:
    valid_impulse: bool
    impulse_direction: Optional[Literal["bullish", "bearish"]]
    expected_correction_direction: Optional[Literal["bullish", "bearish"]]  # opposite of impulse_direction, or None


def _alternating_tail(swing_points: List[SwingPoint], count: int) -> Optional[List[SwingPoint]]:
    """Walks backward through swing_points, keeping only a strictly
    alternating high/low/high/low... sequence (find_swing_points doesn't
    itself guarantee alternation -- two swing highs can appear in a row if
    no qualifying low fell between them). Returns the last `count` of that
    alternating sequence, oldest first, or None if there aren't enough.
    """
    alternating: List[SwingPoint] = []
    for point in reversed(swing_points):
        if not alternating or point.kind != alternating[-1].kind:
            alternating.append(point)
        if len(alternating) == count:
            break
    if len(alternating) < count:
        return None
    return list(reversed(alternating))


def detect_impulse(swing_points: List[SwingPoint]) -> ElliottWaveResult:
    tail = _alternating_tail(swing_points, 6)
    if tail is None:
        return ElliottWaveResult(False, None, None)

    p0, p1, p2, p3, p4, p5 = tail
    bullish = p0.kind == "low"  # alternation guarantees the rest follow: high,low,high,low,high

    if bullish:
        wave1 = p1.price - p0.price
        wave3 = p3.price - p2.price
        wave5 = p5.price - p4.price
        rule1 = p2.price > p0.price                 # wave 2 doesn't retrace past wave 1's start
        rule2 = wave3 >= min(wave1, wave3, wave5) and not (wave3 < wave1 and wave3 < wave5)
        rule3 = p4.price > p1.price                  # wave 4 stays above wave 1's high -- no overlap
        direction = "bullish"
    else:
        wave1 = p0.price - p1.price
        wave3 = p2.price - p3.price
        wave5 = p4.price - p5.price
        rule1 = p2.price < p0.price
        rule2 = not (wave3 < wave1 and wave3 < wave5)
        rule3 = p4.price < p1.price
        direction = "bearish"

    if wave1 <= 0 or wave3 <= 0 or wave5 <= 0 or not (rule1 and rule2 and rule3):
        return ElliottWaveResult(False, None, None)

    correction = "bearish" if direction == "bullish" else "bullish"
    return ElliottWaveResult(True, direction, correction)
