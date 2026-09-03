"""Correlation-based size control: "max correlated exposure" from the spec.

Interpretation call (flagged, not fully specified by the spec): "size
reduction above 0.7, rejection above 0.85" is implemented as a straight
line from multiplier 1.0 at correlation 0.70 down to multiplier 0.0 at
0.85, rather than a step function. A step (1.0 up to 0.70, then some fixed
reduced size, then 0 above 0.85) would create a discontinuity where a
candidate at 0.701 correlation is treated identically to one at 0.849 --
the linear ramp instead makes "more correlated with what you already hold"
continuously worse, which matches the spirit of a risk control better than
an arbitrary cliff.

Correlation is computed against every currently-held symbol independently;
the WORST (highest) pairwise correlation binds, since that's the position
that would move most in lockstep with the candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

DEFAULT_LOOKBACK_DAYS = 60          # settings.yaml: risk.correlation_lookback_days
DEFAULT_REDUCE_THRESHOLD = 0.70     # settings.yaml: risk.correlation_reduce_threshold
DEFAULT_REJECT_THRESHOLD = 0.85     # settings.yaml: risk.correlation_reject_threshold


@dataclass(frozen=True)
class CorrelationResult:
    multiplier: float                        # 0.0 (rejected) .. 1.0 (no reduction)
    worst_correlation: Optional[float]        # None if there were no held positions to compare against
    worst_correlated_symbol: Optional[str]
    reason: str


def rolling_correlation(returns_a: pd.Series, returns_b: pd.Series, window: int = DEFAULT_LOOKBACK_DAYS) -> Optional[float]:
    """Pearson correlation of daily returns over the trailing `window` bars,
    aligned on their shared index. Returns None (not 0.0 -- absence of
    evidence isn't evidence of zero correlation) if there isn't enough
    overlapping history to compute it.
    """
    aligned = pd.concat([returns_a, returns_b], axis=1, join="inner").tail(window)
    if len(aligned) < max(10, window // 3):
        return None
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return None if pd.isna(corr) else float(corr)


def _multiplier_for_correlation(
    corr: float, reduce_threshold: float, reject_threshold: float
) -> float:
    if corr >= reject_threshold:
        return 0.0
    if corr <= reduce_threshold:
        return 1.0
    span = reject_threshold - reduce_threshold
    return 1.0 - (corr - reduce_threshold) / span


def correlation_size_multiplier(
    candidate_symbol: str,
    candidate_returns: pd.Series,
    held_returns_by_symbol: Dict[str, pd.Series],
    window: int = DEFAULT_LOOKBACK_DAYS,
    reduce_threshold: float = DEFAULT_REDUCE_THRESHOLD,
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
) -> CorrelationResult:
    """Compares `candidate_symbol` against every symbol in
    `held_returns_by_symbol` (expected: current open positions, excluding
    the candidate itself if it's already held -- adding to an existing
    position isn't a new correlated bet). Returns a size multiplier to
    apply to an otherwise-approved buy order.
    """
    held_returns_by_symbol = {
        sym: series for sym, series in held_returns_by_symbol.items() if sym != candidate_symbol
    }
    if not held_returns_by_symbol:
        return CorrelationResult(1.0, None, None, "no existing positions to compare against")

    worst_corr: Optional[float] = None
    worst_symbol: Optional[str] = None
    skipped_for_insufficient_data = []
    for symbol, held_returns in held_returns_by_symbol.items():
        corr = rolling_correlation(candidate_returns, held_returns, window)
        if corr is None:
            skipped_for_insufficient_data.append(symbol)
            continue
        if worst_corr is None or corr > worst_corr:
            worst_corr, worst_symbol = corr, symbol

    if worst_corr is None:
        return CorrelationResult(
            1.0, None, None,
            f"insufficient overlapping history to compute correlation against: {skipped_for_insufficient_data}",
        )

    multiplier = _multiplier_for_correlation(worst_corr, reduce_threshold, reject_threshold)
    if multiplier == 0.0:
        reason = f"correlation with {worst_symbol} ({worst_corr:.2f}) >= reject threshold ({reject_threshold:.2f})"
    elif multiplier < 1.0:
        reason = f"correlation with {worst_symbol} ({worst_corr:.2f}) reduced size to {multiplier:.0%}"
    else:
        reason = f"worst correlation ({worst_symbol}, {worst_corr:.2f}) below reduce threshold ({reduce_threshold:.2f})"

    return CorrelationResult(multiplier, worst_corr, worst_symbol, reason)
