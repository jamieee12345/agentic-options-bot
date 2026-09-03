"""Turns Fair Value Gap + volume confirmation into a long-call/long-put
options decision, gated by brain/confluence.py's multi-indicator confluence
check. Independent of the equity regime brain (brain/signal_generator.Signal)
entirely -- this only needs raw OHLCV bars. One real upside of that
independence: the options pipeline doesn't need a real RegimeProvider to be
built before it can run, only a real DataFeed. See README's "Next up" list.

Two-stage decision, in order:
  1. TRIGGER -- is there a fresh Fair Value Gap on the MOST RECENT bar
     (brain/fvg_indicators.py), confirmed by above-average volume on the
     displacement candle? Without this, nothing else runs -- no candidate
     direction to even evaluate.
  2. CONFLUENCE GATE -- given a candidate direction, does it clear
     brain/confluence.py's combined check across market structure,
     support/resistance, supply/demand, liquidity sweeps, volume profile,
     Elliott Wave, and the 200-SMA trend? See that module's docstring for
     the important honesty note about what "confluence score" does and
     does NOT mean (a rule-agreement count, not a backtested probability).

"hold" (no fresh trigger, or a trigger that fails confluence) is
deliberately NOT the same as "close": the absence of a new signal doesn't
by itself exit whatever's already open -- see
orchestration/options_execution.py for what actually closes a position
(a fresh opposing signal, or approaching expiration).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from brain.confluence import DEFAULT_MIN_CONFLUENCE_SCORE, evaluate_confluence
from brain.fvg_indicators import (
    DEFAULT_BODY_MULTIPLIER,
    DEFAULT_LOOKBACK_PERIOD,
    DEFAULT_VOLUME_MULTIPLIER,
    detect_fair_value_gaps,
)
from brain.trend_indicators import DEFAULT_SMA_PERIOD

MIN_BARS_REQUIRED = 3  # need at least one full 3-candle window to evaluate the most recent bar


@dataclass(frozen=True)
class OptionsDecision:
    symbol: str
    action: str              # "buy_call", "buy_put", "hold", or "close"
    conviction: float        # 0.0-1.0, scales the premium budget in safety/options_sizing.py
    reasoning: str
    # The triggering gap's bounds, set only on buy_call/buy_put -- carried
    # through so the executor can persist them (orchestration/trade_log.py)
    # and later check for FVG invalidation (price closing back through the
    # entire gap) as a structural stop, independent of the option's own
    # dollar P&L. None for hold/close, where there's no gap to remember.
    gap_low: Optional[float] = None
    gap_high: Optional[float] = None
    # Everything below is for the dashboard's live-reasoning panel ("what
    # exactly is the bot looking at right now") -- none of it feeds any
    # decision, it's the already-computed inputs to the decision above,
    # carried out so a caller doesn't have to recompute them to explain the
    # "hold". gap_kind/volume_confirmed are set whenever a fresh FVG was
    # found on the latest bar at all, even if it didn't clear volume
    # confirmation or confluence. confluence_* are set only once confluence
    # actually ran (gap + volume both cleared) -- empty/None before that,
    # since there's no candidate direction to have evaluated it against yet.
    gap_kind: Optional[str] = None
    volume_confirmed: Optional[bool] = None
    confluence_details: Dict[str, str] = field(default_factory=dict)
    confluence_score: Optional[float] = None
    confluence_applicable: int = 0


def decide_options_action(
    symbol: str,
    bars: pd.DataFrame,
    lookback_period: int = DEFAULT_LOOKBACK_PERIOD,
    body_multiplier: float = DEFAULT_BODY_MULTIPLIER,
    volume_multiplier: float = DEFAULT_VOLUME_MULTIPLIER,
    sma_period: int = DEFAULT_SMA_PERIOD,
    min_confluence_score: float = DEFAULT_MIN_CONFLUENCE_SCORE,
    daily_bars: Optional[pd.DataFrame] = None,
) -> OptionsDecision:
    """`bars` is whatever interval the live strategy is actually watching
    for FVG/momentum (intraday, for live trading -- see
    orchestration/run_live.py). `daily_bars`, if provided, is used ONLY for
    the 200-SMA trend veto inside evaluate_confluence -- omit it (as the
    backtester does) when `bars` already IS daily.
    """
    if len(bars) < MIN_BARS_REQUIRED:
        return OptionsDecision(symbol, "hold", 0.0, f"only {len(bars)} bar(s) available, need at least {MIN_BARS_REQUIRED}")

    gaps = detect_fair_value_gaps(bars, lookback_period, body_multiplier, volume_multiplier)
    latest_gap = gaps[-1]  # the most recent bar's result -- "fresh" means formed right now, not a stale historical gap

    if latest_gap is None:
        return OptionsDecision(symbol, "hold", 0.0, "no fair value gap on the most recent bar")

    if not latest_gap.volume_confirmed:
        return OptionsDecision(
            symbol, "hold", 0.0,
            f"{latest_gap.kind} FVG formed (${latest_gap.gap_low:.2f}-${latest_gap.gap_high:.2f}) but volume didn't confirm it -- treating as noise",
            gap_kind=latest_gap.kind, volume_confirmed=False,
        )

    direction = "bullish" if latest_gap.kind == "bullish" else "bearish"
    confluence = evaluate_confluence(
        bars, direction, sma_period=sma_period, min_confluence_score=min_confluence_score,
        fvg_lookback_period=lookback_period, fvg_body_multiplier=body_multiplier, daily_bars=daily_bars,
    )

    if not confluence.passed:
        return OptionsDecision(
            symbol, "hold", 0.0,
            f"{latest_gap.kind} FVG + volume triggered, but confluence check failed: {confluence.veto_reason}",
            gap_kind=latest_gap.kind, volume_confirmed=True,
            confluence_details=confluence.details, confluence_score=confluence.score,
            confluence_applicable=confluence.applicable_checks,
        )

    action = "buy_call" if direction == "bullish" else "buy_put"
    score_str = f"{confluence.score:.0%}" if confluence.score is not None else "n/a"
    reasoning = (
        f"{latest_gap.kind} FVG (${latest_gap.gap_low:.2f}-${latest_gap.gap_high:.2f}) confirmed by volume, "
        f"confluence {score_str} across {confluence.applicable_checks} check(s): {confluence.details}"
    )
    return OptionsDecision(
        symbol, action, confluence.score or 1.0, reasoning, gap_low=latest_gap.gap_low, gap_high=latest_gap.gap_high,
        gap_kind=latest_gap.kind, volume_confirmed=True,
        confluence_details=confluence.details, confluence_score=confluence.score,
        confluence_applicable=confluence.applicable_checks,
    )
