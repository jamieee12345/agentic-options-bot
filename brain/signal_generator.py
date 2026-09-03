"""Signal generation: turns (symbol, market data, regime state) into a
concrete Signal the rest of the system can act on.

Three layers, in the order data flows through them:
  1. Confidence/uncertainty handling — dampens position size when the HMM
     itself isn't confident, independent of which strategy is active.
  2. Per-tier Strategy classes — thin wrappers around the already-tested
     decision functions in regime_strategies.py, each producing an
     Optional[Signal] (None when there isn't enough data yet).
  3. StrategyOrchestrator — owns the state -> volatility rank -> tier ->
     Strategy mapping and rebuilds it whenever the HMM is retrained, since
     state indices from a fresh fit aren't comparable to the previous fit's.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from brain.regime_strategies import (
    StrategyDecision,
    high_volatility_decision,
    low_volatility_decision,
    mid_volatility_decision,
    rank_states_by_volatility,
    volatility_tier,
)

logger = logging.getLogger(__name__)

MIN_CONFIDENCE_THRESHOLD = 0.55
UNCERTAINTY_SIZE_MULTIPLIER = 0.50  # matches settings.yaml: strategy.uncertainty_size_multiplier
UNCERTAINTY_LEVERAGE = 1.0
UNCERTAINTY_REASON = "UNCERTAINTY - size halved"

REBALANCE_THRESHOLD = 0.10  # matches settings.yaml: strategy.rebalance_threshold

# ASSUMPTION (not specified): flicker window and sensitivity. Flag if you had
# a specific definition in mind.
FLICKER_WINDOW = 5
FLICKER_MIN_TRANSITIONS = 2


# ---------------------------------------------------------------------------
# Regime state input, and the Signal/RegimeInfo output format
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeState:
    """What the regime detector/tracker hands to the signal generator for a
    given symbol at a given point in time. `recent_states` is a trailing
    window of decoded states (most recent last) — maintaining that rolling
    history as new bars arrive is the live loop's job, not this module's.
    """
    state: int
    probability: float
    label: str
    recent_states: Sequence[int]
    timestamp: datetime


@dataclass(frozen=True)
class RegimeInfo:
    name: str
    probability: float
    timestamp: datetime
    reasoning: str
    strategy_name: str


@dataclass(frozen=True)
class Signal:
    symbol: str
    direction: str  # "LONG" or "FLAT"
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    position_size_pct: float
    leverage: float
    regime: RegimeInfo


# ---------------------------------------------------------------------------
# Confidence / uncertainty
# ---------------------------------------------------------------------------

def is_flickering(
    recent_states: Sequence[int],
    window: int = FLICKER_WINDOW,
    min_transitions: int = FLICKER_MIN_TRANSITIONS,
) -> bool:
    """True if the state changed at least `min_transitions` times within the
    trailing `window` observations — i.e. the regime classification is
    bouncing around rather than settled.
    """
    tail = list(recent_states)[-window:]
    if len(tail) < 2:
        return False
    transitions = sum(1 for a, b in zip(tail, tail[1:]) if a != b)
    return transitions >= min_transitions


def apply_uncertainty(
    position_size_pct: float, leverage: float, reasoning: str, regime_state: RegimeState
) -> Tuple[float, float, str]:
    """If confidence is below threshold OR the regime is flickering: halve
    size, force leverage to 1.0x, and append the uncertainty note to the
    reasoning trail. Otherwise pass everything through unchanged.
    """
    low_confidence = regime_state.probability < MIN_CONFIDENCE_THRESHOLD
    flickering = is_flickering(regime_state.recent_states)

    if not (low_confidence or flickering):
        return position_size_pct, leverage, reasoning

    new_size = position_size_pct * UNCERTAINTY_SIZE_MULTIPLIER
    new_reasoning = f"{reasoning} | {UNCERTAINTY_REASON}"
    logger.info(
        "Uncertainty triggered (low_confidence=%s, flickering=%s): size %.3f -> %.3f",
        low_confidence, flickering, position_size_pct, new_size,
    )
    return new_size, UNCERTAINTY_LEVERAGE, new_reasoning


# ---------------------------------------------------------------------------
# Rebalancing gate
# ---------------------------------------------------------------------------

def should_rebalance(current_allocation_pct: float, target_allocation_pct: float, threshold: float = REBALANCE_THRESHOLD) -> bool:
    """Only rebalance when target differs from current by more than
    `threshold`. Filters out churn from minor probability fluctuations —
    fewer trades means less slippage.
    """
    return abs(target_allocation_pct - current_allocation_pct) > threshold


# ---------------------------------------------------------------------------
# Per-tier strategies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _LatestMarketData:
    price: float
    atr: float
    ema_50: float


def _extract_latest(bars: pd.DataFrame) -> Optional[_LatestMarketData]:
    """Pull the fields a strategy needs from the most recent row of `bars`
    (expected to already have features computed — see data/features.py).
    Returns None if there isn't enough history yet for any required field
    (e.g. still inside the EMA/ATR warmup window) — the caller treats that
    as "no signal yet", not an error.
    """
    required = ("close", "atr_14", "price_ema_50")
    if bars.empty or any(col not in bars.columns for col in required):
        return None

    last = bars.iloc[-1]
    if last[list(required)].isna().any():
        return None

    return _LatestMarketData(price=float(last["close"]), atr=float(last["atr_14"]), ema_50=float(last["price_ema_50"]))


def _build_signal(symbol: str, decision: StrategyDecision, regime_state: RegimeState, entry_price: float) -> Signal:
    reasoning = f"{decision.strategy_name}: regime={regime_state.label} (p={regime_state.probability:.2f})"
    size, leverage, reasoning = apply_uncertainty(decision.allocation_pct, decision.leverage, reasoning, regime_state)

    return Signal(
        symbol=symbol,
        direction=decision.direction,
        confidence=regime_state.probability,
        entry_price=entry_price,
        stop_loss=decision.stop_price,
        take_profit=None,
        position_size_pct=size,
        leverage=leverage,
        regime=RegimeInfo(
            name=regime_state.label,
            probability=regime_state.probability,
            timestamp=regime_state.timestamp,
            reasoning=reasoning,
            strategy_name=decision.strategy_name,
        ),
    )


class Strategy(ABC):
    name: str

    @abstractmethod
    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Optional[Signal]:
        raise NotImplementedError


class LowVolatilityBullStrategy(Strategy):
    name = "low_volatility_bull"

    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Optional[Signal]:
        latest = _extract_latest(bars)
        if latest is None:
            return None
        decision = low_volatility_decision(latest.price, latest.atr, latest.ema_50)
        return _build_signal(symbol, decision, regime_state, latest.price)


class MidVolatilityCautiousStrategy(Strategy):
    name = "mid_volatility_cautious"

    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Optional[Signal]:
        latest = _extract_latest(bars)
        if latest is None:
            return None
        decision = mid_volatility_decision(latest.price, latest.atr, latest.ema_50)
        return _build_signal(symbol, decision, regime_state, latest.price)


class HighVolatilityDefensiveStrategy(Strategy):
    name = "high_volatility_defensive"

    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Optional[Signal]:
        latest = _extract_latest(bars)
        if latest is None:
            return None
        decision = high_volatility_decision(latest.price, latest.atr, latest.ema_50)
        return _build_signal(symbol, decision, regime_state, latest.price)


# --- Backward-compatible aliases -------------------------------------------
# Earlier naming used return-based regime labels (crash/bear/neutral/bull/
# euphoria, matching regime_detector.REGIME_LABELS) rather than volatility
# tiers. These are literally the same classes under different names, not new
# behavior. The crash/bear -> high-vol and bull/euphoria -> low-vol pairing
# isn't enforced anywhere in code — it holds empirically, because turbulent
# periods tend to skew negative-return and calm periods positive-return
# (the same correlation regime_detector's post-hoc labeling relies on).
CrashDefensiveStrategy = HighVolatilityDefensiveStrategy
BearTrendStrategy = HighVolatilityDefensiveStrategy
MeanReversionStrategy = MidVolatilityCautiousStrategy
BullTrendStrategy = LowVolatilityBullStrategy
EuphoriaCautiousStrategy = LowVolatilityBullStrategy


# ---------------------------------------------------------------------------
# Strategy orchestrator
# ---------------------------------------------------------------------------

class StrategyOrchestrator:
    """Owns the current state -> volatility rank -> tier -> Strategy
    mapping. Ranks states by volatility, buckets them into low/mid/high, and
    dispatches signal generation per symbol.

    `rebuild_mapping()` must be called every time the HMM is retrained —
    state indices aren't stable across separate fits (state 0 in a fresh fit
    has no relationship to state 0 in the previous one), so the mapping
    can't just be reused.
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, Strategy] = {
            "low": LowVolatilityBullStrategy(),
            "mid": MidVolatilityCautiousStrategy(),
            "high": HighVolatilityDefensiveStrategy(),
        }
        self._rank_by_state: Dict[int, int] = {}
        self._n_states: int = 0

    def rebuild_mapping(self, model, feature_columns: Sequence[str], n_states: int) -> None:
        self._rank_by_state = rank_states_by_volatility(model, feature_columns)
        self._n_states = n_states
        tiers = {state: volatility_tier(rank, n_states) for state, rank in self._rank_by_state.items()}
        logger.info("Rebuilt state->tier mapping for %d states: %s", n_states, tiers)

    def strategy_for_state(self, state: int) -> Strategy:
        if state not in self._rank_by_state:
            raise ValueError(f"state {state} not in current mapping — has rebuild_mapping() been called?")
        rank = self._rank_by_state[state]
        tier = volatility_tier(rank, self._n_states)
        return self._strategies[tier]

    def generate_signals(
        self, symbol_bars: Dict[str, pd.DataFrame], regime_states: Dict[str, RegimeState]
    ) -> Dict[str, Signal]:
        """symbol_bars: symbol -> feature DataFrame (see data/features.py).
        regime_states: symbol -> that symbol's current RegimeState.
        Returns only symbols that actually produced a signal; symbols with
        insufficient data or missing regime info are silently skipped.
        """
        signals: Dict[str, Signal] = {}
        for symbol, bars in symbol_bars.items():
            regime_state = regime_states.get(symbol)
            if regime_state is None:
                continue
            strategy = self.strategy_for_state(regime_state.state)
            signal = strategy.generate_signal(symbol, bars, regime_state)
            if signal is not None:
                signals[symbol] = signal
        return signals
