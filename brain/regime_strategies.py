"""Per-regime allocation strategies.

Three volatility tiers (low / mid / high, by expected volatility), each
resolving to a concrete allocation decision given current price, ATR, and
50-EMA. All three are LONG-only or flat — this framework times *how much*
to be invested based on volatility, never *which direction* to bet, so
there's no short strategy here by design (see the high-vol strategy's
explicit "not short" note below).

`resolve_strategy()` is the single entry point the signal generator will
call: given an HMM state and the fitted model, it ranks that state by
volatility, buckets it into low/mid/high, and returns a fully resolved
decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

LOW_TIER_MAX_POSITION = 1 / 3
HIGH_TIER_MIN_POSITION = 2 / 3


@dataclass(frozen=True)
class StopLossResult:
    stop_price: float
    atr_stop: float
    trend_stop: float
    binding: str  # "atr" or "trend" — which candidate is actually in force


@dataclass(frozen=True)
class AtrEmaStopRule:
    """stop = max(price - atr_multiplier * ATR, EMA - ema_atr_multiplier * ATR)

    Used only by the low-volatility tier: the higher (tighter, closer to
    price) of a pure volatility stop and an EMA-anchored trend stop binds,
    since that's whichever level price would reach first on the way down.
    """
    atr_multiplier: float
    ema_window: int
    ema_atr_multiplier: float

    def compute(self, price: float, atr: float, ema: float) -> StopLossResult:
        _validate(price, atr)
        atr_stop = price - self.atr_multiplier * atr
        trend_stop = ema - self.ema_atr_multiplier * atr
        stop_price = max(atr_stop, trend_stop)
        binding = "atr" if atr_stop >= trend_stop else "trend"
        return StopLossResult(stop_price=stop_price, atr_stop=atr_stop, trend_stop=trend_stop, binding=binding)


@dataclass(frozen=True)
class EmaAtrOffsetStopRule:
    """stop = EMA - ema_atr_multiplier * ATR — a pure trend-anchored stop,
    no ATR-from-price alternative. Used by the mid- and high-volatility
    tiers (with different multipliers).
    """
    ema_window: int
    ema_atr_multiplier: float

    def compute(self, price: float, atr: float, ema: float) -> float:
        _validate(price, atr)
        return ema - self.ema_atr_multiplier * atr


def _validate(price: float, atr: float) -> None:
    if price <= 0 or atr < 0:
        raise ValueError(f"invalid inputs: price={price}, atr={atr}")


@dataclass(frozen=True)
class StrategyDecision:
    strategy_name: str
    tier: str  # "low", "mid", "high"
    direction: str
    allocation_pct: float
    leverage: float
    stop_price: float


# --- Stop rules, one per tier ------------------------------------------------

LOW_VOL_STOP_RULE = AtrEmaStopRule(atr_multiplier=3.0, ema_window=50, ema_atr_multiplier=0.5)
MID_VOL_STOP_RULE = EmaAtrOffsetStopRule(ema_window=50, ema_atr_multiplier=0.5)
HIGH_VOL_STOP_RULE = EmaAtrOffsetStopRule(ema_window=50, ema_atr_multiplier=1.0)  # wider: volatile conditions


# --- Decision functions, one per tier ---------------------------------------

def low_volatility_decision(price: float, atr: float, ema_50: float) -> StrategyDecision:
    """Lowest third by expected volatility. Calm conditions -> fully
    invested with modest leverage."""
    stop = LOW_VOL_STOP_RULE.compute(price, atr, ema_50)
    return StrategyDecision("low_volatility_bull", "low", "LONG", 0.95, 1.25, stop.stop_price)


def mid_volatility_decision(price: float, atr: float, ema_50: float) -> StrategyDecision:
    """Middle third by expected volatility. Direction of the price/EMA
    relationship decides whether the intermediate trend looks intact
    (stay invested) or broken (reduce)."""
    if price > ema_50:
        allocation_pct, leverage = 0.95, 1.0  # trend intact, stay invested
    else:
        allocation_pct, leverage = 0.60, 1.0  # trend broken, reduce
    stop_price = MID_VOL_STOP_RULE.compute(price, atr, ema_50)
    return StrategyDecision("mid_volatility_cautious", "mid", "LONG", allocation_pct, leverage, stop_price)


def high_volatility_decision(price: float, atr: float, ema_50: float) -> StrategyDecision:
    """Top third by expected volatility. Reduced size, no leverage, wider
    stop. Stays LONG rather than going flat or short — 60% invested is
    deliberately enough exposure to catch a sharp post-selloff rebound,
    which a directional short bet would miss (and this framework doesn't
    make directional bets in the first place — see the module docstring)."""
    stop_price = HIGH_VOL_STOP_RULE.compute(price, atr, ema_50)
    return StrategyDecision("high_volatility_defensive", "high", "LONG", 0.60, 1.0, stop_price)


TIER_DECISION_FUNCTIONS = {
    "low": low_volatility_decision,
    "mid": mid_volatility_decision,
    "high": high_volatility_decision,
}


# --- Rank -> tier mapping, works for any state count 3-7 --------------------

def volatility_tier(rank: int, n_states: int) -> str:
    """rank: 0 = lowest-volatility state, n_states - 1 = highest.

    position <= 1/3 -> "low", position >= 2/3 -> "high", else "mid".
    Thresholds are exact thirds (not two-decimal 0.33/0.67) so behavior is
    consistent across every state count instead of drifting from
    floating-point rounding at the boundary.
    """
    if n_states < 2:
        raise ValueError(f"need at least 2 states to compute a position, got {n_states}")
    if not (0 <= rank < n_states):
        raise ValueError(f"rank {rank} out of range for n_states={n_states}")

    position = rank / (n_states - 1)
    if position <= LOW_TIER_MAX_POSITION:
        return "low"
    if position >= HIGH_TIER_MIN_POSITION:
        return "high"
    return "mid"


def rank_states_by_volatility(model, feature_columns, primary_vol_feature: str = "realized_vol_20") -> Dict[int, int]:
    """Rank each HMM state 0..n_states-1 by its fitted mean value of
    `primary_vol_feature`, ascending (rank 0 = calmest state).

    Uses the model's own fitted parameters (`means_`) rather than
    re-decoding the training sequence, since the volatility features were
    literally what the model was fit on — this is the model's own judgment
    of relative volatility per state, not a second derived computation.
    """
    feature_columns = list(feature_columns)
    if primary_vol_feature not in feature_columns:
        raise ValueError(f"'{primary_vol_feature}' not in feature_columns {feature_columns}")

    idx = feature_columns.index(primary_vol_feature)
    means = model.means_[:, idx]
    order = np.argsort(means)  # ascending: order[0] is the lowest-vol state
    return {int(state): int(rank) for rank, state in enumerate(order)}


def resolve_strategy(
    state: int,
    rank_by_state: Dict[int, int],
    n_states: int,
    price: float,
    atr: float,
    ema_50: float,
) -> StrategyDecision:
    """Single entry point: HMM state in, fully resolved allocation decision out."""
    if state not in rank_by_state:
        raise ValueError(f"state {state} not found in rank_by_state {rank_by_state}")

    rank = rank_by_state[state]
    tier = volatility_tier(rank, n_states)
    return TIER_DECISION_FUNCTIONS[tier](price, atr, ema_50)
