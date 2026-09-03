"""Position-level risk sizing.

Every position must have a stop loss: compute_position_size() raises if one
isn't provided, rather than silently sizing a stop-less trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

MAX_RISK_PER_TRADE = 0.01     # settings.yaml: risk.max_risk_per_trade
MAX_SINGLE_POSITION = 0.15    # settings.yaml: risk.max_single_position
MIN_POSITION_USD = 100.0
GAP_MULTIPLIER = 3.0
GAP_RISK_PCT = 0.02            # deliberately 2x the normal 1% per-trade risk — per spec, not a typo


class MissingStopLossError(ValueError):
    """Raised when a position is requested without a stop loss."""


@dataclass(frozen=True)
class PositionSizeResult:
    shares: int
    position_value: float
    risk_based_shares: int
    regime_cap_shares: int
    portfolio_cap_shares: int
    binding_cap: str        # whichever cap actually determined the final size
    below_minimum: bool     # True if the result was zeroed out for being under MIN_POSITION_USD
    gap_adjusted: bool      # True if overnight gap risk reduced the size further


def compute_position_size(
    equity: float,
    entry_price: float,
    stop_loss: Optional[float],
    regime_allocation_pct: float,
    max_risk_per_trade: float = MAX_RISK_PER_TRADE,
    max_single_position: float = MAX_SINGLE_POSITION,
    min_position_usd: float = MIN_POSITION_USD,
    overnight: bool = False,
) -> PositionSizeResult:
    """size = (equity * max_risk_per_trade) / |entry - stop|, then capped by
    the regime's own allocation ceiling, then by the hard portfolio-wide
    max_single_position -- whichever of the three is smallest wins.

    If `overnight`, an additional cap applies: assume the position could gap
    through the stop by GAP_MULTIPLIER (3x) the normal stop distance
    overnight, and size so that worst case is capped at GAP_RISK_PCT (2%,
    not 1%) of equity.
    """
    if stop_loss is None:
        raise MissingStopLossError("Every position must have a stop loss — refusing to size an order without one.")
    if entry_price <= 0:
        raise ValueError(f"invalid entry_price: {entry_price}")

    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        raise ValueError(f"stop_loss ({stop_loss}) must differ from entry_price ({entry_price})")

    risk_based_shares = int((equity * max_risk_per_trade) / stop_distance)
    regime_cap_shares = int((equity * regime_allocation_pct) / entry_price)
    portfolio_cap_shares = int((equity * max_single_position) / entry_price)

    caps = {
        "risk_based": risk_based_shares,
        "regime_cap": regime_cap_shares,
        "portfolio_cap": portfolio_cap_shares,
    }
    shares = min(caps.values())
    binding_cap = min(caps, key=caps.get)

    gap_adjusted = False
    if overnight:
        gap_risk_shares = int((equity * GAP_RISK_PCT) / (GAP_MULTIPLIER * stop_distance))
        if gap_risk_shares < shares:
            shares = gap_risk_shares
            binding_cap = "gap_risk"
            gap_adjusted = True

    position_value = shares * entry_price
    below_minimum = 0 < position_value < min_position_usd
    if below_minimum:
        shares = 0
        position_value = 0.0

    return PositionSizeResult(
        shares=shares,
        position_value=position_value,
        risk_based_shares=risk_based_shares,
        regime_cap_shares=regime_cap_shares,
        portfolio_cap_shares=portfolio_cap_shares,
        binding_cap=binding_cap,
        below_minimum=below_minimum,
        gap_adjusted=gap_adjusted,
    )
