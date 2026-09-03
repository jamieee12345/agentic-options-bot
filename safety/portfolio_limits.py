"""Portfolio-level limit enforcement: max_exposure, max_concurrent_positions,
max_sector_exposure from settings.yaml's `risk` section.

These were configured from the start (see config/config_loader.py's
RiskConfig) but, per the README's own status log, never actually enforced
against a live set of positions -- this is that enforcement. All three
checks are gates on OPENING OR INCREASING a position; a sell that only
reduces exposure never needs to ask permission to de-risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# Manually maintained sector map for settings.yaml's broker.core_watchlist.
# ASSUMPTION (not specified by the spec): sector exposure can only be
# enforced for symbols this map actually covers. A symbol outside both the
# core watchlist and this map is treated as "unknown" and exempted from the
# sector cap (logged, not silently ignored) rather than guessed at -- an
# incorrect sector guess would be worse than no check at all. Extend this
# map when the dynamic universe screen (settings.yaml:
# broker.universe_screen) starts admitting names beyond the core watchlist.
SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AMD": "Technology",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "SPY": "Diversified (index ETF)",
    "QQQ": "Diversified (index ETF)",
}
UNKNOWN_SECTOR = "unknown"


@dataclass(frozen=True)
class LimitCheckResult:
    ok: bool
    reason: str


def check_max_exposure(
    current_exposure_pct: float, additional_pct: float, max_exposure: float
) -> LimitCheckResult:
    projected = current_exposure_pct + additional_pct
    if projected > max_exposure:
        return LimitCheckResult(
            False,
            f"projected exposure {projected:.1%} would exceed max_exposure {max_exposure:.1%} "
            f"(current {current_exposure_pct:.1%} + this order {additional_pct:.1%})",
        )
    return LimitCheckResult(True, "ok")


def check_max_concurrent_positions(
    current_count: int, is_new_position: bool, max_concurrent_positions: int
) -> LimitCheckResult:
    """`is_new_position`: False when this order only adds to a symbol
    already held -- that doesn't increase the count of distinct positions."""
    if not is_new_position:
        return LimitCheckResult(True, "ok (adding to an existing position, not a new one)")
    if current_count + 1 > max_concurrent_positions:
        return LimitCheckResult(
            False, f"opening a new position would make {current_count + 1}, exceeding max_concurrent_positions ({max_concurrent_positions})"
        )
    return LimitCheckResult(True, "ok")


def check_max_sector_exposure(
    symbol: str,
    current_sector_exposure_pct: float,
    additional_pct: float,
    max_sector_exposure: float,
    sector_map: Optional[Dict[str, str]] = None,
) -> LimitCheckResult:
    sector_map = sector_map if sector_map is not None else SECTOR_MAP
    sector = sector_map.get(symbol, UNKNOWN_SECTOR)
    if sector == UNKNOWN_SECTOR:
        return LimitCheckResult(True, f"ok ({symbol} not in sector map -- sector exposure check skipped, not enforced)")

    projected = current_sector_exposure_pct + additional_pct
    if projected > max_sector_exposure:
        return LimitCheckResult(
            False,
            f"{symbol} ({sector}): projected sector exposure {projected:.1%} would exceed "
            f"max_sector_exposure {max_sector_exposure:.1%}",
        )
    return LimitCheckResult(True, "ok")


def sector_exposure_pct(
    positions_value_by_symbol: Dict[str, float],
    equity: float,
    sector: str,
    sector_map: Optional[Dict[str, str]] = None,
) -> float:
    """Fraction of `equity` currently held in `sector`, per `sector_map`.
    Symbols not in the map don't count toward any sector's exposure (see
    UNKNOWN_SECTOR handling above)."""
    sector_map = sector_map if sector_map is not None else SECTOR_MAP
    if equity <= 0:
        return 0.0
    sector_value = sum(
        value for symbol, value in positions_value_by_symbol.items() if sector_map.get(symbol) == sector
    )
    return sector_value / equity


def check_portfolio_limits(
    symbol: str,
    order_notional: float,
    equity: float,
    current_positions_value_by_symbol: Dict[str, float],
    max_exposure: float,
    max_concurrent_positions: int,
    max_sector_exposure: float,
    sector_map: Optional[Dict[str, str]] = None,
) -> LimitCheckResult:
    """Runs all three portfolio-level checks for a proposed buy, short-
    circuiting on the first failure."""
    if equity <= 0:
        return LimitCheckResult(False, f"non-positive equity ({equity}) -- refusing to size any new exposure")

    additional_pct = order_notional / equity
    current_total_value = sum(current_positions_value_by_symbol.values())
    current_exposure_pct = current_total_value / equity
    is_new_position = symbol not in current_positions_value_by_symbol

    exposure_check = check_max_exposure(current_exposure_pct, additional_pct, max_exposure)
    if not exposure_check.ok:
        return exposure_check

    count_check = check_max_concurrent_positions(len(current_positions_value_by_symbol), is_new_position, max_concurrent_positions)
    if not count_check.ok:
        return count_check

    sector_map = sector_map if sector_map is not None else SECTOR_MAP
    sector = sector_map.get(symbol, UNKNOWN_SECTOR)
    current_sector_pct = sector_exposure_pct(current_positions_value_by_symbol, equity, sector, sector_map)
    sector_check = check_max_sector_exposure(symbol, current_sector_pct, additional_pct, max_sector_exposure, sector_map)
    if not sector_check.ok:
        return sector_check

    return LimitCheckResult(True, "ok")
