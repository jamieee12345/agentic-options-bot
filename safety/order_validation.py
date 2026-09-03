"""Pre-trade order validation: buying power, tradability, spread, duplicates.

Four independent checks, run in this order (cheapest/most-decisive first):
buying power -> tradability -> spread -> duplicate order. Each returns
(ok, reason) rather than raising, so the caller can log *why* an order was
skipped without wrapping every call in try/except -- consistent with how
the rest of the safety layer (circuit_breakers, position_sizing) prefers
explicit result objects over exceptions for expected "this trade doesn't
happen" outcomes.

None of this can run against a live account in this sandbox (no network, no
Robinhood session) -- it's built and reasoned through against hand-picked
cases the same way circuit_breakers.py and position_sizing.py originally
were, but hasn't been exercised against real broker responses. Run it
against a live paper/small-size order before trusting it with real size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

DEFAULT_MAX_SPREAD_PCT = 0.01          # settings.yaml: risk.max_bid_ask_spread_pct
DEFAULT_DUPLICATE_COOLDOWN_MINUTES = 5  # don't resubmit for a symbol with an order still in flight


@dataclass(frozen=True)
class OrderCheckResult:
    ok: bool
    reason: str  # "ok" on success, otherwise the specific check that failed and why


def check_buying_power(order_notional: float, buying_power: float, side: str) -> OrderCheckResult:
    """Only buys consume buying power. A sell of a position you already hold
    doesn't need it (and blocking sells on a buying-power technicality would
    fight the entire point of a stop/derisking trade)."""
    if side == "sell":
        return OrderCheckResult(True, "ok")
    if order_notional <= 0:
        return OrderCheckResult(False, f"non-positive order notional ({order_notional})")
    if order_notional > buying_power:
        return OrderCheckResult(
            False, f"order notional ${order_notional:,.2f} exceeds buying power ${buying_power:,.2f}"
        )
    return OrderCheckResult(True, "ok")


def check_tradability(symbol: str, tradable: bool, reason_if_not: Optional[str] = None) -> OrderCheckResult:
    """`tradable` comes from the broker (e.g. instrument halted, restricted,
    or not marginable when margin was required). This function doesn't call
    the broker itself -- the caller is expected to have already asked it."""
    if not tradable:
        detail = reason_if_not or "broker reports not tradable"
        return OrderCheckResult(False, f"{symbol} not tradable: {detail}")
    return OrderCheckResult(True, "ok")


def check_spread(
    symbol: str,
    bid: Optional[float],
    ask: Optional[float],
    max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
) -> OrderCheckResult:
    """Rejects orders when the quoted spread is wide relative to the mid
    price -- a proxy for "thin/illiquid right now," where a market order
    could fill much worse than the last trade price. Skipped (passes
    trivially) when bid/ask aren't available, since RobinhoodQuoteFetcher
    doesn't currently populate them -- flagged rather than silently assumed
    tight."""
    if bid is None or ask is None:
        return OrderCheckResult(True, "ok (bid/ask unavailable -- spread check skipped)")
    if bid <= 0 or ask <= 0 or ask < bid:
        return OrderCheckResult(False, f"{symbol}: invalid quote (bid={bid}, ask={ask})")

    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid
    if spread_pct > max_spread_pct:
        return OrderCheckResult(
            False, f"{symbol}: spread {spread_pct:.2%} exceeds max {max_spread_pct:.2%} (bid={bid}, ask={ask})"
        )
    return OrderCheckResult(True, "ok")


@dataclass
class DuplicateOrderGuard:
    """Tracks symbols with an order submitted-but-not-yet-resolved, so a
    retry or a repeated signal on the next bar doesn't stack a second order
    on top of one still working at the broker.

    In-memory only, scoped to one process's lifetime -- a restart relies on
    `Broker.get_open_orders()` (queried separately, see
    orchestration/order_execution.py) to recover any orders still open at
    the broker after a crash, since this guard's own memory doesn't survive
    a restart.
    """
    _pending: Dict[str, datetime] = field(default_factory=dict)
    cooldown: timedelta = field(default_factory=lambda: timedelta(minutes=DEFAULT_DUPLICATE_COOLDOWN_MINUTES))

    def check(self, symbol: str, now: datetime, open_order_symbols: Optional[List[str]] = None) -> OrderCheckResult:
        if open_order_symbols and symbol in open_order_symbols:
            return OrderCheckResult(False, f"{symbol}: already has an open order at the broker")
        submitted_at = self._pending.get(symbol)
        if submitted_at is not None and now - submitted_at < self.cooldown:
            return OrderCheckResult(
                False, f"{symbol}: order submitted {(now - submitted_at).seconds}s ago, still within cooldown"
            )
        return OrderCheckResult(True, "ok")

    def record_submitted(self, symbol: str, now: datetime) -> None:
        self._pending[symbol] = now

    def clear(self, symbol: str) -> None:
        self._pending.pop(symbol, None)


def run_order_checks(
    symbol: str,
    side: str,
    order_notional: float,
    buying_power: float,
    tradable: bool,
    bid: Optional[float],
    ask: Optional[float],
    now: datetime,
    duplicate_guard: DuplicateOrderGuard,
    open_order_symbols: Optional[List[str]] = None,
    max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
    tradability_reason: Optional[str] = None,
) -> OrderCheckResult:
    """Runs all four checks in order, short-circuiting on the first failure."""
    for result in (
        check_buying_power(order_notional, buying_power, side),
        check_tradability(symbol, tradable, tradability_reason),
        check_spread(symbol, bid, ask, max_spread_pct),
        duplicate_guard.check(symbol, now, open_order_symbols),
    ):
        if not result.ok:
            return result
    return OrderCheckResult(True, "ok")
