"""Turns Signals into validated orders and (optionally) places them.

This is the piece that was explicitly left out of main_loop.run_bar before:
"Trailing-stop updates and order placement are integration points, left out
here since the broker wrapper's order methods aren't wired up yet." It sits
between the Brain's output (brain.signal_generator.Signal, one per symbol)
and the Broker, applying every safety gate built for this in order:

    rebalance threshold -> position sizing -> correlation check
    -> order validation (buying power / tradability / spread / duplicate)
    -> portfolio limits (exposure / concurrent positions / sector)

A sizing note carried over from safety/position_sizing.py, not
allocation/portfolio.py: `compute_position_size` (fixed-fractional risk
sizing, aware of per-position and portfolio-wide caps) is used here rather
than `compute_rebalance`/`PortfolioState` from allocation/portfolio.py.
That module models a single instrument's cash+shares and is what the
backtester uses; it was never extended to a real multi-symbol live book
(ten-plus tickers sharing one equity pool), which is exactly the gap the
README flagged under "Portfolio-level limit enforcement." Revisit if the
allocation layer is ever generalized to multi-symbol.

DRY-RUN BY DEFAULT: `live_trading_enabled` must be explicitly True (see
settings.yaml: broker.live_trading_enabled) before this will call
`broker.place_order`. With it False (the default), every approved order is
logged as "[DRY RUN] would place order" and never reaches the broker. This
defaults off because none of this pipeline has run against a live account
or been exercised with real data in this sandbox -- flip it only after
watching dry-run output agree with your own read of what the bot should be
doing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from brain.signal_generator import Signal, should_rebalance
from safety.correlation import correlation_size_multiplier
from safety.order_validation import DuplicateOrderGuard, run_order_checks
from safety.portfolio_limits import check_portfolio_limits
from safety.position_sizing import MissingStopLossError, compute_position_size

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Quote:
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    tradable: bool = True
    tradability_reason: Optional[str] = None


@dataclass(frozen=True)
class OrderPlan:
    symbol: str
    side: str          # "buy" or "sell"
    quantity: int
    order_type: str = "market"


@dataclass(frozen=True)
class OrderExecutionRecord:
    symbol: str
    plan: Optional[OrderPlan]
    placed: bool         # True only if broker.place_order was actually called (never True in dry-run)
    order_id: Optional[str]
    skipped_reason: Optional[str]  # None if a plan was built and (dry-run or live) accepted


class OrderExecutor:
    def __init__(
        self,
        broker,
        max_exposure: float,
        max_concurrent_positions: int,
        max_sector_exposure: float,
        max_spread_pct: float,
        correlation_lookback_days: int,
        correlation_reduce_threshold: float,
        correlation_reject_threshold: float,
        max_risk_per_trade: float,
        max_single_position: float,
        live_trading_enabled: bool = False,
        duplicate_guard: Optional[DuplicateOrderGuard] = None,
    ) -> None:
        self.broker = broker
        self.max_exposure = max_exposure
        self.max_concurrent_positions = max_concurrent_positions
        self.max_sector_exposure = max_sector_exposure
        self.max_spread_pct = max_spread_pct
        self.correlation_lookback_days = correlation_lookback_days
        self.correlation_reduce_threshold = correlation_reduce_threshold
        self.correlation_reject_threshold = correlation_reject_threshold
        # Passed through explicitly rather than left to safety.position_sizing's
        # module-level defaults, so this always reflects settings.yaml's
        # risk.max_risk_per_trade / risk.max_single_position even if those
        # ever diverge from that module's hardcoded fallback constants.
        self.max_risk_per_trade = max_risk_per_trade
        self.max_single_position = max_single_position
        self.live_trading_enabled = live_trading_enabled
        self.duplicate_guard = duplicate_guard or DuplicateOrderGuard()

    def run(
        self,
        signals: Dict[str, Signal],
        equity: float,
        positions: Dict[str, float],
        buying_power: float,
        quotes: Dict[str, Quote],
        held_returns_by_symbol: Dict[str, pd.Series],
        open_order_symbols: List[str],
        now: datetime,
        size_multiplier: float = 1.0,
        overnight: bool = False,
    ) -> List[OrderExecutionRecord]:
        """One pass over every symbol with a signal this bar. `size_multiplier`
        is the circuit breaker's current value (0.5 during a reduce, 1.0
        normally) -- callers should not call this at all when the breaker
        reports `trading_halted=True`.

        `held_returns_by_symbol` despite the name needs a daily-returns
        Series for every symbol that might be traded this bar, not only
        symbols currently held -- the correlation check looks up the
        candidate's own returns from this same dict (see
        `correlation_size_multiplier`, which then compares it against every
        *other* entry). A symbol missing from this dict just skips the
        correlation check for that bar (logged), it isn't treated as a
        rejection.
        """
        records: List[OrderExecutionRecord] = []
        positions_value_by_symbol = {
            symbol: shares * quotes[symbol].price
            for symbol, shares in positions.items()
            if symbol in quotes and shares != 0
        }

        for symbol, signal in signals.items():
            record = self._evaluate_one(
                symbol, signal, equity, positions, buying_power, quotes,
                held_returns_by_symbol, open_order_symbols, positions_value_by_symbol,
                now, size_multiplier, overnight,
            )
            records.append(record)
        return records

    def _evaluate_one(
        self, symbol, signal, equity, positions, buying_power, quotes,
        held_returns_by_symbol, open_order_symbols, positions_value_by_symbol,
        now, size_multiplier, overnight,
    ) -> OrderExecutionRecord:
        quote = quotes.get(symbol)
        if quote is None:
            return OrderExecutionRecord(symbol, None, False, None, "no quote available")

        current_shares = int(positions.get(symbol, 0))
        current_value = current_shares * quote.price
        current_pct = current_value / equity if equity > 0 else 0.0
        target_pct = 0.0 if signal.direction == "FLAT" else signal.position_size_pct * size_multiplier

        if not should_rebalance(current_pct, target_pct):
            return OrderExecutionRecord(symbol, None, False, None, "within rebalance threshold")

        try:
            sizing = compute_position_size(
                equity=equity, entry_price=signal.entry_price, stop_loss=signal.stop_loss,
                regime_allocation_pct=target_pct, max_risk_per_trade=self.max_risk_per_trade,
                max_single_position=self.max_single_position, overnight=overnight,
            )
        except MissingStopLossError as exc:
            return OrderExecutionRecord(symbol, None, False, None, str(exc))

        delta = sizing.shares - current_shares
        if delta == 0:
            return OrderExecutionRecord(symbol, None, False, None, f"sizing produced no change (binding_cap={sizing.binding_cap})")

        side = "buy" if delta > 0 else "sell"
        quantity = abs(delta)

        if side == "buy":
            candidate_returns = held_returns_by_symbol.get(symbol)
            if candidate_returns is not None:
                corr_result = correlation_size_multiplier(
                    symbol, candidate_returns, held_returns_by_symbol,
                    window=self.correlation_lookback_days,
                    reduce_threshold=self.correlation_reduce_threshold,
                    reject_threshold=self.correlation_reject_threshold,
                )
                if corr_result.multiplier == 0.0:
                    return OrderExecutionRecord(symbol, None, False, None, corr_result.reason)
                quantity = int(quantity * corr_result.multiplier)
                if quantity == 0:
                    return OrderExecutionRecord(symbol, None, False, None, f"correlation-adjusted size rounded to zero ({corr_result.reason})")
            else:
                logger.warning("%s: no return history available for correlation check -- proceeding unadjusted.", symbol)

        order_notional = quantity * quote.price
        check = run_order_checks(
            symbol=symbol, side=side, order_notional=order_notional, buying_power=buying_power,
            tradable=quote.tradable, bid=quote.bid, ask=quote.ask, now=now,
            duplicate_guard=self.duplicate_guard, open_order_symbols=open_order_symbols,
            max_spread_pct=self.max_spread_pct, tradability_reason=quote.tradability_reason,
        )
        if not check.ok:
            return OrderExecutionRecord(symbol, None, False, None, check.reason)

        if side == "buy":
            limits_check = check_portfolio_limits(
                symbol=symbol, order_notional=order_notional, equity=equity,
                current_positions_value_by_symbol=positions_value_by_symbol,
                max_exposure=self.max_exposure, max_concurrent_positions=self.max_concurrent_positions,
                max_sector_exposure=self.max_sector_exposure,
            )
            if not limits_check.ok:
                return OrderExecutionRecord(symbol, None, False, None, limits_check.reason)

        plan = OrderPlan(symbol=symbol, side=side, quantity=quantity, order_type="market")

        if not self.live_trading_enabled:
            logger.info("[DRY RUN] would place order: %s %d %s (~$%.2f)", side, quantity, symbol, order_notional)
            return OrderExecutionRecord(symbol, plan, False, None, None)

        order_id = self.broker.place_order(symbol, side, quantity, order_type="market")
        self.duplicate_guard.record_submitted(symbol, now)
        return OrderExecutionRecord(symbol, plan, True, order_id, None)
