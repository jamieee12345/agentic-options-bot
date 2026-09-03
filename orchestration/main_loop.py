"""Main loop orchestration: startup, per-bar loop, shutdown.

Defines the CONTROL FLOW and error-handling policy using dependency
injection (Protocol interfaces for Broker / RegimeProvider / DataFeed /
Dashboard), so it's fully testable with fakes even though none of the real
integrations can run here: Broker needs a live Robinhood session, HMM
needs hmmlearn, DataFeed needs a live market data connection, and the
Dashboard component doesn't exist yet. Wiring in the real implementations
is the next integration step once each one is ready.
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from orchestration.market_hours import is_market_open, is_model_stale
from orchestration.retry import retry_with_backoff
from orchestration.session_state import SessionState, load_session_state, save_session_state

logger = logging.getLogger(__name__)


class BrokerError(Exception):
    """Raised by a Broker implementation on an unrecoverable Robinhood API failure."""


class HmmError(Exception):
    """Raised by a RegimeProvider when regime detection/filtering fails."""


class DataFeedError(Exception):
    """Raised by a DataFeed when fresh market data can't be fetched."""


class Broker(Protocol):
    def connect(self) -> None: ...
    def verify_account(self) -> bool: ...
    def get_portfolio_equity(self) -> float: ...
    def get_positions(self) -> Dict[str, float]: ...
    def get_buying_power(self) -> float: ...
    def get_open_orders(self) -> List[dict]: ...
    def place_order(
        self, symbol: str, side: str, quantity: int,
        order_type: str = "market", limit_price: Optional[float] = None,
    ) -> str: ...
    def cancel_order(self, order_id: str) -> None: ...
    def get_option_positions(self) -> List[dict]: ...
    def place_option_order(
        self, symbol: str, option_type: str, expiration_date: str, strike_price: Optional[float],
        side: str, position_effect: str, quantity: int, limit_price: Optional[float],
    ) -> str: ...
    def cancel_option_order(self, order_id: str) -> None: ...


class RegimeProvider(Protocol):
    def load_or_train(self, force: bool) -> Tuple[object, datetime]: ...
    def current_regime(self, model: object, bars: dict) -> Tuple[int, float, str]: ...  # (state, confidence, label)


class DataFeed(Protocol):
    def get_latest_bars(self, symbols: Sequence[str]) -> dict: ...


class Dashboard(Protocol):
    def refresh(self, snapshot: dict) -> None: ...
    def alert(self, message: str, severity: str = "error") -> None: ...


@dataclass
class LoopContext:
    """Threaded through startup -> loop -> shutdown instead of scattering
    state across globals.
    """
    model: object = None
    model_trained_at: Optional[datetime] = None
    current_regime_label: Optional[str] = None
    current_regime_state: Optional[int] = None
    current_regime_confidence: float = 0.0
    recent_regime_states: List[int] = field(default_factory=list)  # rolling history, for is_flickering()
    signals_paused: bool = False  # True after a data-feed drop, until data recovers
    last_good_bars: Optional[dict] = None


def startup(
    broker: Broker,
    regime_provider: RegimeProvider,
    now: datetime,
    state_path=None,
) -> LoopContext:
    """Connect once (covers both "connect to Robinhood, verify account" and
    the later "log onto Robinhood" from the spec -- treated as the same
    step, done first, since nearly everything else needs an active
    session). Then: market-hours check (logged, not enforced here -- waiting
    for open is the caller's responsibility), load-or-retrain the model,
    sync the portfolio, and recover prior session state if present.
    """
    retry_with_backoff(broker.connect)
    if not retry_with_backoff(broker.verify_account):
        raise BrokerError("Account verification failed at startup")

    if not is_market_open(now):
        logger.info("Market is closed at startup (%s).", now)

    prior_state = load_session_state(state_path) if state_path else None
    force_retrain = prior_state is None or is_model_stale(
        prior_state.model_trained_at if prior_state else None, now
    )
    model, trained_at = regime_provider.load_or_train(force=force_retrain)

    equity = retry_with_backoff(broker.get_portfolio_equity)
    positions = retry_with_backoff(broker.get_positions)
    logger.info(
        "Startup complete: equity=%.2f, %d open position(s), model trained_at=%s, retrained=%s",
        equity, len(positions), trained_at, force_retrain,
    )

    ctx = LoopContext(model=model, model_trained_at=trained_at)
    if prior_state is not None:
        ctx.current_regime_label = prior_state.current_regime_label
        ctx.current_regime_state = prior_state.current_regime_state
        logger.info("Recovered prior session state (last updated %s).", prior_state.last_updated)

    return ctx


def run_bar(
    ctx: LoopContext,
    broker: Broker,
    regime_provider: RegimeProvider,
    data_feed: DataFeed,
    dashboard: Dashboard,
    circuit_breaker,  # safety.circuit_breakers.CircuitBreakerEngine
    symbols: Sequence[str],
    now: datetime,
    orchestrator=None,          # Optional[brain.signal_generator.StrategyOrchestrator]
    order_executor=None,        # Optional[orchestration.order_execution.OrderExecutor]
    quotes: Optional[dict] = None,               # symbol -> orchestration.order_execution.Quote, required if order_executor is set
    held_returns_by_symbol: Optional[dict] = None,  # symbol -> pd.Series of daily returns, for the correlation check
    options_executor=None,      # Optional[orchestration.options_execution.OptionsOrderExecutor]
    options_enabled: bool = False,  # settings.yaml: options.enabled -- separate gate from live_trading_enabled
):
    """One loop iteration. Error policy:
      - Robinhood calls go through retry_with_backoff (3 retries, backoff);
        a still-failing call propagates as an exception to the caller.
      - HMM error: hold the current regime rather than resetting or crashing.
      - Data feed drop: pause new signal generation; circuit breakers and
        existing stop orders are unaffected (they run off the broker's own
        reported equity/positions, not off fresh market data).

    `orchestrator`/`order_executor` are optional so every existing caller
    (and every existing test) that doesn't pass them keeps working exactly
    as before -- this only places orders when both are explicitly wired in.

    KNOWN SIMPLIFICATION: `regime_provider.current_regime()` returns one
    regime call for the whole bar (originally meant for dashboard/circuit-
    breaker display), not one per symbol. Signal generation below reuses
    that single regime state for every symbol in `symbols`, since per-symbol
    regime tracking was never built here (it needs its own rolling
    `recent_states` history per symbol, which `RegimeProvider` doesn't
    currently expose). Fine as a stand-in as long as the watchlist is voting
    on one shared market regime; revisit if per-symbol regimes matter later.
    """
    try:
        bars = data_feed.get_latest_bars(symbols)
        ctx.last_good_bars = bars
        ctx.signals_paused = False
    except DataFeedError as exc:
        logger.warning("Data feed drop: %s -- pausing new signals, stops remain active.", exc)
        ctx.signals_paused = True
        bars = ctx.last_good_bars

    if not ctx.signals_paused and bars is not None:
        try:
            state, confidence, label = regime_provider.current_regime(ctx.model, bars)
            ctx.current_regime_state = state
            ctx.current_regime_confidence = confidence
            ctx.current_regime_label = label
            ctx.recent_regime_states.append(state)
            ctx.recent_regime_states = ctx.recent_regime_states[-20:]  # matches dashboard's flicker window
        except HmmError as exc:
            logger.warning("HMM error: %s -- holding current regime (%s).", exc, ctx.current_regime_label)

    equity = retry_with_backoff(broker.get_portfolio_equity)
    positions = retry_with_backoff(broker.get_positions)
    status = circuit_breaker.evaluate(now, equity, list(positions.keys()), ctx.current_regime_label)

    run_options = options_enabled and options_executor is not None
    run_equity = orchestrator is not None and order_executor is not None
    order_records = []

    need_shared_data = (run_equity or run_options) and not status.trading_halted and not ctx.signals_paused and bars is not None
    buying_power = None
    open_order_symbols: List[str] = []
    if need_shared_data:
        buying_power = retry_with_backoff(broker.get_buying_power)
        open_orders = retry_with_backoff(broker.get_open_orders)
        open_order_symbols = [o.get("symbol") for o in open_orders if o.get("symbol")]

    if run_equity and not status.trading_halted:
        if ctx.signals_paused or bars is None or ctx.current_regime_state is None:
            logger.info("Skipping equity signal generation/order placement this bar: no fresh regime state.")
        else:
            from brain.signal_generator import RegimeState  # local import avoids a hard brain->main_loop dependency for callers that never trade

            regime_state = RegimeState(
                state=ctx.current_regime_state, probability=ctx.current_regime_confidence,
                label=ctx.current_regime_label, recent_states=ctx.recent_regime_states, timestamp=now,
            )
            regime_states = {symbol: regime_state for symbol in symbols}
            signals = orchestrator.generate_signals(bars, regime_states)

            order_records = order_executor.run(
                signals=signals, equity=equity, positions=positions, buying_power=buying_power,
                quotes=quotes or {}, held_returns_by_symbol=held_returns_by_symbol or {},
                open_order_symbols=open_order_symbols, now=now, size_multiplier=status.size_multiplier,
            )
            for record in order_records:
                if record.plan is not None:
                    logger.info(
                        "%s: %s %d %s -- %s", record.symbol, record.plan.side, record.plan.quantity,
                        record.plan.order_type, "placed" if record.placed else "dry-run/not placed",
                    )
                elif record.skipped_reason:
                    logger.debug("%s: no order (%s)", record.symbol, record.skipped_reason)
    elif status.trading_halted and run_equity:
        logger.warning("Circuit breaker halted -- skipping equity signal generation and order placement this bar.")

    if run_options and not status.trading_halted:
        # Unlike the equity path above, this doesn't need `orchestrator` or a
        # trained regime model at all -- brain/options_strategy.py runs
        # directly off raw OHLCV bars (Fair Value Gap + volume momentum),
        # so it can trade before a real RegimeProvider ever exists.
        if ctx.signals_paused or bars is None:
            logger.info("Skipping options signal generation/order placement this bar: no fresh bars.")
        else:
            from orchestration.options_execution import build_open_positions_from_broker

            raw_option_positions = retry_with_backoff(broker.get_option_positions)
            open_option_positions = build_open_positions_from_broker(raw_option_positions)
            options_records = options_executor.run(
                bars=bars, equity=equity, open_positions=open_option_positions,
                buying_power=buying_power, open_order_symbols=open_order_symbols, now=now,
            )
            for record in options_records:
                if record.action is not None:
                    logger.info(
                        "%s: %s %s%s -- %s", record.symbol, record.action, record.option_type,
                        f" x{record.contracts}" if record.contracts else "",
                        "placed" if record.placed else "dry-run/not placed",
                    )
                elif record.skipped_reason:
                    logger.debug("%s: no options action (%s)", record.symbol, record.skipped_reason)
            order_records = order_records + options_records
    elif status.trading_halted and run_options:
        logger.warning("Circuit breaker halted -- skipping options signal generation and order placement this bar.")

    dashboard.refresh({
        "timestamp": now,
        "equity": equity,
        "regime": ctx.current_regime_label,
        "breaker_status": status,
        "signals_paused": ctx.signals_paused,
        "order_records": order_records,
    })

    return ctx, status


def _persist_state(ctx: LoopContext, broker: Broker, status, now: datetime, state_path) -> None:
    if state_path is None:
        return
    try:
        equity = broker.get_portfolio_equity()
        positions = broker.get_positions()
    except Exception:
        equity, positions = 0.0, {}
    state = SessionState(
        last_updated=now,
        current_regime_label=ctx.current_regime_label,
        current_regime_state=ctx.current_regime_state,
        model_trained_at=ctx.model_trained_at,
        open_positions=positions,
        equity=equity,
        circuit_breaker_size_multiplier=status.size_multiplier if status else 1.0,
        trading_halted=status.trading_halted if status else False,
    )
    save_session_state(state, state_path)


def run_forever(
    ctx: LoopContext,
    broker: Broker,
    regime_provider: RegimeProvider,
    data_feed: DataFeed,
    dashboard: Dashboard,
    circuit_breaker,
    symbols: Sequence[str],
    state_path=None,
    now_fn=datetime.now,
    sleep_fn=None,
    max_iterations: Optional[int] = None,  # for tests; real usage leaves this None
    orchestrator=None,
    order_executor=None,
    quote_provider=None,               # Optional[Callable[[], Dict[str, order_execution.Quote]]], called fresh each bar
    returns_provider=None,             # Optional[Callable[[], Dict[str, pd.Series]]], called fresh each bar
    options_executor=None,
    options_enabled: bool = False,
) -> LoopContext:
    """Unhandled-exception policy: log the full traceback, persist whatever
    state is available, alert via the dashboard, then re-raise. Deliberately
    does NOT swallow-and-continue after something truly unexpected -- that's
    a decision for whatever's supervising this process (e.g. a process
    manager that restarts it), not something to paper over silently here.
    """
    iteration = 0
    last_status = None
    while max_iterations is None or iteration < max_iterations:
        now = now_fn()
        try:
            quotes = quote_provider() if quote_provider is not None else None
            held_returns = returns_provider() if returns_provider is not None else None
            ctx, last_status = run_bar(
                ctx, broker, regime_provider, data_feed, dashboard, circuit_breaker, symbols, now,
                orchestrator=orchestrator, order_executor=order_executor,
                quotes=quotes, held_returns_by_symbol=held_returns,
                options_executor=options_executor, options_enabled=options_enabled,
            )
            _persist_state(ctx, broker, last_status, now, state_path)
        except Exception:
            tb = traceback.format_exc()
            logger.error("Unhandled exception in main loop:\n%s", tb)
            _persist_state(ctx, broker, last_status, now, state_path)
            dashboard.alert(f"Unhandled exception, state saved: {tb.strip().splitlines()[-1]}", severity="critical")
            raise
        iteration += 1
        if sleep_fn is not None:
            sleep_fn()
    return ctx


def shutdown(ctx: LoopContext, broker: Broker, dashboard: Dashboard, now: datetime, state_path=None) -> dict:
    """Per spec: does NOT close positions -- existing stop orders at the
    broker remain live while the bot is offline. Persists final state and
    returns a session summary.
    """
    equity = retry_with_backoff(broker.get_portfolio_equity)
    positions = retry_with_backoff(broker.get_positions)

    summary = {
        "shutdown_at": now.isoformat(),
        "final_equity": equity,
        "open_positions": dict(positions),
        "final_regime": ctx.current_regime_label,
    }
    _persist_state(ctx, broker, None, now, state_path)
    logger.info("Shutdown summary: %s", summary)
    dashboard.alert(
        f"Bot shutting down. Equity=${equity:,.2f}, {len(positions)} position(s) left open (stops active).",
        severity="info",
    )
    return summary
