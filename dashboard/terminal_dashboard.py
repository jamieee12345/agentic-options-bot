"""Streamlined always-reviewable dashboard.

Stdlib-only text rendering (no `rich` dependency required, though `rich` in
requirements.txt remains a fine visual upgrade later) — kept this way so it
can actually be rendered and tested here.

Builder functions at the bottom adapt the real production dataclasses
(RegimeState, PortfolioState, Signal) into this module's snapshot types, so
wiring this into the live loop later is a thin adapter, not a rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence

DEFAULT_WIDTH = 78
DASHBOARD_FLICKER_WINDOW = 20  # deliberately longer than is_flickering()'s 5-bar trigger window -- this is a
# display/context metric, not a change to the actual uncertainty-mode trigger logic.


# ---------------------------------------------------------------------------
# Snapshot types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeSnapshot:
    label: str
    probability: float
    stability_bars: int   # bars since the last regime change (lower bound if it hits the buffer length)
    flicker_count: int    # transitions within flicker_window
    flicker_window: int


@dataclass(frozen=True)
class RiskSnapshot:
    daily_dd_pct: float  # same number CircuitBreakerEngine.evaluate() acts on -- read from BreakerStatus, not recomputed


@dataclass(frozen=True)
class PortfolioSnapshot:
    equity: float
    daily_pnl_dollars: float
    daily_pnl_pct: float
    allocation_pct: float


@dataclass(frozen=True)
class PositionRow:
    ticker: str
    direction: str  # always "Long" given the system's long-only design
    price: float
    pct_of_portfolio: float
    stop_price: float
    hours_in: float


@dataclass(frozen=True)
class SignalRow:
    timestamp: datetime
    ticker: str
    allocation_before_pct: float
    allocation_after_pct: float
    tier: str


@dataclass(frozen=True)
class DashboardSnapshot:
    regime: RegimeSnapshot
    portfolio: PortfolioSnapshot
    risk: RiskSnapshot
    positions: List[PositionRow]
    recent_signals: List[SignalRow]


# ---------------------------------------------------------------------------
# Regime helper metrics
# ---------------------------------------------------------------------------

def count_transitions(recent_states: Sequence[int], window: int) -> int:
    tail = list(recent_states)[-window:]
    if len(tail) < 2:
        return 0
    return sum(1 for a, b in zip(tail, tail[1:]) if a != b)


def bars_since_last_change(recent_states: Sequence[int]) -> int:
    """Consecutive bars (counting back from the end) matching the most
    recent state. If this equals len(recent_states), the run may actually
    be longer than we can see — it's a lower bound in that case, not a cap.
    """
    if not recent_states:
        return 0
    states = list(recent_states)
    current = states[-1]
    count = 0
    for s in reversed(states):
        if s != current:
            break
        count += 1
    return count


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def _format_pct(x: float, signed: bool = False) -> str:
    if signed:
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.1%}"
    return f"{x:.1%}"


def _section_header(name: str, width: int) -> str:
    prefix = f"{name} "
    return prefix + "-" * max(0, width - len(prefix))


def _render_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_regime_section(r: RegimeSnapshot) -> str:
    return (
        f"{r.label} ({r.probability:.0%}) | "
        f"Stability: {r.stability_bars} bars | "
        f"Flicker: {r.flicker_count}/{r.flicker_window}"
    )


def render_portfolio_section(p: PortfolioSnapshot) -> str:
    return (
        f"Equity: {_format_money(p.equity)} | "
        f"Daily Gain/Loss: {_format_money(p.daily_pnl_dollars)} ({_format_pct(p.daily_pnl_pct, signed=True)}) | "
        f"Allocation: {_format_pct(p.allocation_pct)}"
    )


def render_risk_section(r: RiskSnapshot) -> str:
    return f"Daily DD: {_format_pct(r.daily_dd_pct, signed=True)}"


def render_positions_table(positions: List[PositionRow]) -> str:
    if not positions:
        return "(no open positions)"
    headers = ["Ticker", "Side", "Price", "%", "Stop", "Hours In"]
    rows = [
        [p.ticker, p.direction, _format_money(p.price), _format_pct(p.pct_of_portfolio),
         _format_money(p.stop_price), f"{p.hours_in:.1f}"]
        for p in positions
    ]
    return _render_table(headers, rows)


def render_signals_table(signals: List[SignalRow]) -> str:
    if not signals:
        return "(no recent signals)"
    headers = ["Time", "Ticker", "Rebalance", "Regime"]
    rows = [
        [s.timestamp.strftime("%H:%M:%S"), s.ticker,
         f"{_format_pct(s.allocation_before_pct)} -> {_format_pct(s.allocation_after_pct)}", s.tier]
        for s in signals
    ]
    return _render_table(headers, rows)


def render_dashboard(snapshot: DashboardSnapshot, width: int = DEFAULT_WIDTH) -> str:
    lines = [
        _section_header("REGIME TYPE", width),
        render_regime_section(snapshot.regime),
        _section_header("PORTFOLIO", width),
        render_portfolio_section(snapshot.portfolio),
        _section_header("RISK STATUS", width),
        render_risk_section(snapshot.risk),
        _section_header("POSITIONS", width),
        render_positions_table(snapshot.positions),
        _section_header("RECENT SIGNALS", width),
        render_signals_table(snapshot.recent_signals),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adapters from the real production dataclasses
# ---------------------------------------------------------------------------

def build_regime_snapshot(regime_state, flicker_window: int = DASHBOARD_FLICKER_WINDOW) -> RegimeSnapshot:
    """regime_state: brain.signal_generator.RegimeState. Note its
    `recent_states` buffer must hold at least `flicker_window` entries for
    this to be meaningful -- the uncertainty-trigger logic only needs 5, so
    the live loop's rolling buffer needs to keep the larger of the two.
    """
    return RegimeSnapshot(
        label=regime_state.label,
        probability=regime_state.probability,
        stability_bars=bars_since_last_change(regime_state.recent_states),
        flicker_count=count_transitions(regime_state.recent_states, window=flicker_window),
        flicker_window=flicker_window,
    )


def build_risk_snapshot(breaker_status) -> RiskSnapshot:
    """breaker_status: safety.circuit_breakers.BreakerStatus. Reads
    daily_dd_pct directly off the real circuit breaker's own output rather
    than recomputing it here, so the dashboard can never drift from what
    the breaker is actually acting on.
    """
    return RiskSnapshot(daily_dd_pct=breaker_status.daily_dd_pct)


def build_portfolio_snapshot(portfolio_state, price: float, day_start_equity: float, current_allocation_pct_fn) -> PortfolioSnapshot:
    """portfolio_state: allocation.portfolio.PortfolioState. Takes
    current_allocation_pct_fn as a parameter rather than importing it
    directly, to keep this module decoupled from where that function
    happens to live (allocation.portfolio.current_allocation_pct).
    """
    equity = portfolio_state.equity(price)
    daily_pnl_dollars = equity - day_start_equity
    daily_pnl_pct = daily_pnl_dollars / day_start_equity if day_start_equity else 0.0
    return PortfolioSnapshot(
        equity=equity,
        daily_pnl_dollars=daily_pnl_dollars,
        daily_pnl_pct=daily_pnl_pct,
        allocation_pct=current_allocation_pct_fn(portfolio_state, price),
    )


def build_signal_row(signal, allocation_before_pct: float) -> SignalRow:
    """signal: brain.signal_generator.Signal."""
    tier_display = {
        "low_volatility_bull": "Low Vol",
        "mid_volatility_cautious": "Mid Vol",
        "high_volatility_defensive": "High Vol",
    }.get(signal.regime.strategy_name, signal.regime.strategy_name)
    return SignalRow(
        timestamp=signal.regime.timestamp,
        ticker=signal.symbol,
        allocation_before_pct=allocation_before_pct,
        allocation_after_pct=signal.position_size_pct,
        tier=tier_display,
    )
