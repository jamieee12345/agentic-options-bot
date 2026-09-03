"""P&L-based circuit breakers, independent of the HMM/regime layer.

These fire on realized portfolio equity alone. Regime info is accepted only
for LOGGING (to later check whether the HMM's regime call looked reasonable
at the moment a breaker fired) — it never enters the trigger condition.
That independence is the entire point: if the Brain is producing garbage,
this still has to work on its own.

Design choices flagged for review (not fully specified):
  - "Daily/weekly DD" is measured from that period's START-of-period equity
    (a simple loss-from-open), not from an intraday running peak. A peak-
    based definition would be stricter; flag if that's what was intended.
  - Once a reduce or halt condition triggers, it's STICKY for the rest of
    that day/week ("reduce all sizes 50% rest of day") — it does not
    un-trigger if equity partially recovers before the period rolls over.
  - If daily AND weekly reduce are both active simultaneously, the size
    multiplier is 0.5 (whichever is most restrictive), not 0.25 — they
    don't stack multiplicatively.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DAILY_DD_REDUCE = 0.02
DAILY_DD_HALT = 0.03
WEEKLY_DD_REDUCE = 0.05
WEEKLY_DD_HALT = 0.07
PEAK_DD_HALT = 0.10
REDUCE_MULTIPLIER = 0.5

DEFAULT_LOCK_FILE = Path("trading_halted.lock")


@dataclass(frozen=True)
class BreakerEvent:
    timestamp: datetime
    breaker_type: str          # e.g. "daily_dd_halt", "peak_dd_halt"
    actual_drawdown_pct: float
    equity: float
    positions_closed: Sequence[str]
    regime_at_time: Optional[str]  # logging only -- never read by trigger logic

    def to_log_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "breaker_type": self.breaker_type,
            "actual_drawdown_pct": self.actual_drawdown_pct,
            "equity": self.equity,
            "positions_closed": list(self.positions_closed),
            "regime_at_time": self.regime_at_time,
        }


@dataclass(frozen=True)
class BreakerStatus:
    trading_halted: bool         # True if ANY halt (day, week, or all) is active -- no new trades
    close_all_positions: bool    # True only on the bar a halt NEWLY triggers -- liquidate signal, fires once
    size_multiplier: float       # 1.0 normal, 0.5 if a reduce is active, 0.0 if halted
    active_breakers: List[str]
    daily_dd_pct: float          # current daily drawdown, exposed so the dashboard reads the same number the breaker acts on


@dataclass
class _State:
    day_start_equity: float
    week_start_equity: float
    peak_equity: float
    current_day: date
    current_week: Tuple[int, int]
    day_reduce_triggered: bool = False
    day_halt_triggered: bool = False
    week_reduce_triggered: bool = False
    week_halt_triggered: bool = False
    all_halted: bool = False


class CircuitBreakerEngine:
    def __init__(self, initial_equity: float, timestamp: datetime, lock_file_path: Path = DEFAULT_LOCK_FILE):
        self.lock_file_path = Path(lock_file_path)
        self.events: List[BreakerEvent] = []
        self._state = _State(
            day_start_equity=initial_equity,
            week_start_equity=initial_equity,
            peak_equity=initial_equity,
            current_day=timestamp.date(),
            current_week=timestamp.isocalendar()[:2],
        )
        if self.lock_file_path.exists():
            self._state.all_halted = True
            logger.warning(
                "Lock file %s already present at startup -- trading starts HALTED until it's manually removed.",
                self.lock_file_path,
            )

    def _roll_period_boundaries(self, timestamp: datetime, current_equity: float) -> None:
        day = timestamp.date()
        week = timestamp.isocalendar()[:2]
        if day != self._state.current_day:
            self._state.current_day = day
            self._state.day_start_equity = current_equity
            self._state.day_reduce_triggered = False
            self._state.day_halt_triggered = False
        if week != self._state.current_week:
            self._state.current_week = week
            self._state.week_start_equity = current_equity
            self._state.week_reduce_triggered = False
            self._state.week_halt_triggered = False

    def _write_lock_file(self, event: BreakerEvent) -> None:
        self.lock_file_path.write_text(json.dumps(event.to_log_dict(), indent=2))
        logger.critical(
            "PEAK DRAWDOWN HALT: wrote %s. Trading will not resume until this file is manually deleted.",
            self.lock_file_path,
        )

    def evaluate(
        self,
        timestamp: datetime,
        current_equity: float,
        open_positions: Sequence[str] = (),
        regime_label: Optional[str] = None,
    ) -> BreakerStatus:
        was_halted_before = self._is_any_halt_active()

        if self._state.all_halted:
            # day_start_equity may be stale here (period rolling stops once fully halted),
            # but it's still more informative than hardcoding 0.0 -- shows drawdown relative
            # to whenever daily tracking last reset, even if that was a while ago.
            stale_daily_dd = current_equity / self._state.day_start_equity - 1
            return BreakerStatus(True, False, 0.0, ["all_halted"], stale_daily_dd)

        self._roll_period_boundaries(timestamp, current_equity)

        # Peak drawdown: terminal, never resets on its own.
        self._state.peak_equity = max(self._state.peak_equity, current_equity)
        peak_dd = current_equity / self._state.peak_equity - 1
        daily_dd = current_equity / self._state.day_start_equity - 1
        if peak_dd <= -PEAK_DD_HALT:
            self._state.all_halted = True
            event = BreakerEvent(timestamp, "peak_dd_halt", peak_dd, current_equity, list(open_positions), regime_label)
            self.events.append(event)
            logger.critical("Circuit breaker fired: %s", event.to_log_dict())
            self._write_lock_file(event)
            return BreakerStatus(True, True, 0.0, ["peak_dd_halt"], daily_dd_pct=daily_dd)

        active: List[str] = []

        if daily_dd <= -DAILY_DD_HALT and not self._state.day_halt_triggered:
            self._state.day_halt_triggered = True
            self._log_event(timestamp, "daily_dd_halt", daily_dd, current_equity, open_positions, regime_label)
        elif daily_dd <= -DAILY_DD_REDUCE and not self._state.day_reduce_triggered and not self._state.day_halt_triggered:
            self._state.day_reduce_triggered = True
            self._log_event(timestamp, "daily_dd_reduce", daily_dd, current_equity, open_positions, regime_label)

        weekly_dd = current_equity / self._state.week_start_equity - 1
        if weekly_dd <= -WEEKLY_DD_HALT and not self._state.week_halt_triggered:
            self._state.week_halt_triggered = True
            self._log_event(timestamp, "weekly_dd_halt", weekly_dd, current_equity, open_positions, regime_label)
        elif weekly_dd <= -WEEKLY_DD_REDUCE and not self._state.week_reduce_triggered and not self._state.week_halt_triggered:
            self._state.week_reduce_triggered = True
            self._log_event(timestamp, "weekly_dd_reduce", weekly_dd, current_equity, open_positions, regime_label)

        if self._state.day_halt_triggered:
            active.append("daily_dd_halt")
        elif self._state.day_reduce_triggered:
            active.append("daily_dd_reduce")
        if self._state.week_halt_triggered:
            active.append("weekly_dd_halt")
        elif self._state.week_reduce_triggered:
            active.append("weekly_dd_reduce")

        trading_halted = self._state.day_halt_triggered or self._state.week_halt_triggered
        is_reduced = self._state.day_reduce_triggered or self._state.week_reduce_triggered
        size_multiplier = 0.0 if trading_halted else (REDUCE_MULTIPLIER if is_reduced else 1.0)
        close_all = trading_halted and not was_halted_before

        return BreakerStatus(
            trading_halted=trading_halted,
            close_all_positions=close_all,
            size_multiplier=size_multiplier,
            active_breakers=active or ["normal"],
            daily_dd_pct=daily_dd,
        )

    def _log_event(self, timestamp, breaker_type, dd, equity, positions, regime_label) -> None:
        event = BreakerEvent(timestamp, breaker_type, dd, equity, list(positions), regime_label)
        self.events.append(event)
        logger.warning("Circuit breaker fired: %s", event.to_log_dict())

    def _is_any_halt_active(self) -> bool:
        return self._state.all_halted or self._state.day_halt_triggered or self._state.week_halt_triggered

    def clear_lock_file_if_present(self) -> bool:
        """Convenience for tests/manual ops -- NOT called automatically.
        The whole point of the lock file is that it requires a deliberate
        manual action to remove.
        """
        if self.lock_file_path.exists():
            self.lock_file_path.unlink()
            return True
        return False
