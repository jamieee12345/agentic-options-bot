"""Append-only local trade log: one line of JSON per open/close event,
written by the executors (order_execution.py, options_execution.py) and
read back by the dashboard to answer "what has this bot actually done."

Deliberately NOT reconstructed from Robinhood's own order history after the
fact -- that would mean correlating buy-to-open/sell-to-close legs from
`get_option_orders()`/`get_equity_orders()`, handling partial fills, and
guessing at pairing when multiple round-trips happen on the same symbol,
all against an API whose exact response shape is already flagged elsewhere
in this project as unverified. Logging the event directly, from the one
place that actually knows what just happened (the executor, right after
`broker.place_order`/`place_option_order` returns), is simpler and doesn't
depend on any of that.

Logs BOTH live and dry-run events (`dry_run: true`/`false`) -- this means
the dashboard can show a trade history, clearly marked as simulated,
before `live_trading_enabled` is ever flipped to true. Useful for building
confidence in the strategy's actual behavior before it risks real money.

In-memory pairing only, not a database: `build_closed_trades()` walks the
log in order, keeping the most recent unmatched "open" per symbol, and
pairs it with the next "close" for that symbol. Correct as long as this
project's own "at most one open position per underlying at a time" rule
(orchestration/options_execution.py, safety/order_validation.py's
duplicate guard) actually holds -- which is the whole point of that rule
existing.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_LOG_PATH = Path("trade_log.jsonl")


@dataclass(frozen=True)
class TradeLogEntry:
    event: str            # "open" or "close"
    timestamp: str         # ISO 8601
    symbol: str
    asset_type: str         # "option" or "equity"
    trade_type: str          # "call", "put", or "long" (equity)
    quantity: int
    price: float              # per-contract premium (options) or per-share price (equity)
    notional: float           # price * quantity * (100 if option else 1) -- "how much was put in" for an open, proceeds for a close
    dry_run: bool
    order_id: Optional[str] = None
    reason: Optional[str] = None   # close only: "signal", "flip", "stop_loss", "take_profit", "fvg_invalidated", "expiration"
    # Open only: the triggering FVG's bounds (brain/options_strategy.OptionsDecision.gap_low/gap_high).
    # Persisted here specifically so a structural stop (has price closed back
    # through the entire gap that justified this trade?) survives the bot
    # stopping and restarting overnight via the market-hours timers --
    # in-memory state alone wouldn't make it to the next trading day.
    gap_low: Optional[float] = None
    gap_high: Optional[float] = None
    # Everything below is open-only, from data.options_data.OptionContract
    # and brain.confluence.ConfluenceResult at the moment this trade was
    # opened -- persisted so "what contract, at what price, on what
    # reasoning" survives past the moment of the trade, for the dashboard's
    # full contract-detail display and orchestration/trade_grading.py's
    # post-trade grade (which needs to know how well-supported the ENTRY
    # was, not just how it turned out).
    strike_price: Optional[float] = None
    expiration_date: Optional[str] = None   # ISO date
    dte_at_entry: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_pct: Optional[float] = None      # (ask-bid)/mid at entry -- already required to clear risk.max_bid_ask_spread_pct to have opened at all, kept here for display/record, not re-checked
    confluence_score: Optional[float] = None
    confluence_applicable: Optional[int] = None
    confluence_details: Dict[str, str] = field(default_factory=dict)


def append_entry(entry: TradeLogEntry, path: Path = DEFAULT_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def read_entries(path: Path = DEFAULT_LOG_PATH) -> List[TradeLogEntry]:
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(TradeLogEntry(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue  # skip a malformed line (e.g. a partially-written entry from a crash mid-write) rather than fail the whole read
    return entries


@dataclass(frozen=True)
class ClosedTrade:
    symbol: str
    asset_type: str
    trade_type: str
    quantity: int
    opened_at: str
    closed_at: str
    entry_notional: float     # "how much was put in"
    exit_notional: float
    pnl_dollars: float
    pnl_pct: Optional[float]  # None if entry_notional was 0
    close_reason: Optional[str]
    dry_run: bool
    strike_price: Optional[float] = None
    expiration_date: Optional[str] = None
    dte_at_entry: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_pct: Optional[float] = None
    confluence_score: Optional[float] = None
    confluence_applicable: Optional[int] = None
    confluence_details: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenTrade:
    symbol: str
    asset_type: str
    trade_type: str
    quantity: int
    opened_at: str
    entry_notional: float
    dry_run: bool
    gap_low: Optional[float] = None
    gap_high: Optional[float] = None
    strike_price: Optional[float] = None
    expiration_date: Optional[str] = None
    dte_at_entry: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_pct: Optional[float] = None
    confluence_score: Optional[float] = None
    confluence_applicable: Optional[int] = None
    confluence_details: Dict[str, str] = field(default_factory=dict)


def build_trade_history(entries: List[TradeLogEntry]) -> "tuple[List[ClosedTrade], List[OpenTrade]]":
    pending: dict[str, TradeLogEntry] = {}
    closed: List[ClosedTrade] = []

    for entry in entries:
        if entry.event == "open":
            pending[entry.symbol] = entry
        elif entry.event == "close":
            open_entry = pending.pop(entry.symbol, None)
            if open_entry is None:
                continue  # a close with no matching open in this log (e.g. log started after the position was already open) -- can't compute P&L for it, skip
            pnl_dollars = entry.notional - open_entry.notional
            pnl_pct = (pnl_dollars / open_entry.notional) if open_entry.notional > 0 else None
            closed.append(ClosedTrade(
                symbol=entry.symbol, asset_type=entry.asset_type, trade_type=open_entry.trade_type,
                quantity=open_entry.quantity, opened_at=open_entry.timestamp, closed_at=entry.timestamp,
                entry_notional=open_entry.notional, exit_notional=entry.notional, pnl_dollars=pnl_dollars,
                pnl_pct=pnl_pct, close_reason=entry.reason, dry_run=entry.dry_run,
                strike_price=open_entry.strike_price, expiration_date=open_entry.expiration_date,
                dte_at_entry=open_entry.dte_at_entry, bid=open_entry.bid, ask=open_entry.ask,
                spread_pct=open_entry.spread_pct, confluence_score=open_entry.confluence_score,
                confluence_applicable=open_entry.confluence_applicable, confluence_details=open_entry.confluence_details,
            ))

    still_open = [
        OpenTrade(
            e.symbol, e.asset_type, e.trade_type, e.quantity, e.timestamp, e.notional, e.dry_run, e.gap_low, e.gap_high,
            strike_price=e.strike_price, expiration_date=e.expiration_date, dte_at_entry=e.dte_at_entry,
            bid=e.bid, ask=e.ask, spread_pct=e.spread_pct, confluence_score=e.confluence_score,
            confluence_applicable=e.confluence_applicable, confluence_details=e.confluence_details,
        )
        for e in pending.values()
    ]
    closed.sort(key=lambda t: t.closed_at, reverse=True)
    return closed, still_open
