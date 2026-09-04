"""Shared read/write for the two files that let dashboard/live_account_dashboard.py
show live-ish account state WITHOUT ever calling robin_stocks itself.

Why this exists: the dashboard used to fetch equity/positions directly via
RobinhoodBroker (robin_stocks), polling every 15s. That's exactly the kind
of automated/programmatic Robinhood API access this project's whole MCP
migration was meant to eliminate -- read-only or not, it's still an
unofficial-API bot polling on a loop. But a plain Python script (which is
all the dashboard can be -- it's not a Claude Code session) has no way to
call Robinhood MCP tools itself; only an interactive session or a scheduled
routine can do that.

So instead: the hourly MCP routine, which already reads equity/buying_power/
option_positions/quotes via MCP as part of its normal cycle (see
evaluate_for_agent.py's cmd_evaluate), ALSO writes that state here --
`account_snapshot.json` (latest state, overwritten each cycle) and one more
line in `equity_history.jsonl` (append-only, for the equity curve). Both are
committed and pushed back to the repo alongside trade_log.jsonl/
activity_log.jsonl, same persistence mechanism, same reason (a routine's
sandbox is fresh every firing). The dashboard then becomes a pure, static
reader of committed repo data -- no broker calls, no MCP calls, nothing that
could be mistaken for an automated trading bot. Its "live" refresh just
means "re-read whatever's currently on disk," on an interval matched to how
often a human would bother re-pulling the repo, not real API polling.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_EQUITY_HISTORY_PATH = Path("equity_history.jsonl")
DEFAULT_SNAPSHOT_PATH = Path("account_snapshot.json")
MAX_EQUITY_POINTS = 600  # bounded growth -- see append_equity_point/prune_equity_history


@dataclass(frozen=True)
class EquityPoint:
    timestamp: str
    equity: float


def append_equity_point(point: EquityPoint, path: Path = DEFAULT_EQUITY_HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(point)) + "\n")


def read_equity_history(path: Path = DEFAULT_EQUITY_HISTORY_PATH, max_points: int = MAX_EQUITY_POINTS) -> List[EquityPoint]:
    if not path.exists():
        return []
    points: List[EquityPoint] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                points.append(EquityPoint(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return points[-max_points:]


def prune_equity_history(path: Path = DEFAULT_EQUITY_HISTORY_PATH, keep: int = MAX_EQUITY_POINTS) -> None:
    points = read_equity_history(path, max_points=keep + 1)
    if len(points) <= keep:
        return
    trimmed = points[-keep:]
    path.write_text("\n".join(json.dumps(asdict(p)) for p in trimmed) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class SnapshotOptionPosition:
    symbol: str
    option_type: str
    strike_price: float
    quantity: int
    expiration_date: str  # ISO date
    average_premium_paid: float
    # Priced once, by the routine, from the same MCP quote it already
    # fetched for the stop-loss/take-profit check that cycle -- the
    # dashboard never re-fetches these itself. None if that quote wasn't
    # available this cycle (same fail-open behavior as the executor).
    bid: Optional[float] = None
    ask: Optional[float] = None
    current_value: Optional[float] = None
    pnl_dollars: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass(frozen=True)
class AccountSnapshot:
    fetched_at: str
    equity: float
    buying_power: float
    option_positions: List[SnapshotOptionPosition] = field(default_factory=list)
    # A count of SYMBOLS with a non-terminal order, not a literal order
    # count -- that's the level of detail open_order_symbols (what the
    # routine's MCP reads actually give us) supports. Good enough for the
    # dashboard's "N open orders" stat, not meant as an exact figure.
    open_order_count: int = 0


def write_account_snapshot(snapshot: AccountSnapshot, path: Path = DEFAULT_SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(snapshot), indent=2), encoding="utf-8")


def read_account_snapshot(path: Path = DEFAULT_SNAPSHOT_PATH) -> Optional[AccountSnapshot]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["option_positions"] = [SnapshotOptionPosition(**p) for p in raw.get("option_positions", [])]
        return AccountSnapshot(**raw)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None
