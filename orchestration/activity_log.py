"""Append-only local log of every per-symbol outcome each cycle produces --
not just opens/closes (orchestration/trade_log.py's job), but holds and
skips too. This is what lets the dashboard narrate "what happened today,"
including the (usual) quiet cycles where nothing traded -- trade_log.py
alone can't tell that story, since a bar with no trigger never touches it.

Same append-only JSONL pattern as trade_log.py, same reasoning for why:
one place (the executor, right after it decides) writes down what just
happened, rather than trying to reconstruct it later from something else.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_LOG_PATH = Path("activity_log.jsonl")
MAX_ENTRIES_KEPT = 5000  # rotation point -- see prune_old_entries()


@dataclass(frozen=True)
class ActivityEntry:
    timestamp: str        # ISO 8601
    symbol: str
    outcome: str            # "hold", "open", "close", "skipped" -- mirrors OptionsExecutionRecord.action (None -> "hold"/"skipped")
    option_type: Optional[str]
    contracts: Optional[int]
    detail: str              # the reasoning/skipped_reason string -- this is the actual narrative content
    # Everything below is optional and defaults to "nothing to show" so old
    # log lines (written before these fields existed) still parse fine via
    # ActivityEntry(**json.loads(line)) -- the missing keys just fall back
    # to these defaults. Populated from brain/options_strategy.OptionsDecision
    # whenever that ran for this symbol this cycle (see
    # orchestration/options_execution.py's run()); feeds the dashboard's
    # live-reasoning panel -- "exactly what the bot is looking at right now"
    # for every watched symbol, not just the one-line summary in `detail`.
    price: Optional[float] = None
    gap_kind: Optional[str] = None                 # "bullish"/"bearish" if a fresh FVG formed on the latest bar, else None
    volume_confirmed: Optional[bool] = None         # None if no gap; True/False once one formed
    confluence_details: Dict[str, str] = field(default_factory=dict)  # per-check "pass"/"fail"/"n/a", only once confluence actually ran
    confluence_score: Optional[float] = None        # soft-check score (0-1), independent of hard vetoes -- see brain/confluence.py
    confluence_applicable: int = 0


def append_entry(entry: ActivityEntry, path: Path = DEFAULT_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def read_entries(path: Path = DEFAULT_LOG_PATH) -> List[ActivityEntry]:
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(ActivityEntry(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return entries


def entries_for_date(entries: List[ActivityEntry], target_date: date) -> List[ActivityEntry]:
    return [e for e in entries if datetime.fromisoformat(e.timestamp).date() == target_date]


def prune_old_entries(path: Path = DEFAULT_LOG_PATH, keep: int = MAX_ENTRIES_KEPT) -> None:
    """Every cycle logs one entry per symbol, every ~5 minutes, all day --
    this file grows fast (a 4-symbol watchlist polled every 5 min for a
    6.5-hour session is already ~300 lines/day). Called occasionally (not
    every cycle) by whatever's writing to keep the file from growing
    unbounded over weeks of unattended operation. Keeps the most RECENT
    `keep` entries -- old ones aren't archived anywhere, just dropped.
    """
    entries = read_entries(path)
    if len(entries) <= keep:
        return
    trimmed = entries[-keep:]
    path.write_text("\n".join(json.dumps(asdict(e)) for e in trimmed) + "\n", encoding="utf-8")
