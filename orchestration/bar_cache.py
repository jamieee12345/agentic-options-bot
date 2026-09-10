"""Git-persisted cache of the OHLCV bars the routine feeds into
orchestration/evaluate_for_agent.py, so each hourly cycle only has to pull
the bars that are NEW since the previous cycle instead of re-fetching the
whole history every hour.

Why this exists -- cost, not correctness. Every bar the routine's agent
fetches through Robinhood MCP's get_equity_historicals lands in the agent's
own context (and then gets re-emitted by the agent to reach Python via
stdin). Re-fetching 400 daily bars + 3 days of 5-minute bars for 10 symbols
every hour was ~800K characters of tool output per cycle, which exhausted
the account's 5-hour usage window after two cycles and silently killed the
other seven firings of the day. With the history cached here (committed
back to the repo each cycle exactly like trade_log.jsonl -- the sandbox is
thrown away after every run, so the repo IS the persistence layer), a
steady-state cycle fetches roughly a dozen 5-minute bars and one or two
hour/4hour/day bars per symbol: ~25K characters total, ~30x less.

Robinhood MCP stays the ONLY source of bars -- this file never fetches
anything. It only (a) tells the agent what to fetch (`plan_fetches`: the
smallest set of get_equity_historicals calls that brings every symbol/
interval up to date), and (b) merges what came back into the cache
(`merge_into_cache`) so `evaluate` can run on the full window.

Layout: one CSV per symbol per interval under market_data/, e.g.
market_data/SPY_5minute.csv, columns timestamp,open,high,low,close,volume,
ascending, UTC. Line-per-bar so git diffs are appends. Each interval keeps
a bounded retention window (see INTERVAL_SPECS) so the files -- and the
per-cycle commit -- stay small.

Two details that matter for correctness:

* The last cached bar is always RE-FETCHED (plan_fetches starts the range
  AT the last cached timestamp, not after it). Any bar fetched while it
  was still forming gets overwritten by its completed version next cycle
  -- merge keeps the newest copy of a timestamp -- so the cache
  self-heals instead of permanently holding a partial bar.
* Backfill is BUDGETED and CHUNKED, never one-shot. Every bar the agent
  fetches has to be re-emitted by the agent as output to reach Python,
  and a full backfill (~6,500 bars for 10 symbols x 4 intervals) proved
  to exhaust the account's 5-hour usage window mid-run -- twice. So
  `plan_fetches` caps each cycle at MAX_BARS_PER_CYCLE: forward
  increments for every symbol/interval come first (the bars that
  actually matter for today's decision), then history is extended
  BACKWARD one `chunk_days` slice at a time (5-minute first, then hour,
  4hour, day) until each interval's backfill window is full. Until the
  daily history fills in (~3-4 cycles) the 200-SMA soft check simply
  reads "n/a" (fails open); the 1h/4h reads are complete after 1-2.
* `drop_incomplete_last_bar` removes a bar whose interval hasn't elapsed
  yet at evaluation time. The FVG trigger keys off "the most recent bar";
  a half-formed 5-minute candle is not a confirmed gap, and the backtest
  only ever sees completed bars, so evaluating live on a partial bar would
  be the one place live and backtest disagree by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

DEFAULT_CACHE_DIR = Path("market_data")

# Maximum bars we're willing to put in ONE get_equity_historicals call.
# Empirically a bar is ~150-160 characters of tool output and the MCP host
# rejects results somewhere past ~100-150K characters, so 300 bars (~48K)
# leaves a wide margin. Robinhood also caps symbols at 10 per call.
MAX_BARS_PER_CALL = 300
MAX_SYMBOLS_PER_CALL = 10

# Soft cap on bars planned per CYCLE (the last call that crosses it is
# still included, so a cycle plans at most this plus one call's worth).
# Sized from evidence: the pre-cache routine re-emitted ~4,500 bars per
# cycle and two such cycles exhausted a 5-hour window, so ~1,500 leaves
# room for four-plus cycles per window with the agent's other work.
MAX_BARS_PER_CYCLE = 1500


@dataclass(frozen=True)
class IntervalSpec:
    interval: str          # the exact get_equity_historicals `interval` value
    payload_key: str       # the key evaluate_for_agent's stdin payload uses for it
    backfill_days: int     # calendar days of history to end up with (filled in over several cycles)
    chunk_days: int        # calendar days fetched per backward-extension step
    retention_days: int    # calendar days to keep in the cache
    bars_per_day: float    # regular-session bars per trading day, for sizing calls
    bar_length: timedelta  # used to decide whether the latest bar is still forming


# Backfill windows are sized to what brain/ actually consumes: 5-minute bars
# feed the FVG trigger + structure/S-R/VPVR (look back 10-20 bars, so 3
# trading days is generous -- and the retention window of 10 calendar days
# means Monday morning still has all of last week, which the old
# 3-calendar-day fetch did NOT after a weekend/holiday); hour and 4hour feed
# the trend_1h/trend_4h SMA(20) hard vetoes in brain/confluence.py, so 20
# and 60 calendar days give ~100 and ~80 bars respectively; day feeds the
# 200-SMA soft check, hence 400 calendar days (~252 trading days).
INTERVAL_SPECS: Dict[str, IntervalSpec] = {
    "5minute": IntervalSpec("5minute", "intraday_bars", backfill_days=3, chunk_days=1, retention_days=10, bars_per_day=78, bar_length=timedelta(minutes=5)),
    "hour": IntervalSpec("hour", "hourly_bars", backfill_days=20, chunk_days=5, retention_days=45, bars_per_day=7, bar_length=timedelta(hours=1)),
    "4hour": IntervalSpec("4hour", "four_hour_bars", backfill_days=60, chunk_days=16, retention_days=120, bars_per_day=2, bar_length=timedelta(hours=4)),
    "day": IntervalSpec("day", "daily_bars", backfill_days=400, chunk_days=60, retention_days=420, bars_per_day=1, bar_length=timedelta(days=1)),
}

PAYLOAD_KEY_TO_INTERVAL = {spec.payload_key: spec.interval for spec in INTERVAL_SPECS.values()}

COLUMNS = ["open", "high", "low", "close", "volume"]


def cache_path(symbol: str, interval: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / f"{symbol.upper()}_{interval}.csv"


def load_cached(symbol: str, interval: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Optional[pd.DataFrame]:
    """The cached bars, same DataFrame shape every fetcher produces
    (UTC DatetimeIndex, ascending, float OHLCV), or None if nothing's cached."""
    path = cache_path(symbol, interval, cache_dir)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[COLUMNS].astype(float)


def save_cached(symbol: str, interval: str, bars: pd.DataFrame, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    path = cache_path(symbol, interval, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bars[COLUMNS].copy()
    out.index = out.index.tz_convert("UTC")
    out.index.name = "timestamp"
    out.to_csv(path, date_format="%Y-%m-%dT%H:%M:%SZ", float_format="%.6g")
    return path


def merge_bars(cached: Optional[pd.DataFrame], fresh: Optional[pd.DataFrame], retention_days: int, now: datetime) -> Optional[pd.DataFrame]:
    """Union of cached + fresh, newest copy wins on a duplicate timestamp
    (so a re-fetched, now-complete bar replaces the partial one cached
    last cycle), pruned to the retention window."""
    parts = [p for p in (cached, fresh) if p is not None and not p.empty]
    if not parts:
        return None
    merged = pd.concat(parts)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    cutoff = now - timedelta(days=retention_days)
    return merged[merged.index >= cutoff]


def drop_incomplete_last_bar(bars: Optional[pd.DataFrame], interval: str, now: datetime) -> Optional[pd.DataFrame]:
    """Drops the final bar if its interval hasn't fully elapsed yet -- see
    the module docstring. Daily bars are exempt from the strict rule
    (a 'day' bar's close is the session close, not midnight+24h; the
    trend SMA it feeds is insensitive to one still-forming bar anyway)."""
    if bars is None or bars.empty or interval == "day":
        return bars
    spec = INTERVAL_SPECS[interval]
    last_start = bars.index[-1].to_pydatetime()
    if last_start + spec.bar_length > now:
        return bars.iloc[:-1]
    return bars


@dataclass(frozen=True)
class FetchCall:
    symbols: List[str]
    interval: str
    start_time: str            # RFC3339 UTC, ready to paste into get_equity_historicals
    payload_key: str
    estimated_bars: int
    reason: str                # "incremental" (new bars since last cycle) or "backfill" (extending history backward)
    end_time: Optional[str] = None  # set on backward-extension calls only; pass through when present


def _fmt(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _estimate_bars(spec: IntervalSpec, start: datetime, end: datetime) -> int:
    span_days = max((end - start).total_seconds() / 86400.0, 0.0)
    return int(span_days * spec.bars_per_day * (5.0 / 7.0)) + 2  # +2 slack: inclusive re-fetch + whatever formed since


def _batch(symbols: List[str], spec: IntervalSpec, start: datetime, end: Optional[datetime], now: datetime, reason: str) -> List[FetchCall]:
    per_symbol = _estimate_bars(spec, start, end or now)
    per_call = max(1, min(MAX_SYMBOLS_PER_CALL, MAX_BARS_PER_CALL // max(per_symbol, 1)))
    return [
        FetchCall(
            symbols=symbols[i:i + per_call], interval=spec.interval, start_time=_fmt(start), payload_key=spec.payload_key,
            estimated_bars=per_symbol * len(symbols[i:i + per_call]), reason=reason,
            end_time=_fmt(end) if end is not None else None,
        )
        for i in range(0, len(symbols), per_call)
    ]


def plan_fetches(
    symbols: Iterable[str], now: datetime, intervals: Iterable[str] = tuple(INTERVAL_SPECS),
    cache_dir: Path = DEFAULT_CACHE_DIR, max_bars_per_cycle: int = MAX_BARS_PER_CYCLE,
) -> List[FetchCall]:
    """The get_equity_historicals calls for THIS cycle, in priority order,
    under the per-cycle bar budget (see the module docstring):

      1. incremental -- for every (symbol, interval) with a cache, the bars
         since the last cached one (inclusive). Always planned; this is
         the data today's decision actually depends on.
      2. backfill -- for every (symbol, interval) whose history is shorter
         than its backfill window, one chunk_days slice extending it
         backward (or the most recent chunk, if the cache is empty).
         Intervals in the order given (5minute, hour, 4hour, day), added
         until the budget is crossed; the rest wait for the next cycle.

    Symbols at the same point are batched into one call (up to
    MAX_BARS_PER_CALL / MAX_SYMBOLS_PER_CALL)."""
    symbols = [s.upper() for s in symbols]
    incremental: List[FetchCall] = []
    backfill: List[FetchCall] = []
    for interval in intervals:
        spec = INTERVAL_SPECS[interval]
        target_start = now - timedelta(days=spec.backfill_days)
        stale_cutoff = now - timedelta(days=spec.retention_days)
        fwd_groups: Dict[str, Dict] = {}
        back_groups: Dict[str, Dict] = {}
        for symbol in symbols:
            cached = load_cached(symbol, interval, cache_dir)
            if cached is None or cached.index[-1].to_pydatetime() < stale_cutoff:
                # empty/stale: most recent chunk only (no end_time), history extends next cycle
                start = max(target_start, now - timedelta(days=spec.chunk_days))
                g = back_groups.setdefault((_fmt(start), None), {"start": start, "end": None, "symbols": []})
                g["symbols"].append(symbol)
                continue
            last = cached.index[-1].to_pydatetime()
            g = fwd_groups.setdefault(_fmt(last), {"start": last, "symbols": []})
            g["symbols"].append(symbol)
            earliest = cached.index[0].to_pydatetime()
            if earliest > target_start + timedelta(days=spec.chunk_days):
                start = max(target_start, earliest - timedelta(days=spec.chunk_days))
                g = back_groups.setdefault((_fmt(start), _fmt(earliest)), {"start": start, "end": earliest, "symbols": []})
                g["symbols"].append(symbol)
        for key in sorted(fwd_groups):
            g = fwd_groups[key]
            incremental.extend(_batch(g["symbols"], spec, g["start"], None, now, "incremental"))
        for key in sorted(back_groups, key=lambda k: (k[0], k[1] or "")):
            g = back_groups[key]
            backfill.extend(_batch(g["symbols"], spec, g["start"], g["end"], now, "backfill"))

    calls = list(incremental)
    planned = sum(c.estimated_bars for c in calls)
    for call in backfill:
        if planned >= max_bars_per_cycle:
            break
        calls.append(call)
        planned += call.estimated_bars
    return calls


def merge_into_cache(
    fresh_by_symbol: Dict[str, pd.DataFrame], interval: str, now: datetime, cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Dict[str, Path]:
    """Merges freshly fetched bars for one interval into the on-disk cache.
    Returns the paths written (for the routine to `git add`)."""
    spec = INTERVAL_SPECS[interval]
    written = {}
    for symbol, fresh in fresh_by_symbol.items():
        merged = merge_bars(load_cached(symbol, interval, cache_dir), fresh, spec.retention_days, now)
        if merged is not None and not merged.empty:
            written[symbol] = save_cached(symbol, interval, merged, cache_dir)
    return written


def load_all(symbols: Iterable[str], interval: str, now: datetime, cache_dir: Path = DEFAULT_CACHE_DIR) -> Dict[str, pd.DataFrame]:
    """Every symbol's cached bars for one interval, with a still-forming
    last bar dropped -- the shape evaluate_for_agent hands to the executor."""
    out = {}
    for symbol in symbols:
        bars = drop_incomplete_last_bar(load_cached(symbol, interval, cache_dir), interval, now)
        if bars is not None and not bars.empty:
            out[symbol.upper()] = bars
    return out
