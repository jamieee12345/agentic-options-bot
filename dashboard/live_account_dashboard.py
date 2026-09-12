"""Refreshing HTML dashboard for the Agentic Robinhood account -- MCP era.

How this actually gets you a "refreshing" dashboard: this script runs
forever, re-reading local files and rewriting one HTML file to disk every
`--refresh-seconds`. The HTML itself has a <meta http-equiv="refresh"> tag,
so a browser tab left open on that file refreshes ITSELF on the same
interval, just by re-reading the file from disk -- no web server needed,
and no bridge from this chat to your browser is needed either (there isn't
one; a chat conversation can't push updates to a page on its own).

Run it (needs the venv set up per README's Setup section -- no Robinhood
credentials of any kind, see below for why):

    PYTHONPATH=. python3 dashboard/live_account_dashboard.py

Then open dashboard/output/live_dashboard.html in a browser and leave the
tab open.

THIS FILE MAKES NO BROKER OR MCP CALLS AT ALL -- deliberately, not just as
an optimization. It used to call RobinhoodBroker (robin_stocks) directly
for live equity/positions, polling every 15s; that's exactly the kind of
automated/programmatic Robinhood API access this project's whole MCP
migration was meant to eliminate, read-only or not. A plain Python script
like this one has no way to call Robinhood MCP tools anyway -- only an
interactive Claude Code session or a scheduled routine can. So instead:
the hourly MCP routine writes `account_snapshot.json` (equity, buying
power, open option positions with fresh quotes) every cycle as part of its
normal work (see orchestration/account_snapshot.py and
evaluate_for_agent.py's cmd_evaluate), commits it, and pushes it back to
the repo alongside the other logs. This script only ever reads that file
(plus trade_log.jsonl/activity_log.jsonl/equity_history.jsonl) and,
optionally, runs a plain `git pull` each cycle to sync in whatever the
routine last pushed (see `_git_pull` -- disable with --no-pull). "Refresh"
here means "re-read local disk," not "poll a live API" -- this can only
ever be as fresh as the routine's last hourly push, never truer real-time
than that.

Trade history comes from orchestration/trade_log.py's local log file, not
from Robinhood's own order history -- see that module's docstring for why.
It shows up here whether or not `broker.live_trading_enabled` is on:
dry-run trades are logged too (clearly marked "SIMULATED"), so you can see
what the strategy would have done before ever risking real money on it.

Two more local logs feed this page: `equity_history.jsonl` (one point per
routine cycle, appended by evaluate_for_agent.py, not by this file -- this
is what draws the equity curve) and orchestration/activity_log.py's
per-cycle, per-symbol log (every outcome, not just trades -- this is what
backs the "today" narrative section, since trade_log.py alone can't say
anything about the quiet cycles where nothing triggered, which is most of
them at this project's confluence bar).

Chart rendering is hand-rolled inline SVG, not a charting library -- no
CDN, no network dependency, nothing that can silently fail to load on a
machine with restricted egress (this dashboard may run on a headless
server; see _equity_curve_svg/_pnl_bar_svg below). Colors below follow the
dataviz skill's validated dark-mode palette (references/palette.md) --
series-1 blue for the equity line, the status good/critical pair for P&L
polarity, not a re-purposed categorical hue.

The "Live reasoning" section shows, per watched symbol, the full
gap/volume/confluence read-out from that symbol's most recent activity_log
entry (see orchestration/activity_log.py's confluence_details field) --
literally what brain/confluence.py looked at on its last cycle, not a
one-line summary of it. It's a display of already-computed data, not a
second opinion -- this file never recomputes indicators itself.

Default refresh is 60s -- not settings.yaml's monitoring.dashboard_refresh_seconds
(5s, sized for the old broker-polling loop), and not tied to any rate limit
any more either, since there's no API being polled: it's just how often
this re-reads local files (and, if --no-pull isn't set, runs `git pull`).
Since the underlying data only actually changes once per hour anyway (the
MCP routine's cadence), there's little reason to go much tighter than this.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Dict, List, Optional

from orchestration.account_snapshot import DEFAULT_EQUITY_HISTORY_PATH, DEFAULT_SNAPSHOT_PATH
from orchestration.account_snapshot import EquityPoint, read_account_snapshot, read_equity_history
from orchestration.activity_log import DEFAULT_LOG_PATH as DEFAULT_ACTIVITY_LOG_PATH
from orchestration.activity_log import ActivityEntry, entries_for_date
from orchestration.activity_log import read_entries as read_activity_entries
from orchestration.market_hours import MARKET_CLOSE, MARKET_OPEN, MARKET_TZ
from orchestration.options_execution import OpenOptionPosition
from orchestration.trade_grading import CheckPerformance, MIN_TRADES_FOR_AGGREGATE, TradeGrade, aggregate_check_performance, grade_trade
from orchestration.trade_log import ClosedTrade, DEFAULT_LOG_PATH as DEFAULT_TRADE_LOG_PATH, build_trade_history, read_entries
from dashboard.page import render_html
from dashboard.structure_panel import FvgParams, SymbolStructure, read_watchlist

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path(__file__).parent / "output" / "live_dashboard.html"
# Not a live-API polling interval any more -- see account_snapshot.py's
# module docstring. This just controls how often the local repo checkout is
# re-read and the HTML regenerated; the underlying data only actually
# changes once per hour, when the MCP routine's next cycle pushes.
DEFAULT_REFRESH_SECONDS = 60
MAX_HISTORY_ROWS = 50            # most recent N closed trades shown in the table -- the log itself keeps everything
ACCOUNT_LABEL = "account ...9190"  # Robinhood "Agentic" account 954079190 -- fixed, not read from a broker any more

@dataclass
class OptionPositionView:
    position: OpenOptionPosition
    current_value: Optional[float]   # None if a fresh quote couldn't be fetched this cycle
    pnl_dollars: Optional[float]
    pnl_pct: Optional[float]
    bid: Optional[float] = None      # CURRENT bid/ask -- distinct from trade_log's entry-time bid/ask, this is "what could I get out at right now"
    ask: Optional[float] = None
    spread_pct: Optional[float] = None


@dataclass(frozen=True)
class SymbolActivitySummary:
    symbol: str
    quiet_cycles: int          # evaluated, no fair value gap at all
    near_misses: int           # a gap formed but volume/confluence rejected it -- real activity, not silence


@dataclass
class DashboardSnapshot:
    fetched_at: datetime
    equity: float
    buying_power: float
    cash: Optional[float]
    stock_positions: Dict[str, float]
    option_positions: List[OptionPositionView]
    open_order_count: int
    trade_history: List[ClosedTrade]
    equity_history: List[EquityPoint]
    today_significant_events: List[ActivityEntry]   # opens, closes, and near-misses, most recent first
    today_symbol_summaries: List[SymbolActivitySummary]
    live_reasoning: List[ActivityEntry]              # most recent entry per watched symbol, sorted by symbol -- "what is it looking at right now"
    trade_grades: List[Optional[TradeGrade]]         # same order/length as trade_history -- orchestration/trade_grading.py's per-trade grade
    check_performance: List[CheckPerformance]        # empty until MIN_TRADES_FOR_AGGREGATE closed trades exist
    error: Optional[str] = None  # set instead of raising, so one bad cycle doesn't kill the loop
    # What the bot is configured to do right now (from settings.yaml): the
    # entry model, its session window, the protection gates, the DTE window
    # and the dry-run/live switch -- shown in a strip under the header so
    # the dashboard says which strategy produced what it shows.
    strategy: Dict[str, str] = field(default_factory=dict)
    # Per-symbol market-structure read from the bar cache (dashboard/structure_panel.py).
    structure: List[SymbolStructure] = field(default_factory=list)
    today_entries: List[ActivityEntry] = field(default_factory=list)   # every entry logged today (ET) -- drives the session timeline
    data_as_of: Optional[datetime] = None                             # account_snapshot.json's fetched_at = the last bot cycle
    policy_hard_keys: List[str] = field(default_factory=list)
    policy_soft_keys: List[str] = field(default_factory=list)


def _classify_activity(entries: List[ActivityEntry]) -> tuple[List[ActivityEntry], List[SymbolActivitySummary]]:
    significant: List[ActivityEntry] = []
    tallies: Dict[str, Dict[str, int]] = {}

    for e in entries:
        if e.outcome in ("open", "close"):
            significant.append(e)
            continue
        # A fresh gap that got rejected (volume, or confluence) is real
        # activity worth narrating -- only "no fair value gap at all" is
        # genuinely quiet.
        is_quiet = "no fair value gap" in e.detail
        tallies.setdefault(e.symbol, {"quiet": 0, "near_miss": 0})
        tallies[e.symbol]["quiet" if is_quiet else "near_miss"] += 1
        if not is_quiet:
            significant.append(e)

    significant.sort(key=lambda e: e.timestamp, reverse=True)
    summaries = [
        SymbolActivitySummary(symbol=sym, quiet_cycles=counts["quiet"], near_misses=counts["near_miss"])
        for sym, counts in sorted(tallies.items())
    ]
    return significant, summaries


def _latest_per_symbol(entries: List[ActivityEntry], watchlist: Optional[Iterable[str]] = None) -> List[ActivityEntry]:
    """One entry per symbol -- whichever has the latest timestamp -- sorted
    by symbol so the panel's card order doesn't jump around between
    refreshes. Deliberately NOT date-filtered to "today" (unlike
    _classify_activity's input): right at market open, or before the bot's
    first cycle of the day, this should still show yesterday's last read
    rather than an empty panel.

    `watchlist`, when given, restricts the panel to symbols the bot is
    CURRENTLY watching (settings.yaml's broker.core_watchlist). Without it,
    a symbol dropped from the watchlist would keep showing its last-ever
    read indefinitely, since the activity log is append-only history.
    """
    allowed = {sym.upper() for sym in watchlist} if watchlist is not None else None
    latest: Dict[str, ActivityEntry] = {}
    for e in entries:
        if allowed is not None and e.symbol.upper() not in allowed:
            continue
        current = latest.get(e.symbol)
        if current is None or e.timestamp > current.timestamp:
            latest[e.symbol] = e
    return [latest[sym] for sym in sorted(latest)]


def fetch_snapshot(
    stop_loss_pct: float,
    take_profit_pct: float,
    trade_log_path: Path = DEFAULT_TRADE_LOG_PATH,
    activity_log_path: Path = DEFAULT_ACTIVITY_LOG_PATH,
    equity_history_path: Path = DEFAULT_EQUITY_HISTORY_PATH,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    watchlist: Optional[Iterable[str]] = None,
    strategy: Optional[Dict[str, str]] = None,
    policy=None,
    min_confluence_score: float = 0.6,
    fvg_params: Optional[FvgParams] = None,
) -> DashboardSnapshot:
    """Pure read of files the MCP routine already committed and pushed --
    see orchestration/account_snapshot.py's module docstring for why this
    is a read, never a broker or MCP call. This can only ever be as fresh
    as the last `git pull` picked up from the routine's last push; it is
    NOT a live feed in the sense the old robin_stocks version was.
    """
    account = read_account_snapshot(snapshot_path)
    now = datetime.now(timezone.utc)

    open_positions: Dict[str, OpenOptionPosition] = {}
    option_views: List[OptionPositionView] = []
    if account is not None:
        equity, buying_power, open_order_count = account.equity, account.buying_power, account.open_order_count
        for p in account.option_positions:
            pos = OpenOptionPosition(
                symbol=p.symbol, option_type=p.option_type, strike_price=p.strike_price,
                quantity=p.quantity, expiration_date=date.fromisoformat(p.expiration_date),
                average_premium_paid=p.average_premium_paid,
            )
            open_positions[p.symbol] = pos
            mid = (p.bid + p.ask) / 2 if (p.bid is not None and p.ask is not None) else None
            spread_pct = ((p.ask - p.bid) / mid) if (mid and p.bid is not None and p.ask is not None) else None
            option_views.append(OptionPositionView(pos, p.current_value, p.pnl_dollars, p.pnl_pct, p.bid, p.ask, spread_pct))
    else:
        equity, buying_power, open_order_count = 0.0, 0.0, 0

    # This bot only ever trades options, never holds the underlying itself --
    # always empty, same as it effectively always was under the old
    # broker.get_positions() call too.
    stock_positions: Dict[str, float] = {}

    try:
        closed_trades, _still_open = build_trade_history(read_entries(trade_log_path))
    except Exception:
        logger.exception("Failed to read trade log at %s this cycle", trade_log_path)
        closed_trades = []

    trade_grades = [grade_trade(t, stop_loss_pct, take_profit_pct) for t in closed_trades[:MAX_HISTORY_ROWS]]
    check_performance = aggregate_check_performance(closed_trades)

    try:
        all_activity_entries = read_activity_entries(activity_log_path)
        today_local = now.astimezone(MARKET_TZ).date()
        today_entries = entries_for_date(all_activity_entries, today_local)
        significant_events, symbol_summaries = _classify_activity(today_entries)
        live_reasoning = _latest_per_symbol(all_activity_entries, watchlist)
    except Exception:
        logger.exception("Failed to read activity log at %s this cycle", activity_log_path)
        significant_events, symbol_summaries, live_reasoning, today_entries = [], [], [], []

    # Market-structure read from the committed bar cache -- same functions
    # and policy the bot uses, so the page's "calls only" is the bot's.
    structure: List[SymbolStructure] = []
    if watchlist is not None and policy is not None:
        try:
            structure = read_watchlist(list(watchlist), now, policy, min_confluence_score, fvg_params or FvgParams())
        except Exception:
            logger.exception("Market-structure read failed this cycle")

    try:
        equity_history = read_equity_history(equity_history_path)
    except Exception:
        logger.exception("Failed to read equity history at %s this cycle", equity_history_path)
        equity_history = []

    error = None
    if account is None:
        error = (
            "No account_snapshot.json in this checkout yet -- run `git pull` after the MCP "
            "routine's first cycle (it writes this file every run), or run `python -m "
            "orchestration.evaluate_for_agent evaluate` once locally against real account state."
        )

    return DashboardSnapshot(
        fetched_at=now, equity=equity, buying_power=buying_power, cash=None,
        stock_positions=stock_positions, option_positions=option_views, open_order_count=open_order_count,
        trade_history=closed_trades[:MAX_HISTORY_ROWS], equity_history=equity_history,
        today_significant_events=significant_events[:30], today_symbol_summaries=symbol_summaries,
        live_reasoning=live_reasoning,
        trade_grades=trade_grades, check_performance=check_performance,
        error=error, strategy=strategy or {}, structure=structure, today_entries=today_entries,
        data_as_of=datetime.fromisoformat(account.fetched_at) if account is not None else None,
        policy_hard_keys=list(policy.effective_keys()[0]) if policy is not None else [],
        policy_soft_keys=list(policy.effective_keys()[1]) if policy is not None else [],
    )


def _git_pull() -> None:
    """Best-effort sync with whatever the MCP routine's last cycle pushed.
    Not a broker/MCP call -- a plain `git pull` against this repo's own
    remote, same as a person would run by hand. Failure (no network, local
    uncommitted changes, detached HEAD) is logged and swallowed rather than
    crashing the refresh loop -- the dashboard just renders whatever's
    already in the local checkout that cycle, same as if pull were skipped.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"], capture_output=True, text=True, timeout=30,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if result.returncode != 0:
            logger.warning("git pull failed this cycle (non-fatal): %s", result.stderr.strip())
    except Exception:
        logger.exception("git pull raised this cycle (non-fatal)")


def _strategy_summary(settings) -> Dict[str, str]:
    o = settings.options
    session = f"{o.entry_session_start}-{o.entry_session_end} ET (exits managed all day)"
    model_desc = {
        "confluence": "FVG + lean confluence (1h/4h trend + structure hard; volume profile / 200-SMA / S-R soft)",
        "sweep": "liquidity sweep -> displacement gap",
        "orb": f"{o.orb_minutes}-min opening range breakout (vol >{o.orb_volume_multiplier}x, VWAP)",
    }.get(o.model, o.model)
    return {
        "live_trading_enabled": "true" if settings.broker.live_trading_enabled else "false",
        "model": f"{o.model} -- {model_desc}",
        "entries": session,
        "DTE": f"{o.target_dte_min}-{o.target_dte_max}" + (" (0DTE TEST -- allow_0dte on)" if o.allow_0dte and o.target_dte_min == 0 else ""),
        "max hold": f"{o.max_hold_days} day",
        "per trade": f"{o.max_premium_pct_per_trade:.0%} of equity",
        "gates": f"{o.max_entries_per_day}/day, {o.max_open_positions} open, day -{o.daily_loss_limit_pct:.0%}, week -{o.weekly_loss_limit_pct:.0%}",
        "max_open": str(o.max_open_positions),
        "watchlist": ", ".join(settings.broker.core_watchlist),
    }


def run_forever(
    output_path: Path = DEFAULT_OUTPUT_PATH, refresh_seconds: int = DEFAULT_REFRESH_SECONDS,
    settings_path: str = "config/settings.yaml", auto_pull: bool = True, once: bool = False,
) -> None:
    """Render the dashboard on a loop -- or exactly once when ``once`` is set.

    ``once`` exists for publishing a snapshot elsewhere (e.g. as a private
    Claude artifact on request): render one HTML file from the current repo
    state and exit. Everything else is identical to the local watch mode."""
    # Settings loaded here only for stop_loss_pct/take_profit_pct, which
    # orchestration/trade_grading.py needs to bucket a closed trade's P&L
    # into "clean win"/"small loss"/etc against THIS account's actual
    # thresholds, not an arbitrary number picked in this file. Everything
    # else this dashboard shows comes from committed repo files -- see
    # fetch_snapshot's docstring -- never a broker or MCP call, so this
    # stays a read-only monitor by construction, not just by convention.
    from config.config_loader import load_settings
    settings = load_settings(settings_path)

    o = settings.options
    policy = o.confluence_policy()
    fvg_params = FvgParams(o.fvg_lookback_period, o.fvg_body_multiplier, o.fvg_volume_multiplier, o.fvg_min_gap_atr_multiplier)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Writing %s every %ds (re-reading local files%s, no broker/MCP calls). Open it in a browser and leave the tab open.",
        output_path, refresh_seconds, " + git pull" if auto_pull else "",
    )

    last_good: Optional[DashboardSnapshot] = None
    while True:
        if auto_pull:
            _git_pull()
        try:
            snapshot = fetch_snapshot(
                o.stop_loss_pct, o.take_profit_pct,
                watchlist=settings.broker.core_watchlist, strategy=_strategy_summary(settings),
                policy=policy, min_confluence_score=o.min_confluence_score, fvg_params=fvg_params,
            )
            last_good = snapshot
        except Exception as exc:
            logger.exception("Dashboard refresh failed this cycle -- will retry next cycle")
            if last_good is None:
                time.sleep(refresh_seconds)
                continue
            snapshot = last_good
            snapshot.error = f"{type(exc).__name__}: {exc}"

        output_path.write_text(render_html(snapshot, refresh_seconds, ACCOUNT_LABEL), encoding="utf-8")
        if once:
            return
        time.sleep(refresh_seconds)


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Explicit path, not load_dotenv()'s bare default -- see
    # orchestration/run_live.py's __main__ block for why that default (a
    # stack-frame-based search from the calling file, not the process cwd)
    # is not safe to rely on here.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--refresh-seconds", type=int, default=DEFAULT_REFRESH_SECONDS)
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--no-pull", action="store_true", help="Don't auto `git pull` each cycle -- just re-read whatever's on disk")
    parser.add_argument("--once", action="store_true", help="Render one HTML file and exit; implies --no-pull")
    args = parser.parse_args()
    run_forever(
        Path(args.output), args.refresh_seconds, args.settings,
        auto_pull=not (args.no_pull or args.once), once=args.once,
    )
