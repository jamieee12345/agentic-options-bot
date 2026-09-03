"""Continuously-refreshing HTML dashboard for the Agentic Robinhood account.

How this actually gets you a "constantly refreshing" dashboard: this script
runs forever, re-fetching account state and rewriting one HTML file to disk
every `--refresh-seconds`. The HTML itself has a <meta http-equiv="refresh">
tag, so a browser tab left open on that file refreshes ITSELF on the same
interval, just by re-reading the file from disk -- no web server needed,
and no bridge from this chat to your browser is needed either (there isn't
one; a chat conversation can't push updates to a page on its own).

Run it (needs the venv set up per README's Setup section, and
ROBINHOOD_USERNAME/PASSWORD/MFA_SECRET/ACCOUNT_NUMBER in your .env --
never put those values in this file):

    PYTHONPATH=. python3 dashboard/live_account_dashboard.py

Then open dashboard/output/live_dashboard.html in a browser and leave the
tab open.

Deliberately read-only: this only ever calls the broker's GET-style methods
(equity, positions, option positions, open orders, quotes) -- never
place_order/place_option_order. It's a monitor, not a trading loop.

Trade history comes from orchestration/trade_log.py's local log file, not
from Robinhood's own order history -- see that module's docstring for why.
It shows up here whether or not `broker.live_trading_enabled` is on:
dry-run trades are logged too (clearly marked "SIMULATED"), so you can see
what the strategy would have done before ever risking real money on it.

Two more local logs feed this page, both new: `equity_history.jsonl` (one
point appended per refresh cycle, right here, independent of whether
run_live.py itself is running -- this is what draws the equity curve) and
orchestration/activity_log.py's per-cycle, per-symbol log (every outcome,
not just trades -- this is what backs the "today" narrative section, since
trade_log.py alone can't say anything about the quiet cycles where nothing
triggered, which is most of them at this project's confluence bar).

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

Default refresh is 15s, not settings.yaml's monitoring.dashboard_refresh_seconds
(5s) -- that default was sized for the full regime/signal loop, not for a
standalone script hitting robin_stocks (unofficial, rate-limit-sensitive)
with a burst of calls every cycle: account equity/buying power/positions/
option positions/open orders, PLUS one fresh quote per open options
position. Tighten it if you've confirmed your account doesn't get rate-
limited at that cadence.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from broker.robinhood_broker import RobinhoodBroker
from data.news import AlpacaNewsFetcher, NewsItem
from data.options_data import RobinhoodOptionChainFetcher
from orchestration.activity_log import DEFAULT_LOG_PATH as DEFAULT_ACTIVITY_LOG_PATH
from orchestration.activity_log import ActivityEntry, entries_for_date, prune_old_entries
from orchestration.activity_log import read_entries as read_activity_entries
from orchestration.market_hours import MARKET_CLOSE, MARKET_OPEN, MARKET_TZ
from orchestration.options_execution import OpenOptionPosition, build_open_positions_from_broker
from orchestration.trade_grading import CheckPerformance, MIN_TRADES_FOR_AGGREGATE, TradeGrade, aggregate_check_performance, grade_trade
from orchestration.trade_log import ClosedTrade, DEFAULT_LOG_PATH as DEFAULT_TRADE_LOG_PATH, build_trade_history, read_entries

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path(__file__).parent / "output" / "live_dashboard.html"
DEFAULT_EQUITY_HISTORY_PATH = Path("equity_history.jsonl")
DEFAULT_REFRESH_SECONDS = 15
MAX_HISTORY_ROWS = 50            # most recent N closed trades shown in the table -- the log itself keeps everything
MAX_EQUITY_POINTS = 600          # ~2.5 days of history at 15s intervals -- enough for the chart, bounded growth

# Display order + friendly labels for the confluence checklist -- keys must
# match brain/confluence.py's `details` dict exactly (HARD_VETO_KEYS then
# SOFT_CHECK_KEYS there). Hard vetoes first since any one of those failing
# is decisive regardless of how the soft checks below it score.
CONFLUENCE_CHECK_LABELS = [
    ("trend_200sma", "200-SMA trend"),
    ("market_structure", "Market structure"),
    ("elliott_wave", "Elliott Wave"),
    ("break_of_structure", "Break of structure"),
    ("support_resistance", "Support/resistance"),
    ("supply_demand", "Supply/demand"),
    ("liquidity_sweep", "Liquidity sweep"),
    ("volume_profile", "Volume profile (VPVR)"),
    ("vpvr_node_quality", "VPVR node quality"),
    ("rsi_momentum", "RSI momentum"),
    ("volatility_expansion", "Volatility expansion"),
]


@dataclass(frozen=True)
class EquityPoint:
    timestamp: str
    equity: float


def append_equity_point(point: EquityPoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(point)) + "\n")


def read_equity_history(path: Path, max_points: int = MAX_EQUITY_POINTS) -> List[EquityPoint]:
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


def prune_equity_history(path: Path, keep: int = MAX_EQUITY_POINTS) -> None:
    points = read_equity_history(path, max_points=keep + 1)
    if len(points) <= keep:
        return
    trimmed = points[-keep:]
    path.write_text("\n".join(json.dumps(asdict(p)) for p in trimmed) + "\n", encoding="utf-8")


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
    news_by_symbol: Dict[str, List[NewsItem]]        # recent headlines for symbols with an actual signal or open position -- context only, see data/news.py
    trade_grades: List[Optional[TradeGrade]]         # same order/length as trade_history -- orchestration/trade_grading.py's per-trade grade
    check_performance: List[CheckPerformance]        # empty until MIN_TRADES_FOR_AGGREGATE closed trades exist
    error: Optional[str] = None  # set instead of raising, so one bad cycle doesn't kill the loop


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


def _latest_per_symbol(entries: List[ActivityEntry]) -> List[ActivityEntry]:
    """One entry per symbol -- whichever has the latest timestamp -- sorted
    by symbol so the panel's card order doesn't jump around between
    refreshes. Deliberately NOT date-filtered to "today" (unlike
    _classify_activity's input): right at market open, or before the bot's
    first cycle of the day, this should still show yesterday's last read
    rather than an empty panel.
    """
    latest: Dict[str, ActivityEntry] = {}
    for e in entries:
        current = latest.get(e.symbol)
        if current is None or e.timestamp > current.timestamp:
            latest[e.symbol] = e
    return [latest[sym] for sym in sorted(latest)]


def fetch_snapshot(
    broker: RobinhoodBroker,
    chain_fetcher: RobinhoodOptionChainFetcher,
    stop_loss_pct: float,
    take_profit_pct: float,
    news_fetcher: Optional[AlpacaNewsFetcher] = None,
    trade_log_path: Path = DEFAULT_TRADE_LOG_PATH,
    activity_log_path: Path = DEFAULT_ACTIVITY_LOG_PATH,
    equity_history_path: Path = DEFAULT_EQUITY_HISTORY_PATH,
) -> DashboardSnapshot:
    equity = broker.get_portfolio_equity()
    buying_power = broker.get_buying_power()
    stock_positions = broker.get_positions()
    raw_options = broker.get_option_positions()
    open_positions = build_open_positions_from_broker(raw_options)
    open_orders = broker.get_open_orders()

    option_views: List[OptionPositionView] = []
    for pos in open_positions.values():
        try:
            quote = chain_fetcher.get_quote_for_known_contract(pos.symbol, pos.option_type, pos.expiration_date, pos.strike_price)
        except Exception:
            logger.exception("%s: quote fetch failed this cycle", pos.symbol)
            quote = None

        if quote is not None and quote.mid_price is not None:
            current_value = quote.mid_price * pos.quantity * 100
            entry_value = pos.average_premium_paid * pos.quantity * 100
            pnl_dollars = current_value - entry_value
            pnl_pct = (pnl_dollars / entry_value) if entry_value > 0 else None
        else:
            current_value, pnl_dollars, pnl_pct = None, None, None

        bid = quote.bid if quote is not None else None
        ask = quote.ask if quote is not None else None
        spread_pct = ((ask - bid) / quote.mid_price) if (quote is not None and bid is not None and ask is not None and quote.mid_price) else None

        option_views.append(OptionPositionView(pos, current_value, pnl_dollars, pnl_pct, bid, ask, spread_pct))

    try:
        closed_trades, _still_open = build_trade_history(read_entries(trade_log_path))
    except Exception:
        logger.exception("Failed to read trade log at %s this cycle", trade_log_path)
        closed_trades = []

    trade_grades = [grade_trade(t, stop_loss_pct, take_profit_pct) for t in closed_trades[:MAX_HISTORY_ROWS]]
    check_performance = aggregate_check_performance(closed_trades)

    now = datetime.now(timezone.utc)
    try:
        all_activity_entries = read_activity_entries(activity_log_path)
        today_local = now.astimezone(MARKET_TZ).date()
        today_entries = entries_for_date(all_activity_entries, today_local)
        significant_events, symbol_summaries = _classify_activity(today_entries)
        live_reasoning = _latest_per_symbol(all_activity_entries)
    except Exception:
        logger.exception("Failed to read activity log at %s this cycle", activity_log_path)
        significant_events, symbol_summaries, live_reasoning = [], [], []

    # News is context for a human reading the dashboard, not an input to any
    # trading decision (see data/news.py's module docstring) -- fetched only
    # for symbols where there's actually something to explain: a real gap
    # formed on the latest bar, or a position is open. Fetching it for every
    # quiet symbol every cycle would be both noisy (nothing to explain) and
    # wasteful (an API call for no reason).
    news_by_symbol: Dict[str, List[NewsItem]] = {}
    if news_fetcher is not None:
        newsworthy_symbols = {e.symbol for e in live_reasoning if e.gap_kind is not None} | set(open_positions.keys())
        for symbol in newsworthy_symbols:
            try:
                items = news_fetcher.get_recent_news(symbol)
            except Exception:
                logger.exception("%s: news fetch failed this cycle", symbol)
                items = []
            if items:
                news_by_symbol[symbol] = items

    try:
        append_equity_point(EquityPoint(timestamp=now.isoformat(), equity=equity), equity_history_path)
        prune_equity_history(equity_history_path)
        prune_old_entries(activity_log_path)
        equity_history = read_equity_history(equity_history_path)
    except Exception:
        logger.exception("Failed to update equity history this cycle")
        equity_history = []

    return DashboardSnapshot(
        fetched_at=now, equity=equity, buying_power=buying_power, cash=None,
        stock_positions=stock_positions, option_positions=option_views, open_order_count=len(open_orders),
        trade_history=closed_trades[:MAX_HISTORY_ROWS], equity_history=equity_history,
        today_significant_events=significant_events[:30], today_symbol_summaries=symbol_summaries,
        live_reasoning=live_reasoning, news_by_symbol=news_by_symbol,
        trade_grades=trade_grades, check_performance=check_performance,
    )


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "&mdash;"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "&mdash;"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1%}"


def _market_phase(now_utc: datetime) -> str:
    """Drives the pipeline stepper's "you are here" state -- deliberately
    simple (weekday + time-of-day only), same known gap as
    orchestration/market_hours.py itself: no holiday calendar.
    """
    local = now_utc.astimezone(MARKET_TZ)
    if local.weekday() >= 5:
        return "weekend"
    if local.time() < MARKET_OPEN:
        return "pre_market"
    if local.time() < MARKET_CLOSE:
        return "open"
    return "after_close"


def _equity_curve_svg(points: List[EquityPoint], width: int = 560, height: int = 200) -> str:
    """Hand-rolled inline SVG, not a charting library -- no CDN, no network
    dependency, nothing that can silently fail to load on a machine with
    restricted egress (this dashboard may run on a headless server). The
    animated "draw-in" effect uses SVG's `pathLength` attribute to
    normalize the line to exactly 1000 units regardless of its actual
    pixel length, so a plain CSS stroke-dashoffset animation works without
    computing real path length in Python.
    """
    if len(points) < 2:
        return "<div class='chart-empty'>Building history…</div>"

    values = [p.equity for p in points]
    vmin, vmax = min(values), max(values)
    vrange = (vmax - vmin) or max(abs(vmin), 1.0) * 0.02  # a flat series still gets a visible band, not a div-by-zero
    pad_t, pad_b, pad_l, pad_r = 14, 10, 4, 4
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(values)

    def x(i: int) -> float:
        return pad_l + (i / (n - 1)) * plot_w

    def y(v: float) -> float:
        return pad_t + (1 - (v - vmin) / vrange) * plot_h

    coords = [(x(i), y(v)) for i, v in enumerate(values)]
    line_points = " ".join(f"{px:.1f},{py:.1f}" for px, py in coords)
    area_points = f"{coords[0][0]:.1f},{pad_t + plot_h:.1f} {line_points} {coords[-1][0]:.1f},{pad_t + plot_h:.1f}"
    gridlines = "".join(
        f'<line x1="{pad_l}" y1="{pad_t + f * plot_h:.1f}" x2="{width - pad_r}" y2="{pad_t + f * plot_h:.1f}" class="chart-grid"/>'
        for f in (0.0, 0.5, 1.0)
    )
    # Hover layer: a title-tooltip dot every few points (all of them, on a
    # typical few-hundred-point history, would be too dense to be useful).
    stride = max(1, n // 24)
    dots = "".join(
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" class="chart-hover-dot"><title>{points[i].timestamp[11:16]} — ${points[i].equity:,.2f}</title></circle>'
        for i, (px, py) in enumerate(coords) if i % stride == 0 or i == n - 1
    )

    return f"""<svg viewBox="0 0 {width} {height}" class="chart-svg" preserveAspectRatio="none">
      {gridlines}
      <polygon points="{area_points}" class="chart-area"/>
      <polyline points="{line_points}" class="chart-line" pathLength="1000"/>
      <circle cx="{coords[-1][0]:.1f}" cy="{coords[-1][1]:.1f}" r="3.5" class="chart-dot"/>
      {dots}
    </svg>"""


def _pnl_bar_svg(trades: List[ClosedTrade], width: int = 400, height: int = 200) -> str:
    """Diverging bar (above/below the zero baseline) for per-trade P&L --
    the form the dataviz skill calls for on a "delta to a baseline" job.
    Same no-library rationale as the equity chart above.
    """
    if not trades:
        return "<div class='chart-empty'>No closed trades yet</div>"

    values = [t.pnl_dollars for t in trades]
    vmax, vmin = max(max(values), 0.0), min(min(values), 0.0)
    vrange = (vmax - vmin) or 1.0
    pad_t, pad_b, pad_l, pad_r = 10, 10, 4, 4
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(values)
    slot_w = plot_w / n
    bar_w = min(slot_w * 0.55, 34)
    zero_y = pad_t + (1 - (0 - vmin) / vrange) * plot_h

    bars = []
    for i, (t, v) in enumerate(zip(trades, values)):
        bar_h = abs(v) / vrange * plot_h
        bx = pad_l + i * slot_w + (slot_w - bar_w) / 2
        by = zero_y - bar_h if v >= 0 else zero_y
        cls = "bar-pos" if v >= 0 else "bar-neg"
        title = f"{t.symbol} — {'+' if v >= 0 else ''}${v:,.2f} ({t.close_reason or 'closed'})"
        bars.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{max(bar_h, 1.5):.1f}" rx="3" '
            f'class="{cls}" style="animation-delay:{i * 0.06:.2f}s"><title>{title}</title></rect>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" class="chart-svg" preserveAspectRatio="none">
      <line x1="{pad_l}" y1="{zero_y:.1f}" x2="{width - pad_r}" y2="{zero_y:.1f}" class="chart-baseline"/>
      {''.join(bars)}
    </svg>"""


def _gauge_svg(score: Optional[float], size: int = 40, stroke: int = 4) -> str:
    """A small radial gauge -- used for a confluence score (live-reasoning
    cards) and for win rate (the Realized P&L stat card). Score expected in
    [0, 1]; clamped defensively since a caller could hand it a stray value.
    Color follows the status palette by band, not the categorical one --
    this is a state read (good/warning/critical), not a data series.
    """
    r = size / 2 - stroke
    if score is None:
        return (
            f"<svg viewBox='0 0 {size} {size}' class='gauge-svg' width='{size}' height='{size}'>"
            f"<circle cx='{size / 2}' cy='{size / 2}' r='{r}' class='gauge-track'/>"
            f"<text x='{size / 2}' y='{size / 2 + 4}' class='gauge-text gauge-text-na' text-anchor='middle'>&mdash;</text></svg>"
        )
    clamped = max(0.0, min(1.0, score))
    # pathLength="1000" normalizes the circle to exactly 1000 units of
    # length regardless of its actual radius -- same trick as
    # _equity_curve_svg's line, so the CSS stroke-dasharray/offset values
    # below are fixed constants, not something Python has to compute per
    # radius.
    offset = 1000 * (1 - clamped)
    band = "gauge-good" if clamped >= 0.8 else ("gauge-warn" if clamped >= 0.6 else "gauge-bad")
    return f"""<svg viewBox="0 0 {size} {size}" class="gauge-svg" width="{size}" height="{size}">
      <circle cx="{size / 2}" cy="{size / 2}" r="{r}" class="gauge-track"/>
      <circle cx="{size / 2}" cy="{size / 2}" r="{r}" class="gauge-fill {band}" pathLength="1000"
        style="--gauge-offset:{offset:.1f}" transform="rotate(-90 {size / 2} {size / 2})"/>
      <text x="{size / 2}" y="{size / 2 + 4}" class="gauge-text" text-anchor="middle">{clamped:.0%}</text>
    </svg>"""


def _sparkline_svg(values: List[float], width: int = 120, height: int = 32) -> str:
    """A minimal trend line for inside a stat card -- no axes, no grid, no
    hover layer (there's no room for a tooltip at this size, and the full
    equity chart above already provides one). Purely a "shape of the last
    stretch" cue, same drawn-in-line technique as _equity_curve_svg.
    """
    if len(values) < 2:
        return ""
    vmin, vmax = min(values), max(values)
    vrange = (vmax - vmin) or max(abs(vmin), 1.0) * 0.02
    pad = 2
    n = len(values)

    def x(i: int) -> float:
        return pad + (i / (n - 1)) * (width - 2 * pad)

    def y(v: float) -> float:
        return pad + (1 - (v - vmin) / vrange) * (height - 2 * pad)

    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    up = values[-1] >= values[0]
    cls = "spark-up" if up else "spark-down"
    return (
        f'<svg viewBox="0 0 {width} {height}" class="spark-svg" preserveAspectRatio="none">'
        f'<polyline points="{points}" class="spark-line {cls}" pathLength="1000"/>'
        f'<circle cx="{x(n - 1):.1f}" cy="{y(values[-1]):.1f}" r="2.2" class="spark-dot {cls}"/>'
        f"</svg>"
    )


def _news_html(items: List[NewsItem]) -> str:
    """Recent headlines for a symbol that actually has something to explain
    -- context for a human, see data/news.py's module docstring for why
    this never feeds any trading decision. Each headline links out to the
    source; time shown as HH:MM UTC, matching this page's other timestamps.
    """
    if not items:
        return ""
    rows = "".join(
        f"<a href='{n.url}' target='_blank' rel='noopener noreferrer' class='news-row'>"
        f"<span class='news-time'>{n.created_at.strftime('%H:%M')}</span>"
        f"<span class='news-headline'>{n.headline}</span>"
        f"<span class='news-source'>{n.source}</span></a>"
        for n in items
    )
    return f"<div class='rc-news'><div class='rc-news-label'>Related news</div>{rows}</div>"


def _reasoning_card(e: ActivityEntry, news_items: List[NewsItem]) -> str:
    """One symbol's full current read-out, not just `e.detail`'s one-line
    summary -- see the module docstring's "Live reasoning" note. Three
    shapes, depending on how far the decision pipeline actually got:
      1. An open/close event: there's no fresh-signal checklist to show
         (see options_execution.py's _open/_close -- those don't carry
         confluence_details), so just show what happened.
      2. No fresh FVG on the latest bar (the common case, ~80% of cycles):
         nothing to check yet.
      3. A gap formed: show it, and if confluence actually ran (gap AND
         volume both cleared), show the full 8-indicator checklist.
    News (when there is any) only shows for shapes 1 and 3 -- an actual
    signal or position, something worth explaining -- never for the quiet
    "no gap" case, where there's nothing for a headline to explain.
    """
    time_str = e.timestamp[11:19]
    price_str = f"${e.price:,.2f}" if e.price is not None else "&mdash;"
    header = f"<div class='rc-head'><span class='rc-symbol'>{e.symbol}</span><span class='rc-price'>{price_str}</span><span class='rc-time'>as of {time_str}</span></div>"

    if e.outcome in ("open", "close"):
        cls = "rc-status-open" if e.outcome == "open" else "rc-status-close"
        body = f"<div class='rc-line {cls}'>{e.detail}</div>"
        return f"<div class='reasoning-card'>{header}{body}{_news_html(news_items)}</div>"

    if e.gap_kind is None:
        body = "<div class='rc-line muted'>No fresh Fair Value Gap on the latest bar &mdash; waiting for one to form.</div>"
        return f"<div class='reasoning-card'>{header}{body}</div>"

    if not e.confluence_details:
        # Gap formed but volume didn't confirm it -- confluence never runs
        # without that, so there's no checklist yet, only the gap itself.
        body = f"<div class='rc-line muted'>{e.detail}</div>"
        return f"<div class='reasoning-card'>{header}{body}{_news_html(news_items)}</div>"

    chips = "".join(
        f"<div class='chk chk-{e.confluence_details.get(key, 'n/a').replace('/', '')}'>"
        f"<span class='chk-dot'></span><span class='chk-label'>{label}</span></div>"
        for key, label in CONFLUENCE_CHECK_LABELS
    )
    # Whenever confluence actually PASSES, options_execution.py opens a
    # position immediately (outcome becomes "open", caught by the branch
    # above) -- so reaching here with confluence_details populated and
    # outcome still "hold" means one of exactly two things: already holding
    # a matching position (fine, not a failure), or confluence blocked it.
    verdict_cls = "rc-status-holding" if "already holding" in e.detail else "rc-status-blocked"
    gauge = _gauge_svg(e.confluence_score, size=44)
    detail = (
        f"<div class='rc-detail'>"
        f"<div class='rc-gap'>{e.gap_kind} FVG, volume-confirmed &middot; {e.confluence_applicable} check(s)</div>"
        f"<div class='chk-grid'>{chips}</div>"
        f"<div class='rc-line {verdict_cls}'>{e.detail}</div>"
        f"</div>"
    )
    body = f"<div class='rc-body'><div class='rc-gauge'>{gauge}<span class='rc-gauge-label'>confluence</span></div>{detail}</div>"
    return f"<div class='reasoning-card'>{header}{body}{_news_html(news_items)}</div>"


def render_html(snapshot: DashboardSnapshot, refresh_seconds: int, account_label: str) -> str:
    pos_color = lambda x: "pos" if (x or 0) >= 0 else "neg"  # noqa: E731 -- small local helper, not worth a def

    # ---- positions / history tables (same data, restyled) ----

    stock_rows = "".join(
        f"<tr><td>{symbol}</td><td>{shares:g}</td></tr>"
        for symbol, shares in snapshot.stock_positions.items()
    ) or "<tr><td colspan='2' class='muted'>No open stock positions</td></tr>"

    def _fmt_spread(spread_pct: Optional[float]) -> str:
        return f"{spread_pct:.1%}" if spread_pct is not None else "&mdash;"

    def _fmt_bid_ask(bid: Optional[float], ask: Optional[float]) -> str:
        if bid is None or ask is None:
            return "&mdash;"
        return f"${bid:.2f}/${ask:.2f}"

    def _fmt_confluence(score: Optional[float], applicable: Optional[int]) -> str:
        if score is None:
            return "&mdash;"
        return f"{score:.0%} ({applicable or 0})"

    option_rows = "".join(
        f"<tr><td>{v.position.symbol}</td><td>{v.position.option_type}</td>"
        f"<td>${v.position.strike_price:.2f}</td><td>{v.position.expiration_date}</td>"
        f"<td>{v.position.quantity}</td>"
        f"<td>{_fmt_bid_ask(v.bid, v.ask)}</td><td>{_fmt_spread(v.spread_pct)}</td>"
        f"<td>{_fmt_money(v.position.average_premium_paid * v.position.quantity * 100)}</td>"
        f"<td>{_fmt_money(v.current_value)}</td>"
        f"<td class='{pos_color(v.pnl_dollars)}'>{_fmt_money(v.pnl_dollars)}</td>"
        f"<td class='{pos_color(v.pnl_pct)}'>{_fmt_pct(v.pnl_pct)}</td></tr>"
        for v in snapshot.option_positions
    ) or "<tr><td colspan='11' class='muted'>No open options positions</td></tr>"

    def _grade_badge(grade: Optional[TradeGrade]) -> str:
        if grade is None:
            return "<span class='grade grade-na' title='Not enough information to grade'>&mdash;</span>"
        cls = {"A": "grade-a", "B": "grade-b", "C": "grade-c", "D": "grade-d", "F": "grade-f"}.get(grade.letter, "grade-na")
        tooltip = f"{grade.process_label} / {grade.outcome_label} -- {grade.explanation}"
        return f"<span class='grade {cls}' title=\"{tooltip}\">{grade.letter}</span>"

    def _history_row(t: ClosedTrade, grade: Optional[TradeGrade]) -> str:
        sim_badge = " <span class='sim'>SIMULATED</span>" if t.dry_run else ""
        strike_str = f"${t.strike_price:.2f}" if t.strike_price is not None else "&mdash;"
        exp_str = f"{t.expiration_date} ({t.dte_at_entry}d)" if t.expiration_date else "&mdash;"
        return (
            f"<tr><td>{t.symbol}{sim_badge}</td><td>{t.trade_type}</td><td>{strike_str}</td><td>{exp_str}</td>"
            f"<td>{t.quantity}</td>"
            f"<td>{t.opened_at[:16].replace('T', ' ')}</td><td>{t.closed_at[:16].replace('T', ' ')}</td>"
            f"<td>{_fmt_spread(t.spread_pct)}</td>"
            f"<td>{_fmt_money(t.entry_notional)}</td><td>{_fmt_money(t.exit_notional)}</td>"
            f"<td class='{pos_color(t.pnl_dollars)}'>{_fmt_money(t.pnl_dollars)}</td>"
            f"<td class='{pos_color(t.pnl_pct)}'>{_fmt_pct(t.pnl_pct)}</td>"
            f"<td>{_fmt_confluence(t.confluence_score, t.confluence_applicable)}</td>"
            f"<td>{_grade_badge(grade)}</td>"
            f"<td>{t.close_reason or '&mdash;'}</td></tr>"
        )

    history_rows = "".join(
        _history_row(t, g) for t, g in zip(snapshot.trade_history, snapshot.trade_grades)
    ) or (
        "<tr><td colspan='15' class='muted'>No trades yet -- this fills in as the bot opens and closes positions "
        "(including dry-run trades, before live_trading_enabled is turned on)</td></tr>"
    )

    # ---- trade quality: per-check win-rate breakdown across closed trades
    # (orchestration/trade_grading.py) -- purely observational, see that
    # module's docstring for why this never writes back into config.

    _check_label_lookup = dict(CONFLUENCE_CHECK_LABELS)
    check_perf_rows = "".join(
        f"<tr><td>{_check_label_lookup.get(cp.check_key, cp.check_key)}</td>"
        f"<td>{_fmt_pct(cp.pass_win_rate) if cp.pass_win_rate is not None else '&mdash;'} <span class='muted'>({cp.pass_trades})</span></td>"
        f"<td>{_fmt_pct(cp.fail_win_rate) if cp.fail_win_rate is not None else '&mdash;'} <span class='muted'>({cp.fail_trades})</span></td></tr>"
        for cp in snapshot.check_performance
    )
    closed_for_grading = len(snapshot.trade_history)
    if check_perf_rows:
        check_quality_html = f"<table><tr><th>Check</th><th>Win rate when PASS</th><th>Win rate when FAIL</th></tr>{check_perf_rows}</table>"
    else:
        check_quality_html = (
            f"<div class='muted' style='padding:16px 0;'>Not enough closed trades yet to break down by check "
            f"(need {MIN_TRADES_FOR_AGGREGATE}, have {closed_for_grading}). This is observational only -- "
            f"even once populated, nothing here changes settings.yaml automatically.</div>"
        )

    closed_count = len(snapshot.trade_history)
    total_pnl = sum(t.pnl_dollars for t in snapshot.trade_history)
    wins = sum(1 for t in snapshot.trade_history if t.pnl_dollars > 0)
    win_rate = (wins / closed_count) if closed_count else None
    any_simulated = any(t.dry_run for t in snapshot.trade_history)

    # ---- today's narrative: pipeline stepper + activity feed ----

    phase = _market_phase(snapshot.fetched_at)
    phase_label = {
        "pre_market": "Before market open", "open": "Market open — actively polling",
        "after_close": "After market close", "weekend": "Weekend — markets closed",
    }[phase]
    steps = ["Market opens (9:30 ET)", "Poll every 5 min", "Evaluate FVG + confluence", "Execute or hold", "Market closes (4:00 ET)"]
    active_step = {"pre_market": 0, "open": 2, "after_close": 4, "weekend": 4}[phase]
    stepper_html = "".join(
        f"<div class='step {'active' if i == active_step else ('done' if i < active_step else '')}'>"
        f"<div class='step-dot'></div><div class='step-label'>{label}</div></div>"
        + ("<div class='step-line'></div>" if i < len(steps) - 1 else "")
        for i, label in enumerate(steps)
    )

    def _event_row(e: ActivityEntry) -> str:
        icon = {"open": "▲", "close": "●"}.get(e.outcome, "◈")
        cls = {"open": "evt-open", "close": "evt-close"}.get(e.outcome, "evt-nearmiss")
        time_str = e.timestamp[11:16]
        return f"<div class='event {cls}'><span class='evt-icon'>{icon}</span><span class='evt-time'>{time_str}</span><span class='evt-symbol'>{e.symbol}</span><span class='evt-detail'>{e.detail}</span></div>"

    events_html = "".join(_event_row(e) for e in snapshot.today_significant_events) or (
        "<div class='muted' style='padding:16px 0;'>No signals or trades yet today — quiet cycles are summarized below, not spammed here.</div>"
    )

    reasoning_html = "".join(
        _reasoning_card(e, snapshot.news_by_symbol.get(e.symbol, [])) for e in snapshot.live_reasoning
    ) or (
        "<div class='muted' style='padding:16px 0;'>No cycles logged yet -- this fills in once the bot's first cycle of the day runs.</div>"
    )

    quiet_html = "".join(
        f"<span class='quiet-pill'>{s.symbol}: {s.quiet_cycles} quiet"
        + (f", {s.near_misses} near-miss" if s.near_misses else "")
        + "</span>"
        for s in snapshot.today_symbol_summaries
    ) or "<span class='muted'>No cycles logged yet today.</span>"

    # ---- charts: equity curve + per-trade P&L (inline SVG, see the two
    # generator functions above -- no charting library, no CDN) ----

    equity_svg = _equity_curve_svg(snapshot.equity_history)
    pnl_svg = _pnl_bar_svg(list(reversed(snapshot.trade_history)))  # chronological, oldest first
    equity_sparkline_svg = _sparkline_svg([p.equity for p in snapshot.equity_history[-40:]])

    equity_delta = None
    if len(snapshot.equity_history) >= 2:
        equity_delta = snapshot.equity_history[-1].equity - snapshot.equity_history[0].equity
    equity_delta_pct = (equity_delta / snapshot.equity_history[0].equity) if equity_delta is not None and snapshot.equity_history[0].equity else None

    error_banner = (
        f"<div class='banner off'><div class='title'>Last refresh had an error</div><div>{snapshot.error}</div>"
        f"<div class='foot'>Showing the most recently successful data below (if any) -- this file keeps retrying every cycle.</div></div>"
        if snapshot.error else ""
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh_seconds}">
<title>Trading bot dashboard</title>
<style>
  :root {{
    --surface-1: #1a1a19; --page: #0d0d0d; --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --good: #0ca30c; --critical: #e66767; --warning: #fab219;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--page); color: var(--ink-1);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 28px; max-width: 1180px; margin: 0 auto;
  }}
  h1 {{ font-size: 21px; font-weight: 700; margin: 0; letter-spacing: -0.01em; }}
  .topbar {{ margin-bottom: 14px; }}
  .brand {{ display: flex; align-items: center; gap: 10px; }}
  .brand-mark {{
    width: 26px; height: 26px; border-radius: 7px; flex-shrink: 0;
    background: linear-gradient(135deg, var(--series-1), #1c5cab);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.08) inset, 0 2px 8px rgba(57,135,229,0.35);
  }}
  .subtitle {{ color: var(--ink-muted); font-size: 12.5px; margin: 8px 0 0; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  .subtitle .sep {{ opacity: 0.5; }}
  .ro-badge {{
    margin-left: 2px; font-size: 10.5px; color: var(--ink-muted); border: 1px solid var(--border);
    background: rgba(255,255,255,0.03); border-radius: 20px; padding: 2px 9px; letter-spacing: 0.01em;
  }}
  .live-dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--good); display: inline-block; box-shadow: 0 0 0 3px rgba(12,163,12,0.20); animation: pulse 2s ease-in-out infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
  .topnav {{
    position: sticky; top: 0; z-index: 10; display: flex; gap: 4px; flex-wrap: wrap;
    background: rgba(13,13,13,0.86); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--border); border-radius: 10px; padding: 6px; margin: 0 0 24px;
  }}
  .topnav a {{
    color: var(--ink-muted); text-decoration: none; font-size: 12px; font-weight: 600; padding: 6px 12px;
    border-radius: 7px; transition: background 0.15s, color 0.15s;
  }}
  .topnav a:hover {{ color: var(--ink-1); background: rgba(255,255,255,0.06); }}
  .section {{ margin-bottom: 30px; opacity: 0; animation: fadeUp 0.5s ease-out forwards; }}
  .section:nth-of-type(1) {{ animation-delay: 0.02s; }}
  .section:nth-of-type(2) {{ animation-delay: 0.08s; }}
  .section:nth-of-type(3) {{ animation-delay: 0.14s; }}
  .section:nth-of-type(4) {{ animation-delay: 0.20s; }}
  .section:nth-of-type(5) {{ animation-delay: 0.26s; }}
  .section:nth-of-type(6) {{ animation-delay: 0.32s; }}
  .section:nth-of-type(7) {{ animation-delay: 0.38s; }}
  .section:nth-of-type(8) {{ animation-delay: 0.44s; }}
  .section:nth-of-type(9) {{ animation-delay: 0.50s; }}
  @keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  .section-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--ink-muted); margin-bottom: 10px; font-weight: 600; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
  .card {{
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px;
    transition: border-color 0.2s, transform 0.2s; box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset;
  }}
  .card:hover {{ border-color: rgba(255,255,255,0.20); transform: translateY(-1px); }}
  .card .label {{ font-size: 12px; color: var(--ink-muted); margin-bottom: 6px; }}
  .card .value {{ font-size: 24px; font-weight: 700; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }}
  .card .delta {{ font-size: 12px; margin-top: 4px; font-weight: 500; }}
  .card-gauge {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; }}
  .card-gauge-text {{ min-width: 0; }}
  .pos {{ color: var(--good); }}
  .neg {{ color: var(--critical); }}
  .chart-row {{ display: grid; grid-template-columns: 1.4fr 1fr; gap: 14px; }}
  @media (max-width: 800px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
  .chart-card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }}
  .chart-card .chart-title {{ font-size: 13px; font-weight: 600; color: var(--ink-2); margin-bottom: 12px; }}
  .chart-wrap {{ height: 220px; position: relative; }}
  .chart-svg {{ width: 100%; height: 100%; overflow: visible; }}
  .chart-empty {{ color: var(--ink-muted); font-size: 13px; display: flex; align-items: center; justify-content: center; height: 100%; }}
  .chart-grid {{ stroke: var(--grid); stroke-width: 1; }}
  .chart-area {{ fill: rgba(57,135,229,0.14); opacity: 0; animation: fadeIn 0.8s ease-out 0.3s forwards; }}
  .chart-line {{
    fill: none; stroke: var(--series-1); stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round;
    stroke-dasharray: 1000; stroke-dashoffset: 1000; animation: drawLine 1.1s ease-out forwards;
  }}
  .chart-dot {{ fill: var(--series-1); opacity: 0; animation: fadeIn 0.4s ease-out 1s forwards; }}
  .chart-hover-dot {{ fill: transparent; cursor: pointer; }}
  .chart-hover-dot:hover {{ fill: rgba(57,135,229,0.25); }}
  .chart-baseline {{ stroke: var(--baseline); stroke-width: 1.5; }}
  .bar-pos, .bar-neg {{ transform-box: fill-box; transform-origin: center; opacity: 0; animation: barIn 0.5s ease-out forwards; }}
  .bar-pos {{ fill: rgba(12,163,12,0.78); }}
  .bar-neg {{ fill: rgba(230,103,103,0.78); }}
  @keyframes drawLine {{ to {{ stroke-dashoffset: 0; }} }}
  @keyframes fadeIn {{ to {{ opacity: 1; }} }}
  @keyframes barIn {{ from {{ opacity: 0; transform: scaleY(0.7); }} to {{ opacity: 1; transform: scaleY(1); }} }}
  .gauge-svg {{ overflow: visible; }}
  .gauge-track {{ fill: none; stroke: var(--baseline); stroke-width: 4; }}
  .gauge-fill {{ fill: none; stroke-width: 4; stroke-linecap: round; stroke-dasharray: 1000; stroke-dashoffset: 1000; animation: gaugeFill 0.9s ease-out 0.15s forwards; }}
  @keyframes gaugeFill {{ to {{ stroke-dashoffset: var(--gauge-offset); }} }}
  .gauge-good {{ stroke: var(--good); }}
  .gauge-warn {{ stroke: var(--warning); }}
  .gauge-bad {{ stroke: var(--critical); }}
  .gauge-text {{ font-size: 11px; font-weight: 700; fill: var(--ink-1); font-variant-numeric: tabular-nums; }}
  .gauge-text-na {{ fill: var(--ink-muted); font-weight: 500; }}
  .spark-svg {{ width: 100%; height: 32px; overflow: visible; display: block; margin-top: 6px; }}
  .spark-line {{ fill: none; stroke-width: 1.8; stroke-linejoin: round; stroke-linecap: round; stroke-dasharray: 1000; stroke-dashoffset: 1000; animation: drawLine 0.9s ease-out 0.2s forwards; }}
  .spark-up {{ stroke: var(--good); }}
  .spark-down {{ stroke: var(--critical); }}
  .spark-dot {{ opacity: 0; animation: fadeIn 0.4s ease-out 1s forwards; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: right; color: var(--ink-muted); font-weight: 500; padding: 9px 8px; border-bottom: 1px solid var(--baseline); font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em; }}
  th:first-child, td:first-child {{ text-align: left; }}
  td {{ padding: 9px 8px; border-bottom: 1px solid var(--grid); text-align: right; font-variant-numeric: tabular-nums; }}
  tr:last-child td {{ border-bottom: none; }}
  tr {{ transition: background 0.15s; }}
  tbody tr:hover {{ background: rgba(255,255,255,0.03); }}
  .muted {{ color: var(--ink-muted); text-align: center; }}
  .banner {{ border-radius: 10px; padding: 14px 16px; border: 1px solid; margin-bottom: 20px; }}
  .banner.off {{ background: rgba(230,103,103,0.12); border-color: rgba(230,103,103,0.35); }}
  .banner .title {{ font-weight: 600; margin-bottom: 6px; }}
  .banner .foot {{ margin-top: 6px; font-size: 12px; color: var(--ink-muted); }}
  .sim {{ font-size: 10px; color: var(--warning); border: 1px solid rgba(250,178,25,0.35); background: rgba(250,178,25,0.12); border-radius: 5px; padding: 1px 6px; font-weight: 600; }}
  .grade {{ display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 6px; font-size: 12px; font-weight: 700; cursor: help; }}
  .grade-a {{ color: var(--good); background: rgba(12,163,12,0.14); border: 1px solid rgba(12,163,12,0.35); }}
  .grade-b {{ color: var(--good); background: rgba(12,163,12,0.08); border: 1px solid rgba(12,163,12,0.22); }}
  .grade-c {{ color: var(--warning); background: rgba(250,178,25,0.12); border: 1px solid rgba(250,178,25,0.35); }}
  .grade-d {{ color: var(--warning); background: rgba(250,178,25,0.08); border: 1px solid rgba(250,178,25,0.22); }}
  .grade-f {{ color: var(--critical); background: rgba(230,103,103,0.12); border: 1px solid rgba(230,103,103,0.35); }}
  .grade-na {{ color: var(--ink-muted); background: rgba(255,255,255,0.04); border: 1px solid var(--border); }}
  .table-card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 4px 18px 6px; overflow-x: auto; }}

  /* pipeline stepper */
  .stepper {{ display: flex; align-items: flex-start; background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 22px 24px 18px; }}
  .step {{ display: flex; flex-direction: column; align-items: center; gap: 8px; flex: 0 0 auto; width: 108px; }}
  .step-dot {{ width: 13px; height: 13px; border-radius: 50%; background: var(--baseline); border: 2px solid var(--baseline); }}
  .step.done .step-dot {{ background: var(--good); border-color: var(--good); }}
  .step.active .step-dot {{ background: var(--series-1); border-color: var(--series-1); box-shadow: 0 0 0 4px rgba(57,135,229,0.22); animation: pulse 1.6s ease-in-out infinite; }}
  .step-label {{ font-size: 11.5px; color: var(--ink-muted); text-align: center; line-height: 1.3; }}
  .step.active .step-label {{ color: var(--ink-1); font-weight: 600; }}
  .step-line {{ flex: 1 1 auto; height: 2px; background: var(--baseline); margin-top: 6px; min-width: 16px; }}
  .phase-badge {{ font-size: 12px; color: var(--ink-2); margin-top: 16px; text-align: center; }}

  /* live reasoning panel */
  .reasoning-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }}
  .reasoning-card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; transition: border-color 0.2s; }}
  .reasoning-card:hover {{ border-color: rgba(255,255,255,0.20); }}
  .rc-head {{ display: flex; align-items: baseline; gap: 8px; margin-bottom: 8px; }}
  .rc-symbol {{ font-weight: 700; font-size: 14px; }}
  .rc-price {{ font-variant-numeric: tabular-nums; color: var(--ink-2); font-size: 13px; }}
  .rc-time {{ margin-left: auto; color: var(--ink-muted); font-size: 11px; }}
  .rc-line {{ font-size: 12.5px; color: var(--ink-2); line-height: 1.45; }}
  .rc-line.muted {{ color: var(--ink-muted); }}
  .rc-status-open {{ color: var(--good); }}
  .rc-status-close {{ color: var(--ink-2); }}
  .rc-status-holding {{ color: var(--series-1); }}
  .rc-status-blocked {{ color: var(--warning); }}
  .rc-body {{ display: flex; gap: 12px; align-items: flex-start; }}
  .rc-gauge {{ display: flex; flex-direction: column; align-items: center; gap: 3px; flex-shrink: 0; padding-top: 2px; }}
  .rc-gauge-label {{ font-size: 8.5px; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .rc-detail {{ flex: 1; min-width: 0; }}
  .rc-gap {{ font-size: 11px; color: var(--ink-muted); margin-bottom: 8px; }}
  .chk-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 5px 10px; margin-bottom: 10px; }}
  .chk {{ display: flex; align-items: center; gap: 6px; font-size: 11.5px; }}
  .chk-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
  .chk-pass .chk-dot {{ background: var(--good); }}
  .chk-pass .chk-label {{ color: var(--ink-2); }}
  .chk-fail .chk-dot {{ background: var(--critical); }}
  .chk-fail .chk-label {{ color: var(--ink-2); }}
  .chk-na .chk-dot {{ background: var(--baseline); }}
  .chk-na .chk-label {{ color: var(--ink-muted); }}
  .rc-news {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--grid); }}
  .rc-news-label {{ font-size: 9.5px; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
  .news-row {{ display: flex; align-items: baseline; gap: 7px; padding: 3px 0; font-size: 11.5px; text-decoration: none; color: inherit; }}
  .news-time {{ color: var(--ink-muted); font-variant-numeric: tabular-nums; flex-shrink: 0; }}
  .news-headline {{ color: var(--ink-2); flex: 1; line-height: 1.35; }}
  .news-row:hover .news-headline {{ color: var(--series-1); text-decoration: underline; }}
  .news-source {{ color: var(--ink-muted); flex-shrink: 0; font-size: 10.5px; text-transform: uppercase; }}

  /* today's activity feed */
  .feed-card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 6px 18px 14px; }}
  .event {{ display: flex; align-items: center; gap: 10px; padding: 9px 0; border-bottom: 1px solid var(--grid); font-size: 13px; }}
  .event:last-child {{ border-bottom: none; }}
  .evt-icon {{ font-size: 10px; width: 14px; text-align: center; }}
  .evt-open .evt-icon {{ color: var(--good); }}
  .evt-close .evt-icon {{ color: var(--ink-2); }}
  .evt-nearmiss .evt-icon {{ color: var(--warning); }}
  .evt-time {{ color: var(--ink-muted); font-variant-numeric: tabular-nums; width: 44px; flex-shrink: 0; }}
  .evt-symbol {{ font-weight: 600; width: 52px; flex-shrink: 0; }}
  .evt-detail {{ color: var(--ink-2); flex: 1; }}
  .quiet-pills {{ margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }}
  .quiet-pill {{ font-size: 11.5px; color: var(--ink-muted); background: rgba(255,255,255,0.04); border: 1px solid var(--border); border-radius: 20px; padding: 4px 10px; }}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand"><span class="brand-mark"></span><h1>Trading bot dashboard</h1></div>
  <div class="subtitle">
    <span class="live-dot"></span>
    <span>{account_label}</span>
    <span class="sep">&middot;</span>
    <span>refreshes every {refresh_seconds}s</span>
    <span class="sep">&middot;</span>
    <span>updated {snapshot.fetched_at.strftime('%H:%M:%S UTC')}</span>
    <span class="ro-badge">read-only &middot; places no orders</span>
  </div>
</header>
<nav class="topnav">
  <a href="#account">Account</a><a href="#performance">Performance</a><a href="#reasoning">Reasoning</a>
  <a href="#activity">Activity</a><a href="#history">History</a><a href="#quality">Quality</a><a href="#positions">Positions</a>
</nav>
{error_banner}

<div class="section" id="account">
  <div class="section-label">Account</div>
  <div class="cards">
    <div class="card">
      <div class="label">Equity</div>
      <div class="value">{_fmt_money(snapshot.equity)}</div>
      {f"<div class='delta {pos_color(equity_delta)}'>{_fmt_money(equity_delta)} ({_fmt_pct(equity_delta_pct)}) since dashboard start</div>" if equity_delta is not None else "<div class='delta muted'>Building history…</div>"}
      {equity_sparkline_svg}
    </div>
    <div class="card"><div class="label">Buying power</div><div class="value">{_fmt_money(snapshot.buying_power)}</div></div>
    <div class="card"><div class="label">Open orders</div><div class="value">{snapshot.open_order_count}</div></div>
    <div class="card card-gauge">
      <div class="card-gauge-text">
        <div class="label">Realized P&amp;L{" (sim)" if any_simulated else ""}</div>
        <div class="value {pos_color(total_pnl) if closed_count else ''}">{_fmt_money(total_pnl) if closed_count else "&mdash;"}</div>
        <div class="delta muted">{closed_count} closed trade{'s' if closed_count != 1 else ''}</div>
      </div>
      {_gauge_svg(win_rate, size=48)}
    </div>
  </div>
</div>

<div class="section" id="performance">
  <div class="section-label">Performance</div>
  <div class="chart-row">
    <div class="chart-card">
      <div class="chart-title">Account equity</div>
      <div class="chart-wrap">{equity_svg}</div>
    </div>
    <div class="chart-card">
      <div class="chart-title">P&amp;L per closed trade</div>
      <div class="chart-wrap">{pnl_svg}</div>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-label">Today's session</div>
  <div class="stepper">
    {stepper_html}
  </div>
  <div class="phase-badge">{phase_label}</div>
</div>

<div class="section" id="reasoning">
  <div class="section-label">Live reasoning &mdash; what it's looking at right now</div>
  <div class="reasoning-grid">
    {reasoning_html}
  </div>
</div>

<div class="section" id="activity">
  <div class="section-label">Today's activity</div>
  <div class="feed-card">
    {events_html}
  </div>
  <div class="quiet-pills">{quiet_html}</div>
</div>

<div class="section" id="history">
  <div class="section-label">Trade history</div>
  <div class="table-card">
    <table>
      <tr><th>Symbol</th><th>Type</th><th>Strike</th><th>Expiration</th><th>Qty</th><th>Opened</th><th>Closed</th><th>Entry spread</th><th>Put in</th><th>Exit value</th><th>P&amp;L $</th><th>P&amp;L %</th><th>Confluence</th><th>Grade</th><th>Reason</th></tr>
      {history_rows}
    </table>
  </div>
</div>

<div class="section" id="quality">
  <div class="section-label">Trade quality &mdash; which checks actually correlate with a win</div>
  <div class="table-card">
    {check_quality_html}
  </div>
</div>

<div class="section" id="positions">
  <div class="section-label">Open options positions</div>
  <div class="table-card">
    <table>
      <tr><th>Symbol</th><th>Type</th><th>Strike</th><th>Expiration</th><th>Qty</th><th>Bid/ask</th><th>Spread</th><th>Cost basis</th><th>Current value</th><th>P&amp;L $</th><th>P&amp;L %</th></tr>
      {option_rows}
    </table>
  </div>
</div>

<div class="section">
  <div class="section-label">Open stock positions</div>
  <div class="table-card">
    <table>
      <tr><th>Symbol</th><th>Shares</th></tr>
      {stock_rows}
    </table>
  </div>
</div>

</body>
</html>"""


def run_forever(
    output_path: Path = DEFAULT_OUTPUT_PATH, refresh_seconds: int = DEFAULT_REFRESH_SECONDS, settings_path: str = "config/settings.yaml",
) -> None:
    # Settings loaded here only for stop_loss_pct/take_profit_pct, which
    # orchestration/trade_grading.py needs to bucket a closed trade's P&L
    # into "clean win"/"small loss"/etc against THIS account's actual
    # thresholds, not an arbitrary number picked in this file. Everything
    # else this dashboard shows still comes straight from the broker/logs,
    # not from settings -- it stays a read-only monitor either way.
    from config.config_loader import load_settings
    settings = load_settings(settings_path)

    broker = RobinhoodBroker()
    broker.connect()
    broker.verify_account()
    chain_fetcher = RobinhoodOptionChainFetcher()
    account_label = f"account ...{broker.account_number[-4:]}"

    # Optional -- a missing/invalid Alpaca key shouldn't take down the whole
    # dashboard (it's already required for the live trading loop itself, but
    # this process runs independently and should degrade gracefully rather
    # than crash on startup over a context-only feature).
    try:
        news_fetcher: Optional[AlpacaNewsFetcher] = AlpacaNewsFetcher()
    except Exception:
        logger.exception("Could not set up the news fetcher -- dashboard will run without it")
        news_fetcher = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %s every %ds. Open it in a browser and leave the tab open.", output_path, refresh_seconds)

    last_good: Optional[DashboardSnapshot] = None
    while True:
        try:
            snapshot = fetch_snapshot(broker, chain_fetcher, settings.options.stop_loss_pct, settings.options.take_profit_pct, news_fetcher)
            last_good = snapshot
        except Exception as exc:
            logger.exception("Dashboard refresh failed this cycle -- will retry next cycle")
            if last_good is None:
                time.sleep(refresh_seconds)
                continue
            snapshot = last_good
            snapshot.error = f"{type(exc).__name__}: {exc}"

        output_path.write_text(render_html(snapshot, refresh_seconds, account_label), encoding="utf-8")
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
    args = parser.parse_args()
    run_forever(Path(args.output), args.refresh_seconds, args.settings)
