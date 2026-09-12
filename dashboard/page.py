"""HTML rendering for the dashboard (dashboard/live_account_dashboard.py owns
the data; this file owns the page). Inline CSS + hand-rolled SVG, no
charting library, no CDN scripts -- the page has to render from a local
file on a machine with restricted egress and inside a Claude artifact's
CSP alike. Google Fonts is the one external resource, with a full system
fallback stack so the page is still fine offline.

Every timestamp on the page is shown in ET (America/New_York): that is the
timezone the bot's session rules are written in, and the routine logs UTC.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from orchestration.activity_log import ActivityEntry
from orchestration.market_hours import MARKET_TZ
from orchestration.trade_grading import MIN_TRADES_FOR_AGGREGATE, TradeGrade
from orchestration.trade_log import ClosedTrade

CHECK_LABELS = {
    "trend_1h": "1h trend", "trend_4h": "4h trend", "market_structure": "5m structure",
    "trend_200sma": "Daily 200-SMA", "volume_profile": "Volume profile", "support_resistance": "Support / resistance",
    "break_of_structure": "Break of structure", "supply_demand": "Supply / demand", "liquidity_sweep": "Liquidity sweep",
    "rsi_momentum": "RSI momentum", "volatility_expansion": "Volatility expansion", "vpvr_node_quality": "VPVR node",
    "elliott_wave": "Elliott wave",
}


# --------------------------------------------------------------------------
# formatting

def money(x: Optional[float]) -> str:
    if x is None:
        return "&mdash;"
    return f"{'-' if x < 0 else ''}${abs(x):,.2f}"


def pct(x: Optional[float], signed: bool = True, digits: int = 1) -> str:
    if x is None:
        return "&mdash;"
    sign = "+" if (signed and x >= 0) else ""
    return f"{sign}{x * 100:.{digits}f}%"


def et(ts, fmt: str = "%-I:%M %p") -> str:
    """ISO string or datetime -> ET clock string."""
    if ts is None:
        return "&mdash;"
    d = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    d = d.astimezone(MARKET_TZ)
    # portable "no leading zero" (%-d / %-I are glibc-only, not Windows)
    fmt = fmt.replace("%-d", str(d.day)).replace("%-I", str(int(d.strftime("%I"))))
    return d.strftime(fmt)


def et_full(ts) -> str:
    return et(ts, "%a %b %-d, %-I:%M %p")


def sign_cls(x: Optional[float]) -> str:
    if x is None:
        return ""
    return "up" if x > 0 else ("down" if x < 0 else "")


# --------------------------------------------------------------------------
# SVG pieces

def sparkline(values: Sequence[float], width: int = 160, height: int = 36, cls: str = "") -> str:
    if len(values) < 2:
        return ""
    vmin, vmax = min(values), max(values)
    vr = (vmax - vmin) or max(abs(vmin), 1.0) * 0.005
    n = len(values)
    pad = 2
    pts = [(pad + i / (n - 1) * (width - 2 * pad), pad + (1 - (v - vmin) / vr) * (height - 2 * pad)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{pts[0][0]:.1f},{height} {line} {pts[-1][0]:.1f},{height}"
    tone = cls or ("up" if values[-1] >= values[0] else "down")
    return (
        f'<svg viewBox="0 0 {width} {height}" class="spark {tone}" preserveAspectRatio="none" aria-hidden="true">'
        f'<polygon points="{area}" class="spark-area"/><polyline points="{line}" class="spark-line"/>'
        f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="2.4" class="spark-dot"/></svg>'
    )


def equity_chart(points, width: int = 640, height: int = 220) -> str:
    if len(points) < 2:
        return "<div class='empty'>Equity history starts with the first bot cycle.</div>"
    values = [p.equity for p in points]
    vmin, vmax = min(values), max(values)
    vr = (vmax - vmin) or max(abs(vmin), 1.0) * 0.01
    lo, hi = vmin - vr * 0.15, vmax + vr * 0.15
    pad_l, pad_r, pad_t, pad_b = 56, 12, 12, 26
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(values)
    x = lambda i: pad_l + i / (n - 1) * pw  # noqa: E731
    y = lambda v: pad_t + (1 - (v - lo) / (hi - lo)) * ph  # noqa: E731
    pts = [(x(i), y(v)) for i, v in enumerate(values)]
    line = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    area = f"{pts[0][0]:.1f},{pad_t + ph:.1f} {line} {pts[-1][0]:.1f},{pad_t + ph:.1f}"
    grid = ""
    for f in (0.0, 0.5, 1.0):
        v = hi - f * (hi - lo)
        gy = pad_t + f * ph
        grid += f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" class="gl"/>'
        grid += f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" class="axis" text-anchor="end">${v:,.0f}</text>'
    labels = ""
    for i in sorted({0, n // 2, n - 1}):
        labels += f'<text x="{x(i):.1f}" y="{height - 8}" class="axis" text-anchor="{"start" if i == 0 else ("end" if i == n - 1 else "middle")}">{et(points[i].timestamp, "%b %-d")}</text>'
    stride = max(1, n // 30)
    dots = "".join(
        f'<circle cx="{a:.1f}" cy="{b:.1f}" r="7" class="hit"><title>{et_full(points[i].timestamp)} ET · ${points[i].equity:,.2f}</title></circle>'
        for i, (a, b) in enumerate(pts) if i % stride == 0 or i == n - 1
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart" preserveAspectRatio="none">{grid}'
        f'<polygon points="{area}" class="area"/><polyline points="{line}" class="line"/>'
        f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="3.5" class="end"/>{labels}{dots}</svg>'
    )


def pnl_chart(trades: List[ClosedTrade], width: int = 420, height: int = 220) -> str:
    if not trades:
        return "<div class='empty'>No closed trades yet.</div>"
    values = [t.pnl_dollars for t in trades]
    vmax, vmin = max(max(values), 0.0), min(min(values), 0.0)
    vr = (vmax - vmin) or 1.0
    pad_l, pad_r, pad_t, pad_b = 48, 8, 12, 10
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(values)
    slot = pw / n
    bw = min(slot * 0.6, 28)
    zero = pad_t + (1 - (0 - vmin) / vr) * ph
    bars = ""
    for i, (t, v) in enumerate(zip(trades, values)):
        h = abs(v) / vr * ph
        bx = pad_l + i * slot + (slot - bw) / 2
        by = zero - h if v >= 0 else zero
        bars += (
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{max(h, 1.5):.1f}" rx="2" class="bar {"up" if v >= 0 else "down"}">'
            f"<title>{t.symbol} {t.trade_type} · {'+' if v >= 0 else ''}${v:,.2f} · {t.close_reason or 'closed'}</title></rect>"
        )
    axis = "".join(
        f'<text x="{pad_l - 8}" y="{pad_t + f * ph + 4:.1f}" class="axis" text-anchor="end">${vmax - f * vr:,.0f}</text>'
        for f in (0.0, 1.0)
    )
    return f'<svg viewBox="0 0 {width} {height}" class="chart" preserveAspectRatio="none"><line x1="{pad_l}" y1="{zero:.1f}" x2="{width - pad_r}" y2="{zero:.1f}" class="zero"/>{axis}{bars}</svg>'


def timeline_svg(cycle_times: List[datetime], now_et: datetime, width: int = 900, height: int = 54) -> str:
    """Today's session on one axis, 9:00-17:30 ET: shaded entry window,
    the bot's scheduled cadence as faint ticks, every logged cycle as a dot."""
    start, end = 9.0, 17.5
    pad_l, pad_r = 24, 24
    pw = width - pad_l - pad_r
    x = lambda h: pad_l + (h - start) / (end - start) * pw  # noqa: E731
    base_y = 30
    out = [f'<rect x="{x(9.5):.1f}" y="10" width="{x(10.5) - x(9.5):.1f}" height="28" class="tl-window"/>']
    out.append(f'<line x1="{pad_l}" y1="{base_y}" x2="{width - pad_r}" y2="{base_y}" class="tl-axis"/>')
    for h in (9.5, 10.5, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0):
        label = {9.5: "9:30", 10.5: "10:30", 11.0: "11", 12.0: "12", 13.0: "1", 14.0: "2", 15.0: "3", 16.0: "4:00", 17.0: "5"}[h]
        out.append(f'<line x1="{x(h):.1f}" y1="{base_y - 4}" x2="{x(h):.1f}" y2="{base_y + 4}" class="tl-tick"/>')
        out.append(f'<text x="{x(h):.1f}" y="{height - 4}" class="axis" text-anchor="middle">{label}</text>')
    sched = [9 + 36 / 60 + 5 * i / 60 for i in range(12)] + [10 + 41 / 60, 10 + 51 / 60, 11 + 1 / 60] + [h + 7 / 60 for h in range(11, 18)]
    for h in sched:
        out.append(f'<circle cx="{x(h):.1f}" cy="{base_y}" r="1.6" class="tl-sched"/>')
    for t in cycle_times:
        h = t.hour + t.minute / 60
        if start <= h <= end:
            out.append(f'<circle cx="{x(h):.1f}" cy="{base_y}" r="4" class="tl-cycle"><title>cycle at {t.strftime("%I:%M %p").lstrip("0")} ET</title></circle>')
    h_now = now_et.hour + now_et.minute / 60
    if now_et.weekday() < 5 and start <= h_now <= end:
        out.append(f'<line x1="{x(h_now):.1f}" y1="8" x2="{x(h_now):.1f}" y2="{base_y + 10}" class="tl-now"/>')
    return f'<svg viewBox="0 0 {width} {height}" class="timeline" preserveAspectRatio="none">{"".join(out)}</svg>'


# --------------------------------------------------------------------------
# blocks

def _pill(text: str, tone: str = "") -> str:
    return f"<span class='pill {tone}'>{text}</span>"


def _trend_pill(label: str, direction: Optional[str]) -> str:
    tone = {"bullish": "up", "bearish": "down"}.get(direction or "", "flat")
    glyph = {"bullish": "▲", "bearish": "▼"}.get(direction or "", "–")
    return f"<span class='trend {tone}'><span class='trend-label'>{label}</span><span class='trend-val'>{glyph} {direction or 'n/a'}</span></span>"


def structure_card(s, last_read: Optional[ActivityEntry], hard_keys: Sequence[str]) -> str:
    if s.error:
        return f"<article class='sym'><header class='sym-head'><span class='sym-name'>{s.symbol}</span></header><div class='empty'>{html.escape(s.error)}</div></article>"

    if s.permitted == ["calls", "puts"]:
        verdict, tone = "calls or puts allowed", "up"
    elif s.permitted:
        verdict, tone = f"{s.permitted[0]} only", "up" if s.permitted[0] == "calls" else "down"
    else:
        verdict, tone = "no entry allowed", "flat"

    veto_lines = ""
    for side in ("calls", "puts"):
        r = s.veto_reason.get(side)
        checks = s.hard_checks.get(side, {})
        chips = "".join(
            f"<span class='chk {v}' title='{CHECK_LABELS.get(k, k)}: {v}'>{CHECK_LABELS.get(k, k)}</span>" for k, v in checks.items()
        )
        status = f"<span class='veto'>{html.escape(r)}</span>" if r else "<span class='ok'>all hard checks agree</span>"
        veto_lines += f"<div class='side'><span class='side-name'>{side}</span><span class='side-chips'>{chips}</span>{status}</div>"

    levels = "".join(
        f"<li><span class='lv-name'>{l.label}</span><span class='lv-price'>${l.price:,.2f}</span>"
        f"<span class='lv-dist {'up' if l.distance_pct >= 0 else 'down'}'>{pct(l.distance_pct, digits=2)}</span></li>"
        for l in s.levels
    ) or "<li class='dim'>no major levels yet</li>"

    regime = "".join(
        f"<span class='rg {x.structure} {'fvg' if x.fvg_in_window else ''}' title='{x.session.strftime('%a %b %d')} · 10:30 read: {x.structure} · day {pct(x.day_move_pct)}{' · FVG formed in window' if x.fvg_in_window else ''}'></span>"
        for x in s.regimes
    )
    fvg = "no qualifying gap in cache"
    if s.last_fvg:
        kind, ago, vol = s.last_fvg
        fvg = f"{kind} gap {ago} bar{'s' if ago != 1 else ''} ago{' · volume confirmed' if vol else ' · volume not confirmed'}"

    range_txt = "&mdash;"
    if s.session_range is not None and s.atr_daily:
        range_txt = f"${s.session_range:,.2f} <span class='dim'>({s.session_range_vs_atr:.0%} of daily ATR ${s.atr_daily:,.2f})</span>"

    read = ""
    if last_read is not None:
        read = f"<div class='last-read'><span class='dim'>bot's last read · {et_full(last_read.timestamp)} ET</span><span>{html.escape(last_read.detail)}</span></div>"

    return f"""<article class="sym">
  <header class="sym-head">
    <span class="sym-name">{s.symbol}</span>
    <span class="sym-price">${s.price:,.2f}</span>
    <span class="sym-chg {sign_cls(s.session_change_pct)}">{pct(s.session_change_pct)} <span class="dim">session</span></span>
    <span class="sym-spark">{sparkline(s.session_closes)}</span>
  </header>
  <div class="sym-verdict {tone}">{verdict}</div>
  <div class="trends">{_trend_pill("1h", s.trend_1h)}{_trend_pill("4h", s.trend_4h)}{_trend_pill("daily", s.trend_daily)}
    <span class="trend {'up' if s.structure_5m == 'uptrend' else ('down' if s.structure_5m == 'downtrend' else 'flat')}"><span class="trend-label">5m structure</span><span class="trend-val">{s.structure_5m}{' · BOS ' + s.bos if s.bos else ''}</span></span>
  </div>
  <div class="sides">{veto_lines}</div>
  <dl class="facts">
    <div><dt>Session range</dt><dd>{range_txt}</dd></div>
    <div><dt>Last swing</dt><dd>{'high $' + format(s.last_swing_high, ',.2f') if s.last_swing_high else '&mdash;'} <span class='dim'>/</span> {'low $' + format(s.last_swing_low, ',.2f') if s.last_swing_low else '&mdash;'}</dd></div>
    <div><dt>Latest FVG</dt><dd>{fvg}</dd></div>
  </dl>
  <div class="levels"><div class="eyebrow">Liquidity levels, nearest first</div><ul>{levels}</ul></div>
  <div class="regime"><div class="eyebrow">10:30 ET structure read, last {len(s.regimes)} sessions</div><div class="rg-row">{regime}</div></div>
  {read}
</article>"""


def grade_badge(grade: Optional[TradeGrade]) -> str:
    if grade is None:
        return "<span class='grade na'>&mdash;</span>"
    return f"<span class='grade g{grade.letter.lower()}' title=\"{html.escape(grade.process_label)} / {html.escape(grade.outcome_label)} — {html.escape(grade.explanation)}\">{grade.letter}</span>"


# --------------------------------------------------------------------------
# page

def render_html(snapshot, refresh_seconds: int, account_label: str) -> str:
    strat = snapshot.strategy or {}
    live = strat.get("live_trading_enabled") == "true"
    now_et = snapshot.fetched_at.astimezone(MARKET_TZ)
    hard_keys = tuple(snapshot.policy_hard_keys or ())
    soft_keys = tuple(snapshot.policy_soft_keys or ())

    # freshness: the account snapshot's timestamp is the last bot cycle
    data_as_of = snapshot.data_as_of
    freshness = f"last bot cycle {et_full(data_as_of)} ET" if data_as_of else "no bot cycle recorded yet"
    bars_as_of = max((s.as_of for s in snapshot.structure if not s.error), default=None)
    bars_txt = f"bars through {et_full(bars_as_of)} ET" if bars_as_of else ""

    if now_et.weekday() >= 5:
        phase = "Markets closed for the weekend"
    elif now_et.time() < datetime.strptime("09:30", "%H:%M").time():
        phase = "Pre-market"
    elif now_et.time() < datetime.strptime("10:30", "%H:%M").time():
        phase = "Entry window open"
    elif now_et.time() < datetime.strptime("16:00", "%H:%M").time():
        phase = "Entry window closed · managing exits"
    else:
        phase = "After the close"

    # ---- overview tiles
    closed = snapshot.trade_history
    n_closed = len(closed)
    total_pnl = sum(t.pnl_dollars for t in closed)
    wins = sum(1 for t in closed if t.pnl_dollars > 0)
    eq_hist = snapshot.equity_history
    eq_delta = (eq_hist[-1].equity - eq_hist[0].equity) if len(eq_hist) >= 2 else None
    eq_delta_pct = (eq_delta / eq_hist[0].equity) if (eq_delta is not None and eq_hist[0].equity) else None
    eq_since = et(eq_hist[0].timestamp, "%b %-d") if eq_hist else ""
    cycles_today = sorted({datetime.fromisoformat(e.timestamp).astimezone(MARKET_TZ) for e in snapshot.today_entries})
    permitted_syms = [s for s in snapshot.structure if s.permitted]

    tiles = f"""
<div class="tiles">
  <div class="tile">
    <div class="t-label">Equity</div><div class="t-value">{money(snapshot.equity)}</div>
    <div class="t-sub {sign_cls(eq_delta)}">{money(eq_delta) if eq_delta is not None else '&mdash;'} ({pct(eq_delta_pct)}) since {eq_since or 'start'}</div>
    {sparkline([p.equity for p in eq_hist[-60:]], 200, 30)}
  </div>
  <div class="tile"><div class="t-label">Buying power</div><div class="t-value">{money(snapshot.buying_power)}</div><div class="t-sub dim">{snapshot.open_order_count} open order{'s' if snapshot.open_order_count != 1 else ''}</div></div>
  <div class="tile"><div class="t-label">Open positions</div><div class="t-value">{len(snapshot.option_positions)}</div><div class="t-sub dim">of {strat.get('max_open', '3')} allowed</div></div>
  <div class="tile"><div class="t-label">Realized P&amp;L{' · simulated' if any(t.dry_run for t in closed) else ''}</div><div class="t-value {sign_cls(total_pnl) if n_closed else ''}">{money(total_pnl) if n_closed else '&mdash;'}</div><div class="t-sub dim">{n_closed} closed · {f'{wins / n_closed:.0%} won' if n_closed else 'no wins or losses yet'}</div></div>
  <div class="tile"><div class="t-label">Right now</div><div class="t-value small">{len(permitted_syms)}<span class="dim"> / {len(snapshot.structure)}</span></div><div class="t-sub dim">symbols where hard checks allow an entry</div></div>
</div>"""

    # ---- strategy strip
    chips = "".join(f"<span class='chip'><b>{html.escape(k)}</b>{html.escape(str(v))}</span>" for k, v in strat.items() if k not in ("live_trading_enabled", "max_open"))

    # ---- structure grid
    last_reads = {e.symbol: e for e in snapshot.live_reasoning}
    cards = "".join(structure_card(s, last_reads.get(s.symbol), hard_keys) for s in snapshot.structure) or "<div class='empty'>No cached bars yet — the first bot cycle fills market_data/.</div>"
    n_fvg_sessions = sum(1 for s in snapshot.structure for r in s.regimes if r.fvg_in_window)
    n_sessions = sum(len(s.regimes) for s in snapshot.structure)
    struct_summary = (
        f"Hard checks ({', '.join(CHECK_LABELS.get(k, k) for k in hard_keys)}) currently allow an entry on "
        f"<b>{', '.join(s.symbol + ' ' + '/'.join(s.permitted) for s in permitted_syms) or 'no symbol'}</b>. "
        f"A qualifying fair value gap formed inside the 9:30–10:30 window in <b>{n_fvg_sessions} of {n_sessions}</b> symbol-sessions on record."
    )

    # ---- activity
    def event_row(e: ActivityEntry) -> str:
        tone = {"open": "up", "close": "flat", "skipped_execution": "warn"}.get(e.outcome, "warn")
        glyph = {"open": "▲", "close": "●", "skipped_execution": "◇"}.get(e.outcome, "◈")
        return f"<div class='evt'><span class='evt-g {tone}'>{glyph}</span><span class='evt-t'>{et(e.timestamp)}</span><span class='evt-s'>{e.symbol}</span><span class='evt-d'>{html.escape(e.detail)}</span></div>"

    events = "".join(event_row(e) for e in snapshot.today_significant_events) or "<div class='empty'>No signals or trades so far today. Quiet cycles are tallied below.</div>"
    quiet = "".join(
        f"<span class='pill flat'>{s.symbol} · {s.quiet_cycles} quiet{f', {s.near_misses} near-miss' if s.near_misses else ''}</span>"
        for s in snapshot.today_symbol_summaries
    ) or "<span class='dim'>No cycles logged today.</span>"

    # ---- history
    def hist_row(t: ClosedTrade, g: Optional[TradeGrade]) -> str:
        plan = []
        if getattr(t, "tier", None):
            plan.append(t.tier)
        if getattr(t, "invalidation_price", None):
            plan.append(f"inv ${t.invalidation_price:,.2f}")
        return (
            f"<tr><td class='l'>{t.symbol}{' <span class=\"sim\">sim</span>' if t.dry_run else ''}</td><td class='l'>{t.trade_type}</td>"
            f"<td>{'$' + format(t.strike_price, ',.2f') if t.strike_price is not None else '&mdash;'}</td>"
            f"<td class='l'>{t.expiration_date or '&mdash;'}{f' <span class=dim>({t.dte_at_entry}d)</span>' if t.dte_at_entry is not None else ''}</td>"
            f"<td>{t.quantity}</td><td class='l'>{et(t.opened_at, '%b %-d %-I:%M %p')}</td><td class='l'>{et(t.closed_at, '%b %-d %-I:%M %p')}</td>"
            f"<td>{money(t.entry_notional)}</td><td>{money(t.exit_notional)}</td>"
            f"<td class='{sign_cls(t.pnl_dollars)}'>{money(t.pnl_dollars)}</td><td class='{sign_cls(t.pnl_pct)}'>{pct(t.pnl_pct)}</td>"
            f"<td class='l'>{' · '.join(plan) or '&mdash;'}</td><td>{grade_badge(g)}</td><td class='l reason'>{html.escape(t.close_reason or '')}</td></tr>"
        )

    hist = "".join(hist_row(t, g) for t, g in zip(closed, snapshot.trade_grades)) or "<tr><td colspan='14' class='empty'>No trades yet. Dry-run trades appear here too, with a “sim” tag.</td></tr>"

    quality = ""
    if snapshot.check_performance:
        rows = "".join(
            f"<tr><td class='l'>{CHECK_LABELS.get(c.check_key, c.check_key)}</td><td>{pct(c.pass_win_rate, signed=False) if c.pass_win_rate is not None else '&mdash;'} <span class='dim'>({c.pass_trades})</span></td>"
            f"<td>{pct(c.fail_win_rate, signed=False) if c.fail_win_rate is not None else '&mdash;'} <span class='dim'>({c.fail_trades})</span></td></tr>"
            for c in snapshot.check_performance
        )
        quality = f"<table><thead><tr><th class='l'>Check</th><th>Win rate when pass</th><th>Win rate when fail</th></tr></thead><tbody>{rows}</tbody></table>"
    else:
        quality = f"<div class='empty'>Needs {MIN_TRADES_FOR_AGGREGATE} closed trades to say which checks correlate with wins (have {n_closed}).</div>"

    positions = "".join(
        f"<tr><td class='l'>{v.position.symbol}</td><td class='l'>{v.position.option_type}</td><td>${v.position.strike_price:,.2f}</td><td class='l'>{v.position.expiration_date}</td><td>{v.position.quantity}</td>"
        f"<td>{f'${v.bid:.2f} / ${v.ask:.2f}' if v.bid is not None and v.ask is not None else '&mdash;'}</td>"
        f"<td>{money(v.position.average_premium_paid * v.position.quantity * 100)}</td><td>{money(v.current_value)}</td>"
        f"<td class='{sign_cls(v.pnl_dollars)}'>{money(v.pnl_dollars)}</td><td class='{sign_cls(v.pnl_pct)}'>{pct(v.pnl_pct)}</td></tr>"
        for v in snapshot.option_positions
    ) or "<tr><td colspan='10' class='empty'>No open options positions.</td></tr>"

    error = f"<div class='banner'><b>Last refresh had a problem.</b> {html.escape(snapshot.error)}</div>" if snapshot.error else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{refresh_seconds}">
<title>Agentic Options Bot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --page:#0b1118; --surface:#121a24; --raised:#182331; --hair:rgba(168,190,214,.14); --hair-strong:rgba(168,190,214,.28);
  --ink:#eef2f6; --ink-2:#aab7c5; --ink-3:#71818f;
  --accent:#e39a5b; --accent-ink:#0b1118; --series:#5aa9ff; --series-soft:rgba(90,169,255,.18);
  --up:#43d17e; --down:#ff6f6f; --warn:#f3c969; --up-soft:rgba(67,209,126,.14); --down-soft:rgba(255,111,111,.14); --warn-soft:rgba(243,201,105,.14);
  --display:"Sora","Segoe UI",system-ui,sans-serif; --body:"Source Sans 3","Segoe UI",system-ui,sans-serif; --mono:"IBM Plex Mono","SFMono-Regular",Consolas,monospace;
  color-scheme: dark;
}}
@media (prefers-color-scheme: light) {{ :root:not([data-theme="dark"]) {{
  --page:#f3f5f8; --surface:#ffffff; --raised:#eef2f6; --hair:rgba(20,40,60,.12); --hair-strong:rgba(20,40,60,.24);
  --ink:#14202c; --ink-2:#45566a; --ink-3:#7a8a9a; --accent:#c96f2e; --accent-ink:#ffffff; --series:#2f7fe0; --series-soft:rgba(47,127,224,.14);
  --up:#1f9d55; --down:#d64545; --warn:#b8860b; --up-soft:rgba(31,157,85,.12); --down-soft:rgba(214,69,69,.12); --warn-soft:rgba(184,134,11,.14); color-scheme: light;
}} }}
:root[data-theme="light"] {{
  --page:#f3f5f8; --surface:#ffffff; --raised:#eef2f6; --hair:rgba(20,40,60,.12); --hair-strong:rgba(20,40,60,.24);
  --ink:#14202c; --ink-2:#45566a; --ink-3:#7a8a9a; --accent:#c96f2e; --accent-ink:#ffffff; --series:#2f7fe0; --series-soft:rgba(47,127,224,.14);
  --up:#1f9d55; --down:#d64545; --warn:#b8860b; --up-soft:rgba(31,157,85,.12); --down-soft:rgba(214,69,69,.12); --warn-soft:rgba(184,134,11,.14); color-scheme: light;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font-family:var(--body); font-size:14.5px; line-height:1.45; }}
.wrap {{ max-width:1240px; margin:0 auto; padding:28px 24px 56px; display:flex; flex-direction:column; gap:30px; }}
a {{ color:var(--series); }}
.dim {{ color:var(--ink-3); }}
.up {{ color:var(--up); }} .down {{ color:var(--down); }} .warn {{ color:var(--warn); }}
.mono, .t-value, .sym-price, .lv-price, .lv-dist, td, .evt-t, .facts dd, .sym-chg {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}
h1 {{ font-family:var(--display); font-weight:600; font-size:22px; letter-spacing:-.01em; margin:0; }}
h2 {{ font-family:var(--display); font-weight:600; font-size:15px; letter-spacing:.005em; margin:0; }}
.eyebrow {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--ink-3); font-weight:600; }}
/* masthead */
.mast {{ display:flex; flex-wrap:wrap; align-items:center; gap:14px 18px; padding-bottom:18px; border-bottom:1px solid var(--hair); }}
.mast-mark {{ width:34px; height:34px; border-radius:9px; background:var(--accent); display:grid; place-items:center; color:var(--accent-ink); font-family:var(--display); font-weight:700; font-size:15px; }}
.mast-title {{ display:flex; flex-direction:column; gap:2px; }}
.mast-sub {{ font-size:12.5px; color:var(--ink-3); }}
.mode {{ font-family:var(--display); font-size:11px; font-weight:700; letter-spacing:.1em; padding:5px 10px; border-radius:6px; }}
.mode.dry {{ background:var(--warn-soft); color:var(--warn); border:1px solid var(--warn); }}
.mode.live {{ background:var(--down-soft); color:var(--down); border:1px solid var(--down); }}
.mast-right {{ margin-left:auto; text-align:right; font-size:12.5px; color:var(--ink-2); display:flex; flex-direction:column; gap:2px; }}
.mast-right b {{ color:var(--ink); font-weight:600; }}
.strip {{ display:flex; flex-wrap:wrap; gap:6px 8px; }}
.chip {{ font-size:12px; padding:4px 9px; border-radius:6px; background:var(--raised); color:var(--ink-2); }}
.chip b {{ color:var(--ink-3); font-weight:600; margin-right:6px; text-transform:uppercase; font-size:10px; letter-spacing:.06em; }}
/* tiles */
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
.tile {{ background:var(--surface); border:1px solid var(--hair); border-radius:10px; padding:14px 16px 12px; display:flex; flex-direction:column; gap:4px; }}
.t-label {{ font-size:11.5px; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-3); font-weight:600; }}
.t-value {{ font-size:26px; font-weight:500; letter-spacing:-.01em; }}
.t-value.small {{ font-size:22px; }}
.t-sub {{ font-size:12px; }}
.spark {{ width:100%; height:30px; display:block; margin-top:4px; }}
.spark-line {{ fill:none; stroke-width:1.8; stroke-linejoin:round; }}
.spark-area {{ opacity:.18; }}
.spark.up .spark-line, .spark.up .spark-dot {{ stroke:var(--up); fill:var(--up); }} .spark.up .spark-line {{ fill:none; }} .spark.up .spark-area {{ fill:var(--up); }}
.spark.down .spark-line, .spark.down .spark-dot {{ stroke:var(--down); fill:var(--down); }} .spark.down .spark-line {{ fill:none; }} .spark.down .spark-area {{ fill:var(--down); }}
/* sections */
section {{ display:flex; flex-direction:column; gap:12px; }}
.sec-head {{ display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }}
.sec-head .note {{ font-size:13px; color:var(--ink-2); }}
.panel {{ background:var(--surface); border:1px solid var(--hair); border-radius:10px; padding:14px 16px; }}
.empty {{ color:var(--ink-3); padding:14px 4px; text-align:center; font-size:13.5px; font-family:var(--body); }}
.banner {{ background:var(--down-soft); border:1px solid var(--down); border-radius:8px; padding:10px 14px; font-size:13px; }}
/* structure grid */
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:12px; }}
.sym {{ background:var(--surface); border:1px solid var(--hair); border-radius:10px; padding:14px 16px; display:flex; flex-direction:column; gap:10px; }}
.sym-head {{ display:grid; grid-template-columns:auto auto 1fr auto; align-items:center; gap:10px; }}
.sym-name {{ font-family:var(--display); font-weight:700; font-size:16px; }}
.sym-price {{ font-size:15px; }}
.sym-chg {{ font-size:12px; }}
.sym-spark {{ width:110px; }} .sym-spark .spark {{ height:26px; margin:0; }}
.sym-verdict {{ font-family:var(--display); font-size:12px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; padding:5px 10px; border-radius:6px; align-self:flex-start; }}
.sym-verdict.up {{ background:var(--up-soft); color:var(--up); }} .sym-verdict.down {{ background:var(--down-soft); color:var(--down); }} .sym-verdict.flat {{ background:var(--raised); color:var(--ink-2); }}
.trends {{ display:grid; grid-template-columns:repeat(4,1fr); gap:6px; }}
.trend {{ display:flex; flex-direction:column; gap:1px; background:var(--raised); border-radius:6px; padding:6px 8px; font-size:12px; }}
.trend-label {{ font-size:10px; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-3); }}
.trend-val {{ font-weight:600; }} .trend.up .trend-val {{ color:var(--up); }} .trend.down .trend-val {{ color:var(--down); }} .trend.flat .trend-val {{ color:var(--ink-2); }}
.sides {{ display:flex; flex-direction:column; gap:5px; font-size:12px; }}
.side {{ display:grid; grid-template-columns:38px auto 1fr; gap:8px; align-items:center; }}
.side-name {{ font-weight:600; text-transform:uppercase; font-size:10.5px; letter-spacing:.06em; color:var(--ink-3); }}
.side-chips {{ display:flex; gap:4px; }}
.chk {{ font-size:10.5px; padding:2px 6px; border-radius:4px; background:var(--raised); color:var(--ink-3); border:1px solid transparent; }}
.chk.pass {{ color:var(--up); border-color:var(--up); background:var(--up-soft); }} .chk.fail {{ color:var(--down); border-color:var(--down); background:var(--down-soft); }}
.veto {{ color:var(--ink-2); }} .ok {{ color:var(--up); }}
.facts {{ margin:0; display:grid; grid-template-columns:1fr; gap:3px; font-size:12.5px; }}
.facts div {{ display:grid; grid-template-columns:96px 1fr; gap:8px; }} .facts dt {{ color:var(--ink-3); }} .facts dd {{ margin:0; font-size:12px; }}
.levels ul {{ list-style:none; margin:6px 0 0; padding:0; display:grid; grid-template-columns:1fr 1fr; gap:3px 14px; font-size:12px; }}
.levels li {{ display:grid; grid-template-columns:1fr auto auto; gap:8px; }}
.lv-name {{ color:var(--ink-2); }} .lv-dist {{ min-width:52px; text-align:right; }}
.rg-row {{ display:flex; gap:4px; margin-top:6px; }}
.rg {{ width:22px; height:14px; border-radius:3px; background:var(--raised); position:relative; }}
.rg.uptrend {{ background:var(--up); opacity:.85; }} .rg.downtrend {{ background:var(--down); opacity:.85; }} .rg.ranging {{ background:var(--hair-strong); }}
.rg.fvg::after {{ content:""; position:absolute; left:8px; bottom:-5px; width:6px; height:3px; border-radius:2px; background:var(--accent); }}
.last-read {{ border-top:1px solid var(--hair); padding-top:8px; font-size:12px; display:flex; flex-direction:column; gap:2px; color:var(--ink-2); }}
/* timeline */
.timeline {{ width:100%; height:54px; display:block; }}
.tl-window {{ fill:var(--series-soft); }} .tl-axis {{ stroke:var(--hair-strong); }} .tl-tick {{ stroke:var(--hair-strong); }}
.tl-sched {{ fill:var(--ink-3); opacity:.5; }} .tl-cycle {{ fill:var(--accent); }} .tl-now {{ stroke:var(--ink); stroke-width:1.5; stroke-dasharray:3 2; }}
.axis {{ font-family:var(--mono); font-size:10.5px; fill:var(--ink-3); }}
.legend {{ display:flex; gap:16px; font-size:12px; color:var(--ink-2); flex-wrap:wrap; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:-1px; }}
/* charts */
.charts {{ display:grid; grid-template-columns:1.5fr 1fr; gap:12px; }} @media (max-width:820px) {{ .charts {{ grid-template-columns:1fr; }} }}
.chart {{ width:100%; height:220px; display:block; overflow:visible; }}
.chart .gl {{ stroke:var(--hair); }} .chart .zero {{ stroke:var(--hair-strong); }}
.chart .area {{ fill:var(--series-soft); }} .chart .line {{ fill:none; stroke:var(--series); stroke-width:2; stroke-linejoin:round; }} .chart .end {{ fill:var(--series); }}
.chart .hit {{ fill:transparent; }} .chart .hit:hover {{ fill:var(--series-soft); }}
.chart .bar.up {{ fill:var(--up); }} .chart .bar.down {{ fill:var(--down); }}
/* activity */
.evt {{ display:grid; grid-template-columns:16px 70px 56px 1fr; gap:10px; padding:8px 0; border-bottom:1px solid var(--hair); font-size:13px; align-items:baseline; }}
.evt:last-child {{ border-bottom:none; }} .evt-g {{ font-size:10px; }} .evt-t {{ color:var(--ink-3); font-size:12px; }} .evt-s {{ font-weight:600; }} .evt-d {{ color:var(--ink-2); }}
.pills {{ display:flex; flex-wrap:wrap; gap:6px; }}
.pill {{ font-size:12px; padding:3px 9px; border-radius:999px; background:var(--raised); color:var(--ink-2); }}
/* tables */
.tbl {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
th {{ text-align:right; font-family:var(--body); font-size:10.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-3); font-weight:600; padding:8px 8px; border-bottom:1px solid var(--hair-strong); white-space:nowrap; }}
td {{ text-align:right; padding:8px 8px; border-bottom:1px solid var(--hair); white-space:nowrap; }}
th.l, td.l {{ text-align:left; }} td.reason {{ white-space:normal; max-width:320px; font-family:var(--body); color:var(--ink-2); }}
tbody tr:last-child td {{ border-bottom:none; }}
.sim {{ font-size:9.5px; padding:1px 5px; border-radius:4px; background:var(--warn-soft); color:var(--warn); font-family:var(--body); vertical-align:1px; }}
.grade {{ display:inline-grid; place-items:center; width:22px; height:22px; border-radius:5px; font-weight:600; font-size:12px; cursor:help; }}
.grade.ga, .grade.gb {{ background:var(--up-soft); color:var(--up); }} .grade.gc, .grade.gd {{ background:var(--warn-soft); color:var(--warn); }} .grade.gf {{ background:var(--down-soft); color:var(--down); }} .grade.na {{ background:var(--raised); color:var(--ink-3); }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }} @media (max-width:820px) {{ .two {{ grid-template-columns:1fr; }} }}
@media (prefers-reduced-motion: no-preference) {{ .tile, .sym, .panel {{ animation:in .35s ease-out both; }} @keyframes in {{ from {{ opacity:.001; transform:translateY(4px); }} to {{ opacity:1; transform:none; }} }} }}
</style>
</head>
<body>
<div class="wrap">
<header class="mast">
  <div class="mast-mark">Ao</div>
  <div class="mast-title"><h1>Agentic Options Bot</h1><span class="mast-sub">{account_label} · {phase} · rendered {now_et.strftime('%a %b %d, %I:%M %p').replace(' 0', ' ').lstrip('0')} ET</span></div>
  <span class="mode {'live' if live else 'dry'}">{'LIVE ORDERS' if live else 'DRY RUN'}</span>
  <div class="mast-right"><span>{freshness}</span><span>{bars_txt}</span></div>
</header>
{error}
<div class="strip">{chips}</div>
{tiles}

<section>
  <div class="sec-head"><h2>Market structure</h2><span class="note">{struct_summary}</span></div>
  <div class="grid">{cards}</div>
  <div class="legend"><span><i style="background:var(--up)"></i>uptrend</span><span><i style="background:var(--down)"></i>downtrend</span><span><i style="background:var(--hair-strong)"></i>ranging</span><span><i style="background:var(--accent);border-radius:2px;height:4px;width:12px"></i>a fair value gap formed inside the entry window</span><span class="dim">Structure is read at 10:30 ET each session from 5-minute bars, the same read the bot makes.</span></div>
</section>

<section>
  <div class="sec-head"><h2>Today's session</h2><span class="note">Entry window 9:30–10:30 ET shaded. Small ticks are the scheduled cadence: every 5 minutes inside the window, then hourly through 5 PM for exits. Dots are cycles the bot actually logged today.</span></div>
  <div class="panel">{timeline_svg(cycles_today, now_et)}</div>
</section>

<section>
  <div class="sec-head"><h2>Performance</h2></div>
  <div class="charts">
    <div class="panel"><div class="eyebrow">Account equity</div>{equity_chart(eq_hist)}</div>
    <div class="panel"><div class="eyebrow">P&amp;L per closed trade</div>{pnl_chart(list(reversed(closed)))}</div>
  </div>
</section>

<section>
  <div class="sec-head"><h2>Activity today</h2><span class="note">Signals, opens, closes and anything the bot wanted to do but couldn't.</span></div>
  <div class="panel">{events}</div>
  <div class="pills">{quiet}</div>
</section>

<section>
  <div class="sec-head"><h2>Trade history</h2><span class="note">Exits: max hold → expiration-day 3 PM close → trend invalidation. No fixed profit target or stop percentage.</span></div>
  <div class="panel tbl"><table><thead><tr><th class="l">Symbol</th><th class="l">Type</th><th>Strike</th><th class="l">Expiry</th><th>Qty</th><th class="l">Opened</th><th class="l">Closed</th><th>Put in</th><th>Exit value</th><th>P&amp;L $</th><th>P&amp;L %</th><th class="l">Plan</th><th>Grade</th><th class="l">Exit reason</th></tr></thead><tbody>{hist}</tbody></table></div>
</section>

<div class="two">
  <section>
    <div class="sec-head"><h2>Open positions</h2></div>
    <div class="panel tbl"><table><thead><tr><th class="l">Symbol</th><th class="l">Type</th><th>Strike</th><th class="l">Expiry</th><th>Qty</th><th>Bid / ask</th><th>Cost</th><th>Value</th><th>P&amp;L $</th><th>P&amp;L %</th></tr></thead><tbody>{positions}</tbody></table></div>
  </section>
  <section>
    <div class="sec-head"><h2>Which checks predict a win</h2></div>
    <div class="panel tbl">{quality}</div>
  </section>
</div>
</div>
</body>
</html>"""
