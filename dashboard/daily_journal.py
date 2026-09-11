"""End-of-day trading journal for one market date, built purely from the
files the MCP routine commits back to the repo (activity_log.jsonl,
trade_log.jsonl, equity_history.jsonl, account_snapshot.json). No broker
or MCP call -- same read-only stance as dashboard/live_account_dashboard.py.

Written so that a DRY-RUN day reads exactly like a real one: every trade
the bot intended is journaled with the contract it picked, the price it
saw, the gap and confluence read-out that justified it, and what happened
to it -- executed (dry or live), closed (and why), or not executable (no
expiration in the DTE window / empty chain / sizing rejected, journaled by
evaluate_for_agent's `note` subcommand). Signals that fired but were vetoed
get their own section, so "nothing traded" is always explained, never
silent.

Usage:
    python -m dashboard.daily_journal --date 2026-09-11 [--out reports/]

Writes reports/journal-<date>.html and reports/journal-<date>.md and
prints the markdown to stdout.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from orchestration.account_snapshot import DEFAULT_SNAPSHOT_PATH, read_account_snapshot
from orchestration.activity_log import DEFAULT_LOG_PATH as ACTIVITY_PATH
from orchestration.activity_log import ActivityEntry, entries_for_date, read_entries as read_activity
from orchestration.trade_log import DEFAULT_LOG_PATH as TRADE_PATH
from orchestration.trade_log import TradeLogEntry, build_trade_history, read_entries as read_trades

ET = ZoneInfo("America/New_York")
DEFAULT_OUT_DIR = Path("reports")


def _et(ts: str) -> str:
    return datetime.fromisoformat(ts).astimezone(ET).strftime("%H:%M")


def _money(x: Optional[float]) -> str:
    return "—" if x is None else f"${x:,.2f}"


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:+.1%}"


def _cycle_key(ts: str) -> str:
    """Entries written by one `evaluate` call share a timestamp to the second;
    bucket to the minute so a cycle's records group together."""
    return datetime.fromisoformat(ts).astimezone(ET).strftime("%H:%M")


def _load_equity_for_date(path: Path, target: date) -> List[tuple]:
    points = []
    if not path.exists():
        return points
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ts = datetime.fromisoformat(r["timestamp"]).astimezone(ET)
        if ts.date() == target:
            points.append((ts, float(r["equity"])))
    return points


def build_journal(target: date, activity_path: Path = ACTIVITY_PATH, trade_path: Path = TRADE_PATH,
                  equity_path: Path = Path("equity_history.jsonl"), snapshot_path: Path = DEFAULT_SNAPSHOT_PATH) -> dict:
    activity = entries_for_date(read_activity(activity_path), target)
    trades_all = read_trades(trade_path)
    trades = [t for t in trades_all if datetime.fromisoformat(t.timestamp).astimezone(ET).date() == target]
    closed, still_open = build_trade_history(trades_all)
    closed_today = [c for c in closed if datetime.fromisoformat(c.closed_at).astimezone(ET).date() == target]
    opened_today = [o for o in still_open if datetime.fromisoformat(o.opened_at).astimezone(ET).date() == target]
    equity = _load_equity_for_date(equity_path, target)
    snapshot = read_account_snapshot(snapshot_path)

    cycles: Dict[str, List[ActivityEntry]] = defaultdict(list)
    for e in activity:
        cycles[_cycle_key(e.timestamp)].append(e)
    symbols = sorted({e.symbol for e in activity})

    signals = [e for e in activity if e.gap_kind is not None and e.outcome not in ("open", "close")]
    skipped_exec = [e for e in activity if e.outcome == "skipped_execution"]
    return {
        "date": target, "activity": activity, "trades": trades, "closed_today": closed_today,
        "opened_today": opened_today, "still_open": still_open, "equity": equity, "snapshot": snapshot,
        "cycles": dict(sorted(cycles.items())), "symbols": symbols, "signals": signals, "skipped_exec": skipped_exec,
    }


# ----------------------------------------------------------------------------- markdown
def render_markdown(j: dict) -> str:
    d = j["date"]
    L = [f"# Trading journal — {d.strftime('%A, %B %d, %Y')}", ""]
    eq = j["equity"]
    if eq:
        L.append(f"**Equity:** {_money(eq[0][1])} at {eq[0][0]:%H:%M} ET → {_money(eq[-1][1])} at {eq[-1][0]:%H:%M} ET ({_pct(eq[-1][1]/eq[0][1]-1) if eq[0][1] else '—'})")
    L.append(f"**Cycles run:** {len(j['cycles'])} ({', '.join(j['cycles'].keys())} ET)")
    L.append(f"**Symbols watched:** {', '.join(j['symbols'])}")
    L.append(f"**Trades journaled:** {len([t for t in j['trades'] if t.event == 'open'])} opened, {len(j['closed_today'])} closed")
    L.append("")

    L.append("## Trades")
    if not j["trades"]:
        L.append("_No trade was opened or closed today._")
    for t in j["trades"]:
        tag = "LIVE" if not t.dry_run else "DRY RUN"
        if t.event == "open":
            L.append(f"### {_et(t.timestamp)} ET — OPEN {t.symbol} {t.trade_type.upper()} ×{t.quantity}  `[{tag}]`")
            L.append(f"- Contract: {t.symbol} {t.expiration_date} {t.strike_price} {t.trade_type} ({t.dte_at_entry} DTE at entry)")
            L.append(f"- Premium: {_money(t.price)} per contract (bid {_money(t.bid)} / ask {_money(t.ask)}), notional {_money(t.notional)}")
            if t.gap_low is not None:
                L.append(f"- Triggering gap: {t.gap_low:.2f} – {t.gap_high:.2f}")
            if t.confluence_score is not None:
                L.append(f"- Confluence: {t.confluence_score:.0%} over {t.confluence_applicable} applicable — {t.confluence_details}")
            if t.reason:
                L.append(f"- Reasoning: {t.reason}")
            if t.order_id:
                L.append(f"- Order id: `{t.order_id}`")
        else:
            L.append(f"### {_et(t.timestamp)} ET — CLOSE {t.symbol} {t.trade_type.upper()} ×{t.quantity}  `[{tag}]`")
            L.append(f"- Exit premium: {_money(t.price)} per contract, proceeds {_money(t.notional)}")
            L.append(f"- Reason: {t.reason}")
        L.append("")
    for c in j["closed_today"]:
        L.append(f"- **Round trip** {c.symbol} {c.trade_type} ×{c.quantity}: opened {_et(c.opened_at)} → closed {_et(c.closed_at)} ET, "
                 f"{_money(c.entry_notional)} → {_money(c.exit_notional)}, **P&L {_money(c.pnl_dollars)} ({_pct(c.pnl_pct)})**, reason: {c.close_reason}")
    if j["still_open"]:
        L.append("")
        L.append("### Still open at end of day")
        for o in j["still_open"]:
            L.append(f"- {o.symbol} {o.trade_type} ×{o.quantity}, opened {datetime.fromisoformat(o.opened_at).astimezone(ET):%b %d %H:%M} ET, "
                     f"{o.expiration_date} {o.strike_price}, in {_money(o.entry_notional)}")
    L.append("")

    L.append("## Signals that did not become trades")
    if not j["signals"] and not j["skipped_exec"]:
        L.append("_No fair value gap formed on any symbol's latest bar at any cycle today._")
    for e in j["signals"]:
        L.append(f"- {_et(e.timestamp)} ET **{e.symbol}** {e.gap_kind} FVG @ {e.price}: {e.detail}")
        if e.confluence_details:
            fails = [k for k, v in e.confluence_details.items() if v == "fail"]
            L.append(f"  - checks failing: {', '.join(fails) or 'none'}; score {e.confluence_score if e.confluence_score is None else f'{e.confluence_score:.0%}'} over {e.confluence_applicable}")
    for e in j["skipped_exec"]:
        L.append(f"- {_et(e.timestamp)} ET **{e.symbol}** {e.option_type}: {e.detail}")
    L.append("")

    L.append("## Cycle timeline")
    L.append("Legend: `·` hold (no gap) · `◦` gap, volume unconfirmed · `✗` gap vetoed · `!` wanted to open, not executable · `●` OPEN · `■` CLOSE")
    L.append("")
    L.append("| ET | " + " | ".join(j["symbols"]) + " |")
    L.append("|---|" + "---|" * len(j["symbols"]))
    for ck, entries in j["cycles"].items():
        by_sym = {}
        for e in entries:
            by_sym[e.symbol] = _glyph(e)
        L.append(f"| {ck} | " + " | ".join(by_sym.get(s, " ") for s in j["symbols"]) + " |")
    L.append("")
    snap = j["snapshot"]
    if snap is not None:
        L.append(f"_Account at last cycle: equity {_money(snap.equity)}, buying power {_money(snap.buying_power)}, "
                 f"{len(snap.option_positions)} option position(s), {snap.open_order_count} open order(s)._")
    return "\n".join(L)


def _glyph(e: ActivityEntry) -> str:
    if e.outcome == "open":
        return "●"
    if e.outcome == "close":
        return "■"
    if e.outcome == "skipped_execution":
        return "!"
    if e.gap_kind is None:
        return "·"
    if e.volume_confirmed is False:
        return "◦"
    return "✗"


# ----------------------------------------------------------------------------- html
def render_html(j: dict, md: str) -> str:
    """A simple, printable page: the markdown's content re-laid-out with
    real structure. Kept dependency-free (no markdown library)."""
    d = j["date"]
    esc = html.escape

    def trade_card(t: TradeLogEntry) -> str:
        tag = "LIVE" if not t.dry_run else "DRY RUN"
        cls = "live" if not t.dry_run else "dry"
        if t.event == "open":
            rows = [
                ("Contract", f"{t.symbol} {t.expiration_date} {t.strike_price} {t.trade_type} · {t.dte_at_entry} DTE"),
                ("Premium", f"{_money(t.price)} / contract · bid {_money(t.bid)} · ask {_money(t.ask)} · notional {_money(t.notional)}"),
            ]
            if t.gap_low is not None:
                rows.append(("Gap", f"{t.gap_low:.2f} – {t.gap_high:.2f}"))
            if t.confluence_score is not None:
                checks = " ".join(f"<span class='chk {v}'>{esc(k)}</span>" for k, v in t.confluence_details.items())
                rows.append(("Confluence", f"{t.confluence_score:.0%} over {t.confluence_applicable} applicable<br>{checks}"))
            if t.reason:
                rows.append(("Reasoning", esc(t.reason)))
            if t.order_id:
                rows.append(("Order id", f"<code>{esc(t.order_id)}</code>"))
            head = f"OPEN {esc(t.symbol)} {esc(t.trade_type.upper())} ×{t.quantity}"
        else:
            rows = [("Exit", f"{_money(t.price)} / contract · proceeds {_money(t.notional)}"), ("Reason", esc(t.reason or ""))]
            head = f"CLOSE {esc(t.symbol)} {esc(t.trade_type.upper())} ×{t.quantity}"
        body = "".join(f"<div class='row'><span class='k'>{k}</span><span class='v'>{v}</span></div>" for k, v in rows)
        return f"<div class='card {cls}'><div class='head'><span class='time'>{_et(t.timestamp)} ET</span> {head} <span class='tag'>{tag}</span></div>{body}</div>"

    trades_html = "".join(trade_card(t) for t in j["trades"]) or "<p class='muted'>No trade was opened or closed today.</p>"
    rt = "".join(
        f"<li><b>{esc(c.symbol)} {esc(c.trade_type)} ×{c.quantity}</b>: {_et(c.opened_at)} → {_et(c.closed_at)} ET, "
        f"{_money(c.entry_notional)} → {_money(c.exit_notional)}, <b class='{'pos' if c.pnl_dollars >= 0 else 'neg'}'>{_money(c.pnl_dollars)} ({_pct(c.pnl_pct)})</b>, {esc(c.close_reason or '')}</li>"
        for c in j["closed_today"])
    sig = "".join(
        f"<li><span class='time'>{_et(e.timestamp)} ET</span> <b>{esc(e.symbol)}</b> {esc(e.gap_kind or '')} FVG @ {e.price}: {esc(e.detail)}"
        + ("<br><span class='chks'>" + " ".join(f"<span class='chk {v}'>{esc(k)}</span>" for k, v in e.confluence_details.items()) + "</span>" if e.confluence_details else "")
        + "</li>" for e in j["signals"])
    sig += "".join(f"<li><span class='time'>{_et(e.timestamp)} ET</span> <b>{esc(e.symbol)}</b> {esc(e.option_type or '')}: {esc(e.detail)}</li>" for e in j["skipped_exec"])
    if not sig:
        sig = "<li class='muted'>No fair value gap formed on any symbol's latest bar at any cycle today.</li>"

    head_cells = "".join(f"<th>{esc(s)}</th>" for s in j["symbols"])
    rows_html = ""
    for ck, entries in j["cycles"].items():
        by_sym = {e.symbol: e for e in entries}
        cells = ""
        for s in j["symbols"]:
            e = by_sym.get(s)
            g = _glyph(e) if e else ""
            title = esc(e.detail) if e else ""
            cells += f"<td class='g g-{g if g in '·◦✗!●■' else 'x'}' title='{title}'>{g}</td>"
        rows_html += f"<tr><td class='time'>{ck}</td>{cells}</tr>"

    eq = j["equity"]
    eq_line = (f"{_money(eq[0][1])} → {_money(eq[-1][1])} ({_pct(eq[-1][1]/eq[0][1]-1) if eq[0][1] else '—'})" if eq else "—")
    snap = j["snapshot"]
    snap_line = (f"equity {_money(snap.equity)} · buying power {_money(snap.buying_power)} · {len(snap.option_positions)} position(s) · {snap.open_order_count} open order(s)"
                 if snap is not None else "—")
    n_open = len([t for t in j["trades"] if t.event == "open"])

    return f"""<title>Journal {d.isoformat()}</title>
<style>
:root {{ --bg:#f7f6f2; --ink:#1d1c19; --muted:#6b6862; --line:#dedbd3; --card:#fffdf8; --accent:#8a4b1f; --pos:#1f6f43; --neg:#a3352b; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#17171a; --ink:#ebe8e1; --muted:#9a968d; --line:#33333a; --card:#1f1f24; --accent:#d9955a; --pos:#5fbf86; --neg:#e07b6f; }} }}
:root[data-theme="dark"] {{ --bg:#17171a; --ink:#ebe8e1; --muted:#9a968d; --line:#33333a; --card:#1f1f24; --accent:#d9955a; --pos:#5fbf86; --neg:#e07b6f; }}
body {{ background:var(--bg); color:var(--ink); font: 15px/1.5 Georgia, 'Times New Roman', serif; max-width: 900px; margin: 0 auto; padding: 32px 20px 60px; }}
h1 {{ font-size: 28px; margin: 0 0 4px; text-wrap: balance; }}
h2 {{ font-size: 13px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin: 36px 0 12px; border-bottom: 1px solid var(--line); padding-bottom: 6px; font-family: system-ui, sans-serif; }}
.meta {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 10px 20px; margin: 18px 0 6px; font-family: system-ui, sans-serif; font-size: 13px; }}
.meta b {{ display:block; color: var(--muted); font-weight: 500; font-size: 11px; letter-spacing:.08em; text-transform: uppercase; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-left: 4px solid var(--accent); border-radius: 6px; padding: 12px 16px; margin: 12px 0; }}
.card.live {{ border-left-color: var(--pos); }}
.head {{ font-family: system-ui, sans-serif; font-weight: 600; margin-bottom: 8px; }}
.tag {{ font-size: 11px; letter-spacing:.08em; padding: 2px 6px; border: 1px solid var(--line); border-radius: 4px; margin-left: 8px; color: var(--muted); }}
.row {{ display: grid; grid-template-columns: 110px 1fr; gap: 12px; padding: 3px 0; font-size: 14px; }}
.k {{ color: var(--muted); font-family: system-ui, sans-serif; font-size: 12px; padding-top: 2px; }}
.time {{ font-family: ui-monospace, Menlo, monospace; font-size: 13px; color: var(--muted); }}
.chk {{ display:inline-block; font: 11px ui-monospace, Menlo, monospace; padding: 1px 6px; border-radius: 3px; border: 1px solid var(--line); margin: 2px 2px 0 0; }}
.chk.pass {{ border-color: var(--pos); color: var(--pos); }} .chk.fail {{ border-color: var(--neg); color: var(--neg); }}
ul {{ padding-left: 18px; }} li {{ margin: 6px 0; }}
.muted {{ color: var(--muted); }} .pos {{ color: var(--pos); }} .neg {{ color: var(--neg); }}
.timeline {{ overflow-x: auto; }} table {{ border-collapse: collapse; font-family: system-ui, sans-serif; font-size: 13px; }}
th, td {{ border: 1px solid var(--line); padding: 4px 8px; text-align: center; }} th {{ background: var(--card); }}
td.g {{ font-size: 15px; }} td.g-● {{ color: var(--pos); font-weight: 700; }} td.g-■ {{ color: var(--accent); font-weight: 700; }} td.g-✗ {{ color: var(--neg); }} td.g-\\! {{ color: var(--accent); font-weight: 700; }}
.legend {{ font-size: 12px; color: var(--muted); font-family: system-ui, sans-serif; margin-bottom: 8px; }}
</style>
<h1>Trading journal — {d.strftime('%A, %B %d, %Y')}</h1>
<div class="muted">Account …9190 · built from the routine's committed logs · every intended trade journaled, executed or not</div>
<div class="meta">
  <div><b>Equity (first → last cycle)</b>{eq_line}</div>
  <div><b>Cycles run</b>{len(j['cycles'])} — {esc(', '.join(j['cycles'].keys()))} ET</div>
  <div><b>Trades journaled</b>{n_open} opened · {len(j['closed_today'])} closed</div>
  <div><b>Account at last cycle</b>{snap_line}</div>
</div>
<h2>Trades</h2>
{trades_html}
{('<ul>' + rt + '</ul>') if rt else ''}
<h2>Signals that did not become trades</h2>
<ul>{sig}</ul>
<h2>Cycle timeline</h2>
<div class="legend">· hold (no gap) &nbsp; ◦ gap, volume unconfirmed &nbsp; ✗ gap vetoed &nbsp; ! wanted to open, not executable &nbsp; ● OPEN &nbsp; ■ CLOSE &nbsp; (hover a cell for the reason)</div>
<div class="timeline"><table><tr><th>ET</th>{head_cells}</tr>{rows_html}</table></div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (America/New_York market date); default today")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()
    target = date.fromisoformat(args.date) if args.date else datetime.now(ET).date()
    j = build_journal(target)
    md = render_markdown(j)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"journal-{target.isoformat()}.md").write_text(md, encoding="utf-8")
    (out / f"journal-{target.isoformat()}.html").write_text(render_html(j, md), encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    except Exception:
        pass
    print(md)


if __name__ == "__main__":
    main()
