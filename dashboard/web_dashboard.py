"""HTML dashboard renderer.

Same data model as terminal_dashboard.py (DashboardSnapshot etc.), rendered
as a self-contained HTML page instead of monospace text.

Intended usage once the live loop is wired up: call render_html_dashboard()
and write the result to a fixed path (e.g. dashboard.html) after every
iteration. A browser tab left open on that file auto-refreshes on its own
via the meta-refresh tag below — genuinely continuous monitoring without
needing a running web server, since the browser just re-reads the file from
disk on its own schedule.
"""
from __future__ import annotations

from dashboard.terminal_dashboard import DashboardSnapshot

DEFAULT_REFRESH_SECONDS = 5  # matches settings.yaml: monitoring.dashboard_refresh_seconds

_TIER_COLORS = {"Low Vol": "#2ecc71", "Mid Vol": "#f1c40f", "High Vol": "#e74c3c"}


def _fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def _fmt_pct(x: float, signed: bool = False) -> str:
    sign = "+" if signed and x >= 0 else ""
    return f"{sign}{x:.1%}"


def render_html_dashboard(snapshot: DashboardSnapshot, refresh_seconds: int = DEFAULT_REFRESH_SECONDS) -> str:
    """`refresh_seconds` defaults to 5 per the requirement that this must
    refresh every 5 seconds (also matches settings.yaml). Left configurable
    for testing, but live risk-status monitoring should not increase this.
    """
    regime = snapshot.regime
    p = snapshot.portfolio
    r = snapshot.risk
    pnl_color = "#2ecc71" if p.daily_pnl_dollars >= 0 else "#e74c3c"
    dd_color = "#2ecc71" if r.daily_dd_pct >= 0 else "#e74c3c"

    positions_rows = "".join(
        f"<tr><td>{pos.ticker}</td><td><span class='badge long'>{pos.direction}</span></td>"
        f"<td>{_fmt_money(pos.price)}</td><td>{_fmt_pct(pos.pct_of_portfolio)}</td>"
        f"<td>{_fmt_money(pos.stop_price)}</td><td>{pos.hours_in:.1f}</td></tr>"
        for pos in snapshot.positions
    ) or "<tr><td colspan='6' class='muted'>No open positions</td></tr>"

    signals_rows = "".join(
        f"<tr><td>{s.timestamp.strftime('%H:%M:%S')}</td><td>{s.ticker}</td>"
        f"<td>{_fmt_pct(s.allocation_before_pct)} &rarr; {_fmt_pct(s.allocation_after_pct)}</td>"
        f"<td><span class='badge' style='background:{_TIER_COLORS.get(s.tier, '#8b949e')}22; "
        f"color:{_TIER_COLORS.get(s.tier, '#8b949e')};'>{s.tier}</span></td></tr>"
        for s in snapshot.recent_signals
    ) or "<tr><td colspan='4' class='muted'>No recent signals</td></tr>"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh_seconds}">
<title>Trading bot dashboard</title>
<style>
  body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, Helvetica, Arial, sans-serif; padding: 24px; }}
  .section {{ margin-bottom: 24px; }}
  .section-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #8b949e; margin-bottom: 8px; }}
  .badge {{ padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 500; }}
  .badge.long {{ background: #2ecc7122; color: #2ecc71; }}
  .cards {{ display: flex; gap: 16px; }}
  .card {{ background: #161b22; border-radius: 8px; padding: 16px; flex: 1; }}
  .card .label {{ font-size: 12px; color: #8b949e; margin-bottom: 4px; }}
  .card .value {{ font-size: 22px; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; color: #8b949e; font-weight: 400; padding: 8px 6px; border-bottom: 1px solid #30363d; }}
  td {{ padding: 8px 6px; border-bottom: 1px solid #21262d; }}
  .muted {{ color: #8b949e; text-align: center; }}
  .regime-badge {{ font-size: 18px; font-weight: 600; }}
  .meta {{ color: #8b949e; font-size: 13px; margin-left: 12px; }}
</style>
</head>
<body>
  <div class="section">
    <div class="section-label">Regime</div>
    <span class="regime-badge">{regime.label} ({regime.probability:.0%})</span>
    <span class="meta">Stability: {regime.stability_bars} bars</span>
    <span class="meta">Flicker: {regime.flicker_count}/{regime.flicker_window}</span>
  </div>

  <div class="section">
    <div class="section-label">Portfolio</div>
    <div class="cards">
      <div class="card"><div class="label">Equity</div><div class="value">{_fmt_money(p.equity)}</div></div>
      <div class="card"><div class="label">Daily gain/loss</div><div class="value" style="color:{pnl_color}">{_fmt_money(p.daily_pnl_dollars)} ({_fmt_pct(p.daily_pnl_pct, signed=True)})</div></div>
      <div class="card"><div class="label">Allocation</div><div class="value">{_fmt_pct(p.allocation_pct)}</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-label">Risk status</div>
    <span class="meta" style="margin-left:0; color:{dd_color};">Daily DD: {_fmt_pct(r.daily_dd_pct, signed=True)}</span>
  </div>

  <div class="section">
    <div class="section-label">Positions</div>
    <table>
      <tr><th>Ticker</th><th>Side</th><th>Price</th><th>%</th><th>Stop</th><th>Hours in</th></tr>
      {positions_rows}
    </table>
  </div>

  <div class="section">
    <div class="section-label">Recent signals</div>
    <table>
      <tr><th>Time</th><th>Ticker</th><th>Rebalance</th><th>Regime</th></tr>
      {signals_rows}
    </table>
  </div>
</body>
</html>"""
