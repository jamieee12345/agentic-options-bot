"""Per-indicator attribution for the FVG + confluence options strategy, on
the intraday (5-minute) walk-forward backtest under the CURRENT
config/settings.yaml. Answers "which checks are carrying information and
which are noise" -- the input to deciding what to keep, not a tuning loop.

For every confluence check (all of brain/confluence.ALL_CHECK_KEYS, whether
or not the active policy uses it -- they're all still computed):

  * at TRIGGERS (an FVG formed and volume confirmed): how often it read
    pass / fail / n/a, and how often it was THE named veto;
  * at ENTRIES (trades actually taken): win rate and average P&L split by
    the check's reading at entry. A check whose "pass" trades do no better
    than its "fail" trades carries no information; one whose fail trades
    do BETTER is pulling against the others.

Plus the headline stats of the same run (trade count, win rate, profit
factor, exit-reason breakdown) so the reader sees the strategy's overall
state next to the per-check detail.

Same caveats as backtest/options_strategy_backtest.py -- theoretical
option prices, no costs, ~60-day yfinance window -- and one more that
matters here: with a few dozen trades, any split with fewer than ~10
trades on a side is a hint, not a finding. The report marks those.

Usage:
    python -m backtest.indicator_attribution [--days 58] [--symbols ...] [--out reports/indicators-<date>.md]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from backtest.options_strategy_backtest import (
    DEFAULT_INTRADAY_DAYS, HOURLY_WARMUP_DAYS, SimulatedTrade, resample_ohlcv, run_symbol_backtest_intraday,
)
from brain.confluence import ALL_CHECK_KEYS
from config.config_loader import load_settings
from data.fetchers import YFinanceHistoricalFetcher

THIN_SAMPLE = 10  # below this many trades on a side, a split is flagged as unreadable


def run_attribution(settings, symbols: List[str], days: int, log=print) -> dict:
    f = YFinanceHistoricalFetcher()
    end = date.today()
    i_start = end - timedelta(days=days)
    d_start = end - timedelta(days=500)
    h_start = i_start - timedelta(days=HOURLY_WARMUP_DAYS)
    policy = settings.options.confluence_policy()
    hard_keys, soft_keys = policy.effective_keys()

    trigger_reads = {k: Counter() for k in ALL_CHECK_KEYS}
    veto_named: Counter = Counter()
    closed: List[SimulatedTrade] = []
    counts: Counter = Counter()

    def hook(ts, d):
        for k in ALL_CHECK_KEYS:
            trigger_reads[k][d.confluence_details.get(k, "missing")] += 1
        if d.action == "hold":
            r = d.reasoning
            for tag, key in (("1-hour trend", "trend_1h"), ("4-hour trend", "trend_4h"), ("ranging", "market_structure (ranging)"),
                             ("market structure is a clear", "market_structure (opposing)"), ("Elliott", "elliott_wave"),
                             ("applicable confluence", "too few applicable soft checks"), ("confluence score", "soft score below minimum")):
                if tag in r:
                    veto_named[key] += 1
                    break

    t0 = time.time()
    for sym in symbols:
        five = f.get_bars(sym, i_start.isoformat(), end.isoformat(), "5m")
        day = f.get_bars(sym, d_start.isoformat(), end.isoformat(), "1D")
        hr = f.get_bars(sym, h_start.isoformat(), end.isoformat(), "1h")
        fh = resample_ohlcv(hr, "4h", offset="9h30min")
        r = run_symbol_backtest_intraday(sym, five, day, settings, hourly_bars=hr, four_hour_bars=fh, decision_hook=hook)
        closed.extend(t for t in r.trades if t.is_closed and t.pnl_pct is not None)
        counts["triggers"] += r.fvg_triggers
        counts["rejected"] += r.confluence_rejections
        counts["opened"] += r.trades_opened
        log(f"{sym} done ({time.time() - t0:.0f}s)")

    return {
        "window": (i_start, end), "symbols": symbols, "policy": policy, "hard_keys": hard_keys, "soft_keys": soft_keys,
        "trigger_reads": trigger_reads, "veto_named": veto_named, "closed": closed, "counts": counts,
    }


def _stats(trades: List[SimulatedTrade]) -> Optional[dict]:
    if not trades:
        return None
    wins = [t.pnl_pct for t in trades if t.pnl_pct > 0]
    losses = [t.pnl_pct for t in trades if t.pnl_pct <= 0]
    gp, gl = sum(wins), -sum(losses)
    return {
        "n": len(trades), "win_rate": len(wins) / len(trades), "avg": sum(t.pnl_pct for t in trades) / len(trades),
        "sum": sum(t.pnl_pct for t in trades), "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0, "pf": (gp / gl) if gl > 0 else float("inf"),
    }


def render_markdown(a: dict) -> str:
    i_start, end = a["window"]
    closed = a["closed"]
    c = a["counts"]
    L = [f"# Indicator attribution — {i_start} .. {end}", ""]
    L.append(f"Symbols: {', '.join(a['symbols'])}. Policy: hard = {list(a['hard_keys'])}, soft = {list(a['soft_keys'])}, "
             f"ranging-structure veto = {a['policy'].structure_must_agree}, min applicable = {a['policy'].min_applicable_checks}.")
    L.append("")
    L.append("## Headline (current config)")
    s = _stats(closed)
    L.append(f"- Triggers (FVG + volume): **{c['triggers']}** · rejected by confluence: {c['rejected']} · trades opened: {c['opened']} · closed: {len(closed)}")
    if s:
        L.append(f"- Win rate **{s['win_rate']:.1%}** · avg win {s['avg_win']:+.1%} · avg loss {s['avg_loss']:+.1%} · **profit factor {s['pf']:.3f}** · sum of trade P&L {s['sum']:+.1%}")
    else:
        L.append("- No closed trades in the window.")
    by_reason: Dict[str, List[SimulatedTrade]] = defaultdict(list)
    for t in closed:
        by_reason[t.exit_reason or "?"].append(t)
    if by_reason:
        L.append("")
        L.append("| exit reason | n | win rate | avg P&L | total |")
        L.append("|---|---|---|---|---|")
        for reason, ts in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            st = _stats(ts)
            L.append(f"| {reason} | {st['n']} | {st['win_rate']:.0%} | {st['avg']:+.1%} | {st['sum']:+.1%} |")
    L.append("")
    L.append("## What is doing the filtering")
    L.append(f"Of {c['triggers']} triggers, the named rejection reasons were:")
    for k, n in a["veto_named"].most_common():
        L.append(f"- {k}: {n}")
    L.append("")
    L.append("## How each check read at triggers")
    L.append("| check | role | pass | fail | n/a |")
    L.append("|---|---|---|---|---|")
    for k in ALL_CHECK_KEYS:
        r = a["trigger_reads"][k]
        tot = sum(r.values()) or 1
        role = "HARD" if k in a["hard_keys"] else ("soft" if k in a["soft_keys"] else "unused")
        L.append(f"| {k} | {role} | {r['pass']} ({r['pass']/tot:.0%}) | {r['fail']} ({r['fail']/tot:.0%}) | {r['n/a']} ({r['n/a']/tot:.0%}) |")
    L.append("")
    L.append("## Outcome by each check's reading at entry")
    L.append("A check is informative when its PASS trades beat its FAIL trades; equal = noise; inverted = pulling against the others. "
             f"Splits with fewer than {THIN_SAMPLE} trades on a side are marked ⚠ thin.")
    L.append("")
    L.append("| check | role | pass n / win / avg | fail n / win / avg | n/a n / win / avg | edge (pass−fail) |")
    L.append("|---|---|---|---|---|---|")
    rows = []
    for k in ALL_CHECK_KEYS:
        by: Dict[str, List[SimulatedTrade]] = defaultdict(list)
        for t in closed:
            by[t.entry_details.get(k, "missing")].append(t)
        p, fl, na = _stats(by.get("pass", [])), _stats(by.get("fail", [])), _stats(by.get("n/a", []))
        edge = (p["avg"] - fl["avg"]) if (p and fl) else None
        thin = (p is None or p["n"] < THIN_SAMPLE) or (fl is None or fl["n"] < THIN_SAMPLE)
        rows.append((k, edge, thin))
        fmt = lambda st: f"{st['n']} / {st['win_rate']:.0%} / {st['avg']:+.1%}" if st else "—"
        role = "HARD" if k in a["hard_keys"] else ("soft" if k in a["soft_keys"] else "unused")
        L.append(f"| {k} | {role} | {fmt(p)} | {fmt(fl)} | {fmt(na)} | {('%+.1f%%' % (edge*100)) if edge is not None else '—'}{' ⚠ thin' if thin and edge is not None else ''} |")
    L.append("")
    L.append("## Ranked")
    readable = sorted([r for r in rows if r[1] is not None and not r[2]], key=lambda r: r[1])
    thin_rows = [r for r in rows if r[1] is not None and r[2]]
    if readable:
        L.append("Readable splits, most harmful/uninformative first:")
        for k, edge, _ in readable:
            verdict = "harmful (inverted)" if edge < -0.05 else ("noise" if abs(edge) <= 0.05 else "informative")
            L.append(f"- {k}: edge {edge:+.1%} — {verdict}")
    if thin_rows:
        L.append("")
        L.append("Too thin to call (one side under %d trades): %s" % (THIN_SAMPLE, ", ".join(f"{k} ({edge:+.1%})" for k, edge, _ in thin_rows)))
    no_split = [k for k, edge, _ in rows if edge is None]
    if no_split:
        L.append("")
        L.append("No pass-vs-fail split available (hard vetoes never enter on a fail; or a check never read fail on an entered trade): " + ", ".join(no_split))
    return "\n".join(L)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=DEFAULT_INTRADAY_DAYS)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--out", default=None, help="Markdown path; default reports/indicators-<today>.md")
    args = ap.parse_args()
    settings = load_settings(args.settings)
    symbols = [s.upper() for s in (args.symbols or settings.broker.core_watchlist)]
    a = run_attribution(settings, symbols, args.days, log=lambda m: print(m, file=sys.stderr, flush=True))
    md = render_markdown(a)
    out = Path(args.out) if args.out else Path("reports") / f"indicators-{date.today().isoformat()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(md)


if __name__ == "__main__":
    main()
