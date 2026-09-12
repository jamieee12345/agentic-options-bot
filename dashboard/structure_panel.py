"""Per-symbol market-structure read for the dashboard, computed from the
committed bar cache (market_data/) at render time -- no broker or MCP call.

This is the "what does the market look like right now, and what would the
bot be ALLOWED to do" panel. Everything here reuses the exact functions the
live decision path uses (brain/confluence.py, brain/market_structure.py,
brain/liquidity.py, brain/fvg_indicators.py, brain/volatility.py), with the
same policy from settings.yaml -- so a "puts only" verdict on the page is
the same verdict the bot would reach on its next cycle, not a re-
implementation that can drift.

Observational only: nothing here writes to any log or changes any setting.
The point is to accumulate a picture over time of WHICH regimes produce
signals and which don't, so the strategy can be tuned on evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from brain.confluence import ConfluencePolicy, evaluate_confluence
from brain.fvg_indicators import detect_fair_value_gaps
from brain.liquidity import LiquidityPool, find_session_pools
from brain.market_structure import classify_structure, detect_break_of_structure, find_swing_points
from brain.trend_indicators import sma_trend
from brain.volatility import compute_atr
from orchestration.market_hours import MARKET_TZ

REGIME_HISTORY_SESSIONS = 10
ENTRY_WINDOW_END = time(10, 30)


@dataclass(frozen=True)
class LevelRead:
    label: str            # "prev day high", "opening range low", ...
    price: float
    side: str             # "above" / "below" current price
    distance_pct: float   # signed: positive = level above price


@dataclass(frozen=True)
class SessionRegime:
    session: date
    structure: str         # uptrend / downtrend / ranging, read at 10:30 ET
    day_move_pct: float    # session close vs session open
    fvg_in_window: bool    # did a fresh FVG form inside 09:30-10:30


@dataclass
class SymbolStructure:
    symbol: str
    price: float
    as_of: datetime                       # timestamp of the last completed 5-min bar (UTC-aware)
    session_change_pct: Optional[float]   # vs the session's first bar open
    prior_close_change_pct: Optional[float]
    trend_1h: Optional[str]               # bullish / bearish / None
    trend_4h: Optional[str]
    trend_daily: Optional[str]
    structure_5m: str                      # uptrend / downtrend / ranging
    last_swing_high: Optional[float]
    last_swing_low: Optional[float]
    bos: Optional[str]                     # bullish / bearish / None
    atr_5m: Optional[float]
    atr_daily: Optional[float]
    session_range: Optional[float]
    session_range_vs_atr: Optional[float]  # session high-low / daily ATR
    levels: List[LevelRead] = field(default_factory=list)
    permitted: List[str] = field(default_factory=list)          # "calls" / "puts" the hard checks would allow right now
    hard_checks: Dict[str, Dict[str, str]] = field(default_factory=dict)  # direction -> {check: pass/fail/n/a}
    veto_reason: Dict[str, Optional[str]] = field(default_factory=dict)
    last_fvg: Optional[Tuple[str, int, bool]] = None            # (kind, bars ago, volume confirmed)
    regimes: List[SessionRegime] = field(default_factory=list)
    session_closes: List[float] = field(default_factory=list)   # last session's 5-min closes, for the sparkline
    error: Optional[str] = None


def _et(ts) -> datetime:
    return ts.to_pydatetime().astimezone(MARKET_TZ)


def _session_dates(bars: pd.DataFrame) -> List[date]:
    return sorted({_et(ts).date() for ts in bars.index})


def _level_reads(pools: List[LiquidityPool], price: float) -> List[LevelRead]:
    names = {
        "previous_day_high": "prev day high", "previous_day_low": "prev day low",
        "opening_range_high": "opening range high", "opening_range_low": "opening range low",
        "equal_levels": "equal levels",
    }
    out = []
    for p in pools:
        if p.tier != "major":
            continue
        label = names.get(p.label, p.label)
        if p.label == "equal_levels":
            label = "equal highs" if p.kind == "buy_side" else "equal lows"
        dist = (p.price - price) / price
        out.append(LevelRead(label, p.price, "above" if dist >= 0 else "below", dist))
    # nearest first, above and below interleaved by distance
    out.sort(key=lambda l: abs(l.distance_pct))
    return out[:6]


@dataclass(frozen=True)
class FvgParams:
    lookback_period: int = 10
    body_multiplier: float = 1.5
    volume_multiplier: float = 1.5
    min_gap_atr_multiplier: float = 0.5


def _gaps(bars: pd.DataFrame, fvg: FvgParams):
    return detect_fair_value_gaps(bars, fvg.lookback_period, fvg.body_multiplier, fvg.volume_multiplier, fvg.min_gap_atr_multiplier)


def _regime_history(bars: pd.DataFrame, sessions: List[date], n: int, fvg: FvgParams) -> List[SessionRegime]:
    out = []
    for d in sessions[-n:]:
        cutoff = datetime.combine(d, ENTRY_WINDOW_END, tzinfo=MARKET_TZ)
        upto = bars[bars.index <= pd.Timestamp(cutoff)]
        day = bars[[_et(ts).date() == d for ts in bars.index]]
        if len(upto) < 12 or day.empty:
            continue
        structure = classify_structure(find_swing_points(upto))
        move = (float(day["close"].iloc[-1]) - float(day["open"].iloc[0])) / float(day["open"].iloc[0])
        window = day[[_et(ts).time() <= ENTRY_WINDOW_END for ts in day.index]]
        had_fvg = False
        if len(window) >= 3:
            gaps = _gaps(upto, fvg)
            first_idx = len(upto) - len(window)
            had_fvg = any(g is not None and g.index >= first_idx for g in gaps)
        out.append(SessionRegime(d, structure, move, had_fvg))
    return out


def read_symbol(
    symbol: str,
    bars_5m: Optional[pd.DataFrame],
    hourly: Optional[pd.DataFrame],
    four_hour: Optional[pd.DataFrame],
    daily: Optional[pd.DataFrame],
    policy: ConfluencePolicy,
    min_confluence_score: float,
    fvg: FvgParams = FvgParams(),
) -> Optional[SymbolStructure]:
    if bars_5m is None or bars_5m.empty:
        return None
    price = float(bars_5m["close"].iloc[-1])
    as_of = bars_5m.index[-1].to_pydatetime()
    sessions = _session_dates(bars_5m)
    last_session = sessions[-1]
    today_bars = bars_5m[[_et(ts).date() == last_session for ts in bars_5m.index]]

    session_change = None
    session_range = None
    if not today_bars.empty:
        o = float(today_bars["open"].iloc[0])
        session_change = (price - o) / o if o else None
        session_range = float(today_bars["high"].max() - today_bars["low"].min())

    prior_close_change = None
    if daily is not None and not daily.empty:
        prior = daily[[(_et(ts).date() if ts.tzinfo else ts.date()) < last_session for ts in daily.index]]
        if not prior.empty:
            pc = float(prior["close"].iloc[-1])
            prior_close_change = (price - pc) / pc if pc else None

    def _dir(bars, period):
        if bars is None or bars.empty:
            return None
        return sma_trend(bars, period).direction

    swings = find_swing_points(bars_5m)
    highs = [p for p in swings if p.kind == "high"]
    lows = [p for p in swings if p.kind == "low"]
    bos = detect_break_of_structure(bars_5m, swings)
    atr_5m = compute_atr(bars_5m)
    atr_daily = compute_atr(daily) if daily is not None and len(daily) > 15 else None

    # "now" for pool detection = just after the last bar, so the last
    # session in the cache counts as "today" even on a weekend render.
    pools = find_session_pools(bars_5m, daily, as_of + timedelta(minutes=1))

    permitted, hard_checks, veto = [], {}, {}
    for direction, side in (("bullish", "calls"), ("bearish", "puts")):
        try:
            res = evaluate_confluence(
                bars_5m, direction, min_confluence_score=min_confluence_score, daily_bars=daily,
                hourly_bars=hourly, four_hour_bars=four_hour, policy=policy,
            )
        except Exception as exc:  # never let one symbol take the page down
            hard_checks[side] = {}
            veto[side] = f"{type(exc).__name__}: {exc}"
            continue
        hard, _soft = policy.effective_keys()
        hard_checks[side] = {k: res.details.get(k, "n/a") for k in hard}
        veto[side] = res.veto_reason if res.hard_vetoed else None
        if not res.hard_vetoed:
            permitted.append(side)

    last_fvg = None
    gaps = _gaps(bars_5m, fvg)
    for g in reversed(gaps):
        if g is not None:
            last_fvg = (g.kind, len(bars_5m) - 1 - g.index, bool(g.volume_confirmed))
            break

    return SymbolStructure(
        symbol=symbol, price=price, as_of=as_of,
        session_change_pct=session_change, prior_close_change_pct=prior_close_change,
        trend_1h=_dir(hourly, 20), trend_4h=_dir(four_hour, 20), trend_daily=_dir(daily, 200),
        structure_5m=classify_structure(swings),
        last_swing_high=highs[-1].price if highs else None, last_swing_low=lows[-1].price if lows else None,
        bos=bos.direction if bos else None,
        atr_5m=atr_5m, atr_daily=atr_daily, session_range=session_range,
        session_range_vs_atr=(session_range / atr_daily) if (session_range is not None and atr_daily) else None,
        levels=_level_reads(pools, price), permitted=permitted, hard_checks=hard_checks, veto_reason=veto,
        last_fvg=last_fvg, regimes=_regime_history(bars_5m, sessions, REGIME_HISTORY_SESSIONS, fvg),
        session_closes=[float(c) for c in today_bars["close"]] if not today_bars.empty else [],
    )


def read_watchlist(symbols, now: datetime, policy: ConfluencePolicy, min_confluence_score: float, fvg: FvgParams = FvgParams()) -> List[SymbolStructure]:
    from orchestration.bar_cache import load_all

    b5 = load_all(symbols, "5minute", now)
    b1h = load_all(symbols, "hour", now)
    b4h = load_all(symbols, "4hour", now)
    bd = load_all(symbols, "day", now)
    out = []
    for s in symbols:
        s = s.upper()
        try:
            r = read_symbol(s, b5.get(s), b1h.get(s), b4h.get(s), bd.get(s), policy, min_confluence_score, fvg)
        except Exception as exc:
            r = SymbolStructure(symbol=s, price=0.0, as_of=now, session_change_pct=None, prior_close_change_pct=None,
                                trend_1h=None, trend_4h=None, trend_daily=None, structure_5m="ranging",
                                last_swing_high=None, last_swing_low=None, bos=None, atr_5m=None, atr_daily=None,
                                session_range=None, session_range_vs_atr=None, error=f"{type(exc).__name__}: {exc}")
        if r is not None:
            out.append(r)
    return out
