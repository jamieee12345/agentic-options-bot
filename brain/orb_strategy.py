"""Opening Range Breakout (ORB) -- `options.model: orb`. Built 2026-09-12 from
the user's spec (an intraday index/stock ORB plan, translated to US hours):

  1. RANGE.   The first `orb_minutes` of the regular session (9:30-9:45 ET
              = the first three 5-minute bars). Nothing is tradable until
              the range is complete.
  2. BREAKOUT. The FIRST 5-minute bar of the day that CLOSES beyond the
              range -- above the ORB high for a long, below the ORB low for
              a short -- with volume > `volume_multiplier` x the average of
              the previous `volume_lookback` bars, and the close on the
              right side of the session VWAP (above for longs, below for
              shorts). One breakout per day per symbol: the first one that
              qualifies is THE signal; later ones are ignored.
  3. DON'T CHASE. The breakout must be within `breakout_lookback_bars` of
              the latest bar (an hourly cycle still catches it), and the
              current close must be no more than `max_chase_atr` x ATR
              beyond the range edge. If it ran, it ran.
  4. STOP.    The opposite end of the range, or 1 x ATR(`atr_period`, 5m)
              from entry -- whichever is TIGHTER. Fixed at entry.
  5. TARGET.  `target_r` x R (R = entry - stop) for a partial; the executor
              trails the remainder at VWAP / a 2-bar low-high, and hard
              time-stops everything at `time_stop` ET. Those exit rules
              live in orchestration/options_execution.py (orb mode); this
              module only produces the entry decision with its plan.

Sizing is by premium at risk (safety/options_sizing.py); conviction is
always 1.0 here -- the spec's tiers are about instrument choice (ITM,
delta ~0.6), which the routine prompt handles at contract selection, not
about scaling the budget.

Returns brain.options_strategy.OptionsDecision (same shape as the other
models) with invalidation_price / target_price / tier set and the ORB
read-out in `confluence_details` (orb_complete / breakout / volume / vwap
/ no_chase).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from brain.liquidity import MARKET_TZ
from brain.options_strategy import OptionsDecision


@dataclass(frozen=True)
class OrbConfig:
    orb_minutes: int = 15
    volume_multiplier: float = 1.5
    volume_lookback: int = 20
    atr_period: int = 14
    target_r: float = 1.5
    breakout_lookback_bars: int = 12
    max_chase_atr: float = 0.5


def _et_date(ts):
    return ts.date() if ts.tzinfo is None else ts.astimezone(MARKET_TZ).date()


def todays_bars(bars: pd.DataFrame, now: datetime) -> pd.DataFrame:
    today = now.astimezone(MARKET_TZ).date()
    return bars[[_et_date(ts) == today for ts in bars.index]]


def session_vwap(todays: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP from the session open, one value per bar."""
    typical = (todays["high"] + todays["low"] + todays["close"]) / 3.0
    pv = (typical * todays["volume"]).cumsum()
    vol = todays["volume"].cumsum().replace(0, float("nan"))
    return pv / vol


def atr(bars: pd.DataFrame, period: int) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    prev_close = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - prev_close).abs(), (bars["low"] - prev_close).abs()], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if pd.notna(val) and val > 0 else None


@dataclass(frozen=True)
class OrbRange:
    high: float
    low: float
    bars: int  # how many bars formed it


def opening_range(todays: pd.DataFrame, orb_minutes: int) -> Optional[OrbRange]:
    """The completed opening range, or None while it is still forming."""
    if todays.empty:
        return None
    first_ts = todays.index[0]
    inside = todays[todays.index < first_ts + pd.Timedelta(minutes=orb_minutes)]
    if inside.empty or len(todays) <= len(inside):
        return None
    return OrbRange(float(inside["high"].max()), float(inside["low"].min()), len(inside))


def decide_orb_action(symbol: str, bars: pd.DataFrame, now: datetime, cfg: OrbConfig = OrbConfig()) -> OptionsDecision:
    details: Dict[str, str] = {}
    todays = todays_bars(bars, now)
    rng = opening_range(todays, cfg.orb_minutes)
    if rng is None:
        details["orb_complete"] = "fail"
        return OptionsDecision(symbol, "hold", 0.0, f"opening range ({cfg.orb_minutes} min) not complete yet", confluence_details=details)
    details["orb_complete"] = "pass"

    vwap = session_vwap(todays)
    # volume average over the previous `volume_lookback` bars, using the full series (yesterday's bars count early in the day)
    vol_avg_all = bars["volume"].shift(1).rolling(cfg.volume_lookback).mean()

    # first qualifying breakout bar of the day, after the range
    post = todays.iloc[rng.bars:]
    breakout_idx = None
    direction = None
    for ts, bar in post.iterrows():
        close = float(bar["close"])
        v_avg = vol_avg_all.get(ts)
        if v_avg is None or pd.isna(v_avg) or v_avg <= 0:
            continue
        vol_ok = float(bar["volume"]) > cfg.volume_multiplier * float(v_avg)
        vw = vwap.get(ts)
        if close > rng.high:
            if vol_ok and pd.notna(vw) and close > vw:
                breakout_idx, direction = ts, "bullish"
                break
        elif close < rng.low:
            if vol_ok and pd.notna(vw) and close < vw:
                breakout_idx, direction = ts, "bearish"
                break

    if breakout_idx is None:
        details["breakout"] = "fail"
        return OptionsDecision(
            symbol, "hold", 0.0,
            f"no qualifying breakout of the {rng.low:.2f}-{rng.high:.2f} opening range yet (needs a 5m close beyond it with volume >{cfg.volume_multiplier}x avg and the right side of VWAP)",
            confluence_details=details,
        )
    details["breakout"] = "pass"
    details["volume"] = "pass"
    details["vwap"] = "pass"

    bars_ago = len(bars) - 1 - bars.index.get_loc(breakout_idx)
    price = float(bars["close"].iloc[-1])
    a = atr(bars, cfg.atr_period) or 0.0
    edge = rng.high if direction == "bullish" else rng.low
    if bars_ago > cfg.breakout_lookback_bars:
        details["no_chase"] = "fail"
        return OptionsDecision(symbol, "hold", 0.0, f"{direction} ORB breakout was {bars_ago} bars ago (limit {cfg.breakout_lookback_bars}) -- missed, not chasing",
                               confluence_details=details)
    chased = (price - edge) > cfg.max_chase_atr * a if direction == "bullish" else (edge - price) > cfg.max_chase_atr * a
    if chased:
        details["no_chase"] = "fail"
        return OptionsDecision(symbol, "hold", 0.0, f"{direction} ORB breakout {bars_ago} bar(s) ago, but price {price:.2f} is >{cfg.max_chase_atr} ATR ({a:.2f}) past the range edge {edge:.2f} -- not chasing",
                               confluence_details=details)
    # already back inside the range? then the breakout failed
    if (direction == "bullish" and price <= rng.high) or (direction == "bearish" and price >= rng.low):
        details["no_chase"] = "fail"
        return OptionsDecision(symbol, "hold", 0.0, f"{direction} ORB breakout {bars_ago} bar(s) ago, but price {price:.2f} is back inside the range -- failed breakout",
                               confluence_details=details)
    details["no_chase"] = "pass"

    if direction == "bullish":
        stop = max(rng.low, price - a) if a > 0 else rng.low   # tighter of the two = the higher stop
        r = price - stop
        target = price + cfg.target_r * r
        action = "buy_call"
    else:
        stop = min(rng.high, price + a) if a > 0 else rng.high
        r = stop - price
        target = price - cfg.target_r * r
        action = "buy_put"
    if r <= 0:
        return OptionsDecision(symbol, "hold", 0.0, "degenerate stop (R <= 0)", confluence_details=details)
    stop_kind = "ATR" if (a > 0 and ((direction == "bullish" and price - a > rng.low) or (direction == "bearish" and price + a < rng.high))) else "opposite ORB edge"
    reasoning = (
        f"{direction} ORB breakout {bars_ago} bar(s) ago: range {rng.low:.2f}-{rng.high:.2f}, close beyond it with volume >{cfg.volume_multiplier}x and on the right side of VWAP; "
        f"entry {price:.2f}, stop {stop:.2f} ({stop_kind}, R={r:.2f}), target {target:.2f} ({cfg.target_r}R partial, then trail at VWAP / 2-bar); time-stop at the configured hour"
    )
    return OptionsDecision(
        symbol, action, 1.0, reasoning, gap_low=rng.low, gap_high=rng.high, gap_kind=direction, volume_confirmed=True,
        confluence_details=details, invalidation_price=stop, target_price=target, tier="full",
    )
