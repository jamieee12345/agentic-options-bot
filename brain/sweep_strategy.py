"""ONE model: liquidity sweep -> displacement gap -> entry. Built 2026-09-11
to replace the multi-indicator confluence gate (brain/options_strategy.py +
brain/confluence.py), which per-indicator attribution showed was rejecting
~90% of triggers for no measurable edge.

The model, in order -- every step must hold or the answer is "hold":

  1. POOLS.  Where is engineered liquidity right now? Previous-day high/
     low, today's completed opening range, equal highs/lows (major); single
     swing highs/lows (minor). brain/liquidity.find_session_pools.
  2. SWEEP.  Within the last `sweep_lookback_bars` 5-minute bars, did a bar
     wick through a pool and CLOSE back inside? Sell-side swept -> the
     trade is bullish (calls); buy-side swept -> bearish (puts). The sweep
     bar's wick extreme is the invalidation price, fixed now.
  3. DISPLACEMENT GAP.  After the sweep bar (within
     `displacement_window_bars`), a Fair Value Gap formed in the reversal
     direction (brain/fvg_indicators). No gap, no trade -- a sweep without
     displacement is just a wick.
  4. DON'T CHASE.  Price is still at the gap: no further than
     `max_chase_atr` x ATR beyond the gap's far edge, and not already past
     the invalidation. If it ran without us, it ran.

Everything else is CONTEXT and only sets SIZE (never a veto):
  * pool tier (major/minor),
  * volume confirmation on the displacement candle,
  * price inside a higher-timeframe (1h or 4h) FVG in the trade direction
    -- "know what gap you're inside of".
  Full size (conviction 1.0) only when all three line up; otherwise half
  (0.5). safety/options_sizing.compute_contract_count already scales the
  premium budget by conviction, so no new sizing code.

Target: the nearest opposing pool beyond the gap (the next place liquidity
rests in the trade direction). Session, entries-per-day and the daily loss
limit are enforced by the executor, not here -- this function only
answers "is there a setup on these bars right now".

Returns brain.options_strategy.OptionsDecision so the executor, activity
log, journal and dashboard need no new record shapes; the model's own
read-out goes into `confluence_details` under sweep-specific keys
(sweep / displacement_gap / no_chase / pool_tier / volume / htf_gap).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from brain.fvg_indicators import DEFAULT_BODY_MULTIPLIER, DEFAULT_LOOKBACK_PERIOD, DEFAULT_MIN_GAP_ATR_MULTIPLIER, DEFAULT_VOLUME_MULTIPLIER, FairValueGap, detect_fair_value_gaps
from brain.liquidity import LiquidityPool, LiquiditySweep, detect_sweep_in_window, find_session_pools
from brain.options_strategy import OptionsDecision
from brain.volatility import DEFAULT_ATR_PERIOD


@dataclass(frozen=True)
class SweepConfig:
    sweep_lookback_bars: int = 12        # one hour of 5-minute bars
    displacement_window_bars: int = 6    # the gap must form within this many bars after the sweep
    max_chase_atr: float = 0.5           # how far past the gap's far edge price may already be
    opening_range_minutes: int = 15
    htf_gap_lookback_bars: int = 30      # how many 1h/4h bars back to look for an FVG price sits inside
    fvg_lookback_period: int = DEFAULT_LOOKBACK_PERIOD
    fvg_body_multiplier: float = DEFAULT_BODY_MULTIPLIER
    fvg_volume_multiplier: float = DEFAULT_VOLUME_MULTIPLIER
    fvg_min_gap_atr_multiplier: float = DEFAULT_MIN_GAP_ATR_MULTIPLIER
    atr_period: int = DEFAULT_ATR_PERIOD
    full_conviction: float = 1.0
    half_conviction: float = 0.5


def _atr(bars: pd.DataFrame, period: int) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    prev_close = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - prev_close).abs(), (bars["low"] - prev_close).abs()], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if pd.notna(val) and val > 0 else None


def _gap_after_sweep(bars: pd.DataFrame, sweep: LiquiditySweep, cfg: SweepConfig) -> Optional[FairValueGap]:
    gaps = detect_fair_value_gaps(bars, cfg.fvg_lookback_period, cfg.fvg_body_multiplier, cfg.fvg_volume_multiplier, cfg.fvg_min_gap_atr_multiplier)
    lo, hi = sweep.index + 1, min(len(bars) - 1, sweep.index + cfg.displacement_window_bars)
    for i in range(lo, hi + 1):
        g = gaps[i]
        if g is not None and g.kind == sweep.direction:
            return g
    return None


def _inside_htf_gap(price: float, direction: str, htf_bars: Optional[pd.DataFrame], cfg: SweepConfig) -> Optional[bool]:
    """True if price sits inside a same-direction FVG on this higher
    timeframe (within the last cfg.htf_gap_lookback_bars bars); None if no
    bars were supplied (context unavailable -- counts as not lined up)."""
    if htf_bars is None or len(htf_bars) < 3:
        return None
    gaps = detect_fair_value_gaps(htf_bars, cfg.fvg_lookback_period, cfg.fvg_body_multiplier, volume_multiplier=0.0, min_gap_atr_multiplier=0.0)
    for g in gaps[-cfg.htf_gap_lookback_bars:]:
        if g is not None and g.kind == direction and g.gap_low <= price <= g.gap_high:
            return True
    return False


def _target(price: float, direction: str, pools: List[LiquidityPool]) -> Optional[float]:
    if direction == "bullish":
        above = [p.price for p in pools if p.kind == "buy_side" and p.price > price]
        return min(above) if above else None
    below = [p.price for p in pools if p.kind == "sell_side" and p.price < price]
    return max(below) if below else None


def decide_sweep_action(
    symbol: str,
    bars: pd.DataFrame,
    now: datetime,
    daily_bars: Optional[pd.DataFrame] = None,
    hourly_bars: Optional[pd.DataFrame] = None,
    four_hour_bars: Optional[pd.DataFrame] = None,
    cfg: SweepConfig = SweepConfig(),
) -> OptionsDecision:
    if len(bars) < 3:
        return OptionsDecision(symbol, "hold", 0.0, f"only {len(bars)} bar(s) available")
    price = float(bars["close"].iloc[-1])
    details: Dict[str, str] = {}

    pools = find_session_pools(bars, daily_bars, now, cfg.opening_range_minutes)
    labels = sorted({p.label for p in pools})
    sweep = detect_sweep_in_window(bars, pools, cfg.sweep_lookback_bars)
    if sweep is None:
        details["sweep"] = "fail"
        return OptionsDecision(
            symbol, "hold", 0.0,
            f"no liquidity sweep in the last {cfg.sweep_lookback_bars} bars (pools in play: {', '.join(labels) or 'none'})",
            confluence_details=details,
        )
    details["sweep"] = "pass"
    details["pool_tier"] = "pass" if sweep.pool.tier == "major" else "fail"
    direction = sweep.direction
    bars_ago = len(bars) - 1 - sweep.index
    swept = f"{sweep.pool.label} @ {sweep.level:.2f} ({sweep.pool.tier}) swept {bars_ago} bar(s) ago, wick {sweep.extreme:.2f}"

    gap = _gap_after_sweep(bars, sweep, cfg)
    if gap is None:
        details["displacement_gap"] = "fail"
        return OptionsDecision(
            symbol, "hold", 0.0,
            f"{direction} sweep of {swept} -- no displacement gap yet within {cfg.displacement_window_bars} bars; waiting",
            gap_kind=None, confluence_details=details,
        )
    details["displacement_gap"] = "pass"
    volume_confirmed = bool(gap.volume_confirmed)  # numpy bool from the detector -> plain bool (json-serializable downstream)
    details["volume"] = "pass" if volume_confirmed else "fail"

    # invalidation first: if price already closed beyond the sweep wick the setup is dead
    invalidated = price < sweep.extreme if direction == "bullish" else price > sweep.extreme
    if invalidated:
        details["no_chase"] = "fail"
        return OptionsDecision(
            symbol, "hold", 0.0, f"{direction} sweep of {swept} with gap {gap.gap_low:.2f}-{gap.gap_high:.2f}, but price {price:.2f} already closed beyond the sweep wick -- invalidated before entry",
            gap_kind=gap.kind, volume_confirmed=volume_confirmed, gap_low=None, gap_high=None, confluence_details=details,
        )
    atr = _atr(bars, cfg.atr_period) or 0.0
    far_edge = gap.gap_high if direction == "bullish" else gap.gap_low
    chased = (price - far_edge) > cfg.max_chase_atr * atr if direction == "bullish" else (far_edge - price) > cfg.max_chase_atr * atr
    if chased:
        details["no_chase"] = "fail"
        return OptionsDecision(
            symbol, "hold", 0.0,
            f"{direction} sweep of {swept}, gap {gap.gap_low:.2f}-{gap.gap_high:.2f} formed, but price {price:.2f} has run more than {cfg.max_chase_atr} ATR ({atr:.2f}) past it -- not chasing",
            gap_kind=gap.kind, volume_confirmed=volume_confirmed, confluence_details=details,
        )
    details["no_chase"] = "pass"

    htf = _inside_htf_gap(price, direction, hourly_bars, cfg)
    htf4 = _inside_htf_gap(price, direction, four_hour_bars, cfg)
    inside_htf = bool(htf) or bool(htf4)
    details["htf_gap"] = "pass" if inside_htf else ("n/a" if htf is None and htf4 is None else "fail")

    lined_up = sweep.pool.tier == "major" and volume_confirmed and inside_htf
    tier = "full" if lined_up else "half"
    conviction = cfg.full_conviction if lined_up else cfg.half_conviction
    target = _target(price, direction, pools)
    action = "buy_call" if direction == "bullish" else "buy_put"
    missing = [k for k, v in (("major pool", sweep.pool.tier == "major"), ("volume", volume_confirmed), ("HTF gap", inside_htf)) if not v]
    reasoning = (
        f"{direction} sweep of {swept}; displacement gap {gap.gap_low:.2f}-{gap.gap_high:.2f}"
        f"{' (volume confirmed)' if volume_confirmed else ''}; price {price:.2f} at the gap; "
        f"invalidation {sweep.extreme:.2f}; target {f'{target:.2f}' if target is not None else 'none in view'}; "
        f"{'inside' if inside_htf else 'not inside'} a {direction} 1h/4h gap -> {tier.upper()} size"
        + (f" (missing: {', '.join(missing)})" if missing else " (everything lined up)")
    )
    return OptionsDecision(
        symbol, action, conviction, reasoning, gap_low=gap.gap_low, gap_high=gap.gap_high,
        gap_kind=gap.kind, volume_confirmed=volume_confirmed, confluence_details=details,
        confluence_score=None, confluence_applicable=0,
        invalidation_price=sweep.extreme, target_price=target, tier=tier,
    )
