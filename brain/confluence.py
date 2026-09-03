"""Combines every indicator in the brain/ package into one confluence gate
for a candidate FVG+volume trade.

IMPORTANT HONESTY NOTE, read before trusting any number this module
produces: "confluence_score" below is a COUNT of how many independent,
rule-based checks agree with the candidate trade direction. It is NOT a
statistically calibrated probability of profit. Nothing in this codebase
has backtested this specific rule set against historical outcomes to
measure an actual historical win rate -- doing that honestly would mean
running it through backtest/ against real data and reporting what
actually happened, not asserting a number up front. Anyone who tells you a
specific win probability for a rule-based TA system without having
backtested that EXACT rule set is not telling you something well-founded.
More checks agreeing is evidence, not a guarantee -- there is no
confluence bar, however strict, that makes every trade a winner. What this
system CAN honestly do is take fewer, better-supported trades and cap the
damage on the ones that are wrong anyway (see stop_loss_pct/take_profit_pct
in orchestration/options_execution.py) -- that is the real, achievable
version of "limited risk," not a rule set that never loses.

Two tiers, deliberately different from each other:

  HARD VETOES -- any one true blocks the trade outright, no matter how the
  soft checks below score. These are the "we should never fight this"
  conditions: trading against the 200-day trend, trading against a clear
  opposing market structure trend, or entering right as Elliott Wave rules
  say a same-direction impulse just completed (i.e., chasing exhaustion).

  SOFT CONFLUENCE CHECKS -- break of structure, support/resistance,
  supply/demand, a liquidity sweep, VPVR (volume profile) value-area
  breakout, VPVR node quality (a second read of the same profile -- see
  brain/volume_profile.node_quality_check), RSI momentum, and ATR
  volatility expansion. Each is pass / fail / not-applicable (e.g., "no
  supply/demand zone nearby" is not-applicable, not a fail -- there's
  nothing to disagree with). The score is pass / (pass + fail) among only
  the applicable checks. Requires both a minimum score (settings.yaml:
  options.min_confluence_score) AND a minimum number of applicable checks,
  so a technically-perfect score from only one or two lucky applicable
  checks can't pass on its own -- shallow confluence is chosen not to
  count as real confluence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import pandas as pd

from brain.elliott_wave import detect_impulse
from brain.liquidity import detect_liquidity_sweep, find_liquidity_pools
from brain.market_structure import classify_structure, detect_break_of_structure, find_swing_points
from brain.momentum_indicators import DEFAULT_RSI_PERIOD, rsi_momentum_check
from brain.supply_demand import detect_zones, nearest_zone, price_in_zone
from brain.support_resistance import cluster_levels, distance_pct, nearest_level
from brain.trend_indicators import sma_trend
from brain.volatility import DEFAULT_ATR_PERIOD, volatility_expansion_check
from brain.volume_profile import compute_volume_profile, node_quality_check, price_in_value_area

# Lowered from 1.0 (literal unanimity) on request: at 100%, one disagreeing
# soft check out of any number of applicable ones blocked the trade outright
# -- day one logged 42 real gap+volume signals and zero trades because of
# it. 0.6 still requires a clear majority (e.g. 3 of 5, or 4 of 7 once
# RSI/ATR below are both applicable) rather than unanimity -- meaningfully
# more selective than "any agreement at all," just not maximally so. The
# hard vetoes (200-SMA/market-structure/Elliott-Wave) are UNCHANGED by this
# -- still block the trade outright regardless of the soft score.
DEFAULT_MIN_CONFLUENCE_SCORE = 0.6
DEFAULT_MIN_APPLICABLE_CHECKS = 3
DEFAULT_SR_BLOCK_DISTANCE_PCT = 0.005  # a level within 0.5% of price counts as "immediately in the way"

# The three hard-veto checks, kept separate from the soft ones below so the
# score (passes / applicable) only ever counts the soft checks -- exactly
# the original semantics, even though `details` now always carries every
# key (hard + soft) for the dashboard's live-reasoning display.
HARD_VETO_KEYS = ("trend_200sma", "market_structure", "elliott_wave")
# rsi_momentum and volatility_expansion added on request ("multiple [more]
# checks to confirm trades") -- both read genuinely different signals
# (momentum, volatility) than the other five, which all read price
# structure off the same series, so they're additive rather than
# duplicating an existing vote.
SOFT_CHECK_KEYS = (
    "break_of_structure", "support_resistance", "supply_demand", "liquidity_sweep", "volume_profile",
    "rsi_momentum", "volatility_expansion", "vpvr_node_quality",
)


@dataclass(frozen=True)
class ConfluenceResult:
    passed: bool
    score: Optional[float]              # None if vetoed before scoring, or too few applicable checks to score meaningfully
    applicable_checks: int
    veto_reason: Optional[str]
    details: Dict[str, str] = field(default_factory=dict)  # check name -> "pass"/"fail"/"n/a", for logging/debugging


def evaluate_confluence(
    bars: pd.DataFrame,
    direction: Literal["bullish", "bearish"],
    sma_period: int = 200,
    min_confluence_score: float = DEFAULT_MIN_CONFLUENCE_SCORE,
    min_applicable_checks: int = DEFAULT_MIN_APPLICABLE_CHECKS,
    fvg_lookback_period: int = 10,
    fvg_body_multiplier: float = 1.5,
    daily_bars: Optional[pd.DataFrame] = None,
    rsi_period: int = DEFAULT_RSI_PERIOD,
    atr_period: int = DEFAULT_ATR_PERIOD,
) -> ConfluenceResult:
    """`bars` drives every check except the 200-SMA trend veto -- for live,
    intraday-interval trading, `bars` is expected to be intraday (so
    market structure/S-R/supply-demand/liquidity/volume-profile/Elliott-Wave
    all reflect CURRENT price action, not yesterday's close), while
    `daily_bars` (falls back to `bars` if omitted, e.g. for the backtester,
    which only ever has daily bars anyway) keeps the SMA200 filter meaning
    what it's supposed to mean -- the past 200 DAYS, not 200 five-minute
    bars, which would only be a couple of trading days and defeat the
    entire point of a long-term trend filter.
    """
    price = float(bars["close"].iloc[-1])
    swings = find_swing_points(bars)
    details: Dict[str, str] = {}

    # --- hard vetoes ------------------------------------------------------
    # Always evaluated and recorded in `details`, even the ones that don't
    # end up firing -- so a caller displaying "everything we looked at" (the
    # dashboard's live-reasoning panel) sees the full picture, not just
    # whichever single check happened to trip first. The veto DECISION
    # itself (which trade this blocks) is unchanged from before: any one of
    # these three being unfavorable still blocks the trade outright.

    trend = sma_trend(daily_bars if daily_bars is not None else bars, sma_period)
    if trend.direction is None:
        details["trend_200sma"] = "n/a"
    else:
        details["trend_200sma"] = "pass" if trend.direction == direction else "fail"

    structure = classify_structure(swings)
    opposing_structure = {"bullish": "downtrend", "bearish": "uptrend"}[direction]
    agreeing_structure = {"bullish": "uptrend", "bearish": "downtrend"}[direction]
    if structure == agreeing_structure:
        details["market_structure"] = "pass"
    elif structure == opposing_structure:
        details["market_structure"] = "fail"
    else:
        details["market_structure"] = "n/a"  # "ranging" -- no clear structure to agree or disagree with

    impulse = detect_impulse(swings)
    # Veto-only by original design (see module docstring): a same-direction
    # impulse just completing is a reason to hold off, but an OPPOSING
    # impulse completing isn't claimed as support for this trade -- that
    # would be a new judgment call this codebase hasn't validated, so it
    # stays "n/a" rather than "pass".
    details["elliott_wave"] = "fail" if (impulse.valid_impulse and impulse.impulse_direction == direction) else "n/a"

    veto_reason: Optional[str] = None
    if details["trend_200sma"] == "fail":
        veto_reason = f"200-SMA (daily) trend is {trend.direction}, opposing a {direction} trade"
    elif details["market_structure"] == "fail":
        veto_reason = f"market structure is a clear {structure}, opposing a {direction} trade"
    elif details["elliott_wave"] == "fail":
        veto_reason = (
            f"a {direction} Elliott-Wave-rule-valid impulse just completed -- "
            "entering now would be chasing exhaustion, not the move"
        )

    # --- soft confluence checks ---------------------------------------------
    # Computed unconditionally too (even on a hard veto) -- cheap relative to
    # a 5-minute poll cycle, and it means `details` always reflects every
    # indicator that ran, not just whichever ones mattered for the final
    # pass/fail. Only these five feed the score below; the hard-veto keys
    # above are excluded from it (SOFT_CHECK_KEYS), preserving the original
    # score semantics exactly.

    bos = detect_break_of_structure(bars, swings)
    bos_matches = {"bullish": "bullish", "bearish": "bearish"}[direction]
    details["break_of_structure"] = "pass" if (bos is not None and bos.direction == bos_matches) else "fail"

    levels = cluster_levels(swings)
    blocking_kind = "resistance" if direction == "bullish" else "support"
    blocking_level = nearest_level(price, levels, blocking_kind)
    if blocking_level is None:
        details["support_resistance"] = "n/a"
    else:
        details["support_resistance"] = "fail" if distance_pct(price, blocking_level) < DEFAULT_SR_BLOCK_DISTANCE_PCT else "pass"

    zones = detect_zones(bars, fvg_lookback_period, fvg_body_multiplier)
    zone_kind = "demand" if direction == "bullish" else "supply"
    zone = nearest_zone(price, zones, zone_kind)
    if zone is None:
        details["supply_demand"] = "n/a"
    else:
        details["supply_demand"] = "pass" if price_in_zone(price, zone) else "fail"

    pools = find_liquidity_pools(swings)
    sweep = detect_liquidity_sweep(bars, pools)
    if sweep is None:
        details["liquidity_sweep"] = "n/a"
    else:
        details["liquidity_sweep"] = "pass" if sweep.direction == direction else "fail"

    if len(bars) < 20:
        details["volume_profile"] = "n/a"
    else:
        profile = compute_volume_profile(bars)
        # Trading OUTSIDE the value area suggests a directional break, not chop.
        details["volume_profile"] = "fail" if price_in_value_area(price, profile) else "pass"

    # A second, richer read of VPVR -- see node_quality_check's docstring.
    # Deliberately built from bars EXCLUDING the current one, not the same
    # `profile` above: the current bar is the FVG's own displacement
    # candle, which by definition just traded above-average volume (that's
    # the volume-confirmation requirement upstream) -- including it would
    # inflate its own bin and make "is this a low-volume node" fail almost
    # tautologically, on every single signal, regardless of the actual
    # HISTORICAL character of this price level before the move happened.
    # Direction-independent, same as volume_profile above: this asks
    # whether HERE is a decent-quality place to enter, not which way price
    # should go.
    prior_bars = bars.iloc[:-1]
    if len(prior_bars) < 20:
        details["vpvr_node_quality"] = "n/a"
    else:
        prior_profile = compute_volume_profile(prior_bars)
        details["vpvr_node_quality"] = node_quality_check(price, prior_profile)

    details["rsi_momentum"] = rsi_momentum_check(bars, direction, rsi_period)
    details["volatility_expansion"] = volatility_expansion_check(bars, atr_period)

    passes = sum(1 for k in SOFT_CHECK_KEYS if details[k] == "pass")
    fails = sum(1 for k in SOFT_CHECK_KEYS if details[k] == "fail")
    applicable = passes + fails
    score = (passes / applicable) if applicable > 0 else None

    # A hard veto blocks the trade regardless of how the soft checks scored
    # -- but `score`/`applicable` above are still the REAL computed soft
    # numbers, not None/0 placeholders, so a caller showing "everything we
    # looked at" can display them for reference even on a vetoed read.
    if veto_reason is not None:
        return ConfluenceResult(False, score, applicable, veto_reason, details)

    if applicable < min_applicable_checks:
        return ConfluenceResult(False, score, applicable, f"only {applicable} applicable confluence check(s), need >= {min_applicable_checks}", details)

    passed = score >= min_confluence_score
    reason = None if passed else f"confluence score {score:.0%} below required {min_confluence_score:.0%}"
    return ConfluenceResult(passed, score, applicable, reason, details)
