"""Walk-forward backtest of the FVG + confluence options strategy
(brain/options_strategy.py) against real historical daily bars.

READ THIS BEFORE TRUSTING ANY NUMBER THIS PRODUCES:

1. **This tests the REAL production decision logic**, not a reimplementation
   of it -- every bar calls `brain.options_strategy.decide_options_action()`
   directly, the exact function `orchestration/options_execution.py` calls
   live. If the strategy logic has a bug, this backtest will show it (that's
   the actual point of running it, per the request that led to this file).

2. **The option PRICES are theoretical, not real historical market prices**
   (see backtest/options_pricing.py's docstring for why: there's no free
   historical options data source available anywhere in this project's
   toolset). Entry/exit premiums are computed with Black-Scholes, spot price
   from real historical bars, strike = ATM at entry (matching
   data/options_data.py's live selection), and volatility = trailing
   realized volatility of the underlying, held CONSTANT for the life of each
   simulated trade. Real option prices would differ from this, sometimes
   substantially -- this cannot tell you the real historical dollar P&L of
   this strategy, only a theoretically-grounded approximation of it.

3. **No transaction costs, slippage, or bid-ask spread are modeled.** Real
   fills would be worse than these theoretical prices.

4. **Walk-forward, no lookahead**: at simulated bar i, the strategy only
   ever sees `bars.iloc[:i+1]` -- everything after that is invisible to it,
   the same as it would be live. This is the one thing this backtest DOES
   get right with confidence.

5. **`run_symbol_backtest`/`run_backtest` (daily bars) cannot distinguish
   same-day exits from overnight ones, at all** -- one confluence
   evaluation per calendar day. The live strategy (max_hold_days,
   target_dte_min=1/target_dte_max=2 in settings.yaml) is explicitly built
   around same-day and single-overnight holds -- something with no
   representation at daily granularity. Every simulated trade here
   effectively spans "opened day i, force-closed on day i+1"
   (max_hold_days=1 triggers on the very next daily bar, regardless of
   whether the real position would have been closed same-day by the FVG
   signal, or held overnight), so THIS PARTICULAR function cannot
   meaningfully validate the same-day-vs-overnight split, or trend
   invalidation (max_hold_days wins the race before trend has any real
   chance to move) -- use `run_symbol_backtest_intraday`/
   `run_backtest_intraday` (`--intraday` on the CLI) instead, which walks
   5-minute bars and can represent both. Real constraint on THAT tool
   though: yfinance's 5-minute history caps out around ~60 days, so it can
   only ever cover a recent window, not a multi-year one like the daily
   version -- there's no free source of longer intraday history available
   to this project. Treat any number from either version for this
   strategy shape as even rougher than caveat 2 already implies.

What this IS good for: catching bugs in the decision logic (crashes, dumb
outputs, a threshold that never triggers or always triggers), sanity-
checking how selective the confluence gate actually is in practice, and a
rough, clearly-approximate read on whether the underlying signal has any
historical edge at all. What it is NOT: a number you should size real
capital off of.

Usage (needs a real Python environment with pandas/numpy/yfinance
installed -- this cannot run in a sandbox with no network or Python at
all, same limitation flagged throughout this project's other backtest/
files):

    PYTHONPATH=. python3 backtest/options_strategy_backtest.py \\
        --symbols SPY QQQ AAPL NVDA \\
        --start 2022-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Literal, Optional

import pandas as pd

from backtest.metrics import avg_win_loss, max_consecutive_losses, max_drawdown, profit_factor, win_rate
from backtest.options_pricing import black_scholes_price, realized_volatility
from brain.confluence import evaluate_confluence
from brain.options_strategy import OptionsDecision, decide_options_action
from brain.sweep_strategy import decide_sweep_action
from brain.orb_strategy import decide_orb_action, session_vwap, todays_bars
from orchestration.market_hours import MARKET_TZ
from config.config_loader import Settings, load_settings
from data.fetchers import YFinanceHistoricalFetcher

logger = logging.getLogger(__name__)

DEFAULT_RISK_FREE_RATE = 0.045  # rough constant assumption -- not pulled from a live rate anywhere
DEFAULT_WARMUP_BARS = 210        # sma_period (200) + a little slack for swing-point warmup


@dataclass
class SimulatedTrade:
    symbol: str
    option_type: Literal["call", "put"]
    entry_date: date
    entry_spot: float
    strike: float
    expiration_date: date
    entry_iv: float
    entry_premium: float
    gap_low: Optional[float] = None   # the triggering FVG's bounds, for the same structural stop the live executor uses
    gap_high: Optional[float] = None
    exit_date: Optional[date] = None
    exit_reason: Optional[str] = None
    exit_premium: Optional[float] = None
    # The confluence read-out at entry (check name -> "pass"/"fail"/"n/a"),
    # kept so per-indicator attribution can be done after the fact -- which
    # checks were passing on the trades that won vs lost. Diagnostic only.
    entry_details: Dict[str, str] = field(default_factory=dict)
    # Sweep model: sizing tier and the entry-time invalidation/target.
    tier: Optional[str] = None
    invalidation_price: Optional[float] = None
    target_price: Optional[float] = None

    # ORB model: premium at which the first half was taken at target_r
    # (None = no partial). pnl_pct then averages the two legs.
    partial_exit_premium: Optional[float] = None

    @property
    def size_weight(self) -> float:
        """Half-size trades count half in pooled P&L (conviction 0.5)."""
        return 0.5 if self.tier == "half" else 1.0

    @property
    def is_closed(self) -> bool:
        return self.exit_premium is not None

    @property
    def pnl_pct(self) -> Optional[float]:
        if not self.is_closed or self.entry_premium <= 0:
            return None
        if self.partial_exit_premium is not None:
            return 0.5 * (self.partial_exit_premium / self.entry_premium - 1) + 0.5 * (self.exit_premium / self.entry_premium - 1)
        return self.exit_premium / self.entry_premium - 1


@dataclass
class SymbolBacktestResult:
    symbol: str
    trades: List[SimulatedTrade] = field(default_factory=list)
    bars_evaluated: int = 0
    fvg_triggers: int = 0             # a fair value gap formed AND was volume-confirmed
    confluence_rejections: int = 0    # triggered, but the confluence gate said no
    trades_opened: int = 0


def _target_dte(settings: Settings) -> int:
    return settings.options.target_dte_min + (settings.options.target_dte_max - settings.options.target_dte_min) // 2


def _price_position(trade: SimulatedTrade, spot: float, current_date: date, risk_free_rate: float) -> float:
    years_to_expiry = max((trade.expiration_date - current_date).days, 0) / 365
    return black_scholes_price(spot, trade.strike, years_to_expiry, risk_free_rate, trade.entry_iv, trade.option_type)


def run_symbol_backtest(
    symbol: str,
    bars: pd.DataFrame,
    settings: Settings,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
) -> SymbolBacktestResult:
    opt = settings.options
    result = SymbolBacktestResult(symbol=symbol)
    open_trade: Optional[SimulatedTrade] = None
    just_closed_this_bar = False

    # Bounded window (400 bars ~ orchestration/run_live.py's DAILY_LOOKBACK_DAYS),
    # not the full history-to-date -- two reasons, not just speed. (1) The
    # unbounded `bars.iloc[:i+1]` made every bar's confluence computation
    # O(window size), so the whole backtest was O(n^2) in the number of
    # bars -- a 20-symbol/5-year run took 40+ minutes for exactly this
    # reason. (2) More importantly, an unbounded window is NOT what the
    # live bot ever sees: run_live.py always fetches a capped lookback
    # (400 daily bars), so testing against ever-growing history was
    # already a live/backtest fidelity gap, independent of speed -- this
    # fix corrects both at once, not just the slow one.
    WINDOW_BARS = 400
    for i in range(warmup_bars, len(bars)):
        window = bars.iloc[max(0, i + 1 - WINDOW_BARS): i + 1]
        current_date = bars.index[i].date() if hasattr(bars.index[i], "date") else bars.index[i]
        spot = float(bars["close"].iloc[i])
        result.bars_evaluated += 1
        just_closed_this_bar = False

        # Unconditional holding-time cap, checked FIRST -- matches
        # orchestration/options_execution.py's ordering. See this module's
        # docstring (caveat 5) for why daily-bar granularity can't actually
        # distinguish a same-day exit from an overnight one: this will
        # almost always be what closes a trade, on the very next bar.
        if open_trade is not None:
            days_held = (current_date - open_trade.entry_date).days
            if days_held >= opt.max_hold_days:
                open_trade.exit_date, open_trade.exit_reason = current_date, "max_hold"
                open_trade.exit_premium = _price_position(open_trade, spot, current_date, risk_free_rate)
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True

        if open_trade is not None:
            days_left = (open_trade.expiration_date - current_date).days
            if days_left <= opt.close_before_expiration_days:
                open_trade.exit_date, open_trade.exit_reason = current_date, "expiration"
                open_trade.exit_premium = _price_position(open_trade, spot, current_date, risk_free_rate)
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True

        # Structural stop, checked before the dollar-based one, matching
        # orchestration/options_execution.py's ordering: has price closed
        # back through the ENTIRE gap that triggered this trade?
        if open_trade is not None and open_trade.gap_low is not None and open_trade.gap_high is not None:
            invalidated = (
                (open_trade.option_type == "call" and spot < open_trade.gap_low) or
                (open_trade.option_type == "put" and spot > open_trade.gap_high)
            )
            if invalidated:
                open_trade.exit_date, open_trade.exit_reason = current_date, "fvg_invalidated"
                open_trade.exit_premium = _price_position(open_trade, spot, current_date, risk_free_rate)
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True

        # Trend invalidation -- matches orchestration/options_execution.py's
        # _check_trend_invalidation exactly: re-run the SAME hard-veto
        # checks (trend_200sma/market_structure/elliott_wave) a fresh entry
        # would face, against this trade's own direction. Any one failing
        # means the premise is gone -- close regardless of P&L. This is the
        # ONLY profit/loss exit now (stop_loss_pct/take_profit_pct no
        # longer drive any exit, live or here -- see that method's
        # docstring for why NOT `result.veto_reason is not None` alone,
        # which also fires on a merely-low soft score).
        if open_trade is not None:
            direction = "bullish" if open_trade.option_type == "call" else "bearish"
            confluence = evaluate_confluence(
                window, direction, sma_period=opt.sma_period, min_confluence_score=opt.min_confluence_score,
                fvg_lookback_period=opt.fvg_lookback_period, fvg_body_multiplier=opt.fvg_body_multiplier,
                policy=opt.confluence_policy(),
            )
            if confluence.hard_vetoed:
                open_trade.exit_date, open_trade.exit_reason = current_date, "trend_invalidated"
                open_trade.exit_premium = _price_position(open_trade, spot, current_date, risk_free_rate)
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True

        if open_trade is not None:
            current_value = _price_position(open_trade, spot, current_date, risk_free_rate)
            pnl_pct = (current_value / open_trade.entry_premium) - 1 if open_trade.entry_premium > 0 else 0
            # "Close if going nowhere" -- matches
            # orchestration/options_execution.py's stagnation check exactly
            # (same fraction-of-own-DTE threshold, same P&L bar), only
            # reachable here too if trend invalidation didn't already fire
            # this bar.
            dte_at_entry = (open_trade.expiration_date - open_trade.entry_date).days
            days_held = (current_date - open_trade.entry_date).days
            if days_held >= opt.stagnant_exit_hold_fraction * dte_at_entry and pnl_pct < opt.stagnant_exit_min_pnl_pct:
                open_trade.exit_date, open_trade.exit_reason = current_date, "stagnant"
                open_trade.exit_premium = current_value
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True

        decision = decide_options_action(
            symbol, window, opt.fvg_lookback_period, opt.fvg_body_multiplier, opt.fvg_volume_multiplier,
            opt.sma_period, opt.min_confluence_score, min_gap_atr_multiplier=opt.fvg_min_gap_atr_multiplier,
            policy=opt.confluence_policy(),
        )

        if "no fair value gap" not in decision.reasoning and "didn't confirm it" not in decision.reasoning:
            if decision.action in ("buy_call", "buy_put"):
                result.fvg_triggers += 1
            elif "confluence check failed" in decision.reasoning:
                result.fvg_triggers += 1
                result.confluence_rejections += 1

        if open_trade is not None and decision.action in ("buy_call", "buy_put"):
            wanted_type = "call" if decision.action == "buy_call" else "put"
            if wanted_type != open_trade.option_type:
                open_trade.exit_date, open_trade.exit_reason = current_date, "flip"
                open_trade.exit_premium = _price_position(open_trade, spot, current_date, risk_free_rate)
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True

        if open_trade is None and not just_closed_this_bar and decision.action in ("buy_call", "buy_put"):
            wanted_type = "call" if decision.action == "buy_call" else "put"
            expiration = current_date + timedelta(days=_target_dte(settings))
            iv = realized_volatility(window)
            try:
                entry_premium = black_scholes_price(spot, spot, (expiration - current_date).days / 365, risk_free_rate, iv, wanted_type)
            except ValueError as exc:
                # Degenerate pricing inputs (e.g. ~zero trailing volatility) --
                # skip opening a trade this bar rather than crash the whole run.
                logger.warning("%s @ %s: skipping trade, couldn't price entry (%s)", symbol, current_date, exc)
                continue
            if entry_premium <= 0:
                logger.warning("%s @ %s: skipping trade, theoretical entry premium was %.4f", symbol, current_date, entry_premium)
                continue
            open_trade = SimulatedTrade(
                symbol=symbol, option_type=wanted_type, entry_date=current_date, entry_spot=spot,
                strike=spot, expiration_date=expiration, entry_iv=iv, entry_premium=entry_premium,
                gap_low=decision.gap_low, gap_high=decision.gap_high,
                entry_details=dict(decision.confluence_details),
            )
            result.trades_opened += 1

    if open_trade is not None:
        result.trades.append(open_trade)  # still open at backtest end -- pnl_pct is None, excluded from closed-trade stats

    return result


DEFAULT_INTRADAY_WARMUP_BARS = 30  # fvg_lookback_period + swing-point slack -- far smaller than daily's 210, since this is only about bar-count for FVG/swing detection; the 200-SMA veto and realized-vol IV proxy come from daily_bars, not this window
INTRADAY_WINDOW_BARS = 400          # matches production's bounded live intraday window, same constant as run_symbol_backtest's daily WINDOW_BARS


def _completed_before(bars: Optional[pd.DataFrame], bar_length: timedelta, ts) -> Optional[pd.DataFrame]:
    """The prefix of `bars` whose candles had fully CLOSED by `ts` -- a
    1h/4h bar that's still forming at the current 5-minute bar's time is
    not something the live routine can see either (orchestration/
    bar_cache.drop_incomplete_last_bar), so the walk-forward must not
    peek at it. None stays None (trend read reports n/a, fails open)."""
    if bars is None or bars.empty:
        return None
    n = bars.index.searchsorted(ts - bar_length, side="right")
    return bars.iloc[:n] if n > 0 else None


def resample_ohlcv(bars: pd.DataFrame, rule: str, offset: Optional[str] = None) -> pd.DataFrame:
    """Aggregates finer bars into coarser ones (e.g. 1h -> 4h). yfinance
    has no native 4h interval, so the backtest builds it from 60m bars;
    `offset="9h30min"` aligns bins to the US regular-session open so a
    4h bar means 9:30-13:30 / 13:30-16:00 ET, matching Robinhood's own
    regular-session 4hour bars (2 per day) that the live routine feeds."""
    agg = bars.resample(rule, offset=offset, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return agg.dropna(subset=["open"])


def run_symbol_backtest_intraday(
    symbol: str,
    intraday_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    settings: Settings,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    warmup_bars: int = DEFAULT_INTRADAY_WARMUP_BARS,
    hourly_bars: Optional[pd.DataFrame] = None,
    four_hour_bars: Optional[pd.DataFrame] = None,
    decision_hook: Optional[Callable[[pd.Timestamp, OptionsDecision], None]] = None,
) -> SymbolBacktestResult:
    """Same walk-forward simulation as run_symbol_backtest, but at intraday
    (typically 5-minute) bar resolution instead of daily -- the only way to
    meaningfully test max_hold_days/same-day exits and trend-invalidation
    at all, since a daily backtest evaluates confluence once per calendar
    day and max_hold_days=1 always wins the race before trend has any real
    chance to move (see this module's docstring, caveat 5).

    `daily_bars` is used ONLY for the 200-SMA veto and the realized-vol IV
    proxy (both need daily granularity to mean what they're supposed to
    mean, e.g. realized_volatility() annualizes assuming each bar is one
    trading day) -- never for FVG/market-structure/confluence detection
    itself, matching exactly how orchestration/evaluate_for_agent.py and
    run_live.py split these two inputs live. `daily_window` is sliced to
    STRICTLY BEFORE the current intraday bar's calendar date, so "today"'s
    not-yet-complete daily candle never leaks into either calculation --
    recomputed once per calendar day (not once per intraday bar) since it
    only actually changes then.

    `hourly_bars`/`four_hour_bars` feed the trend_1h/trend_4h HARD vetoes
    (brain/confluence.py) for both entry gating and trend-invalidation
    exits, sliced per bar to candles that had fully closed by the current
    5-minute bar's time (_completed_before) -- same no-peeking rule as the
    daily slice. Omit either and that veto reports n/a (fails open), which
    is a materially more permissive strategy than production runs, so
    run_backtest_intraday always supplies both.
    """
    opt = settings.options
    result = SymbolBacktestResult(symbol=symbol)
    open_trade: Optional[SimulatedTrade] = None

    last_date = None
    daily_window = daily_bars.iloc[:0]

    for i in range(warmup_bars, len(intraday_bars)):
        window = intraday_bars.iloc[max(0, i + 1 - INTRADAY_WINDOW_BARS): i + 1]
        bar_ts = intraday_bars.index[i]
        current_date = bar_ts.date() if hasattr(bar_ts, "date") else bar_ts
        spot = float(intraday_bars["close"].iloc[i])
        result.bars_evaluated += 1
        just_closed_this_bar = False

        if current_date != last_date:
            daily_window = daily_bars[daily_bars.index.date < current_date]
            last_date = current_date
        hourly_window = _completed_before(hourly_bars, timedelta(hours=1), bar_ts)
        four_hour_window = _completed_before(four_hour_bars, timedelta(hours=4), bar_ts)

        if open_trade is not None:
            days_held = (current_date - open_trade.entry_date).days
            if days_held >= opt.max_hold_days:
                open_trade.exit_date, open_trade.exit_reason = current_date, "max_hold"
                open_trade.exit_premium = _price_position(open_trade, spot, current_date, risk_free_rate)
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True

        if open_trade is not None:
            days_left = (open_trade.expiration_date - current_date).days
            if days_left <= opt.close_before_expiration_days:
                open_trade.exit_date, open_trade.exit_reason = current_date, "expiration"
                open_trade.exit_premium = _price_position(open_trade, spot, current_date, risk_free_rate)
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True

        if open_trade is not None and open_trade.gap_low is not None and open_trade.gap_high is not None:
            invalidated = (
                (open_trade.option_type == "call" and spot < open_trade.gap_low) or
                (open_trade.option_type == "put" and spot > open_trade.gap_high)
            )
            if invalidated:
                open_trade.exit_date, open_trade.exit_reason = current_date, "fvg_invalidated"
                open_trade.exit_premium = _price_position(open_trade, spot, current_date, risk_free_rate)
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True

        if open_trade is not None:
            direction = "bullish" if open_trade.option_type == "call" else "bearish"
            confluence = evaluate_confluence(
                window, direction, sma_period=opt.sma_period, min_confluence_score=opt.min_confluence_score,
                fvg_lookback_period=opt.fvg_lookback_period, fvg_body_multiplier=opt.fvg_body_multiplier,
                daily_bars=daily_window if not daily_window.empty else None,
                hourly_bars=hourly_window, four_hour_bars=four_hour_window,
                trend_1h_period=opt.trend_1h_period, trend_4h_period=opt.trend_4h_period,
                policy=opt.confluence_policy(),
            )
            if confluence.hard_vetoed:
                open_trade.exit_date, open_trade.exit_reason = current_date, "trend_invalidated"
                open_trade.exit_premium = _price_position(open_trade, spot, current_date, risk_free_rate)
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True

        if open_trade is not None:
            current_value = _price_position(open_trade, spot, current_date, risk_free_rate)
            pnl_pct = (current_value / open_trade.entry_premium) - 1 if open_trade.entry_premium > 0 else 0
            dte_at_entry = (open_trade.expiration_date - open_trade.entry_date).days
            days_held = (current_date - open_trade.entry_date).days
            if days_held >= opt.stagnant_exit_hold_fraction * dte_at_entry and pnl_pct < opt.stagnant_exit_min_pnl_pct:
                open_trade.exit_date, open_trade.exit_reason = current_date, "stagnant"
                open_trade.exit_premium = current_value
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True

        # Same entry-session gate as the executor (exits above are unaffected).
        et = bar_ts.tz_convert(MARKET_TZ) if bar_ts.tzinfo is not None else bar_ts.tz_localize("UTC").tz_convert(MARKET_TZ)
        if not (opt.entry_session_start <= et.strftime("%H:%M") < opt.entry_session_end):
            continue

        decision = decide_options_action(
            symbol, window, opt.fvg_lookback_period, opt.fvg_body_multiplier, opt.fvg_volume_multiplier,
            opt.sma_period, opt.min_confluence_score,
            daily_bars=daily_window if not daily_window.empty else None,
            min_gap_atr_multiplier=opt.fvg_min_gap_atr_multiplier,
            hourly_bars=hourly_window, four_hour_bars=four_hour_window,
            trend_1h_period=opt.trend_1h_period, trend_4h_period=opt.trend_4h_period,
            policy=opt.confluence_policy(),
        )

        if "no fair value gap" not in decision.reasoning and "didn't confirm it" not in decision.reasoning:
            if decision.action in ("buy_call", "buy_put"):
                result.fvg_triggers += 1
            elif "confluence check failed" in decision.reasoning:
                result.fvg_triggers += 1
                result.confluence_rejections += 1
            # Diagnostic tap: every decision that carried a confluence
            # read (i.e. the FVG+volume trigger fired), taken or rejected.
            if decision_hook is not None and decision.confluence_details:
                decision_hook(bar_ts, decision)

        if open_trade is not None and decision.action in ("buy_call", "buy_put"):
            wanted_type = "call" if decision.action == "buy_call" else "put"
            if wanted_type != open_trade.option_type:
                open_trade.exit_date, open_trade.exit_reason = current_date, "flip"
                open_trade.exit_premium = _price_position(open_trade, spot, current_date, risk_free_rate)
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True

        if open_trade is None and not just_closed_this_bar and decision.action in ("buy_call", "buy_put"):
            wanted_type = "call" if decision.action == "buy_call" else "put"
            expiration = current_date + timedelta(days=_target_dte(settings))
            iv = realized_volatility(daily_window) if len(daily_window) >= 2 else 0.20
            try:
                entry_premium = black_scholes_price(spot, spot, (expiration - current_date).days / 365, risk_free_rate, iv, wanted_type)
            except ValueError as exc:
                logger.warning("%s @ %s: skipping trade, couldn't price entry (%s)", symbol, current_date, exc)
                continue
            if entry_premium <= 0:
                logger.warning("%s @ %s: skipping trade, theoretical entry premium was %.4f", symbol, current_date, entry_premium)
                continue
            open_trade = SimulatedTrade(
                symbol=symbol, option_type=wanted_type, entry_date=current_date, entry_spot=spot,
                strike=spot, expiration_date=expiration, entry_iv=iv, entry_premium=entry_premium,
                gap_low=decision.gap_low, gap_high=decision.gap_high,
                entry_details=dict(decision.confluence_details),
            )
            result.trades_opened += 1

    if open_trade is not None:
        result.trades.append(open_trade)

    return result


def run_symbol_backtest_sweep(
    symbol: str,
    intraday_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    settings: Settings,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    warmup_bars: int = DEFAULT_INTRADAY_WARMUP_BARS,
    hourly_bars: Optional[pd.DataFrame] = None,
    four_hour_bars: Optional[pd.DataFrame] = None,
    decision_hook: Optional[Callable[[pd.Timestamp, OptionsDecision], None]] = None,
) -> SymbolBacktestResult:
    """Walk-forward of the SWEEP model (brain/sweep_strategy.py) at 5-minute
    resolution, mirroring orchestration/options_execution.py's sweep-mode
    rules exactly: entries only inside options.entry_session_start/end
    (ET), at most max_entries_per_day, none after the daily loss limit;
    exits in order max_hold -> expiration -> sweep_invalidated (close
    beyond the sweep wick) -> target_reached -> stagnant. Half-size trades
    carry size_weight 0.5 in pooled P&L.

    Same pricing caveats as the rest of this module (theoretical premiums,
    no costs). Realized daily P&L for the loss limit is measured on the
    same theoretical premiums, as a fraction of one unit of equity.
    """
    opt = settings.options
    cfg = opt.sweep_config()
    result = SymbolBacktestResult(symbol=symbol)
    open_trade: Optional[SimulatedTrade] = None
    last_date = None
    daily_window = daily_bars.iloc[:0]
    entries_today = 0
    day_pnl = 0.0  # sum of size-weighted pnl_pct closed today, as a fraction of the per-trade budget
    equity_units = 1.0 / max(opt.max_premium_pct_per_trade, 1e-9)  # one full-size trade = max_premium_pct_per_trade of equity

    for i in range(warmup_bars, len(intraday_bars)):
        window = intraday_bars.iloc[max(0, i + 1 - INTRADAY_WINDOW_BARS): i + 1]
        bar_ts = intraday_bars.index[i]
        et = bar_ts.tz_convert(MARKET_TZ) if bar_ts.tzinfo is not None else bar_ts.tz_localize("UTC").tz_convert(MARKET_TZ)
        current_date = et.date()
        spot = float(intraday_bars["close"].iloc[i])
        result.bars_evaluated += 1
        just_closed_this_bar = False

        if current_date != last_date:
            daily_window = daily_bars[daily_bars.index.date < current_date]
            last_date = current_date
            entries_today, day_pnl = 0, 0.0

        def _close_trade(reason: str, premium: Optional[float] = None):
            nonlocal open_trade, just_closed_this_bar, day_pnl
            open_trade.exit_date, open_trade.exit_reason = current_date, reason
            open_trade.exit_premium = premium if premium is not None else _price_position(open_trade, spot, current_date, risk_free_rate)
            result.trades.append(open_trade)
            if open_trade.pnl_pct is not None:
                day_pnl += open_trade.pnl_pct * open_trade.size_weight
            open_trade, just_closed_this_bar = None, True

        if open_trade is not None and (current_date - open_trade.entry_date).days >= opt.max_hold_days:
            _close_trade("max_hold")
        if open_trade is not None and (open_trade.expiration_date - current_date).days <= opt.close_before_expiration_days:
            _close_trade("expiration")
        if open_trade is not None and open_trade.invalidation_price is not None:
            if (open_trade.option_type == "call" and spot < open_trade.invalidation_price) or (open_trade.option_type == "put" and spot > open_trade.invalidation_price):
                _close_trade("sweep_invalidated")
        if open_trade is not None and open_trade.target_price is not None:
            if (open_trade.option_type == "call" and spot >= open_trade.target_price) or (open_trade.option_type == "put" and spot <= open_trade.target_price):
                _close_trade("target_reached")
        if open_trade is not None:
            current_value = _price_position(open_trade, spot, current_date, risk_free_rate)
            pnl_pct = (current_value / open_trade.entry_premium) - 1 if open_trade.entry_premium > 0 else 0
            dte_at_entry = (open_trade.expiration_date - open_trade.entry_date).days
            days_held = (current_date - open_trade.entry_date).days
            if days_held >= opt.stagnant_exit_hold_fraction * dte_at_entry and pnl_pct < opt.stagnant_exit_min_pnl_pct:
                _close_trade("stagnant", current_value)

        hhmm = et.strftime("%H:%M")
        in_session = opt.entry_session_start <= hhmm < opt.entry_session_end
        loss_limited = opt.daily_loss_limit_pct > 0 and (day_pnl / equity_units) <= -opt.daily_loss_limit_pct
        if not in_session or open_trade is not None or just_closed_this_bar:
            continue
        if opt.max_entries_per_day and entries_today >= opt.max_entries_per_day:
            continue
        if loss_limited:
            continue

        decision = decide_sweep_action(
            symbol, window, bar_ts.to_pydatetime(), daily_bars=daily_window if not daily_window.empty else None,
            hourly_bars=_completed_before(hourly_bars, timedelta(hours=1), bar_ts),
            four_hour_bars=_completed_before(four_hour_bars, timedelta(hours=4), bar_ts), cfg=cfg,
        )
        if decision.confluence_details.get("sweep") == "pass":
            result.fvg_triggers += 1  # "triggers" = sweeps seen, for this model
            if decision.action == "hold":
                result.confluence_rejections += 1
        if decision_hook is not None and decision.confluence_details:
            decision_hook(bar_ts, decision)
        if decision.action not in ("buy_call", "buy_put"):
            continue

        wanted_type = "call" if decision.action == "buy_call" else "put"
        expiration = current_date + timedelta(days=_target_dte(settings))
        iv = realized_volatility(daily_window) if len(daily_window) >= 2 else 0.20
        try:
            entry_premium = black_scholes_price(spot, spot, (expiration - current_date).days / 365, risk_free_rate, iv, wanted_type)
        except ValueError as exc:
            logger.warning("%s @ %s: skipping trade, couldn't price entry (%s)", symbol, current_date, exc)
            continue
        if entry_premium <= 0:
            continue
        open_trade = SimulatedTrade(
            symbol=symbol, option_type=wanted_type, entry_date=current_date, entry_spot=spot, strike=spot,
            expiration_date=expiration, entry_iv=iv, entry_premium=entry_premium,
            gap_low=decision.gap_low, gap_high=decision.gap_high, entry_details=dict(decision.confluence_details),
            tier=decision.tier, invalidation_price=decision.invalidation_price, target_price=decision.target_price,
        )
        result.trades_opened += 1
        entries_today += 1

    if open_trade is not None:
        result.trades.append(open_trade)
    return result


def run_symbol_backtest_orb(
    symbol: str,
    intraday_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    settings: Settings,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    warmup_bars: int = DEFAULT_INTRADAY_WARMUP_BARS,
    hourly_bars: Optional[pd.DataFrame] = None,
    four_hour_bars: Optional[pd.DataFrame] = None,
    decision_hook: Optional[Callable[[pd.Timestamp, OptionsDecision], None]] = None,
) -> SymbolBacktestResult:
    """Walk-forward of the ORB model (brain/orb_strategy.py), mirroring
    orchestration/options_execution.py's orb-mode rules: entries only in
    the entry session, protection gates, exits time_stop -> stop_hit ->
    target partial (half at target_r x R, modelled as averaging two legs)
    -> trail (VWAP / 2-bar) -> and the generic max_hold/expiration
    backstops. Always flat by time_stop, so DTE only affects pricing."""
    opt = settings.options
    cfg = opt.orb_config()
    result = SymbolBacktestResult(symbol=symbol)
    open_trade: Optional[SimulatedTrade] = None
    partial_taken = False
    last_date = None
    daily_window = daily_bars.iloc[:0]
    entries_today = 0
    day_pnl = 0.0
    equity_units = 1.0 / max(opt.max_premium_pct_per_trade, 1e-9)

    for i in range(warmup_bars, len(intraday_bars)):
        window = intraday_bars.iloc[max(0, i + 1 - INTRADAY_WINDOW_BARS): i + 1]
        bar_ts = intraday_bars.index[i]
        et = bar_ts.tz_convert(MARKET_TZ) if bar_ts.tzinfo is not None else bar_ts.tz_localize("UTC").tz_convert(MARKET_TZ)
        current_date = et.date()
        hhmm = et.strftime("%H:%M")
        spot = float(intraday_bars["close"].iloc[i])
        result.bars_evaluated += 1
        just_closed_this_bar = False

        if current_date != last_date:
            daily_window = daily_bars[daily_bars.index.date < current_date]
            last_date = current_date
            entries_today, day_pnl = 0, 0.0

        def _close_trade(reason: str, premium: Optional[float] = None):
            nonlocal open_trade, just_closed_this_bar, day_pnl, partial_taken
            open_trade.exit_date, open_trade.exit_reason = current_date, reason
            open_trade.exit_premium = premium if premium is not None else _price_position(open_trade, spot, current_date, risk_free_rate)
            result.trades.append(open_trade)
            if open_trade.pnl_pct is not None:
                day_pnl += open_trade.pnl_pct * open_trade.size_weight
            open_trade, just_closed_this_bar, partial_taken = None, True, False

        if open_trade is not None:
            is_call = open_trade.option_type == "call"
            if hhmm >= opt.time_stop:
                _close_trade("time_stop")
            elif (current_date - open_trade.entry_date).days >= opt.max_hold_days:
                _close_trade("max_hold")
            elif (open_trade.expiration_date - current_date).days <= opt.close_before_expiration_days:
                _close_trade("expiration")
            elif open_trade.invalidation_price is not None and ((is_call and spot < open_trade.invalidation_price) or (not is_call and spot > open_trade.invalidation_price)):
                _close_trade("stop_hit")
            elif open_trade.target_price is not None and not partial_taken and ((is_call and spot >= open_trade.target_price) or (not is_call and spot <= open_trade.target_price)):
                open_trade.partial_exit_premium = _price_position(open_trade, spot, current_date, risk_free_rate)
                partial_taken = True
            elif partial_taken:
                todays = todays_bars(window, bar_ts.to_pydatetime())
                if len(todays) >= 3:
                    vwap_now = float(session_vwap(todays).iloc[-1])
                    prior2 = todays.iloc[-3:-1]
                    if is_call and (spot < vwap_now or spot < float(prior2["low"].min())):
                        _close_trade("trail_exit")
                    elif not is_call and (spot > vwap_now or spot > float(prior2["high"].max())):
                        _close_trade("trail_exit")

        in_session = opt.entry_session_start <= hhmm < opt.entry_session_end
        loss_limited = opt.daily_loss_limit_pct > 0 and (day_pnl / equity_units) <= -opt.daily_loss_limit_pct
        if not in_session or open_trade is not None or just_closed_this_bar or loss_limited:
            continue
        if opt.max_entries_per_day and entries_today >= opt.max_entries_per_day:
            continue

        decision = decide_orb_action(symbol, window, bar_ts.to_pydatetime(), cfg=cfg)
        if decision.confluence_details.get("breakout") == "pass":
            result.fvg_triggers += 1  # "triggers" = qualifying breakouts seen (repeats while it stays in the lookback)
            if decision.action == "hold":
                result.confluence_rejections += 1
        if decision_hook is not None and decision.confluence_details:
            decision_hook(bar_ts, decision)
        if decision.action not in ("buy_call", "buy_put"):
            continue

        wanted_type = "call" if decision.action == "buy_call" else "put"
        expiration = current_date + timedelta(days=_target_dte(settings))
        iv = realized_volatility(daily_window) if len(daily_window) >= 2 else 0.20
        try:
            entry_premium = black_scholes_price(spot, spot, (expiration - current_date).days / 365, risk_free_rate, iv, wanted_type)
        except ValueError:
            continue
        if entry_premium <= 0:
            continue
        open_trade = SimulatedTrade(
            symbol=symbol, option_type=wanted_type, entry_date=current_date, entry_spot=spot, strike=spot,
            expiration_date=expiration, entry_iv=iv, entry_premium=entry_premium,
            gap_low=decision.gap_low, gap_high=decision.gap_high, entry_details=dict(decision.confluence_details),
            tier=decision.tier, invalidation_price=decision.invalidation_price, target_price=decision.target_price,
        )
        partial_taken = False
        result.trades_opened += 1
        entries_today += 1

    if open_trade is not None:
        result.trades.append(open_trade)
    return result


@dataclass
class BacktestReport:
    per_symbol: Dict[str, SymbolBacktestResult]
    pooled_closed_pnls: List[float]
    pooled_equity_curve: pd.Series


def run_backtest(symbols: List[str], start: str, end: str, settings: Settings) -> BacktestReport:
    fetcher = YFinanceHistoricalFetcher()
    per_symbol: Dict[str, SymbolBacktestResult] = {}
    all_closed: List[SimulatedTrade] = []

    for symbol in symbols:
        bars = fetcher.get_bars(symbol, start, end, timeframe="1D")
        result = run_symbol_backtest(symbol, bars, settings)
        per_symbol[symbol] = result
        all_closed.extend(t for t in result.trades if t.is_closed)

    all_closed = [t for t in all_closed if t.pnl_pct is not None]  # guards a degenerate entry_premium == 0 edge case
    all_closed.sort(key=lambda t: t.exit_date)
    pooled_pnls = [t.pnl_pct for t in all_closed]

    # Simplification, flagged: treats every closed trade (across every
    # symbol) as if it used 100% of one shared capital pool, one at a time,
    # in exit-date order -- ignores that several symbols can have positions
    # open concurrently in reality. A rough shape for the equity curve, not
    # a real multi-asset capital-allocation simulation.
    equity = [1.0]
    for pnl in pooled_pnls:
        equity.append(equity[-1] * (1 + pnl))
    equity_curve = pd.Series(equity[1:], index=[t.exit_date for t in all_closed]) if all_closed else pd.Series([1.0])

    return BacktestReport(per_symbol, pooled_pnls, equity_curve)


DEFAULT_INTRADAY_DAYS = 58     # yfinance's 5-minute history caps out around ~60 days -- see data/fetchers.py; 58 leaves a little slack
DEFAULT_DAILY_LOOKBACK_DAYS = 500  # comfortably covers a 200-day SMA with warmup room, matches orchestration/run_live.py's DAILY_LOOKBACK_DAYS order of magnitude
HOURLY_WARMUP_DAYS = 60            # extra 1h history before the 5-minute window, so the resampled 4h SMA(20) is warm from bar one -- mirrors bar_cache's 4hour backfill_days


def run_backtest_intraday(
    symbols: List[str], settings: Settings,
    intraday_days: int = DEFAULT_INTRADAY_DAYS, daily_lookback_days: int = DEFAULT_DAILY_LOOKBACK_DAYS,
) -> BacktestReport:
    """Same shape as run_backtest, but walks 5-minute bars via
    run_symbol_backtest_intraday -- see that function's docstring for why
    this exists. `intraday_days` is hard-capped by yfinance's own ~60-day
    limit on intraday history (data/fetchers.py's YFinanceHistoricalFetcher
    docstring) -- asking for more here won't actually get more, yfinance
    will just return what it has. This means, unlike run_backtest, this
    can only ever cover a recent ~2-month window, not a multi-year one --
    a real, unavoidable constraint of the only free data source this
    project has, not a choice.
    """
    fetcher = YFinanceHistoricalFetcher()
    end = date.today()
    intraday_start = end - timedelta(days=intraday_days)
    daily_start = end - timedelta(days=daily_lookback_days)
    # 1h bars start well before the 5-minute window so the 4h SMA(20)
    # (~10 trading days of 2 bars/day) is already warm on the first
    # evaluated bar. yfinance keeps ~730 days of 60m history, so no cap issue.
    hourly_start = intraday_start - timedelta(days=HOURLY_WARMUP_DAYS)

    per_symbol: Dict[str, SymbolBacktestResult] = {}
    all_closed: List[SimulatedTrade] = []

    for symbol in symbols:
        intraday_bars = fetcher.get_bars(symbol, intraday_start.isoformat(), end.isoformat(), timeframe="5m")
        daily_bars = fetcher.get_bars(symbol, daily_start.isoformat(), end.isoformat(), timeframe="1D")
        hourly_bars = fetcher.get_bars(symbol, hourly_start.isoformat(), end.isoformat(), timeframe="1h")
        four_hour_bars = resample_ohlcv(hourly_bars, "4h", offset="9h30min")
        walk = {"sweep": run_symbol_backtest_sweep, "orb": run_symbol_backtest_orb}.get(settings.options.model, run_symbol_backtest_intraday)
        result = walk(
            symbol, intraday_bars, daily_bars, settings, hourly_bars=hourly_bars, four_hour_bars=four_hour_bars,
        )
        per_symbol[symbol] = result
        all_closed.extend(t for t in result.trades if t.is_closed)

    all_closed = [t for t in all_closed if t.pnl_pct is not None]
    all_closed.sort(key=lambda t: t.exit_date)
    pooled_pnls = [t.pnl_pct for t in all_closed]

    equity = [1.0]
    for pnl in pooled_pnls:
        equity.append(equity[-1] * (1 + pnl))
    equity_curve = pd.Series(equity[1:], index=[t.exit_date for t in all_closed]) if all_closed else pd.Series([1.0])

    return BacktestReport(per_symbol, pooled_pnls, equity_curve)


def print_report(report: BacktestReport) -> None:
    print("\n=== Per-symbol summary ===")
    for symbol, r in report.per_symbol.items():
        closed = [t for t in r.trades if t.is_closed]
        still_open = len(r.trades) - len(closed)
        pass_rate = (r.fvg_triggers - r.confluence_rejections) / r.fvg_triggers if r.fvg_triggers else None
        print(
            f"{symbol}: {r.bars_evaluated} bars, {r.fvg_triggers} FVG+volume trigger(s), "
            f"{r.confluence_rejections} rejected by confluence "
            f"({'n/a' if pass_rate is None else f'{pass_rate:.1%} pass rate'}), "
            f"{r.trades_opened} trade(s) opened, {len(closed)} closed, {still_open} still open at backtest end"
        )

    print("\n=== Pooled closed-trade stats (theoretical premiums, see module docstring) ===")
    if not report.pooled_closed_pnls:
        print("No closed trades -- nothing to report. A strict min_confluence_score (see config/settings.yaml) makes this expected sometimes.")
        return

    avg_win, avg_loss = avg_win_loss(report.pooled_closed_pnls)
    dd = max_drawdown(report.pooled_equity_curve)
    print(f"Closed trades: {len(report.pooled_closed_pnls)}")
    print(f"Win rate: {win_rate(report.pooled_closed_pnls):.1%}")
    print(f"Avg win: {avg_win:+.1%}   Avg loss: {avg_loss:+.1%}")
    print(f"Profit factor: {profit_factor(report.pooled_closed_pnls):.2f}")
    print(f"Max consecutive losses: {max_consecutive_losses(report.pooled_closed_pnls)}")
    print(f"Max drawdown (pooled equity curve): {dd.max_drawdown_pct:.1%}")
    print(f"Final pooled equity (from 1.0): {report.pooled_equity_curve.iloc[-1]:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", help="Daily-bar backtest only (--intraday ignores this, see --intraday-days)")
    parser.add_argument("--end", help="Daily-bar backtest only")
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument(
        "--intraday", action="store_true",
        help="Walk 5-minute bars via run_backtest_intraday instead of daily bars -- the only way to meaningfully "
             "test max_hold_days/same-day exits and trend-invalidation (see module docstring, caveat 5). "
             "Limited to a recent ~2-month window by yfinance's own intraday history cap.",
    )
    parser.add_argument("--intraday-days", type=int, default=DEFAULT_INTRADAY_DAYS, help="--intraday only")
    args = parser.parse_args()

    loaded_settings = load_settings(args.settings)
    if args.intraday:
        backtest_report = run_backtest_intraday(args.symbols, loaded_settings, intraday_days=args.intraday_days)
    else:
        if not args.start or not args.end:
            parser.error("--start/--end are required unless --intraday is set")
        backtest_report = run_backtest(args.symbols, args.start, args.end, loaded_settings)
    print_report(backtest_report)
