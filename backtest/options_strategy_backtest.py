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
from typing import Dict, List, Literal, Optional

import pandas as pd

from backtest.metrics import avg_win_loss, max_consecutive_losses, max_drawdown, profit_factor, win_rate
from backtest.options_pricing import black_scholes_price, realized_volatility
from brain.confluence import HARD_VETO_KEYS, evaluate_confluence
from brain.options_strategy import decide_options_action
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

    @property
    def is_closed(self) -> bool:
        return self.exit_premium is not None

    @property
    def pnl_pct(self) -> Optional[float]:
        if not self.is_closed or self.entry_premium <= 0:
            return None
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
            )
            if any(confluence.details.get(k) == "fail" for k in HARD_VETO_KEYS):
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
            opt.sma_period, opt.min_confluence_score,
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
            )
            result.trades_opened += 1

    if open_trade is not None:
        result.trades.append(open_trade)  # still open at backtest end -- pnl_pct is None, excluded from closed-trade stats

    return result


DEFAULT_INTRADAY_WARMUP_BARS = 30  # fvg_lookback_period + swing-point slack -- far smaller than daily's 210, since this is only about bar-count for FVG/swing detection; the 200-SMA veto and realized-vol IV proxy come from daily_bars, not this window
INTRADAY_WINDOW_BARS = 400          # matches production's bounded live intraday window, same constant as run_symbol_backtest's daily WINDOW_BARS


def run_symbol_backtest_intraday(
    symbol: str,
    intraday_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    settings: Settings,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    warmup_bars: int = DEFAULT_INTRADAY_WARMUP_BARS,
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
            )
            if any(confluence.details.get(k) == "fail" for k in HARD_VETO_KEYS):
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

        decision = decide_options_action(
            symbol, window, opt.fvg_lookback_period, opt.fvg_body_multiplier, opt.fvg_volume_multiplier,
            opt.sma_period, opt.min_confluence_score,
            daily_bars=daily_window if not daily_window.empty else None,
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
            )
            result.trades_opened += 1

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

    per_symbol: Dict[str, SymbolBacktestResult] = {}
    all_closed: List[SimulatedTrade] = []

    for symbol in symbols:
        intraday_bars = fetcher.get_bars(symbol, intraday_start.isoformat(), end.isoformat(), timeframe="5m")
        daily_bars = fetcher.get_bars(symbol, daily_start.isoformat(), end.isoformat(), timeframe="1D")
        result = run_symbol_backtest_intraday(symbol, intraday_bars, daily_bars, settings)
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
