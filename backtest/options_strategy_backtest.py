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

    for i in range(warmup_bars, len(bars)):
        window = bars.iloc[: i + 1]
        current_date = bars.index[i].date() if hasattr(bars.index[i], "date") else bars.index[i]
        spot = float(bars["close"].iloc[i])
        result.bars_evaluated += 1
        just_closed_this_bar = False

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

        if open_trade is not None:
            current_value = _price_position(open_trade, spot, current_date, risk_free_rate)
            pnl_pct = (current_value / open_trade.entry_premium) - 1 if open_trade.entry_premium > 0 else 0
            if -pnl_pct >= opt.stop_loss_pct:
                open_trade.exit_date, open_trade.exit_reason = current_date, "stop_loss"
                open_trade.exit_premium = current_value
                result.trades.append(open_trade)
                open_trade, just_closed_this_bar = None, True
            elif pnl_pct >= opt.take_profit_pct:
                open_trade.exit_date, open_trade.exit_reason = current_date, "take_profit"
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
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--settings", default="config/settings.yaml")
    args = parser.parse_args()

    loaded_settings = load_settings(args.settings)
    backtest_report = run_backtest(args.symbols, args.start, args.end, loaded_settings)
    print_report(backtest_report)
