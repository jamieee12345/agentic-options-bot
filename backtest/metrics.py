"""Performance metrics for the walk-forward backtester.

Every function takes a plain equity curve, return series, or trade P&L list
as input, so each is testable in isolation without needing a full backtest
run first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return(equity_curve: pd.Series) -> float:
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def cagr(equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    n_periods = len(equity_curve) - 1
    if n_periods <= 0:
        return 0.0
    growth = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if growth <= 0:
        return -1.0  # wiped out (or worse) -- not representable as a normal CAGR
    years = n_periods / periods_per_year
    return float(growth ** (1 / years) - 1)


def periodic_returns(equity_curve: pd.Series) -> pd.Series:
    return equity_curve.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR, risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    std = excess.std(ddof=1)
    if std < 1e-12:  # tolerance, not exact ==, since floating-point std of a
        return 0.0    # "constant" series is rarely bit-exact zero
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR, risk_free_rate: float = 0.0) -> float:
    """Downside deviation uses the full sample (zeros for non-negative
    excess returns), not just the negative subset -- the standard
    semi-deviation definition.
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    downside = np.minimum(excess, 0.0)
    downside_deviation = np.sqrt((downside ** 2).mean())
    if downside_deviation < 1e-12:
        return 0.0
    return float((excess.mean() / downside_deviation) * np.sqrt(periods_per_year))


@dataclass(frozen=True)
class DrawdownResult:
    max_drawdown_pct: float          # negative, e.g. -0.23 for a 23% drawdown
    peak_date: object
    trough_date: object
    recovery_date: Optional[object]  # None if never recovered by series end
    duration_bars: int               # peak -> trough, in bars
    underwater_bars: int             # peak -> recovery (or series end), in bars; >= duration_bars


def max_drawdown(equity_curve: pd.Series) -> DrawdownResult:
    values = equity_curve.to_numpy()
    index = equity_curve.index
    n = len(values)

    running_max = np.maximum.accumulate(values)
    drawdown = values / running_max - 1
    trough_pos = int(np.argmin(drawdown))
    peak_value = running_max[trough_pos]

    # last position at/before the trough where equity equals the peak driving this drawdown
    peak_candidates = np.where(values[: trough_pos + 1] == peak_value)[0]
    peak_pos = int(peak_candidates[-1])

    # first position after the trough where equity recovers to >= that peak
    recovery_candidates = np.where(values[trough_pos + 1 :] >= peak_value)[0]
    if len(recovery_candidates) > 0:
        recovery_pos = trough_pos + 1 + int(recovery_candidates[0])
        recovery_date = index[recovery_pos]
        underwater_bars = recovery_pos - peak_pos
    else:
        recovery_date = None
        underwater_bars = (n - 1) - peak_pos  # still underwater at series end

    return DrawdownResult(
        max_drawdown_pct=float(drawdown[trough_pos]),
        peak_date=index[peak_pos],
        trough_date=index[trough_pos],
        recovery_date=recovery_date,
        duration_bars=trough_pos - peak_pos,
        underwater_bars=underwater_bars,
    )


def longest_underwater_streak(equity_curve: pd.Series) -> int:
    """Longest run of consecutive bars below the running peak. Can exceed
    the worst drawdown's own `underwater_bars` — a smaller drawdown that
    takes a long time to recover can outlast a sharper one that recovers
    quickly.
    """
    values = equity_curve.to_numpy()
    running_max = np.maximum.accumulate(values)
    underwater = values < running_max

    longest = current = 0
    for uw in underwater:
        current = current + 1 if uw else 0
        longest = max(longest, current)
    return longest


def calmar_ratio(cagr_value: float, max_drawdown_pct: float) -> float:
    if max_drawdown_pct == 0:
        return 0.0
    return cagr_value / abs(max_drawdown_pct)


def win_rate(trade_pnls: Sequence[float]) -> float:
    if not trade_pnls:
        return 0.0
    return sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)


def avg_win_loss(trade_pnls: Sequence[float]) -> Tuple[float, float]:
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    return (
        float(np.mean(wins)) if wins else 0.0,
        float(np.mean(losses)) if losses else 0.0,
    )


def profit_factor(trade_pnls: Sequence[float]) -> float:
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_consecutive_losses(trade_pnls: Sequence[float]) -> int:
    longest = current = 0
    for p in trade_pnls:
        current = current + 1 if p < 0 else 0
        longest = max(longest, current)
    return longest


def worst_period_return(returns: pd.Series, freq: str) -> float:
    """freq='D' for daily (no resampling needed), or a pandas offset alias
    ('W', 'ME') to compound daily returns within each period first.
    Requires a DatetimeIndex on `returns` for anything other than 'D'.
    """
    if freq == "D":
        return float(returns.min()) if len(returns) else 0.0
    compounded = (1 + returns).resample(freq).prod() - 1
    return float(compounded.min()) if len(compounded) else 0.0
