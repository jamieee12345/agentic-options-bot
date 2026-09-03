"""Theoretical (Black-Scholes) option pricing, used ONLY by the backtester.

Why theoretical pricing at all: there is no free, readily available source
of HISTORICAL options prices anywhere in this project's data stack.
yfinance gives free historical stock bars but only a live snapshot of
option chains (no time series); robin_stocks gives live chains, not
history either. Real historical options data (strikes/premiums/IV over
time) generally means a paid vendor (CBOE, ORATS, Polygon, ThetaData,
etc.), which this project doesn't have. Black-Scholes, fed by the
underlying's actual historical prices and a trailing-realized-volatility
proxy for IV, is the best approximation achievable with only free data --
clearly an approximation, not real market prices. It ignores the
volatility skew/smile real option prices have (it prices every strike off
the same single volatility number) and assumes IV stays constant for the
life of a simulated trade rather than actually moving day to day. Both are
real sources of error versus what a live account would have actually paid
and received. See backtest/options_strategy_backtest.py's module docstring
for how this is used and what it can and can't tell you.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_price(spot: float, strike: float, years_to_expiry: float, risk_free_rate: float, volatility: float, option_type: str) -> float:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if spot <= 0 or strike <= 0 or volatility <= 0:
        raise ValueError(f"spot ({spot}), strike ({strike}), and volatility ({volatility}) must all be positive")
    if years_to_expiry <= 0:
        # At/past expiry: intrinsic value only.
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)

    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * years_to_expiry) / (volatility * math.sqrt(years_to_expiry))
    d2 = d1 - volatility * math.sqrt(years_to_expiry)

    if option_type == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-risk_free_rate * years_to_expiry) * _norm_cdf(d2)
    return strike * math.exp(-risk_free_rate * years_to_expiry) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


DEFAULT_REALIZED_VOL_LOOKBACK = 20


def realized_volatility(bars: pd.DataFrame, lookback: int = DEFAULT_REALIZED_VOL_LOOKBACK, trading_days_per_year: int = 252) -> float:
    """Annualized stdev of daily log returns over the trailing `lookback`
    bars -- the standard, simplest realized-vol estimator, used here as the
    IV proxy for black_scholes_price(). Real implied volatility (what
    options actually trade on) differs from trailing realized volatility --
    often significantly, especially around known catalysts -- so this is a
    genuine approximation, not a measurement of what IV actually was.
    """
    closes = bars["close"].iloc[-(lookback + 1):]
    if len(closes) < 2:
        return 0.20  # arbitrary but reasonable fallback for a symbol with almost no history yet
    log_returns = np.log(closes / closes.shift(1)).dropna()
    daily_std = float(log_returns.std(ddof=1))
    return daily_std * math.sqrt(trading_days_per_year)
