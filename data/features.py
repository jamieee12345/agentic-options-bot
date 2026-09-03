"""Technical indicators and feature computation.

Pure pandas/numpy — no `pandas-ta`/`ta-lib` dependency, so this is easy to
install, easy to audit, and behaves identically whether it's running inside
the backtester or the live loop. That parity matters: if backtesting and
live trading compute features differently, backtest results stop meaning
anything.

Organized into the same categories used to spec this out: returns,
volatility, volume, trend, mean reversion, momentum, range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """OLS slope of `series` against a simple 0..window-1 time index,
    computed over a trailing rolling window. Used for every "slope of the
    N-period SMA" style feature below — the window IS the N.
    """
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / x_var)

    return series.rolling(window).apply(_slope, raw=True)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close / close.shift(periods))


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def realized_volatility(close: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling realized volatility of 1-period log returns, annualized by default."""
    vol = log_return(close, 1).rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol


def volatility_ratio(close: pd.Series, short_window: int = 5, long_window: int = 20) -> pd.Series:
    """Short-term realized vol / long-term realized vol. >1 means volatility
    is currently expanding relative to its recent baseline; <1 means it's
    contracting. Annualization cancels out in the ratio either way.
    """
    short_vol = realized_volatility(close, window=short_window, annualize=False)
    long_vol = realized_volatility(close, window=long_window, annualize=False)
    return short_vol / long_vol.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def normalized_volume(volume: pd.Series, window: int = 50) -> pd.Series:
    """Z-score of volume against its own rolling mean/std."""
    return rolling_zscore(volume, window)


def volume_trend_slope(volume: pd.Series, sma_window: int = 10) -> pd.Series:
    """Slope of the volume's own N-period SMA — is trading activity ramping
    up or drying up, independent of its current absolute level.
    """
    return rolling_slope(sma(volume, sma_window), window=sma_window)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def _directional_movement(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    return pd.Series(plus_dm, index=high.index), pd.Series(minus_dm, index=high.index)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index — trend *strength* (not direction).
    Conventionally: below ~20 is non-trending/choppy, above ~25 is trending.
    """
    plus_dm, minus_dm = _directional_movement(high, low)
    smoothed_tr = true_range(high, low, close).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / smoothed_tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / smoothed_tr

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def sma_slope(close: pd.Series, window: int = 50) -> pd.Series:
    return rolling_slope(sma(close, window), window=window)


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_zscore(close: pd.Series, rsi_window: int = 14, zscore_window: int = 50) -> pd.Series:
    return rolling_zscore(rsi(close, rsi_window), window=zscore_window)


def distance_from_sma_pct(close: pd.Series, window: int = 200) -> pd.Series:
    """(close - SMA) / close — how far price has stretched from its long-run
    average, as a fraction of current price. Positive = above the average.
    """
    return (close - sma(close, window)) / close


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def roc(close: pd.Series, periods: int) -> pd.Series:
    """Rate of change: simple (not log) percent change over N periods."""
    return (close - close.shift(periods)) / close.shift(periods)


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed (approximated via EWM, the
    standard practical equivalent).
    """
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def normalized_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return atr(high, low, close, window) / close


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def compute_feature_set(bars: pd.DataFrame, trend_threshold: float = 25.0) -> pd.DataFrame:
    """Given OHLCV bars (columns: open, high, low, close, volume), compute
    the full feature set: returns, volatility, volume, trend, mean
    reversion, momentum, and range.
    """
    for col in ("high", "low", "close", "volume"):
        if col not in bars.columns:
            raise ValueError(f"compute_feature_set: missing required column '{col}'")

    close, high, low, volume = bars["close"], bars["high"], bars["low"], bars["volume"]
    out = pd.DataFrame(index=bars.index)
    out["close"] = close

    # Returns
    out["log_return_1"] = log_return(close, 1)
    out["log_return_5"] = log_return(close, 5)
    out["log_return_20"] = log_return(close, 20)

    # Volatility
    out["realized_vol_5"] = realized_volatility(close, window=5)
    out["realized_vol_20"] = realized_volatility(close, window=20)
    out["vol_ratio_5_20"] = volatility_ratio(close, short_window=5, long_window=20)

    # Volume
    out["volume_zscore_50"] = normalized_volume(volume, window=50)
    out["volume_sma_10"] = sma(volume, window=10)
    out["volume_trend_slope_10"] = volume_trend_slope(volume, sma_window=10)

    # Trend
    out["adx_14"] = adx(high, low, close, window=14)
    out["price_sma_50"] = sma(close, window=50)
    out["price_ema_50"] = ema(close, span=50)
    out["sma_50_slope"] = sma_slope(close, window=50)
    out["is_trending"] = out["adx_14"] > trend_threshold

    # Mean reversion
    out["rsi_14"] = rsi(close, window=14)
    out["rsi_zscore_50"] = rsi_zscore(close, rsi_window=14, zscore_window=50)
    out["dist_from_sma200_pct"] = distance_from_sma_pct(close, window=200)

    # Momentum
    out["roc_10"] = roc(close, 10)
    out["roc_20"] = roc(close, 20)

    # Range
    out["atr_14"] = atr(high, low, close, window=14)
    out["normalized_atr_14"] = normalized_atr(high, low, close, window=14)

    out["dollar_volume"] = close * volume

    return out
