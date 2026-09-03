"""Historical and real-time market data fetchers.

Defines a common interface so the rest of the system (feature computation,
backtesting, live trading) doesn't care which data source sits behind it.
External libraries are imported lazily, inside each method, so this module
can be imported and unit tested even in environments without network access
or those optional packages installed.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None


class HistoricalDataFetcher(ABC):
    """Fetches historical OHLCV bars for a symbol."""

    @abstractmethod
    def get_bars(self, symbol: str, start: str, end: str, timeframe: str = "1D") -> pd.DataFrame:
        """Return a DataFrame indexed by timestamp (ascending) with columns:
        open, high, low, close, volume.
        """
        raise NotImplementedError


class QuoteFetcher(ABC):
    """Fetches the latest quote for a symbol, for live trading."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        raise NotImplementedError


def _validate_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: fetched data missing columns {missing}")
    if df.empty:
        raise ValueError(f"{symbol}: fetched data is empty")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(f"{symbol}: fetched data contains NaNs — check the date range/symbol")
    return df.sort_index()[REQUIRED_COLUMNS]


class YFinanceHistoricalFetcher(HistoricalDataFetcher):
    """Free, no-auth historical data source. Good for backtesting and for
    warming up the regime model on startup.

    Not a substitute for a real-time feed: it's delayed and not officially
    supported for production trading use — treat it as a backtesting/research
    data source only.
    """

    # Intraday entries added for the live FVG/confluence options strategy,
    # which needs to react within a trading day, not once it closes. yfinance's
    # own limits apply: 1m data only covers the last ~7 days, other intraday
    # intervals ~60 days -- comfortably enough for this project's lookback
    # windows (10-20 bars), nowhere near enough for a 200-period SMA on an
    # intraday interval, which is exactly why the SMA200 trend filter
    # (brain/trend_indicators.py) is deliberately fed DAILY bars separately
    # from whatever interval the rest of the strategy runs on -- see
    # orchestration/run_live.py.
    _TIMEFRAME_MAP = {"1D": "1d", "1W": "1wk", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m"}

    def get_bars(self, symbol: str, start: str, end: str, timeframe: str = "1D") -> pd.DataFrame:
        import yfinance as yf  # imported lazily — optional dependency

        interval = self._TIMEFRAME_MAP.get(timeframe)
        if interval is None:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. Supported: {list(self._TIMEFRAME_MAP)}"
            )

        raw = yf.download(
            symbol, start=start, end=end, interval=interval,
            progress=False, auto_adjust=True,
        )
        if raw.empty:
            raise ValueError(f"No data returned for {symbol} between {start} and {end}")

        # yfinance returns MultiIndex columns for some call shapes — flatten defensively.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.rename(columns=str.lower)
        df.index.name = "timestamp"
        return _validate_bars(df, symbol)


class RobinhoodQuoteFetcher(QuoteFetcher):
    """Live quotes via the unofficial `robin_stocks` client library.

    Robinhood does not publish an official equities/options market data API.
    This wraps the community `robin_stocks` package, which replicates the
    private endpoints used by Robinhood's own mobile app. Two things to keep
    in mind before relying on this in production:
      - It is unsupported and can break without notice if Robinhood changes
        their app's internal API.
      - Automated use of these endpoints for non-crypto accounts sits outside
        Robinhood's terms of service — that's a risk you're taking on, not
        something this code can remove.

    Credentials are read from environment variables and never touch this
    code as literals. Set them in a local, git-ignored .env file:
        ROBINHOOD_USERNAME=...
        ROBINHOOD_PASSWORD=...
        ROBINHOOD_MFA_SECRET=...   # TOTP secret, if using app-based 2FA
    """

    def __init__(self) -> None:
        import robin_stocks.robinhood as rh  # imported lazily — optional dependency
        import pyotp

        username = os.environ.get("ROBINHOOD_USERNAME")
        password = os.environ.get("ROBINHOOD_PASSWORD")
        mfa_secret = os.environ.get("ROBINHOOD_MFA_SECRET")

        if not username or not password:
            raise RuntimeError(
                "Set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD as environment "
                "variables before starting live trading."
            )

        mfa_code = pyotp.TOTP(mfa_secret).now() if mfa_secret else None
        self._rh = rh
        self._rh.login(username, password, mfa_code=mfa_code, store_session=True)

    def get_quote(self, symbol: str) -> Quote:
        data = self._rh.stocks.get_latest_price(symbol, includeExtendedHours=True)
        if not data or data[0] is None:
            raise ValueError(f"No quote returned for {symbol}")
        return Quote(symbol=symbol, price=float(data[0]), timestamp=datetime.now(timezone.utc))

    def logout(self) -> None:
        self._rh.logout()


class RobinhoodIntradayHistoricalFetcher(HistoricalDataFetcher):
    """Live intraday bars via `robin_stocks`, fresher than
    YFinanceHistoricalFetcher's free ~15-20min-delayed feed since it's the
    same authenticated account session.

    FIELD NAMES CONFIRMED, not guessed: pulled a real 5-minute bar from a
    live Robinhood account via a separate tool during development and
    checked the actual response shape -- `begins_at`, `open_price`,
    `high_price`, `low_price`, `close_price`, `volume`. That's a real
    improvement in confidence over the rest of this project's robin_stocks
    integration, which has had to guess at field/parameter names without
    ever seeing a live response. What's still NOT verified: robin_stocks'
    own Python function signature (`stocks.get_stock_historicals`) and
    exactly which interval/span PAIRS it accepts -- that call happens
    through the `robin_stocks` library, not the tool used to confirm field
    names above, and the library's parameter validation could differ from
    the raw API's. `_INTERVAL_SPAN_MAP` below sticks to the two
    combinations most consistently documented across robin_stocks usage
    (5minute/week, hour/month) rather than the finer-grained intervals
    Robinhood's raw API turned out to support (30minute, 4hour, etc.) --
    confirm those work with your installed version before assuming a
    tighter interval is available.

    Unlike YFinanceHistoricalFetcher, `get_bars`'s `start`/`end` are NOT
    honored precisely -- robin_stocks' historicals endpoint takes a fixed
    `span` (how far back), not an arbitrary date range. This always
    returns everything in that span; trim it yourself if you need an exact
    window.
    """

    _INTERVAL_SPAN_MAP = {"5m": ("5minute", "week"), "1h": ("hour", "month")}

    def __init__(self, rh_session=None) -> None:
        if rh_session is not None:
            self._rh = rh_session
            return

        import robin_stocks.robinhood as rh  # lazy import, matches the rest of the codebase's convention
        import pyotp

        username = os.environ.get("ROBINHOOD_USERNAME")
        password = os.environ.get("ROBINHOOD_PASSWORD")
        mfa_secret = os.environ.get("ROBINHOOD_MFA_SECRET")
        if not username or not password:
            raise RuntimeError("Set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD as environment variables before starting live trading.")

        mfa_code = pyotp.TOTP(mfa_secret).now() if mfa_secret else None
        rh.login(username, password, mfa_code=mfa_code, store_session=True)
        self._rh = rh

    def get_bars(self, symbol: str, start: str, end: str, timeframe: str = "5m") -> pd.DataFrame:
        if timeframe not in self._INTERVAL_SPAN_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {list(self._INTERVAL_SPAN_MAP)}")
        interval, span = self._INTERVAL_SPAN_MAP[timeframe]

        raw = self._rh.stocks.get_stock_historicals(symbol, interval=interval, span=span, bounds="regular")
        if not raw:
            raise ValueError(f"No historicals returned for {symbol} (interval={interval}, span={span})")

        df = pd.DataFrame(raw)
        df = df.rename(columns={"open_price": "open", "high_price": "high", "low_price": "low", "close_price": "close"})
        df["timestamp"] = pd.to_datetime(df["begins_at"])
        df = df.set_index("timestamp")
        for col in REQUIRED_COLUMNS:
            df[col] = df[col].astype(float)
        return _validate_bars(df, symbol)


class AlpacaIntradayHistoricalFetcher(HistoricalDataFetcher):
    """Intraday bars via Alpaca's official, documented market-data API
    (`alpaca-py`) -- a real API contract, not a reverse-engineered one like
    robin_stocks/yfinance/TradingView scrapers. Free tier: real-time (not
    delayed) bars from the IEX exchange only, not the full consolidated
    tape -- explicitly requested via `feed=DataFeed.IEX` below, since the
    default feed on a free account otherwise errors (SIP, the full-tape
    feed, requires a paid subscription).

    Needs a free account at alpaca.markets and an API key pair -- market
    data access doesn't require funding a brokerage account with them, only
    signing up. Set in your local, git-ignored .env (never as literals here,
    never pasted anywhere else):
        ALPACA_API_KEY_ID=<your key id, starts with PK>
        ALPACA_API_SECRET_KEY=<your secret key>

    Meaningfully more verified than this project's robin_stocks/options
    integrations: `alpaca-py` was actually installed in this environment
    and every field/enum used below (`StockBarsRequest`'s
    `symbol_or_symbols`/`timeframe`/`start`/`end`/`feed`,
    `TimeFrameUnit.Minute`/`.Hour`, `DataFeed.IEX`) was checked against the
    real installed package (`StockBarsRequest.model_fields`, `dir()` on the
    enums) -- not guessed from memory. What's still NOT verified is an
    actual live API call: no real Alpaca credentials were available in this
    sandbox, so the request could still be malformed in a way that only a
    real response would reveal, and `.df`'s exact shape (MultiIndex vs not,
    for a single-symbol request) is inferred from `alpaca-py`'s documented
    behavior, not observed directly. Confirm your first real call returns
    sane bars before trusting it unattended.
    """

    _TIMEFRAME_MAP = {"1m": (1, "Minute"), "5m": (5, "Minute"), "15m": (15, "Minute"), "30m": (30, "Minute"), "1h": (1, "Hour")}

    def __init__(self) -> None:
        api_key = os.environ.get("ALPACA_API_KEY_ID")
        secret_key = os.environ.get("ALPACA_API_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY as environment variables before using this fetcher.")

        from alpaca.data.historical import StockHistoricalDataClient  # lazy import, optional dependency

        self._client = StockHistoricalDataClient(api_key, secret_key)

    def get_bars(self, symbol: str, start: Optional[str], end: Optional[str], timeframe: str = "5m") -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        if timeframe not in self._TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {list(self._TIMEFRAME_MAP)}")
        amount, unit_name = self._TIMEFRAME_MAP[timeframe]

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(amount, getattr(TimeFrameUnit, unit_name)),
            start=datetime.fromisoformat(start) if start else None,
            end=datetime.fromisoformat(end) if end else None,
            feed=DataFeed.IEX,
        )
        bar_set = self._client.get_stock_bars(request)
        df = bar_set.df
        if df is None or df.empty:
            raise ValueError(f"No bars returned for {symbol}")

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        df.index.name = "timestamp"
        return _validate_bars(df, symbol)
