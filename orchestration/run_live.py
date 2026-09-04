"""The entrypoint that was missing: assembles every already-built piece
(RobinhoodBroker, RobinhoodOptionChainFetcher, OptionsOrderExecutor,
brain/options_strategy.py's FVG+confluence engine) into one process that
runs continuously and can place real orders. Nothing before this file did
that -- the backtest exercised the decision logic against history, and the
dashboard reads account state, but neither calls
`broker.place_option_order` in a standing loop. This is that loop.

RUNS ON INTRADAY BARS, not daily ones -- reacts within the trading day, not
once when it closes. Every `--poll-seconds`, it re-fetches recent intraday
bars for the whole watchlist and re-evaluates every symbol. The 200-SMA
trend veto is the one exception: brain/confluence.py deliberately keeps
that on a DAILY feed (fetched once per cycle too, cheap) since "200
periods" only means "200 days" if the bars actually are daily -- 200
five-minute bars is under two trading days and would gut the entire point
of a long-term trend filter. See that module's docstring.

`--poll-seconds` (how often this LOOKS) and `--interval` (what timeframe
the bars it looks AT are) are deliberately decoupled, on request -- default
60s / 5m, not matched 1:1 the way an earlier version of this docstring
described. The signals themselves (FVG, market structure, volume profile,
etc.) still read 5-minute bars -- a 5m bar only closes every 5 minutes
regardless of how often this polls, so the signal quality/noise level is
unchanged. What faster polling actually buys: (1) less lag between a bar
closing and this process noticing it -- up to poll_seconds late instead of
up to interval late, (2) faster reaction on the EXIT side, since
_check_price_based_exit (orchestration/options_execution.py) re-fetches a
fresh option quote every cycle regardless of bar interval, so a stop-loss/
take-profit trigger is caught within one poll_seconds window, not one bar
close. It does NOT mean the strategy trades off 1-minute noise -- that
would require also lowering --interval, which isn't part of this change.
Going much below ~60s risks tripping Robinhood's unofficial-API rate
limiting/fraud detection (this account already hit one "suspicious login"
challenge this project) for a diminishing-returns reaction-speed gain --
tune down cautiously, watch the logs, not blindly.

NO FRESHNESS/DEDUP STATE FILE -- an earlier daily-bar version of this
script tracked "have I already acted on this bar" per symbol to avoid
reprocessing. That's not needed here: re-evaluating the same still-current
intraday bar every cycle is idempotent by construction --
OptionsOrderExecutor already refuses to add to a position it's already
holding, and safety/order_validation.py's DuplicateOrderGuard independently
blocks resubmitting an order for a symbol with one already working. Instead
of separate bookkeeping to prevent re-triggering, this leans on the
guarantees that already have to hold for the executor to be correct at all.

DATA SOURCE: intraday bars come from `data.fetchers.RobinhoodIntradayHistoricalFetcher`
-- the same authenticated account session as the broker (one login, shared
via `RobinhoodBroker.rh_session`, not a second one), which is materially
fresher than a free delayed feed. Its field names were checked against a
real live response during development (see that class's docstring); its
exact robin_stocks interval/span parameter pairing was not, since that
goes through the `robin_stocks` library rather than the tool used to check
field names -- confirm your first real run's bars look sane before trusting
it unattended. Only "5m" and "1h" are supported (the two interval/span
pairs most consistently documented for robin_stocks); yfinance's
`YFinanceHistoricalFetcher` still covers 15m/30m if you need those, at the
cost of its usual delay.

DAILY bars (the SMA200 feed only) stay on `YFinanceHistoricalFetcher` --
free, no extra robin_stocks call, and a day-old daily close doesn't
meaningfully change a 200-day average, so the freshness that matters for
the intraday feed doesn't matter here.

SAFETY, unchanged from every other executor in this project: respects
`broker.live_trading_enabled` exactly like the backtester and dashboard's
writers already do -- a single shared kill switch for both the equity and
options pipelines, not a separate one per asset class. False (the
settings.yaml default) means every approved
trade is logged as a would-be order and NEVER reaches the broker. This
script is safe to run continuously, including on an always-on machine,
until that flag is deliberately flipped. The startup banner states which
mode it's in every time, loudly.

Run it (needs .env credentials -- see README's Setup section):

    PYTHONPATH=. python3 orchestration/run_live.py --symbols SPY QQQ AAPL NVDA
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd

from broker.robinhood_broker import RobinhoodBroker
from config.config_loader import Settings, load_settings
from data.fetchers import AlpacaIntradayHistoricalFetcher, RobinhoodIntradayHistoricalFetcher, YFinanceHistoricalFetcher
from data.options_data import RobinhoodOptionChainFetcher
from orchestration.options_execution import OptionsOrderExecutor, build_open_positions_from_broker
from orchestration.retry import retry_with_backoff

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = "5m"  # the TRADED timeframe -- signals/indicators read this bar size regardless of poll cadence below
DEFAULT_DATA_SOURCE = "alpaca"
DEFAULT_POLL_SECONDS = 60  # how often this LOOKS, independent of DEFAULT_INTERVAL (what it looks at) -- see module docstring
DAILY_LOOKBACK_DAYS = 400     # covers sma_period=200 + swing/FVG warmup with calendar-day slack
ALPACA_INTRADAY_LOOKBACK_DAYS = 20  # comfortably inside Alpaca's IEX free-feed practical window, plenty for this project's lookback windows (10-20 bars)


def build_intraday_fetcher(source: str, broker: RobinhoodBroker):
    if source == "robinhood":
        return RobinhoodIntradayHistoricalFetcher(rh_session=broker.rh_session)  # shares the broker's login, no second auth
    if source == "alpaca":
        return AlpacaIntradayHistoricalFetcher()
    raise ValueError(f"Unknown data source {source!r}")


def fetch_intraday_bars(fetcher, symbols: List[str], interval: str) -> Dict[str, pd.DataFrame]:
    # RobinhoodIntradayHistoricalFetcher ignores start/end (robin_stocks
    # takes a fixed span, not a date range -- see its class docstring);
    # AlpacaIntradayHistoricalFetcher honors them. Computing a real window
    # here and passing it to both keeps this call site source-agnostic.
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=ALPACA_INTRADAY_LOOKBACK_DAYS)
    bars: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            bars[symbol] = fetcher.get_bars(symbol, start.isoformat(), end.isoformat(), timeframe=interval)
        except Exception:
            logger.exception("%s: failed to fetch intraday bars this cycle -- skipping it, not the whole cycle", symbol)
    return bars


def fetch_daily_bars(fetcher: YFinanceHistoricalFetcher, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=DAILY_LOOKBACK_DAYS)
    bars: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            bars[symbol] = fetcher.get_bars(symbol, start.isoformat(), end.isoformat(), timeframe="1D")
        except Exception:
            logger.exception("%s: failed to fetch daily bars this cycle -- SMA200 veto will fall back to intraday bars for this symbol", symbol)
    return bars


def build_executor(broker: RobinhoodBroker, chain_fetcher: RobinhoodOptionChainFetcher, settings: Settings) -> OptionsOrderExecutor:
    opt = settings.options
    return OptionsOrderExecutor(
        broker=broker, chain_fetcher=chain_fetcher,
        max_premium_pct_per_trade=opt.max_premium_pct_per_trade,
        max_total_premium_pct_of_equity=opt.max_total_premium_pct_of_equity,
        close_before_expiration_days=opt.close_before_expiration_days,
        dte_min=opt.target_dte_min, dte_max=opt.target_dte_max,
        max_spread_pct=settings.risk.max_bid_ask_spread_pct,
        fvg_lookback_period=opt.fvg_lookback_period, fvg_body_multiplier=opt.fvg_body_multiplier,
        fvg_volume_multiplier=opt.fvg_volume_multiplier, sma_period=opt.sma_period,
        min_confluence_score=opt.min_confluence_score,
        stagnant_exit_hold_fraction=opt.stagnant_exit_hold_fraction,
        stagnant_exit_min_pnl_pct=opt.stagnant_exit_min_pnl_pct,
        max_hold_days=opt.max_hold_days,
        # NOTE: this flag lives under settings.yaml's broker: section, not
        # options: -- it's a single shared kill switch for both pipelines,
        # not a separate one per asset class.
        live_trading_enabled=settings.broker.live_trading_enabled,
    )


def run_once(
    broker: RobinhoodBroker,
    intraday_fetcher,  # RobinhoodIntradayHistoricalFetcher or AlpacaIntradayHistoricalFetcher -- see build_intraday_fetcher
    daily_fetcher: YFinanceHistoricalFetcher,
    executor: OptionsOrderExecutor,
    symbols: List[str],
    interval: str,
) -> None:
    intraday_bars = fetch_intraday_bars(intraday_fetcher, symbols, interval)
    if not intraday_bars:
        logger.warning("No intraday bars fetched for any symbol this cycle -- skipping.")
        return
    daily_bars = fetch_daily_bars(daily_fetcher, symbols)

    equity = retry_with_backoff(broker.get_portfolio_equity)
    buying_power = retry_with_backoff(broker.get_buying_power)
    open_orders = retry_with_backoff(broker.get_open_orders)
    open_order_symbols = [o.get("symbol") for o in open_orders if o.get("symbol")]
    raw_option_positions = retry_with_backoff(broker.get_option_positions)
    open_positions = build_open_positions_from_broker(raw_option_positions)

    records = executor.run(
        bars=intraday_bars, equity=equity, open_positions=open_positions, buying_power=buying_power,
        open_order_symbols=open_order_symbols, now=datetime.now(timezone.utc), daily_bars=daily_bars,
    )
    for r in records:
        if r.action is not None:
            logger.info("%s: %s %s x%s -- %s", r.symbol, r.action, r.option_type, r.contracts, "placed" if r.placed else "dry-run/not placed")
        elif r.skipped_reason:
            logger.info("%s: no action (%s)", r.symbol, r.skipped_reason)


def run_forever(symbols: List[str], settings_path: str, interval: str, poll_seconds: int, data_source: str) -> None:
    settings = load_settings(settings_path)
    if not settings.options.enabled:
        raise SystemExit(
            "options.enabled is false in settings.yaml -- refusing to start. "
            "This is the feature flag for whether the options pipeline runs at all; "
            "flip it to true deliberately before starting this process."
        )

    logger.info(
        "%s -- live_trading_enabled=%s. %s",
        "=" * 60, settings.broker.live_trading_enabled,
        "REAL ORDERS WILL BE PLACED." if settings.broker.live_trading_enabled else "DRY RUN ONLY -- no real orders will be placed.",
    )

    broker = RobinhoodBroker()
    retry_with_backoff(broker.connect)
    retry_with_backoff(broker.verify_account)
    chain_fetcher = RobinhoodOptionChainFetcher()
    intraday_fetcher = build_intraday_fetcher(data_source, broker)
    daily_fetcher = YFinanceHistoricalFetcher()
    executor = build_executor(broker, chain_fetcher, settings)

    logger.info("Watching %s on %s bars (source=%s), polling every %ds", symbols, interval, data_source, poll_seconds)

    while True:
        try:
            run_once(broker, intraday_fetcher, daily_fetcher, executor, symbols, interval)
        except Exception:
            logger.exception("Cycle failed -- will retry next cycle rather than crash the process")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    from pathlib import Path

    from dotenv import load_dotenv

    # Explicit path, not load_dotenv()'s bare default. That default walks
    # up from the CALLING FILE's location (stack-frame based), not the
    # process's working directory -- confirmed the hard way: running this
    # script by an absolute path from outside the project silently failed
    # to find .env even though `cwd` was correct, while `python -c "..."`
    # from the same cwd found it fine. Under systemd (WorkingDirectory=
    # set, but ExecStart also uses an absolute path) this exact ambiguity
    # would have made .env loading depend on undocumented dotenv internals
    # instead of something predictable -- not a risk worth taking for a
    # process that unattended-loads live trading credentials.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--data-source", default=DEFAULT_DATA_SOURCE, choices=["alpaca", "robinhood"])
    parser.add_argument(
        "--interval", default=DEFAULT_INTERVAL,
        choices=sorted(set(RobinhoodIntradayHistoricalFetcher._INTERVAL_SPAN_MAP) | set(AlpacaIntradayHistoricalFetcher._TIMEFRAME_MAP)),
        help="Not every interval is supported by every --data-source (robinhood: 5m/1h only; alpaca: 1m/5m/15m/30m/1h) -- an unsupported combo fails fast with a clear error.",
    )
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    args = parser.parse_args()
    run_forever(args.symbols, args.settings, args.interval, args.poll_seconds, args.data_source)
