"""The Bash-callable half of the MCP-agent architecture (see the migration
plan this was built from). A Claude Code scheduled routine, holding the
Robinhood MCP tools, is the only thing that can actually read the account or
place an order under the sanctioned "Agentic Trading" path -- this script
supplies the DECISION, never the execution. It never imports robin_stocks
and never touches the network for a broker call; the only outbound call
here is Alpaca market data (unchanged from orchestration/run_live.py).

Three subcommands, matching the three points in a cycle where the routine
needs a verdict from tested Python instead of its own judgment:

  evaluate     Run the full FVG+confluence pipeline (unchanged --
               brain/options_strategy.decide_options_action, the same
               force-close/FVG-invalidation/stop-loss/take-profit checks
               orchestration/options_execution.py has always run) against
               the account state the agent already fetched via MCP. Prints
               one action per symbol: hold, a fully-resolved close (a
               known contract, priced from a quote the agent supplied), or
               an open SIGNAL ONLY (direction + conviction -- no contract
               yet, since picking one needs a live MCP chain lookup only
               the agent can do).

  size_check   Given a specific contract's live price (from the agent's own
               MCP chain lookup), wraps safety/options_sizing.compute_contract_count
               and safety/order_validation.run_order_checks UNCHANGED --
               same spread cap, same per-trade/portfolio sizing caps, same
               duplicate-order guard. The agent never computes any of this
               itself, only relays the verdict.

  record       Appends the REAL outcome (did the agent actually place the
               order? what order_id?) to trade_log.jsonl/activity_log.jsonl,
               using the exact same TradeLogEntry/ActivityEntry shapes the
               old broker-mode executor always wrote. This is deferred
               here, not written by `evaluate`, because `evaluate` runs
               BEFORE the agent has actually called place_option_order --
               see orchestration/options_execution.py's OptionsOrderExecutor
               class docstring for why agent mode never writes trade_log.jsonl
               itself.

`live_trading_enabled` is read straight from settings.yaml, exactly like it
always has been -- this script reports it in `evaluate`'s output but never
overrides it, and the routine's own prompt (not this file) is what
determines whether review_option_order/place_option_order get called at
all. Nothing in this file is a kill switch; settings.yaml still is.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config.config_loader import load_settings
from data.fetchers import AlpacaIntradayHistoricalFetcher, YFinanceHistoricalFetcher
from data.options_data import OptionContract
from orchestration.activity_log import ActivityEntry
from orchestration.activity_log import DEFAULT_LOG_PATH as DEFAULT_ACTIVITY_LOG_PATH
from orchestration.activity_log import append_entry as append_activity_entry
from orchestration.options_execution import OpenOptionPosition, OptionsOrderExecutor
from orchestration.trade_log import DEFAULT_LOG_PATH as DEFAULT_TRADE_LOG_PATH
from orchestration.trade_log import TradeLogEntry, append_entry as append_trade_entry
from safety.options_sizing import compute_contract_count
from safety.order_validation import DuplicateOrderGuard, run_order_checks

DAILY_LOOKBACK_DAYS = 400
ALPACA_INTRADAY_LOOKBACK_DAYS = 20


def _build_executor(settings) -> OptionsOrderExecutor:
    opt = settings.options
    return OptionsOrderExecutor(
        broker=None, chain_fetcher=None,  # agent mode -- see OptionsOrderExecutor's class docstring
        max_premium_pct_per_trade=opt.max_premium_pct_per_trade,
        max_total_premium_pct_of_equity=opt.max_total_premium_pct_of_equity,
        close_before_expiration_days=opt.close_before_expiration_days,
        dte_min=opt.target_dte_min, dte_max=opt.target_dte_max,
        max_spread_pct=settings.risk.max_bid_ask_spread_pct,
        fvg_lookback_period=opt.fvg_lookback_period, fvg_body_multiplier=opt.fvg_body_multiplier,
        fvg_volume_multiplier=opt.fvg_volume_multiplier, sma_period=opt.sma_period,
        min_confluence_score=opt.min_confluence_score, stop_loss_pct=opt.stop_loss_pct,
        take_profit_pct=opt.take_profit_pct,
        live_trading_enabled=settings.broker.live_trading_enabled,
    )


def _parse_open_positions(raw: list) -> Dict[str, OpenOptionPosition]:
    positions = {}
    for p in raw:
        pos = OpenOptionPosition(
            symbol=p["symbol"], option_type=p["option_type"], strike_price=float(p["strike_price"]),
            quantity=int(p["quantity"]), expiration_date=date.fromisoformat(p["expiration_date"]),
            average_premium_paid=float(p["average_premium_paid"]),
        )
        positions[pos.symbol] = pos
    return positions


def _parse_quotes(raw: dict) -> Dict[str, OptionContract]:
    quotes = {}
    for symbol, q in raw.items():
        quotes[symbol] = OptionContract(
            id=q.get("id", symbol), symbol=symbol, option_type=q.get("option_type", ""),
            strike_price=float(q.get("strike_price", 0) or 0),
            expiration_date=date.fromisoformat(q["expiration_date"]) if q.get("expiration_date") else date.today(),
            bid=float(q["bid"]) if q.get("bid") is not None else None,
            ask=float(q["ask"]) if q.get("ask") is not None else None,
            last_price=float(q["last_price"]) if q.get("last_price") is not None else None,
        )
    return quotes


def cmd_evaluate(args: argparse.Namespace) -> None:
    settings = load_settings(args.settings)
    symbols = args.symbols or settings.broker.core_watchlist

    payload = json.load(sys.stdin)
    equity = float(payload["equity"])
    buying_power = float(payload["buying_power"])
    open_positions = _parse_open_positions(payload.get("open_positions", []))
    open_position_quotes = _parse_quotes(payload.get("open_position_quotes", {}))
    open_order_symbols = payload.get("open_order_symbols", [])

    intraday_fetcher = AlpacaIntradayHistoricalFetcher()
    daily_fetcher = YFinanceHistoricalFetcher()
    now = datetime.now(timezone.utc)

    end = now
    start = end - pd.Timedelta(days=ALPACA_INTRADAY_LOOKBACK_DAYS)
    intraday_bars = {}
    for symbol in symbols:
        try:
            intraday_bars[symbol] = intraday_fetcher.get_bars(symbol, start.isoformat(), end.isoformat(), timeframe="5m")
        except Exception as exc:
            print(f"WARNING: {symbol}: intraday fetch failed ({exc}) -- skipping this cycle", file=sys.stderr)

    # end.date() - 1, not end.date(): `now` is UTC, and UTC is ahead of US
    # market time -- asking yfinance for a daily bar "through today (UTC)"
    # can mean a US trading day that hasn't happened yet (e.g. late evening
    # ET is already past midnight UTC), which comes back as a NaN row and
    # fails the fetcher's own NaN check. One day of slack is irrelevant to
    # a 200-day SMA, so just don't ask for a day that might not exist yet.
    daily_end = end.date() - pd.Timedelta(days=1)
    daily_start = daily_end - pd.Timedelta(days=DAILY_LOOKBACK_DAYS)
    daily_bars = {}
    for symbol in symbols:
        try:
            daily_bars[symbol] = daily_fetcher.get_bars(symbol, daily_start.isoformat(), daily_end.isoformat(), timeframe="1D")
        except Exception as exc:
            print(f"WARNING: {symbol}: daily fetch failed ({exc}) -- SMA200 veto will fall back to intraday bars", file=sys.stderr)

    if not intraday_bars:
        print(json.dumps({"live_trading_enabled": settings.broker.live_trading_enabled, "fetched_at": now.isoformat(), "records": [], "error": "no intraday bars fetched for any symbol"}))
        return

    executor = _build_executor(settings)
    records = executor.run(
        bars=intraday_bars, equity=equity, open_positions=open_positions, buying_power=buying_power,
        open_order_symbols=open_order_symbols, now=now, daily_bars=daily_bars,
        open_position_quotes=open_position_quotes,
    )

    print(json.dumps({
        "live_trading_enabled": settings.broker.live_trading_enabled,
        "fetched_at": now.isoformat(),
        "records": [asdict(r) for r in records],
    }, default=str))


def cmd_size_check(args: argparse.Namespace) -> None:
    settings = load_settings(args.settings)
    opt = settings.options

    sizing = compute_contract_count(
        equity=args.equity, contract_price=args.contract_price, conviction=args.conviction,
        current_total_premium_at_risk=args.current_total_premium_at_risk,
        max_premium_pct_per_trade=opt.max_premium_pct_per_trade,
        max_total_premium_pct_of_equity=opt.max_total_premium_pct_of_equity,
        buying_power=args.buying_power,
    )

    result = {"contracts": sizing.contracts, "premium": sizing.premium, "binding_cap": sizing.binding_cap, "below_minimum": sizing.below_minimum}

    if sizing.contracts == 0:
        result["ok"] = False
        result["reason"] = f"sizing produced zero contracts (binding_cap={sizing.binding_cap}, below_minimum={sizing.below_minimum})"
        print(json.dumps(result))
        return

    open_order_symbols = json.loads(args.open_order_symbols) if args.open_order_symbols else []
    check = run_order_checks(
        symbol=args.symbol, side="buy", order_notional=sizing.premium, buying_power=args.buying_power,
        tradable=True, bid=args.bid, ask=args.ask, now=datetime.now(timezone.utc),
        duplicate_guard=DuplicateOrderGuard(), open_order_symbols=open_order_symbols,
        max_spread_pct=settings.risk.max_bid_ask_spread_pct,
    )
    result["ok"] = check.ok
    result["reason"] = check.reason
    print(json.dumps(result))


def cmd_record(args: argparse.Namespace) -> None:
    now = datetime.now(timezone.utc)
    confluence_details = json.loads(args.confluence_details) if args.confluence_details else {}

    append_trade_entry(TradeLogEntry(
        event=args.event, timestamp=now.isoformat(), symbol=args.symbol, asset_type="option",
        trade_type=args.option_type, quantity=args.quantity, price=args.price, notional=args.notional,
        dry_run=not args.live, order_id=args.order_id, reason=args.reason,
        gap_low=args.gap_low, gap_high=args.gap_high,
        strike_price=args.strike_price,
        expiration_date=args.expiration_date,
        dte_at_entry=args.dte_at_entry, bid=args.bid, ask=args.ask, spread_pct=args.spread_pct,
        confluence_score=args.confluence_score, confluence_applicable=args.confluence_applicable,
        confluence_details=confluence_details,
    ), path=Path(args.trade_log_path))

    outcome = args.event
    append_activity_entry(ActivityEntry(
        timestamp=now.isoformat(), symbol=args.symbol, outcome=outcome,
        option_type=args.option_type, contracts=args.quantity,
        detail=f"{'[LIVE]' if args.live else '[DRY RUN]'} {args.event} {args.quantity} {args.option_type} -- {args.reason or ''}".strip(),
        price=args.price,
    ), path=Path(args.activity_log_path))

    print(json.dumps({"recorded": True, "event": args.event, "symbol": args.symbol, "live": args.live}))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="Run the decision pipeline; account state comes from stdin JSON")
    p_eval.add_argument("--symbols", nargs="*", default=None, help="Defaults to settings.yaml's broker.core_watchlist")
    p_eval.add_argument("--settings", default="config/settings.yaml")
    p_eval.set_defaults(func=cmd_evaluate)

    p_size = sub.add_parser("size_check", help="Validate sizing/spread/duplicate checks for a specific contract")
    p_size.add_argument("--symbol", required=True)
    p_size.add_argument("--contract-price", type=float, required=True)
    p_size.add_argument("--conviction", type=float, required=True)
    p_size.add_argument("--equity", type=float, required=True)
    p_size.add_argument("--buying-power", type=float, required=True)
    p_size.add_argument("--current-total-premium-at-risk", type=float, required=True)
    p_size.add_argument("--bid", type=float, default=None)
    p_size.add_argument("--ask", type=float, default=None)
    p_size.add_argument("--open-order-symbols", default=None, help="JSON array")
    p_size.add_argument("--settings", default="config/settings.yaml")
    p_size.set_defaults(func=cmd_size_check)

    p_rec = sub.add_parser("record", help="Persist the real outcome of an executed (or dry-run) order")
    p_rec.add_argument("--event", choices=["open", "close"], required=True)
    p_rec.add_argument("--symbol", required=True)
    p_rec.add_argument("--option-type", required=True)
    p_rec.add_argument("--quantity", type=int, required=True)
    p_rec.add_argument("--price", type=float, required=True)
    p_rec.add_argument("--notional", type=float, required=True)
    p_rec.add_argument("--live", action="store_true", help="Omit for a dry-run record")
    p_rec.add_argument("--order-id", default=None)
    p_rec.add_argument("--reason", default=None)
    p_rec.add_argument("--gap-low", type=float, default=None)
    p_rec.add_argument("--gap-high", type=float, default=None)
    p_rec.add_argument("--strike-price", type=float, default=None)
    p_rec.add_argument("--expiration-date", default=None)
    p_rec.add_argument("--dte-at-entry", type=int, default=None)
    p_rec.add_argument("--bid", type=float, default=None)
    p_rec.add_argument("--ask", type=float, default=None)
    p_rec.add_argument("--spread-pct", type=float, default=None)
    p_rec.add_argument("--confluence-score", type=float, default=None)
    p_rec.add_argument("--confluence-applicable", type=int, default=None)
    p_rec.add_argument("--confluence-details", default=None, help="JSON object")
    p_rec.add_argument("--trade-log-path", default=str(DEFAULT_TRADE_LOG_PATH))
    p_rec.add_argument("--activity-log-path", default=str(DEFAULT_ACTIVITY_LOG_PATH))
    p_rec.set_defaults(func=cmd_record)

    parsed = parser.parse_args()
    parsed.func(parsed)
