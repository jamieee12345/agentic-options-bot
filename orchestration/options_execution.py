"""Turns a Fair Value Gap + volume momentum signal into simple long-call/
long-put options orders. Runs entirely off raw OHLCV bars now (via
brain/options_strategy.decide_options_action) -- no dependency on the
equity regime brain (brain/signal_generator.Signal) at all anymore.

Deliberately simple position model, matching the "smallest lift" scope this
was built for: **at most one open options position per underlying at a
time** (never stacks calls and puts on the same symbol, never adds to an
existing options position). Each bar, per symbol, exactly one of:

    force-close (approaching expiration, checked FIRST and unconditionally)
    -> flip (a fresh opposing FVG signal) -> close (signal explicitly says
    "close") -> open a new position (signal wants a side and nothing's
    currently open) -> hold (already correctly positioned, no new signal,
    or nothing to do)

"hold" is NOT the same as "close": the strategy saying "no fresh FVG this
bar" does not by itself exit an existing position -- only a fresh opposing
signal (flip), an explicit close, or approaching expiration does. A quiet
bar between momentum bursts is expected, not a reason to bail.

A flip closes the wrong-side position this bar and opens the new one on a
LATER bar once the close has actually cleared -- this never tries to close
and open the same symbol in one pass, to avoid double-counting buying power
or the portfolio premium cap against a position that hasn't settled yet.

No assignment-risk handling needed here (that's specific to being SHORT an
option -- covered calls, cash-secured puts, naked writes -- none of which
this does). Six things this DOES manage for an existing position, checked
BEFORE the signal, in this order:
  1. Max hold (`max_hold_days`) -- unconditional, checked FIRST: once a
     position has been open this many calendar days, force-close it no
     matter what, regardless of DTE-at-entry, P&L, or the FVG signal. This
     is what actually enforces "same-day and single-overnight trades
     only" (target_dte_min=1/target_dte_max=2) -- narrowed from a
     multi-day swing window on request, specifically to bound how much
     theta decay any position can ever be exposed to.
  2. Expiration (`close_before_expiration_days`) -- now a LAST-RESORT
     backstop (0 = force-close once actually on the expiration day
     itself), since #1 fires first in the overwhelming majority of cases
     given the narrow 1-2 DTE window.
  3. FVG invalidation -- structural stop, not a dollar one: has price
     closed back through the ENTIRE gap that triggered this trade? If so
     the premise is gone regardless of what the option's current value
     says. Needs the triggering gap's bounds, persisted at open time via
     orchestration/trade_log.py (survives the bot stopping overnight).
  4. Stop-loss (`stop_loss_pct`) -- the dollar backstop underneath #3:
     force-close once current value has fallen this fraction below entry,
     independent of any structural read. Catches gap risk / cases the FVG
     check doesn't -- both #3 and #4 exist because either can fire first.
  5. Take-profit (`take_profit_pct`) -- lock in a gain rather than ride it
     indefinitely. Nothing in this system confirms a move will CONTINUE
     once it's already worked -- confluence only validates entries, not
     continuation -- so an open-ended hold on a winner has no more
     informational backing than one on a loser.
  6. Stagnation (`stagnant_exit_hold_fraction`/`stagnant_exit_min_pnl_pct`)
     -- "close if going nowhere": only reachable if #4/#5 didn't already
     fire, so this only affects a position in the boring middle. Once held
     for `stagnant_exit_hold_fraction` of its own dte_at_entry without
     reaching `stagnant_exit_min_pnl_pct`, force-close rather than let it
     ride toward an expiration close. Still meaningfully tighter than #1
     for a 1-DTE position specifically (0.5 * 1 day = 12h vs. max_hold_days'
     24h) -- for a 2-DTE position the two thresholds coincide (0.5 * 2 = 1
     day), so #1 wins the race there since it's unconditional.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from brain.options_strategy import OptionsDecision, decide_options_action
from data.options_data import OptionContract, RobinhoodOptionChainFetcher
from orchestration.activity_log import ActivityEntry
from orchestration.activity_log import DEFAULT_LOG_PATH as DEFAULT_ACTIVITY_LOG_PATH
from orchestration.activity_log import append_entry as append_activity_entry
from orchestration.trade_log import DEFAULT_LOG_PATH, TradeLogEntry, append_entry, build_trade_history, read_entries
from safety.options_sizing import compute_contract_count
from safety.order_validation import DuplicateOrderGuard, run_order_checks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenOptionPosition:
    symbol: str
    option_type: str          # "call" or "put"
    strike_price: float
    quantity: int
    expiration_date: date
    average_premium_paid: float  # per contract, i.e. per 100 shares of exposure -- same convention as Robinhood's average_buy_price for equities


def build_open_positions_from_broker(raw_positions: List[dict]) -> Dict[str, OpenOptionPosition]:
    """Adapts broker.get_option_positions()'s raw robin_stocks dicts into
    OpenOptionPosition. Field names below (chain_symbol, strike_price,
    expiration_date, quantity, average_price, type) match robin_stocks'
    commonly documented get_open_option_positions() shape -- like the rest
    of this file, **not verified against a live response in this sandbox**;
    confirm against your own account's actual output before relying on it.
    Positions with zero quantity are dropped (a fully-closed position that
    still shows up as a zero-quantity row, same pattern as the equity side).
    """
    positions: Dict[str, OpenOptionPosition] = {}
    for raw in raw_positions:
        quantity = int(float(raw.get("quantity", 0) or 0))
        if quantity == 0:
            continue
        symbol = raw["chain_symbol"]
        positions[symbol] = OpenOptionPosition(
            symbol=symbol,
            option_type=raw["type"],
            strike_price=float(raw["strike_price"]),
            quantity=quantity,
            expiration_date=datetime.strptime(raw["expiration_date"], "%Y-%m-%d").date(),
            average_premium_paid=float(raw.get("average_price", 0) or 0),
        )
    return positions


@dataclass(frozen=True)
class OptionsExecutionRecord:
    symbol: str
    action: Optional[str]      # "open", "close", None if nothing happened
    option_type: Optional[str]
    contracts: Optional[int]
    placed: bool
    order_id: Optional[str]
    skipped_reason: Optional[str]
    # Carried straight from brain.options_strategy.OptionsDecision (when one
    # ran this cycle -- see run()) so the activity log can persist the full
    # reasoning, not just `skipped_reason`'s one-line summary. Left at
    # defaults for the executor-level branches that manage an EXISTING
    # position (force-close, FVG invalidation, stop-loss/take-profit) --
    # those aren't evaluating a fresh signal, so there's no decision to show.
    price: Optional[float] = None
    gap_kind: Optional[str] = None
    volume_confirmed: Optional[bool] = None
    confluence_details: Dict[str, str] = field(default_factory=dict)
    confluence_score: Optional[float] = None
    confluence_applicable: int = 0
    # Set only in AGENT MODE (chain_fetcher=None, see OptionsOrderExecutor's
    # docstring) for a close or open-signal this run() couldn't fully resolve
    # itself -- no broker/chain_fetcher means it can't fetch a contract, fetch
    # a fresh quote, or place an order, so it hands back exactly what a
    # caller (a Claude agent driving Robinhood's MCP tools) needs to finish
    # the job: for a close, the known contract + a limit price (already
    # resolved from a supplied quote); for an open, only a direction/
    # conviction -- the agent must still pick a contract via MCP and run a
    # separate size-check before anything can be priced or sized. Never
    # combined with placed=True -- agent-mode never places an order itself.
    pending_action: Optional[Dict] = None


class OptionsOrderExecutor:
    """Two modes, selected purely by whether `chain_fetcher` is given:

    BROKER MODE (chain_fetcher set, the original/default behavior, still
    used by orchestration/run_live.py) -- this class does everything itself:
    fetches contracts and quotes via chain_fetcher, places orders via
    `broker.place_option_order` when live_trading_enabled, and writes
    trade_log.jsonl immediately as each decision is made.

    AGENT MODE (chain_fetcher=None, broker=None) -- for the Robinhood MCP
    migration (see the plan this was built from): this class can't reach
    the broker or fetch its own contracts/quotes, so it never calls
    place_option_order and never writes trade_log.jsonl itself. Instead it
    returns `OptionsExecutionRecord.pending_action` for anything that would
    need a broker call, and the CALLER (orchestration/evaluate_for_agent.py,
    driven by a Claude agent with the Robinhood MCP tools) finishes the job:
    fetching contracts/quotes via MCP, sizing/validating via this same
    module's compute_contract_count/run_order_checks, calling
    review_option_order/place_option_order itself, and recording the result
    to trade_log.jsonl via evaluate_for_agent.py's own `record` mode. This
    keeps every safety number (spread cap, sizing caps, DTE window, the
    confluence gate) in this exact tested code either way -- only WHO makes
    the final broker call changes between modes.
    """

    def __init__(
        self,
        broker,
        chain_fetcher: Optional[RobinhoodOptionChainFetcher],
        max_premium_pct_per_trade: float,
        max_total_premium_pct_of_equity: float,
        close_before_expiration_days: int,
        dte_min: int,
        dte_max: int,
        max_spread_pct: float,
        fvg_lookback_period: int,
        fvg_body_multiplier: float,
        fvg_volume_multiplier: float,
        sma_period: int,
        min_confluence_score: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        stagnant_exit_hold_fraction: float,
        stagnant_exit_min_pnl_pct: float,
        max_hold_days: int,
        live_trading_enabled: bool = False,
        duplicate_guard: Optional[DuplicateOrderGuard] = None,
        trade_log_path: Path = DEFAULT_LOG_PATH,
        activity_log_path: Path = DEFAULT_ACTIVITY_LOG_PATH,
    ) -> None:
        self.broker = broker
        self.chain_fetcher = chain_fetcher
        self._agent_mode = chain_fetcher is None  # see class docstring
        self.max_premium_pct_per_trade = max_premium_pct_per_trade
        self.max_total_premium_pct_of_equity = max_total_premium_pct_of_equity
        self.close_before_expiration_days = close_before_expiration_days
        self.dte_min = dte_min
        self.dte_max = dte_max
        self.max_spread_pct = max_spread_pct
        self.fvg_lookback_period = fvg_lookback_period
        self.fvg_body_multiplier = fvg_body_multiplier
        self.fvg_volume_multiplier = fvg_volume_multiplier
        self.sma_period = sma_period
        self.min_confluence_score = min_confluence_score
        # Fraction of entry premium a long position is allowed to lose before
        # being force-closed -- e.g. 0.50 closes a position once it's worth
        # half what was paid for it. A long option's max loss is already
        # capped at 100% of premium by construction; this cuts losses well
        # before that, rather than riding every losing trade to worthless or
        # expiration.
        self.stop_loss_pct = stop_loss_pct
        # Fraction of entry premium a position must GAIN before being
        # force-closed to lock it in. No indicator in this system confirms
        # a favorable move will keep going -- confluence is an entry
        # filter, not a continuation forecast -- so an unbounded hold on a
        # winner is exposed the same way an unbounded hold on a loser
        # would be without stop_loss_pct.
        self.take_profit_pct = take_profit_pct
        # "Close if going nowhere" -- see the class/module docstrings for
        # why. Only ever reached after stop_loss_pct/take_profit_pct have
        # both failed to fire this bar (see _check_price_based_exit).
        self.stagnant_exit_hold_fraction = stagnant_exit_hold_fraction
        self.stagnant_exit_min_pnl_pct = stagnant_exit_min_pnl_pct
        # Unconditional holding-time cap, checked FIRST in run() -- see
        # class/module docstrings. Independent of DTE-at-entry and P&L.
        self.max_hold_days = max_hold_days
        self.live_trading_enabled = live_trading_enabled
        self.duplicate_guard = duplicate_guard or DuplicateOrderGuard()
        self.trade_log_path = trade_log_path
        self.activity_log_path = activity_log_path

    def run(
        self,
        bars: Dict[str, pd.DataFrame],
        equity: float,
        open_positions: Dict[str, OpenOptionPosition],
        buying_power: float,
        open_order_symbols: List[str],
        now: datetime,
        daily_bars: Optional[Dict[str, pd.DataFrame]] = None,
        open_position_quotes: Optional[Dict[str, "OptionContract"]] = None,
    ) -> List[OptionsExecutionRecord]:
        """`bars` drives FVG/structure/S-R/etc -- pass intraday bars here for
        live trading so the strategy reacts within the trading day, not just
        once it closes. `daily_bars`, if given, is used only for the 200-SMA
        trend veto per symbol (see brain/confluence.py's docstring for why
        that stays on a daily interval regardless of what `bars` is).

        `open_position_quotes` is AGENT MODE ONLY (ignored in broker mode,
        which fetches its own quotes via chain_fetcher): a fresh quote per
        symbol with an existing open position, keyed by symbol -- the caller
        (an agent with the Robinhood MCP tools) fetches these once up front
        via get_option_quotes, since this class has no chain_fetcher to do
        it itself in agent mode. Needed for the stop-loss/take-profit check
        and for pricing a close's limit price. A symbol with an open
        position but no entry here just skips that symbol's price-based
        exit check this cycle (fails open, same as a failed quote fetch
        does in broker mode) rather than erroring.
        """
        current_total_premium_at_risk = sum(p.quantity * p.average_premium_paid * 100 for p in open_positions.values())
        records: List[OptionsExecutionRecord] = []

        # Read once per run(), not once per symbol -- one open position per
        # symbol at a time means this covers every symbol's entry context
        # in a single pass over the log.
        _closed_unused, still_open_trades = build_trade_history(read_entries(self.trade_log_path))
        open_trade_by_symbol = {t.symbol: t for t in still_open_trades}

        for symbol, symbol_bars in bars.items():
            existing = open_positions.get(symbol)
            current_price = float(symbol_bars["close"].iloc[-1])

            if existing is not None:
                open_trade = open_trade_by_symbol.get(symbol)
                if open_trade is not None:
                    try:
                        opened_at = datetime.fromisoformat(open_trade.opened_at)
                    except (TypeError, ValueError):
                        opened_at = None
                    if opened_at is not None and (now.date() - opened_at.date()).days >= self.max_hold_days:
                        records.append(self._close(
                            symbol, existing, now, open_order_symbols,
                            f"max_hold: held since {opened_at.date().isoformat()}, {self.max_hold_days}d cap reached -- "
                            f"same-day/overnight-only strategy, forced close regardless of DTE or P&L",
                            open_position_quotes,
                        ))
                        continue

                if (existing.expiration_date - now.date()).days <= self.close_before_expiration_days:
                    records.append(self._close(symbol, existing, now, open_order_symbols, "approaching expiration -- forced close", open_position_quotes))
                    continue

                fvg_record = self._check_fvg_invalidation(
                    symbol, existing, current_price, open_trade_by_symbol.get(symbol), now, open_order_symbols, open_position_quotes,
                )
                if fvg_record is not None:
                    records.append(fvg_record)
                    continue

                price_exit_record = self._check_price_based_exit(
                    symbol, existing, open_trade_by_symbol.get(symbol), now, open_order_symbols, open_position_quotes,
                )
                if price_exit_record is not None:
                    records.append(price_exit_record)
                    continue

            decision = decide_options_action(
                symbol, symbol_bars, self.fvg_lookback_period, self.fvg_body_multiplier, self.fvg_volume_multiplier,
                self.sma_period, self.min_confluence_score,
                daily_bars=(daily_bars or {}).get(symbol),
            )
            # Every branch below that stems from `decision` (not from an
            # executor-level position-management check above) carries the
            # full read-out forward -- gap/volume/confluence detail, not
            # just the one-line `decision.reasoning` -- so the dashboard's
            # live-reasoning panel can show exactly what was looked at.
            decision_fields = dict(
                price=current_price, gap_kind=decision.gap_kind, volume_confirmed=decision.volume_confirmed,
                confluence_details=decision.confluence_details, confluence_score=decision.confluence_score,
                confluence_applicable=decision.confluence_applicable,
            )

            if decision.action == "hold":
                records.append(OptionsExecutionRecord(
                    symbol, None, existing.option_type if existing else None, None, False, None, decision.reasoning,
                    **decision_fields,
                ))
                continue

            if decision.action == "close":
                if existing is not None:
                    records.append(self._close(symbol, existing, now, open_order_symbols, decision.reasoning, open_position_quotes))
                else:
                    records.append(OptionsExecutionRecord(
                        symbol, None, None, None, False, None, f"nothing open -- {decision.reasoning}", **decision_fields,
                    ))
                continue

            wanted_type = "call" if decision.action == "buy_call" else "put"

            if existing is not None and existing.option_type != wanted_type:
                records.append(self._close(
                    symbol, existing, now, open_order_symbols, f"flipping {existing.option_type}->{wanted_type}: {decision.reasoning}", open_position_quotes,
                ))
                continue

            if existing is not None and existing.option_type == wanted_type:
                records.append(OptionsExecutionRecord(
                    symbol, None, wanted_type, None, False, None, f"already holding a {wanted_type}, not adding to it",
                    **decision_fields,
                ))
                continue

            if self._agent_mode:
                # No chain_fetcher to pick a contract with -- the caller
                # (an agent with the Robinhood MCP tools) must fetch a
                # chain/instrument/quote itself, then run this module's
                # compute_contract_count/run_order_checks (via
                # evaluate_for_agent.py's size_check mode) before anything
                # here can be priced, sized, or placed.
                records.append(OptionsExecutionRecord(
                    symbol, None, wanted_type, None, False, None, f"awaiting agent execution -- {decision.reasoning}",
                    pending_action={
                        "type": "open", "wanted_type": wanted_type, "conviction": decision.conviction,
                        "gap_low": decision.gap_low, "gap_high": decision.gap_high, "reasoning": decision.reasoning,
                    },
                    **decision_fields,
                ))
            else:
                records.append(self._open(symbol, current_price, wanted_type, decision, equity, current_total_premium_at_risk, buying_power, now, open_order_symbols))

        # One pass, logged after the loop rather than at each individual
        # records.append() call above -- simpler to keep correct than
        # threading a log call through every branch, and the outcome is
        # identical either way (nothing here depends on log-write timing).
        for record in records:
            outcome = record.action or ("skipped" if record.skipped_reason and "already holding" not in record.skipped_reason else "hold")
            append_activity_entry(ActivityEntry(
                timestamp=now.isoformat(), symbol=record.symbol, outcome=outcome,
                option_type=record.option_type, contracts=record.contracts,
                detail=record.skipped_reason or f"{record.action} {record.contracts or ''} {record.option_type or ''}".strip(),
                price=record.price, gap_kind=record.gap_kind, volume_confirmed=record.volume_confirmed,
                confluence_details=record.confluence_details, confluence_score=record.confluence_score,
                confluence_applicable=record.confluence_applicable,
            ), path=self.activity_log_path)

        return records

    def _check_fvg_invalidation(
        self, symbol, existing: OpenOptionPosition, current_price: float, open_trade, now, open_order_symbols,
        open_position_quotes: Optional[Dict[str, "OptionContract"]] = None,
    ) -> Optional[OptionsExecutionRecord]:
        """Structural stop: has price closed back through the ENTIRE Fair
        Value Gap that triggered this position? If so the setup's premise
        is gone, independent of what the option's dollar value says --
        checked BEFORE the price-based stop/take-profit for exactly that
        reason. `open_trade` is this symbol's own logged "open" entry
        (orchestration/trade_log.py), which is where the triggering gap's
        bounds actually live -- they don't exist anywhere else, since
        OpenOptionPosition comes from the broker and has no memory of why
        the position was opened.

        Runs off `current_price` (the underlying's last intraday close),
        not an options quote -- cheaper (no extra API call) and more
        direct: the gap was defined on the underlying's price, not the
        option's.
        """
        if open_trade is None or open_trade.gap_low is None or open_trade.gap_high is None:
            return None  # no recorded entry context (e.g. log predates this feature, or position wasn't opened by this bot) -- can't check, not a reason to block anything else

        if existing.option_type == "call" and current_price < open_trade.gap_low:
            return self._close(
                symbol, existing, now, open_order_symbols,
                f"fvg_invalidated: price ({current_price:.2f}) closed below the entry gap's low (${open_trade.gap_low:.2f}) -- bullish setup filled and broken",
                open_position_quotes,
            )
        if existing.option_type == "put" and current_price > open_trade.gap_high:
            return self._close(
                symbol, existing, now, open_order_symbols,
                f"fvg_invalidated: price ({current_price:.2f}) closed above the entry gap's high (${open_trade.gap_high:.2f}) -- bearish setup filled and broken",
                open_position_quotes,
            )
        return None

    def _check_price_based_exit(
        self, symbol, existing: OpenOptionPosition, open_trade, now, open_order_symbols,
        open_position_quotes: Optional[Dict[str, "OptionContract"]] = None,
    ) -> Optional[OptionsExecutionRecord]:
        """Stop-loss, take-profit, and the stagnation ("close if going
        nowhere") check, all from one quote -- checked every bar an
        existing position isn't already being closed for expiration or FVG
        invalidation. Needs a fresh quote regardless of what the FVG/
        confluence signal says this bar -- monitoring an open long option's
        current value is a bar-by-bar obligation, not something that can
        wait for a new signal to show up.

        Broker mode fetches that quote itself via chain_fetcher; agent mode
        has no chain_fetcher, so it reads from the caller-supplied
        `open_position_quotes` instead (see run()'s docstring) -- a missing
        entry there is treated the same as a failed fetch in broker mode:
        fail open, skip the check this cycle, don't error.

        `open_trade` is this symbol's own logged "open" entry
        (orchestration/trade_log.py) -- same source _check_fvg_invalidation
        uses, needed here for `opened_at`/`dte_at_entry` to evaluate the
        stagnation check. A missing `open_trade` (or a missing
        dte_at_entry, e.g. a log predating this field) just skips the
        stagnation check, not the stop-loss/take-profit checks above it.
        """
        if existing.average_premium_paid <= 0:
            return None  # no known entry cost to measure P&L against (e.g. adapter couldn't read average_price) -- fail open rather than close on bad data

        if self._agent_mode:
            quote = (open_position_quotes or {}).get(symbol)
        else:
            quote = self.chain_fetcher.get_quote_for_known_contract(
                symbol, existing.option_type, existing.expiration_date, existing.strike_price,
            )
        if quote is None or quote.mid_price is None:
            logger.warning("%s: no fresh quote available for the stop-loss/take-profit check this bar -- skipping", symbol)
            return None

        current_value = quote.mid_price * existing.quantity * 100
        entry_value = existing.average_premium_paid * existing.quantity * 100
        pnl_pct = (current_value / entry_value) - 1  # positive = gain, negative = loss

        if -pnl_pct >= self.stop_loss_pct:
            return self._close(
                symbol, existing, now, open_order_symbols,
                f"stop_loss: down {-pnl_pct:.0%} from entry (${entry_value:.2f} -> ${current_value:.2f}), limit is {self.stop_loss_pct:.0%}",
                open_position_quotes,
            )
        if pnl_pct >= self.take_profit_pct:
            return self._close(
                symbol, existing, now, open_order_symbols,
                f"take_profit: up {pnl_pct:.0%} from entry (${entry_value:.2f} -> ${current_value:.2f}), target is {self.take_profit_pct:.0%}",
                open_position_quotes,
            )

        if open_trade is not None and open_trade.dte_at_entry:
            try:
                opened_at = datetime.fromisoformat(open_trade.opened_at)
            except (TypeError, ValueError):
                opened_at = None
            if opened_at is not None:
                days_held = (now.date() - opened_at.date()).days
                hold_threshold = self.stagnant_exit_hold_fraction * open_trade.dte_at_entry
                if days_held >= hold_threshold and pnl_pct < self.stagnant_exit_min_pnl_pct:
                    return self._close(
                        symbol, existing, now, open_order_symbols,
                        f"stagnant: held {days_held}d ({hold_threshold:.1f}d threshold on a {open_trade.dte_at_entry}d "
                        f"DTE-at-entry position), pnl {pnl_pct:+.0%} hasn't reached the "
                        f"{self.stagnant_exit_min_pnl_pct:+.0%} bar -- closing before theta decay accelerates into expiration",
                        open_position_quotes,
                    )
        return None

    def _close(
        self, symbol, existing: OpenOptionPosition, now, open_order_symbols, reason,
        open_position_quotes: Optional[Dict[str, "OptionContract"]] = None,
    ) -> OptionsExecutionRecord:
        check = self.duplicate_guard.check(symbol, now, open_order_symbols)
        if not check.ok:
            return OptionsExecutionRecord(symbol, "close", existing.option_type, existing.quantity, False, None, check.reason)

        logger.info("%s: closing %d %s contract(s) -- %s", symbol, existing.quantity, existing.option_type, reason)

        # Fetched before the live/dry-run branch below so the trade log gets
        # a real estimated exit notional either way -- a dry-run entry with
        # no price attached would be far less useful for judging the
        # strategy's behavior before it's trusted with real orders. Agent
        # mode has no chain_fetcher, so it uses the caller-supplied quote
        # (see run()'s docstring) instead of fetching its own.
        if self._agent_mode:
            quote = (open_position_quotes or {}).get(symbol)
        else:
            quote = self.chain_fetcher.get_quote_for_known_contract(
                symbol, existing.option_type, existing.expiration_date, existing.strike_price,
            )
        limit_price = quote.mid_price if quote is not None else None
        if not limit_price or limit_price <= 0:
            return OptionsExecutionRecord(
                symbol, "close", existing.option_type, existing.quantity, False, None,
                "no usable quote to close this position -- refusing to submit a close order with no limit price",
            )

        exit_notional = limit_price * existing.quantity * 100

        if self._agent_mode:
            # Can't place the order or write trade_log.jsonl here -- no
            # broker, and the REAL outcome (was it actually filled? what
            # order_id?) isn't known until the caller (an agent) executes
            # this via the Robinhood MCP tools and reports back through
            # evaluate_for_agent.py's `record` mode. Everything the agent
            # needs to do that is in pending_action.
            return OptionsExecutionRecord(
                symbol, "close", existing.option_type, existing.quantity, False, None, None,
                pending_action={
                    "type": "close", "option_type": existing.option_type,
                    "expiration_date": existing.expiration_date.isoformat(), "strike_price": existing.strike_price,
                    "side": "sell", "position_effect": "close", "quantity": existing.quantity,
                    "limit_price": limit_price, "notional": exit_notional, "reason": reason,
                },
            )

        order_id = None
        if self.live_trading_enabled:
            order_id = self.broker.place_option_order(
                symbol=symbol, option_type=existing.option_type, expiration_date=existing.expiration_date.isoformat(),
                strike_price=existing.strike_price, side="sell", position_effect="close",
                quantity=existing.quantity, limit_price=limit_price,
            )
            self.duplicate_guard.record_submitted(symbol, now)

        append_entry(TradeLogEntry(
            event="close", timestamp=now.isoformat(), symbol=symbol, asset_type="option",
            trade_type=existing.option_type, quantity=existing.quantity, price=limit_price,
            notional=exit_notional, dry_run=not self.live_trading_enabled, order_id=order_id, reason=reason,
        ), path=self.trade_log_path)
        return OptionsExecutionRecord(symbol, "close", existing.option_type, existing.quantity, self.live_trading_enabled, order_id, None)

    def _open(self, symbol, current_price: float, wanted_type, decision: OptionsDecision, equity, current_total_premium_at_risk, buying_power, now, open_order_symbols) -> OptionsExecutionRecord:
        contract = self.chain_fetcher.get_atm_contract(symbol, wanted_type, current_price, self.dte_min, self.dte_max)
        if contract is None:
            return OptionsExecutionRecord(symbol, None, wanted_type, None, False, None, f"no tradable {wanted_type} found in [{self.dte_min},{self.dte_max}] DTE window")

        contract_price = contract.mid_price
        if contract_price is None or contract_price <= 0:
            return OptionsExecutionRecord(symbol, None, wanted_type, None, False, None, f"no usable price for contract {contract.id}")

        sizing = compute_contract_count(
            equity=equity, contract_price=contract_price, conviction=decision.conviction,
            current_total_premium_at_risk=current_total_premium_at_risk,
            max_premium_pct_per_trade=self.max_premium_pct_per_trade,
            max_total_premium_pct_of_equity=self.max_total_premium_pct_of_equity,
            buying_power=buying_power,
        )
        if sizing.contracts == 0:
            return OptionsExecutionRecord(symbol, None, wanted_type, 0, False, None, f"sizing produced zero contracts (binding_cap={sizing.binding_cap}, below_minimum={sizing.below_minimum})")

        check = run_order_checks(
            symbol=symbol, side="buy", order_notional=sizing.premium, buying_power=buying_power,
            tradable=True,  # find_tradable_options() already filters to tradable instruments; no separate halted/restricted flag surfaced here
            bid=contract.bid, ask=contract.ask, now=now, duplicate_guard=self.duplicate_guard,
            open_order_symbols=open_order_symbols, max_spread_pct=self.max_spread_pct,
        )
        if not check.ok:
            return OptionsExecutionRecord(symbol, None, wanted_type, sizing.contracts, False, None, check.reason)

        logger.info(
            "%s: opening %d %s contract(s), strike $%.2f exp %s (~$%.2f premium) -- %s",
            symbol, sizing.contracts, wanted_type, contract.strike_price, contract.expiration_date, sizing.premium, decision.reasoning,
        )

        order_id = None
        if self.live_trading_enabled:
            order_id = self.broker.place_option_order(
                symbol=symbol, option_type=wanted_type, expiration_date=contract.expiration_date.isoformat(),
                strike_price=contract.strike_price, side="buy", position_effect="open",
                quantity=sizing.contracts, limit_price=contract_price,
            )
            self.duplicate_guard.record_submitted(symbol, now)

        spread_pct = ((contract.ask - contract.bid) / contract_price) if (contract.bid is not None and contract.ask is not None and contract_price) else None

        append_entry(TradeLogEntry(
            event="open", timestamp=now.isoformat(), symbol=symbol, asset_type="option",
            trade_type=wanted_type, quantity=sizing.contracts, price=contract_price,
            notional=sizing.premium, dry_run=not self.live_trading_enabled, order_id=order_id,
            gap_low=decision.gap_low, gap_high=decision.gap_high,
            strike_price=contract.strike_price, expiration_date=contract.expiration_date.isoformat(),
            dte_at_entry=contract.days_to_expiration, bid=contract.bid, ask=contract.ask, spread_pct=spread_pct,
            confluence_score=decision.confluence_score, confluence_applicable=decision.confluence_applicable,
            confluence_details=decision.confluence_details,
        ), path=self.trade_log_path)
        return OptionsExecutionRecord(symbol, "open", wanted_type, sizing.contracts, self.live_trading_enabled, order_id, None)
