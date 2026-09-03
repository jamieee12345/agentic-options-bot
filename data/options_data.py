"""Options chain/quote data, via the unofficial `robin_stocks` client.

Same unofficial-API caveats as data/fetchers.py's RobinhoodQuoteFetcher
apply here, doubled: Robinhood has no official options API either, and
robin_stocks' options module has changed shape across versions more than
its equities module has. Everything in this file is written against the
most commonly documented pattern as of when it was written and **could not
be verified against a live account or even a live robin_stocks install in
this sandbox** (no network, no Python runtime at all here). Before trusting
this: run `get_chain_expirations` and `get_atm_contract` against one liquid
symbol (e.g. SPY) in your own environment and eyeball the output before
wiring it into anything that places real orders.

Deliberately simple contract selection for a v1 "simple long calls/puts"
strategy: nearest available expiration inside a target DTE window, and the
strike closest to the current stock price (ATM) -- no delta targeting.
Picking by delta would give more consistent directional exposure across
different implied-vol environments, but needs the chain's Greeks, which
adds another thing that could silently be missing/malformed from an
unofficial endpoint. Flagged here as the natural v2 upgrade, not built now.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional

DEFAULT_TARGET_DTE_MIN = 30
DEFAULT_TARGET_DTE_MAX = 45

# HARD FLOOR, enforced here independently of config/config_loader.py's own
# check on options.target_dte_min: 0DTE (same-day expiration) trades are
# never permitted, by explicit requirement. Two independent enforcement
# points on purpose -- a misconfigured or bypassed settings.yaml still
# can't produce a 0DTE contract selection through this class.
MIN_ALLOWED_DTE = 1


@dataclass(frozen=True)
class OptionContract:
    id: str
    symbol: str                # underlying, e.g. "AAPL"
    option_type: str           # "call" or "put"
    strike_price: float
    expiration_date: date
    bid: Optional[float]
    ask: Optional[float]
    last_price: Optional[float]

    @property
    def mid_price(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return self.last_price
        return (self.bid + self.ask) / 2

    @property
    def days_to_expiration(self) -> int:
        return (self.expiration_date - date.today()).days


class RobinhoodOptionChainFetcher:
    """Credentials/session are shared with whatever already called
    `robin_stocks.robinhood.login()` (e.g. broker/robinhood_broker.py's
    `connect()`) -- this class doesn't log in on its own, it just wraps the
    options-specific read endpoints on an already-authenticated session.
    """

    def __init__(self) -> None:
        import robin_stocks.robinhood as rh  # lazy import, matches the rest of the codebase's convention

        self._rh = rh

    def get_chain_expirations(self, symbol: str) -> List[date]:
        chain = self._rh.options.get_chains(symbol)
        raw_dates = (chain or {}).get("expiration_dates", [])
        return sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in raw_dates)

    def _pick_expiration(self, symbol: str, dte_min: int, dte_max: int) -> Optional[date]:
        if dte_min < MIN_ALLOWED_DTE:
            raise ValueError(
                f"dte_min ({dte_min}) is below MIN_ALLOWED_DTE ({MIN_ALLOWED_DTE}) -- "
                f"0DTE (same-day expiration) trades are never permitted"
            )
        today = date.today()
        candidates = [
            d for d in self.get_chain_expirations(symbol)
            if dte_min <= (d - today).days <= dte_max
        ]
        if not candidates:
            return None
        target = dte_min + (dte_max - dte_min) / 2
        return min(candidates, key=lambda d: abs((d - today).days - target))

    def get_atm_contract(
        self,
        symbol: str,
        option_type: str,
        current_price: float,
        dte_min: int = DEFAULT_TARGET_DTE_MIN,
        dte_max: int = DEFAULT_TARGET_DTE_MAX,
    ) -> Optional[OptionContract]:
        """Returns the tradable contract of `option_type` ("call"/"put")
        whose strike is closest to `current_price`, at the expiration
        closest to the midpoint of [dte_min, dte_max]. None if no
        expiration in that window has any tradable strikes. Never returns
        a 0DTE contract -- `_pick_expiration` enforces `dte_min >=
        MIN_ALLOWED_DTE` unconditionally, independent of whatever
        settings.yaml says.
        """
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

        expiration = self._pick_expiration(symbol, dte_min, dte_max)
        if expiration is None:
            return None

        instruments = self._rh.options.find_tradable_options(
            symbol, expirationDate=expiration.isoformat(), optionType=option_type,
        ) or []
        if not instruments:
            return None

        closest = min(instruments, key=lambda inst: abs(float(inst["strike_price"]) - current_price))
        market_data = self._rh.options.get_option_market_data_by_id(closest["id"])
        return self._to_contract(closest["id"], symbol, option_type, float(closest["strike_price"]), expiration, market_data)

    def get_quote_for_known_contract(
        self, symbol: str, option_type: str, expiration_date: date, strike_price: float
    ) -> Optional[OptionContract]:
        """For a contract you already hold (e.g. closing an existing
        position) -- fetches a fresh quote without re-running strike
        selection. Returns None if the contract can't be found (e.g. it's
        no longer listed, which shouldn't happen for something still open,
        but this is an unofficial API -- fail loud via None rather than a
        stale/guessed price on something about to be closed for real money.
        """
        instruments = self._rh.options.find_tradable_options(
            symbol, expirationDate=expiration_date.isoformat(), strikePrice=str(strike_price), optionType=option_type,
        ) or []
        if not instruments:
            return None
        market_data = self._rh.options.get_option_market_data_by_id(instruments[0]["id"])
        return self._to_contract(instruments[0]["id"], symbol, option_type, strike_price, expiration_date, market_data)

    def _to_contract(self, contract_id, symbol, option_type, strike_price, expiration, market_data) -> OptionContract:
        md = (market_data or [{}])[0] if isinstance(market_data, list) else (market_data or {})

        def _f(key: str) -> Optional[float]:
            value = md.get(key)
            return float(value) if value not in (None, "") else None

        return OptionContract(
            id=contract_id, symbol=symbol, option_type=option_type,
            strike_price=strike_price, expiration_date=expiration,
            bid=_f("bid_price"), ask=_f("ask_price"), last_price=_f("last_trade_price"),
        )
