"""Real Robinhood broker implementation (`orchestration.main_loop.Broker`),
hard-restricted to a single account number.

This is the piece the README's "Next up" list called "wire real
implementations into the orchestration layer" -- `broker/__init__.py` was
empty before this. Per the account restriction requested when this was
built: this class will only ever act against ONE Robinhood account, read
from the ROBINHOOD_ACCOUNT_NUMBER environment variable, never the account
your Robinhood login defaults to. Add to your .env (never commit it):

    ROBINHOOD_USERNAME=...
    ROBINHOOD_PASSWORD=...
    ROBINHOOD_MFA_SECRET=...          # only if using app-based 2FA
    ROBINHOOD_ACCOUNT_NUMBER=...      # the ONE account this bot may touch

How the restriction is enforced, and its limit:
  - The account number is threaded through every account-scoped robin_stocks
    call via that library's `account_number=` keyword argument.
  - `verify_account()` re-fetches the account profile for that exact number
    and confirms it comes back before allowing anything else to run --
    every other method refuses to execute if verification hasn't happened.
  - The real limit, stated plainly: robin_stocks is an unofficial,
    reverse-engineered client, and its `account_number` parameter support
    has changed across versions and isn't uniform across every function.
    This was written against the documented pattern as of when it was
    written, but **could not be tested against a live account or a live
    robin_stocks install in this sandbox (no network, no Python runtime
    available at all here)**. Before running this against real money:
    confirm your installed robin_stocks version actually honors
    `account_number` on `orders.order_buy_market` / `order_sell_market` /
    `get_all_open_stock_orders` / `load_account_profile` -- if any of those
    silently ignore it and fall back to your default account, this
    restriction is broken. A cheap way to check: call `place_order` for one
    share of something you'd be fine holding, then verify in the Robinhood
    app which account it landed in before ever running the live loop
    unattended.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from orchestration.main_loop import BrokerError

logger = logging.getLogger(__name__)

VALID_SIDES = {"buy", "sell"}
VALID_ORDER_TYPES = {"market", "limit"}


class AccountMismatchError(BrokerError):
    """Raised when the account robin_stocks actually returns doesn't match
    the configured ROBINHOOD_ACCOUNT_NUMBER -- fail closed, never guess."""


class RobinhoodBroker:
    """Implements the `Broker` protocol from orchestration/main_loop.py."""

    def __init__(self, account_number: Optional[str] = None) -> None:
        self.account_number = account_number or os.environ.get("ROBINHOOD_ACCOUNT_NUMBER")
        if not self.account_number:
            raise BrokerError(
                "ROBINHOOD_ACCOUNT_NUMBER is not set. Refusing to start without an "
                "explicit, single target account -- this bot must never fall back to "
                "whatever account your Robinhood login happens to default to."
            )
        self._rh = None
        self._verified = False

    # -- connection --------------------------------------------------------

    def connect(self) -> None:
        import robin_stocks.robinhood as rh  # lazy import, matches data/fetchers.py's convention
        import pyotp

        username = os.environ.get("ROBINHOOD_USERNAME")
        password = os.environ.get("ROBINHOOD_PASSWORD")
        mfa_secret = os.environ.get("ROBINHOOD_MFA_SECRET")
        if not username or not password:
            raise BrokerError("Set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD before connecting.")

        mfa_code = pyotp.TOTP(mfa_secret).now() if mfa_secret else None
        rh.login(username, password, mfa_code=mfa_code, store_session=True)
        self._rh = rh
        self._verified = False  # a fresh login always needs re-verification

    def verify_account(self) -> bool:
        if self._rh is None:
            raise BrokerError("connect() must succeed before verify_account().")

        profile = self._rh.profiles.load_account_profile(account_number=self.account_number)
        returned_number = (profile or {}).get("account_number")
        if not profile or returned_number != self.account_number:
            raise AccountMismatchError(
                f"Configured account {self.account_number!r} did not come back from "
                f"load_account_profile (got {returned_number!r}). Refusing to proceed -- "
                f"this almost certainly means robin_stocks' account_number targeting "
                f"isn't working the way this code assumes for your installed version."
            )
        self._verified = True
        return True

    def _require_verified(self) -> None:
        if not self._verified:
            raise BrokerError("verify_account() must succeed before any account-scoped call.")

    @property
    def rh_session(self):
        """The authenticated `robin_stocks.robinhood` module, for callers
        that need a second robin_stocks-based client (e.g.
        data.fetchers.RobinhoodIntradayHistoricalFetcher) without triggering
        a second, separate login. Only the login itself is shared -- this
        does NOT bypass this class's own account_number restriction, since
        market-data reads (unlike orders/positions) aren't account-scoped
        on Robinhood to begin with.
        """
        if self._rh is None:
            raise BrokerError("connect() must succeed before rh_session is available.")
        return self._rh

    # -- account state -------------------------------------------------------

    def get_portfolio_equity(self) -> float:
        self._require_verified()
        profile = self._rh.profiles.load_portfolio_profile(account_number=self.account_number)
        equity = (profile or {}).get("equity")
        if equity is None:
            raise BrokerError(f"No 'equity' field in portfolio profile: {profile}")
        return float(equity)

    def get_buying_power(self) -> float:
        self._require_verified()
        profile = self._rh.profiles.load_account_profile(account_number=self.account_number)
        bp = (profile or {}).get("buying_power")
        if bp is None:
            raise BrokerError(f"No 'buying_power' field in account profile: {profile}")
        return float(bp)

    def get_positions(self) -> Dict[str, float]:
        self._require_verified()
        holdings = self._rh.account.get_open_stock_positions(account_number=self.account_number) or []
        positions: Dict[str, float] = {}
        for h in holdings:
            quantity = float(h.get("quantity", 0) or 0)
            if quantity == 0:
                continue
            symbol = self._rh.stocks.get_symbol_by_url(h["instrument"])
            positions[symbol] = quantity
        return positions

    def get_open_orders(self) -> List[dict]:
        self._require_verified()
        return self._rh.orders.get_all_open_stock_orders(account_number=self.account_number) or []

    # -- order placement -----------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> str:
        self._require_verified()
        if side not in VALID_SIDES:
            raise ValueError(f"side must be one of {VALID_SIDES}, got {side!r}")
        if order_type not in VALID_ORDER_TYPES:
            raise ValueError(f"order_type must be one of {VALID_ORDER_TYPES}, got {order_type!r}")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if order_type == "limit" and limit_price is None:
            raise ValueError("limit_price is required for a limit order")

        order_fn = {
            ("buy", "market"): lambda: self._rh.orders.order_buy_market(symbol, quantity, account_number=self.account_number),
            ("sell", "market"): lambda: self._rh.orders.order_sell_market(symbol, quantity, account_number=self.account_number),
            ("buy", "limit"): lambda: self._rh.orders.order_buy_limit(symbol, quantity, limit_price, account_number=self.account_number),
            ("sell", "limit"): lambda: self._rh.orders.order_sell_limit(symbol, quantity, limit_price, account_number=self.account_number),
        }[(side, order_type)]

        result = order_fn()
        order_id = (result or {}).get("id")
        if not order_id:
            raise BrokerError(f"Order for {symbol} was not accepted: {result}")

        logger.info(
            "Placed %s %s order: %d shares of %s (account %s, order_id=%s)",
            order_type, side, quantity, symbol, self.account_number, order_id,
        )
        return order_id

    def cancel_order(self, order_id: str) -> None:
        self._require_verified()
        result = self._rh.orders.cancel_stock_order(order_id)
        logger.info("Cancel requested for order_id=%s: %s", order_id, result)

    # -- options -------------------------------------------------------------
    # Same account restriction as everything above (account_number threaded
    # through every call). Extra caveat specific to options: robin_stocks'
    # options order functions have a less stable, less-documented signature
    # history than its equity ones -- `order_buy_option_limit` and
    # `order_sell_option_limit` below are written against the commonly
    # documented (positionEffect, creditOrDebit, price, symbol, quantity,
    # expirationDate, strike, optionType) shape, but **this could not be
    # verified against a live robin_stocks install in this sandbox**. Check
    # your installed version's actual signature (`help(rh.orders.order_buy_option_limit)`)
    # before this ever places a real order.

    def get_option_positions(self) -> List[dict]:
        self._require_verified()
        return self._rh.options.get_open_option_positions(account_number=self.account_number) or []

    def place_option_order(
        self,
        symbol: str,
        option_type: str,       # "call" or "put"
        expiration_date: str,   # "YYYY-MM-DD"
        strike_price: float,
        side: str,               # "buy" or "sell"
        position_effect: str,    # "open" or "close"
        quantity: int,
        limit_price: float,
    ) -> str:
        self._require_verified()
        if side not in VALID_SIDES:
            raise ValueError(f"side must be one of {VALID_SIDES}, got {side!r}")
        if position_effect not in ("open", "close"):
            raise ValueError(f"position_effect must be 'open' or 'close', got {position_effect!r}")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if limit_price <= 0:
            raise ValueError(f"limit_price must be positive, got {limit_price}")

        credit_or_debit = "debit" if side == "buy" else "credit"
        order_fn = self._rh.orders.order_buy_option_limit if side == "buy" else self._rh.orders.order_sell_option_limit

        result = order_fn(
            positionEffect=position_effect, creditOrDebit=credit_or_debit, price=limit_price,
            symbol=symbol, quantity=quantity, expirationDate=expiration_date,
            strike=strike_price, optionType=option_type, account_number=self.account_number,
        )
        order_id = (result or {}).get("id")
        if not order_id:
            raise BrokerError(f"Option order for {symbol} {expiration_date} {strike_price}{option_type[0]} was not accepted: {result}")

        logger.info(
            "Placed %s to %s: %d contract(s) of %s %s %s $%.2f @ limit $%.2f (account %s, order_id=%s)",
            side, position_effect, quantity, symbol, expiration_date, option_type, strike_price,
            limit_price, self.account_number, order_id,
        )
        return order_id

    def cancel_option_order(self, order_id: str) -> None:
        self._require_verified()
        result = self._rh.orders.cancel_option_order(order_id)
        logger.info("Option order cancel requested for order_id=%s: %s", order_id, result)
