"""Portfolio state and rebalance mechanics.

Shared between the backtester and (eventually) live trading — this is the
allocation layer's core job: translating a target allocation from the
Brain's signal into actual share deltas given current cash and positions.

This is also the part of the original backtester spec explicitly flagged as
needing to be exactly correct, so every formula here is implemented
literally and tested against hand-computed numbers, including the
leveraged/negative-cash (margin) case.

One correction from that spec, flagged prominently: `cash = delta *
price` (a literal overwrite) would discard the existing cash balance on
every single rebalance. The correct, implemented version is `cash = cash -
delta * price` — buying `delta` shares costs `delta * price`, selling
returns it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioState:
    cash: float
    shares: int

    def equity(self, price: float) -> float:
        """equity = cash + shares * current_price. Still correct under
        leverage/margin: if shares*price exceeds equity, cash is negative
        (margin debt), and this formula nets it out correctly — the position
        value minus the debt.
        """
        return self.cash + self.shares * price


@dataclass(frozen=True)
class RebalanceResult:
    new_state: PortfolioState
    delta_shares: int
    fill_price: float          # post-slippage price the trade executed at
    commission_paid: float
    traded: bool                # False if delta was 0 (no-op, no cost incurred)


def current_allocation_pct(state: PortfolioState, price: float) -> float:
    """Fraction of equity currently held in the position. Can exceed 1.0
    under leverage — that's expected, not an error.
    """
    equity = state.equity(price)
    if equity <= 0:
        raise ValueError(f"non-positive equity ({equity}) at price {price} — portfolio has blown up")
    return (state.shares * price) / equity


def should_rebalance(current_pct: float, target_pct: float, threshold: float = 0.10) -> bool:
    """Only rebalance when target differs from current by more than
    `threshold` — prevents churn from minor probability fluctuations.
    """
    return abs(target_pct - current_pct) > threshold


def _apply_slippage(price: float, delta_shares: int, slippage_pct: float) -> float:
    """Slippage always works against you: buying fills higher, selling
    fills lower. No slippage applied if there's no trade.
    """
    if delta_shares > 0:
        return price * (1 + slippage_pct)
    if delta_shares < 0:
        return price * (1 - slippage_pct)
    return price


def compute_rebalance(
    state: PortfolioState,
    current_price: float,
    next_open_price: float,
    target_allocation_pct: float,
    slippage_pct: float = 0.0,
    commission: float = 0.0,
) -> RebalanceResult:
    """Implements, in order:
        equity = cash + shares * current_price          (sizing reference price)
        target_shares = int(equity * target_allocation / current_price)
        delta = target_shares - current_shares
        fill_price = next_open_price, adjusted for slippage by delta's sign
        cash = cash - delta * fill_price - commission     (only if delta != 0)
        shares = target_shares

    `current_price` is the signal bar's price (used for sizing only).
    `next_open_price` is the fill bar's open (fill delay = 1 bar), before
    slippage — matching "signal bar N -> rebalance at bar N+1 open".
    """
    if current_price <= 0 or next_open_price <= 0:
        raise ValueError(f"invalid prices: current={current_price}, next_open={next_open_price}")

    equity = state.equity(current_price)
    target_shares = int(equity * target_allocation_pct / current_price)
    delta = target_shares - state.shares

    if delta == 0:
        return RebalanceResult(
            new_state=state, delta_shares=0, fill_price=next_open_price,
            commission_paid=0.0, traded=False,
        )

    fill_price = _apply_slippage(next_open_price, delta, slippage_pct)
    new_cash = state.cash - delta * fill_price - commission
    new_state = PortfolioState(cash=new_cash, shares=target_shares)

    return RebalanceResult(
        new_state=new_state, delta_shares=delta, fill_price=fill_price,
        commission_paid=commission, traded=True,
    )
