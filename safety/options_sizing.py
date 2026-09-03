"""Premium-based position sizing for long options (calls/puts).

Doesn't reuse safety/position_sizing.py's stop-distance sizing -- that
formula (`equity * max_risk_per_trade / |entry - stop|`) assumes a stock
position can gap through a stop and lose more than expected. A long
option's maximum possible loss is exactly the premium paid, full stop, by
construction (this is short options -- covered calls, cash-secured puts,
naked writes -- where risk gets complicated; none of that applies to a
long-only calls/puts strategy). So sizing here is direct: decide how much
premium you're willing to risk, then buy as many contracts as that buys.

Three independent caps -- the smallest one wins:
  - `max_premium_pct_per_trade` (settings.yaml: options.max_premium_pct_per_trade)
    -- how much to risk on this one position, as a fraction of total
    equity, scaled by the strategy's conviction
    (brain/options_strategy.OptionsDecision.conviction).
  - `max_total_premium_pct_of_equity` (settings.yaml: options.max_total_premium_pct_of_equity)
    -- a portfolio-wide ceiling on premium tied up across ALL open options
    positions at once, independent of the equity-side max_exposure. Options
    decay and can go to zero; capping total premium at risk bounds the
    worst case ("every option you hold expires worthless") to a number
    chosen up front.
  - `buying_power` -- the account's actual real, currently-spendable cash.
    The two caps above are both fractions of EQUITY (total account value),
    which can exceed what's actually liquid right now (equity tied up in
    existing stock/option positions isn't spendable on a new one). Without
    this cap, sizing could compute a contract count the account can't
    actually afford, and the only thing catching it would be
    safety/order_validation.py's buying-power check rejecting the ENTIRE
    order outright -- capping it here instead means the bot buys as many
    contracts as it can actually afford (bounded by the other two caps
    too), rather than an all-or-nothing reject on an order that was only
    slightly too big. order_validation's check stays in place as a second,
    independent confirmation right before the order is submitted (buying
    power can move between sizing and submission) -- belt and suspenders,
    not a duplicate of this cap's job.
"""
from __future__ import annotations

from dataclasses import dataclass

MIN_PREMIUM_USD = 50.0  # below this, per-contract lot sizes make the position not worth the commission/spread cost


@dataclass(frozen=True)
class OptionsSizeResult:
    contracts: int
    premium: float
    per_trade_budget: float
    portfolio_remaining_budget: float
    buying_power: float
    binding_cap: str        # "per_trade", "portfolio", or "buying_power" -- whichever budget actually limited the size
    below_minimum: bool


def compute_contract_count(
    equity: float,
    contract_price: float,
    conviction: float,
    current_total_premium_at_risk: float,
    max_premium_pct_per_trade: float,
    max_total_premium_pct_of_equity: float,
    buying_power: float,
    min_premium_usd: float = MIN_PREMIUM_USD,
) -> OptionsSizeResult:
    if equity <= 0:
        raise ValueError(f"non-positive equity ({equity})")
    if contract_price <= 0:
        raise ValueError(f"non-positive contract_price ({contract_price})")
    if not (0.0 <= conviction <= 1.0):
        raise ValueError(f"conviction must be in [0, 1], got {conviction}")
    if buying_power < 0:
        raise ValueError(f"negative buying_power ({buying_power})")

    per_trade_budget = equity * max_premium_pct_per_trade * conviction
    portfolio_ceiling = equity * max_total_premium_pct_of_equity
    portfolio_remaining_budget = max(0.0, portfolio_ceiling - current_total_premium_at_risk)

    caps = {"per_trade": per_trade_budget, "portfolio": portfolio_remaining_budget, "buying_power": buying_power}
    budget = min(caps.values())
    binding_cap = min(caps, key=caps.get)

    contract_cost = contract_price * 100  # one contract = 100 shares of exposure
    contracts = int(budget / contract_cost)
    premium = contracts * contract_cost

    below_minimum = 0 < premium < min_premium_usd
    if below_minimum:
        contracts, premium = 0, 0.0

    return OptionsSizeResult(
        contracts=contracts, premium=premium, per_trade_budget=per_trade_budget,
        portfolio_remaining_budget=portfolio_remaining_budget, buying_power=buying_power,
        binding_cap=binding_cap, below_minimum=below_minimum,
    )
