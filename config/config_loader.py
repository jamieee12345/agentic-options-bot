"""Loads and validates settings.yaml into typed, immutable config objects.

Deliberately stdlib-only (dataclasses + yaml) so this module has zero extra
dependencies and can be imported by every other layer without dragging in
a dependency tree.

Validation encodes the sanity checks a human reviewer would do by eye:
loss-limit ordering (reduce < halt < peak drawdown), and a warning if the
position-count/size caps mean max_exposure can never actually bind.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when settings.yaml fails validation."""


@dataclass(frozen=True)
class UniverseScreen:
    min_market_cap_usd: float
    min_avg_daily_dollar_volume_usd: float
    volume_lookback_days: int


@dataclass(frozen=True)
class BrokerConfig:
    provider: str
    live_trading_enabled: bool
    core_watchlist: List[str]
    universe_screen: UniverseScreen
    timeframes: List[str]


@dataclass(frozen=True)
class AllocationConfig:
    low_volatility: float
    mid_volatility_trending: float
    mid_volatility_no_trend: float
    high_volatility: float


@dataclass(frozen=True)
class LeverageConfig:
    low_volatility: float


@dataclass(frozen=True)
class StrategyConfig:
    allocation: AllocationConfig
    leverage: LeverageConfig
    rebalance_threshold: float
    uncertainty_size_multiplier: float


@dataclass(frozen=True)
class DrawdownControls:
    daily_loss_reduce: float
    daily_loss_halt: float
    weekly_loss_reduce: float
    weekly_loss_halt: float
    max_loss_from_peak: float

    def __post_init__(self) -> None:
        if not (0 < self.daily_loss_reduce < self.daily_loss_halt):
            raise ConfigError(
                f"daily_loss_reduce ({self.daily_loss_reduce}) must be > 0 and "
                f"< daily_loss_halt ({self.daily_loss_halt})"
            )
        if not (0 < self.weekly_loss_reduce < self.weekly_loss_halt):
            raise ConfigError(
                f"weekly_loss_reduce ({self.weekly_loss_reduce}) must be > 0 and "
                f"< weekly_loss_halt ({self.weekly_loss_halt})"
            )
        if self.weekly_loss_halt >= self.max_loss_from_peak:
            raise ConfigError(
                f"weekly_loss_halt ({self.weekly_loss_halt}) should be < "
                f"max_loss_from_peak ({self.max_loss_from_peak}) — the peak "
                f"drawdown breaker is meant to be the outermost backstop"
            )


@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade: float
    max_exposure: float
    max_leverage: float
    max_single_position: float
    max_sector_exposure: float
    max_concurrent_positions: int
    max_daily_trades: int
    max_bid_ask_spread_pct: float
    correlation_lookback_days: int
    correlation_reduce_threshold: float
    correlation_reject_threshold: float
    drawdown_controls: DrawdownControls

    def __post_init__(self) -> None:
        if not (0 < self.max_risk_per_trade <= 0.05):
            raise ConfigError(
                f"max_risk_per_trade ({self.max_risk_per_trade}) looks out of "
                f"range — expected a small fraction like 0.01 (1%)"
            )
        if not (0 < self.max_exposure <= 1.0):
            raise ConfigError(f"max_exposure ({self.max_exposure}) must be in (0, 1]")
        if not (0 < self.max_single_position <= self.max_exposure):
            raise ConfigError(
                f"max_single_position ({self.max_single_position}) must be > 0 "
                f"and <= max_exposure ({self.max_exposure})"
            )
        if not (0 < self.max_bid_ask_spread_pct < 1.0):
            raise ConfigError(f"max_bid_ask_spread_pct ({self.max_bid_ask_spread_pct}) must be in (0, 1)")
        if self.correlation_lookback_days <= 0:
            raise ConfigError(f"correlation_lookback_days ({self.correlation_lookback_days}) must be positive")
        if not (0 < self.correlation_reduce_threshold < self.correlation_reject_threshold <= 1.0):
            raise ConfigError(
                f"correlation_reduce_threshold ({self.correlation_reduce_threshold}) must be > 0 and "
                f"< correlation_reject_threshold ({self.correlation_reject_threshold}) <= 1.0"
            )

        implied_max = self.max_single_position * self.max_concurrent_positions
        if implied_max < self.max_exposure:
            warnings.warn(
                f"max_single_position * max_concurrent_positions = {implied_max:.2f}, "
                f"which is below max_exposure ({self.max_exposure}). max_exposure can "
                f"never actually bind — it's dead weight as currently configured.",
                stacklevel=2,
            )


MIN_ALLOWED_DTE = 1  # HARD FLOOR: 0DTE (same-day expiration) trades are never permitted, by explicit requirement


@dataclass(frozen=True)
class OptionsConfig:
    enabled: bool
    target_dte_min: int
    target_dte_max: int
    close_before_expiration_days: int
    max_premium_pct_per_trade: float
    max_total_premium_pct_of_equity: float
    fvg_lookback_period: int
    fvg_body_multiplier: float
    fvg_volume_multiplier: float
    sma_period: int
    min_confluence_score: float
    stop_loss_pct: float
    take_profit_pct: float
    # "Close if going nowhere": checked after stop_loss_pct/take_profit_pct
    # (only reachable if neither of those already fired), so this only
    # affects positions sitting in the boring middle. Once a position has
    # been held for stagnant_exit_hold_fraction of its OWN dte_at_entry
    # (e.g. 0.5 = halfway to expiration) without reaching
    # stagnant_exit_min_pnl_pct, it's force-closed rather than left to ride
    # the back half of its life -- where theta decay accelerates fastest --
    # toward an expiration close. Added after a 3-year/20-symbol diagnostic
    # backtest showed 48% of trades were closing on expiration at a -12.8%
    # average loss, the single largest loss bucket by trade count, while
    # take-profit trades (only 18% of trades) averaged +147%: too many
    # positions were being allowed to decay all the way to expiration
    # instead of being cut once they'd already shown they weren't working.
    stagnant_exit_hold_fraction: float
    stagnant_exit_min_pnl_pct: float

    def __post_init__(self) -> None:
        if self.target_dte_min < MIN_ALLOWED_DTE:
            raise ConfigError(
                f"options.target_dte_min ({self.target_dte_min}) must be >= {MIN_ALLOWED_DTE} -- "
                f"0DTE (same-day expiration) trades are never permitted"
            )
        if self.target_dte_min >= self.target_dte_max:
            raise ConfigError(
                f"options.target_dte_min ({self.target_dte_min}) must be < target_dte_max ({self.target_dte_max})"
            )
        if self.close_before_expiration_days <= 0:
            raise ConfigError(f"options.close_before_expiration_days ({self.close_before_expiration_days}) must be positive")
        if self.close_before_expiration_days >= self.target_dte_min:
            raise ConfigError(
                f"options.close_before_expiration_days ({self.close_before_expiration_days}) must be < "
                f"target_dte_min ({self.target_dte_min}) -- otherwise a freshly opened position could already be inside its own force-close window"
            )
        if not (0 < self.max_premium_pct_per_trade <= 1.0):
            raise ConfigError(f"options.max_premium_pct_per_trade ({self.max_premium_pct_per_trade}) must be in (0, 1]")
        if not (self.max_premium_pct_per_trade <= self.max_total_premium_pct_of_equity <= 1.0):
            raise ConfigError(
                f"options.max_total_premium_pct_of_equity ({self.max_total_premium_pct_of_equity}) must be >= "
                f"max_premium_pct_per_trade ({self.max_premium_pct_per_trade}) and <= 1.0"
            )
        if self.fvg_lookback_period <= 0:
            raise ConfigError(f"options.fvg_lookback_period ({self.fvg_lookback_period}) must be positive")
        if self.fvg_body_multiplier <= 0:
            raise ConfigError(f"options.fvg_body_multiplier ({self.fvg_body_multiplier}) must be positive")
        if self.fvg_volume_multiplier <= 0:
            raise ConfigError(f"options.fvg_volume_multiplier ({self.fvg_volume_multiplier}) must be positive")
        if self.sma_period <= 0:
            raise ConfigError(f"options.sma_period ({self.sma_period}) must be positive")
        if not (0 < self.min_confluence_score <= 1.0):
            raise ConfigError(f"options.min_confluence_score ({self.min_confluence_score}) must be in (0, 1]")
        if not (0 < self.stop_loss_pct <= 1.0):
            raise ConfigError(f"options.stop_loss_pct ({self.stop_loss_pct}) must be in (0, 1]")
        if self.take_profit_pct <= 0:
            raise ConfigError(f"options.take_profit_pct ({self.take_profit_pct}) must be positive")
        if not (0 < self.stagnant_exit_hold_fraction < 1):
            raise ConfigError(f"options.stagnant_exit_hold_fraction ({self.stagnant_exit_hold_fraction}) must be in (0, 1)")
        if self.stagnant_exit_min_pnl_pct <= -self.stop_loss_pct:
            warnings.warn(
                f"options.stagnant_exit_min_pnl_pct ({self.stagnant_exit_min_pnl_pct}) is <= -stop_loss_pct "
                f"(-{self.stop_loss_pct}), so stop_loss_pct will always fire first -- the stagnant-exit check "
                f"can never actually trigger as currently configured.",
                stacklevel=2,
            )
        if self.stagnant_exit_min_pnl_pct >= self.take_profit_pct:
            warnings.warn(
                f"options.stagnant_exit_min_pnl_pct ({self.stagnant_exit_min_pnl_pct}) is >= take_profit_pct "
                f"({self.take_profit_pct}), so take_profit_pct will always fire first -- the stagnant-exit check "
                f"can never actually trigger as currently configured.",
                stacklevel=2,
            )


@dataclass(frozen=True)
class BacktestingConfig:
    slippage_pct: float
    initial_capital_usd: float


@dataclass(frozen=True)
class MonitoringConfig:
    dashboard_refresh_seconds: int
    alert_rate_limit_minutes: int


@dataclass(frozen=True)
class Settings:
    broker: BrokerConfig
    strategy: StrategyConfig
    risk: RiskConfig
    options: OptionsConfig
    backtesting: BacktestingConfig
    monitoring: MonitoringConfig


def _build(cls, raw: dict):
    """Recursively construct a nested dataclass from a plain dict, so the
    yaml's nested structure maps 1:1 onto the nested dataclasses above.
    """
    kwargs = {}
    for f in field_types(cls):
        name, ftype = f
        value = raw[name]
        if hasattr(ftype, "__dataclass_fields__"):
            value = _build(ftype, value)
        kwargs[name] = value
    return cls(**kwargs)


def field_types(cls):
    # `from __future__ import annotations` makes every dataclass field's
    # `.type` a plain string (e.g. "AllocationConfig") instead of the actual
    # class object, which breaks the `hasattr(ftype, "__dataclass_fields__")`
    # recursion check below. get_type_hints() resolves those strings back
    # into real types.
    hints = get_type_hints(cls)
    return [(f.name, hints[f.name]) for f in cls.__dataclass_fields__.values()]


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """Load, parse, and validate settings.yaml. Raises ConfigError (or a
    warning for non-fatal issues) if anything is inconsistent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No settings file at {path.resolve()}")

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    try:
        return _build(Settings, raw)
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"settings.yaml is missing or misshapes a field: {exc}") from exc


if __name__ == "__main__":
    import sys

    settings_path = sys.argv[1] if len(sys.argv) > 1 else "config/settings.yaml"
    settings = load_settings(settings_path)
    print(f"Loaded settings from {settings_path}")
    print(f"  Broker: {settings.broker.provider}, {len(settings.broker.core_watchlist)} core tickers")
    print(f"  Max exposure: {settings.risk.max_exposure}, max concurrent: {settings.risk.max_concurrent_positions}")
    print(f"  Initial capital: ${settings.backtesting.initial_capital_usd:,.0f}")
