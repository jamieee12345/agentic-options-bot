"""Full walk-forward backtest, using synthetic data.

Same honesty note as the pipeline demo: `hmmlearn` isn't available in every
environment this runs in, so `FakeGaussianHMM` below stands in for a fitted
model. Unlike the earlier demo's simpler stand-in, this one:
  - Is fit ONLY on each window's in-sample (IS) data — never sees OOS data
    at fit time, same discipline a real retrain would need.
  - Estimates real Gaussian-HMM-shaped parameters (startprob_, transmat_,
    means_, covars_) via empirical moments, so it can drive the ACTUAL
    `forward_filter()` from walk_forward.py bar-by-bar through OOS — not a
    simplified confidence heuristic.

Everything else in this run — rank_states_by_volatility, volatility_tier,
StrategyOrchestrator, compute_rebalance, should_rebalance, and every metric
in metrics.py — is the real production code, exercised end to end.

Run: PYTHONPATH=. python3 backtest/run_backtest_demo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.features import compute_feature_set
from brain.signal_generator import StrategyOrchestrator, RegimeState
from backtest.walk_forward import generate_walk_forward_windows, forward_filter_from_log_emission
from allocation.portfolio import PortfolioState, compute_rebalance, current_allocation_pct, should_rebalance
from backtest import metrics as m

FEATURE_COLUMNS = ["realized_vol_20"]
N_STATES = 3
SLIPPAGE_PCT = 0.0005       # settings.yaml: backtesting.slippage_pct
REBALANCE_THRESHOLD = 0.10  # settings.yaml: strategy.rebalance_threshold
INITIAL_CAPITAL = 2000.0    # settings.yaml: backtesting.initial_capital_usd
FLICKER_WINDOW = 5


class FakeGaussianHMM:
    """Stand-in for a fitted hmmlearn.GaussianHMM (diag covariance, 1
    feature). Parameters estimated by empirical moments on IS data alone —
    a reasonable approximation of what EM converges to for well-separated
    clusters, not a substitute for actually running hmmlearn.
    """

    def __init__(self, vol_series: pd.Series, n_states: int = N_STATES):
        q_edges = vol_series.quantile(np.linspace(0, 1, n_states + 1)).to_numpy().copy()
        q_edges[0], q_edges[-1] = -np.inf, np.inf
        state_seq = pd.cut(vol_series, bins=q_edges, labels=False, include_lowest=True).to_numpy()

        self.n_components = n_states
        self.means_ = np.zeros((n_states, 1))
        variances = np.zeros((n_states, 1))
        counts = np.zeros(n_states)
        for k in range(n_states):
            vals = vol_series.to_numpy()[state_seq == k]
            counts[k] = len(vals)
            self.means_[k, 0] = vals.mean()
            variances[k, 0] = max(vals.var(ddof=0), 1e-8)  # floor to avoid a degenerate zero-variance state
        self.covars_ = variances

        self.startprob_ = (counts + 1e-3) / (counts + 1e-3).sum()

        transmat = np.zeros((n_states, n_states))
        for t in range(len(state_seq) - 1):
            transmat[state_seq[t], state_seq[t + 1]] += 1
        transmat += 1e-3  # small smoothing floor -- avoids exact-zero transition probabilities,
        # which are numerically noisy (log(0)) and unrealistic for an EM-fit model anyway
        transmat = transmat / transmat.sum(axis=1, keepdims=True)
        self.transmat_ = transmat

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Diagonal-covariance Gaussian log-likelihood, matching what
        hmmlearn computes internally for covariance_type='diag'.
        """
        n_states = self.n_components
        T = X.shape[0]
        log_prob = np.zeros((T, n_states))
        for k in range(n_states):
            var = self.covars_[k, 0]
            mean = self.means_[k, 0]
            x = X[:, 0]
            log_prob[:, k] = -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)
        return log_prob


def generate_synthetic_bars(seed: int = 11) -> pd.DataFrame:
    np.random.seed(seed)
    phases = [
        (200, 0.0006, 0.007),   # calm bull
        (60, -0.0050, 0.035),   # crash
        (200, 0.0006, 0.008),   # calm recovery
        (150, 0.0001, 0.018),   # choppy/moderate
        (50, -0.0040, 0.030),   # second turbulent patch
        (200, 0.0007, 0.007),   # calm bull again
    ]
    returns = np.concatenate([np.random.normal(mu, sigma, n) for n, mu, sigma in phases])
    n = len(returns)
    dates = pd.bdate_range("2019-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + np.random.normal(0, 0.001, n))  # small independent open-vs-prior-close noise
    high = np.maximum(close, open_) * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low = np.minimum(close, open_) * (1 - np.abs(np.random.normal(0, 0.003, n)))
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def run_backtest() -> dict:
    bars = generate_synthetic_bars()
    feats = compute_feature_set(bars)

    windows = generate_walk_forward_windows(len(bars), is_size=252, oos_size=126, step_size=126)
    print(f"{len(windows)} walk-forward windows over {len(bars)} bars\n")

    orchestrator = StrategyOrchestrator()
    portfolio = PortfolioState(cash=INITIAL_CAPITAL, shares=0)

    equity_records = []   # (date, equity)
    trade_records = []    # dict per rebalance event
    recent_states: list[int] = []

    for window in windows:
        is_vol = feats["realized_vol_20"].iloc[window.is_start: window.is_end].dropna()
        model = FakeGaussianHMM(is_vol, n_states=N_STATES)
        orchestrator.rebuild_mapping(model, FEATURE_COLUMNS, n_states=N_STATES)

        oos_vol = feats["realized_vol_20"].iloc[window.oos_start: window.oos_end]
        oos_dates = feats.index[window.oos_start: window.oos_end]
        log_emission_oos = model._compute_log_likelihood(oos_vol.to_numpy().reshape(-1, 1))

        for i, global_idx in enumerate(range(window.oos_start, window.oos_end)):
            date = feats.index[global_idx]
            current_price = float(feats["close"].iloc[global_idx])

            # Forward-filter using only OOS observations up to and including this bar --
            # the real production function, not a simplified stand-in.
            filtered = forward_filter_from_log_emission(
                model.startprob_, model.transmat_, log_emission_oos[: i + 1]
            )
            probs_t = filtered[-1]
            state = int(np.argmax(probs_t))
            confidence = float(probs_t[state])

            recent_states.append(state)
            recent_states[:] = recent_states[-FLICKER_WINDOW:]

            regime_state = RegimeState(
                state=state, probability=confidence, label=f"state_{state}",
                recent_states=list(recent_states), timestamp=date,
            )
            bars_so_far = feats.iloc[: global_idx + 1]
            signals = orchestrator.generate_signals({"SYM": bars_so_far}, {"SYM": regime_state})
            signal = signals.get("SYM")

            if signal is not None:
                target_pct = signal.position_size_pct
                current_pct = current_allocation_pct(portfolio, current_price)

                if should_rebalance(current_pct, target_pct, threshold=REBALANCE_THRESHOLD):
                    # Fill delay: signal at bar `global_idx`, execute at bar `global_idx + 1`'s open.
                    if global_idx + 1 < len(bars):
                        next_open = float(bars["open"].iloc[global_idx + 1])
                        equity_before = portfolio.equity(current_price)
                        result = compute_rebalance(
                            portfolio, current_price=current_price, next_open_price=next_open,
                            target_allocation_pct=target_pct, slippage_pct=SLIPPAGE_PCT, commission=0.0,
                        )
                        if result.traded:
                            trade_records.append({
                                "date": date, "delta_shares": result.delta_shares,
                                "fill_price": result.fill_price, "equity_before": equity_before,
                                "target_pct": target_pct, "strategy": signal.regime.strategy_name,
                            })
                        portfolio = result.new_state

            equity_records.append((date, portfolio.equity(current_price)))

    equity_curve = pd.Series(
        [e for _, e in equity_records], index=pd.DatetimeIndex([d for d, _ in equity_records]), name="equity"
    )

    # Trade P&L: equity change attributable to the holding period between
    # consecutive rebalances (this system continuously rebalances rather
    # than opening/closing discrete round trips, so a "trade" here is
    # defined as one such holding stint -- flagged as an interpretation
    # choice, not something the spec defined precisely).
    trade_pnls = []
    holding_periods = []
    for j in range(len(trade_records) - 1):
        entry_date = trade_records[j]["date"]
        exit_date = trade_records[j + 1]["date"]
        entry_equity = equity_curve.loc[entry_date]
        exit_equity = equity_curve.loc[exit_date]
        trade_pnls.append(exit_equity - entry_equity)
        holding_periods.append((exit_date - entry_date).days)
    if trade_records:
        last_entry = trade_records[-1]["date"]
        trade_pnls.append(equity_curve.iloc[-1] - equity_curve.loc[last_entry])
        holding_periods.append((equity_curve.index[-1] - last_entry).days)

    returns = m.periodic_returns(equity_curve)

    dd = m.max_drawdown(equity_curve)
    results = {
        "equity_curve": equity_curve,
        "trades": trade_records,
        "n_trades": len(trade_records),
        "avg_holding_period_days": float(np.mean(holding_periods)) if holding_periods else 0.0,
        "total_return": m.total_return(equity_curve),
        "cagr": m.cagr(equity_curve),
        "sharpe": m.sharpe_ratio(returns),
        "sortino": m.sortino_ratio(returns),
        "max_drawdown": dd,
        "longest_underwater_streak": m.longest_underwater_streak(equity_curve),
        "calmar": m.calmar_ratio(m.cagr(equity_curve), dd.max_drawdown_pct),
        "win_rate": m.win_rate(trade_pnls),
        "avg_win_loss": m.avg_win_loss(trade_pnls),
        "profit_factor": m.profit_factor(trade_pnls),
        "max_consecutive_losses": m.max_consecutive_losses(trade_pnls),
        "worst_day": m.worst_period_return(returns, "D"),
        "worst_week": m.worst_period_return(returns, "W"),
        "worst_month": m.worst_period_return(returns, "ME"),
    }
    return results


def print_report(r: dict) -> None:
    dd = r["max_drawdown"]
    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"{'Total Return':<28}{r['total_return']:>15.2%}")
    print(f"{'CAGR':<28}{r['cagr']:>15.2%}")
    print(f"{'Sharpe Ratio':<28}{r['sharpe']:>15.2f}")
    print(f"{'Sortino Ratio':<28}{r['sortino']:>15.2f}")
    print(f"{'Calmar Ratio':<28}{r['calmar']:>15.2f}")
    print(f"{'Max Drawdown':<28}{dd.max_drawdown_pct:>15.2%}")
    print(f"{'  Duration (peak->trough)':<28}{dd.duration_bars:>15} bars")
    print(f"{'  Underwater (peak->recover)':<28}{dd.underwater_bars:>15} bars")
    print(f"{'Longest Underwater Streak':<28}{r['longest_underwater_streak']:>15} bars")
    print("-" * 60)
    print(f"{'Total Trades':<28}{r['n_trades']:>15}")
    print(f"{'Avg Holding Period':<28}{r['avg_holding_period_days']:>15.1f} days")
    print(f"{'Win Rate':<28}{r['win_rate']:>15.1%}")
    avg_w, avg_l = r["avg_win_loss"]
    print(f"{'Avg Win / Avg Loss':<28}{'$' + format(avg_w, '.2f'):>8} / {'$' + format(avg_l, '.2f'):<8}")
    print(f"{'Profit Factor':<28}{r['profit_factor']:>15.2f}")
    print(f"{'Max Consecutive Losses':<28}{r['max_consecutive_losses']:>15}")
    print("-" * 60)
    print(f"{'Worst Day':<28}{r['worst_day']:>15.2%}")
    print(f"{'Worst Week':<28}{r['worst_week']:>15.2%}")
    print(f"{'Worst Month':<28}{r['worst_month']:>15.2%}")
    print("=" * 60)


if __name__ == "__main__":
    results = run_backtest()
    print_report(results)
