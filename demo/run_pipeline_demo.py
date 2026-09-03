"""End-to-end pipeline demo, using synthetic data.

Two things this is NOT:
  - Not a test of the HMM itself. `hmmlearn` isn't available in every
    environment this runs in, so `SimpleQuantileRegimeAssigner` below stands
    in for it: a plain tercile split on realized_vol_20. It produces the
    same *shape* of output a fitted GaussianHMM would (a state index per bar
    plus a `means_`-like array), which is enough to exercise everything
    downstream — ranking, tiering, the three strategies, the uncertainty
    gate, the orchestrator — for real. It is not a substitute for actually
    validating regime_detector.py against hmmlearn.
  - Not a backtest. No P&L, no slippage, no portfolio state. It just shows
    what signal the pipeline would produce at a handful of points in time.

Run: python3 demo/run_pipeline_demo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.features import compute_feature_set
from brain.signal_generator import StrategyOrchestrator, RegimeState

FEATURE_COLUMNS = ["realized_vol_20"]  # kept to 1 feature so the demo is easy to follow by eye
N_STATES = 3


class SimpleQuantileRegimeAssigner:
    """Stand-in for a fitted GaussianHMM: splits realized_vol_20 into
    terciles and exposes a `.means_` array in the same shape hmmlearn would
    produce, so rank_states_by_volatility() works unmodified.
    """

    def __init__(self, vol_series: pd.Series, state_ids=(2, 0, 1)):
        # Deliberately non-obvious state indices (calm=2, mid=0, turbulent=1)
        # so this demo doesn't accidentally rely on state 0 meaning "calm" —
        # the real pipeline has to sort that out from means_, same as it
        # would have to for a real HMM's arbitrary state ordering.
        self.calm_id, self.mid_id, self.turbulent_id = state_ids
        q1, q2 = vol_series.quantile([1 / 3, 2 / 3])
        self._q1, self._q2 = q1, q2

        self.state_sequence = vol_series.apply(self._assign)
        valid = self.state_sequence.notna()
        grouped = vol_series[valid].groupby(self.state_sequence[valid]).mean()

        self.means_ = np.zeros((N_STATES, 1))
        for state_id in state_ids:
            self.means_[state_id, 0] = grouped.get(state_id, np.nan)

    def _assign(self, v: float):
        if pd.isna(v):
            return pd.NA
        if v <= self._q1:
            return self.calm_id
        if v <= self._q2:
            return self.mid_id
        return self.turbulent_id


def confidence_for(idx: int, state_sequence: pd.Series) -> float:
    """Simulated confidence: high in the interior of a regime, dips near a
    transition (the state just changed within the last 3 bars) — mimicking
    how a real HMM is genuinely less certain right at a regime boundary.
    """
    window = state_sequence.iloc[max(0, idx - 3): idx + 1]
    if window.nunique(dropna=True) > 1:
        return 0.45  # below the 0.55 threshold -> should trigger uncertainty mode
    return 0.85


def main() -> None:
    np.random.seed(7)

    n_calm_1, n_turbulent, n_calm_2 = 150, 80, 150
    n = n_calm_1 + n_turbulent + n_calm_2
    dates = pd.bdate_range("2023-01-01", periods=n)

    returns = np.concatenate([
        np.random.normal(0.0006, 0.006, n_calm_1),     # calm bull
        np.random.normal(-0.0040, 0.032, n_turbulent),  # turbulent selloff
        np.random.normal(0.0007, 0.007, n_calm_2),      # calm recovery
    ])
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.004, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.004, n)))
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)

    bars = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    feats = compute_feature_set(bars)

    assigner = SimpleQuantileRegimeAssigner(feats["realized_vol_20"])
    print("Stand-in 'model' means_ (mean realized_vol_20 per state):")
    for state_id, label in [(assigner.calm_id, "calm"), (assigner.mid_id, "mid"), (assigner.turbulent_id, "turbulent")]:
        print(f"  state {state_id} ({label}): {assigner.means_[state_id, 0]:.4f}")
    print()

    orchestrator = StrategyOrchestrator()
    orchestrator.rebuild_mapping(assigner, FEATURE_COLUMNS, n_states=N_STATES)

    print("Resulting state -> strategy mapping (this is the real production code, not a mock):")
    for state_id, label in [(assigner.calm_id, "calm"), (assigner.mid_id, "mid"), (assigner.turbulent_id, "turbulent")]:
        print(f"  state {state_id} ({label}) -> {orchestrator.strategy_for_state(state_id).name}")
    print()

    # Sample one point from deep inside each phase, plus the first bar of
    # the turbulent phase (a genuine regime transition) to see the
    # uncertainty gate fire.
    sample_indices = {
        "deep in calm bull": 100,
        "transition into turbulence": n_calm_1 + 1,
        "deep in turbulent selloff": n_calm_1 + 40,
        "deep in calm recovery": n_calm_1 + n_turbulent + 100,
    }

    results = []
    for description, idx in sample_indices.items():
        symbol_bars = feats.iloc[: idx + 1]  # only data available "as of" this point
        state = int(assigner.state_sequence.iloc[idx])
        recent_states = [
            int(s) for s in assigner.state_sequence.iloc[max(0, idx - 4): idx + 1].dropna()
        ]
        prob = confidence_for(idx, assigner.state_sequence)

        regime_state = RegimeState(
            state=state, probability=prob, label=description,
            recent_states=recent_states, timestamp=dates[idx],
        )
        signals = orchestrator.generate_signals({"SPY": symbol_bars}, {"SPY": regime_state})
        signal = signals.get("SPY")

        print(f"--- {description} ({dates[idx].date()}) ---")
        print(f"  realized_vol_20={feats['realized_vol_20'].iloc[idx]:.3f}, price={feats['close'].iloc[idx]:.2f}")
        if signal is None:
            print("  No signal (insufficient data)")
        else:
            print(f"  strategy={signal.regime.strategy_name}, confidence={signal.confidence:.2f}")
            print(f"  allocation={signal.position_size_pct:.1%}, leverage={signal.leverage}x, stop={signal.stop_loss:.2f}")
            print(f"  reasoning: {signal.regime.reasoning}")
        print()

        results.append({
            "description": description,
            "date": dates[idx],
            "close": feats["close"].iloc[idx],
            "realized_vol_20": feats["realized_vol_20"].iloc[idx],
            "signal": signal,
        })

    return bars, feats, assigner, results


if __name__ == "__main__":
    main()
