"""Walk-forward windowing and forward-only HMM filtering.

Two pieces, both foundational to the walk-forward engine and both easy to
get subtly wrong in a way that leaks future information into the backtest:

1. `generate_walk_forward_windows` — the IS/OOS/step schedule.
2. `forward_filter` — deliberately NOT hmmlearn's `predict_proba()`, which
   does forward-*backward* smoothing. Smoothing P(state_t | all data in the
   slice) is fine as long as the slice never extends past t, but it's easy
   to misuse by accident. This implements the forward algorithm directly
   (alpha recursion) so there's no way to accidentally pass in future bars.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: int
    is_start: int   # index into the full bar series, inclusive
    is_end: int     # exclusive
    oos_start: int  # == is_end
    oos_end: int    # exclusive


def generate_walk_forward_windows(
    n_bars: int, is_size: int = 252, oos_size: int = 126, step_size: int = 126
) -> List[WalkForwardWindow]:
    """Generates walk-forward windows. A trailing partial window that
    doesn't have enough bars for a full OOS period is dropped rather than
    truncated — flag if a partial final window should be kept instead.
    """
    if is_size <= 0 or oos_size <= 0 or step_size <= 0:
        raise ValueError("is_size, oos_size, and step_size must all be positive")

    windows = []
    is_start = 0
    window_id = 0
    while True:
        is_end = is_start + is_size
        oos_start = is_end
        oos_end = oos_start + oos_size
        if oos_end > n_bars:
            break
        windows.append(WalkForwardWindow(window_id, is_start, is_end, oos_start, oos_end))
        window_id += 1
        is_start += step_size

    return windows


def forward_filter_from_log_emission(
    startprob: np.ndarray, transmat: np.ndarray, log_emission: np.ndarray
) -> np.ndarray:
    """Pure forward-algorithm math, split out from `forward_filter()` so it's
    testable without needing an actual fitted hmmlearn model.

    Returns filtered_probs of shape (T, n_states), where row t is
    P(state_t | observations_0..t) — using only data up to and including t,
    never beyond it. Computed in log-space for numerical stability.

    log_emission: shape (T, n_states), log P(observation_t | state=k).
    """
    T, n_states = log_emission.shape
    log_startprob = np.log(startprob)
    log_transmat = np.log(transmat)

    log_alpha = np.zeros((T, n_states))
    log_alpha[0] = log_startprob + log_emission[0]
    log_alpha[0] -= logsumexp(log_alpha[0])

    for t in range(1, T):
        # log_alpha[t, j] = logsumexp_k(log_alpha[t-1, k] + log_transmat[k, j]) + log_emission[t, j]
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0) + log_emission[t]
        log_alpha[t] -= logsumexp(log_alpha[t])

    return np.exp(log_alpha)


def forward_filter(model, X: np.ndarray) -> np.ndarray:
    """Forward-filter observations X through a fitted hmmlearn GaussianHMM.

    Uses the model's own `_compute_log_likelihood` (hmmlearn's internal
    emission-probability computation, which correctly handles whatever
    covariance_type the model was fit with) so this works regardless of
    covariance_type, then runs the pure forward recursion above.

    Relies on a private hmmlearn method (`_compute_log_likelihood`), which
    is a real (if minor) risk: it could change or be renamed between
    hmmlearn versions. Worth pinning a hmmlearn version in requirements.txt
    if this becomes a problem.
    """
    log_emission = model._compute_log_likelihood(X)
    return forward_filter_from_log_emission(model.startprob_, model.transmat_, log_emission)
