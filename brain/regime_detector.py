"""Gaussian HMM volatility regime detector, with automatic model selection via BIC.

Design philosophy: this classifies market VOLATILITY regime only. It is
explicitly not a return/direction forecaster — the features driving state
assignment are volatility-related (realized vol, ATR%), never signed
returns.

After fitting, states are labeled using the *realized* mean return
conditional on each state. That's a post-hoc, descriptive labeling step —
useful because volatility and returns are empirically correlated (turbulent
periods skew negative, calm periods skew positive) — not something the model
uses predictively. Returns never enter the model as an input feature.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_STATES = 3
MAX_STATES = 7
N_INIT = 10

REGIME_LABELS: Dict[int, List[str]] = {
    3: ["bear", "neutral", "bull"],
    4: ["crash", "bear", "bull", "euphoria"],
    5: ["crash", "bear", "neutral", "bull", "euphoria"],
    6: ["crash", "strong bear", "weak bear", "weak bull", "strong bull", "euphoria"],
    7: ["crash", "strong bear", "weak bear", "neutral", "weak bull", "strong bull", "euphoria"],
}


@dataclass(frozen=True)
class BicCandidate:
    n_states: int
    log_likelihood: float
    n_params: int
    n_samples: int
    bic: float
    converged: bool


@dataclass(frozen=True)
class RegimeModel:
    model: "object"                 # fitted hmmlearn.hmm.GaussianHMM
    n_states: int
    state_order: List[int]          # state indices sorted by mean return, ascending
    state_labels: Dict[int, str]    # state index -> label string
    candidates: List[BicCandidate]  # every (n_states, BIC) tested, for audit/logging


def n_params(n_states: int, n_features: int, covariance_type: str) -> int:
    """Number of free parameters in a Gaussian HMM, for the BIC penalty term.

    startprob: n_states - 1 free values (they sum to 1)
    transmat: n_states rows, each summing to 1 -> n_states * (n_states - 1)
    means: n_states * n_features
    covars: depends on covariance_type
    """
    startprob = n_states - 1
    transmat = n_states * (n_states - 1)
    means = n_states * n_features

    if covariance_type == "diag":
        covars = n_states * n_features
    elif covariance_type == "full":
        covars = n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "spherical":
        covars = n_states
    elif covariance_type == "tied":
        covars = n_features * (n_features + 1) // 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    return startprob + transmat + means + covars


def compute_bic(log_likelihood: float, n_states: int, n_features: int, n_samples: int, covariance_type: str) -> float:
    k = n_params(n_states, n_features, covariance_type)
    return -2 * log_likelihood + k * np.log(n_samples)


def _fit_best_of_n_init(X: np.ndarray, n_states: int, covariance_type: str, n_init: int, base_seed: int):
    """Fit n_init random restarts, keep the one with highest log-likelihood.
    EM only finds a local optimum, so restarts aren't decoration — they're
    how a bad initialization gets caught.
    """
    from hmmlearn.hmm import GaussianHMM  # imported lazily — heavy optional dependency

    best_model = None
    best_ll = -np.inf

    for i in range(n_init):
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=200,
                random_state=base_seed + i,
                init_params="stmc",
            )
            model.fit(X)
            ll = model.score(X)
            if np.isfinite(ll) and ll > best_ll:
                best_ll = ll
                best_model = model
        except Exception as exc:  # hmmlearn can raise on degenerate covariance, etc.
            logger.warning("n_states=%d init=%d failed to fit: %s", n_states, i, exc)
            continue

    return best_model


def select_regime_model(
    features: pd.DataFrame,
    feature_columns: Sequence[str] = ("realized_vol_20", "normalized_atr_14", "vol_ratio_5_20"),
    covariance_type: str = "diag",
    min_states: int = MIN_STATES,
    max_states: int = MAX_STATES,
    n_init: int = N_INIT,
    random_state: int = 42,
) -> RegimeModel:
    """Fit Gaussian HMMs for n_states in [min_states, max_states], select the
    lowest-BIC model, and label its states by realized mean return.

    `features` must contain the volatility feature_columns (used to fit the
    model — see data/features.py) and a `log_return_1` column (used only for
    post-hoc labeling, never as a model input).
    """
    feature_columns = list(feature_columns)
    required = set(feature_columns) | {"log_return_1"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"select_regime_model: missing required columns {missing}")

    clean = features[feature_columns + ["log_return_1"]].dropna()
    if len(clean) < 100:
        raise ValueError(
            f"Only {len(clean)} clean samples after dropping NaNs — need substantially "
            f"more history to fit a stable regime model."
        )

    X = clean[feature_columns].to_numpy()
    n_samples, n_feat = X.shape

    candidates: List[BicCandidate] = []
    fitted_models: Dict[int, object] = {}

    for k_states in range(min_states, max_states + 1):
        model = _fit_best_of_n_init(X, k_states, covariance_type, n_init, base_seed=random_state)
        if model is None:
            logger.warning("n_states=%d: every init failed to converge, skipping", k_states)
            continue

        ll = model.score(X)
        bic = compute_bic(ll, k_states, n_feat, n_samples, covariance_type)

        candidates.append(
            BicCandidate(
                n_states=k_states,
                log_likelihood=ll,
                n_params=n_params(k_states, n_feat, covariance_type),
                n_samples=n_samples,
                bic=bic,
                converged=bool(model.monitor_.converged),
            )
        )
        fitted_models[k_states] = model

        logger.info(
            "n_states=%d: log_likelihood=%.2f, n_params=%d, BIC=%.2f, converged=%s",
            k_states, ll, n_params(k_states, n_feat, covariance_type), bic, model.monitor_.converged,
        )

    if not candidates:
        raise RuntimeError("No candidate model converged for any state count — check input data.")

    best = min(candidates, key=lambda c: c.bic)
    best_model = fitted_models[best.n_states]

    logger.info(
        "Selected n_states=%d (BIC=%.2f). All candidates: %s",
        best.n_states, best.bic,
        {c.n_states: round(c.bic, 1) for c in candidates},
    )

    state_order, state_labels = label_states(best_model, X, clean["log_return_1"].to_numpy(), best.n_states)

    return RegimeModel(
        model=best_model,
        n_states=best.n_states,
        state_order=state_order,
        state_labels=state_labels,
        candidates=candidates,
    )


def label_states(model, X: np.ndarray, returns: np.ndarray, n_states: int):
    """Decode the training sequence, compute mean realized return per state,
    and assign labels ordered from lowest to highest mean return.

    This is the only place realized returns touch the regime model, and it's
    purely descriptive: naming an already-fit volatility state, not
    influencing what the state boundaries are.
    """
    _, state_sequence = model.decode(X)
    return _label_states_from_sequence(state_sequence, returns, n_states)


def _label_states_from_sequence(state_sequence: np.ndarray, returns: np.ndarray, n_states: int):
    """Pure logic split out from label_states() so it's testable without an
    actual fitted hmmlearn model.
    """
    mean_return_by_state = {}
    for s in range(n_states):
        mask = state_sequence == s
        mean_return_by_state[s] = returns[mask].mean() if np.any(mask) else np.nan

    empty_states = [s for s, r in mean_return_by_state.items() if np.isnan(r)]
    if empty_states:
        raise RuntimeError(
            f"States {empty_states} were never assigned during decoding — the model has "
            f"more states than the data supports. Consider lowering max_states."
        )

    state_order = sorted(mean_return_by_state, key=lambda s: mean_return_by_state[s])

    if n_states not in REGIME_LABELS:
        raise ValueError(f"No label scheme defined for n_states={n_states}")
    names = REGIME_LABELS[n_states]

    state_labels = {state: names[rank] for rank, state in enumerate(state_order)}
    return state_order, state_labels
