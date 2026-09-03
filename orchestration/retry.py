"""Generic retry-with-exponential-backoff.

Per the error-handling spec: Robinhood API failures get 3 retries with
exponential backoff. Built as a generic utility rather than Robinhood-
specific, so it's reusable and independently testable.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_SECONDS = 1.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0


class RetryExhaustedError(Exception):
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Failed after {attempts} attempts: {last_exception}")


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Calls fn(), retrying up to max_retries times on any exception, with
    the delay doubling each attempt (base, base*2, base*4, ...).
    `sleep_fn` is injectable so tests don't have to actually sleep.
    """
    last_exception: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any broker call failure should retry
            last_exception = exc
            if attempt < max_retries:
                delay = base_delay_seconds * (backoff_multiplier**attempt)
                logger.warning(
                    "Attempt %d/%d failed: %s -- retrying in %.1fs", attempt + 1, max_retries + 1, exc, delay
                )
                sleep_fn(delay)
            else:
                logger.error("All %d attempts failed. Last error: %s", max_retries + 1, exc)

    raise RetryExhaustedError(max_retries + 1, last_exception)
