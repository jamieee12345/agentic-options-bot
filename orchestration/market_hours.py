"""Market hours and model-staleness checks, used at startup."""
from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

MODEL_STALENESS_DAYS = 7


def is_market_open(timestamp: datetime) -> bool:
    """US equity regular-hours check: weekday, 9:30am-4:00pm ET (inclusive
    of open, exclusive of close). Does NOT account for market holidays — a
    fixed holiday calendar needs updating every year, so it's left as a
    known gap rather than a half-implemented list that goes stale.
    """
    local = timestamp.astimezone(MARKET_TZ)
    if local.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return MARKET_OPEN <= local.time() < MARKET_CLOSE


def is_model_stale(model_trained_at: Optional[datetime], now: datetime, max_age_days: int = MODEL_STALENESS_DAYS) -> bool:
    """True if the model is missing entirely (None) or older than
    max_age_days — matches "if model > 7 days old or missing, retrain".
    """
    if model_trained_at is None:
        return True
    return (now - model_trained_at) > timedelta(days=max_age_days)
