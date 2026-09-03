"""Recent news headlines per symbol, via Alpaca's News API -- same account/
credentials as data/fetchers.py's AlpacaIntradayHistoricalFetcher, no new
signup needed. Field names below (NewsRequest's symbols/start/limit/
exclude_contentless, News's headline/source/url/created_at/symbols) were
checked against the real installed alpaca-py package (0.44.0) --
NewsRequest.model_fields, News.model_fields -- and a real live API call was
made and returned real headlines during development. Not guessed from
memory, same verification standard as the rest of data/fetchers.py.

PURELY A DASHBOARD/EXPLANATORY FEATURE, not a trading signal: this project's
actual decision logic (brain/confluence.py) reads price/volume only.
Headlines fetched here are NEVER passed into decide_options_action or
evaluate_confluence, and nothing in orchestration/ imports this module.
Turning "here's a headline near this move" into "trade because of this
headline" would mean building and validating a real sentiment-to-price
model -- a materially bigger and riskier undertaking than "show a human
what might explain this," which is what this exists for. If that's ever
wanted, it should be its own deliberate project, not a quiet addition here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

DEFAULT_LOOKBACK_HOURS = 48
DEFAULT_LIMIT = 4


@dataclass(frozen=True)
class NewsItem:
    headline: str
    source: str
    url: Optional[str]
    created_at: datetime
    symbols: List[str]


class AlpacaNewsFetcher:
    def __init__(self) -> None:
        from alpaca.data.historical.news import NewsClient  # lazy import, matches this project's convention

        api_key = os.environ.get("ALPACA_API_KEY_ID")
        secret_key = os.environ.get("ALPACA_API_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY as environment variables before using this fetcher.")
        self._client = NewsClient(api_key=api_key, secret_key=secret_key)

    def get_recent_news(self, symbol: str, hours: int = DEFAULT_LOOKBACK_HOURS, limit: int = DEFAULT_LIMIT) -> List[NewsItem]:
        from alpaca.data.requests import NewsRequest

        request = NewsRequest(
            symbols=symbol, start=datetime.now(timezone.utc) - timedelta(hours=hours),
            limit=limit, exclude_contentless=True,
        )
        result = self._client.get_news(request)
        raw_items = result.data.get("news", []) if hasattr(result, "data") else list(result)
        return [
            NewsItem(headline=n.headline, source=n.source, url=n.url, created_at=n.created_at, symbols=n.symbols)
            for n in raw_items
        ]
