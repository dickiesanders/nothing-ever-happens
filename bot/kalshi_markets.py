"""Kalshi market discovery — mirrors ``bot.standalone_markets`` for the Kalshi venue.

Kalshi markets are individually tradable tickers organised into events and
series. For the nothing-happens strategy we want binary yes/no markets,
non-sports, closing within a bounded window. This module produces
``StandaloneMarket`` objects so the rest of the strategy needs no changes.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import aiohttp

from bot.standalone_markets import (
    DEFAULT_MAX_END_DATE_MONTHS,
    EXCLUDED_KEYWORDS,
    EXCLUDED_TITLE_PHRASES,
    StandaloneMarket,
    _parse_iso_ts,
)

logger = logging.getLogger(__name__)

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
PAGE_LIMIT = 200
PAGE_MAX_RETRIES = 5
PAGE_RETRY_BASE_DELAY_SEC = 1.0
PAGE_RETRY_MAX_DELAY_SEC = 20.0
PAGE_DELAY_SEC = 0.1

# Kalshi exposes a ``category`` field on markets; we drop sports-heavy
# categories entirely rather than keyword-matching every ticker.
EXCLUDED_CATEGORIES = {
    "sports",
    "sports scores",
    "athletics",
    "climate",  # noisy weather markets flap sub-hour; strategy assumes slower markets.
}


class KalshiMarketFetchError(RuntimeError):
    pass


def _category_excluded(market: dict) -> bool:
    cat = str(market.get("category") or "").strip().lower()
    if cat in EXCLUDED_CATEGORIES:
        return True
    # Kalshi often puts "Sports" at the start of category names.
    if cat.startswith("sports"):
        return True
    return False


def _keyword_excluded(market: dict) -> bool:
    haystack_parts = [
        str(market.get("title") or ""),
        str(market.get("subtitle") or ""),
        str(market.get("event_ticker") or ""),
        str(market.get("category") or ""),
    ]
    haystack = " ".join(haystack_parts).lower()
    if any(phrase in haystack for phrase in EXCLUDED_TITLE_PHRASES):
        return True
    if any(keyword in haystack for keyword in EXCLUDED_KEYWORDS):
        return True
    return False


def _close_ts(market: dict) -> float:
    value = (
        market.get("close_time")
        or market.get("expected_expiration_time")
        or market.get("expiration_time")
    )
    if not value:
        return 0.0
    return _parse_iso_ts(str(value))


def _ends_within_window(market: dict, *, max_end_date_months: int) -> bool:
    close_ts = _close_ts(market)
    if close_ts <= 0:
        return False
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=max_end_date_months * 30)
    end = datetime.fromtimestamp(close_ts, tz=timezone.utc)
    return now <= end <= cutoff


def _is_binary(market: dict) -> bool:
    # Kalshi markets that aren't binary have ``market_type`` != "binary".
    market_type = str(market.get("market_type") or "binary").lower()
    return market_type == "binary"


def _passes_filters(market: dict, *, max_end_date_months: int) -> bool:
    if str(market.get("status") or "").lower() != "active":
        return False
    if not _is_binary(market):
        return False
    if _category_excluded(market):
        return False
    if _keyword_excluded(market):
        return False
    if not _ends_within_window(market, max_end_date_months=max_end_date_months):
        return False
    return True


def build_standalone_market(market: dict) -> StandaloneMarket | None:
    ticker = str(market.get("ticker") or "")
    if not ticker:
        return None
    yes_bid = _cents_to_prob(market.get("yes_bid"))
    yes_ask = _cents_to_prob(market.get("yes_ask"))
    yes_mid = _midpoint(yes_bid, yes_ask)
    no_mid = 1.0 - yes_mid if yes_mid > 0 else 0.0
    close_ts = _close_ts(market)
    end_date = str(
        market.get("close_time")
        or market.get("expected_expiration_time")
        or market.get("expiration_time")
        or ""
    )
    return StandaloneMarket(
        question=str(market.get("title") or ticker),
        slug=ticker,
        condition_id=str(market.get("event_ticker") or ticker),
        yes_token_id=ticker,
        no_token_id=ticker,
        yes_price=yes_mid,
        no_price=no_mid,
        volume=float(market.get("volume") or 0.0),
        liquidity=float(market.get("liquidity") or 0.0),
        min_order_size=1.0,
        end_date=end_date,
        end_ts=close_ts,
        category=str(market.get("category") or ""),
        event_slug=str(market.get("event_ticker") or ""),
    )


def _cents_to_prob(value) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return max(0.0, min(1.0, float(value) / 100.0))
    except (TypeError, ValueError):
        return 0.0


def _midpoint(bid: float, ask: float) -> float:
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return bid or ask or 0.0


async def _iter_market_batches(session: aiohttp.ClientSession, *, base_url: str):
    cursor: str | None = None
    retries = 0
    while True:
        params: dict[str, str] = {"limit": str(PAGE_LIMIT), "status": "open"}
        if cursor:
            params["cursor"] = cursor
        try:
            async with session.get(
                f"{base_url}/markets",
                params=params,
                headers={"User-Agent": "kalshi-scanner/1.0", "Accept": "application/json"},
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        except aiohttp.ClientResponseError as exc:
            if exc.status in {429, 502, 503, 504} and retries < PAGE_MAX_RETRIES:
                delay = min(
                    PAGE_RETRY_BASE_DELAY_SEC * (2 ** retries),
                    PAGE_RETRY_MAX_DELAY_SEC,
                )
                retries += 1
                logger.warning(
                    "kalshi_markets_rate_limited cursor=%s retry=%d delay=%.2f",
                    cursor,
                    retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            logger.warning("kalshi_markets_fetch_aborted cursor=%s status=%s err=%s", cursor, exc.status, exc)
            raise KalshiMarketFetchError(f"kalshi_markets_fetch_aborted status={exc.status}") from exc
        except aiohttp.ClientError as exc:
            logger.warning("kalshi_markets_fetch_failed cursor=%s err=%s", cursor, exc)
            raise KalshiMarketFetchError("kalshi_markets_fetch_failed") from exc

        retries = 0
        markets = payload.get("markets") or []
        if not isinstance(markets, list):
            return
        if not markets:
            return
        yield markets

        cursor = payload.get("cursor") or None
        if not cursor:
            return
        await asyncio.sleep(PAGE_DELAY_SEC)


async def fetch_candidate_markets(
    session: aiohttp.ClientSession,
    *,
    max_end_date_months: int = DEFAULT_MAX_END_DATE_MONTHS,
    base_url: str = KALSHI_API_BASE,
) -> list[StandaloneMarket]:
    """Stream Kalshi ``open`` markets and project them as ``StandaloneMarket`` rows."""

    kept: list[StandaloneMarket] = []
    async for batch in _iter_market_batches(session, base_url=base_url):
        for raw in batch:
            if not _passes_filters(raw, max_end_date_months=max_end_date_months):
                continue
            market = build_standalone_market(raw)
            if market is None:
                continue
            kept.append(market)
    kept.sort(key=lambda market: market.volume, reverse=True)
    return kept


async def fetch_all_open_markets(session: aiohttp.ClientSession) -> list[dict]:
    """Raw passthrough equivalent of ``bot.standalone_markets.fetch_all_open_markets``."""
    out: list[dict] = []
    async for batch in _iter_market_batches(session, base_url=KALSHI_API_BASE):
        out.extend(batch)
    return out
