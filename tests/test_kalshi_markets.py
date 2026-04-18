"""Unit tests for Kalshi market discovery."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from bot.kalshi_markets import (
    _parse_retry_after_seconds,
    _passes_filters,
    _window_cutoff_ts,
    build_standalone_market,
    fetch_candidate_markets,
)


def _future_iso(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat().replace("+00:00", "Z")


def _base_market(**overrides):
    market = {
        "ticker": "KXWILL-25DEC31-Y",
        "event_ticker": "KXWILL",
        "title": "Will something happen by year end?",
        "subtitle": "Yes or no?",
        "category": "Politics",
        "status": "active",
        "market_type": "binary",
        "yes_bid": 35,
        "yes_ask": 37,
        "close_time": _future_iso(30),
        "volume": 1000,
        "liquidity": 500,
    }
    market.update(overrides)
    return market


def test_passes_filters_accepts_binary_non_sports_market() -> None:
    assert _passes_filters(_base_market(), max_end_date_months=3)


def test_passes_filters_rejects_sports_category() -> None:
    assert not _passes_filters(
        _base_market(category="Sports"),
        max_end_date_months=3,
    )


def test_passes_filters_rejects_crypto_keyword_in_title() -> None:
    assert not _passes_filters(
        _base_market(title="Will Bitcoin hit $200k?"),
        max_end_date_months=3,
    )


def test_passes_filters_rejects_markets_closing_outside_window() -> None:
    assert not _passes_filters(
        _base_market(close_time=_future_iso(400)),
        max_end_date_months=3,
    )


def test_passes_filters_rejects_already_closed_market() -> None:
    assert not _passes_filters(
        _base_market(status="closed"),
        max_end_date_months=3,
    )


def test_passes_filters_rejects_non_binary_market() -> None:
    assert not _passes_filters(
        _base_market(market_type="scalar"),
        max_end_date_months=3,
    )


def test_build_standalone_market_derives_no_mid() -> None:
    raw = _base_market(yes_bid=35, yes_ask=37)
    market = build_standalone_market(raw)

    assert market is not None
    assert market.slug == "KXWILL-25DEC31-Y"
    assert market.yes_token_id == "KXWILL-25DEC31-Y"
    assert market.no_token_id == "KXWILL-25DEC31-Y"
    # yes_mid = (0.35 + 0.37)/2 = 0.36; no_mid = 1 - 0.36 = 0.64
    assert market.yes_price == pytest.approx(0.36)
    assert market.no_price == pytest.approx(0.64)
    assert market.event_slug == "KXWILL"


def test_build_standalone_market_returns_none_without_ticker() -> None:
    assert build_standalone_market({"ticker": ""}) is None


def test_parse_retry_after_seconds_handles_integer_header() -> None:
    class _Headers:
        def get(self, key):
            assert key == "Retry-After"
            return "7"

    assert _parse_retry_after_seconds(_Headers()) == 7.0


def test_parse_retry_after_seconds_returns_none_for_missing() -> None:
    assert _parse_retry_after_seconds(None) is None
    assert _parse_retry_after_seconds({}) is None
    assert _parse_retry_after_seconds({"Retry-After": "not-a-number"}) is None


def test_window_cutoff_ts_rounds_to_months_ahead() -> None:
    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff = _window_cutoff_ts(3)
    assert cutoff is not None
    # Cutoff should be ~90 days ahead (3 months * 30 days).
    assert cutoff - now_ts == pytest.approx(90 * 24 * 3600, rel=0.01)
    assert _window_cutoff_ts(0) is None


@pytest.mark.asyncio
async def test_fetch_candidate_markets_passes_max_close_ts_to_iterator() -> None:
    session = object()
    seen: dict = {}

    async def _async_gen(_sess, base_url=None, max_close_ts=None):
        seen["max_close_ts"] = max_close_ts
        return
        yield  # make this an async generator

    with patch("bot.kalshi_markets._iter_market_batches", _async_gen):
        await fetch_candidate_markets(session, max_end_date_months=3)

    assert seen["max_close_ts"] is not None
    assert seen["max_close_ts"] > datetime.now(timezone.utc).timestamp()


@pytest.mark.asyncio
async def test_fetch_candidate_markets_paginates_and_filters() -> None:
    session = object()
    page_one = {
        "markets": [
            _base_market(),  # keep
            _base_market(ticker="KXMLB-25DEC-Y", category="Sports"),  # drop
        ],
        "cursor": "next",
    }
    page_two = {
        "markets": [
            _base_market(ticker="KXRAIN-25DEC31-Y", volume=500),
        ],
        "cursor": None,
    }
    batches = [page_one["markets"], page_two["markets"]]

    async def _async_gen(_sess, base_url=None, max_close_ts=None):
        for batch in batches:
            yield batch

    with patch("bot.kalshi_markets._iter_market_batches", _async_gen):
        markets = await fetch_candidate_markets(session)

    slugs = [m.slug for m in markets]
    assert slugs == ["KXWILL-25DEC31-Y", "KXRAIN-25DEC31-Y"]  # volume-sorted, sports dropped
