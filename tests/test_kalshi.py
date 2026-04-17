"""Unit tests for the Kalshi exchange adapter."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from bot.config import ExchangeConfig
from bot.exchange.kalshi import (
    KalshiExchangeClient,
    _cents,
    _estimate_contract_count,
    _no_best_bid_ask,
    _no_book_as_levels,
    _position_row_to_polymarket_shape,
    _prob,
    _KalshiOrderBook,
)
from bot.models import LimitOrderIntent, MarketOrderIntent, Side


# --------------------------- helpers ---------------------------


def _gen_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


def _paper_config() -> ExchangeConfig:
    # Paper-mode config: no credentials, live send disabled.
    return ExchangeConfig(
        host="https://api.elections.kalshi.com",
        chain_id=137,
        signature_type=0,
        private_key=None,
        funder_address=None,
        live_send_enabled=False,
        venue="kalshi",
    )


def _live_config() -> ExchangeConfig:
    return ExchangeConfig(
        host="https://api.elections.kalshi.com",
        chain_id=137,
        signature_type=0,
        private_key=None,
        funder_address=None,
        live_send_enabled=True,
        venue="kalshi",
        kalshi_api_key_id="test-key-id",
        kalshi_private_key_pem=_gen_pem(),
    )


class _Response:
    def __init__(self, json_body, status: int = 200):
        self._body = json_body
        self.status_code = status
        self.text = json.dumps(json_body) if json_body is not None else ""
        self.content = self.text.encode("utf-8")

    def json(self):
        return self._body


class _FakeSession:
    """Recording requests.Session stand-in."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.next_response: _Response = _Response({})
        self.headers: dict[str, str] = {}

    def request(self, method, url, params=None, data=None, headers=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "data": json.loads(data.decode()) if data else None,
                "headers": headers or {},
            }
        )
        return self.next_response


# --------------------------- primitive helpers ---------------------------


def test_cents_clamps_to_kalshi_range() -> None:
    assert _cents(0.0) == 1
    assert _cents(1.0) == 99
    assert _cents(0.37) == 37


def test_prob_handles_empty_and_strings() -> None:
    assert _prob(None) == 0.0
    assert _prob("") == 0.0
    assert _prob(42) == 0.42
    assert _prob("15") == 0.15


def test_no_best_bid_ask_derives_ask_from_yes_bid() -> None:
    book = _KalshiOrderBook(
        yes_bids=[(62, 100), (60, 200)],
        no_bids=[(37, 300), (35, 50)],
    )
    best_no_bid, best_no_ask = _no_best_bid_ask(book)
    assert best_no_bid == 37
    # Best YES bid is 62¢, so best NO ask = 100 - 62 = 38¢.
    assert best_no_ask == 38


def test_no_book_as_levels_produces_bid_ask_ladder() -> None:
    book = _KalshiOrderBook(
        yes_bids=[(62, 100), (60, 200)],
        no_bids=[(37, 300), (35, 50)],
    )
    bids, asks = _no_book_as_levels(book)
    assert [(lvl.price, lvl.size) for lvl in bids] == [(0.37, 300.0), (0.35, 50.0)]
    assert [(lvl.price, lvl.size) for lvl in asks] == [(0.38, 100.0), (0.40, 200.0)]


def test_estimate_contract_count_for_buy_uses_reference_price() -> None:
    order = MarketOrderIntent(
        token_id="KX",
        side=Side.BUY,
        amount=10.0,
        reference_price=0.4,
    )
    # $10 budget at 40¢ per contract = 25 contracts (floor, not round).
    assert _estimate_contract_count(order) == 25


def test_estimate_contract_count_for_sell_uses_amount_as_contracts() -> None:
    order = MarketOrderIntent(token_id="KX", side=Side.SELL, amount=7.3)
    assert _estimate_contract_count(order) == 7


# --------------------------- client ---------------------------


def test_client_rejects_live_trading_without_credentials() -> None:
    cfg = replace(_paper_config(), live_send_enabled=True)
    with pytest.raises(ValueError, match="KALSHI_API_KEY_ID"):
        KalshiExchangeClient(cfg, allow_trading=True, session=_FakeSession())


def test_client_allows_paper_mode_without_credentials() -> None:
    client = KalshiExchangeClient(_paper_config(), allow_trading=False, session=_FakeSession())
    assert client.get_open_orders("KXABC") == []
    assert client.get_trades("KXABC") == []
    assert client.get_collateral_balance() == 0.0
    assert client.get_conditional_balance("KXABC") == 0.0


def test_get_market_rules_is_constant() -> None:
    client = KalshiExchangeClient(_paper_config(), allow_trading=False, session=_FakeSession())
    rules = client.get_market_rules("KXABC")
    assert rules is not None
    assert rules.tick_size == 0.01
    assert rules.min_order_size == 1.0


def test_get_mid_price_averages_no_bid_and_derived_no_ask() -> None:
    session = _FakeSession()
    session.next_response = _Response(
        {"orderbook": {"yes": [[62, 100], [60, 200]], "no": [[37, 300], [35, 50]]}}
    )
    client = KalshiExchangeClient(_paper_config(), allow_trading=False, session=session)
    # Best NO bid = 37¢, best NO ask = 100 - 62 = 38¢, mid = 0.375
    assert client.get_mid_price("KXABC") == pytest.approx(0.375)
    assert session.calls[0]["method"] == "GET"
    assert session.calls[0]["url"].endswith("/markets/KXABC/orderbook")
    assert "KALSHI-ACCESS-KEY" not in session.calls[0]["headers"]


def test_place_limit_order_builds_no_side_payload() -> None:
    session = _FakeSession()
    session.next_response = _Response({"order": {"order_id": "ord-1", "status": "resting"}})
    client = KalshiExchangeClient(_live_config(), allow_trading=True, session=session)

    result = client.place_limit_order(
        LimitOrderIntent(token_id="KXMARKET-25DEC31", side=Side.BUY, price=0.37, size=10)
    )

    assert result.order_id == "ord-1"
    body = session.calls[0]["data"]
    assert body["action"] == "buy"
    assert body["side"] == "no"
    assert body["ticker"] == "KXMARKET-25DEC31"
    assert body["type"] == "limit"
    assert body["count"] == 10
    assert body["no_price"] == 37
    headers = session.calls[0]["headers"]
    assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert "KALSHI-ACCESS-SIGNATURE" in headers
    assert headers["KALSHI-ACCESS-TIMESTAMP"].isdigit()


def test_place_limit_order_rejected_when_trading_disabled() -> None:
    client = KalshiExchangeClient(_paper_config(), allow_trading=False, session=_FakeSession())
    with pytest.raises(RuntimeError, match="Order transmission is disabled"):
        client.place_limit_order(
            LimitOrderIntent(token_id="KXABC", side=Side.BUY, price=0.3, size=5)
        )


def test_place_market_order_applies_price_cap() -> None:
    session = _FakeSession()
    session.next_response = _Response({"order": {"order_id": "ord-2", "status": "filled"}})
    client = KalshiExchangeClient(_live_config(), allow_trading=True, session=session)

    result = client.place_market_order(
        MarketOrderIntent(
            token_id="KXABC",
            side=Side.BUY,
            amount=20.0,
            reference_price=0.40,
            price_cap=0.50,
        )
    )

    assert result.order_id == "ord-2"
    body = session.calls[0]["data"]
    assert body["type"] == "market"
    assert body["action"] == "buy"
    assert body["count"] == 50  # 20 / 0.40
    assert body["no_price"] == 50
    assert result.raw["_buffered_price"] == pytest.approx(0.50)


def test_cancel_order_returns_false_when_trading_disabled() -> None:
    client = KalshiExchangeClient(_paper_config(), allow_trading=False, session=_FakeSession())
    assert client.cancel_order("some-id") is False


def test_cancel_order_sends_delete() -> None:
    session = _FakeSession()
    session.next_response = _Response({})
    client = KalshiExchangeClient(_live_config(), allow_trading=True, session=session)

    assert client.cancel_order("ord-1") is True
    call = session.calls[0]
    assert call["method"] == "DELETE"
    assert call["url"].endswith("/portfolio/orders/ord-1")


def test_get_collateral_balance_converts_cents_to_dollars() -> None:
    session = _FakeSession()
    session.next_response = _Response({"balance": 12345})
    client = KalshiExchangeClient(_live_config(), allow_trading=True, session=session)

    assert client.get_collateral_balance() == pytest.approx(123.45)


def test_get_conditional_balance_returns_no_position_as_positive() -> None:
    session = _FakeSession()
    session.next_response = _Response(
        {"market_positions": [{"ticker": "KXABC", "position": -42}]}
    )
    client = KalshiExchangeClient(_live_config(), allow_trading=True, session=session)

    # Kalshi uses signed position: negative = long NO. Adapter returns magnitude.
    assert client.get_conditional_balance("KXABC") == 42.0


def test_get_conditional_balance_zero_for_yes_only_position() -> None:
    session = _FakeSession()
    session.next_response = _Response(
        {"market_positions": [{"ticker": "KXABC", "position": 10}]}
    )
    client = KalshiExchangeClient(_live_config(), allow_trading=True, session=session)

    # Long YES → no NO contracts to sell.
    assert client.get_conditional_balance("KXABC") == 0.0


def test_check_order_readiness_checks_cash_for_buy() -> None:
    responses = iter(
        [
            _Response({"balance": 20000}),  # $200
        ]
    )

    class _Iter(_FakeSession):
        def request(self, *a, **kw):
            self.calls.append({"method": kw.get("method") or a[0]})
            return next(responses)

    session = _Iter()
    client = KalshiExchangeClient(_live_config(), allow_trading=True, session=session)

    readiness = client.check_order_readiness(
        LimitOrderIntent(token_id="KXABC", side=Side.BUY, price=0.5, size=10)
    )
    assert readiness.ready is True
    assert readiness.balance == pytest.approx(200.0)


def test_check_order_readiness_requires_credentials() -> None:
    client = KalshiExchangeClient(_paper_config(), allow_trading=False, session=_FakeSession())
    readiness = client.check_order_readiness(
        LimitOrderIntent(token_id="KXABC", side=Side.BUY, price=0.5, size=10)
    )
    assert readiness.ready is False
    assert "Authenticated" in readiness.reason


def test_position_row_to_polymarket_shape_maps_no_long_position() -> None:
    row = {
        "ticker": "KXABC-25DEC31",
        "position": -100,
        "market_exposure": 3000,  # $30.00 cost basis for 100 NO contracts → $0.30 avg
        "realized_pnl": -200,
        "event_ticker": "KXABC",
    }
    mapped = _position_row_to_polymarket_shape(row)
    assert mapped is not None
    assert mapped["slug"] == "KXABC-25DEC31"
    assert mapped["asset"] == "KXABC-25DEC31"
    assert mapped["outcome"] == "No"
    assert mapped["conditionId"] == "KXABC"
    assert mapped["size"] == 100.0
    assert mapped["avgPrice"] == pytest.approx(0.30)
    assert mapped["initialValue"] == pytest.approx(30.00)
    assert mapped["cashPnl"] == pytest.approx(-2.00)


def test_position_row_to_polymarket_shape_drops_yes_long_and_empty() -> None:
    assert _position_row_to_polymarket_shape({"ticker": "KXABC", "position": 50}) is None
    assert _position_row_to_polymarket_shape({"ticker": "KXABC", "position": 0}) is None
    assert _position_row_to_polymarket_shape({"ticker": "", "position": -50}) is None


def test_get_portfolio_positions_paginates_and_filters() -> None:
    session = _FakeSession()
    responses = iter(
        [
            _Response(
                {
                    "market_positions": [
                        {"ticker": "KXA", "position": -10, "market_exposure": 300},
                        {"ticker": "KXB", "position": 5, "market_exposure": 150},  # YES-long, drop
                    ],
                    "cursor": "page2",
                }
            ),
            _Response(
                {
                    "market_positions": [
                        {"ticker": "KXC", "position": -2, "market_exposure": 60},
                    ],
                    "cursor": None,
                }
            ),
        ]
    )

    def _request(method, url, params=None, data=None, headers=None, timeout=None):
        session.calls.append({"method": method, "url": url, "params": params})
        return next(responses)

    session.request = _request  # type: ignore[assignment]
    client = KalshiExchangeClient(_live_config(), allow_trading=True, session=session)

    positions = client.get_portfolio_positions()

    assert [p["slug"] for p in positions] == ["KXA", "KXC"]
    assert session.calls[0]["params"] == {"limit": 100}
    assert session.calls[1]["params"] == {"limit": 100, "cursor": "page2"}


def test_get_portfolio_positions_returns_empty_without_credentials() -> None:
    client = KalshiExchangeClient(_paper_config(), allow_trading=False, session=_FakeSession())
    assert client.get_portfolio_positions() == []


def test_signing_produces_stable_headers_per_call() -> None:
    client = KalshiExchangeClient(_live_config(), allow_trading=True, session=_FakeSession())
    headers = client._sign("GET", "/trade-api/v2/portfolio/balance")
    assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"].isdigit()
    # Base64 signature — length depends on key size but must be non-empty.
    assert len(headers["KALSHI-ACCESS-SIGNATURE"]) > 100
