"""Kalshi exchange client for the nothing-happens runtime.

Kalshi is a US-regulated CFTC exchange with a simple REST API:
  - Auth:   RSA-PSS-SHA256 signing with a per-key ID + PEM private key.
  - Prices: integer cents 1..99.
  - Markets: identified by a ticker string; each market has yes/no sides.

This adapter treats the Polymarket-shaped ``token_id`` as a ticker and always
operates on the NO side, matching the strategy contract (it only buys NO on
standalone binary markets). Internally prices are stored as floats in [0, 1]
so the rest of the bot (which speaks probabilities) does not need to change.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from bot.config import ExchangeConfig
from bot.models import (
    LimitOrderIntent,
    MarketOrderIntent,
    MarketRules,
    OpenOrder,
    OrderBookLevel,
    OrderBookSnapshot,
    OrderReadiness,
    OrderResult,
    Side,
    Trade,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "https://api.elections.kalshi.com"
API_PREFIX = "/trade-api/v2"
TICK_SIZE = 0.01  # Kalshi quotes in 1-cent increments.
MIN_ORDER_SIZE = 1.0  # One contract minimum.
DEFAULT_TIMEOUT_SEC = 15.0


def _cents(probability: float) -> int:
    """Convert a [0, 1] probability to 1..99 cents (Kalshi's limit-price range)."""
    raw = int(round(float(probability) * 100))
    return max(1, min(99, raw))


def _prob(cents: int | float | str | None) -> float:
    if cents is None:
        return 0.0
    try:
        return float(cents) / 100.0
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class _KalshiOrderBook:
    yes_bids: list[tuple[int, int]]
    no_bids: list[tuple[int, int]]


class KalshiExchangeClient:
    """Synchronous Kalshi client exposing the ``ExchangeClient`` protocol.

    All methods operate on the NO side of the market identified by ``token_id``
    (which is the Kalshi ticker). If either authentication credential is
    missing the client runs in read-only mode: public endpoints still work but
    any authenticated call raises unless ``allow_trading`` is False.
    """

    def __init__(
        self,
        config: ExchangeConfig,
        allow_trading: bool,
        *,
        session: Any | None = None,
    ) -> None:
        try:
            import requests  # noqa: F401
            from cryptography.hazmat.primitives import hashes, serialization  # noqa: F401
            from cryptography.hazmat.primitives.asymmetric import padding  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency for Kalshi client (requests, cryptography). "
                "Install with: pip install -r requirements.txt"
            ) from exc

        self.allow_trading = allow_trading
        self.host = (config.host or DEFAULT_HOST).rstrip("/")
        self.api_key_id = config.kalshi_api_key_id
        self._private_key_pem = config.kalshi_private_key_pem
        self._private_key = None
        if self._private_key_pem:
            self._private_key = self._load_private_key(self._private_key_pem)

        if self.allow_trading and (not self.api_key_id or self._private_key is None):
            raise ValueError(
                "Kalshi live trading requires KALSHI_API_KEY_ID and a private key"
            )

        if session is not None:
            self._session = session
        else:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "nothing-ever-happens-kalshi/1.0"})

    # --------------------------- public HTTP helpers ---------------------------

    @staticmethod
    def _load_private_key(pem: str):
        from cryptography.hazmat.primitives import serialization

        return serialization.load_pem_private_key(pem.encode("utf-8"), password=None)

    def _sign(self, method: str, path: str) -> dict[str, str]:
        """Build Kalshi auth headers for an authenticated request."""
        if not self.api_key_id or self._private_key is None:
            raise RuntimeError("Kalshi authentication is not configured")

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + path).encode("utf-8")
        signature = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        auth: bool,
    ) -> Any:
        url = f"{self.host}{path}"
        headers: dict[str, str] = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if auth:
            headers.update(self._sign(method, path))
        data = json.dumps(body).encode("utf-8") if body is not None else None
        resp = self._session.request(
            method,
            url,
            params=params,
            data=data,
            headers=headers,
            timeout=DEFAULT_TIMEOUT_SEC,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Kalshi {method} {path} failed: {resp.status_code} {resp.text[:500]}"
            )
        if not resp.content:
            return {}
        return resp.json()

    # --------------------------- protocol surface ---------------------------

    def bootstrap_live_trading(self, token_id: str | None = None) -> None:
        # Kalshi has no on-chain approvals or balance/allowance sync to run.
        _ = token_id

    def warm_token_cache(self, token_id: str) -> None:
        _ = token_id

    def prepare_sell(self, token_id: str) -> bool:
        _ = token_id
        return True

    def get_mid_price(self, token_id: str) -> float:
        book = self._fetch_order_book(token_id)
        bid, ask = _no_best_bid_ask(book)
        if bid is None and ask is None:
            return 0.0
        if bid is None:
            return _prob(ask)
        if ask is None:
            return _prob(bid)
        return (_prob(bid) + _prob(ask)) / 2.0

    def get_market_rules(self, token_id: str) -> MarketRules | None:
        _ = token_id
        return MarketRules(tick_size=TICK_SIZE, min_order_size=MIN_ORDER_SIZE)

    def get_order_book(self, token_id: str) -> OrderBookSnapshot:
        book = self._fetch_order_book(token_id)
        bids, asks = _no_book_as_levels(book)
        return OrderBookSnapshot(
            token_id=token_id,
            bids=tuple(bids),
            asks=tuple(asks),
            tick_size=TICK_SIZE,
            min_order_size=MIN_ORDER_SIZE,
            timestamp=int(time.time() * 1000),
        )

    def get_open_orders(self, token_id: str) -> list[OpenOrder]:
        if not self.api_key_id or self._private_key is None:
            return []
        payload = self._request(
            "GET",
            f"{API_PREFIX}/portfolio/orders",
            params={"ticker": token_id, "status": "resting"},
            auth=True,
        )
        raw_orders = payload.get("orders") or []
        parsed: list[OpenOrder] = []
        for raw in raw_orders:
            try:
                parsed.append(self._parse_order(raw, default_ticker=token_id))
            except Exception as exc:
                logger.warning(
                    "kalshi_open_order_parse_failed",
                    extra={"error": str(exc), "raw": list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__},
                )
        return parsed

    def get_order(self, order_id: str) -> OpenOrder | None:
        if not self.api_key_id or self._private_key is None:
            return None
        try:
            payload = self._request(
                "GET",
                f"{API_PREFIX}/portfolio/orders/{order_id}",
                auth=True,
            )
        except Exception as exc:
            logger.warning("kalshi_get_order_failed", extra={"order_id": order_id, "error": str(exc)})
            return None
        raw = payload.get("order") or payload
        try:
            return self._parse_order(raw)
        except Exception as exc:
            logger.warning("kalshi_parse_order_failed", extra={"order_id": order_id, "error": str(exc)})
            return None

    def place_limit_order(self, order: LimitOrderIntent) -> OrderResult:
        if not self.allow_trading:
            raise RuntimeError("Order transmission is disabled")
        body = {
            "action": "buy" if order.side == Side.BUY else "sell",
            "client_order_id": str(uuid.uuid4()),
            "count": max(1, int(round(order.size))),
            "side": "no",
            "ticker": order.token_id,
            "type": "limit",
            "no_price": _cents(order.price),
        }
        response = self._request(
            "POST",
            f"{API_PREFIX}/portfolio/orders",
            body=body,
            auth=True,
        )
        raw_order = response.get("order") or response
        order_id = str(raw_order.get("order_id") or raw_order.get("id") or "")
        if not order_id:
            raise ValueError(f"Kalshi limit order response missing id: {response!r}")
        status = str(raw_order.get("status") or "submitted")
        logger.info(
            "kalshi_limit_order_submitted",
            extra={"order_id": order_id, "ticker": order.token_id, "status": status},
        )
        return OrderResult(order_id=order_id, status=status, raw=response)

    def place_market_order(self, order: MarketOrderIntent) -> OrderResult:
        if not self.allow_trading:
            raise RuntimeError("Order transmission is disabled")

        action = "buy" if order.side == Side.BUY else "sell"
        count = _estimate_contract_count(order)
        body: dict[str, Any] = {
            "action": action,
            "client_order_id": str(uuid.uuid4()),
            "count": count,
            "side": "no",
            "ticker": order.token_id,
            "type": "market",
        }
        if order.price_cap is not None:
            body["no_price"] = _cents(order.price_cap)
        elif order.reference_price is not None:
            slippage = order.allowed_slippage if order.allowed_slippage is not None else 0.05
            if action == "buy":
                body["no_price"] = _cents(order.reference_price + slippage)
            else:
                body["no_price"] = _cents(order.reference_price - slippage)

        response = self._request(
            "POST",
            f"{API_PREFIX}/portfolio/orders",
            body=body,
            auth=True,
        )
        raw_order = response.get("order") or response
        order_id = str(raw_order.get("order_id") or raw_order.get("id") or "")
        if not order_id:
            raise ValueError(f"Kalshi market order response missing id: {response!r}")
        status = str(raw_order.get("status") or "submitted")

        fill_count = _coerce_float(raw_order.get("taker_fill_count"), default=float(count))
        fill_cost_cents = _coerce_float(raw_order.get("taker_fill_cost"), default=0.0)
        fill_price = None
        if fill_count and fill_cost_cents:
            fill_price = (fill_cost_cents / fill_count) / 100.0

        enriched = dict(response) if isinstance(response, dict) else {"raw": response}
        enriched["_market_price"] = _prob(body.get("no_price"))
        enriched["_buffered_price"] = _prob(body.get("no_price"))
        enriched["_fill_price"] = fill_price
        enriched["_price_source"] = "kalshi"
        logger.info(
            "kalshi_market_order_submitted",
            extra={
                "order_id": order_id,
                "ticker": order.token_id,
                "action": action,
                "count": count,
                "status": status,
                "fill_price": fill_price,
            },
        )
        return OrderResult(order_id=order_id, status=status, raw=enriched)

    def get_trades(self, token_id: str, after_timestamp: int | None = None) -> list[Trade]:
        if not self.api_key_id or self._private_key is None:
            return []
        params: dict[str, Any] = {"ticker": token_id, "limit": 100}
        if after_timestamp is not None:
            params["min_ts"] = int(after_timestamp)
        try:
            payload = self._request(
                "GET",
                f"{API_PREFIX}/portfolio/fills",
                params=params,
                auth=True,
            )
        except Exception as exc:
            logger.error("kalshi_get_fills_failed", extra={"ticker": token_id, "error": str(exc)})
            return []

        raw_fills = payload.get("fills") or []
        trades: list[Trade] = []
        for raw in raw_fills:
            try:
                trades.append(self._parse_fill(raw, default_ticker=token_id))
            except Exception as exc:
                logger.warning("kalshi_fill_parse_failed", extra={"error": str(exc)})
        return trades

    def check_order_readiness(self, order: LimitOrderIntent | MarketOrderIntent) -> OrderReadiness:
        if not self.api_key_id or self._private_key is None:
            return OrderReadiness(False, "Authenticated Kalshi access is unavailable")
        try:
            balance = self.get_collateral_balance()
        except Exception as exc:
            logger.warning("kalshi_readiness_check_failed", extra={"error": str(exc)})
            return OrderReadiness(False, "Could not verify Kalshi balance")

        required = order.notional
        if order.side == Side.SELL:
            # Selling a NO contract releases cash; we just need the position.
            try:
                position = self.get_conditional_balance(order.token_id)
            except Exception as exc:
                logger.warning(
                    "kalshi_position_check_failed",
                    extra={"ticker": order.token_id, "error": str(exc)},
                )
                return OrderReadiness(False, "Could not verify Kalshi position")
            if position + 1e-9 < order.size:
                return OrderReadiness(
                    False,
                    "Insufficient NO contracts for sell order",
                    balance=position,
                )
            return OrderReadiness(True, "ok", balance=position)

        if balance + 1e-9 < required:
            return OrderReadiness(
                False,
                "Insufficient cash balance for order",
                balance=balance,
            )
        return OrderReadiness(True, "ok", balance=balance)

    def cancel_order(self, order_id: str) -> bool:
        if not self.allow_trading:
            return False
        try:
            self._request(
                "DELETE",
                f"{API_PREFIX}/portfolio/orders/{order_id}",
                auth=True,
            )
            return True
        except Exception as exc:
            logger.warning("kalshi_cancel_order_failed", extra={"order_id": order_id, "error": str(exc)})
            return False

    def cancel_all(self) -> bool:
        if not self.allow_trading:
            return False
        try:
            payload = self._request(
                "GET",
                f"{API_PREFIX}/portfolio/orders",
                params={"status": "resting"},
                auth=True,
            )
        except Exception as exc:
            logger.warning("kalshi_cancel_all_list_failed", extra={"error": str(exc)})
            return False
        orders = payload.get("orders") or []
        all_ok = True
        for raw in orders:
            order_id = str(raw.get("order_id") or raw.get("id") or "")
            if not order_id:
                continue
            if not self.cancel_order(order_id):
                all_ok = False
        return all_ok

    # --------------------------- balances ---------------------------

    def get_collateral_balance(self) -> float:
        """Cash balance in dollars (Kalshi returns cents)."""
        if not self.api_key_id or self._private_key is None:
            return 0.0
        payload = self._request("GET", f"{API_PREFIX}/portfolio/balance", auth=True)
        cents = _coerce_float(payload.get("balance"), default=0.0)
        return cents / 100.0

    def get_conditional_balance(self, token_id: str) -> float:
        """Net NO position (contracts) for the given ticker."""
        if not self.api_key_id or self._private_key is None:
            return 0.0
        try:
            payload = self._request(
                "GET",
                f"{API_PREFIX}/portfolio/positions",
                params={"ticker": token_id},
                auth=True,
            )
        except Exception as exc:
            logger.warning(
                "kalshi_position_fetch_failed",
                extra={"ticker": token_id, "error": str(exc)},
            )
            return 0.0

        positions = payload.get("market_positions") or []
        for pos in positions:
            if str(pos.get("ticker")) != token_id:
                continue
            # ``position`` is signed: positive means long YES, negative means long NO.
            position = _coerce_float(pos.get("position"), default=0.0)
            if position < 0:
                return abs(position)
            return 0.0
        return 0.0

    # --------------------------- parsers ---------------------------

    def _fetch_order_book(self, token_id: str) -> _KalshiOrderBook:
        payload = self._request(
            "GET",
            f"{API_PREFIX}/markets/{token_id}/orderbook",
            auth=False,
        )
        book = payload.get("orderbook") or {}

        def _levels(raw: Any) -> list[tuple[int, int]]:
            if not isinstance(raw, list):
                return []
            out: list[tuple[int, int]] = []
            for level in raw:
                if not isinstance(level, (list, tuple)) or len(level) < 2:
                    continue
                try:
                    out.append((int(level[0]), int(level[1])))
                except (TypeError, ValueError):
                    continue
            return out

        return _KalshiOrderBook(
            yes_bids=_levels(book.get("yes")),
            no_bids=_levels(book.get("no")),
        )

    def _parse_order(self, raw: dict[str, Any], default_ticker: str | None = None) -> OpenOrder:
        order_id = str(raw.get("order_id") or raw.get("id") or "")
        if not order_id:
            raise KeyError(f"Kalshi order missing id; keys present: {list(raw.keys())}")
        ticker = str(raw.get("ticker") or default_ticker or "")
        action = str(raw.get("action") or "buy").lower()
        side = Side.BUY if action == "buy" else Side.SELL
        no_price = raw.get("no_price") or raw.get("price")
        price = _prob(no_price)
        original = _coerce_float(raw.get("count") or raw.get("original_count"), default=0.0)
        remaining = _coerce_float(raw.get("remaining_count"), default=original)
        matched = max(0.0, original - remaining)
        return OpenOrder(
            order_id=order_id,
            token_id=ticker,
            side=side,
            price=price,
            size_matched=matched or None,
            original_size=original or None,
            status=str(raw.get("status")) if raw.get("status") is not None else None,
        )

    def _parse_fill(self, raw: dict[str, Any], default_ticker: str) -> Trade:
        trade_id = str(raw.get("trade_id") or raw.get("id") or "")
        order_id = str(raw.get("order_id") or "")
        ticker = str(raw.get("ticker") or default_ticker)
        action = str(raw.get("action") or raw.get("side") or "buy").lower()
        side = Side.BUY if action == "buy" else Side.SELL
        price = _prob(raw.get("no_price") or raw.get("price"))
        size = _coerce_float(raw.get("count"), default=0.0)
        fee_cents = _coerce_float(raw.get("fee") or raw.get("maker_fee") or raw.get("taker_fee"), default=0.0)
        timestamp = raw.get("created_time") or raw.get("ts")
        return Trade(
            trade_id=trade_id,
            order_id=order_id,
            token_id=ticker,
            side=side,
            price=price,
            size=size,
            fee=fee_cents / 100.0,
            timestamp=timestamp,
        )


def _no_best_bid_ask(book: _KalshiOrderBook) -> tuple[int | None, int | None]:
    """Return best NO bid and ask in cents.

    Kalshi's orderbook returns, for each side, (price_cents, size) pairs where
    ``price_cents`` is the price for buying that side. The best NO bid is the
    highest NO price; the best NO ask is derived from the best YES bid as
    ``100 - yes_bid`` (because selling a NO contract is equivalent to buying
    YES at the complementary price).
    """
    no_prices = [p for p, _ in book.no_bids]
    best_no_bid = max(no_prices) if no_prices else None
    yes_prices = [p for p, _ in book.yes_bids]
    best_yes_bid = max(yes_prices) if yes_prices else None
    best_no_ask = (100 - best_yes_bid) if best_yes_bid is not None else None
    return best_no_bid, best_no_ask


def _no_book_as_levels(
    book: _KalshiOrderBook,
) -> tuple[list[OrderBookLevel], list[OrderBookLevel]]:
    """Project Kalshi's two-sided book onto a NO-perspective bid/ask ladder."""
    bids = [
        OrderBookLevel(price=price / 100.0, size=float(size))
        for price, size in sorted(book.no_bids, key=lambda kv: kv[0], reverse=True)
    ]
    asks = [
        OrderBookLevel(price=(100 - price) / 100.0, size=float(size))
        for price, size in sorted(book.yes_bids, key=lambda kv: kv[0], reverse=True)
    ]
    return bids, asks


def _estimate_contract_count(order: MarketOrderIntent) -> int:
    if order.side == Side.SELL:
        return max(1, int(round(order.amount)))
    ref = order.reference_price
    if not ref or ref <= 0:
        return max(1, int(round(order.amount)))
    # ``amount`` is USD for a BUY; Kalshi prices are in dollars per contract.
    # Compute in integer cents to avoid binary-float rounding (10/0.4 == 24.999…).
    amount_cents = int(round(order.amount * 100))
    ref_cents = int(round(ref * 100))
    if ref_cents <= 0:
        return max(1, int(round(order.amount)))
    return max(1, amount_cents // ref_cents)


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
