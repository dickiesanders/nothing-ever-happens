"""Quick Kalshi balance check.

Hits ``GET /trade-api/v2/portfolio/balance`` using the same RSA-signing path
the bot uses in live mode and prints the balance in dollars. Useful for
verifying demo credentials and network reachability before running the bot.

Usage:
    python -m scripts.kalshi_balance                    # demo env (default)
    python -m scripts.kalshi_balance --host prod        # production
    python -m scripts.kalshi_balance --host https://... # custom URL

Required env vars:
    KALSHI_API_KEY_ID
    KALSHI_PRIVATE_KEY_PATH        (or KALSHI_PRIVATE_KEY_PEM inline)
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from bot.config import ExchangeConfig, _env_optional, _load_kalshi_private_key_pem
from bot.exchange.kalshi import KalshiExchangeClient


HOST_ALIASES = {
    "demo": "https://demo-api.kalshi.co",
    "prod": "https://api.elections.kalshi.com",
    "production": "https://api.elections.kalshi.com",
}


def _resolve_host(raw: str) -> str:
    alias = HOST_ALIASES.get(raw.lower())
    return alias if alias else raw


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Print the Kalshi cash balance.")
    parser.add_argument(
        "--host",
        default="demo",
        help="Base host or alias: 'demo' (default), 'prod', or a full URL.",
    )
    args = parser.parse_args()

    host = _resolve_host(args.host)
    api_key_id = _env_optional("KALSHI_API_KEY_ID")
    pem = _load_kalshi_private_key_pem()

    if not api_key_id or not pem:
        print(
            "ERROR: set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH "
            "(or KALSHI_PRIVATE_KEY_PEM) in .env",
            file=sys.stderr,
        )
        return 1

    config = ExchangeConfig(
        host=host,
        chain_id=0,
        signature_type=0,
        private_key=None,
        funder_address=None,
        live_send_enabled=True,
        venue="kalshi",
        kalshi_api_key_id=api_key_id,
        kalshi_private_key_pem=pem,
    )

    try:
        client = KalshiExchangeClient(config, allow_trading=True)
        balance = client.get_collateral_balance()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"host:     {host}")
    print(f"key id:   {api_key_id}")
    print(f"balance:  ${balance:,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
