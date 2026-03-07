"""
Alpaca Broker Client — wraps the alpaca-py SDK for paper trading.

Responsibilities:
1. Submit market orders to Alpaca paper trading API
2. Query order status and fills
3. Fetch account info (cash, equity, buying power)
4. Fetch current positions from broker (for reconciliation)

This module is the ONLY place that talks to the Alpaca API.
All other modules go through this interface.
"""

import os
import time
from datetime import datetime

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

from core.logging import get_logger

load_dotenv()
logger = get_logger("execution.alpaca_broker")

# Module-level client (initialized lazily)
_client = None


def get_client():
    """Get or create the Alpaca TradingClient singleton."""
    global _client
    if _client is None:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )

        _client = TradingClient(api_key, secret_key, paper=True)
        logger.info("Alpaca TradingClient initialized (paper mode)")

    return _client


def get_account():
    """
    Fetch account info from Alpaca.

    Returns:
        Dict with keys: cash, equity, buying_power, status
    """
    client = get_client()
    account = client.get_account()

    info = {
        "cash": float(account.cash),
        "equity": float(account.equity),
        "buying_power": float(account.buying_power),
        "status": account.status.value if hasattr(account.status, 'value') else str(account.status),
        "currency": account.currency,
        "pattern_day_trader": account.pattern_day_trader,
    }

    logger.info(
        "Account info fetched",
        extra={"extra_data": info},
    )
    return info


def get_positions():
    """
    Fetch all open positions from Alpaca.

    Returns:
        Dict of {symbol: {"qty": float, "market_value": float,
                          "avg_entry": float, "unrealized_pl": float}}
    """
    client = get_client()
    positions = client.get_all_positions()

    result = {}
    for pos in positions:
        # Normalize crypto symbols: Alpaca returns "BTCUSD" but we use "BTC/USD"
        symbol = pos.symbol
        if hasattr(pos, 'asset_class') and str(getattr(pos, 'asset_class', '')) == 'crypto':
            # Convert BTCUSD → BTC/USD
            if '/' not in symbol and symbol.endswith('USD'):
                symbol = symbol[:-3] + '/USD'
        result[symbol] = {
            "qty": float(pos.qty),
            "market_value": float(pos.market_value),
            "avg_entry": float(pos.avg_entry_price),
            "unrealized_pl": float(pos.unrealized_pl),
            "current_price": float(pos.current_price),
        }

    logger.info(
        "Positions fetched",
        extra={"extra_data": {"count": len(result), "symbols": list(result.keys())}},
    )
    return result


def submit_market_order(symbol, qty, side):
    """
    Submit a market order to Alpaca paper trading.

    Supports fractional shares. If qty has decimals, uses fractional
    order mode. This is critical for small accounts — $100 can't buy
    a whole share of SPY.

    Args:
        symbol: Ticker symbol (e.g., "SPY").
        qty: Number of shares (float — fractional allowed).
        side: "buy" or "sell".

    Returns:
        Dict with order details including broker_order_id.
    """
    client = get_client()

    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

    # Crypto orders require GTC; equities use DAY
    is_crypto = "/" in symbol
    tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY

    # Use fractional qty if not a whole number
    qty_float = float(qty)
    if qty_float == int(qty_float) and not is_crypto:
        # Whole shares (equities only)
        request = MarketOrderRequest(
            symbol=symbol,
            qty=int(qty_float),
            side=order_side,
            time_in_force=tif,
        )
    else:
        # Fractional shares or crypto
        request = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty_float, 6),
            side=order_side,
            time_in_force=tif,
        )

    order = client.submit_order(request)

    result = {
        "broker_order_id": str(order.id),
        "symbol": order.symbol,
        "side": side,
        "qty": qty_float,
        "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
        "submitted_at": str(order.submitted_at),
        "type": "market",
    }

    logger.info(
        f"Order submitted: {side} {qty_float:.6f} {symbol}",
        extra={"extra_data": result},
    )
    return result


def get_order_status(broker_order_id):
    """
    Check the status of a specific order.

    Args:
        broker_order_id: The Alpaca order ID string.

    Returns:
        Dict with order status details.
    """
    client = get_client()
    order = client.get_order_by_id(broker_order_id)

    result = {
        "broker_order_id": str(order.id),
        "symbol": order.symbol,
        "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
        "qty": float(order.qty) if order.qty else 0,
        "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
        "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
        "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
        "submitted_at": str(order.submitted_at),
        "filled_at": str(order.filled_at) if order.filled_at else None,
    }
    return result


def get_recent_orders(limit=20):
    """
    Fetch recent orders from Alpaca.

    Args:
        limit: Max number of orders to return.

    Returns:
        List of order dicts.
    """
    client = get_client()
    request = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        limit=limit,
    )
    orders = client.get_orders(request)

    return [
        {
            "broker_order_id": str(o.id),
            "symbol": o.symbol,
            "side": o.side.value if hasattr(o.side, 'value') else str(o.side),
            "qty": float(o.qty) if o.qty else 0,
            "status": o.status.value if hasattr(o.status, 'value') else str(o.status),
            "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
            "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
        }
        for o in orders
    ]


def cancel_all_orders():
    """Cancel all open orders. Used by kill switch."""
    client = get_client()
    cancelled = client.cancel_orders()
    logger.warning(
        "All orders cancelled",
        extra={"extra_data": {"count": len(cancelled) if cancelled else 0}},
    )
    return cancelled


def _retry(func, max_retries=3, base_delay=2):
    """
    Retry a function with exponential backoff for transient network errors.

    Only retries on connection/timeout errors, NOT on API rejections
    (e.g., insufficient funds, invalid symbol).
    """
    from requests.exceptions import ConnectionError, Timeout, ProxyError

    for attempt in range(max_retries + 1):
        try:
            return func()
        except (ConnectionError, Timeout, ProxyError, OSError) as e:
            if attempt == max_retries:
                logger.error(
                    f"API call failed after {max_retries} retries",
                    extra={"extra_data": {"error": str(e)}},
                )
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"API call failed (attempt {attempt + 1}), retrying in {delay}s",
                extra={"extra_data": {"error": str(e)}},
            )
            time.sleep(delay)


def submit_market_order_with_retry(symbol, qty, side, max_retries=3):
    """Submit a market order with automatic retry on network errors."""
    return _retry(lambda: submit_market_order(symbol, qty, side), max_retries=max_retries)


def get_account_with_retry(max_retries=3):
    """Fetch account info with automatic retry on network errors."""
    return _retry(get_account, max_retries=max_retries)


def get_positions_with_retry(max_retries=3):
    """Fetch positions with automatic retry on network errors."""
    return _retry(get_positions, max_retries=max_retries)
