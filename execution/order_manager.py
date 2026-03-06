"""
Order Manager — converts approved decisions into orders.

Pipeline:
1. Take a risk-approved decision
2. Calculate share quantity from dollar allocation
3. Create an Order record in the database
4. Submit to the execution backend (paper or live)

This module does NOT decide what to trade — it only handles the
mechanical conversion of decisions into orders.
"""

import json
from datetime import datetime

from core.logging import get_logger
from data.db import Order, get_session

logger = get_logger("execution.order_manager")


def calculate_shares(dollar_amount, price, fractional=True, min_notional=1.0):
    """
    Convert a dollar allocation to shares (fractional or whole).

    Alpaca supports fractional shares down to $1. For a $100/week
    account, fractional shares are essential — $100 can't buy a
    whole share of SPY at $550+.

    Args:
        dollar_amount: Dollar amount to invest.
        price: Current price per share.
        fractional: If True, return fractional qty (default True).
        min_notional: Minimum dollar value of trade (default $1).

    Returns:
        Float number of shares (0 if below min_notional).
    """
    if price <= 0 or dollar_amount <= 0:
        return 0

    if dollar_amount < min_notional:
        return 0

    if fractional:
        # Round to 6 decimal places (Alpaca's precision)
        shares = round(dollar_amount / price, 6)
    else:
        shares = int(dollar_amount / price)
        if shares < 1:
            return 0

    return shares


def create_order(session, decision, qty, order_type="market"):
    """
    Create an order record in the database.

    Args:
        session: DB session.
        decision: Decision dict from decision engine.
        qty: Number of shares.
        order_type: "market" or "limit".

    Returns:
        Order dict with database id.
    """
    action = decision.get("action", "")
    side = "buy" if action == "BUY" else "sell"
    symbol = decision["symbol"]

    order = Order(
        decision_id=decision.get("signal_id"),
        symbol=symbol,
        side=side,
        qty=float(qty),
        order_type=order_type,
        status="pending",
        submitted_at=None,
        filled_at=None,
        filled_price=None,
        filled_qty=None,
    )
    session.add(order)
    session.flush()  # Get the id

    order_dict = {
        "id": order.id,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "order_type": order_type,
        "status": "pending",
        "decision_id": decision.get("signal_id"),
    }

    logger.info(
        f"Order created: {side} {qty} {symbol}",
        extra={"extra_data": order_dict},
    )
    return order_dict


def mark_order_submitted(session, order_id, broker_order_id=None):
    """Mark an order as submitted to the broker."""
    order = session.query(Order).filter(Order.id == order_id).first()
    if order:
        order.status = "submitted"
        order.submitted_at = datetime.utcnow()
        if broker_order_id:
            order.broker_order_id = broker_order_id
        logger.info(
            f"Order submitted: #{order_id}",
            extra={"extra_data": {"broker_order_id": broker_order_id}},
        )


def mark_order_filled(session, order_id, filled_price, filled_qty):
    """Mark an order as filled."""
    order = session.query(Order).filter(Order.id == order_id).first()
    if order:
        order.status = "filled"
        order.filled_at = datetime.utcnow()
        order.filled_price = filled_price
        order.filled_qty = filled_qty
        logger.info(
            f"Order filled: #{order_id} {order.side} {filled_qty} {order.symbol} @ ${filled_price:.2f}",
            extra={
                "extra_data": {
                    "order_id": order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "filled_price": filled_price,
                    "filled_qty": filled_qty,
                }
            },
        )


def mark_order_cancelled(session, order_id, reason=""):
    """Mark an order as cancelled."""
    order = session.query(Order).filter(Order.id == order_id).first()
    if order:
        order.status = "cancelled"
        logger.info(
            f"Order cancelled: #{order_id}",
            extra={"extra_data": {"reason": reason}},
        )


def mark_order_rejected(session, order_id, reason=""):
    """Mark an order as rejected (risk or broker rejection)."""
    order = session.query(Order).filter(Order.id == order_id).first()
    if order:
        order.status = "rejected"
        logger.info(
            f"Order rejected: #{order_id}",
            extra={"extra_data": {"reason": reason}},
        )


def get_open_orders(session):
    """Get all orders that are pending or submitted (not yet filled/cancelled)."""
    return (
        session.query(Order)
        .filter(Order.status.in_(["pending", "submitted"]))
        .all()
    )


def get_filled_orders_today(session):
    """Get all orders filled today."""
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return (
        session.query(Order)
        .filter(Order.status == "filled", Order.filled_at >= today)
        .all()
    )
