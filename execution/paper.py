"""
Paper Execution Engine — submits orders to Alpaca paper trading API.

Pipeline for each decision:
1. Get current portfolio state (from Alpaca account)
2. Run risk validation
3. Calculate share quantity
4. Submit market order to Alpaca
5. Poll for fill confirmation
6. Record order in local database
7. Update local portfolio state

This uses real Alpaca paper trading — orders execute against
the market simulation at realistic prices with proper fills.
"""

import time
from datetime import datetime

import yaml

from core.logging import get_logger
from data.db import get_session, init_db
from data.feature_store import get_price_history
from capital.manager import (
    get_or_create_portfolio,
    update_position,
    save_portfolio_snapshot,
)
from risk.enforcer import load_risk_params, validate_order
from execution.order_manager import (
    calculate_shares,
    create_order,
    mark_order_submitted,
    mark_order_filled,
    mark_order_rejected,
)
from execution.alpaca_broker import (
    submit_market_order_with_retry as submit_market_order,
    get_order_status,
    get_account_with_retry as get_account,
    get_positions_with_retry as get_alpaca_positions,
)

logger = get_logger("execution.paper")

# Max time to wait for an order to fill (seconds)
ORDER_FILL_TIMEOUT = 30
ORDER_POLL_INTERVAL = 1


def get_latest_prices(symbols, session=None):
    """
    Get the most recent close price for each symbol from local data.
    Used for pre-trade sizing only — actual fills come from Alpaca.
    """
    prices_df = get_price_history(symbols, lookback_days=5, session=session)
    if prices_df.empty:
        return {}

    latest = prices_df.iloc[-1]
    return {symbol: float(latest[symbol]) for symbol in latest.index if latest[symbol] > 0}


def sync_portfolio_from_alpaca(session):
    """
    Sync local portfolio state from Alpaca account.

    Returns:
        Portfolio state dict matching the format expected by other modules.
    """
    account = get_account()
    positions = get_alpaca_positions()

    # Convert to local format: {symbol: qty}
    position_qtys = {sym: int(pos["qty"]) for sym, pos in positions.items()}
    prices = {sym: pos["current_price"] for sym, pos in positions.items()}

    snapshot = save_portfolio_snapshot(
        session,
        cash=account["cash"],
        positions=position_qtys,
        prices=prices,
    )

    logger.info(
        "Portfolio synced from Alpaca",
        extra={
            "extra_data": {
                "alpaca_equity": account["equity"],
                "alpaca_cash": account["cash"],
                "local_equity": snapshot["total_equity"],
                "positions": len(position_qtys),
            }
        },
    )
    return snapshot


def wait_for_fill(broker_order_id, timeout=ORDER_FILL_TIMEOUT):
    """
    Poll Alpaca for order fill status.

    Returns:
        Order status dict, or None if timeout.
    """
    elapsed = 0
    while elapsed < timeout:
        status = get_order_status(broker_order_id)
        if status["status"] in ("filled", "partially_filled"):
            return status
        if status["status"] in ("canceled", "cancelled", "expired", "rejected"):
            return status
        time.sleep(ORDER_POLL_INTERVAL)
        elapsed += ORDER_POLL_INTERVAL

    logger.warning(
        f"Order fill timeout after {timeout}s",
        extra={"extra_data": {"broker_order_id": broker_order_id}},
    )
    return get_order_status(broker_order_id)


def execute_paper_decisions(decisions, config_path="config/settings.yaml"):
    """
    Execute a list of decisions via Alpaca paper trading API.

    Args:
        decisions: List of decision dicts from decision engine.
        config_path: Path to settings config.

    Returns:
        List of execution result dicts.
    """
    logger.info(
        "Paper execution starting (Alpaca)",
        extra={"extra_data": {"decision_count": len(decisions)}},
    )

    init_db()
    session = get_session()
    risk_params = load_risk_params()
    results = []

    try:
        # Sync portfolio from Alpaca before executing
        portfolio = sync_portfolio_from_alpaca(session)

        # Collect symbols we need prices for (pre-trade sizing)
        symbols = list(set(d["symbol"] for d in decisions if d.get("action") in ("BUY", "SELL")))
        if not symbols:
            logger.info("No actionable decisions (all HOLD/SKIP)")
            return results

        prices = get_latest_prices(symbols, session=session)
        if not prices:
            logger.warning("No price data available for sizing — using Alpaca positions for sells")

        for decision in decisions:
            action = decision.get("action")
            symbol = decision.get("symbol")

            # Skip non-actionable decisions
            if action in ("SKIP", "HOLD"):
                results.append({
                    "symbol": symbol,
                    "action": action,
                    "status": "skipped",
                    "reason": decision.get("reason", ""),
                })
                continue

            # Risk validation
            risk_result = validate_order(session, risk_params, decision, portfolio)
            if not risk_result["approved"]:
                logger.info(
                    f"Order rejected by risk: {symbol}",
                    extra={"extra_data": risk_result},
                )
                results.append({
                    "symbol": symbol,
                    "action": action,
                    "status": "rejected",
                    "reason": risk_result["reason"],
                })
                continue

            if action == "BUY":
                result = _execute_alpaca_buy(
                    session, decision, portfolio, prices, risk_result
                )
            elif action == "SELL":
                result = _execute_alpaca_sell(session, decision, portfolio)
            else:
                result = {"symbol": symbol, "action": action, "status": "unknown"}

            results.append(result)

            # Update portfolio locally from fill data (no API call)
            # The position was already updated in the buy/sell handler
            portfolio = get_or_create_portfolio(session)

        # Single final sync from Alpaca after all orders are done
        sync_portfolio_from_alpaca(session)
        session.commit()

        logger.info(
            "Paper execution complete",
            extra={
                "extra_data": {
                    "results": [
                        {"symbol": r["symbol"], "action": r["action"], "status": r["status"]}
                        for r in results
                    ]
                }
            },
        )
        return results

    except Exception as e:
        session.rollback()
        logger.error(
            "Paper execution failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )
        raise
    finally:
        session.close()


def _execute_alpaca_buy(session, decision, portfolio, prices, risk_result):
    """Execute a BUY order via Alpaca paper trading."""
    symbol = decision["symbol"]
    multiplier = decision.get("position_multiplier", 1.0)

    price = prices.get(symbol)
    if price is None or price <= 0:
        logger.warning(f"No price for {symbol} — skipping buy")
        return {
            "symbol": symbol,
            "action": "BUY",
            "status": "skipped",
            "reason": "No price data available for sizing",
        }

    # Calculate allocation — deploy available cash, scaled by signal strength.
    # max_trade_size from risk enforcer reflects the risk mode's deployment %.
    # Signal strength (0.1 to 2.0) scales the allocation: strong momentum = bigger position.
    from capital.manager import get_deployable_cash
    deployable = get_deployable_cash(portfolio)
    max_trade = risk_result.get("max_trade_size", deployable)
    signal_strength = decision.get("signal_strength", 1.0)
    # Clamp strength between 0.5 and 1.0 for sizing (don't go below 50% or above 100% of max)
    strength_factor = max(0.5, min(1.0, signal_strength))
    allocation = min(deployable, max_trade) * multiplier * strength_factor

    shares = calculate_shares(allocation, price, fractional=True, min_notional=1.0)
    if shares <= 0:
        logger.info(f"Insufficient funds for {symbol} at ${price:.2f}")
        return {
            "symbol": symbol,
            "action": "BUY",
            "status": "skipped",
            "reason": f"Insufficient funds: ${allocation:.2f} for {symbol} at ${price:.2f}",
        }

    # Create local order record
    order = create_order(session, decision, shares)

    # Submit to Alpaca
    try:
        alpaca_order = submit_market_order(symbol, shares, "buy")
        broker_order_id = alpaca_order["broker_order_id"]
        mark_order_submitted(session, order["id"], broker_order_id=broker_order_id)

        # Wait for fill
        fill_status = wait_for_fill(broker_order_id)

        if fill_status and fill_status["status"] == "filled":
            filled_price = fill_status["filled_avg_price"] or price
            filled_qty = float(fill_status["filled_qty"]) or shares

            mark_order_filled(session, order["id"], filled_price=filled_price, filled_qty=filled_qty)
            update_position(session, symbol, qty_change=filled_qty, price=filled_price, current_state=portfolio)

            total_cost = filled_qty * filled_price
            logger.info(
                f"Alpaca BUY filled: {filled_qty:.6f} {symbol} @ ${filled_price:.2f} = ${total_cost:.2f}",
                extra={"extra_data": {"order_id": order["id"], "broker_order_id": broker_order_id}},
            )

            return {
                "symbol": symbol,
                "action": "BUY",
                "status": "filled",
                "shares": filled_qty,
                "price": filled_price,
                "total_cost": total_cost,
                "order_id": order["id"],
                "broker_order_id": broker_order_id,
            }
        else:
            status_str = fill_status["status"] if fill_status else "timeout"
            mark_order_rejected(session, order["id"], reason=f"Alpaca: {status_str}")
            return {
                "symbol": symbol,
                "action": "BUY",
                "status": "rejected",
                "reason": f"Order not filled: {status_str}",
                "order_id": order["id"],
            }

    except Exception as e:
        mark_order_rejected(session, order["id"], reason=str(e))
        logger.error(
            f"Alpaca BUY failed: {symbol}",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )
        return {
            "symbol": symbol,
            "action": "BUY",
            "status": "rejected",
            "reason": f"Alpaca API error: {str(e)}",
            "order_id": order["id"],
        }


def _execute_alpaca_sell(session, decision, portfolio):
    """Execute a SELL order via Alpaca paper trading."""
    symbol = decision["symbol"]
    positions = portfolio.get("positions", {})
    current_qty = positions.get(symbol, 0)

    if current_qty <= 0:
        logger.info(f"No position in {symbol} to sell")
        return {
            "symbol": symbol,
            "action": "SELL",
            "status": "skipped",
            "reason": "No position to sell",
        }

    shares = float(current_qty)
    order = create_order(session, decision, shares)

    try:
        alpaca_order = submit_market_order(symbol, shares, "sell")
        broker_order_id = alpaca_order["broker_order_id"]
        mark_order_submitted(session, order["id"], broker_order_id=broker_order_id)

        # Wait for fill
        fill_status = wait_for_fill(broker_order_id)

        if fill_status and fill_status["status"] == "filled":
            filled_price = fill_status["filled_avg_price"] or 0
            filled_qty = float(fill_status["filled_qty"]) or shares

            mark_order_filled(session, order["id"], filled_price=filled_price, filled_qty=filled_qty)
            update_position(session, symbol, qty_change=-filled_qty, price=filled_price, current_state=portfolio)

            total_proceeds = filled_qty * filled_price
            logger.info(
                f"Alpaca SELL filled: {filled_qty} {symbol} @ ${filled_price:.2f} = ${total_proceeds:.2f}",
                extra={"extra_data": {"order_id": order["id"], "broker_order_id": broker_order_id}},
            )

            return {
                "symbol": symbol,
                "action": "SELL",
                "status": "filled",
                "shares": filled_qty,
                "price": filled_price,
                "total_proceeds": total_proceeds,
                "order_id": order["id"],
                "broker_order_id": broker_order_id,
            }
        else:
            status_str = fill_status["status"] if fill_status else "timeout"
            mark_order_rejected(session, order["id"], reason=f"Alpaca: {status_str}")
            return {
                "symbol": symbol,
                "action": "SELL",
                "status": "rejected",
                "reason": f"Order not filled: {status_str}",
                "order_id": order["id"],
            }

    except Exception as e:
        mark_order_rejected(session, order["id"], reason=str(e))
        logger.error(
            f"Alpaca SELL failed: {symbol}",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )
        return {
            "symbol": symbol,
            "action": "SELL",
            "status": "rejected",
            "reason": f"Alpaca API error: {str(e)}",
            "order_id": order["id"],
        }


if __name__ == "__main__":
    # Quick connectivity test — fetches account info
    from execution.alpaca_broker import get_account, get_positions
    print("Testing Alpaca connection...")
    acct = get_account()
    print(f"  Account status: {acct['status']}")
    print(f"  Cash: ${acct['cash']:,.2f}")
    print(f"  Equity: ${acct['equity']:,.2f}")
    print(f"  Buying Power: ${acct['buying_power']:,.2f}")
    pos = get_positions()
    print(f"  Open positions: {len(pos)}")
    for sym, p in pos.items():
        print(f"    {sym}: {p['qty']} shares @ ${p['current_price']:.2f}")
