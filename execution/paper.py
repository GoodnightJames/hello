"""
Paper Execution Engine — simulated order fills for paper trading.

In paper mode:
- BUY orders fill at the latest close price
- SELL orders fill at the latest close price
- All fills are instant (no slippage, no partial fills)
- Portfolio state is updated after each fill

This runs during the paper trading phase before live execution is enabled.
"""

import json
from datetime import datetime

import yaml

from core.logging import get_logger
from data.db import get_session, init_db
from data.feature_store import get_price_history
from capital.manager import (
    get_or_create_portfolio,
    update_position,
    calculate_strategy_allocation,
)
from risk.enforcer import load_risk_params, validate_order
from execution.order_manager import (
    calculate_shares,
    create_order,
    mark_order_submitted,
    mark_order_filled,
    mark_order_rejected,
)

logger = get_logger("execution.paper")


def get_latest_prices(symbols, session=None):
    """
    Get the most recent close price for each symbol.

    Args:
        symbols: List of ticker symbols.
        session: DB session.

    Returns:
        Dict of {symbol: latest_close_price}.
    """
    prices_df = get_price_history(symbols, lookback_days=5, session=session)
    if prices_df.empty:
        return {}

    latest = prices_df.iloc[-1]
    return {symbol: float(latest[symbol]) for symbol in latest.index if latest[symbol] > 0}


def execute_paper_decisions(decisions, config_path="config/settings.yaml"):
    """
    Execute a list of decisions in paper trading mode.

    Pipeline for each decision:
    1. Get current portfolio state
    2. Get latest prices
    3. Run risk validation
    4. Calculate share quantity
    5. Create order
    6. Simulate fill at latest close
    7. Update portfolio state

    Args:
        decisions: List of decision dicts from decision engine.
        config_path: Path to settings config.

    Returns:
        List of execution result dicts.
    """
    logger.info(
        "Paper execution starting",
        extra={"extra_data": {"decision_count": len(decisions)}},
    )

    init_db()
    session = get_session()
    risk_params = load_risk_params()
    results = []

    try:
        portfolio = get_or_create_portfolio(session)

        # Collect symbols we need prices for
        symbols = list(set(d["symbol"] for d in decisions if d.get("action") in ("BUY", "SELL")))
        if not symbols:
            logger.info("No actionable decisions (all HOLD/SKIP)")
            return results

        prices = get_latest_prices(symbols, session=session)
        if not prices:
            logger.warning("No price data available for paper execution")
            return results

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

            price = prices.get(symbol)
            if price is None or price <= 0:
                logger.warning(f"No price for {symbol} — skipping")
                results.append({
                    "symbol": symbol,
                    "action": action,
                    "status": "skipped",
                    "reason": "No price data available",
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
                result = _execute_paper_buy(
                    session, decision, portfolio, price, risk_result, risk_params
                )
            elif action == "SELL":
                result = _execute_paper_sell(session, decision, portfolio, price)
            else:
                result = {"symbol": symbol, "action": action, "status": "unknown"}

            results.append(result)

            # Refresh portfolio state after each fill
            portfolio = get_or_create_portfolio(session)

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


def _execute_paper_buy(session, decision, portfolio, price, risk_result, risk_params):
    """Execute a paper BUY order."""
    symbol = decision["symbol"]
    multiplier = decision.get("position_multiplier", 1.0)

    # Calculate allocation: strategy allocation * regime multiplier
    # Capped by risk max trade size
    strategy_allocation = portfolio["cash"] * 0.90  # Keep 10% cash buffer
    max_trade = risk_result.get("max_trade_size", strategy_allocation)
    allocation = min(strategy_allocation, max_trade) * multiplier

    shares = calculate_shares(allocation, price)
    if shares <= 0:
        logger.info(f"Insufficient funds for {symbol} at ${price:.2f}")
        return {
            "symbol": symbol,
            "action": "BUY",
            "status": "skipped",
            "reason": f"Insufficient funds: ${allocation:.2f} for {symbol} at ${price:.2f}",
        }

    # Create and fill order
    order = create_order(session, decision, shares)
    mark_order_submitted(session, order["id"], broker_order_id=f"PAPER-{order['id']}")
    mark_order_filled(session, order["id"], filled_price=price, filled_qty=shares)

    # Update portfolio
    update_position(session, symbol, qty_change=shares, price=price, current_state=portfolio)

    total_cost = shares * price
    logger.info(
        f"Paper BUY filled: {shares} {symbol} @ ${price:.2f} = ${total_cost:.2f}",
        extra={
            "extra_data": {
                "order_id": order["id"],
                "shares": shares,
                "price": price,
                "total_cost": total_cost,
            }
        },
    )

    return {
        "symbol": symbol,
        "action": "BUY",
        "status": "filled",
        "shares": shares,
        "price": price,
        "total_cost": total_cost,
        "order_id": order["id"],
    }


def _execute_paper_sell(session, decision, portfolio, price):
    """Execute a paper SELL order."""
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

    # Sell entire position
    shares = current_qty
    order = create_order(session, decision, shares)
    mark_order_submitted(session, order["id"], broker_order_id=f"PAPER-{order['id']}")
    mark_order_filled(session, order["id"], filled_price=price, filled_qty=shares)

    # Update portfolio
    update_position(session, symbol, qty_change=-shares, price=price, current_state=portfolio)

    total_proceeds = shares * price
    logger.info(
        f"Paper SELL filled: {shares} {symbol} @ ${price:.2f} = ${total_proceeds:.2f}",
        extra={
            "extra_data": {
                "order_id": order["id"],
                "shares": shares,
                "price": price,
                "total_proceeds": total_proceeds,
            }
        },
    )

    return {
        "symbol": symbol,
        "action": "SELL",
        "status": "filled",
        "shares": shares,
        "price": price,
        "total_proceeds": total_proceeds,
        "order_id": order["id"],
    }


if __name__ == "__main__":
    # For manual testing
    test_decisions = [
        {
            "symbol": "SPY",
            "action": "BUY",
            "signal_type": "BUY",
            "signal_id": None,
            "position_multiplier": 1.0,
            "risk_approved": True,
            "reason": "Test BUY",
        }
    ]
    results = execute_paper_decisions(test_decisions)
    for r in results:
        print(f"  {r['action']:5s} {r['symbol']:5s} — {r['status']}")
