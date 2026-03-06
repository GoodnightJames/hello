"""
Capital Manager — portfolio state tracking and allocation.

Responsibilities:
1. Track portfolio state (cash, equity, positions) in database
2. Calculate allocation amounts per strategy
3. Handle weekly deposits
4. Provide portfolio state to decision and risk engines

This module is the source of truth for "how much capital is available."
The broker is used for reconciliation only.
"""

import json
from datetime import datetime

from core.logging import get_logger
from data.db import PortfolioState, Deposit, get_session, init_db

logger = get_logger("capital.manager")

# Default initial capital for paper trading
DEFAULT_INITIAL_CAPITAL = 500.0


def get_latest_portfolio_state(session):
    """
    Get the most recent portfolio snapshot from the database.

    Returns:
        Dict with keys: total_equity, cash, positions, date
        or None if no state exists.
    """
    state = (
        session.query(PortfolioState)
        .order_by(PortfolioState.date.desc())
        .first()
    )

    if state is None:
        return None

    positions = json.loads(state.positions_json) if state.positions_json else {}

    return {
        "id": state.id,
        "date": state.date,
        "cash": state.cash,
        "total_equity": state.total_equity,
        "positions": positions,
    }


def initialize_portfolio(session, initial_capital=None):
    """
    Create the initial portfolio state (first run or reset).

    Args:
        session: DB session.
        initial_capital: Starting cash amount.

    Returns:
        Portfolio state dict.
    """
    if initial_capital is None:
        initial_capital = DEFAULT_INITIAL_CAPITAL

    state = PortfolioState(
        date=datetime.utcnow(),
        cash=initial_capital,
        total_equity=initial_capital,
        positions_json=json.dumps({}),
    )
    session.add(state)
    session.commit()

    logger.info(
        "Portfolio initialized",
        extra={"extra_data": {"initial_capital": initial_capital}},
    )

    return {
        "id": state.id,
        "date": state.date,
        "cash": initial_capital,
        "total_equity": initial_capital,
        "positions": {},
    }


def get_or_create_portfolio(session, initial_capital=None):
    """Get current portfolio state, creating initial state if needed."""
    state = get_latest_portfolio_state(session)
    if state is None:
        state = initialize_portfolio(session, initial_capital)
    return state


def save_portfolio_snapshot(session, cash, positions, prices=None):
    """
    Save a new portfolio state snapshot.

    Args:
        session: DB session.
        cash: Current cash balance.
        positions: Dict of {symbol: quantity}.
        prices: Dict of {symbol: current_price} for equity calculation.
                If None, equity = cash (positions valued at 0).

    Returns:
        Portfolio state dict.
    """
    # Calculate total equity
    positions_value = 0.0
    if prices and positions:
        for symbol, qty in positions.items():
            if qty > 0 and symbol in prices:
                positions_value += qty * prices[symbol]

    total_equity = cash + positions_value

    state = PortfolioState(
        date=datetime.utcnow(),
        cash=cash,
        total_equity=total_equity,
        positions_json=json.dumps(positions),
    )
    session.add(state)
    session.commit()

    logger.info(
        "Portfolio snapshot saved",
        extra={
            "extra_data": {
                "cash": cash,
                "positions_value": positions_value,
                "total_equity": total_equity,
                "position_count": len([s for s, q in positions.items() if q > 0]),
            }
        },
    )

    return {
        "id": state.id,
        "date": state.date,
        "cash": cash,
        "total_equity": total_equity,
        "positions": positions,
    }


def record_deposit(session, amount, notes=None, mode_params=None):
    """
    Record a cash deposit (weekly $100 per spec).

    When the performance mode is conservative, deposits are held in
    a cash buffer instead of being counted toward deployable capital.
    The deposit is always recorded, but the portfolio snapshot reflects
    whether the cash is available for trading.

    Args:
        session: DB session.
        amount: Deposit amount.
        notes: Optional notes.
        mode_params: Optional dict from performance.manager.get_mode_params().
                     When deploy_deposits is False, deposit goes to cash
                     but a note is added indicating it's buffered.

    Returns:
        Updated portfolio state dict.
    """
    deploy = True
    if mode_params and not mode_params.get("deploy_deposits", True):
        deploy = False
        notes = (notes or "") + " [BUFFERED — conservative mode, not deployed]"

    deposit = Deposit(
        amount=amount,
        date=datetime.utcnow(),
        notes=notes,
    )
    session.add(deposit)

    # Update portfolio state
    current = get_latest_portfolio_state(session)
    if current is None:
        current = initialize_portfolio(session, amount)
    else:
        new_cash = current["cash"] + amount
        current = save_portfolio_snapshot(session, new_cash, current["positions"])

    logger.info(
        "Deposit recorded",
        extra={
            "extra_data": {
                "amount": amount,
                "deployed": deploy,
                "new_cash": current["cash"],
                "new_equity": current["total_equity"],
            }
        },
    )
    return current


def calculate_strategy_allocation(portfolio_state, strategy_capital_pct):
    """
    Calculate the dollar amount available for a strategy.

    Args:
        portfolio_state: Portfolio state dict.
        strategy_capital_pct: Strategy's allocation percentage (e.g., 0.60).

    Returns:
        Float — dollar amount allocated to the strategy.
    """
    total = portfolio_state.get("total_equity", 0)
    allocation = total * strategy_capital_pct

    logger.info(
        "Strategy allocation calculated",
        extra={
            "extra_data": {
                "total_equity": total,
                "capital_pct": strategy_capital_pct,
                "allocation": allocation,
            }
        },
    )
    return allocation


def update_position(session, symbol, qty_change, price, current_state):
    """
    Update a position after a fill.

    Args:
        session: DB session.
        symbol: Ticker symbol.
        qty_change: Positive for buy, negative for sell.
        price: Fill price.
        current_state: Current portfolio state dict.

    Returns:
        Updated portfolio state dict.
    """
    positions = dict(current_state.get("positions", {}))
    cash = current_state["cash"]

    old_qty = positions.get(symbol, 0)
    new_qty = old_qty + qty_change
    trade_value = abs(qty_change) * price

    if qty_change > 0:
        # Buy — reduce cash
        cash -= trade_value
    else:
        # Sell — increase cash
        cash += trade_value

    if new_qty <= 0:
        positions.pop(symbol, None)
    else:
        positions[symbol] = new_qty

    # Get current prices for equity calculation (use fill price as estimate)
    prices = {symbol: price}

    updated = save_portfolio_snapshot(session, cash, positions, prices)

    logger.info(
        f"Position updated: {symbol}",
        extra={
            "extra_data": {
                "symbol": symbol,
                "qty_change": qty_change,
                "price": price,
                "old_qty": old_qty,
                "new_qty": new_qty,
                "cash_after": cash,
            }
        },
    )
    return updated
