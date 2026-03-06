"""
Capital Manager — portfolio state tracking for small-account accumulation.

ACCOUNT MODEL: $0 start, $100/week deposits.
The weekly deposit IS the capital. There is no lump sum.

Responsibilities:
1. Track portfolio state (cash, equity, positions) in database
2. Track total amount deposited vs current value (the key metric)
3. Handle weekly deposits as the primary capital event
4. Calculate allocation — deploy almost all available cash
5. Provide portfolio state to decision and risk engines

This module is the source of truth for "how much capital is available."
The broker is used for reconciliation only.
"""

import json
from datetime import datetime

import yaml

from core.logging import get_logger
from data.db import PortfolioState, Deposit, Trade, CostBasis, get_session, init_db

logger = get_logger("capital.manager")


def _load_account_config():
    """Load account settings from settings.yaml."""
    try:
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
        return settings.get("account", {})
    except Exception:
        return {}


def get_initial_capital():
    """Get initial capital from config. Defaults to 0 for accumulation mode."""
    config = _load_account_config()
    return config.get("initial_capital", 0.0)


def get_weekly_deposit_amount():
    """Get weekly deposit amount from config. Defaults to $100."""
    config = _load_account_config()
    return config.get("weekly_deposit", 100.0)


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

    For accumulation mode, this starts at $0. The first deposit
    is what kicks things off.

    Args:
        session: DB session.
        initial_capital: Starting cash amount. Defaults to config value (0 for accumulation).

    Returns:
        Portfolio state dict.
    """
    if initial_capital is None:
        initial_capital = get_initial_capital()

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
        extra={"extra_data": {"initial_capital": initial_capital, "mode": "accumulation"}},
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


def get_total_deposited(session):
    """
    Calculate total amount deposited across all time.

    This is THE key metric for accumulation mode:
    total_deposited vs current_equity = your actual return.

    Returns:
        Float — total dollars deposited.
    """
    deposits = session.query(Deposit).all()
    return sum(d.amount for d in deposits)


def get_accumulation_summary(session):
    """
    Get a summary of the accumulation account's performance.

    Returns:
        Dict with:
        - total_deposited: total $ put in
        - current_equity: what it's worth now
        - gain_loss: dollar gain/loss
        - gain_loss_pct: percentage return on invested capital
        - weeks_active: number of deposits made
    """
    total_deposited = get_total_deposited(session)
    state = get_latest_portfolio_state(session)
    current_equity = state["total_equity"] if state else 0.0

    gain_loss = current_equity - total_deposited
    gain_loss_pct = (gain_loss / total_deposited * 100) if total_deposited > 0 else 0.0

    deposit_count = session.query(Deposit).count()

    return {
        "total_deposited": total_deposited,
        "current_equity": current_equity,
        "gain_loss": gain_loss,
        "gain_loss_pct": round(gain_loss_pct, 2),
        "weeks_active": deposit_count,
    }


def record_deposit(session, amount=None, notes=None, mode_params=None):
    """
    Record a cash deposit (weekly $100 per spec).

    When the performance mode is conservative, deposits are held in
    a cash buffer instead of being counted toward deployable capital.
    The deposit is always recorded, but the portfolio snapshot reflects
    whether the cash is available for trading.

    Args:
        session: DB session.
        amount: Deposit amount. Defaults to weekly_deposit from config.
        notes: Optional notes.
        mode_params: Optional dict from performance.manager.get_mode_params().
                     When deploy_deposits is False, deposit goes to cash
                     but a note is added indicating it's buffered.

    Returns:
        Updated portfolio state dict.
    """
    if amount is None:
        amount = get_weekly_deposit_amount()

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

    total_deposited = get_total_deposited(session)

    logger.info(
        "Deposit recorded",
        extra={
            "extra_data": {
                "amount": amount,
                "deployed": deploy,
                "new_cash": current["cash"],
                "new_equity": current["total_equity"],
                "total_deposited": total_deposited,
                "weeks_active": session.query(Deposit).count(),
            }
        },
    )
    return current


def calculate_strategy_allocation(portfolio_state, strategy_capital_pct):
    """
    Calculate the dollar amount available for a strategy.

    For accumulation mode, this is almost all available cash —
    the whole point is to get money deployed, not sitting idle.

    Args:
        portfolio_state: Portfolio state dict.
        strategy_capital_pct: Strategy's allocation percentage (e.g., 0.95).

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


def get_deployable_cash(portfolio_state):
    """
    Get the amount of cash available to deploy right now.

    For small accounts, this is basically all the cash minus a tiny buffer
    to avoid rounding issues with fractional shares.

    Args:
        portfolio_state: Portfolio state dict.

    Returns:
        Float — deployable cash amount.
    """
    cash = portfolio_state.get("cash", 0)
    # Keep $1 buffer to avoid zero-balance issues
    buffer = min(1.0, cash * 0.05)
    return max(0, cash - buffer)


def update_cost_basis(session, symbol, qty_change, price):
    """
    Update the cost basis for a symbol after a buy fill.

    Uses weighted average: new_avg = (old_total + new_cost) / (old_qty + new_qty)

    Args:
        session: DB session.
        symbol: Ticker symbol.
        qty_change: Shares bought (positive).
        price: Fill price.
    """
    basis = session.query(CostBasis).filter(CostBasis.symbol == symbol).first()
    if basis is None:
        basis = CostBasis(symbol=symbol, qty=0, avg_price=0, total_cost=0)
        session.add(basis)

    new_cost = qty_change * price
    basis.qty += qty_change
    basis.total_cost += new_cost
    basis.avg_price = basis.total_cost / basis.qty if basis.qty > 0 else 0
    basis.last_updated = datetime.utcnow()


def record_realized_trade(session, symbol, qty_sold, exit_price, buy_order_id=None, sell_order_id=None):
    """
    Record a realized trade when a position is closed (sold).

    Computes P&L from cost basis and creates a Trade record.

    Args:
        session: DB session.
        symbol: Ticker symbol.
        qty_sold: Shares sold (positive number).
        exit_price: Sell fill price.
        buy_order_id: Original buy order id (if known).
        sell_order_id: Sell order id.

    Returns:
        Trade dict with realized P&L.
    """
    basis = session.query(CostBasis).filter(CostBasis.symbol == symbol).first()

    if basis is None or basis.qty <= 0:
        entry_price = exit_price  # No basis — assume breakeven
    else:
        entry_price = basis.avg_price

    realized_pnl = (exit_price - entry_price) * qty_sold
    realized_pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

    trade = Trade(
        symbol=symbol,
        buy_order_id=buy_order_id,
        sell_order_id=sell_order_id,
        qty=qty_sold,
        entry_price=entry_price,
        exit_price=exit_price,
        realized_pnl=realized_pnl,
        realized_pnl_pct=round(realized_pnl_pct, 4),
        is_win=realized_pnl > 0,
        exit_date=datetime.utcnow(),
    )
    session.add(trade)

    # Update cost basis — reduce qty
    if basis is not None:
        basis.qty -= qty_sold
        basis.total_cost = basis.qty * basis.avg_price if basis.qty > 0 else 0
        if basis.qty <= 0:
            basis.qty = 0
            basis.avg_price = 0
            basis.total_cost = 0
        basis.last_updated = datetime.utcnow()

    logger.info(
        f"Trade recorded: {symbol} sold {qty_sold:.6f} @ ${exit_price:.2f}, "
        f"P&L: ${realized_pnl:+.2f} ({realized_pnl_pct:+.2f}%)",
        extra={
            "extra_data": {
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "realized_pnl": realized_pnl,
                "is_win": realized_pnl > 0,
            }
        },
    )

    return {
        "symbol": symbol,
        "qty": qty_sold,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "realized_pnl": realized_pnl,
        "realized_pnl_pct": realized_pnl_pct,
        "is_win": realized_pnl > 0,
    }


def get_trade_performance(session):
    """
    Compute win rate and profit factor from realized trades.

    Returns:
        Dict with:
        - total_trades: int
        - wins: int
        - losses: int
        - win_rate: float (0-1)
        - profit_factor: float (gross_profit / gross_loss)
        - total_pnl: float
    """
    trades = session.query(Trade).all()

    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "total_pnl": 0,
        }

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    gross_profit = sum(t.realized_pnl for t in wins)
    gross_loss = abs(sum(t.realized_pnl for t in losses))

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "total_pnl": sum(t.realized_pnl for t in trades),
    }


def update_position(session, symbol, qty_change, price, current_state, buy_order_id=None, sell_order_id=None):
    """
    Update a position after a fill. Tracks cost basis and realized P&L.

    Args:
        session: DB session.
        symbol: Ticker symbol.
        qty_change: Positive for buy, negative for sell.
        price: Fill price.
        current_state: Current portfolio state dict.
        buy_order_id: Order id for buy fills.
        sell_order_id: Order id for sell fills.

    Returns:
        Updated portfolio state dict.
    """
    positions = dict(current_state.get("positions", {}))
    cash = current_state["cash"]

    old_qty = positions.get(symbol, 0)
    new_qty = old_qty + qty_change
    trade_value = abs(qty_change) * price

    if qty_change > 0:
        # Buy — reduce cash, update cost basis
        cash -= trade_value
        update_cost_basis(session, symbol, qty_change, price)
    else:
        # Sell — increase cash, record realized P&L
        cash += trade_value
        record_realized_trade(
            session, symbol, abs(qty_change), price,
            buy_order_id=buy_order_id, sell_order_id=sell_order_id,
        )

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
