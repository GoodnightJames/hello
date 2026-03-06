"""
Database schema and initialization for the Trading Engine.

Uses SQLAlchemy ORM with SQLite backend. All tables defined here
serve as the local ground truth for decisions, orders, and state.
Broker state is used for reconciliation only.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()


class Base(DeclarativeBase):
    pass


class DailyBar(Base):
    """Daily OHLCV price data for all symbols in universe."""

    __tablename__ = "daily_bars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    vwap = Column(Float, nullable=True)
    trade_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_symbol_date"),
        Index("ix_daily_bars_date", "date"),
        Index("ix_daily_bars_symbol", "symbol"),
    )


class Signal(Base):
    """Strategy-generated signals."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(50), nullable=False)
    symbol = Column(String(10), nullable=False)
    date = Column(DateTime, nullable=False)
    signal_type = Column(String(20), nullable=False)  # BUY, SELL, HOLD
    score = Column(Float, nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("ix_signals_date", "date"),)


class Decision(Base):
    """Decision engine outputs — every decision logged including SKIPs."""

    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(50), nullable=False)
    symbol = Column(String(10), nullable=False)
    date = Column(DateTime, nullable=False)
    action = Column(String(10), nullable=False)  # BUY, SELL, HOLD, SKIP
    reason = Column(Text, nullable=True)
    signal_id = Column(Integer, nullable=True)
    risk_approved = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("ix_decisions_date", "date"),)


class Order(Base):
    """Orders submitted to broker and their fill status."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(Integer, nullable=True)
    broker_order_id = Column(String(100), nullable=True)
    symbol = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)  # buy, sell
    qty = Column(Float, nullable=False)
    order_type = Column(String(20), nullable=False)  # market, limit
    status = Column(String(20), nullable=False)  # submitted, filled, cancelled, rejected
    submitted_at = Column(DateTime, nullable=True)
    filled_at = Column(DateTime, nullable=True)
    filled_price = Column(Float, nullable=True)
    filled_qty = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("ix_orders_status", "status"),)


class RiskEvent(Base):
    """Risk engine events — throttles, shutdowns, kill switch activations."""

    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False)  # daily_loss_limit, consecutive_loss, vix_throttle, kill_switch
    severity = Column(String(10), nullable=False)  # INFO, WARNING, CRITICAL
    details_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("ix_risk_events_created", "created_at"),)


class PortfolioState(Base):
    """Daily portfolio snapshot — cash, equity, positions."""

    __tablename__ = "portfolio_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    cash = Column(Float, nullable=False)
    total_equity = Column(Float, nullable=False)
    positions_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("ix_portfolio_state_date", "date"),)


class ParamVersion(Base):
    """Strategy parameter version log — every change documented."""

    __tablename__ = "param_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False)
    params_json = Column(Text, nullable=False)
    change_reason = Column(Text, nullable=True)
    effective_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Trade(Base):
    """Realized round-trip trades — tracks P&L per closed position.

    A Trade is created when a sell order closes (fully or partially) a position.
    It links the buy and sell fills to compute realized P&L.
    """

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    buy_order_id = Column(Integer, nullable=True)
    sell_order_id = Column(Integer, nullable=True)
    qty = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)          # avg cost basis
    exit_price = Column(Float, nullable=False)            # fill price on sell
    realized_pnl = Column(Float, nullable=False)          # exit - entry, in dollars
    realized_pnl_pct = Column(Float, nullable=False)      # as percentage
    is_win = Column(Boolean, nullable=False)              # True if pnl > 0
    entry_date = Column(DateTime, nullable=True)
    exit_date = Column(DateTime, nullable=True)
    holding_days = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_trades_symbol", "symbol"),
        Index("ix_trades_exit_date", "exit_date"),
    )


class CostBasis(Base):
    """Per-symbol cost basis tracking for open positions.

    Updated on every buy fill. Used to compute realized P&L on sells.
    """

    __tablename__ = "cost_basis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, unique=True)
    qty = Column(Float, nullable=False, default=0)
    avg_price = Column(Float, nullable=False, default=0)     # weighted avg entry
    total_cost = Column(Float, nullable=False, default=0)     # qty * avg_price
    last_updated = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("ix_cost_basis_symbol", "symbol"),)


class PositionHighWater(Base):
    """High-water mark tracker for trailing stop-loss per position.

    Updated daily from Alpaca position data. When current price drops
    trailing_stop_pct below the high-water mark, a SELL signal is generated.
    """

    __tablename__ = "position_high_water"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, unique=True)
    high_price = Column(Float, nullable=False)        # Highest price since entry
    entry_price = Column(Float, nullable=False)        # Price at entry
    last_updated = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("ix_position_high_water_symbol", "symbol"),)


class Deposit(Base):
    """Weekly deposit log."""

    __tablename__ = "deposits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    amount = Column(Float, nullable=False)
    date = Column(DateTime, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("ix_deposits_date", "date"),)


def get_engine(database_url=None):
    """Create SQLAlchemy engine from config or environment."""
    if database_url is None:
        database_url = os.getenv("DATABASE_URL", "sqlite:///trading.db")
    return create_engine(database_url, echo=False)


def get_session(engine=None):
    """Create a new database session."""
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(database_url=None):
    """Initialize database — create all tables if they don't exist."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


if __name__ == "__main__":
    engine = init_db()
    print(f"Database initialized: {engine.url}")
    for table in Base.metadata.sorted_tables:
        print(f"  Table: {table.name}")
