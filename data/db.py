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
