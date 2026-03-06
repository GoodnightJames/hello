"""
Data ingestion module — fetches daily OHLCV bars from Alpaca
and stores them to the local SQLite database.

Daily bars only — no intraday data.
All parameters loaded from YAML config — no magic numbers.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import func

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from core.logging import get_logger
from data.db import DailyBar, get_engine, get_session, init_db

load_dotenv()
logger = get_logger("data.ingestion")


def load_config(config_path="config/settings.yaml"):
    """Load trading engine configuration from YAML."""
    logger.info("Loading config", extra={"extra_data": {"path": config_path}})
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_alpaca_client():
    """Create Alpaca historical data client using .env credentials."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        logger.error("Alpaca API keys not found in environment")
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
        )

    logger.info("Created Alpaca data client")
    return StockHistoricalDataClient(api_key, secret_key)


def fetch_daily_bars(client, symbols, start_date, end_date):
    """
    Fetch daily OHLCV bars from Alpaca for given symbols.

    Args:
        client: StockHistoricalDataClient instance.
        symbols: List of ticker symbols.
        start_date: Start date (datetime).
        end_date: End date (datetime).

    Returns:
        pandas DataFrame with OHLCV data, or empty DataFrame on error.
    """
    logger.info(
        "Fetching daily bars",
        extra={
            "extra_data": {
                "symbols_count": len(symbols),
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            }
        },
    )

    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
        )
        bars = client.get_stock_bars(request_params)
        df = bars.df

        if df.empty:
            logger.info("No bars returned from Alpaca")
            return pd.DataFrame()

        # Reset multi-index (symbol, timestamp) to columns
        df = df.reset_index()
        logger.info(
            "Fetched bars successfully",
            extra={"extra_data": {"rows": len(df)}},
        )
        return df

    except Exception as e:
        logger.error(
            "Failed to fetch bars from Alpaca",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )
        return pd.DataFrame()


def store_bars(df, session):
    """
    Store daily bars DataFrame to database. Skips duplicates.

    Args:
        df: DataFrame with columns: symbol, timestamp, open, high, low, close, volume, vwap, trade_count.
        session: SQLAlchemy session.

    Returns:
        Number of new rows inserted.
    """
    if df.empty:
        logger.info("No bars to store")
        return 0

    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        symbol = row.get("symbol", None)
        timestamp = row.get("timestamp", None)

        if symbol is None or timestamp is None:
            skipped += 1
            continue

        # Check for existing record
        exists = (
            session.query(DailyBar)
            .filter(DailyBar.symbol == symbol, DailyBar.date == timestamp)
            .first()
        )
        if exists:
            skipped += 1
            continue

        bar = DailyBar(
            symbol=symbol,
            date=timestamp,
            open=float(row.get("open", 0)),
            high=float(row.get("high", 0)),
            low=float(row.get("low", 0)),
            close=float(row.get("close", 0)),
            volume=float(row.get("volume", 0)),
            vwap=float(row["vwap"]) if pd.notna(row.get("vwap")) else None,
            trade_count=int(row["trade_count"]) if pd.notna(row.get("trade_count")) else None,
        )
        session.add(bar)
        inserted += 1

    session.commit()
    logger.info(
        "Stored bars to database",
        extra={"extra_data": {"inserted": inserted, "skipped": skipped}},
    )
    return inserted


def get_last_bar_date(session, symbol=None):
    """Get the most recent bar date in the database."""
    query = session.query(func.max(DailyBar.date))
    if symbol:
        query = query.filter(DailyBar.symbol == symbol)
    result = query.scalar()
    return result


def ingest_daily(config_path="config/settings.yaml"):
    """
    Main daily ingestion orchestrator.

    Loads config, determines date range (last stored date to yesterday),
    fetches and stores bars for full universe.
    """
    logger.info("Starting daily ingestion")

    config = load_config(config_path)
    symbols = config["universe"]["equities"]

    # Initialize DB and get session
    init_db()
    session = get_session()

    try:
        # Determine date range
        last_date = get_last_bar_date(session)
        if last_date:
            start_date = last_date + timedelta(days=1)
        else:
            # First run — fetch 1 year of history for momentum calculations
            start_date = datetime.now() - timedelta(days=365)

        end_date = datetime.now() - timedelta(days=1)

        if start_date >= end_date:
            logger.info("Data is up to date, nothing to fetch")
            return 0

        # Fetch and store
        client = create_alpaca_client()
        df = fetch_daily_bars(client, symbols, start_date, end_date)
        count = store_bars(df, session)

        logger.info(
            "Daily ingestion complete",
            extra={"extra_data": {"new_bars": count}},
        )
        return count

    except Exception as e:
        logger.error(
            "Daily ingestion failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )
        raise
    finally:
        session.close()


def backfill(symbols, start_date, end_date, config_path="config/settings.yaml"):
    """
    One-time historical backfill for specific symbols and date range.

    Args:
        symbols: List of ticker symbols to backfill.
        start_date: Start date (datetime).
        end_date: End date (datetime).
        config_path: Path to settings config.
    """
    logger.info(
        "Starting backfill",
        extra={
            "extra_data": {
                "symbols": symbols,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            }
        },
    )

    init_db()
    session = get_session()

    try:
        client = create_alpaca_client()
        df = fetch_daily_bars(client, symbols, start_date, end_date)
        count = store_bars(df, session)

        logger.info(
            "Backfill complete",
            extra={"extra_data": {"new_bars": count}},
        )
        return count

    except Exception as e:
        logger.error(
            "Backfill failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )
        raise
    finally:
        session.close()


if __name__ == "__main__":
    ingest_daily()
