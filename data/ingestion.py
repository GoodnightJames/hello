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
    Store daily bars DataFrame to database. Skips duplicates using batch check.

    Instead of checking each row individually (N queries), fetches all existing
    (symbol, date) pairs in one query, then bulk-inserts only new rows.
    On a 365-day backfill of 24 symbols this reduces 8,760 queries to 1.

    Args:
        df: DataFrame with columns: symbol, timestamp, open, high, low, close, volume, vwap, trade_count.
        session: SQLAlchemy session.

    Returns:
        Number of new rows inserted.
    """
    if df.empty:
        logger.info("No bars to store")
        return 0

    # Filter rows with valid symbol/timestamp
    valid = df.dropna(subset=["symbol", "timestamp"])
    if valid.empty:
        logger.info("No valid bars to store")
        return 0

    # Normalize timestamps to naive UTC for consistent comparison with DB
    def _normalize_ts(ts):
        """Strip timezone info so comparisons match SQLite naive datetimes."""
        if hasattr(ts, 'tz') and ts.tz is not None:
            return ts.tz_convert('UTC').tz_localize(None)
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            return ts.replace(tzinfo=None)
        return ts

    valid = valid.copy()
    valid["timestamp"] = valid["timestamp"].apply(_normalize_ts)

    # Batch-check existing records (single query instead of N queries)
    symbols_in_batch = valid["symbol"].unique().tolist()
    dates_in_batch = valid["timestamp"].unique().tolist()

    existing = set()
    # Query in chunks to avoid SQL parameter limits
    chunk_size = 500
    for i in range(0, len(dates_in_batch), chunk_size):
        date_chunk = dates_in_batch[i:i + chunk_size]
        rows = (
            session.query(DailyBar.symbol, DailyBar.date)
            .filter(
                DailyBar.symbol.in_(symbols_in_batch),
                DailyBar.date.in_(date_chunk),
            )
            .all()
        )
        existing.update((r[0], r[1]) for r in rows)

    # Build list of new bars
    new_bars = []
    for _, row in valid.iterrows():
        key = (row["symbol"], row["timestamp"])
        if key in existing:
            continue

        new_bars.append(DailyBar(
            symbol=row["symbol"],
            date=row["timestamp"],
            open=float(row.get("open", 0)),
            high=float(row.get("high", 0)),
            low=float(row.get("low", 0)),
            close=float(row.get("close", 0)),
            volume=float(row.get("volume", 0)),
            vwap=float(row["vwap"]) if pd.notna(row.get("vwap")) else None,
            trade_count=int(row["trade_count"]) if pd.notna(row.get("trade_count")) else None,
        ))

    # Bulk insert
    if new_bars:
        session.add_all(new_bars)
        session.commit()

    skipped = len(valid) - len(new_bars)
    logger.info(
        "Stored bars to database",
        extra={"extra_data": {"inserted": len(new_bars), "skipped": skipped}},
    )
    return len(new_bars)


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
