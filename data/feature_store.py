"""
Feature store — computes technical features from daily bars.

All features are computed from daily OHLCV data stored in the database.
No intraday data. No lookahead bias — features use only data available
at computation time.
"""

import pandas as pd
from sqlalchemy import func

from core.logging import get_logger
from data.db import DailyBar, get_session

logger = get_logger("data.feature_store")


def get_price_history(symbols, lookback_days=252, session=None):
    """
    Fetch daily close prices for symbols from database.

    Args:
        symbols: List of ticker symbols.
        lookback_days: Number of trading days of history to fetch.
        session: SQLAlchemy session (created if not provided).

    Returns:
        DataFrame with DatetimeIndex and one column per symbol (close prices).
    """
    close_session = False
    if session is None:
        session = get_session()
        close_session = True

    try:
        # Get the latest date in DB to anchor the lookback
        max_date = session.query(func.max(DailyBar.date)).scalar()
        if max_date is None:
            logger.warning("No price data in database")
            return pd.DataFrame()

        rows = (
            session.query(DailyBar.symbol, DailyBar.date, DailyBar.close)
            .filter(DailyBar.symbol.in_(symbols))
            .order_by(DailyBar.date.desc())
            .limit(lookback_days * len(symbols))
            .all()
        )

        if not rows:
            logger.warning("No price data found for requested symbols")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["symbol", "date", "close"])
        pivot = df.pivot(index="date", columns="symbol", values="close").sort_index()

        logger.info(
            "Fetched price history",
            extra={"extra_data": {"symbols": len(symbols), "rows": len(pivot)}},
        )
        return pivot

    finally:
        if close_session:
            session.close()


def compute_returns(prices, periods):
    """
    Compute percentage returns over multiple lookback periods.

    Args:
        prices: DataFrame of close prices (DatetimeIndex, symbol columns).
        periods: Dict of {label: trading_days}, e.g. {"12m": 252, "6m": 126}.

    Returns:
        Dict of {label: DataFrame of returns per symbol}.
    """
    results = {}
    for label, days in periods.items():
        if len(prices) < days:
            logger.warning(
                f"Insufficient data for {label} returns",
                extra={"extra_data": {"need": days, "have": len(prices)}},
            )
            results[label] = pd.DataFrame()
            continue
        results[label] = prices.pct_change(periods=days)

    logger.info(
        "Computed returns",
        extra={"extra_data": {"periods": list(periods.keys())}},
    )
    return results


def compute_sma(prices, windows):
    """
    Compute simple moving averages.

    Args:
        prices: DataFrame of close prices.
        windows: Dict of {label: window_size}, e.g. {"200d": 200, "50d": 50}.

    Returns:
        Dict of {label: DataFrame of SMA values per symbol}.
    """
    results = {}
    for label, window in windows.items():
        results[label] = prices.rolling(window=window).mean()

    logger.info(
        "Computed SMAs",
        extra={"extra_data": {"windows": list(windows.keys())}},
    )
    return results


def compute_rsi(prices, period=2):
    """
    Compute RSI (Relative Strength Index).

    Default period=2 for Connors RSI(2) mean reversion (v1.5).

    Args:
        prices: DataFrame of close prices.
        period: RSI lookback period.

    Returns:
        DataFrame of RSI values per symbol.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Standard RS calculation
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # When avg_loss is zero (no losses), RSI should be 100
    no_loss = avg_loss.abs() < 1e-15
    rsi = rsi.where(~no_loss, 100.0)

    # When avg_gain is zero (no gains), RSI should be 0
    no_gain = avg_gain.abs() < 1e-15
    rsi = rsi.where(~no_gain, 0.0)

    logger.info(
        "Computed RSI",
        extra={"extra_data": {"period": period, "symbols": len(prices.columns)}},
    )
    return rsi


def build_features(symbols, lookback_days=300, session=None):
    """
    Build complete feature set for the given symbols.

    Returns a dict with all computed features:
    - prices: raw close prices
    - returns: {period_label: returns_df}
    - sma: {window_label: sma_df}
    - rsi_2: RSI(2) values

    Args:
        symbols: List of ticker symbols.
        lookback_days: Trading days of history to fetch (extra buffer for SMA computation).
        session: SQLAlchemy session.

    Returns:
        Dict of feature DataFrames.
    """
    logger.info(
        "Building features",
        extra={"extra_data": {"symbols": symbols, "lookback": lookback_days}},
    )

    prices = get_price_history(symbols, lookback_days=lookback_days, session=session)
    if prices.empty:
        logger.warning("No price data available — cannot build features")
        return {"prices": prices, "returns": {}, "sma": {}, "rsi_2": pd.DataFrame()}

    returns = compute_returns(prices, {"12m": 252, "6m": 126, "3m": 63, "1m": 21})
    sma = compute_sma(prices, {"200d": 200, "50d": 50})
    rsi_2 = compute_rsi(prices, period=2)

    features = {
        "prices": prices,
        "returns": returns,
        "sma": sma,
        "rsi_2": rsi_2,
    }

    logger.info("Feature build complete")
    return features
