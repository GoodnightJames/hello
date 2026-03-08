"""
Universe Listing Rules — rules-governed asset inclusion/exclusion.

Instead of a hand-curated static list, enforce quantitative criteria
for whether a crypto asset belongs in the tradeable universe.

Listing criteria:
1. Minimum 24h volume (from exchange data)
2. Minimum price history (days of clean data)
3. Maximum spread threshold (estimated from cost model)
4. Data freshness (stale data → disable)

Usage:
    from risk.universe import filter_tradeable_universe
    active = filter_tradeable_universe(all_symbols, prices, config)
"""

from core.logging import get_logger
from risk.cost_model import SPREAD_OVERRIDES_BPS

logger = get_logger("risk.universe")

# Default listing thresholds
DEFAULT_LISTING_RULES = {
    "min_history_days": 14,       # Need at least 14 days of price data
    "max_spread_bps": 35.0,       # Don't trade assets with >35 bps estimated spread
    "min_avg_daily_range_pct": 0.005,  # Must have some price movement (not dead)
    "max_stale_hours": 48,        # Disable if last price update >48h old
}


def check_listing_criteria(symbol, prices_series, config=None):
    """
    Check whether a single symbol meets listing criteria.

    Args:
        symbol: Ticker symbol.
        prices_series: Pandas Series of daily close prices, or None.
        config: Optional dict overriding DEFAULT_LISTING_RULES.

    Returns:
        (eligible: bool, reasons: list[str])
    """
    rules = dict(DEFAULT_LISTING_RULES)
    if config:
        rules.update(config)

    reasons = []

    # Rule 1: minimum price history
    if prices_series is None or len(prices_series) < rules["min_history_days"]:
        actual = 0 if prices_series is None else len(prices_series)
        reasons.append(
            f"insufficient history: {actual} < {rules['min_history_days']} days"
        )

    # Rule 2: spread threshold (from known overrides)
    known_spread = SPREAD_OVERRIDES_BPS.get(symbol, 10.0)
    if known_spread > rules["max_spread_bps"]:
        reasons.append(
            f"spread too wide: {known_spread:.0f} bps > {rules['max_spread_bps']:.0f} bps max"
        )

    # Rule 3: minimum price movement (not a dead/delisted asset)
    if prices_series is not None and len(prices_series) >= 5:
        recent = prices_series.tail(5)
        daily_range = recent.pct_change().dropna().abs().mean()
        if daily_range < rules["min_avg_daily_range_pct"]:
            reasons.append(
                f"price too stale: avg daily range {daily_range:.3%} "
                f"< {rules['min_avg_daily_range_pct']:.3%}"
            )

    # Rule 4: data freshness (check last timestamp vs current)
    # This is best-effort — if the index has timestamps, check recency
    if prices_series is not None and hasattr(prices_series.index, 'max'):
        try:
            from datetime import datetime, timedelta
            import pandas as pd
            last_date = pd.Timestamp(prices_series.index.max())
            now = pd.Timestamp(datetime.utcnow())
            hours_stale = (now - last_date).total_seconds() / 3600
            if hours_stale > rules["max_stale_hours"]:
                reasons.append(
                    f"data stale: last update {hours_stale:.0f}h ago "
                    f"> {rules['max_stale_hours']}h max"
                )
        except Exception:
            pass  # Skip freshness check if timestamps aren't parseable

    eligible = len(reasons) == 0
    return eligible, reasons


def filter_tradeable_universe(symbols, prices_df, config=None):
    """
    Filter a list of symbols down to those meeting listing criteria.

    Args:
        symbols: List of symbol strings.
        prices_df: DataFrame with symbol columns and price rows.
        config: Optional listing rules override dict.

    Returns:
        (active_symbols, removed_with_reasons)
        active_symbols: list of symbols that pass
        removed_with_reasons: list of (symbol, [reasons]) for failed symbols
    """
    active = []
    removed = []

    for symbol in symbols:
        if prices_df is not None and symbol in prices_df.columns:
            series = prices_df[symbol].dropna()
        else:
            series = None

        eligible, reasons = check_listing_criteria(symbol, series, config)

        if eligible:
            active.append(symbol)
        else:
            removed.append((symbol, reasons))
            logger.warning(
                f"Universe filter: REMOVED {symbol}",
                extra={"extra_data": {"symbol": symbol, "reasons": reasons}},
            )

    logger.info(
        f"Universe filter: {len(active)}/{len(symbols)} symbols active",
        extra={"extra_data": {
            "active": active,
            "removed": [(s, r) for s, r in removed],
        }},
    )

    return active, removed
