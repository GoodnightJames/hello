"""
Trailing Stop-Loss — protects gains by selling when price drops from peak.

Without this, the only exit signal is when 12-month momentum turns negative.
That's a 252-day lookback — a stock can drop 30% before that triggers.

A trailing stop at 8% from peak means:
- Buy SPY at $500, it runs to $550 → stop at $506 (8% below $550)
- If it drops to $506, sell immediately → lock in $6/share gain

Only activates after the position has gained 3% from entry, so we don't
get stopped out on normal noise right after buying.
"""

from datetime import datetime

from core.logging import get_logger
from data.db import PositionHighWater, CostBasis

logger = get_logger("risk.trailing_stop")


def update_high_water_marks(session, positions_with_prices):
    """
    Update high-water marks for all open positions.

    Called during portfolio sync (daily at market close).

    Args:
        session: DB session.
        positions_with_prices: Dict of {symbol: current_price}.
    """
    for symbol, current_price in positions_with_prices.items():
        if current_price <= 0:
            continue

        hw = session.query(PositionHighWater).filter(
            PositionHighWater.symbol == symbol
        ).first()

        if hw is None:
            # New position — initialize high water at current price
            basis = session.query(CostBasis).filter(CostBasis.symbol == symbol).first()
            entry_price = basis.avg_price if basis and basis.avg_price > 0 else current_price

            hw = PositionHighWater(
                symbol=symbol,
                high_price=current_price,
                entry_price=entry_price,
                last_updated=datetime.utcnow(),
            )
            session.add(hw)
            logger.info(
                f"High-water initialized: {symbol} @ ${current_price:.2f}",
            )
        elif current_price > hw.high_price:
            # New high — update
            hw.high_price = current_price
            hw.last_updated = datetime.utcnow()
            logger.info(
                f"High-water updated: {symbol} ${hw.high_price:.2f} → ${current_price:.2f}",
            )


def check_trailing_stops(session, positions_with_prices, risk_params):
    """
    Check all positions against their trailing stop levels.

    Args:
        session: DB session.
        positions_with_prices: Dict of {symbol: current_price}.
        risk_params: Risk parameters dict.

    Returns:
        List of symbols that should be sold (trailing stop triggered).
    """
    stop_config = risk_params.get("trailing_stop", {})
    if not stop_config.get("enabled", False):
        return []

    stop_pct = stop_config.get("stop_pct", 0.08)
    min_gain = stop_config.get("min_gain_to_activate", 0.03)

    triggered = []

    for symbol, current_price in positions_with_prices.items():
        if current_price <= 0:
            continue

        hw = session.query(PositionHighWater).filter(
            PositionHighWater.symbol == symbol
        ).first()

        if hw is None:
            continue

        # Only activate trailing stop after minimum gain from entry
        gain_from_entry = (hw.high_price - hw.entry_price) / hw.entry_price if hw.entry_price > 0 else 0
        if gain_from_entry < min_gain:
            continue

        # Check if price has dropped stop_pct from high
        stop_price = hw.high_price * (1 - stop_pct)
        drop_from_high = (hw.high_price - current_price) / hw.high_price if hw.high_price > 0 else 0

        if current_price <= stop_price:
            triggered.append(symbol)
            logger.warning(
                f"TRAILING STOP triggered: {symbol} "
                f"current=${current_price:.2f} < stop=${stop_price:.2f} "
                f"(high=${hw.high_price:.2f}, drop={drop_from_high:.1%})",
                extra={
                    "extra_data": {
                        "symbol": symbol,
                        "current_price": current_price,
                        "high_price": hw.high_price,
                        "stop_price": stop_price,
                        "entry_price": hw.entry_price,
                        "drop_from_high": round(drop_from_high, 4),
                        "gain_from_entry": round(gain_from_entry, 4),
                    }
                },
            )

    return triggered


def clear_high_water(session, symbol):
    """Remove high-water mark when a position is fully closed."""
    hw = session.query(PositionHighWater).filter(
        PositionHighWater.symbol == symbol
    ).first()
    if hw:
        session.delete(hw)
        logger.info(f"High-water cleared: {symbol}")
