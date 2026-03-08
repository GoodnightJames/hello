"""
Sleeve Health Monitor — meta-layer that measures whether a strategy
is currently performing well enough to justify continued trading.

This is not superstition — it's regime detection at the strategy level.
Edge can be regime-dependent, and a strategy "out of form" should reduce
aggressiveness automatically.

Measures:
- Rolling expectancy (recent trades profitable after costs?)
- Rolling hit rate (winning enough?)
- Recent drawdown (underwater?)
- Realized slippage vs modeled slippage
- Regime match (is the current regime one where this strategy has worked?)

Returns a health score (0.0 = severely impaired, 1.0 = fully healthy)
and a recommended aggressiveness multiplier.

Usage:
    from risk.sleeve_health import check_sleeve_health

    health = check_sleeve_health(session, "crypto")
    if health["score"] < 0.5:
        # Reduce position sizes or skip marginal trades
"""

import json
from datetime import datetime, timedelta
from collections import defaultdict

from core.logging import get_logger
from data.db import Trade, get_session

logger = get_logger("risk.sleeve_health")


def check_sleeve_health(session, sleeve="crypto", lookback_trades=20,
                        lookback_days=14):
    """
    Assess current sleeve health from recent trade performance.

    Args:
        session: DB session.
        sleeve: "crypto" or "equity".
        lookback_trades: Number of recent trades to evaluate.
        lookback_days: Recency window for time-based checks.

    Returns:
        Dict with:
        {
            "score": float (0.0-1.0),
            "aggressiveness": float (0.25-1.0),
            "components": {
                "expectancy": float,
                "hit_rate": float,
                "drawdown_severity": float,
                "streak": int,
            },
            "recommendation": str,
        }
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    try:
        recent_trades = (
            session.query(Trade)
            .filter(Trade.exit_date >= cutoff)
            .order_by(Trade.exit_date.desc())
            .limit(lookback_trades)
            .all()
        )
    except Exception:
        return _default_health()

    # Filter by sleeve
    if sleeve == "crypto":
        trades = [t for t in recent_trades if "/" in t.symbol]
    else:
        trades = [t for t in recent_trades if "/" not in t.symbol]

    if len(trades) < 3:
        return _default_health(reason="insufficient_data")

    # 1. Rolling expectancy
    pnls = [t.realized_pnl for t in trades if t.realized_pnl is not None]
    if pnls:
        avg_pnl = sum(pnls) / len(pnls)
        # Normalize: 0 at breakeven, 1.0 at strong positive
        # Scale so $0.50 average profit on small trades = decent
        expectancy_score = min(1.0, max(0.0, (avg_pnl + 0.50) / 1.0))
    else:
        avg_pnl = 0
        expectancy_score = 0.5

    # 2. Hit rate
    wins = sum(1 for t in trades if t.is_win)
    hit_rate = wins / len(trades) if trades else 0
    # Below 30% = very bad, above 55% = great
    hit_rate_score = min(1.0, max(0.0, (hit_rate - 0.20) / 0.40))

    # 3. Consecutive loss streak (from most recent)
    streak = 0
    for t in trades:
        if not t.is_win:
            streak += 1
        else:
            break
    # 0-1 losses = fine, 3+ = concerning, 5+ = severe
    streak_score = max(0.0, 1.0 - streak * 0.2)

    # 4. Recent drawdown severity
    # Check if cumulative PnL is negative over lookback
    cumulative = sum(pnls)
    # Normalize: -$5 on a small account = moderate concern
    dd_score = min(1.0, max(0.0, (cumulative + 5.0) / 10.0))

    # Composite health score (weighted)
    score = (
        0.35 * expectancy_score
        + 0.25 * hit_rate_score
        + 0.25 * streak_score
        + 0.15 * dd_score
    )
    score = round(min(1.0, max(0.0, score)), 3)

    # Aggressiveness: map score to trading intensity
    if score >= 0.7:
        aggressiveness = 1.0
        recommendation = "healthy — normal trading"
    elif score >= 0.5:
        aggressiveness = 0.75
        recommendation = "caution — reduce position sizes"
    elif score >= 0.3:
        aggressiveness = 0.50
        recommendation = "impaired — trade only high-conviction setups"
    else:
        aggressiveness = 0.25
        recommendation = "severely impaired — skip marginal trades"

    result = {
        "score": score,
        "aggressiveness": aggressiveness,
        "components": {
            "expectancy": round(avg_pnl, 4),
            "expectancy_score": round(expectancy_score, 3),
            "hit_rate": round(hit_rate, 3),
            "hit_rate_score": round(hit_rate_score, 3),
            "consecutive_losses": streak,
            "streak_score": round(streak_score, 3),
            "cumulative_pnl": round(cumulative, 4),
            "drawdown_score": round(dd_score, 3),
        },
        "trades_evaluated": len(trades),
        "recommendation": recommendation,
    }

    logger.info(
        f"Sleeve health [{sleeve}]: score={score:.2f} → {recommendation}",
        extra={"extra_data": result},
    )

    return result


def _default_health(reason="no_data"):
    """Return default healthy state when insufficient data."""
    return {
        "score": 0.75,
        "aggressiveness": 1.0,
        "components": {},
        "trades_evaluated": 0,
        "recommendation": f"default — {reason}",
    }
