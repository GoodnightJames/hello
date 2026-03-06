"""
Performance Manager — dynamic risk mode selection based on equity performance.

Tracks rolling equity metrics and selects one of three operating modes:
  conservative  — protect capital during drawdowns / losing streaks
  normal        — default state
  aggressive    — increase exposure when performance is strong

The strategy logic never changes.  Only *exposure* changes:
  • position size  (risk_per_trade_pct)
  • max concurrent positions
  • deposit deployment speed

Mode transitions are slow and capped — no doubling after a win.
"""

from datetime import datetime, timedelta

from core.logging import get_logger
from data.db import PortfolioState, Order, RiskEvent, Trade

logger = get_logger("performance.manager")

MODE_CONSERVATIVE = "conservative"
MODE_NORMAL = "normal"
MODE_AGGRESSIVE = "aggressive"

# Ordered from safest to riskiest for clamping
_MODE_ORDER = [MODE_CONSERVATIVE, MODE_NORMAL, MODE_AGGRESSIVE]


# ── Equity metrics ──────────────────────────────────────────────────────────


def compute_equity_curve(session, lookback_days=60):
    """
    Fetch the equity curve from portfolio snapshots.

    Args:
        session: DB session.
        lookback_days: Number of calendar days to look back.

    Returns:
        List of (date, total_equity) tuples, oldest first.
    """
    cutoff = datetime.utcnow() - timedelta(days=int(lookback_days * 1.5))
    snapshots = (
        session.query(PortfolioState)
        .filter(PortfolioState.date >= cutoff)
        .order_by(PortfolioState.date.asc())
        .all()
    )
    return [(s.date, s.total_equity) for s in snapshots]


def compute_rolling_return(equity_curve, lookback_days=60):
    """
    Compute rolling return over the lookback window.

    Returns:
        Float return (e.g. 0.08 for 8%), or None if insufficient data.
    """
    if len(equity_curve) < 2:
        return None

    # Find the snapshot closest to `lookback_days` ago
    now = equity_curve[-1][0]
    target = now - timedelta(days=lookback_days)
    past = [e for e in equity_curve if e[0] <= target]

    if not past:
        # Use the oldest available snapshot
        start_equity = equity_curve[0][1]
    else:
        start_equity = past[-1][1]

    if start_equity <= 0:
        return None

    current_equity = equity_curve[-1][1]
    return (current_equity - start_equity) / start_equity


def compute_max_drawdown(equity_curve):
    """
    Compute maximum drawdown from peak over the equity curve.

    Returns:
        Float drawdown as a positive number (e.g. 0.05 for 5%), or 0.0.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0][1]
    max_dd = 0.0

    for _, equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

    return max_dd


def compute_current_drawdown(equity_curve):
    """
    Compute current drawdown from peak (not recovered).

    Returns:
        Float drawdown as a positive number, or 0.0.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = max(e for _, e in equity_curve)
    current = equity_curve[-1][1]

    if peak <= 0:
        return 0.0

    return max(0.0, (peak - current) / peak)


def count_consecutive_losses(session, max_check=10):
    """
    Count consecutive losing trades (most recent first).

    Uses the Trade table which tracks realized P&L per round-trip.

    Returns:
        Int — number of consecutive losses from the most recent trade.
    """
    recent_trades = (
        session.query(Trade)
        .order_by(Trade.exit_date.desc())
        .limit(max_check)
        .all()
    )

    if not recent_trades:
        return 0

    consecutive = 0
    for trade in recent_trades:
        if not trade.is_win:
            consecutive += 1
        else:
            break
    return consecutive


def count_trades_this_week(session):
    """
    Count the number of filled orders in the current calendar week.

    Returns:
        Int — number of filled orders this week (Mon-Sun).
    """
    now = datetime.utcnow()
    # Monday of this week
    monday = now - timedelta(days=now.weekday())
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)

    count = (
        session.query(Order)
        .filter(
            Order.status == "filled",
            Order.filled_at >= monday,
        )
        .count()
    )
    return count


# ── Mode selection ──────────────────────────────────────────────────────────


def select_risk_mode(session, risk_params):
    """
    Determine the current risk mode based on performance metrics.

    Decision tree (evaluated top to bottom, first match wins):
    1. If drawdown ≥ conservative threshold OR consecutive losses ≥ limit → conservative
    2. If rolling return ≥ aggressive threshold AND drawdown < aggressive max → aggressive
    3. Otherwise → normal

    Args:
        session: DB session.
        risk_params: Full risk_params dict (includes mode_transitions).

    Returns:
        Dict:
        {
            "mode": "conservative" | "normal" | "aggressive",
            "metrics": {rolling_return, drawdown, consecutive_losses},
            "reason": str,
        }
    """
    transitions = risk_params.get("mode_transitions", {})
    agg = transitions.get("aggressive_trigger", {})
    con = transitions.get("conservative_trigger", {})

    lookback = agg.get("lookback_days", 60)
    equity_curve = compute_equity_curve(session, lookback_days=lookback)

    rolling_ret = compute_rolling_return(equity_curve, lookback_days=lookback)
    drawdown = compute_current_drawdown(equity_curve)
    consec_losses = count_consecutive_losses(session)

    metrics = {
        "rolling_return": rolling_ret,
        "drawdown": round(drawdown, 6),
        "consecutive_losses": consec_losses,
        "equity_points": len(equity_curve),
    }

    # Not enough data → stay normal
    if rolling_ret is None:
        logger.info(
            "Risk mode: normal (insufficient data)",
            extra={"extra_data": metrics},
        )
        return {"mode": MODE_NORMAL, "metrics": metrics, "reason": "Insufficient data for mode selection"}

    # ── Check conservative triggers ────────────────────────────────────────
    con_max_dd = con.get("max_drawdown", 0.05)
    con_max_losses = con.get("max_consecutive_losses", 4)

    if drawdown >= con_max_dd:
        reason = f"Drawdown {drawdown:.2%} >= {con_max_dd:.2%} threshold"
        logger.warning(f"Risk mode: CONSERVATIVE — {reason}", extra={"extra_data": metrics})
        _log_mode_change(session, MODE_CONSERVATIVE, reason, metrics)
        return {"mode": MODE_CONSERVATIVE, "metrics": metrics, "reason": reason}

    if consec_losses >= con_max_losses:
        reason = f"{consec_losses} consecutive losses >= {con_max_losses} limit"
        logger.warning(f"Risk mode: CONSERVATIVE — {reason}", extra={"extra_data": metrics})
        _log_mode_change(session, MODE_CONSERVATIVE, reason, metrics)
        return {"mode": MODE_CONSERVATIVE, "metrics": metrics, "reason": reason}

    # ── Check aggressive triggers ──────────────────────────────────────────
    agg_min_ret = agg.get("min_rolling_return", 0.08)
    agg_max_dd = agg.get("max_drawdown", 0.04)

    if rolling_ret >= agg_min_ret and drawdown < agg_max_dd:
        reason = f"Return {rolling_ret:.2%} >= {agg_min_ret:.2%} and drawdown {drawdown:.2%} < {agg_max_dd:.2%}"
        logger.info(f"Risk mode: AGGRESSIVE — {reason}", extra={"extra_data": metrics})
        _log_mode_change(session, MODE_AGGRESSIVE, reason, metrics)
        return {"mode": MODE_AGGRESSIVE, "metrics": metrics, "reason": reason}

    # ── Default: normal ────────────────────────────────────────────────────
    reason = "Default — no trigger conditions met"
    logger.info(f"Risk mode: normal — {reason}", extra={"extra_data": metrics})
    return {"mode": MODE_NORMAL, "metrics": metrics, "reason": reason}


def get_mode_params(risk_params, mode):
    """
    Return the risk parameters for a given mode.

    Falls back to the base position_limits if the mode isn't defined.

    Args:
        risk_params: Full risk_params dict.
        mode: One of "conservative", "normal", "aggressive".

    Returns:
        Dict with keys: risk_per_trade_pct, max_positions, deploy_deposits
    """
    modes = risk_params.get("risk_modes", {})
    mode_config = modes.get(mode, {})

    # Fallback to base values
    base = risk_params.get("position_limits", {})

    return {
        "risk_per_trade_pct": mode_config.get("risk_per_trade_pct", base.get("max_risk_per_trade", 0.005)),
        "max_positions": mode_config.get("max_positions", base.get("max_concurrent_positions", 3)),
        "deploy_deposits": mode_config.get("deploy_deposits", True),
    }


def check_trade_throttle(session, risk_params):
    """
    Check if the weekly trade limit has been reached.

    Returns:
        (bool, dict) — (is_throttled, details)
    """
    throttle = risk_params.get("trade_throttle", {})
    max_per_week = throttle.get("max_trades_per_week", 5)
    current = count_trades_this_week(session)

    details = {
        "trades_this_week": current,
        "max_per_week": max_per_week,
    }

    if current >= max_per_week:
        logger.warning(
            "TRADE THROTTLE — weekly limit reached",
            extra={"extra_data": details},
        )
        return True, details

    return False, details


def _log_mode_change(session, new_mode, reason, metrics):
    """Log a risk mode change as a risk event."""
    import json

    event = RiskEvent(
        event_type=f"mode_change_{new_mode}",
        severity="INFO" if new_mode == MODE_NORMAL else "WARNING",
        details_json=json.dumps({"mode": new_mode, "reason": reason, **metrics}),
    )
    session.add(event)
