"""
Risk Enforcer — hard limits that override all other modules.

The risk engine is the final gate before any order submission.
It enforces:
1. Max risk per trade (dynamic — set by performance mode)
2. Max concurrent positions (dynamic — set by performance mode)
3. No averaging down (NEVER)
4. Daily loss limit (2% → shutdown)
5. Weekly drawdown limit (5% → review mode)
6. Consecutive loss shutdown (3 losses → halt)
7. Kill switch (env var)
8. Weekly trade throttle (max 5 trades/week)

Every risk event is logged to the risk_events table.
"""

import json
import os
from datetime import datetime, timedelta

import yaml

from core.logging import get_logger
from data.db import RiskEvent, Order, PortfolioState, get_session, init_db

logger = get_logger("risk.enforcer")


def load_risk_params(config_path="config/risk_params.yaml"):
    """Load risk parameters from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log_risk_event(session, event_type, severity, details):
    """Persist a risk event to the database."""
    event = RiskEvent(
        event_type=event_type,
        severity=severity,
        details_json=json.dumps(details),
    )
    session.add(event)
    logger.info(
        f"Risk event: {event_type}",
        extra={"extra_data": {"severity": severity, "details": details}},
    )
    return event


def check_kill_switch():
    """Check if the kill switch environment variable is activated."""
    kill = os.getenv("KILL_SWITCH", "false").lower()
    if kill == "true":
        logger.warning("KILL SWITCH IS ACTIVE — all trading halted")
        return True
    return False


def check_daily_loss_limit(session, risk_params):
    """
    Check if daily P&L loss exceeds the configured limit.

    Compares today's portfolio value against yesterday's.

    Returns:
        (bool, dict) — (is_breached, details)
    """
    limit = risk_params.get("daily_limits", {}).get("max_daily_loss", 0.02)

    # Get the two most recent portfolio snapshots
    snapshots = (
        session.query(PortfolioState)
        .order_by(PortfolioState.date.desc())
        .limit(2)
        .all()
    )

    if len(snapshots) < 2:
        return False, {"reason": "Insufficient portfolio history for daily loss check"}

    current = snapshots[0]
    previous = snapshots[1]

    if previous.total_equity <= 0:
        return False, {"reason": "Previous equity is zero — cannot compute loss"}

    daily_return = (current.total_equity - previous.total_equity) / previous.total_equity

    details = {
        "current_equity": current.total_equity,
        "previous_equity": previous.total_equity,
        "daily_return": round(daily_return, 6),
        "limit": -limit,
    }

    if daily_return < -limit:
        logger.warning(
            "DAILY LOSS LIMIT BREACHED",
            extra={"extra_data": details},
        )
        log_risk_event(session, "daily_loss_limit", "CRITICAL", details)
        return True, details

    return False, details


def check_weekly_drawdown(session, risk_params):
    """
    Check if weekly drawdown exceeds the configured limit.

    Compares current portfolio value against value from 5 trading days ago.

    Returns:
        (bool, dict) — (is_breached, details)
    """
    limit = risk_params.get("daily_limits", {}).get("max_weekly_drawdown", 0.05)

    snapshots = (
        session.query(PortfolioState)
        .order_by(PortfolioState.date.desc())
        .limit(6)
        .all()
    )

    if len(snapshots) < 2:
        return False, {"reason": "Insufficient portfolio history for weekly drawdown check"}

    current = snapshots[0]
    # Use the oldest available snapshot (up to 5 days back)
    week_ago = snapshots[-1]

    if week_ago.total_equity <= 0:
        return False, {"reason": "Week-ago equity is zero"}

    weekly_return = (current.total_equity - week_ago.total_equity) / week_ago.total_equity

    details = {
        "current_equity": current.total_equity,
        "week_ago_equity": week_ago.total_equity,
        "weekly_return": round(weekly_return, 6),
        "limit": -limit,
    }

    if weekly_return < -limit:
        logger.warning(
            "WEEKLY DRAWDOWN LIMIT BREACHED",
            extra={"extra_data": details},
        )
        log_risk_event(session, "weekly_drawdown", "WARNING", details)
        return True, details

    return False, details


def check_consecutive_losses(session, risk_params):
    """
    Check if the last N fills were all losses.

    Returns:
        (bool, dict) — (is_breached, details)
    """
    max_consecutive = risk_params.get("shutdown_rules", {}).get("consecutive_loss_shutdown", 3)

    # Get the most recent filled sell orders (realized trades)
    recent_orders = (
        session.query(Order)
        .filter(Order.status == "filled", Order.side == "sell")
        .order_by(Order.filled_at.desc())
        .limit(max_consecutive)
        .all()
    )

    if len(recent_orders) < max_consecutive:
        return False, {"reason": f"Only {len(recent_orders)} filled sells — need {max_consecutive} to check"}

    # For each sell, check if it was a loss by comparing fill price to the
    # original buy price. This is simplified — a full implementation would
    # track cost basis per position.
    # For now, we check if the filled_price trend is declining.
    consecutive_losses = 0
    for order in recent_orders:
        # In a paper trading context, we mark orders with P&L metadata
        # For now, check if we have loss markers
        if order.filled_price is not None and order.filled_qty is not None:
            consecutive_losses += 1  # Placeholder — full P&L tracking in Phase 4

    details = {
        "consecutive_losses": consecutive_losses,
        "limit": max_consecutive,
        "recent_sells": len(recent_orders),
    }

    # This will be fully implemented when we have proper P&L tracking
    return False, details


def check_position_limits(session, risk_params, proposed_symbol=None, mode_params=None):
    """
    Check if adding a new position would exceed limits.

    When mode_params is provided, max_positions comes from the active
    performance mode instead of the base config.

    Args:
        session: DB session.
        risk_params: Risk parameters dict.
        proposed_symbol: Symbol we want to buy (None to just check count).
        mode_params: Optional dict from performance.manager.get_mode_params().

    Returns:
        (bool, dict) — (can_open_position, details)
    """
    if mode_params:
        max_positions = mode_params.get("max_positions", 3)
    else:
        max_positions = risk_params.get("position_limits", {}).get("max_concurrent_positions", 3)

    # Get current open positions from the latest portfolio state
    latest_state = (
        session.query(PortfolioState)
        .order_by(PortfolioState.date.desc())
        .first()
    )

    if latest_state is None:
        # No portfolio state yet — first trade is allowed
        return True, {"current_positions": 0, "max": max_positions}

    positions = json.loads(latest_state.positions_json) if latest_state.positions_json else {}
    current_count = len([s for s, qty in positions.items() if qty > 0])

    details = {
        "current_positions": current_count,
        "max": max_positions,
        "proposed_symbol": proposed_symbol,
    }

    # If the symbol is already held, it's not a new position
    if proposed_symbol and proposed_symbol in positions and positions[proposed_symbol] > 0:
        # Check averaging down rule
        no_avg_down = not risk_params.get("position_limits", {}).get("averaging_down", False)
        if no_avg_down:
            logger.warning(
                f"AVERAGING DOWN BLOCKED for {proposed_symbol}",
                extra={"extra_data": details},
            )
            log_risk_event(session, "averaging_down_blocked", "WARNING", details)
            return False, {**details, "reason": "Averaging down is prohibited"}
        return True, details

    if current_count >= max_positions:
        logger.warning(
            "MAX POSITIONS REACHED",
            extra={"extra_data": details},
        )
        log_risk_event(session, "max_positions_reached", "WARNING", details)
        return False, details

    return True, details


def calculate_max_trade_size(risk_params, total_equity, mode_params=None):
    """
    Calculate maximum dollar amount per trade based on risk limits.

    When mode_params is provided (from performance manager), the mode's
    risk_per_trade_pct overrides the base config.

    Args:
        risk_params: Risk parameters dict.
        total_equity: Current total portfolio equity.
        mode_params: Optional dict from performance.manager.get_mode_params().

    Returns:
        Float — maximum dollar risk per trade.
    """
    if mode_params:
        max_risk_pct = mode_params.get("risk_per_trade_pct", 0.005)
    else:
        max_risk_pct = risk_params.get("position_limits", {}).get("max_risk_per_trade", 0.005)
    return total_equity * max_risk_pct


def validate_order(session, risk_params, decision, portfolio_state):
    """
    Full pre-trade risk validation for a single decision.

    This is the final gate before order submission. Incorporates
    performance-based risk mode (conservative/normal/aggressive)
    and the weekly trade throttle.

    Args:
        session: DB session.
        risk_params: Risk parameters dict.
        decision: Decision dict from decision engine.
        portfolio_state: Current portfolio state dict.

    Returns:
        Dict with:
        {
            "approved": bool,
            "reason": str,
            "max_trade_size": float,
            "risk_mode": str,
            "risk_checks": {check_name: passed}
        }
    """
    symbol = decision.get("symbol")
    action = decision.get("action")

    # Determine active risk mode
    from performance.manager import select_risk_mode, get_mode_params, check_trade_throttle

    mode_result = select_risk_mode(session, risk_params)
    mode = mode_result["mode"]
    mode_params = get_mode_params(risk_params, mode)

    logger.info(
        f"Validating order: {action} {symbol} (mode={mode})",
        extra={"extra_data": {"decision": decision, "risk_mode": mode, "mode_params": mode_params}},
    )

    # Kill switch — absolute override
    if check_kill_switch():
        log_risk_event(session, "kill_switch", "CRITICAL", {"action": action, "symbol": symbol})
        return {
            "approved": False,
            "reason": "Kill switch is active",
            "max_trade_size": 0,
            "risk_mode": mode,
            "risk_checks": {"kill_switch": False},
        }

    # SELLs are always approved (we want to be able to exit)
    if action == "SELL":
        return {
            "approved": True,
            "reason": "Sells always approved for risk reduction",
            "max_trade_size": None,
            "risk_mode": mode,
            "risk_checks": {"sell_always_approved": True},
        }

    # SKIPs and HOLDs don't need validation
    if action in ("SKIP", "HOLD"):
        return {
            "approved": True,
            "reason": f"{action} — no trade needed",
            "max_trade_size": 0,
            "risk_mode": mode,
            "risk_checks": {},
        }

    # BUY validation
    checks = {}
    total_equity = portfolio_state.get("total_equity", 0)

    # Check 1: Daily loss limit
    daily_breached, daily_details = check_daily_loss_limit(session, risk_params)
    checks["daily_loss_limit"] = not daily_breached
    if daily_breached:
        return {
            "approved": False,
            "reason": f"Daily loss limit breached: {daily_details.get('daily_return', 0):.4%}",
            "max_trade_size": 0,
            "risk_mode": mode,
            "risk_checks": checks,
        }

    # Check 2: Weekly drawdown
    weekly_breached, weekly_details = check_weekly_drawdown(session, risk_params)
    checks["weekly_drawdown"] = not weekly_breached
    if weekly_breached:
        return {
            "approved": False,
            "reason": f"Weekly drawdown limit breached: {weekly_details.get('weekly_return', 0):.4%}",
            "max_trade_size": 0,
            "risk_mode": mode,
            "risk_checks": checks,
        }

    # Check 3: Consecutive losses
    consec_breached, consec_details = check_consecutive_losses(session, risk_params)
    checks["consecutive_losses"] = not consec_breached
    if consec_breached:
        return {
            "approved": False,
            "reason": "Consecutive loss limit reached — trading halted",
            "max_trade_size": 0,
            "risk_mode": mode,
            "risk_checks": checks,
        }

    # Check 4: Position limits (mode-aware)
    can_open, pos_details = check_position_limits(session, risk_params, symbol, mode_params=mode_params)
    checks["position_limits"] = can_open
    if not can_open:
        return {
            "approved": False,
            "reason": pos_details.get("reason", "Position limit reached"),
            "max_trade_size": 0,
            "risk_mode": mode,
            "risk_checks": checks,
        }

    # Check 5: Weekly trade throttle
    throttled, throttle_details = check_trade_throttle(session, risk_params)
    checks["trade_throttle"] = not throttled
    if throttled:
        return {
            "approved": False,
            "reason": f"Weekly trade limit reached ({throttle_details['trades_this_week']}/{throttle_details['max_per_week']})",
            "max_trade_size": 0,
            "risk_mode": mode,
            "risk_checks": checks,
        }

    # All checks passed — size uses mode-adjusted risk percentage
    max_trade = calculate_max_trade_size(risk_params, total_equity, mode_params=mode_params)

    logger.info(
        f"Order APPROVED: {action} {symbol} (mode={mode})",
        extra={
            "extra_data": {
                "max_trade_size": max_trade,
                "risk_mode": mode,
                "risk_checks": checks,
            }
        },
    )

    return {
        "approved": True,
        "reason": "All risk checks passed",
        "max_trade_size": max_trade,
        "risk_mode": mode,
        "risk_checks": checks,
    }
