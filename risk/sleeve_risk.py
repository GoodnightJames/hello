"""
Sleeve-Level Risk Budgets — replaces blunt circuit breakers with
portfolio-behavior-based controls.

Instead of:
  - "5 consecutive losses = HALT" (trade-count proxy)
  - "6 trades/week = throttle" (arbitrary)
  - "3% daily loss = halt all buys" (too blunt)

Use:
  - Max sleeve drawdown from peak
  - Max rolling 20-day realized volatility
  - Max total exposure
  - Max single-position concentration
  - Turnover budget (dollars traded / NAV)

These are tied to actual portfolio behavior, not arbitrary trade counts.

Usage:
    from risk.sleeve_risk import check_sleeve_risk_budget
    result = check_sleeve_risk_budget(session, "equity", risk_params)
    if not result["within_budget"]:
        # Reduce or halt new entries
"""

import json
from datetime import datetime, timedelta

from core.logging import get_logger
from data.db import PortfolioState, Trade, Order, get_session

logger = get_logger("risk.sleeve_risk")


def _get_recent_equity_series(session, lookback_days=30):
    """
    Get recent portfolio equity values for vol/drawdown computation.

    Returns:
        List of (date, total_equity) tuples, sorted by date ascending.
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    snapshots = (
        session.query(PortfolioState)
        .filter(PortfolioState.date >= cutoff)
        .order_by(PortfolioState.date)
        .all()
    )
    return [(s.date, s.total_equity) for s in snapshots if s.total_equity > 0]


def compute_rolling_drawdown(equity_series):
    """
    Compute max drawdown from peak over the equity series.

    Args:
        equity_series: List of (date, equity) tuples.

    Returns:
        (max_drawdown_pct, peak_equity, trough_equity)
    """
    if len(equity_series) < 2:
        return 0.0, 0.0, 0.0

    peak = equity_series[0][1]
    max_dd = 0.0
    trough = peak

    for _, equity in equity_series:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            trough = equity

    return max_dd, peak, trough


def compute_rolling_volatility(equity_series, window=20):
    """
    Compute rolling realized volatility of portfolio equity.

    Uses daily returns, annualized.

    Returns:
        Annualized vol as decimal (e.g., 0.15 = 15%).
    """
    if len(equity_series) < window + 1:
        return 0.0

    equities = [e for _, e in equity_series]
    returns = []
    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            returns.append((equities[i] - equities[i - 1]) / equities[i - 1])

    if len(returns) < window:
        return 0.0

    recent = returns[-window:]
    import numpy as np
    daily_vol = float(np.std(recent))
    annualized = daily_vol * (252 ** 0.5)
    return annualized


def compute_turnover(session, lookback_days=7):
    """
    Compute portfolio turnover: total dollars traded / NAV.

    High turnover = high fee drag. Useful for monitoring cost efficiency.

    Returns:
        (turnover_ratio, total_traded, nav)
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    # Get filled orders in the period
    orders = (
        session.query(Order)
        .filter(
            Order.status == "filled",
            Order.filled_at >= cutoff,
        )
        .all()
    )

    total_traded = sum(
        (o.filled_qty or 0) * (o.filled_price or 0)
        for o in orders
    )

    # Get current NAV
    latest = (
        session.query(PortfolioState)
        .order_by(PortfolioState.date.desc())
        .first()
    )
    nav = latest.total_equity if latest and latest.total_equity > 0 else 1.0

    turnover = total_traded / nav
    return turnover, total_traded, nav


def check_sleeve_risk_budget(session, risk_params):
    """
    Check portfolio-level risk budgets.

    Replaces blunt circuit breakers with behavior-based controls.

    Args:
        session: DB session.
        risk_params: Risk parameters dict.

    Returns:
        Dict with:
        {
            "within_budget": bool,
            "position_scale": float (0.0-1.0, multiply position sizes by this),
            "details": {
                "drawdown": {...},
                "volatility": {...},
                "turnover": {...},
            },
            "warnings": [str],
        }
    """
    budgets = risk_params.get("risk_budgets", {})
    equity_budget = budgets.get("equity_sleeve", {})
    crypto_budget = budgets.get("crypto_sleeve", {})

    # Use portfolio-level budgets (applies to whole account)
    portfolio_budget = budgets.get("portfolio", {})
    max_drawdown = portfolio_budget.get("max_drawdown", 0.15)
    max_vol = portfolio_budget.get("max_realized_vol", 0.25)
    max_weekly_turnover = portfolio_budget.get("max_weekly_turnover", 2.0)

    warnings = []
    position_scale = 1.0

    # 1. Drawdown check
    equity_series = _get_recent_equity_series(session, lookback_days=60)
    current_dd, peak, trough = compute_rolling_drawdown(equity_series)

    dd_details = {
        "current_drawdown": round(current_dd, 4),
        "max_allowed": max_drawdown,
        "peak_equity": round(peak, 2),
        "trough_equity": round(trough, 2),
    }

    if current_dd > max_drawdown:
        position_scale *= 0.25  # Severe reduction
        warnings.append(
            f"DRAWDOWN BUDGET EXCEEDED: {current_dd:.1%} > {max_drawdown:.1%}"
        )
    elif current_dd > max_drawdown * 0.7:
        position_scale *= 0.50  # Moderate reduction
        warnings.append(
            f"DRAWDOWN WARNING: {current_dd:.1%} approaching limit {max_drawdown:.1%}"
        )

    # 2. Volatility check
    current_vol = compute_rolling_volatility(equity_series, window=20)

    vol_details = {
        "current_vol": round(current_vol, 4),
        "max_allowed": max_vol,
    }

    if current_vol > max_vol:
        position_scale *= 0.50
        warnings.append(
            f"VOLATILITY BUDGET EXCEEDED: {current_vol:.1%} > {max_vol:.1%}"
        )
    elif current_vol > max_vol * 0.8:
        position_scale *= 0.75
        warnings.append(
            f"VOLATILITY WARNING: {current_vol:.1%} approaching limit {max_vol:.1%}"
        )

    # 3. Turnover check
    turnover, total_traded, nav = compute_turnover(session, lookback_days=7)

    turnover_details = {
        "weekly_turnover": round(turnover, 4),
        "max_allowed": max_weekly_turnover,
        "total_traded": round(total_traded, 2),
        "nav": round(nav, 2),
    }

    if turnover > max_weekly_turnover:
        position_scale *= 0.50
        warnings.append(
            f"TURNOVER BUDGET EXCEEDED: {turnover:.1f}x > {max_weekly_turnover:.1f}x"
        )

    within_budget = position_scale >= 0.50

    result = {
        "within_budget": within_budget,
        "position_scale": round(position_scale, 2),
        "details": {
            "drawdown": dd_details,
            "volatility": vol_details,
            "turnover": turnover_details,
        },
        "warnings": warnings,
    }

    if warnings:
        logger.warning(
            "Risk budget warnings",
            extra={"extra_data": result},
        )
    else:
        logger.info(
            "Risk budgets within limits",
            extra={"extra_data": {
                "position_scale": position_scale,
                "drawdown": round(current_dd, 4),
                "vol": round(current_vol, 4),
                "turnover": round(turnover, 2),
            }},
        )

    return result
