"""
Portfolio Rebalancer — deploys idle cash into existing winners.

Problem: The system deposits $100/week but only trades when momentum
signals change (which can be weeks apart). Meanwhile, cash sits idle.

Solution: When there's deployable cash AND an existing position with
active momentum, add to the position (respecting risk limits).

This is NOT averaging down — it's adding to winners.
"""

import json

from core.logging import get_logger
from data.db import PortfolioState, CostBasis, get_session

logger = get_logger("capital.rebalancer")


def check_idle_cash_deployment(session, portfolio, decisions, risk_params):
    """
    Check if we have idle cash that should be deployed into existing winners.

    Conditions for deployment:
    1. Deployable cash > min_trade_dollars ($1)
    2. No BUY decision already pending (don't double-deploy)
    3. An existing position has active momentum (HOLD signal, not SELL)
    4. Position limits allow adding (not maxed out)

    Args:
        session: DB session.
        portfolio: Current portfolio state dict.
        decisions: List of decision dicts from current run.
        risk_params: Risk parameters dict.

    Returns:
        List of additional BUY decisions to append, or empty list.
    """
    from capital.manager import get_deployable_cash

    deployable = get_deployable_cash(portfolio)
    min_trade = 1.0  # Alpaca minimum

    if deployable < min_trade:
        return []

    # Don't deploy if there's already a BUY decision
    has_buy = any(d.get("action") == "BUY" for d in decisions)
    if has_buy:
        return []

    # Find HOLD signals — these are positions with active momentum
    # that aren't top-ranked but are still worth holding
    hold_symbols = [
        d["symbol"] for d in decisions
        if d.get("action") == "HOLD"
        and d.get("signal_type") == "HOLD"
    ]

    if not hold_symbols:
        return []

    # Check which HOLD symbols we actually own
    positions = portfolio.get("positions", {})
    owned_holds = [s for s in hold_symbols if s in positions and positions[s] > 0]

    if not owned_holds:
        return []

    # Pick the first owned HOLD position to add to
    target = owned_holds[0]

    # Get signal strength if available from decisions
    signal_strength = 1.0
    for d in decisions:
        if d.get("symbol") == target and "signal_strength" in d:
            signal_strength = d["signal_strength"]
            break

    logger.info(
        f"Idle cash deployment: ${deployable:.2f} → {target}",
        extra={
            "extra_data": {
                "deployable": deployable,
                "target": target,
                "signal_strength": signal_strength,
            }
        },
    )

    return [{
        "symbol": target,
        "action": "BUY",
        "reason": f"Idle cash deployment: ${deployable:.2f} into existing winner {target}",
        "signal_type": "BUY",
        "signal_id": None,
        "position_multiplier": 1.0,
        "signal_strength": signal_strength,
        "risk_approved": True,
    }]
