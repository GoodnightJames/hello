"""
Trading Engine Status — quick snapshot for small-account accumulation mode.

The key metric: total deposited vs current value.
That's YOUR money in vs what it's worth now.

Usage:
    python status.py
"""

import json
from datetime import datetime, timedelta

from dotenv import load_dotenv

from core.logging import get_logger
from data.db import init_db, get_session, Order, PortfolioState, RiskEvent
from capital.manager import get_accumulation_summary
from execution.alpaca_broker import get_account_with_retry as get_account, get_positions_with_retry as get_positions

load_dotenv()
logger = get_logger("status")


def get_trade_stats(session):
    """Get trading statistics for the paper trading period."""
    all_fills = (
        session.query(Order)
        .filter(Order.status == "filled")
        .order_by(Order.filled_at.asc())
        .all()
    )

    if not all_fills:
        return {
            "total_trades": 0,
            "buys": 0,
            "sells": 0,
            "first_trade": None,
            "days_active": 0,
        }

    buys = [o for o in all_fills if o.side == "buy"]
    sells = [o for o in all_fills if o.side == "sell"]
    first_trade = all_fills[0].filled_at
    days_active = (datetime.utcnow() - first_trade).days if first_trade else 0

    return {
        "total_trades": len(all_fills),
        "buys": len(buys),
        "sells": len(sells),
        "first_trade": str(first_trade) if first_trade else None,
        "days_active": days_active,
    }


def get_equity_history(session):
    """Get equity curve summary."""
    snapshots = (
        session.query(PortfolioState)
        .order_by(PortfolioState.date.asc())
        .all()
    )

    if not snapshots:
        return {"snapshots": 0, "peak": 0, "trough": 0, "current": 0, "drawdown": 0}

    equities = [s.total_equity for s in snapshots]
    peak = max(equities)
    current = equities[-1]
    drawdown = (peak - current) / peak if peak > 0 else 0

    return {
        "snapshots": len(snapshots),
        "start_equity": equities[0],
        "peak": peak,
        "current": current,
        "drawdown_pct": round(drawdown * 100, 2),
        "total_return_pct": round(((current - equities[0]) / equities[0]) * 100, 2) if equities[0] > 0 else 0,
    }


def get_risk_events(session, days=30):
    """Get recent risk events."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    events = (
        session.query(RiskEvent)
        .filter(RiskEvent.created_at >= cutoff)
        .order_by(RiskEvent.created_at.desc())
        .all()
    )

    return [
        {
            "type": e.event_type,
            "severity": e.severity,
            "date": str(e.created_at),
        }
        for e in events
    ]


def check_go_live_readiness(session, trade_stats, equity_history, risk_events):
    """
    Check if paper trading meets go-live criteria.

    Returns:
        Dict with each criterion and pass/fail status.
    """
    import yaml
    with open("config/risk_params.yaml", "r") as f:
        risk_params = yaml.safe_load(f)

    criteria = risk_params.get("go_live_criteria", {})

    checks = {}

    # 1. Minimum days
    min_days = criteria.get("min_days_trading", 21)
    days_active = trade_stats.get("days_active", 0)
    checks["min_days"] = {
        "required": min_days,
        "actual": days_active,
        "passed": days_active >= min_days,
        "label": f"Trading for {days_active}/{min_days} days",
    }

    # 2. Minimum trades
    min_trades = criteria.get("min_trades", 6)
    total_trades = trade_stats.get("total_trades", 0)
    checks["min_trades"] = {
        "required": min_trades,
        "actual": total_trades,
        "passed": total_trades >= min_trades,
        "label": f"{total_trades}/{min_trades} trades completed",
    }

    # 3. Max drawdown
    max_dd = criteria.get("max_drawdown", 0.15)
    actual_dd = equity_history.get("drawdown_pct", 0) / 100
    checks["max_drawdown"] = {
        "required": f"{max_dd:.0%}",
        "actual": f"{actual_dd:.2%}",
        "passed": actual_dd <= max_dd,
        "label": f"Max drawdown {actual_dd:.2%} (limit {max_dd:.0%})",
    }

    # 4. Daily loss events
    max_events = criteria.get("max_daily_loss_events", 3)
    loss_events = len([e for e in risk_events if e["type"] == "daily_loss_limit"])
    checks["daily_loss_events"] = {
        "required": max_events,
        "actual": loss_events,
        "passed": loss_events <= max_events,
        "label": f"{loss_events}/{max_events} daily loss shutdowns",
    }

    all_passed = all(c["passed"] for c in checks.values())

    return {
        "ready": all_passed,
        "checks": checks,
    }


def print_status():
    """Print a formatted status report."""
    print("\n" + "=" * 60)
    print("  TRADING ENGINE STATUS (Accumulation Mode)")
    print("  $100/week — get rich or go broke")
    print("=" * 60)

    # ── The number that matters ──────────────────────────────────────────
    init_db()
    session = get_session()

    summary = get_accumulation_summary(session)
    print(f"\n--- YOUR MONEY ---")
    print(f"  Total deposited: ${summary['total_deposited']:>10,.2f}")
    print(f"  Current value:   ${summary['current_equity']:>10,.2f}")
    gain = summary['gain_loss']
    marker = "+" if gain >= 0 else ""
    print(f"  Gain/Loss:       {marker}${gain:>10,.2f} ({marker}{summary['gain_loss_pct']:.1f}%)")
    print(f"  Weeks active:    {summary['weeks_active']}")

    if summary['weeks_active'] > 0:
        avg_per_week = summary['total_deposited'] / summary['weeks_active']
        print(f"  Avg deposit/wk:  ${avg_per_week:>10,.2f}")

    # Alpaca account
    try:
        acct = get_account()
        print(f"\n--- Alpaca Account ---")
        print(f"  Status:       {acct['status']}")
        print(f"  Cash:         ${acct['cash']:>12,.2f}")
        print(f"  Equity:       ${acct['equity']:>12,.2f}")
        print(f"  Buying Power: ${acct['buying_power']:>12,.2f}")
    except Exception as e:
        print(f"\n--- Alpaca Account ---")
        print(f"  ERROR: {e}")

    # Positions
    try:
        positions = get_positions()
        print(f"\n--- Open Positions ({len(positions)}) ---")
        if positions:
            total_value = 0
            total_pl = 0
            for sym, pos in sorted(positions.items()):
                pl_pct = (pos["unrealized_pl"] / (pos["avg_entry"] * pos["qty"])) * 100 if pos["avg_entry"] > 0 else 0
                marker = "+" if pos["unrealized_pl"] >= 0 else ""
                print(f"  {sym:6s}  {pos['qty']:>8.2f} shares  "
                      f"@ ${pos['avg_entry']:>8.2f}  "
                      f"now ${pos['current_price']:>8.2f}  "
                      f"P&L: {marker}${pos['unrealized_pl']:>8.2f} ({marker}{pl_pct:.1f}%)")
                total_value += pos["market_value"]
                total_pl += pos["unrealized_pl"]
            print(f"  {'':6s}  Total value: ${total_value:>12,.2f}  P&L: ${total_pl:>+12,.2f}")
        else:
            print("  (no positions)")
    except Exception as e:
        print(f"\n--- Positions ---")
        print(f"  ERROR: {e}")

    # Trade stats
    trade_stats = get_trade_stats(session)
    print(f"\n--- Trade History ---")
    print(f"  Total fills:  {trade_stats['total_trades']}")
    print(f"  Buys:         {trade_stats['buys']}")
    print(f"  Sells:        {trade_stats['sells']}")
    print(f"  First trade:  {trade_stats['first_trade'] or 'None yet'}")
    print(f"  Days active:  {trade_stats['days_active']}")

    equity_history = get_equity_history(session)
    print(f"\n--- Equity Curve ---")
    if equity_history["snapshots"] > 0:
        print(f"  Start:        ${equity_history['start_equity']:>12,.2f}")
        print(f"  Peak:         ${equity_history['peak']:>12,.2f}")
        print(f"  Current:      ${equity_history['current']:>12,.2f}")
        print(f"  Drawdown:     {equity_history['drawdown_pct']:.2f}%")
    else:
        print("  No snapshots yet")

    risk_events = get_risk_events(session)
    print(f"\n--- Risk Events (last 30 days) ---")
    if risk_events:
        for e in risk_events[:5]:
            print(f"  [{e['severity']:8s}] {e['type']}  ({e['date'][:19]})")
        if len(risk_events) > 5:
            print(f"  ... and {len(risk_events) - 5} more")
    else:
        print("  None")

    # Go-live readiness
    readiness = check_go_live_readiness(session, trade_stats, equity_history, risk_events)
    print(f"\n--- Go-Live Checklist ---")
    for key, check in readiness["checks"].items():
        icon = "PASS" if check["passed"] else "FAIL"
        print(f"  [{icon}] {check['label']}")

    if readiness["ready"]:
        print(f"\n  >>> ALL CRITERIA MET — Ready for live trading! <<<")
    else:
        passed = sum(1 for c in readiness["checks"].values() if c["passed"])
        total = len(readiness["checks"])
        print(f"\n  >>> {passed}/{total} criteria met — keep paper trading <<<")

    # Projection
    if summary['weeks_active'] > 0 and summary['total_deposited'] > 0:
        weekly_return = summary['gain_loss_pct'] / summary['weeks_active'] / 100 if summary['weeks_active'] > 1 else 0
        print(f"\n--- Projection (if current pace holds) ---")
        deposit_per_week = 100
        current = summary['current_equity']
        for label, weeks in [("6 months", 26), ("1 year", 52), ("2 years", 104), ("5 years", 260)]:
            projected = current
            for _ in range(weeks):
                projected = projected * (1 + weekly_return) + deposit_per_week
            total_in = summary['total_deposited'] + (deposit_per_week * weeks)
            print(f"  {label:10s}: ${projected:>10,.0f}  (${total_in:,.0f} deposited)")

    session.close()
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print_status()
