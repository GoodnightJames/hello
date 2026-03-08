"""
Weekly Report Generator — automated performance summary.

Generates a structured report every Sunday covering:
1. Portfolio P&L (week, month, all-time)
2. Trade log (all fills for the week)
3. Regime history (trend + volatility states)
4. Risk events (any throttles or limit breaches)
5. Position summary (current holdings)

Reports are saved as JSON and human-readable text to the reports/ directory.
"""

import json
import os
from datetime import datetime, timedelta

from core.logging import get_logger
from data.db import (
    Order,
    Decision,
    RiskEvent,
    PortfolioState,
    get_session,
    init_db,
)

logger = get_logger("review.weekly_report")

REPORTS_DIR = "reports"


def get_week_range(reference_date=None):
    """Get Monday-to-Friday date range for the most recent trading week."""
    if reference_date is None:
        reference_date = datetime.utcnow()

    # Go back to the most recent Monday
    days_since_monday = reference_date.weekday()
    monday = reference_date - timedelta(days=days_since_monday)
    monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    friday = monday + timedelta(days=4, hours=23, minutes=59, seconds=59)

    return monday, friday


def compute_pnl(session, week_start, week_end):
    """
    Compute P&L metrics for the week.

    Returns dict with weekly, inception-to-date returns.
    """
    # Get portfolio snapshots
    week_snapshots = (
        session.query(PortfolioState)
        .filter(PortfolioState.date >= week_start, PortfolioState.date <= week_end)
        .order_by(PortfolioState.date)
        .all()
    )

    # Get the snapshot just before the week (prior Friday close)
    prior_snapshot = (
        session.query(PortfolioState)
        .filter(PortfolioState.date < week_start)
        .order_by(PortfolioState.date.desc())
        .first()
    )

    # Get the very first snapshot (inception)
    first_snapshot = (
        session.query(PortfolioState)
        .order_by(PortfolioState.date)
        .first()
    )

    # Get the latest snapshot
    latest_snapshot = (
        session.query(PortfolioState)
        .order_by(PortfolioState.date.desc())
        .first()
    )

    result = {
        "week_start_equity": None,
        "week_end_equity": None,
        "weekly_pnl_dollars": 0.0,
        "weekly_pnl_pct": 0.0,
        "inception_equity": None,
        "current_equity": None,
        "inception_pnl_dollars": 0.0,
        "inception_pnl_pct": 0.0,
    }

    if prior_snapshot:
        result["week_start_equity"] = prior_snapshot.total_equity
    elif week_snapshots:
        result["week_start_equity"] = week_snapshots[0].total_equity

    if week_snapshots:
        result["week_end_equity"] = week_snapshots[-1].total_equity
    elif latest_snapshot:
        result["week_end_equity"] = latest_snapshot.total_equity

    if result["week_start_equity"] and result["week_end_equity"]:
        result["weekly_pnl_dollars"] = result["week_end_equity"] - result["week_start_equity"]
        if result["week_start_equity"] > 0:
            result["weekly_pnl_pct"] = result["weekly_pnl_dollars"] / result["week_start_equity"]

    if first_snapshot:
        result["inception_equity"] = first_snapshot.total_equity
    if latest_snapshot:
        result["current_equity"] = latest_snapshot.total_equity

    if result["inception_equity"] and result["current_equity"]:
        result["inception_pnl_dollars"] = result["current_equity"] - result["inception_equity"]
        if result["inception_equity"] > 0:
            result["inception_pnl_pct"] = result["inception_pnl_dollars"] / result["inception_equity"]

    return result


def get_trade_log(session, week_start, week_end):
    """Get all filled orders during the week."""
    orders = (
        session.query(Order)
        .filter(
            Order.status == "filled",
            Order.filled_at >= week_start,
            Order.filled_at <= week_end,
        )
        .order_by(Order.filled_at)
        .all()
    )

    trades = []
    for o in orders:
        trades.append({
            "order_id": o.id,
            "symbol": o.symbol,
            "side": o.side,
            "qty": o.filled_qty,
            "price": o.filled_price,
            "value": (o.filled_qty or 0) * (o.filled_price or 0),
            "filled_at": o.filled_at.isoformat() if o.filled_at else None,
        })

    return trades


def get_decision_log(session, week_start, week_end):
    """Get all decisions made during the week."""
    decisions = (
        session.query(Decision)
        .filter(Decision.date >= week_start, Decision.date <= week_end)
        .order_by(Decision.date)
        .all()
    )

    return [
        {
            "id": d.id,
            "strategy": d.strategy,
            "symbol": d.symbol,
            "action": d.action,
            "reason": d.reason,
            "risk_approved": d.risk_approved,
            "date": d.date.isoformat(),
        }
        for d in decisions
    ]


def get_risk_events_log(session, week_start, week_end):
    """Get all risk events during the week."""
    events = (
        session.query(RiskEvent)
        .filter(RiskEvent.created_at >= week_start, RiskEvent.created_at <= week_end)
        .order_by(RiskEvent.created_at)
        .all()
    )

    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "severity": e.severity,
            "details": json.loads(e.details_json) if e.details_json else {},
            "created_at": e.created_at.isoformat(),
        }
        for e in events
    ]


def get_position_summary(session):
    """Get current position summary from latest portfolio state."""
    state = (
        session.query(PortfolioState)
        .order_by(PortfolioState.date.desc())
        .first()
    )

    if state is None:
        return {"cash": 0, "total_equity": 0, "positions": {}}

    positions = json.loads(state.positions_json) if state.positions_json else {}
    return {
        "cash": state.cash,
        "total_equity": state.total_equity,
        "positions": positions,
    }


def compute_diagnostics(session, week_start, week_end):
    """
    Compute rich diagnostic metrics beyond basic P&L.

    Includes:
    - Sleeve NAV breakdown
    - Realized volatility (20-day)
    - Drawdown from peak
    - Turnover ratio
    - Hit rate by sleeve
    - Fee/slippage drag estimate
    - Current regime state
    - Rejected signals summary
    """
    from capital.manager import get_sleeve_summary, get_trade_performance
    from risk.sleeve_risk import (
        _get_recent_equity_series,
        compute_rolling_drawdown,
        compute_rolling_volatility,
        compute_turnover,
    )
    from risk.cost_model import estimate_round_trip_cost

    diagnostics = {}

    # Sleeve NAV breakdown
    try:
        sleeve_summary = get_sleeve_summary(session)
        diagnostics["sleeves"] = sleeve_summary
    except Exception:
        diagnostics["sleeves"] = {}

    # Realized vol and drawdown
    try:
        equity_series = _get_recent_equity_series(session, lookback_days=60)
        dd, peak, trough = compute_rolling_drawdown(equity_series)
        vol = compute_rolling_volatility(equity_series, window=20)
        diagnostics["risk_metrics"] = {
            "realized_vol_20d": round(vol, 4),
            "max_drawdown_60d": round(dd, 4),
            "peak_equity": round(peak, 2),
            "trough_equity": round(trough, 2),
        }
    except Exception:
        diagnostics["risk_metrics"] = {}

    # Turnover
    try:
        turnover, total_traded, nav = compute_turnover(session, lookback_days=7)
        diagnostics["turnover"] = {
            "weekly_turnover_ratio": round(turnover, 4),
            "total_traded": round(total_traded, 2),
            "nav": round(nav, 2),
        }
    except Exception:
        diagnostics["turnover"] = {}

    # Hit rate by sleeve
    try:
        from data.db import Trade
        cutoff = week_start
        week_trades = (
            session.query(Trade)
            .filter(Trade.exit_date >= cutoff, Trade.exit_date <= week_end)
            .all()
        )
        equity_trades = [t for t in week_trades if "/" not in t.symbol]
        crypto_trades = [t for t in week_trades if "/" in t.symbol]

        def _hit_rate(trades):
            if not trades:
                return {"trades": 0, "wins": 0, "losses": 0, "hit_rate": 0}
            wins = sum(1 for t in trades if t.is_win)
            return {
                "trades": len(trades),
                "wins": wins,
                "losses": len(trades) - wins,
                "hit_rate": round(wins / len(trades), 3) if trades else 0,
                "total_pnl": round(sum(t.realized_pnl for t in trades), 2),
                "avg_pnl": round(
                    sum(t.realized_pnl for t in trades) / len(trades), 2
                ) if trades else 0,
            }

        diagnostics["hit_rate_by_sleeve"] = {
            "equity": _hit_rate(equity_trades),
            "crypto": _hit_rate(crypto_trades),
            "total": _hit_rate(week_trades),
        }
    except Exception:
        diagnostics["hit_rate_by_sleeve"] = {}

    # Estimated fee drag
    try:
        orders = (
            session.query(Order)
            .filter(
                Order.status == "filled",
                Order.filled_at >= week_start,
                Order.filled_at <= week_end,
            )
            .all()
        )
        total_fee_estimate = 0.0
        for o in orders:
            notional = (o.filled_qty or 0) * (o.filled_price or 0)
            is_crypto = "/" in o.symbol
            # Rough one-way cost estimate
            if is_crypto:
                fee_bps = 25  # fee + spread estimate
            else:
                fee_bps = 2
            total_fee_estimate += notional * fee_bps / 10000

        diagnostics["estimated_fee_drag"] = {
            "weekly_fees_estimate": round(total_fee_estimate, 2),
            "orders_counted": len(orders),
        }
    except Exception:
        diagnostics["estimated_fee_drag"] = {}

    # Rejected signals
    try:
        decisions = (
            session.query(Decision)
            .filter(
                Decision.date >= week_start,
                Decision.date <= week_end,
                Decision.action == "SKIP",
            )
            .all()
        )
        diagnostics["rejected_signals"] = {
            "count": len(decisions),
            "reasons": [
                {"symbol": d.symbol, "reason": d.reason[:80]}
                for d in decisions[:10]  # Top 10
            ],
        }
    except Exception:
        diagnostics["rejected_signals"] = {}

    return diagnostics


def generate_weekly_report(reference_date=None):
    """
    Generate the full weekly report with enhanced diagnostics.

    Includes:
    - Standard P&L, trades, decisions, risk events, positions
    - NEW: sleeve NAV, vol, drawdown, turnover, hit rates, fee drag, rejected signals

    Args:
        reference_date: Date to generate report for (defaults to now).

    Returns:
        Report dict and saves to reports/ directory.
    """
    logger.info("Generating weekly report")

    init_db()
    session = get_session()

    try:
        week_start, week_end = get_week_range(reference_date)

        report = {
            "report_type": "weekly",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "pnl": compute_pnl(session, week_start, week_end),
            "trades": get_trade_log(session, week_start, week_end),
            "decisions": get_decision_log(session, week_start, week_end),
            "risk_events": get_risk_events_log(session, week_start, week_end),
            "positions": get_position_summary(session),
            "diagnostics": compute_diagnostics(session, week_start, week_end),
        }

        # Save as JSON
        os.makedirs(REPORTS_DIR, exist_ok=True)
        week_label = week_start.strftime("%Y-%m-%d")
        json_path = os.path.join(REPORTS_DIR, f"weekly_{week_label}.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save as human-readable text
        text_path = os.path.join(REPORTS_DIR, f"weekly_{week_label}.txt")
        with open(text_path, "w") as f:
            f.write(_format_text_report(report))

        logger.info(
            "Weekly report generated",
            extra={
                "extra_data": {
                    "week": week_label,
                    "json_path": json_path,
                    "text_path": text_path,
                    "trade_count": len(report["trades"]),
                    "decision_count": len(report["decisions"]),
                    "risk_event_count": len(report["risk_events"]),
                }
            },
        )
        return report

    finally:
        session.close()


def _format_text_report(report):
    """Format report dict as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("WEEKLY TRADING REPORT")
    lines.append(f"Week: {report['week_start'][:10]} to {report['week_end'][:10]}")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("=" * 60)

    # P&L Section
    pnl = report["pnl"]
    lines.append("")
    lines.append("--- P&L SUMMARY ---")
    lines.append(f"  Weekly P&L:     ${pnl['weekly_pnl_dollars']:>10.2f}  ({pnl['weekly_pnl_pct']:>+.2%})")
    lines.append(f"  Inception P&L:  ${pnl['inception_pnl_dollars']:>10.2f}  ({pnl['inception_pnl_pct']:>+.2%})")
    lines.append(f"  Current Equity: ${pnl['current_equity'] or 0:>10.2f}")

    # Positions
    pos = report["positions"]
    lines.append("")
    lines.append("--- CURRENT POSITIONS ---")
    lines.append(f"  Cash: ${pos['cash']:>10.2f}")
    if pos["positions"]:
        for symbol, qty in pos["positions"].items():
            lines.append(f"  {symbol:6s}: {qty} shares")
    else:
        lines.append("  (no positions)")

    # Trades
    lines.append("")
    lines.append(f"--- TRADE LOG ({len(report['trades'])} fills) ---")
    for t in report["trades"]:
        lines.append(
            f"  {t['filled_at'][:10]}  {t['side'].upper():4s}  "
            f"{t['qty']:>6.0f}  {t['symbol']:6s}  "
            f"@ ${t['price']:>8.2f}  = ${t['value']:>10.2f}"
        )
    if not report["trades"]:
        lines.append("  (no trades this week)")

    # Decisions
    lines.append("")
    lines.append(f"--- DECISIONS ({len(report['decisions'])}) ---")
    for d in report["decisions"]:
        approved = "OK" if d["risk_approved"] else "BLOCKED"
        lines.append(f"  {d['date'][:10]}  {d['action']:5s}  {d['symbol']:6s}  [{approved}]  {d['reason'][:60]}")
    if not report["decisions"]:
        lines.append("  (no decisions this week)")

    # Risk Events
    lines.append("")
    lines.append(f"--- RISK EVENTS ({len(report['risk_events'])}) ---")
    for e in report["risk_events"]:
        lines.append(f"  {e['created_at'][:10]}  [{e['severity']}]  {e['event_type']}")
    if not report["risk_events"]:
        lines.append("  (no risk events this week)")

    # Diagnostics
    diag = report.get("diagnostics", {})
    if diag:
        lines.append("")
        lines.append("--- DIAGNOSTICS ---")

        # Risk metrics
        risk = diag.get("risk_metrics", {})
        if risk:
            lines.append(f"  Realized Vol (20d):  {risk.get('realized_vol_20d', 0):.1%}")
            lines.append(f"  Max Drawdown (60d):  {risk.get('max_drawdown_60d', 0):.1%}")

        # Turnover
        turn = diag.get("turnover", {})
        if turn:
            lines.append(f"  Weekly Turnover:     {turn.get('weekly_turnover_ratio', 0):.2f}x")
            lines.append(f"  Total Traded:        ${turn.get('total_traded', 0):,.2f}")

        # Hit rate by sleeve
        hr = diag.get("hit_rate_by_sleeve", {})
        for sleeve_name in ("equity", "crypto", "total"):
            s = hr.get(sleeve_name, {})
            if s.get("trades", 0) > 0:
                lines.append(
                    f"  {sleeve_name.title()} Hit Rate: "
                    f"{s['wins']}/{s['trades']} ({s.get('hit_rate', 0):.0%}) "
                    f"P&L: ${s.get('total_pnl', 0):+.2f}"
                )

        # Fee drag
        fees = diag.get("estimated_fee_drag", {})
        if fees:
            lines.append(f"  Est. Fee Drag:       ${fees.get('weekly_fees_estimate', 0):.2f}")

        # Sleeve balances
        sleeves = diag.get("sleeves", {})
        for name, data in sleeves.items():
            if isinstance(data, dict):
                lines.append(
                    f"  Sleeve [{name}]:  cash=${data.get('cash', 0):.2f}  "
                    f"deposited=${data.get('total_deposited', 0):.2f}  "
                    f"spent=${data.get('total_spent', 0):.2f}"
                )

        # Rejected signals
        rej = diag.get("rejected_signals", {})
        if rej.get("count", 0) > 0:
            lines.append(f"  Rejected Signals:    {rej['count']}")
            for r in rej.get("reasons", [])[:3]:
                lines.append(f"    {r['symbol']}: {r['reason'][:60]}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_weekly_report()
    print(_format_text_report(report))
