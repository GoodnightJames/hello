"""
Backtest Simulator — replay historical data through the decision pipeline
and simulate portfolio growth over time.

Unlike the replay engine (which validates consistency), this module
simulates actual portfolio evolution: deposits, buys, sells, and tracks
the equity curve, drawdown, and trade distribution.

Key properties:
- Uses the SAME decision logic as live trading (score_dual_momentum)
- No lookahead bias (build_features_as_of)
- Simulates weekly deposits and sleeve accounting
- Tracks exit path distribution (rank_drop vs trailing_stop)

Usage:
    python -m replay.backtest --start 2025-01-01 --end 2025-12-31

Output:
    - Equity curve (per-date portfolio value)
    - Trade list with entry/exit reasons
    - Exit path distribution
    - Drawdown profile
    - Sleeve allocation history
"""

import json
import os
from datetime import datetime, timedelta

import pandas as pd

from core.logging import get_logger
from replay.engine import load_historical_prices, build_features_as_of
from research.momentum_scorer import score_dual_momentum
from research.regime import classify_regime

logger = get_logger("replay.backtest")

REPORTS_DIR = "reports"


def _load_strategy_config():
    """Load momentum strategy config."""
    import yaml
    with open("config/strategies/momentum_v1.yaml", "r") as f:
        return yaml.safe_load(f)


def _load_risk_params():
    """Load risk params for regime classification."""
    import yaml
    with open("config/risk_params.yaml", "r") as f:
        return yaml.safe_load(f)


def _get_fridays(start_date, end_date):
    """Generate all Fridays (rebalance days) in a date range."""
    current = start_date
    # Advance to first Friday
    while current.weekday() != 4:
        current += timedelta(days=1)
    fridays = []
    while current <= end_date:
        fridays.append(current)
        current += timedelta(days=7)
    return fridays


def run_backtest(start_date, end_date, weekly_deposit=100.0, initial_capital=0.0):
    """
    Run a backtest from start_date to end_date.

    Simulates:
    - Weekly $100 deposits (Monday)
    - Friday rebalance via score_dual_momentum
    - Regime filtering
    - Equal-weight top-3 allocation with 40% cap
    - 8% trailing stop (checked weekly at rebalance)
    - Sleeve accounting (70/30 equity/crypto)

    Note: Crypto DCA is NOT simulated (no intraday data in daily bars).
    This backtest covers the equity momentum sleeve only.

    Args:
        start_date: datetime — backtest start
        end_date: datetime — backtest end
        weekly_deposit: float — weekly deposit amount
        initial_capital: float — starting capital

    Returns:
        Dict with equity_curve, trades, exit_distribution, drawdown, etc.
    """
    from data.db import init_db, get_session

    init_db()
    session = get_session()

    strategy_config = _load_strategy_config()
    risk_params = _load_risk_params()
    instruments = strategy_config.get("instruments", {})
    risk_assets = instruments.get("risk_assets", [])
    safe_assets = instruments.get("safe_assets", [])
    all_symbols = list(set(risk_assets + safe_assets + ["SPY"]))

    # Trailing stop config
    stop_config = risk_params.get("trailing_stop", {})
    trailing_stop_enabled = stop_config.get("enabled", True)
    stop_pct = stop_config.get("stop_pct", 0.08)
    min_gain_to_activate = stop_config.get("min_gain_to_activate", 0.03)

    # Max per-position cap
    max_weight = 0.40

    # Portfolio state
    equity_sleeve_cash = initial_capital * 0.70
    positions = {}  # {symbol: {"qty": float, "entry_price": float, "high_price": float}}
    equity_curve = []
    trades = []
    exit_counts = {"rank_drop": 0, "trailing_stop": 0, "regime_redirect": 0}
    sleeve_history = []

    # Get all rebalance dates (Fridays)
    rebalance_dates = _get_fridays(start_date, end_date)

    if not rebalance_dates:
        logger.warning("No rebalance dates in range")
        return {"error": "No rebalance dates in range"}

    # Track deposits (Monday before each Friday)
    last_deposit_week = None

    logger.info(
        f"Backtest starting: {start_date.date()} to {end_date.date()}, "
        f"{len(rebalance_dates)} rebalance dates",
    )

    try:
        for rebalance_date in rebalance_dates:
            # Weekly deposit (simulate Monday deposit)
            week_num = rebalance_date.isocalendar()[1]
            if last_deposit_week != week_num:
                equity_deposit = round(weekly_deposit * 0.70, 2)
                equity_sleeve_cash += equity_deposit
                last_deposit_week = week_num

            # Build features as-of this date (no lookahead)
            features = build_features_as_of(session, all_symbols, rebalance_date)
            if features["prices"].empty:
                # Record equity curve even if no data
                portfolio_value = _compute_portfolio_value(
                    equity_sleeve_cash, positions, features
                )
                equity_curve.append({
                    "date": rebalance_date.isoformat()[:10],
                    "equity": portfolio_value,
                    "cash": equity_sleeve_cash,
                    "positions": len(positions),
                })
                continue

            prices = features["prices"]
            latest_prices = {}
            if not prices.empty:
                latest_row = prices.iloc[-1]
                latest_prices = {
                    sym: float(latest_row[sym])
                    for sym in latest_row.index
                    if sym in prices.columns and float(latest_row[sym]) > 0
                }

            # Check trailing stops before rebalancing
            if trailing_stop_enabled:
                stopped_out = []
                for sym, pos in list(positions.items()):
                    current_price = latest_prices.get(sym)
                    if current_price is None:
                        continue

                    # Update high-water mark
                    if current_price > pos["high_price"]:
                        pos["high_price"] = current_price

                    gain_from_entry = (pos["high_price"] - pos["entry_price"]) / pos["entry_price"]
                    if gain_from_entry < min_gain_to_activate:
                        continue

                    stop_price = pos["high_price"] * (1 - stop_pct)
                    if current_price <= stop_price:
                        # Trailing stop triggered
                        proceeds = pos["qty"] * current_price
                        equity_sleeve_cash += proceeds
                        pnl = (current_price - pos["entry_price"]) * pos["qty"]

                        trades.append({
                            "symbol": sym,
                            "action": "SELL",
                            "exit_path": "trailing_stop",
                            "date": rebalance_date.isoformat()[:10],
                            "entry_price": pos["entry_price"],
                            "exit_price": current_price,
                            "qty": pos["qty"],
                            "pnl": round(pnl, 2),
                        })
                        exit_counts["trailing_stop"] += 1
                        stopped_out.append(sym)

                for sym in stopped_out:
                    del positions[sym]

            # Run scoring
            signals = score_dual_momentum(features, strategy_config)

            # Classify regime
            regime = classify_regime(features, risk_params)
            allow_entries = regime.get("allow_new_entries", True)
            trend = regime.get("trend", {})

            # Determine target holdings
            buy_signals = [s for s in signals if s["signal_type"] == "BUY"]
            sell_signals = [s for s in signals if s["signal_type"] == "SELL"]

            # Sell positions that lost momentum (rank drop)
            for sig in sell_signals:
                sym = sig["symbol"]
                if sym in positions:
                    current_price = latest_prices.get(sym)
                    if current_price is None:
                        continue
                    pos = positions[sym]
                    proceeds = pos["qty"] * current_price
                    equity_sleeve_cash += proceeds
                    pnl = (current_price - pos["entry_price"]) * pos["qty"]

                    trades.append({
                        "symbol": sym,
                        "action": "SELL",
                        "exit_path": "rank_drop",
                        "date": rebalance_date.isoformat()[:10],
                        "entry_price": pos["entry_price"],
                        "exit_price": current_price,
                        "qty": pos["qty"],
                        "pnl": round(pnl, 2),
                    })
                    exit_counts["rank_drop"] += 1
                    del positions[sym]

            # Buy new positions
            if allow_entries and buy_signals:
                # Filter out symbols we already hold
                new_buys = [s for s in buy_signals if s["symbol"] not in positions]
                n_buys = len(new_buys) + len(positions)  # total target positions
                if n_buys > 0:
                    weight = min(1.0 / n_buys, max_weight)
                    for sig in new_buys:
                        sym = sig["symbol"]
                        price = latest_prices.get(sym)
                        if price is None or price <= 0:
                            continue
                        allocation = equity_sleeve_cash * weight
                        if allocation < 1.0:
                            continue
                        qty = allocation / price
                        equity_sleeve_cash -= allocation

                        positions[sym] = {
                            "qty": qty,
                            "entry_price": price,
                            "high_price": price,
                        }
                        trades.append({
                            "symbol": sym,
                            "action": "BUY",
                            "exit_path": None,
                            "date": rebalance_date.isoformat()[:10],
                            "entry_price": price,
                            "exit_price": None,
                            "qty": round(qty, 6),
                            "pnl": None,
                        })

            elif not allow_entries:
                # Regime risk-off — check if we should redirect to safe assets
                if not positions:
                    # Rank safe assets by 12m return
                    safe_candidates = []
                    benchmark = strategy_config.get("signals", {}).get("benchmark", "SHY")
                    for sym in safe_assets:
                        if sym == benchmark:
                            continue
                        price = latest_prices.get(sym)
                        ret_12m = features.get("returns", {}).get("12m")
                        if ret_12m is not None and not ret_12m.empty and sym in ret_12m.columns:
                            r = float(ret_12m.iloc[-1][sym])
                            if price and price > 0:
                                safe_candidates.append((sym, r, price))

                    if safe_candidates:
                        safe_candidates.sort(key=lambda x: x[1], reverse=True)
                        sym, _, price = safe_candidates[0]
                        allocation = equity_sleeve_cash * max_weight
                        if allocation >= 1.0:
                            qty = allocation / price
                            equity_sleeve_cash -= allocation
                            positions[sym] = {
                                "qty": qty,
                                "entry_price": price,
                                "high_price": price,
                            }
                            trades.append({
                                "symbol": sym,
                                "action": "BUY",
                                "exit_path": None,
                                "date": rebalance_date.isoformat()[:10],
                                "entry_price": price,
                                "exit_price": None,
                                "qty": round(qty, 6),
                                "pnl": None,
                            })
                            exit_counts["regime_redirect"] += 1

            # Compute portfolio value
            portfolio_value = equity_sleeve_cash
            for sym, pos in positions.items():
                price = latest_prices.get(sym, pos["entry_price"])
                portfolio_value += pos["qty"] * price

            equity_curve.append({
                "date": rebalance_date.isoformat()[:10],
                "equity": round(portfolio_value, 2),
                "cash": round(equity_sleeve_cash, 2),
                "positions": len(positions),
                "held": list(positions.keys()),
            })

            sleeve_history.append({
                "date": rebalance_date.isoformat()[:10],
                "equity_cash": round(equity_sleeve_cash, 2),
                "equity_invested": round(portfolio_value - equity_sleeve_cash, 2),
            })

    finally:
        session.close()

    # Compute summary stats
    result = _compute_backtest_summary(
        equity_curve, trades, exit_counts, sleeve_history,
        start_date, end_date, weekly_deposit,
    )

    logger.info(
        "Backtest complete",
        extra={"extra_data": {
            "dates": len(equity_curve),
            "trades": len(trades),
            "final_equity": equity_curve[-1]["equity"] if equity_curve else 0,
        }},
    )

    return result


def _compute_portfolio_value(cash, positions, features):
    """Compute total portfolio value from cash + positions."""
    value = cash
    prices = features.get("prices")
    if prices is not None and not prices.empty:
        latest_row = prices.iloc[-1]
        for sym, pos in positions.items():
            if sym in latest_row.index:
                price = float(latest_row[sym])
                if price > 0:
                    value += pos["qty"] * price
    return round(value, 2)


def _compute_backtest_summary(equity_curve, trades, exit_counts, sleeve_history,
                               start_date, end_date, weekly_deposit):
    """Compute summary statistics from backtest results."""
    if not equity_curve:
        return {"error": "No data points"}

    equities = [e["equity"] for e in equity_curve]
    peak = equities[0]
    max_drawdown = 0
    drawdown_series = []

    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, dd)
        drawdown_series.append(round(dd, 4))

    # Trade stats
    sells = [t for t in trades if t["action"] == "SELL"]
    wins = [t for t in sells if t.get("pnl", 0) and t["pnl"] > 0]
    losses = [t for t in sells if t.get("pnl", 0) and t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in sells if t.get("pnl") is not None)

    win_rate = len(wins) / len(sells) if sells else 0
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Total deposited
    weeks = len(equity_curve)
    total_deposited = weekly_deposit * 0.70 * weeks  # equity sleeve only

    # Hold time
    hold_times = []
    buy_dates = {}
    for t in trades:
        if t["action"] == "BUY":
            buy_dates[t["symbol"]] = t["date"]
        elif t["action"] == "SELL" and t["symbol"] in buy_dates:
            buy_dt = datetime.fromisoformat(buy_dates[t["symbol"]])
            sell_dt = datetime.fromisoformat(t["date"])
            hold_times.append((sell_dt - buy_dt).days)
            del buy_dates[t["symbol"]]

    avg_hold_days = sum(hold_times) / len(hold_times) if hold_times else 0

    return {
        "period": {
            "start": start_date.isoformat()[:10],
            "end": end_date.isoformat()[:10],
            "weeks": weeks,
        },
        "performance": {
            "final_equity": equities[-1],
            "total_deposited": round(total_deposited, 2),
            "total_return_pct": round(
                ((equities[-1] - total_deposited) / total_deposited * 100), 2
            ) if total_deposited > 0 else 0,
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "total_pnl": round(total_pnl, 2),
        },
        "trades": {
            "total_buys": len([t for t in trades if t["action"] == "BUY"]),
            "total_sells": len(sells),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 3),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
            "avg_hold_days": round(avg_hold_days, 1),
        },
        "exit_distribution": exit_counts,
        "equity_curve": equity_curve,
        "drawdown_series": drawdown_series,
        "sleeve_history": sleeve_history,
        "trade_log": trades,
    }


def format_backtest_report(result):
    """Format backtest results as human-readable text."""
    if "error" in result:
        return f"Backtest error: {result['error']}"

    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST REPORT — Equity Momentum Sleeve")
    lines.append("=" * 60)

    p = result["period"]
    lines.append(f"Period:     {p['start']} to {p['end']} ({p['weeks']} weeks)")

    perf = result["performance"]
    lines.append(f"\n--- Performance ---")
    lines.append(f"  Final equity:   ${perf['final_equity']:>10,.2f}")
    lines.append(f"  Total deposited:${perf['total_deposited']:>10,.2f}")
    lines.append(f"  Return:         {perf['total_return_pct']:>+10.2f}%")
    lines.append(f"  Max drawdown:   {perf['max_drawdown_pct']:>10.2f}%")
    lines.append(f"  Total P&L:      ${perf['total_pnl']:>+10.2f}")

    t = result["trades"]
    lines.append(f"\n--- Trades ---")
    lines.append(f"  Buys:           {t['total_buys']}")
    lines.append(f"  Sells:          {t['total_sells']} ({t['wins']}W / {t['losses']}L)")
    lines.append(f"  Win rate:       {t['win_rate']:.0%}")
    pf = t['profit_factor']
    lines.append(f"  Profit factor:  {pf}")
    lines.append(f"  Avg hold:       {t['avg_hold_days']:.0f} days")

    ex = result["exit_distribution"]
    lines.append(f"\n--- Exit Distribution ---")
    total_exits = sum(ex.values())
    for path, count in sorted(ex.items(), key=lambda x: -x[1]):
        pct = count / total_exits * 100 if total_exits > 0 else 0
        lines.append(f"  {path:20s}: {count:>4d} ({pct:.0f}%)")

    # Last 10 trades
    trade_log = result.get("trade_log", [])
    sells = [t for t in trade_log if t["action"] == "SELL"]
    if sells:
        lines.append(f"\n--- Recent Sells (last 10) ---")
        for t in sells[-10:]:
            pnl_str = f"${t['pnl']:+.2f}" if t["pnl"] is not None else "?"
            lines.append(
                f"  {t['date']}  {t['symbol']:6s}  "
                f"${t['entry_price']:.2f} → ${t['exit_price']:.2f}  "
                f"{pnl_str:>8s}  [{t['exit_path']}]"
            )

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def save_backtest_report(result):
    """Save backtest results to reports/ directory."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    period = result.get("period", {})
    label = f"{period.get('start', 'unknown')}_to_{period.get('end', 'unknown')}"

    json_path = os.path.join(REPORTS_DIR, f"backtest_{label}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    text_path = os.path.join(REPORTS_DIR, f"backtest_{label}.txt")
    with open(text_path, "w") as f:
        f.write(format_backtest_report(result))

    logger.info(f"Backtest report saved: {json_path}")
    return json_path, text_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest equity momentum strategy")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--deposit", type=float, default=100.0, help="Weekly deposit")
    parser.add_argument("--initial", type=float, default=0.0, help="Initial capital")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    print(f"Running backtest: {args.start} to {args.end}")
    result = run_backtest(start, end, args.deposit, args.initial)
    print(format_backtest_report(result))
    save_backtest_report(result)
