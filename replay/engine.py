"""
Replay Engine — deterministic decision replay for auditing.

Given a date range, replays the decision pipeline using historical data
to verify that today's code produces the same decisions as were originally
recorded. This catches:
1. Unintended logic changes (regressions)
2. Data integrity issues
3. Parameter drift

The replay engine does NOT execute orders. It only produces decisions
and compares them against the historical record.

Key property: same inputs (data + params) = same outputs (signals + decisions).
"""

import json
from datetime import datetime, timedelta

import pandas as pd

from core.logging import get_logger
from data.db import Decision, Signal, DailyBar, get_session, init_db
from data.feature_store import compute_returns, compute_sma, compute_rsi
from research.momentum_scorer import score_dual_momentum
from research.regime import classify_regime
from review.param_log import get_params_at_date

logger = get_logger("replay.engine")


def load_historical_prices(session, symbols, as_of_date, lookback_days=352):
    """
    Load price data as it existed on a specific date (no lookahead).

    Args:
        session: DB session.
        symbols: List of ticker symbols.
        as_of_date: Cut-off date — only data on or before this date.
        lookback_days: Number of days of history to include.

    Returns:
        DataFrame with DatetimeIndex and symbol columns (close prices).
    """
    cutoff = as_of_date
    start = as_of_date - timedelta(days=int(lookback_days * 1.5))  # Calendar day buffer

    rows = (
        session.query(DailyBar.symbol, DailyBar.date, DailyBar.close)
        .filter(
            DailyBar.symbol.in_(symbols),
            DailyBar.date >= start,
            DailyBar.date <= cutoff,
        )
        .order_by(DailyBar.date)
        .all()
    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["symbol", "date", "close"])
    pivot = df.pivot(index="date", columns="symbol", values="close").sort_index()
    return pivot


def build_features_as_of(session, symbols, as_of_date, lookback_days=352):
    """
    Build features using only data available on as_of_date.

    No lookahead bias — strictly uses data <= as_of_date.
    """
    prices = load_historical_prices(session, symbols, as_of_date, lookback_days)

    if prices.empty:
        return {"prices": prices, "returns": {}, "sma": {}, "rsi_2": pd.DataFrame()}

    returns = compute_returns(prices, {"12m": 252, "6m": 126, "3m": 63, "1m": 21})
    sma = compute_sma(prices, {"200d": 200, "50d": 50})
    rsi_2 = compute_rsi(prices, period=2)

    return {
        "prices": prices,
        "returns": returns,
        "sma": sma,
        "rsi_2": rsi_2,
    }


def get_historical_decisions(session, strategy_name, target_date):
    """Get the decisions that were originally made on a given date."""
    # Search within the day
    day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    decisions = (
        session.query(Decision)
        .filter(
            Decision.strategy == strategy_name,
            Decision.date >= day_start,
            Decision.date < day_end,
        )
        .order_by(Decision.date)
        .all()
    )

    return [
        {
            "id": d.id,
            "symbol": d.symbol,
            "action": d.action,
            "reason": d.reason,
            "risk_approved": d.risk_approved,
        }
        for d in decisions
    ]


def replay_date(session, strategy_name, target_date, strategy_config=None):
    """
    Replay the decision pipeline for a single date.

    Args:
        session: DB session.
        strategy_name: Strategy to replay.
        target_date: Date to replay.
        strategy_config: Strategy config dict. If None, loads from param_log
                         using the config that was active on target_date.

    Returns:
        Dict with replay results:
        {
            "date": str,
            "strategy": str,
            "replayed_signals": [...],
            "historical_decisions": [...],
            "match": bool,
            "mismatches": [...]
        }
    """
    logger.info(
        f"Replaying {strategy_name} for {target_date.date()}",
    )

    # Load config that was active on that date
    if strategy_config is None:
        strategy_config = get_params_at_date(strategy_name, target_date, session)
        if strategy_config is None:
            logger.warning(f"No param version found for {strategy_name} on {target_date.date()}")
            return {
                "date": target_date.isoformat(),
                "strategy": strategy_name,
                "error": "No parameter version found for this date",
            }

    # Get instruments from config
    instruments = strategy_config.get("instruments", {})
    risk_assets = instruments.get("risk_assets", [])
    safe_assets = instruments.get("safe_assets", [])
    all_symbols = list(set(risk_assets + safe_assets + ["SPY"]))

    # Build features as-of that date (no lookahead)
    features = build_features_as_of(session, all_symbols, target_date)

    if features["prices"].empty:
        return {
            "date": target_date.isoformat(),
            "strategy": strategy_name,
            "error": "No price data available for this date",
        }

    # Replay signal generation
    replayed_signals = score_dual_momentum(features, strategy_config)

    # Get what was actually decided
    historical = get_historical_decisions(session, strategy_name, target_date)

    # Compare
    mismatches = _compare_decisions(replayed_signals, historical)
    match = len(mismatches) == 0

    result = {
        "date": target_date.isoformat(),
        "strategy": strategy_name,
        "replayed_signals": [
            {"symbol": s["symbol"], "signal_type": s["signal_type"], "score": round(s["score"], 6)}
            for s in replayed_signals
        ],
        "historical_decisions": historical,
        "match": match,
        "mismatches": mismatches,
    }

    if match:
        logger.info(f"Replay MATCH for {target_date.date()}")
    else:
        logger.warning(
            f"Replay MISMATCH for {target_date.date()}",
            extra={"extra_data": {"mismatches": mismatches}},
        )

    return result


def _compare_decisions(replayed_signals, historical_decisions):
    """
    Compare replayed signals against historical decisions.

    Returns list of mismatch descriptions.
    """
    mismatches = []

    # Build lookup by symbol
    replayed_by_symbol = {s["symbol"]: s["signal_type"] for s in replayed_signals}
    historical_by_symbol = {d["symbol"]: d["action"] for d in historical_decisions}

    # Signal type to decision action mapping
    # Note: regime filter may change BUY→SKIP, so we only flag clear mismatches
    signal_to_possible_actions = {
        "BUY": {"BUY", "SKIP"},  # BUY signal might be SKIP'd by regime
        "SELL": {"SELL"},
        "HOLD": {"HOLD"},
    }

    all_symbols = set(list(replayed_by_symbol.keys()) + list(historical_by_symbol.keys()))

    for symbol in all_symbols:
        replayed = replayed_by_symbol.get(symbol)
        historical = historical_by_symbol.get(symbol)

        if replayed is None and historical is not None:
            mismatches.append({
                "symbol": symbol,
                "type": "missing_in_replay",
                "historical_action": historical,
            })
        elif replayed is not None and historical is None:
            mismatches.append({
                "symbol": symbol,
                "type": "extra_in_replay",
                "replayed_signal": replayed,
            })
        elif replayed is not None and historical is not None:
            possible = signal_to_possible_actions.get(replayed, {replayed})
            if historical not in possible:
                mismatches.append({
                    "symbol": symbol,
                    "type": "action_mismatch",
                    "replayed_signal": replayed,
                    "historical_action": historical,
                })

    return mismatches


def replay_range(strategy_name, start_date, end_date, strategy_config=None):
    """
    Replay decisions for a date range.

    Args:
        strategy_name: Strategy to replay.
        start_date: Start of replay range.
        end_date: End of replay range.
        strategy_config: Config dict (uses param_log if None).

    Returns:
        List of daily replay result dicts.
    """
    logger.info(
        f"Replaying {strategy_name} from {start_date.date()} to {end_date.date()}"
    )

    init_db()
    session = get_session()

    try:
        results = []
        current = start_date

        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:
                result = replay_date(session, strategy_name, current, strategy_config)
                results.append(result)
            current += timedelta(days=1)

        # Summary
        total = len(results)
        matches = sum(1 for r in results if r.get("match", False))
        errors = sum(1 for r in results if "error" in r)

        logger.info(
            "Replay range complete",
            extra={
                "extra_data": {
                    "strategy": strategy_name,
                    "total_days": total,
                    "matches": matches,
                    "mismatches": total - matches - errors,
                    "errors": errors,
                }
            },
        )

        return results

    finally:
        session.close()


def generate_replay_report(results):
    """
    Format replay results as a human-readable report.

    Args:
        results: List of daily replay result dicts.

    Returns:
        String report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("DECISION REPLAY REPORT")
    lines.append("=" * 60)

    total = len(results)
    matches = sum(1 for r in results if r.get("match", False))
    errors = sum(1 for r in results if "error" in r)
    mismatches = total - matches - errors

    lines.append(f"Total days replayed: {total}")
    lines.append(f"Matches:             {matches}")
    lines.append(f"Mismatches:          {mismatches}")
    lines.append(f"Errors:              {errors}")
    lines.append("")

    if mismatches > 0:
        lines.append("--- MISMATCHES ---")
        for r in results:
            if not r.get("match", True) and "error" not in r:
                lines.append(f"  Date: {r['date'][:10]}")
                for m in r.get("mismatches", []):
                    lines.append(f"    {m['symbol']}: {m['type']} "
                                 f"(replay={m.get('replayed_signal', '?')}, "
                                 f"historical={m.get('historical_action', '?')})")
        lines.append("")

    if errors > 0:
        lines.append("--- ERRORS ---")
        for r in results:
            if "error" in r:
                lines.append(f"  {r['date'][:10]}: {r['error']}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    # Example: replay last 5 trading days
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    results = replay_range("dual_momentum_trend_core", start, end)
    print(generate_replay_report(results))
