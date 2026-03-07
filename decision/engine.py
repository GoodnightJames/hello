"""
Decision Engine — deterministic signal-to-decision pipeline.

Same inputs = same output. No randomness.

Pipeline:
1. Load features from feature store
2. Run strategy signal generation
3. Classify regime
4. Apply regime filter to signals
5. Log every decision (including SKIPs with reason)
6. Store decisions to database

The decision engine does NOT execute trades. It produces decisions
that the execution layer (Phase 3) will act on.
"""

import json
from datetime import datetime

import yaml

from core.logging import get_logger
from data.db import Signal, Decision, get_session, init_db
from data.feature_store import build_features
from research.regime import classify_regime, load_risk_params
from review.param_log import log_param_version
from strategies.momentum_v1 import DualMomentumStrategy

logger = get_logger("decision.engine")


def load_config(config_path="config/settings.yaml"):
    """Load main settings config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def store_signal(session, strategy_name, signal_dict):
    """
    Persist a signal to the database.

    Returns the created Signal's id.
    """
    signal = Signal(
        strategy=strategy_name,
        symbol=signal_dict["symbol"],
        date=datetime.utcnow(),
        signal_type=signal_dict["signal_type"],
        score=signal_dict.get("score"),
        metadata_json=json.dumps(signal_dict.get("metadata", {})),
    )
    session.add(signal)
    session.flush()  # Get the id before commit
    return signal.id


def store_decision(session, strategy_name, symbol, action, reason, signal_id=None, risk_approved=False):
    """Persist a decision to the database."""
    decision = Decision(
        strategy=strategy_name,
        symbol=symbol,
        date=datetime.utcnow(),
        action=action,
        reason=reason,
        signal_id=signal_id,
        risk_approved=risk_approved,
    )
    session.add(decision)
    return decision


def get_held_positions(session):
    """
    Get list of currently held symbols from the latest portfolio state.

    Returns:
        List of symbol strings with qty > 0.
    """
    from data.db import PortfolioState
    latest = (
        session.query(PortfolioState)
        .order_by(PortfolioState.date.desc())
        .first()
    )
    if latest is None or not latest.positions_json:
        return []

    positions = json.loads(latest.positions_json)
    return [sym for sym, qty in positions.items() if qty > 0]


def get_exit_log(session):
    """
    Build exit log from recent fast-exit trades for cooldown tracking.

    Returns:
        Dict of {symbol: last_exit_date} for symbols exited via fast exit
        in the last 10 trading days.
    """
    from data.db import Trade
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(days=14)  # ~10 trading days

    recent_exits = (
        session.query(Trade)
        .filter(Trade.exit_date >= cutoff)
        .order_by(Trade.exit_date.desc())
        .all()
    )

    exit_log = {}
    for trade in recent_exits:
        sym = trade.symbol
        if sym not in exit_log:
            exit_log[sym] = trade.exit_date
    return exit_log


def apply_regime_filter(signals, regime, strategy_name, session):
    """
    Apply regime filters to signals, producing decisions.

    Rules:
    - If trend regime is risk_off AND signal is BUY on a risk asset → SKIP
    - If all equity BUYs are blocked by regime → redirect to best safe asset
    - Position multiplier applied by execution layer (stored in decision metadata)

    Every signal produces a decision (BUY, SELL, HOLD, or SKIP).
    Every decision is logged with reason.

    Args:
        signals: List of signal dicts from strategy.
        regime: Dict from classify_regime().
        strategy_name: Strategy identifier string.
        session: SQLAlchemy session.

    Returns:
        List of decision dicts.
    """
    decisions = []
    allow_entries = regime.get("allow_new_entries", True)
    multiplier = regime.get("position_multiplier", 1.0)
    trend = regime.get("trend", {})
    equity_buys_blocked = 0  # Track how many equity BUYs regime blocked

    for sig in signals:
        symbol = sig["symbol"]
        signal_type = sig["signal_type"]
        is_crypto = "/" in symbol  # Crypto symbols use slash (BTC/USD, ETH/USD)

        # Store signal to DB
        signal_id = store_signal(session, strategy_name, sig)

        if signal_type == "BUY" and not allow_entries and not is_crypto:
            # Regime says no new entries — SKIP this BUY
            # Crypto is exempt: SPY's 200d SMA has nothing to do with crypto markets
            spy_price = trend.get("spy_price")
            spy_sma = trend.get("spy_200d_sma")
            if spy_price is not None and spy_sma is not None:
                reason = (
                    f"SKIP: Regime filter blocked BUY. "
                    f"SPY below 200d SMA ({spy_price:.2f} < {spy_sma:.2f}). "
                    f"Holding cash."
                )
            else:
                reason = "SKIP: Regime risk_off — insufficient data for trend regime, no new entries"

            decision = store_decision(
                session, strategy_name, symbol,
                action="SKIP",
                reason=reason,
                signal_id=signal_id,
                risk_approved=False,
            )
            decisions.append({
                "symbol": symbol,
                "action": "SKIP",
                "reason": reason,
                "signal_type": signal_type,
                "signal_id": signal_id,
                "position_multiplier": multiplier,
                "risk_approved": False,
            })
            equity_buys_blocked += 1

            logger.info(
                f"Decision: SKIP {symbol}",
                extra={"extra_data": {"reason": reason, "signal_id": signal_id}},
            )

        elif signal_type == "SELL":
            metadata = sig.get("metadata", {})
            if metadata.get("fast_exit"):
                exit_details = metadata.get("exit_details", {})
                reason = (
                    f"SELL: Fast exit — 3m return ({exit_details.get('return_3m', 0):.4f}) "
                    f"below benchmark ({exit_details.get('benchmark_3m', 0):.4f})"
                )
            elif metadata.get("scan_type") == "daily":
                reason = f"SELL: Daily scan exit — 3m momentum breakdown"
            else:
                reason = f"SELL: No absolute momentum (12m return below benchmark)"
            decision = store_decision(
                session, strategy_name, symbol,
                action="SELL",
                reason=reason,
                signal_id=signal_id,
                risk_approved=True,  # Sells are always risk-approved
            )
            decisions.append({
                "symbol": symbol,
                "action": "SELL",
                "reason": reason,
                "signal_type": signal_type,
                "signal_id": signal_id,
                "position_multiplier": 1.0,
                "risk_approved": True,
            })

            logger.info(
                f"Decision: SELL {symbol}",
                extra={"extra_data": {"reason": reason, "signal_id": signal_id}},
            )

        elif signal_type == "BUY":
            signal_strength = sig.get("signal_strength", 1.0)
            reason = (
                f"BUY: Top-ranked asset with positive absolute momentum. "
                f"12m return: {sig.get('score', 0):.4f}. "
                f"Strength: {signal_strength:.2f}. "
                f"Position multiplier: {multiplier:.2f}"
            )
            decision = store_decision(
                session, strategy_name, symbol,
                action="BUY",
                reason=reason,
                signal_id=signal_id,
                risk_approved=True,  # Risk validation happens at execution
            )
            decisions.append({
                "symbol": symbol,
                "action": "BUY",
                "reason": reason,
                "signal_type": signal_type,
                "signal_id": signal_id,
                "position_multiplier": multiplier,
                "signal_strength": signal_strength,
                "risk_approved": True,
            })

            logger.info(
                f"Decision: BUY {symbol}",
                extra={"extra_data": {"reason": reason, "signal_id": signal_id, "multiplier": multiplier}},
            )

        elif signal_type == "HOLD":
            reason = f"HOLD: Asset has absolute momentum but is not top-ranked"
            decision = store_decision(
                session, strategy_name, symbol,
                action="HOLD",
                reason=reason,
                signal_id=signal_id,
                risk_approved=True,
            )
            decisions.append({
                "symbol": symbol,
                "action": "HOLD",
                "reason": reason,
                "signal_type": signal_type,
                "signal_id": signal_id,
                "position_multiplier": multiplier,
                "risk_approved": True,
            })

            logger.info(
                f"Decision: HOLD {symbol}",
                extra={"extra_data": {"reason": reason, "signal_id": signal_id}},
            )

    # If regime blocked equity BUYs, redirect to safe assets
    if equity_buys_blocked > 0:
        equity_buys_remaining = [d for d in decisions if d["action"] == "BUY" and "/" not in d["symbol"]]
        if not equity_buys_remaining:
            safe_decisions = _generate_safe_asset_redirect(regime, strategy_name, session, multiplier)
            decisions.extend(safe_decisions)

    # Assign equal-weight target allocations to BUY decisions (max 40% per symbol)
    decisions = assign_target_weights(decisions)

    return decisions


# ── Regime-Off Safe-Asset Redirect ────────────────────────────────────────
SAFE_ASSETS = ["AGG", "TLT"]  # SHY excluded — it's the benchmark
SAFE_ASSET_STRENGTH = 0.5     # Conservative signal strength for safe assets


def _generate_safe_asset_redirect(regime, strategy_name, session, multiplier):
    """
    When regime blocks all equity BUYs, redirect to the best safe asset.

    Decision tree:
    1. Get 12-month returns for AGG and TLT from feature store
    2. Pick the one with higher 12-month return
    3. If both have data, BUY the winner
    4. If neither has data, hold cash (no decision emitted)

    Returns:
        List of decision dicts (0 or 1 BUY decisions).
    """
    trend = regime.get("trend", {})

    try:
        features = build_features(SAFE_ASSETS + ["SHY"], lookback_days=260)
        prices = features.get("prices")
        if prices is None or prices.empty:
            logger.warning("Safe-asset redirect: no price data — holding cash")
            return []

        # Compute 12-month returns
        if len(prices) < 252:
            logger.warning("Safe-asset redirect: insufficient history — holding cash")
            return []

        returns_12m = prices.pct_change(252).iloc[-1]
        benchmark_return = float(returns_12m.get("SHY", 0))

        # Rank safe assets by 12m return
        candidates = []
        for sym in SAFE_ASSETS:
            if sym in returns_12m.index:
                ret = float(returns_12m[sym])
                candidates.append((sym, ret))

        if not candidates:
            logger.info("Safe-asset redirect: no safe asset data — holding cash")
            return []

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_sym, best_ret = candidates[0]

        spy_price = trend.get("spy_price")
        spy_sma = trend.get("spy_200d_sma")
        reason = (
            f"BUY: Regime risk_off redirect to safe asset. "
            f"SPY below 200d SMA ({spy_price:.2f} < {spy_sma:.2f}). "
            f"Best safe asset: {best_sym} (12m return: {best_ret:.4f})"
        )

        signal_id = store_signal(session, strategy_name, {
            "symbol": best_sym,
            "signal_type": "BUY",
            "score": best_ret,
            "metadata": {
                "reason": "Regime risk_off — safe asset redirect",
                "safe_asset": True,
                "benchmark_return": benchmark_return,
            },
        })

        decision = store_decision(
            session, strategy_name, best_sym,
            action="BUY",
            reason=reason,
            signal_id=signal_id,
            risk_approved=True,
        )

        logger.info(
            f"Safe-asset redirect: BUY {best_sym}",
            extra={"extra_data": {"reason": reason, "12m_return": best_ret}},
        )

        return [{
            "symbol": best_sym,
            "action": "BUY",
            "reason": reason,
            "signal_type": "BUY",
            "signal_id": signal_id,
            "position_multiplier": multiplier,
            "signal_strength": SAFE_ASSET_STRENGTH,
            "risk_approved": True,
        }]

    except Exception as e:
        logger.error(
            f"Safe-asset redirect failed — holding cash: {e}",
            extra={"extra_data": {"error": str(e)}},
        )
        return []


# ── Top-N Equal-Weight Allocation ────────────────────────────────────────
MAX_WEIGHT_PER_POSITION = 0.40  # Hard cap: no single position > 40% of sleeve


def assign_target_weights(decisions):
    """
    Assign equal-weight target_weight to BUY decisions.

    For N buy decisions, each gets 1/N of sleeve capital, capped at 40%.
    The target_weight is used by the execution layer for position sizing
    instead of signal-strength-based sizing.

    Non-BUY decisions are passed through unchanged.
    """
    buys = [d for d in decisions if d.get("action") == "BUY"]
    if not buys:
        return decisions

    n = len(buys)
    raw_weight = 1.0 / n
    capped_weight = min(raw_weight, MAX_WEIGHT_PER_POSITION)

    for d in decisions:
        if d.get("action") == "BUY":
            d["target_weight"] = round(capped_weight, 4)

    logger.info(
        f"Target weights assigned: {n} positions × {capped_weight:.1%} each",
        extra={"extra_data": {"n_buys": n, "weight": capped_weight, "cap": MAX_WEIGHT_PER_POSITION}},
    )

    return decisions


# ── Decision Audit Trail ─────────────────────────────────────────────────


def _get_return_for_symbol(features, symbol, timeframe):
    """Safely extract a return value from features."""
    ret_df = features.get("returns", {}).get(timeframe)
    if ret_df is not None and not ret_df.empty and symbol in ret_df.columns:
        val = ret_df.iloc[-1][symbol]
        try:
            return round(float(val), 6)
        except (ValueError, TypeError):
            return None
    return None


def _emit_decision_audit(decisions, signals, regime, features, held_positions):
    """
    Emit a structured audit log entry for every decision in this cycle.

    Each entry captures the full context needed to answer:
    - Why did the system take this action?
    - What were the alternatives?
    - What was the regime state?

    This is the primary debugging tool for live systems.
    """
    trend = regime.get("trend", {})
    vol = regime.get("volatility", {})
    regime_summary = {
        "trend_regime": trend.get("trend_regime", "unknown"),
        "spy_price": trend.get("spy_price"),
        "spy_200d_sma": trend.get("spy_200d_sma"),
        "vol_regime": vol.get("vol_regime", "unknown"),
        "position_multiplier": regime.get("position_multiplier", 1.0),
        "allow_new_entries": regime.get("allow_new_entries", True),
    }

    # Build signal lookup for cross-referencing
    signal_by_symbol = {s["symbol"]: s for s in signals}

    audit_entries = []
    for d in decisions:
        symbol = d["symbol"]
        sig = signal_by_symbol.get(symbol, {})
        is_crypto = "/" in symbol
        sleeve = "crypto" if is_crypto else "equity"

        entry = {
            "symbol": symbol,
            "action": d["action"],
            "sleeve": sleeve,
            "target_weight": d.get("target_weight"),
            "signal_strength": d.get("signal_strength", sig.get("signal_strength")),
            "regime": regime_summary,
            "returns": {
                "12m": _get_return_for_symbol(features, symbol, "12m"),
                "6m": _get_return_for_symbol(features, symbol, "6m"),
                "3m": _get_return_for_symbol(features, symbol, "3m"),
                "1m": _get_return_for_symbol(features, symbol, "1m"),
            },
            "benchmark_returns": {
                "12m": _get_return_for_symbol(features, "SHY", "12m"),
                "6m": _get_return_for_symbol(features, "SHY", "6m"),
                "3m": _get_return_for_symbol(features, "SHY", "3m"),
            },
            "relative_rank": sig.get("metadata", {}).get("relative_rank"),
            "is_held": symbol in held_positions,
            "exit_path": _classify_exit_path(d, sig),
            "reason": d.get("reason", ""),
        }
        audit_entries.append(entry)

    logger.info(
        "Decision audit trail",
        extra={"extra_data": {
            "cycle_type": "weekly_rebalance",
            "regime": regime_summary,
            "held_positions": held_positions,
            "decisions": audit_entries,
        }},
    )


def _classify_exit_path(decision, signal):
    """Classify which exit path triggered this action."""
    action = decision.get("action")
    if action != "SELL":
        return None

    reason = decision.get("reason", "")
    metadata = signal.get("metadata", {})

    if "TRAILING STOP" in reason:
        return "trailing_stop"
    if metadata.get("fast_exit"):
        return "fast_exit_3m"
    if "No absolute momentum" in reason or "below benchmark" in reason:
        return "rank_drop"
    if "momentum breakdown" in reason:
        return "daily_scan"
    return "unknown"


def run_decision_engine(config_path="config/settings.yaml"):
    """
    Main decision engine pipeline.

    Orchestrates: features → signals → regime → decisions.
    All decisions logged to database and structured logs.

    Returns:
        List of decision dicts.
    """
    logger.info("=" * 50)
    logger.info("Decision engine starting")
    logger.info("=" * 50)

    config = load_config(config_path)
    strategies = config.get("active_strategies", {})
    active_strategy = strategies.get("equity", config.get("active_strategy", "momentum_v1"))

    # Initialize DB
    init_db()
    session = get_session()

    try:
        # Step 1: Load strategy and log params
        logger.info(f"Loading strategy: {active_strategy}")
        strategy = DualMomentumStrategy()
        log_param_version(strategy.name, strategy.config, session=session)
        instruments = strategy.get_instruments()

        # Ensure SPY is in the feature set for regime classification
        feature_symbols = list(set(instruments + ["SPY"]))

        # Step 2: Build features
        logger.info("Building features from database")
        features = build_features(feature_symbols, lookback_days=352)  # 252 + 100 buffer

        if features["prices"].empty:
            logger.warning("No price data in database — run data ingestion first")
            return []

        # Step 3: Generate signals (with cooldown awareness)
        logger.info("Generating strategy signals")
        exit_log = get_exit_log(session)
        signals = strategy.generate_signals(data=features, exit_log=exit_log)

        if not signals:
            logger.info("No signals generated — nothing to decide")
            return []

        # Step 4: Classify regime
        logger.info("Classifying market regime")
        risk_params = load_risk_params()
        regime = classify_regime(features, risk_params)

        # Step 5: Apply regime filter → produce decisions
        logger.info("Applying regime filter to signals")
        decisions = apply_regime_filter(signals, regime, strategy.name, session)

        # Step 6: Stamp full audit context onto decisions and emit audit log
        held_positions = get_held_positions(session)
        _emit_decision_audit(decisions, signals, regime, features, held_positions)

        # Commit all signals and decisions
        session.commit()

        logger.info(
            "Decision engine complete",
            extra={
                "extra_data": {
                    "strategy": strategy.name,
                    "signals_count": len(signals),
                    "decisions_count": len(decisions),
                    "regime": {
                        "trend": regime["trend"]["trend_regime"],
                        "vol": regime["volatility"]["vol_regime"],
                        "multiplier": regime["position_multiplier"],
                    },
                    "decisions_summary": [
                        {"symbol": d["symbol"], "action": d["action"]}
                        for d in decisions
                    ],
                }
            },
        )
        return decisions

    except Exception as e:
        session.rollback()
        logger.error(
            "Decision engine failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )
        raise
    finally:
        session.close()


def run_daily_decision_engine(config_path="config/settings.yaml"):
    """
    Daily decision engine — exit losers and reallocate immediately.

    Runs every trading day (not just weekly rebalance).
    Checks held positions for 3m momentum breakdown, finds replacements.

    Returns:
        List of decision dicts (SELLs + replacement BUYs).
    """
    logger.info("=" * 50)
    logger.info("Daily decision engine starting")
    logger.info("=" * 50)

    config = load_config(config_path)
    init_db()
    session = get_session()

    try:
        strategy = DualMomentumStrategy()
        instruments = strategy.get_instruments()
        feature_symbols = list(set(instruments + ["SPY"]))

        # Build features
        features = build_features(feature_symbols, lookback_days=352)
        if features["prices"].empty:
            logger.warning("No price data — run data ingestion first")
            return []

        # Get current holdings and exit log
        held_positions = get_held_positions(session)
        exit_log = get_exit_log(session)

        if not held_positions:
            logger.info("No positions held — daily scan has nothing to check")
            return []

        logger.info(
            f"Daily scan: checking {len(held_positions)} positions",
            extra={"extra_data": {"held": held_positions, "exit_log_symbols": list(exit_log.keys())}},
        )

        # Generate daily signals (exit + replace)
        signals = strategy.generate_daily_signals(
            data=features,
            held_positions=held_positions,
            exit_log=exit_log,
        )

        if not signals:
            logger.info("Daily scan: all positions healthy, no action needed")
            return []

        # Classify regime (for replacement BUYs)
        risk_params = load_risk_params()
        regime = classify_regime(features, risk_params)

        # Apply regime filter → produce decisions
        decisions = apply_regime_filter(signals, regime, strategy.name, session)

        # Assign equal-weight allocations to any replacement BUYs
        decisions = assign_target_weights(decisions)

        # Emit audit trail for daily scan too
        _emit_decision_audit(decisions, signals, regime, features, held_positions)

        session.commit()

        logger.info(
            "Daily decision engine complete",
            extra={
                "extra_data": {
                    "signals_count": len(signals),
                    "decisions_count": len(decisions),
                    "sells": [d["symbol"] for d in decisions if d["action"] == "SELL"],
                    "buys": [d["symbol"] for d in decisions if d["action"] == "BUY"],
                }
            },
        )
        return decisions

    except Exception as e:
        session.rollback()
        logger.error(
            "Daily decision engine failed",
            extra={"extra_data": {"error": str(e)}},
            exc_info=True,
        )
        raise
    finally:
        session.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "daily":
        decisions = run_daily_decision_engine()
        print("=== Daily Scan Results ===")
    else:
        decisions = run_decision_engine()
        print("=== Weekly Rebalance Results ===")
    for d in decisions:
        print(f"  {d['action']:5s} {d['symbol']:5s} — {d['reason']}")
