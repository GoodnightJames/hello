"""
Automatic Allocation Policy — enforces symbol culling in the live pipeline.

Turns SymbolTracker classifications into hard trading constraints:
- dead_weight → DISABLED (cannot trade)
- conditional_earner → regime-gated (only trades in profitable regimes)
- core_earner → normal (always tradeable)
- insufficient_data → monitored (trades with caution)

The policy is rebuilt from trade history on each cycle and persisted
to disk so decisions are auditable. Symbols don't just get flagged —
they get starved of capital automatically.

Usage:
    from risk.allocation_policy import load_allocation_policy, apply_policy_filter

    policy = load_allocation_policy()
    eligible = apply_policy_filter(symbols, policy, current_regime)
"""

import json
import os
from datetime import datetime

from core.logging import get_logger
from review.symbol_tracker import load_symbol_history
from research.validation import build_culling_policy

logger = get_logger("risk.allocation_policy")

POLICY_PATH = "reports/allocation_policy.json"
POLICY_LOG_PATH = "reports/allocation_policy_log.jsonl"

# Minimum trades before policy has teeth — avoids culling on thin evidence
MIN_TRADES_FOR_POLICY = 5


def build_live_policy(min_trades=None):
    """
    Build allocation policy from accumulated trade history.

    Loads all historical trades from the symbol tracker, classifies
    each symbol, then converts classifications into enforceable rules.

    Returns:
        Dict of {symbol: policy_dict} where policy_dict has:
        - action: "normal" | "restrict_to_regimes" | "disable" | "monitor"
        - allowed_regimes: list of regime phases where trading is allowed
        - reason: human-readable explanation
        - classification: underlying classification
        - stats: performance stats backing the decision
    """
    if min_trades is None:
        min_trades = MIN_TRADES_FOR_POLICY

    tracker = load_symbol_history()

    if not tracker._trades:
        logger.info("Allocation policy: no trade history yet — all symbols allowed")
        return {}

    # Classify symbols from accumulated history
    classification = tracker.classify_symbols(min_trades=min_trades)

    if not classification:
        return {}

    # Build culling policy from classification
    raw_policy = build_culling_policy(classification)

    # Enrich with stats and upgrade "flag_for_removal" to "disable"
    policy = {}
    for sym, rule in raw_policy.items():
        cls_data = classification.get(sym, {})
        action = rule["action"]

        # Upgrade flag_for_removal to disable — this is the enforcement step
        if action == "flag_for_removal":
            action = "disable"

        policy[sym] = {
            "action": action,
            "allowed_regimes": rule.get("allowed_regimes", ["all"]),
            "reason": rule["reason"],
            "classification": cls_data.get("classification", "unknown"),
            "stats": cls_data.get("stats", {}),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

    return policy


def save_policy(policy):
    """Persist current policy to disk for auditability."""
    if not policy:
        return

    os.makedirs(os.path.dirname(POLICY_PATH), exist_ok=True)

    # Save current snapshot
    snapshot = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "symbols": policy,
    }
    with open(POLICY_PATH, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    # Append to log for history
    log_entry = {
        "timestamp": snapshot["timestamp"],
        "policy_summary": {
            sym: {"action": p["action"], "classification": p["classification"]}
            for sym, p in policy.items()
        },
    }
    with open(POLICY_LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry, default=str) + "\n")


def load_allocation_policy():
    """
    Load the most recent saved policy from disk.

    Returns empty dict if no policy exists yet.
    """
    if not os.path.exists(POLICY_PATH):
        return {}

    try:
        with open(POLICY_PATH, "r") as f:
            snapshot = json.load(f)
        return snapshot.get("symbols", {})
    except (json.JSONDecodeError, KeyError):
        logger.warning("Failed to load allocation policy — returning empty")
        return {}


def apply_policy_filter(symbols, policy, regime=None):
    """
    Filter symbols through allocation policy.

    This is the enforcement point — called in the trading pipeline
    AFTER universe listing rules but BEFORE scoring.

    Args:
        symbols: List of symbol strings that passed universe filter.
        policy: Dict from build_live_policy() or load_allocation_policy().
        regime: Current regime dict with "phase" key.

    Returns:
        (allowed_symbols, blocked_with_reasons)
        allowed_symbols: list of symbols cleared to trade
        blocked_with_reasons: list of (symbol, reason) for blocked symbols
    """
    if not policy:
        return list(symbols), []

    current_phase = "unknown"
    if regime and isinstance(regime, dict):
        current_phase = regime.get("phase", "unknown")

    allowed = []
    blocked = []

    for sym in symbols:
        rule = policy.get(sym)

        if rule is None:
            # No policy for this symbol — allow by default (new/untracked)
            allowed.append(sym)
            continue

        action = rule.get("action", "normal")

        if action == "disable":
            blocked.append((sym, f"DISABLED: {rule.get('reason', 'dead weight')}"))
            logger.info(
                f"Policy block: {sym} DISABLED — {rule.get('reason', '')}",
            )
            continue

        if action == "restrict_to_regimes":
            allowed_regimes = rule.get("allowed_regimes", [])
            if "all" in allowed_regimes or current_phase in allowed_regimes:
                allowed.append(sym)
            else:
                blocked.append((
                    sym,
                    f"REGIME BLOCKED: {sym} only allowed in "
                    f"{allowed_regimes}, current={current_phase}",
                ))
                logger.info(
                    f"Policy block: {sym} regime-gated — allowed in "
                    f"{allowed_regimes}, current={current_phase}",
                )
                continue

        elif action in ("normal", "monitor"):
            allowed.append(sym)

    if blocked:
        logger.info(
            f"Allocation policy: {len(allowed)}/{len(symbols)} symbols allowed, "
            f"{len(blocked)} blocked",
            extra={"extra_data": {
                "allowed": allowed,
                "blocked": [(s, r) for s, r in blocked],
                "regime_phase": current_phase,
            }},
        )

    return allowed, blocked


def refresh_and_apply(symbols, regime=None, min_trades=None):
    """
    Convenience: rebuild policy from history, save it, and apply.

    Use this in the trading pipeline for a single-call integration.

    Returns:
        (allowed_symbols, blocked_with_reasons, policy)
    """
    policy = build_live_policy(min_trades=min_trades)

    if policy:
        save_policy(policy)
        _log_policy_summary(policy)

    allowed, blocked = apply_policy_filter(symbols, policy, regime)
    return allowed, blocked, policy


def _log_policy_summary(policy):
    """Log a compact summary of the current policy."""
    if not policy:
        return

    by_action = {}
    for sym, rule in policy.items():
        action = rule.get("action", "unknown")
        by_action.setdefault(action, []).append(sym)

    parts = []
    for action in ["normal", "restrict_to_regimes", "disable", "monitor"]:
        syms = by_action.get(action, [])
        if syms:
            parts.append(f"{action}: {', '.join(syms)}")

    logger.info(f"Allocation policy: {' | '.join(parts)}")
