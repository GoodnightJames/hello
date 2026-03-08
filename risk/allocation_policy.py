"""
Automatic Allocation Policy — enforces symbol culling in the live pipeline.

Turns SymbolTracker classifications into hard trading constraints:
- dead_weight → DISABLED (cannot trade, after sufficient evidence)
- conditional_earner → regime-gated (only trades in profitable regimes)
- core_earner → normal (always tradeable)
- insufficient_data → monitored (trades with caution)

Evidence tiers prevent overreacting to small samples:
- 5+ trades  → classification + monitoring only
- 10+ trades → regime restrictions allowed
- 20+ trades → hard disable allowed

Hysteresis prevents flickering: a symbol must hold its classification
for N consecutive review windows before policy changes take effect.

Shadow mode: blocked symbols are scored hypothetically so we can
measure whether blocking them actually improves outcomes.

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
SHADOW_LOG_PATH = "reports/allocation_shadow.jsonl"

# ── Evidence tiers ────────────────────────────────────────────────────
# Different actions require different amounts of evidence.
# Five trades can suggest a pattern; twenty trades can confirm it.
MIN_TRADES_CLASSIFY = 5       # Enough to classify and monitor
MIN_TRADES_RESTRICT = 10      # Enough for regime restrictions
MIN_TRADES_DISABLE = 20       # Enough for hard disable
MIN_TRADES_PER_REGIME = 3     # Per-regime minimum for regime-specific rules

# ── Hysteresis ────────────────────────────────────────────────────────
# Require persistent classification before acting.
# Prevents one bad week from culling a decent symbol.
HYSTERESIS_WINDOWS_RESTRICT = 2   # Must be conditional_earner 2x before restricting
HYSTERESIS_WINDOWS_DISABLE = 3    # Must be dead_weight 3x before disabling


def build_live_policy(min_trades=None):
    """
    Build allocation policy from accumulated trade history.

    Uses tiered evidence thresholds: more trades required for
    more aggressive actions (disable > restrict > monitor).

    Returns:
        Dict of {symbol: policy_dict} where policy_dict has:
        - action: "normal" | "restrict_to_regimes" | "disable" | "monitor"
        - confidence: "provisional" | "moderate" | "high"
        - allowed_regimes: list of regime phases where trading is allowed
        - reason: human-readable explanation
        - classification: underlying classification
        - stats: performance stats backing the decision
    """
    if min_trades is None:
        min_trades = MIN_TRADES_CLASSIFY

    tracker = load_symbol_history()

    if not tracker._trades:
        logger.info("Allocation policy: no trade history yet — all symbols allowed")
        return {}

    # Classify symbols from accumulated history
    classification = tracker.classify_symbols(min_trades=min_trades)

    if not classification:
        return {}

    # Build raw culling policy from classification
    raw_policy = build_culling_policy(classification)

    # Load previous policy for hysteresis checks
    prev_policy = load_allocation_policy()
    prev_log = _load_policy_log()

    # Apply evidence tiers and hysteresis
    policy = {}
    for sym, rule in raw_policy.items():
        cls_data = classification.get(sym, {})
        cls = cls_data.get("classification", "unknown")
        stats = cls_data.get("stats", {})
        trade_count = stats.get("trades", 0)
        raw_action = rule["action"]

        # Determine confidence from trade count
        if trade_count >= MIN_TRADES_DISABLE:
            confidence = "high"
        elif trade_count >= MIN_TRADES_RESTRICT:
            confidence = "moderate"
        else:
            confidence = "provisional"

        # Apply evidence tier gates
        action = _apply_evidence_tiers(
            raw_action, cls, trade_count, cls_data, confidence,
        )

        # Apply hysteresis — check how long this classification has persisted
        action = _apply_hysteresis(
            sym, action, cls, prev_log,
        )

        # Check re-entry: disabled symbols with improving shadow data
        # can be promoted back to monitor for fresh evaluation
        if action == "disable":
            action = _check_reentry(sym, action, prev_policy)

        policy[sym] = {
            "action": action,
            "allowed_regimes": rule.get("allowed_regimes", ["all"]),
            "reason": rule["reason"],
            "classification": cls,
            "confidence": confidence,
            "trade_count": trade_count,
            "stats": stats,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

    return policy


def _check_reentry(sym, action, prev_policy):
    """
    Check if a disabled symbol qualifies for re-entry via shadow data.

    A disabled symbol is promoted back to "monitor" if its shadow
    performance shows it's competitive again. This prevents the
    policy from being permanently punitive.

    Re-entry requires:
    - Previously disabled (in prior policy)
    - Enough shadow observations (5+)
    - Recent shadow data shows positive scores or selection
    """
    if action != "disable":
        return action

    # Only check re-entry for symbols that were already disabled
    prev = prev_policy.get(sym, {})
    if prev.get("action") != "disable":
        return action  # Newly disabled — no re-entry yet

    try:
        from review.policy_scorecard import find_reentry_candidates
        candidates = find_reentry_candidates()
        reentry_syms = {c["symbol"] for c in candidates}

        if sym in reentry_syms:
            logger.info(
                f"Re-entry: {sym} promoted from disable → monitor "
                f"(improving shadow performance)",
            )
            return "monitor"
    except Exception:
        pass  # Non-critical — keep disabled if scorecard fails

    return action


def _apply_evidence_tiers(raw_action, classification, trade_count,
                          cls_data, confidence):
    """
    Gate policy actions by evidence level.

    - flag_for_removal with <20 trades → downgrade to monitor
    - restrict_to_regimes with <10 trades → downgrade to monitor
    - restrict_to_regimes with sparse regime data → downgrade to monitor
    """
    if raw_action == "flag_for_removal":
        if trade_count >= MIN_TRADES_DISABLE:
            return "disable"
        elif trade_count >= MIN_TRADES_RESTRICT:
            # Not enough to disable, but enough to restrict
            logger.info(
                f"Evidence tier: {classification} has {trade_count} trades "
                f"(need {MIN_TRADES_DISABLE} for disable) — restricting instead",
            )
            return "restrict_to_regimes"
        else:
            logger.info(
                f"Evidence tier: {classification} has {trade_count} trades "
                f"(need {MIN_TRADES_RESTRICT} for restrict) — monitoring only",
            )
            return "monitor"

    if raw_action == "restrict_to_regimes":
        if trade_count < MIN_TRADES_RESTRICT:
            logger.info(
                f"Evidence tier: conditional_earner has {trade_count} trades "
                f"(need {MIN_TRADES_RESTRICT} for regime restriction) — monitoring",
            )
            return "monitor"

        # Check per-regime sample sizes
        regime_bd = cls_data.get("regime_breakdown", {})
        allowed = []
        for regime, regime_stats in regime_bd.items():
            if regime_stats and regime_stats.get("trades", 0) >= MIN_TRADES_PER_REGIME:
                if regime_stats.get("expectancy", 0) > 0:
                    allowed.append(regime)
            # else: not enough data to trust this regime's verdict

        if not allowed:
            # No regime has enough data to be trustworthy
            logger.info(
                f"Evidence tier: conditional_earner regime data too sparse "
                f"(need {MIN_TRADES_PER_REGIME} trades/regime) — monitoring",
            )
            return "monitor"

        return "restrict_to_regimes"

    return raw_action


def _apply_hysteresis(sym, action, classification, policy_log):
    """
    Require persistent classification before escalating policy.

    A symbol must hold its classification for N consecutive review
    windows before restrictions or disable take effect.
    """
    if action in ("normal", "monitor"):
        return action  # No hysteresis needed for permissive actions

    # Count consecutive windows this symbol had the same classification
    consecutive = _count_consecutive_classification(sym, classification, policy_log)

    if action == "disable" and consecutive < HYSTERESIS_WINDOWS_DISABLE:
        logger.info(
            f"Hysteresis: {sym} classified as {classification} for "
            f"{consecutive}/{HYSTERESIS_WINDOWS_DISABLE} windows — "
            f"restricting instead of disabling",
        )
        return "restrict_to_regimes"

    if action == "restrict_to_regimes" and consecutive < HYSTERESIS_WINDOWS_RESTRICT:
        logger.info(
            f"Hysteresis: {sym} classified as {classification} for "
            f"{consecutive}/{HYSTERESIS_WINDOWS_RESTRICT} windows — "
            f"monitoring instead of restricting",
        )
        return "monitor"

    return action


def _count_consecutive_classification(sym, classification, policy_log):
    """
    Count how many recent consecutive policy log entries show this
    symbol with the same classification.
    """
    if not policy_log:
        return 1  # First time — count as 1

    count = 0
    for entry in reversed(policy_log):
        summary = entry.get("policy_summary", {})
        sym_entry = summary.get(sym, {})
        if sym_entry.get("classification") == classification:
            count += 1
        else:
            break

    return count + 1  # +1 for the current window


def _load_policy_log():
    """Load policy log history for hysteresis checks."""
    if not os.path.exists(POLICY_LOG_PATH):
        return []

    entries = []
    try:
        with open(POLICY_LOG_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except (json.JSONDecodeError, IOError):
        return []

    return entries


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

    # Append to log for history (includes classification for hysteresis)
    log_entry = {
        "timestamp": snapshot["timestamp"],
        "policy_summary": {
            sym: {
                "action": p["action"],
                "classification": p["classification"],
                "confidence": p.get("confidence", "unknown"),
                "trade_count": p.get("trade_count", 0),
            }
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
        confidence = rule.get("confidence", "unknown")

        if action == "disable":
            blocked.append((
                sym,
                f"DISABLED ({confidence}): {rule.get('reason', 'dead weight')}",
            ))
            logger.info(
                f"Policy block: {sym} DISABLED ({confidence}) — "
                f"{rule.get('reason', '')}",
            )
            continue

        if action == "restrict_to_regimes":
            allowed_regimes = rule.get("allowed_regimes", [])
            if "all" in allowed_regimes or current_phase in allowed_regimes:
                allowed.append(sym)
            else:
                blocked.append((
                    sym,
                    f"REGIME BLOCKED ({confidence}): {sym} only allowed in "
                    f"{allowed_regimes}, current={current_phase}",
                ))
                logger.info(
                    f"Policy block: {sym} regime-gated ({confidence}) — "
                    f"allowed in {allowed_regimes}, current={current_phase}",
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


def record_shadow_outcome(symbol, action, regime_phase,
                          would_have_scored=None, would_have_selected=False,
                          hypothetical_pnl=None):
    """
    Record a shadow-mode outcome for a blocked symbol.

    Called after scoring to track what WOULD have happened if the
    symbol hadn't been blocked. This is how we validate the policy.

    Args:
        symbol: The blocked symbol.
        action: Policy action that blocked it ("disable" or "restrict_to_regimes").
        regime_phase: Current regime when blocked.
        would_have_scored: Rank score the symbol got (scored even though blocked).
        would_have_selected: Whether it would have been the top pick.
        hypothetical_pnl: If we can estimate forward PnL (e.g., from next cycle).
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "symbol": symbol,
        "policy_action": action,
        "regime_phase": regime_phase,
        "would_have_scored": round(would_have_scored, 4) if would_have_scored is not None else None,
        "would_have_selected": would_have_selected,
        "hypothetical_pnl": round(hypothetical_pnl, 6) if hypothetical_pnl is not None else None,
    }

    os.makedirs(os.path.dirname(SHADOW_LOG_PATH), exist_ok=True)
    with open(SHADOW_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def analyze_shadow_outcomes():
    """
    Analyze shadow mode data to measure policy effectiveness.

    Returns:
        Dict with:
        - total_blocks: number of times policy blocked a symbol
        - would_have_been_selected: times the blocked symbol was top-ranked
        - avoided_trades: breakdown of hypothetical outcomes
        - policy_verdict: whether blocking is helping or hurting
    """
    if not os.path.exists(SHADOW_LOG_PATH):
        return {"total_blocks": 0, "verdict": "no_data"}

    entries = []
    try:
        with open(SHADOW_LOG_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except (json.JSONDecodeError, IOError):
        return {"total_blocks": 0, "verdict": "read_error"}

    if not entries:
        return {"total_blocks": 0, "verdict": "no_data"}

    total = len(entries)
    would_selected = sum(1 for e in entries if e.get("would_have_selected"))
    has_pnl = [e for e in entries if e.get("hypothetical_pnl") is not None]

    result = {
        "total_blocks": total,
        "would_have_been_selected": would_selected,
        "selection_rate": round(would_selected / total, 3) if total > 0 else 0,
    }

    if has_pnl:
        pnls = [e["hypothetical_pnl"] for e in has_pnl]
        avoided_wins = sum(1 for p in pnls if p > 0)
        avoided_losses = sum(1 for p in pnls if p <= 0)
        avg_avoided_pnl = sum(pnls) / len(pnls)

        result["avoided_trades_with_pnl"] = len(has_pnl)
        result["avoided_wins"] = avoided_wins
        result["avoided_losses"] = avoided_losses
        result["avg_avoided_pnl"] = round(avg_avoided_pnl, 6)
        result["total_avoided_pnl"] = round(sum(pnls), 6)

        # Verdict: if average avoided PnL is negative, policy is helping
        if avg_avoided_pnl < -0.001:
            result["verdict"] = "HELPING — avoided net-negative trades"
        elif avg_avoided_pnl > 0.001:
            result["verdict"] = "HURTING — blocked net-positive trades"
        else:
            result["verdict"] = "NEUTRAL — blocked trades were roughly flat"
    else:
        result["verdict"] = "insufficient_pnl_data"

    return result


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
        confidence = rule.get("confidence", "?")
        by_action.setdefault(action, []).append(f"{sym}({confidence})")

    parts = []
    for action in ["normal", "restrict_to_regimes", "disable", "monitor"]:
        syms = by_action.get(action, [])
        if syms:
            parts.append(f"{action}: {', '.join(syms)}")

    logger.info(f"Allocation policy: {' | '.join(parts)}")
