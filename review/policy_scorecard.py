"""
Policy Scorecard — audits allocation policy effectiveness.

Answers the question: is the policy layer actually helping?

Reports:
- Blocked trade counts by symbol and regime
- Shadow outcomes: would-have-selected rate, hypothetical PnL
- Per-symbol policy verdict (helping/hurting/neutral)
- Per-regime verdict
- Confidence distribution
- Re-entry candidates (disabled symbols with improving shadow data)

Usage:
    from review.policy_scorecard import generate_policy_scorecard

    text = generate_policy_scorecard()
    print(text)
"""

import json
import os
from collections import defaultdict

from core.logging import get_logger

logger = get_logger("review.policy_scorecard")

SHADOW_LOG_PATH = "reports/allocation_shadow.jsonl"
POLICY_LOG_PATH = "reports/allocation_policy_log.jsonl"
POLICY_PATH = "reports/allocation_policy.json"

# Re-entry thresholds: how many positive shadow windows before
# a disabled symbol can be reconsidered
REENTRY_SHADOW_WINDOWS = 3       # Need 3 windows of positive shadow data
REENTRY_MIN_SHADOW_ENTRIES = 5   # Need at least 5 shadow observations


def generate_policy_scorecard():
    """
    Generate a comprehensive policy audit report.

    Returns:
        Human-readable text report.
    """
    shadow_data = _load_shadow_log()
    policy_log = _load_policy_log()
    current_policy = _load_current_policy()

    lines = []
    lines.append("=" * 85)
    lines.append("ALLOCATION POLICY SCORECARD")
    lines.append("=" * 85)
    lines.append("")

    # ── Current policy snapshot ───────────────────────────────────────
    lines.append("  CURRENT POLICY:")
    if not current_policy:
        lines.append("    No policy active yet.")
    else:
        lines.append(
            f"    {'SYMBOL':10s} {'ACTION':22s} {'CONF':12s} "
            f"{'TRADES':>6s} {'CLASS':20s}"
        )
        lines.append("    " + "-" * 72)
        for sym in sorted(current_policy.keys()):
            p = current_policy[sym]
            lines.append(
                f"    {sym:10s} {p.get('action', '?'):22s} "
                f"{p.get('confidence', '?'):12s} "
                f"{p.get('trade_count', 0):>6d} "
                f"{p.get('classification', '?'):20s}"
            )
    lines.append("")

    # ── Confidence distribution ───────────────────────────────────────
    if current_policy:
        conf_dist = defaultdict(int)
        for p in current_policy.values():
            conf_dist[p.get("confidence", "unknown")] += 1
        lines.append("  CONFIDENCE DISTRIBUTION:")
        for conf in ["high", "moderate", "provisional", "unknown"]:
            if conf in conf_dist:
                lines.append(f"    {conf:15s} {conf_dist[conf]:>3d} symbols")
        lines.append("")

    # ── Shadow mode analysis ──────────────────────────────────────────
    if not shadow_data:
        lines.append("  SHADOW MODE: No data yet (policy hasn't blocked anything).")
        lines.append("")
    else:
        lines.append("  SHADOW MODE ANALYSIS:")
        lines.append(f"    Total block events: {len(shadow_data)}")

        would_selected = sum(1 for e in shadow_data if e.get("would_have_selected"))
        lines.append(f"    Would have been selected: {would_selected}")
        lines.append(
            f"    Selection rate: "
            f"{would_selected / len(shadow_data):.0%} of blocked cycles"
        )
        lines.append("")

        # ── Per-symbol shadow breakdown ───────────────────────────────
        by_symbol = defaultdict(list)
        for e in shadow_data:
            by_symbol[e["symbol"]].append(e)

        lines.append("  PER-SYMBOL SHADOW RESULTS:")
        lines.append(
            f"    {'SYMBOL':10s} {'BLOCKS':>6s} {'WOULD_SEL':>9s} "
            f"{'SEL_RATE':>8s} {'AVG_SCORE':>9s} {'VERDICT':20s}"
        )
        lines.append("    " + "-" * 65)

        for sym in sorted(by_symbol.keys()):
            entries = by_symbol[sym]
            n = len(entries)
            sel = sum(1 for e in entries if e.get("would_have_selected"))
            scores = [
                e["would_have_scored"] for e in entries
                if e.get("would_have_scored") is not None
            ]
            avg_score = sum(scores) / len(scores) if scores else 0

            # Per-symbol PnL verdict
            pnl_entries = [e for e in entries if e.get("hypothetical_pnl") is not None]
            if pnl_entries:
                avg_pnl = sum(e["hypothetical_pnl"] for e in pnl_entries) / len(pnl_entries)
                if avg_pnl < -0.001:
                    verdict = "HELPING"
                elif avg_pnl > 0.001:
                    verdict = "HURTING"
                else:
                    verdict = "NEUTRAL"
            elif sel == 0:
                verdict = "HELPING (never top)"
            else:
                verdict = "AWAITING PnL"

            lines.append(
                f"    {sym:10s} {n:>6d} {sel:>9d} "
                f"{sel / n:>7.0%} {avg_score:>+9.3f} {verdict:20s}"
            )
        lines.append("")

        # ── Per-regime shadow breakdown ───────────────────────────────
        by_regime = defaultdict(list)
        for e in shadow_data:
            by_regime[e.get("regime_phase", "unknown")].append(e)

        lines.append("  PER-REGIME SHADOW RESULTS:")
        lines.append(
            f"    {'REGIME':12s} {'BLOCKS':>6s} {'WOULD_SEL':>9s} "
            f"{'SEL_RATE':>8s}"
        )
        lines.append("    " + "-" * 38)

        for regime in sorted(by_regime.keys()):
            entries = by_regime[regime]
            n = len(entries)
            sel = sum(1 for e in entries if e.get("would_have_selected"))
            lines.append(
                f"    {regime:12s} {n:>6d} {sel:>9d} "
                f"{sel / n:>7.0%}"
            )
        lines.append("")

        # ── PnL impact (if available) ─────────────────────────────────
        pnl_entries = [e for e in shadow_data if e.get("hypothetical_pnl") is not None]
        if pnl_entries:
            pnls = [e["hypothetical_pnl"] for e in pnl_entries]
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p <= 0)
            avg = sum(pnls) / len(pnls)

            lines.append("  HYPOTHETICAL PnL IMPACT:")
            lines.append(f"    Trades with PnL data: {len(pnl_entries)}")
            lines.append(f"    Avoided wins: {wins}")
            lines.append(f"    Avoided losses: {losses}")
            lines.append(f"    Avg avoided PnL: {avg:+.4%}")
            lines.append(f"    Total avoided PnL: {sum(pnls):+.4%}")
            lines.append("")

            if avg < -0.001:
                lines.append("    VERDICT: POLICY IS HELPING — avoided net-negative trades")
            elif avg > 0.001:
                lines.append("    VERDICT: POLICY IS HURTING — blocked net-positive trades")
            else:
                lines.append("    VERDICT: POLICY IS NEUTRAL — blocked trades were flat")
            lines.append("")

    # ── Policy stability ──────────────────────────────────────────────
    if len(policy_log) >= 2:
        lines.append("  POLICY STABILITY:")
        lines.append(f"    Review windows logged: {len(policy_log)}")

        # Track classification changes
        changes = _count_classification_changes(policy_log)
        if changes:
            lines.append("    Classification changes detected:")
            for sym, change_count in sorted(changes.items(), key=lambda x: -x[1]):
                lines.append(f"      {sym:10s} changed {change_count} time(s)")
        else:
            lines.append("    No classification changes — policy is stable.")
        lines.append("")

    # ── Re-entry candidates ───────────────────────────────────────────
    reentry = find_reentry_candidates(shadow_data, current_policy)
    if reentry:
        lines.append("  RE-ENTRY CANDIDATES:")
        lines.append("  (Disabled symbols showing improving shadow performance)")
        for candidate in reentry:
            lines.append(
                f"    {candidate['symbol']:10s} — {candidate['reason']}"
            )
        lines.append("")

    return "\n".join(lines)


def find_reentry_candidates(shadow_data=None, current_policy=None):
    """
    Find disabled symbols that may deserve re-entry.

    A symbol qualifies for re-entry consideration if:
    1. Currently disabled
    2. Has enough shadow observations (MIN 5)
    3. Recent shadow windows show improving scores
    4. Would-have-selected rate suggests the symbol is competitive

    Returns:
        List of {symbol, reason, shadow_stats} dicts.
    """
    if shadow_data is None:
        shadow_data = _load_shadow_log()
    if current_policy is None:
        current_policy = _load_current_policy()

    if not shadow_data or not current_policy:
        return []

    # Find disabled symbols
    disabled = {
        sym for sym, p in current_policy.items()
        if p.get("action") == "disable"
    }

    if not disabled:
        return []

    candidates = []

    by_symbol = defaultdict(list)
    for e in shadow_data:
        by_symbol[e["symbol"]].append(e)

    for sym in disabled:
        entries = by_symbol.get(sym, [])
        if len(entries) < REENTRY_MIN_SHADOW_ENTRIES:
            continue

        # Check recent shadow performance
        recent = entries[-REENTRY_SHADOW_WINDOWS:]
        scores = [
            e["would_have_scored"] for e in recent
            if e.get("would_have_scored") is not None
        ]
        selected_count = sum(1 for e in recent if e.get("would_have_selected"))

        # Check PnL if available
        pnl_entries = [e for e in recent if e.get("hypothetical_pnl") is not None]

        reason_parts = []
        qualifies = False

        # Criterion 1: consistently scoring well in shadow
        if scores and sum(scores) / len(scores) > 0.10:
            reason_parts.append(
                f"avg shadow score {sum(scores) / len(scores):+.3f}"
            )
            qualifies = True

        # Criterion 2: would have been selected recently
        if selected_count >= 2:
            reason_parts.append(
                f"would-have-selected {selected_count}/{len(recent)} recent windows"
            )
            qualifies = True

        # Criterion 3: positive hypothetical PnL
        if pnl_entries:
            avg_pnl = sum(e["hypothetical_pnl"] for e in pnl_entries) / len(pnl_entries)
            if avg_pnl > 0:
                reason_parts.append(f"positive shadow PnL ({avg_pnl:+.4%})")
                qualifies = True

        if qualifies:
            candidates.append({
                "symbol": sym,
                "reason": "; ".join(reason_parts),
                "shadow_entries": len(entries),
                "recent_avg_score": sum(scores) / len(scores) if scores else 0,
                "recent_selection_rate": selected_count / len(recent) if recent else 0,
            })

    return candidates


def _count_classification_changes(policy_log):
    """Count how many times each symbol's classification changed."""
    if len(policy_log) < 2:
        return {}

    changes = defaultdict(int)
    prev_entry = policy_log[0].get("policy_summary", {})

    for log_entry in policy_log[1:]:
        current = log_entry.get("policy_summary", {})
        all_syms = set(prev_entry.keys()) | set(current.keys())
        for sym in all_syms:
            prev_cls = prev_entry.get(sym, {}).get("classification")
            curr_cls = current.get(sym, {}).get("classification")
            if prev_cls and curr_cls and prev_cls != curr_cls:
                changes[sym] += 1
        prev_entry = current

    return dict(changes)


def _load_shadow_log():
    """Load shadow log entries."""
    if not os.path.exists(SHADOW_LOG_PATH):
        return []
    entries = []
    try:
        with open(SHADOW_LOG_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except (json.JSONDecodeError, IOError):
        return []
    return entries


def _load_policy_log():
    """Load policy log entries."""
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


def _load_current_policy():
    """Load current policy snapshot."""
    if not os.path.exists(POLICY_PATH):
        return {}
    try:
        with open(POLICY_PATH, "r") as f:
            snapshot = json.load(f)
        return snapshot.get("symbols", {})
    except (json.JSONDecodeError, KeyError):
        return {}
