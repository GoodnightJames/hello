"""
Strategy Validation Framework — proves which features earn their keep.

This module provides the research tools needed BEFORE adding more features:

1. Interaction Matrix — per-trade rejection cascade showing how many
   candidates are blocked by each gate and which gates combine to
   cause accidental paralysis.

2. Feature Isolation — runs the strategy with each v6 feature toggled
   on/off to measure individual contribution.

3. Regime Performance Tables — expectancy/hit rate/profit factor by
   regime phase, with opportunity-loss estimates.

4. Recovery Sensitivity — measures first N entries after regime
   improvement to detect under-participation from over-defense.

5. Health Layer Contribution — quantifies when the health meta-layer
   helps vs hurts, specifically around drawdown bottoms and recovery.

Usage:
    from research.validation import (
        build_interaction_matrix,
        run_feature_isolation,
        build_regime_tables,
        analyze_recovery_sensitivity,
        analyze_health_contribution,
        generate_validation_report,
    )

    report = generate_validation_report(trades, cycles, scored_history)
"""

import json
import os
from collections import defaultdict
from datetime import datetime

from core.logging import get_logger

logger = get_logger("research.validation")

VALIDATION_DIR = "reports/validation"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. INTERACTION MATRIX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_interaction_matrix(cycle_history):
    """
    Build a rejection cascade showing where candidates die in the pipeline.

    For each candidate across all recorded cycles, tracks:
    - Passed raw ranking (score > 0)?
    - Failed regime-adjusted threshold?
    - Failed cost gate?
    - Failed universe eligibility?
    - Failed budget scale / min notional?
    - Reduced by sleeve health?
    - Actually entered?
    - Exit type (if entered)?

    Args:
        cycle_history: List of cycle dicts from CycleDashboard.
                       Each has "meta" and "candidates" keys.

    Returns:
        Dict with rejection counts, combination rates, and marginal analysis.
    """
    total_candidates = 0
    rejection_counts = defaultdict(int)
    combination_counts = defaultdict(int)
    entered_count = 0
    entered_scores = []
    rejected_scores = defaultdict(list)

    for cycle in cycle_history:
        meta = cycle.get("meta", {})
        candidates = cycle.get("candidates", [])

        for c in candidates:
            total_candidates += 1
            score = c.get("rank_score", 0)
            reasons = []

            # Check each gate
            if not c.get("threshold_pass", False):
                reasons.append("below_threshold")
                rejection_counts["below_threshold"] += 1

            if not c.get("cost_gate_pass", False):
                reasons.append("cost_gate_fail")
                rejection_counts["cost_gate_fail"] += 1

            reject_reason = c.get("reject_reason", "")
            if "budget" in reject_reason:
                reasons.append("budget_blocked")
                rejection_counts["budget_blocked"] += 1
            if "min_notional" in reject_reason:
                reasons.append("min_notional_fail")
                rejection_counts["min_notional_fail"] += 1
            if "not_top_ranked" in reject_reason:
                reasons.append("not_top_ranked")
                rejection_counts["not_top_ranked"] += 1

            if c.get("selected", False):
                entered_count += 1
                entered_scores.append(score)
            else:
                combo_key = "+".join(sorted(reasons)) if reasons else "other"
                combination_counts[combo_key] += 1
                for r in reasons:
                    rejected_scores[r].append(score)

    # Marginal analysis: what would happen if each gate were removed?
    marginal = {}
    for gate, scores in rejected_scores.items():
        marginal[gate] = {
            "blocked_count": len(scores),
            "avg_score_of_blocked": round(
                sum(scores) / len(scores), 4
            ) if scores else 0,
            "max_score_blocked": round(max(scores), 4) if scores else 0,
            "pct_of_total": round(len(scores) / total_candidates, 3) if total_candidates > 0 else 0,
        }

    result = {
        "total_candidates": total_candidates,
        "entered": entered_count,
        "entry_rate": round(entered_count / total_candidates, 3) if total_candidates > 0 else 0,
        "rejection_by_gate": dict(rejection_counts),
        "rejection_combinations": dict(sorted(
            combination_counts.items(), key=lambda x: x[1], reverse=True
        )),
        "marginal_analysis": marginal,
        "avg_entered_score": round(
            sum(entered_scores) / len(entered_scores), 4
        ) if entered_scores else 0,
    }

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. FEATURE ISOLATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_feature_isolation(trades, feature_tags=None):
    """
    Evaluate each v6 feature's individual contribution.

    Each trade should be tagged with which features were active when it
    was taken. This allows comparing:
    - v5 baseline (no v6 features)
    - v5 + regime_thresholds only
    - v5 + time_decay only
    - v5 + momentum_collapse only
    - v5 + health_layer only
    - v6 full stack

    Args:
        trades: List of trade dicts, each with:
            - pnl_after_cost: float
            - regime_thresholds_active: bool
            - time_decay_active: bool
            - momentum_collapse_active: bool
            - health_layer_active: bool
            - exit_type: str
        feature_tags: List of feature names to analyze.
                     Defaults to v6 feature set.

    Returns:
        Dict with per-feature and combined performance comparison.
    """
    if feature_tags is None:
        feature_tags = [
            "regime_thresholds",
            "time_decay",
            "momentum_collapse",
            "health_layer",
        ]

    results = {}

    # Overall baseline
    results["all_trades"] = _compute_trade_stats(trades)

    # Per feature: trades where this feature was active
    for feature in feature_tags:
        key = f"{feature}_active"
        active = [t for t in trades if t.get(key, False)]
        inactive = [t for t in trades if not t.get(key, False)]

        results[feature] = {
            "active": _compute_trade_stats(active),
            "inactive": _compute_trade_stats(inactive),
            "contribution": _compute_contribution(active, inactive),
        }

    # Trades prevented by each exit feature
    for exit_feature in ["time_decay", "momentum_collapse"]:
        exits = [t for t in trades if t.get("exit_type") == exit_feature]
        results[f"{exit_feature}_exits"] = _compute_trade_stats(exits)

    return results


def _compute_trade_stats(trades):
    """Compute standard performance metrics for a trade set."""
    if not trades:
        return {
            "count": 0, "expectancy": 0, "hit_rate": 0,
            "avg_winner": 0, "avg_loser": 0, "profit_factor": 0,
            "total_pnl": 0,
        }

    pnls = [t.get("pnl_after_cost", t.get("pnl_pct", 0)) for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    gross_wins = sum(winners)
    gross_losses = abs(sum(losers))

    return {
        "count": len(trades),
        "expectancy": round(sum(pnls) / len(pnls), 6),
        "hit_rate": round(len(winners) / len(pnls), 3),
        "avg_winner": round(gross_wins / len(winners), 6) if winners else 0,
        "avg_loser": round(-gross_losses / len(losers), 6) if losers else 0,
        "profit_factor": (
            round(gross_wins / gross_losses, 2) if gross_losses > 0 else
            float("inf") if gross_wins > 0 else 0
        ),
        "total_pnl": round(sum(pnls), 6),
    }


def _compute_contribution(active_trades, inactive_trades):
    """Compute the marginal contribution of a feature."""
    active_stats = _compute_trade_stats(active_trades)
    inactive_stats = _compute_trade_stats(inactive_trades)

    return {
        "expectancy_delta": round(
            active_stats["expectancy"] - inactive_stats["expectancy"], 6
        ),
        "hit_rate_delta": round(
            active_stats["hit_rate"] - inactive_stats["hit_rate"], 3
        ),
        "trade_count_delta": active_stats["count"] - inactive_stats["count"],
        "verdict": (
            "helpful" if active_stats["expectancy"] > inactive_stats["expectancy"]
            else "harmful" if active_stats["expectancy"] < inactive_stats["expectancy"]
            else "neutral"
        ),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. REGIME PERFORMANCE TABLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_regime_tables(trades, cycles=None):
    """
    Build comprehensive regime-by-regime performance tables.

    Args:
        trades: List of trade dicts with regime_phase, pnl_after_cost.
        cycles: Optional list of cycle dicts for opportunity-loss analysis.

    Returns:
        Dict with per-regime stats and opportunity-loss estimates.
    """
    by_regime = defaultdict(list)
    for t in trades:
        phase = t.get("regime_phase", "unknown")
        by_regime[phase].append(t)

    regime_stats = {}
    for phase in sorted(by_regime.keys()):
        stats = _compute_trade_stats(by_regime[phase])
        regime_stats[phase] = stats

    # Opportunity loss: how many cycles had no entry per regime?
    opportunity_loss = {}
    if cycles:
        regime_cycles = defaultdict(lambda: {"total": 0, "traded": 0, "skipped": 0})
        for cycle in cycles:
            meta = cycle.get("meta", {})
            regime = meta.get("regime", {})
            phase = regime.get("phase", "unknown")
            regime_cycles[phase]["total"] += 1

            candidates = cycle.get("candidates", [])
            had_entry = any(c.get("selected") for c in candidates)
            if had_entry:
                regime_cycles[phase]["traded"] += 1
            else:
                regime_cycles[phase]["skipped"] += 1

        for phase, counts in regime_cycles.items():
            total = counts["total"]
            opportunity_loss[phase] = {
                "total_cycles": total,
                "traded_cycles": counts["traded"],
                "skipped_cycles": counts["skipped"],
                "skip_rate": round(counts["skipped"] / total, 3) if total > 0 else 0,
            }

    # False negative analysis: would skipped trades have been profitable?
    # (Requires scored_history with hypothetical entries)

    return {
        "by_regime": regime_stats,
        "opportunity_loss": opportunity_loss,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. RECOVERY SENSITIVITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def analyze_recovery_sensitivity(trades, cycles, window=5):
    """
    Measure first N entries after regime improvement.

    Detects under-participation from over-defense — the classic problem
    in systems that get smarter about defense.

    Looks for regime transitions:
    - crisis → correction
    - correction → ranging
    - ranging → trending

    And measures:
    - How many of the first `window` cycles had entries?
    - What was the performance of those entries?
    - Were good opportunities missed?

    Args:
        trades: List of trade dicts with timestamps and regime.
        cycles: List of cycle dicts with regime metadata.
        window: Number of cycles after transition to examine.

    Returns:
        Dict with recovery participation analysis.
    """
    if not cycles or len(cycles) < 2:
        return {"transitions": [], "note": "insufficient_data"}

    # Detect regime transitions
    improvement_order = ["crisis", "correction", "ranging", "trending"]
    transitions = []

    prev_phase = None
    for i, cycle in enumerate(cycles):
        meta = cycle.get("meta", {})
        phase = meta.get("regime", {}).get("phase", "unknown")

        if prev_phase and phase != prev_phase:
            # Check if this is an improvement
            prev_rank = improvement_order.index(prev_phase) if prev_phase in improvement_order else -1
            curr_rank = improvement_order.index(phase) if phase in improvement_order else -1

            if curr_rank > prev_rank:
                transitions.append({
                    "cycle_index": i,
                    "from_phase": prev_phase,
                    "to_phase": phase,
                    "timestamp": meta.get("timestamp", ""),
                })

        prev_phase = phase

    # Analyze post-transition behavior
    recovery_analysis = []
    for transition in transitions:
        idx = transition["cycle_index"]
        post_cycles = cycles[idx:idx + window]

        entries = 0
        skips = 0
        entry_pnls = []

        for cycle in post_cycles:
            candidates = cycle.get("candidates", [])
            selected = [c for c in candidates if c.get("selected")]

            if selected:
                entries += 1
                # Find matching trades by timestamp proximity
                for s in selected:
                    sym = s.get("symbol", "")
                    matching = [
                        t for t in trades
                        if t.get("symbol") == sym
                    ]
                    if matching:
                        entry_pnls.append(matching[-1].get("pnl_after_cost", 0))
            else:
                skips += 1

        recovery_analysis.append({
            **transition,
            "window_size": len(post_cycles),
            "entries_taken": entries,
            "entries_skipped": skips,
            "participation_rate": round(
                entries / len(post_cycles), 3
            ) if post_cycles else 0,
            "avg_recovery_pnl": round(
                sum(entry_pnls) / len(entry_pnls), 6
            ) if entry_pnls else 0,
            "recovery_positive": sum(1 for p in entry_pnls if p > 0),
        })

    # Summary
    all_participation = [r["participation_rate"] for r in recovery_analysis]
    avg_participation = (
        round(sum(all_participation) / len(all_participation), 3)
        if all_participation else 0
    )

    return {
        "transitions_found": len(transitions),
        "recovery_analysis": recovery_analysis,
        "avg_recovery_participation": avg_participation,
        "under_participation_risk": avg_participation < 0.40,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. HEALTH LAYER CONTRIBUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def analyze_health_contribution(trades, health_history):
    """
    Quantify when the sleeve health meta-layer helps vs hurts.

    Specifically tests for:
    - Does reducing aggressiveness during drawdowns help?
    - Does it cause under-participation at bottoms?
    - Does re-engagement lag after recovery?

    Args:
        trades: List of trade dicts with health_score, aggressiveness.
        health_history: List of health check dicts over time.

    Returns:
        Dict with health layer analysis.
    """
    if not trades:
        return {"note": "no_trades"}

    # Split trades by health state
    healthy = [t for t in trades if t.get("health_score", 1.0) >= 0.7]
    caution = [t for t in trades if 0.5 <= t.get("health_score", 1.0) < 0.7]
    impaired = [t for t in trades if t.get("health_score", 1.0) < 0.5]

    # Performance by health state
    by_health = {
        "healthy (>=0.7)": _compute_trade_stats(healthy),
        "caution (0.5-0.7)": _compute_trade_stats(caution),
        "impaired (<0.5)": _compute_trade_stats(impaired),
    }

    # Key question: do trades taken during impaired state perform worse?
    # If yes, the health layer is correctly reducing exposure.
    # If no, it's just limiting upside without reducing risk.
    impaired_expect = by_health["impaired (<0.5)"]["expectancy"]
    healthy_expect = by_health["healthy (>=0.7)"]["expectancy"]

    verdict = (
        "protective" if impaired_expect < healthy_expect and impaired_expect < 0
        else "overcautious" if impaired_expect >= 0
        else "neutral"
    )

    # Analyze drawdown-to-recovery transitions in health history
    recovery_lag = None
    if health_history and len(health_history) > 3:
        # Find sequences where health dropped below 0.5 then recovered
        was_impaired = False
        recovery_delays = []
        impaired_start = None

        for i, h in enumerate(health_history):
            score = h.get("score", 1.0)
            if score < 0.5 and not was_impaired:
                was_impaired = True
                impaired_start = i
            elif score >= 0.7 and was_impaired:
                was_impaired = False
                if impaired_start is not None:
                    recovery_delays.append(i - impaired_start)

        if recovery_delays:
            recovery_lag = {
                "avg_recovery_periods": round(
                    sum(recovery_delays) / len(recovery_delays), 1
                ),
                "max_recovery_periods": max(recovery_delays),
                "recovery_episodes": len(recovery_delays),
            }

    return {
        "by_health_state": by_health,
        "verdict": verdict,
        "trades_during_impairment": len(impaired),
        "trades_during_health": len(healthy),
        "recovery_lag": recovery_lag,
        "recommendation": (
            "keep as control layer" if verdict == "protective"
            else "demote to reporting only" if verdict == "overcautious"
            else "needs more data"
        ),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. SYMBOL CULLING POLICY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_culling_policy(symbol_classification, review_window=4):
    """
    Build enforceable symbol culling rules.

    Policy:
    - dead_weight for N consecutive review windows → DISABLE
    - conditional_earner → only trade in favorable regimes
    - core_earner → normal treatment
    - insufficient_data → continue monitoring

    Args:
        symbol_classification: Dict from SymbolTracker.classify_symbols().
                               Can be a list of historical classifications.
        review_window: Consecutive dead_weight reviews before culling.

    Returns:
        Dict with per-symbol policy decisions.
    """
    # If single classification (not historical), treat as latest snapshot
    if isinstance(symbol_classification, dict) and "by_symbol" not in symbol_classification:
        # Single point-in-time classification
        policies = {}
        for sym, data in symbol_classification.items():
            cls = data.get("classification", "insufficient_data")
            stats = data.get("stats", {})

            if cls == "dead_weight":
                policies[sym] = {
                    "action": "flag_for_removal",
                    "reason": (
                        f"Dead weight: expectancy={stats.get('expectancy', 0):+.4%}, "
                        f"PF={stats.get('profit_factor', 0):.2f}"
                    ),
                    "allowed_regimes": [],
                }
            elif cls == "conditional_earner":
                # Find which regimes are profitable
                regime_bd = data.get("regime_breakdown", {})
                good_regimes = [
                    r for r, s in regime_bd.items()
                    if s and s.get("expectancy", 0) > 0
                ]
                policies[sym] = {
                    "action": "restrict_to_regimes",
                    "reason": f"Conditional earner: profitable in {good_regimes}",
                    "allowed_regimes": good_regimes,
                }
            elif cls == "core_earner":
                policies[sym] = {
                    "action": "normal",
                    "reason": "Core earner: consistently profitable after costs",
                    "allowed_regimes": ["all"],
                }
            else:
                policies[sym] = {
                    "action": "monitor",
                    "reason": "Insufficient data for classification",
                    "allowed_regimes": ["all"],
                }

        return policies

    return {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMBINED REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_validation_report(trades=None, cycles=None, health_history=None,
                               symbol_classification=None):
    """
    Generate a comprehensive validation report combining all analyses.

    Args:
        trades: List of completed trade dicts.
        cycles: List of cycle dashboard dicts.
        health_history: List of sleeve health check results.
        symbol_classification: Dict from SymbolTracker.classify_symbols().

    Returns:
        (report_dict, report_text)
    """
    report = {}

    # 1. Interaction matrix
    if cycles:
        report["interaction_matrix"] = build_interaction_matrix(cycles)

    # 2. Feature isolation
    if trades:
        report["feature_isolation"] = run_feature_isolation(trades)

    # 3. Regime tables
    if trades:
        report["regime_tables"] = build_regime_tables(trades, cycles)

    # 4. Recovery sensitivity
    if trades and cycles:
        report["recovery_sensitivity"] = analyze_recovery_sensitivity(
            trades, cycles
        )

    # 5. Health contribution
    if trades:
        report["health_contribution"] = analyze_health_contribution(
            trades, health_history or []
        )

    # 6. Culling policy
    if symbol_classification:
        report["culling_policy"] = build_culling_policy(symbol_classification)

    # Format text report
    text = format_validation_report(report)

    # Save to disk
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(VALIDATION_DIR, f"validation_{timestamp}.json")
    text_path = os.path.join(VALIDATION_DIR, f"validation_{timestamp}.txt")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(text_path, "w") as f:
        f.write(text)

    logger.info(
        f"Validation report saved: {json_path}",
        extra={"extra_data": {"sections": list(report.keys())}},
    )

    return report, text


def format_validation_report(report):
    """Format validation report as human-readable text."""
    lines = []
    lines.append("=" * 90)
    lines.append("STRATEGY VALIDATION REPORT")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("=" * 90)

    # 1. Interaction Matrix
    im = report.get("interaction_matrix", {})
    if im:
        lines.append("")
        lines.append("--- INTERACTION MATRIX ---")
        lines.append(f"  Total candidates evaluated: {im.get('total_candidates', 0)}")
        lines.append(f"  Entered: {im.get('entered', 0)} ({im.get('entry_rate', 0):.0%})")
        lines.append("")
        lines.append("  Rejection by gate:")
        for gate, count in sorted(
            im.get("rejection_by_gate", {}).items(),
            key=lambda x: x[1], reverse=True
        ):
            pct = count / im["total_candidates"] if im["total_candidates"] > 0 else 0
            lines.append(f"    {gate:25s} {count:>5d} ({pct:.0%})")

        lines.append("")
        lines.append("  Top rejection combinations:")
        for combo, count in list(im.get("rejection_combinations", {}).items())[:8]:
            lines.append(f"    {combo:40s} {count:>5d}")

        lines.append("")
        lines.append("  Marginal analysis (if gate removed):")
        for gate, data in im.get("marginal_analysis", {}).items():
            lines.append(
                f"    {gate:25s} would unblock {data['blocked_count']:>4d} "
                f"(avg score {data['avg_score_of_blocked']:+.3f})"
            )

    # 2. Feature Isolation
    fi = report.get("feature_isolation", {})
    if fi:
        lines.append("")
        lines.append("--- FEATURE ISOLATION ---")
        all_stats = fi.get("all_trades", {})
        lines.append(
            f"  All trades: {all_stats.get('count', 0)} trades, "
            f"expectancy={all_stats.get('expectancy', 0):+.4%}, "
            f"PF={all_stats.get('profit_factor', 0):.2f}"
        )
        lines.append("")

        for feature in ["regime_thresholds", "time_decay",
                        "momentum_collapse", "health_layer"]:
            data = fi.get(feature, {})
            if not data:
                continue
            contrib = data.get("contribution", {})
            active = data.get("active", {})
            inactive = data.get("inactive", {})
            verdict = contrib.get("verdict", "?")
            lines.append(
                f"  {feature:25s}  "
                f"active={active.get('count', 0):>4d} expect={active.get('expectancy', 0):+.4%}  "
                f"inactive={inactive.get('count', 0):>4d} expect={inactive.get('expectancy', 0):+.4%}  "
                f"→ {verdict.upper()}"
            )

    # 3. Regime Tables
    rt = report.get("regime_tables", {})
    if rt:
        lines.append("")
        lines.append("--- REGIME PERFORMANCE ---")
        lines.append(
            f"  {'PHASE':15s} {'N':>5s} {'HIT%':>6s} {'EXPECT':>9s} "
            f"{'PF':>6s} {'TOTAL':>9s}"
        )
        lines.append("  " + "-" * 55)
        for phase, stats in sorted(rt.get("by_regime", {}).items()):
            pf = stats.get("profit_factor", 0)
            pf_str = f"{pf:>6.2f}" if pf != float("inf") else "   inf"
            lines.append(
                f"  {phase:15s} {stats.get('count', 0):>5d} "
                f"{stats.get('hit_rate', 0):>5.0%} "
                f"{stats.get('expectancy', 0):>+9.4%} "
                f"{pf_str} "
                f"{stats.get('total_pnl', 0):>+9.4%}"
            )

        opp = rt.get("opportunity_loss", {})
        if opp:
            lines.append("")
            lines.append("  Opportunity loss (skipped cycles by regime):")
            for phase, data in sorted(opp.items()):
                lines.append(
                    f"    {phase:15s} {data['traded_cycles']}/{data['total_cycles']} "
                    f"traded (skip rate: {data['skip_rate']:.0%})"
                )

    # 4. Recovery Sensitivity
    rs = report.get("recovery_sensitivity", {})
    if rs and rs.get("transitions_found", 0) > 0:
        lines.append("")
        lines.append("--- RECOVERY SENSITIVITY ---")
        lines.append(f"  Regime transitions found: {rs['transitions_found']}")
        lines.append(f"  Avg recovery participation: {rs.get('avg_recovery_participation', 0):.0%}")

        if rs.get("under_participation_risk"):
            lines.append(
                "  WARNING: Under-participation risk detected (<40% participation "
                "in first 5 cycles after regime improvement)"
            )

        for r in rs.get("recovery_analysis", []):
            lines.append(
                f"    {r['from_phase']} → {r['to_phase']}: "
                f"{r['entries_taken']}/{r['window_size']} entries "
                f"({r['participation_rate']:.0%}), "
                f"avg PnL: {r['avg_recovery_pnl']:+.4%}"
            )

    # 5. Health Contribution
    hc = report.get("health_contribution", {})
    if hc and hc.get("note") != "no_trades":
        lines.append("")
        lines.append("--- HEALTH LAYER ANALYSIS ---")
        lines.append(f"  Verdict: {hc.get('verdict', '?').upper()}")
        lines.append(f"  Recommendation: {hc.get('recommendation', '?')}")
        lines.append(
            f"  Trades during health: {hc.get('trades_during_health', 0)} | "
            f"during impairment: {hc.get('trades_during_impairment', 0)}"
        )

        for state, stats in hc.get("by_health_state", {}).items():
            if stats.get("count", 0) > 0:
                lines.append(
                    f"    {state:20s} {stats['count']:>4d} trades, "
                    f"expect={stats['expectancy']:+.4%}, "
                    f"PF={stats['profit_factor']:.2f}"
                )

        lag = hc.get("recovery_lag")
        if lag:
            lines.append(
                f"  Recovery lag: avg {lag['avg_recovery_periods']:.0f} periods, "
                f"max {lag['max_recovery_periods']} periods"
            )

    # 6. Culling Policy
    cp = report.get("culling_policy", {})
    if cp:
        lines.append("")
        lines.append("--- SYMBOL CULLING POLICY ---")
        for sym, policy in sorted(cp.items()):
            action = policy.get("action", "?")
            reason = policy.get("reason", "")
            regimes = policy.get("allowed_regimes", [])
            regime_str = ", ".join(regimes) if regimes else "none"
            lines.append(
                f"  {sym:10s} [{action:20s}] regimes={regime_str}"
            )
            lines.append(f"             {reason}")

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)
