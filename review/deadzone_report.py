"""
Budget Dead-Zone Report — tracks signal funnel attrition.

Monitors how many trade candidates pass each gate in the pipeline
and where they get blocked. A healthy strategy has selective filtering,
not accidental paralysis.

Funnel stages:
1. Universe eligible (pass listing rules)
2. Score threshold (pass rank gate)
3. Cost gate (pass edge_ratio >= min_edge_ratio)
4. Budget approved (not blocked by drawdown/vol/turnover)
5. Min notional (not too small after budget scaling)
6. Executed (actually filled)

If stages 4-5 consistently block all candidates that passed 1-3,
the system is in a "dead zone" — architecturally correct but
functionally paralyzed.

Usage:
    from review.deadzone_report import DeadzoneTracker

    tracker = DeadzoneTracker()
    tracker.record_cycle(...)
    report = tracker.summarize()
"""

import json
import os
from collections import defaultdict
from datetime import datetime

from core.logging import get_logger

logger = get_logger("review.deadzone_report")

DEADZONE_LOG_PATH = "reports/deadzone_history.jsonl"


class DeadzoneTracker:
    """Tracks signal funnel attrition across cycles."""

    def __init__(self):
        self._cycles = []

    def record_cycle(self, cycle_type="crypto",
                     universe_total=0,
                     universe_eligible=0,
                     passed_score_threshold=0,
                     passed_cost_gate=0,
                     passed_budget=0,
                     passed_min_notional=0,
                     executed=0,
                     budget_scale=1.0,
                     blocked_reasons=None,
                     timestamp=None):
        """
        Record funnel counts for one cycle.

        Args:
            universe_total: Total symbols in configured universe.
            universe_eligible: After listing rules filter.
            passed_score_threshold: Symbols with rank_score >= threshold.
            passed_cost_gate: Symbols with edge_ratio >= min_edge_ratio.
            passed_budget: Not blocked by sleeve risk budgets.
            passed_min_notional: Not below min notional after scaling.
            executed: Actually filled orders.
            budget_scale: Current budget scale factor.
            blocked_reasons: Dict of {reason: count} for blocked candidates.
        """
        entry = {
            "timestamp": (timestamp or datetime.utcnow()).isoformat() + "Z",
            "cycle_type": cycle_type,
            "universe_total": universe_total,
            "universe_eligible": universe_eligible,
            "passed_score_threshold": passed_score_threshold,
            "passed_cost_gate": passed_cost_gate,
            "passed_budget": passed_budget,
            "passed_min_notional": passed_min_notional,
            "executed": executed,
            "budget_scale": round(budget_scale, 3),
            "blocked_reasons": blocked_reasons or {},
        }
        self._cycles.append(entry)

        # Persist
        os.makedirs(os.path.dirname(DEADZONE_LOG_PATH), exist_ok=True)
        with open(DEADZONE_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def summarize(self, last_n=None):
        """
        Compute aggregate funnel statistics.

        Args:
            last_n: Only consider the last N cycles (None = all).

        Returns:
            Dict with funnel rates, dead-zone detection, and reason breakdown.
        """
        cycles = self._cycles[-last_n:] if last_n else self._cycles
        if not cycles:
            return {"total_cycles": 0}

        n = len(cycles)

        # Aggregate counts
        totals = {
            "universe_total": sum(c["universe_total"] for c in cycles),
            "universe_eligible": sum(c["universe_eligible"] for c in cycles),
            "passed_score_threshold": sum(c["passed_score_threshold"] for c in cycles),
            "passed_cost_gate": sum(c["passed_cost_gate"] for c in cycles),
            "passed_budget": sum(c["passed_budget"] for c in cycles),
            "passed_min_notional": sum(c["passed_min_notional"] for c in cycles),
            "executed": sum(c["executed"] for c in cycles),
        }

        # Per-cycle rates
        no_signal_cycles = sum(1 for c in cycles if c["passed_score_threshold"] == 0)
        no_edge_cycles = sum(
            1 for c in cycles
            if c["passed_score_threshold"] > 0 and c["passed_cost_gate"] == 0
        )
        budget_blocked_cycles = sum(
            1 for c in cycles
            if c["passed_cost_gate"] > 0 and c["passed_budget"] == 0
        )
        notional_blocked_cycles = sum(
            1 for c in cycles
            if c["passed_budget"] > 0 and c["passed_min_notional"] == 0
        )
        executed_cycles = sum(1 for c in cycles if c["executed"] > 0)

        # Dead zone detection: cost gate passed but downstream blocked
        dead_zone_cycles = budget_blocked_cycles + notional_blocked_cycles

        # Budget scale distribution
        scales = [c["budget_scale"] for c in cycles]
        avg_budget_scale = sum(scales) / len(scales) if scales else 1.0
        below_half_pct = sum(1 for s in scales if s < 0.5) / n if n > 0 else 0

        # Aggregate blocked reasons
        all_reasons = defaultdict(int)
        for c in cycles:
            for reason, count in c.get("blocked_reasons", {}).items():
                all_reasons[reason] += count

        report = {
            "total_cycles": n,
            "totals": totals,
            "cycle_rates": {
                "no_signal_pct": round(no_signal_cycles / n, 3),
                "no_edge_pct": round(no_edge_cycles / n, 3),
                "budget_blocked_pct": round(budget_blocked_cycles / n, 3),
                "notional_blocked_pct": round(notional_blocked_cycles / n, 3),
                "executed_pct": round(executed_cycles / n, 3),
                "dead_zone_pct": round(dead_zone_cycles / n, 3),
            },
            "budget_scale": {
                "average": round(avg_budget_scale, 3),
                "below_50_pct": round(below_half_pct, 3),
            },
            "blocked_reasons": dict(sorted(
                all_reasons.items(), key=lambda x: x[1], reverse=True
            )),
            "is_dead_zone": dead_zone_cycles / n > 0.50 if n > 0 else False,
        }

        return report

    def format_text(self, report=None, last_n=None):
        """Format dead-zone report as human-readable text."""
        if report is None:
            report = self.summarize(last_n=last_n)

        if report.get("total_cycles", 0) == 0:
            return "DEAD-ZONE REPORT: No cycles recorded.\n"

        lines = []
        lines.append("=" * 70)
        lines.append("BUDGET DEAD-ZONE REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Total cycles analyzed: {report['total_cycles']}")
        lines.append("")

        # Funnel
        t = report["totals"]
        lines.append("  SIGNAL FUNNEL (aggregate across all cycles):")
        lines.append(f"    Universe total:        {t['universe_total']}")
        lines.append(f"    Universe eligible:     {t['universe_eligible']}")
        lines.append(f"    Passed score threshold: {t['passed_score_threshold']}")
        lines.append(f"    Passed cost gate:      {t['passed_cost_gate']}")
        lines.append(f"    Passed budget:         {t['passed_budget']}")
        lines.append(f"    Passed min notional:   {t['passed_min_notional']}")
        lines.append(f"    Executed:              {t['executed']}")
        lines.append("")

        # Per-cycle rates
        r = report["cycle_rates"]
        lines.append("  PER-CYCLE RATES:")
        lines.append(f"    No signal (below threshold): {r['no_signal_pct']:.0%}")
        lines.append(f"    No edge (cost gate fail):    {r['no_edge_pct']:.0%}")
        lines.append(f"    Budget blocked:              {r['budget_blocked_pct']:.0%}")
        lines.append(f"    Min notional blocked:        {r['notional_blocked_pct']:.0%}")
        lines.append(f"    Executed at least 1 trade:   {r['executed_pct']:.0%}")
        lines.append(f"    Dead zone (good signal, downstream block): {r['dead_zone_pct']:.0%}")
        lines.append("")

        # Budget scale
        bs = report["budget_scale"]
        lines.append("  BUDGET SCALE:")
        lines.append(f"    Average scale factor:  {bs['average']:.2f}")
        lines.append(f"    Cycles below 0.5x:     {bs['below_50_pct']:.0%}")
        lines.append("")

        # Blocked reasons
        reasons = report.get("blocked_reasons", {})
        if reasons:
            lines.append("  TOP BLOCKED REASONS:")
            for reason, count in list(reasons.items())[:10]:
                lines.append(f"    [{count:>4d}] {reason}")
            lines.append("")

        # Alert
        if report.get("is_dead_zone"):
            lines.append(
                "  ALERT: Dead zone detected. More than 50% of cycles have "
                "viable signals blocked by budget/notional constraints. "
                "The strategy is architecturally sound but functionally "
                "paralyzed. Review budget thresholds and min notional floors."
            )
            lines.append("")

        return "\n".join(lines)


def load_deadzone_history():
    """Load historical dead-zone observations from the JSONL log."""
    tracker = DeadzoneTracker()
    if not os.path.exists(DEADZONE_LOG_PATH):
        return tracker

    with open(DEADZONE_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tracker._cycles.append(json.loads(line))

    return tracker
