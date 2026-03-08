"""
Universe Stability Report — tracks symbol eligibility over time.

A flickering universe (symbols toggling eligible/ineligible every few cycles)
causes churn and undermines signal quality. Stable inclusion is a sign of
a governed, well-calibrated listing rule set.

Per-symbol metrics:
- Eligibility rate (% of cycles where symbol was eligible)
- Average consecutive eligible periods
- Most common rejection reason
- Rejection breakdown (spread / stale / history / range)

Usage:
    from review.universe_report import UniverseTracker

    tracker = UniverseTracker()
    tracker.record_cycle(eligible=["BTC/USD", "ETH/USD"],
                         removed=[("DOGE/USD", ["spread too wide: 15 bps > 10 bps max"])])
    report = tracker.summarize()
"""

import json
import os
from collections import defaultdict
from datetime import datetime

from core.logging import get_logger

logger = get_logger("review.universe_report")

UNIVERSE_LOG_PATH = "reports/universe_history.jsonl"


class UniverseTracker:
    """Tracks universe eligibility stability across cycles."""

    def __init__(self):
        self._cycles = []

    def record_cycle(self, eligible, removed, all_symbols=None, timestamp=None):
        """
        Record one universe filter cycle.

        Args:
            eligible: List of symbols that passed listing rules.
            removed: List of (symbol, [reasons]) for rejected symbols.
            all_symbols: Full configured universe (eligible + removed).
            timestamp: Override timestamp (default: now).
        """
        if all_symbols is None:
            all_symbols = list(eligible) + [s for s, _ in removed]

        entry = {
            "timestamp": (timestamp or datetime.utcnow()).isoformat() + "Z",
            "all_symbols": all_symbols,
            "eligible": eligible,
            "removed": {sym: reasons for sym, reasons in removed},
        }
        self._cycles.append(entry)

        # Persist
        os.makedirs(os.path.dirname(UNIVERSE_LOG_PATH), exist_ok=True)
        with open(UNIVERSE_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def summarize(self):
        """
        Compute per-symbol eligibility stability metrics.

        Returns:
            Dict with per-symbol stats and overall stability score.
        """
        if not self._cycles:
            return {"total_cycles": 0}

        n = len(self._cycles)

        # Collect all symbols ever seen
        all_syms = set()
        for c in self._cycles:
            all_syms.update(c.get("all_symbols", []))

        symbol_stats = {}
        for sym in sorted(all_syms):
            eligible_count = 0
            rejection_reasons = defaultdict(int)
            consecutive_eligible = []
            current_streak = 0

            for c in self._cycles:
                if sym in c.get("eligible", []):
                    eligible_count += 1
                    current_streak += 1
                else:
                    if current_streak > 0:
                        consecutive_eligible.append(current_streak)
                    current_streak = 0
                    # Count rejection reasons
                    removed = c.get("removed", {})
                    if sym in removed:
                        for reason in removed[sym]:
                            # Categorize by first word
                            category = _categorize_reason(reason)
                            rejection_reasons[category] += 1

            if current_streak > 0:
                consecutive_eligible.append(current_streak)

            eligibility_rate = eligible_count / n if n > 0 else 0
            avg_streak = (
                sum(consecutive_eligible) / len(consecutive_eligible)
                if consecutive_eligible else 0
            )

            # Most common rejection
            top_reason = max(rejection_reasons.items(), key=lambda x: x[1])[0] if rejection_reasons else None

            # Flicker score: high eligibility variance = bad
            # A symbol that's always in or always out is stable.
            # A symbol that toggles is flickering.
            is_flickering = (
                0.15 < eligibility_rate < 0.85 and len(consecutive_eligible) > 2
            )

            symbol_stats[sym] = {
                "eligibility_rate": round(eligibility_rate, 3),
                "eligible_cycles": eligible_count,
                "total_cycles": n,
                "avg_consecutive_eligible": round(avg_streak, 1),
                "max_consecutive_eligible": max(consecutive_eligible) if consecutive_eligible else 0,
                "top_rejection_reason": top_reason,
                "rejection_breakdown": dict(rejection_reasons),
                "is_flickering": is_flickering,
            }

        # Overall stability
        flickering_count = sum(
            1 for s in symbol_stats.values() if s["is_flickering"]
        )
        fully_stable = sum(
            1 for s in symbol_stats.values()
            if s["eligibility_rate"] >= 0.95 or s["eligibility_rate"] <= 0.05
        )

        report = {
            "total_cycles": n,
            "total_symbols": len(all_syms),
            "fully_stable_symbols": fully_stable,
            "flickering_symbols": flickering_count,
            "stability_score": round(
                fully_stable / len(all_syms), 3
            ) if all_syms else 1.0,
            "by_symbol": symbol_stats,
        }

        return report

    def format_text(self, report=None):
        """Format universe stability report as human-readable text."""
        if report is None:
            report = self.summarize()

        if report.get("total_cycles", 0) == 0:
            return "UNIVERSE STABILITY REPORT: No cycles recorded.\n"

        lines = []
        lines.append("=" * 80)
        lines.append("UNIVERSE STABILITY REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"  Total cycles: {report['total_cycles']}")
        lines.append(f"  Total symbols tracked: {report['total_symbols']}")
        lines.append(f"  Fully stable (>95% or <5% eligible): {report['fully_stable_symbols']}")
        lines.append(f"  Flickering (toggling eligibility): {report['flickering_symbols']}")
        lines.append(f"  Stability score: {report['stability_score']:.0%}")
        lines.append("")

        # Per-symbol table
        lines.append(
            f"  {'SYMBOL':10s} {'ELIG%':>6s} {'STREAK':>6s} {'MAX':>4s} "
            f"{'FLICKER':>7s} {'TOP REJECTION REASON'}"
        )
        lines.append("  " + "-" * 70)

        for sym in sorted(report["by_symbol"].keys()):
            s = report["by_symbol"][sym]
            flicker = "YES" if s["is_flickering"] else ""
            top_reason = s["top_rejection_reason"] or ""
            lines.append(
                f"  {sym:10s} {s['eligibility_rate']:>5.0%} "
                f"{s['avg_consecutive_eligible']:>6.1f} "
                f"{s['max_consecutive_eligible']:>4d} "
                f"{flicker:>7s} {top_reason}"
            )

        # Rejection breakdown for flickering symbols
        flickering = [
            (sym, data) for sym, data in report["by_symbol"].items()
            if data["is_flickering"]
        ]
        if flickering:
            lines.append("")
            lines.append("  FLICKERING SYMBOLS — REJECTION BREAKDOWN:")
            for sym, data in flickering:
                lines.append(f"    {sym}:")
                for reason, count in sorted(
                    data["rejection_breakdown"].items(),
                    key=lambda x: x[1], reverse=True
                ):
                    lines.append(f"      [{count:>3d}] {reason}")

        lines.append("")

        if report["flickering_symbols"] > 0:
            lines.append(
                "  NOTE: Flickering symbols cause churn. Consider tightening "
                "listing thresholds to make inclusion binary, or add hysteresis "
                "(require N consecutive eligible cycles before re-admission)."
            )
            lines.append("")

        return "\n".join(lines)


def _categorize_reason(reason_str):
    """Extract a short category from a rejection reason string."""
    reason_lower = reason_str.lower()
    if "spread" in reason_lower:
        return "spread_violation"
    elif "stale" in reason_lower or "freshness" in reason_lower:
        return "stale_data"
    elif "history" in reason_lower or "insufficient" in reason_lower:
        return "insufficient_history"
    elif "range" in reason_lower or "movement" in reason_lower:
        return "low_price_movement"
    else:
        return "other"


def load_universe_history():
    """Load historical universe observations from the JSONL log."""
    tracker = UniverseTracker()
    if not os.path.exists(UNIVERSE_LOG_PATH):
        return tracker

    with open(UNIVERSE_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tracker._cycles.append(json.loads(line))

    return tracker
