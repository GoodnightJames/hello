"""
Symbol Performance Tracker — ranks assets by fee-adjusted expectancy.

Classifies each symbol as:
- core_earner: positive expectancy after costs, consistent
- conditional_earner: positive only in certain regimes
- dead_weight: consistently underperforms after costs → remove

Tracks per symbol:
- Fee-adjusted expectancy (avg PnL after estimated costs)
- Hit rate
- Average winner / average loser
- Profit factor (gross wins / gross losses)
- Average hold time
- Regime dependence
- Average slippage estimate

Usage:
    from review.symbol_tracker import SymbolTracker

    tracker = SymbolTracker()
    tracker.add_trade(symbol, pnl, cost_bps, hold_hours, regime, ...)
    report = tracker.classify_symbols()
"""

import json
import os
from collections import defaultdict
from datetime import datetime

from core.logging import get_logger

logger = get_logger("review.symbol_tracker")

SYMBOL_LOG_PATH = "reports/symbol_performance.jsonl"


class SymbolTracker:
    """Tracks per-symbol trade performance for asset classification."""

    def __init__(self):
        self._trades = []

    def add_trade(self, symbol, pnl_pct, cost_bps=0, hold_hours=0,
                  regime_phase="unknown", entry_score=0, edge_ratio=0,
                  exit_type="unknown", timestamp=None):
        """Record one completed trade."""
        entry = {
            "timestamp": (timestamp or datetime.utcnow()).isoformat() + "Z",
            "symbol": symbol,
            "pnl_pct": round(pnl_pct, 6),
            "cost_bps": round(cost_bps, 2),
            "cost_pct": round(cost_bps / 10000.0, 6),
            "pnl_after_cost": round(pnl_pct - cost_bps / 10000.0, 6),
            "hold_hours": round(hold_hours, 1),
            "regime_phase": regime_phase,
            "entry_score": round(entry_score, 4),
            "edge_ratio": round(edge_ratio, 2),
            "exit_type": exit_type,
            "is_win": pnl_pct > 0,
            "is_win_after_cost": (pnl_pct - cost_bps / 10000.0) > 0,
        }
        self._trades.append(entry)

        os.makedirs(os.path.dirname(SYMBOL_LOG_PATH), exist_ok=True)
        with open(SYMBOL_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _symbol_stats(self, trades):
        """Compute stats for a list of trades."""
        if not trades:
            return None

        n = len(trades)
        pnls = [t["pnl_after_cost"] for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        gross_wins = sum(winners) if winners else 0
        gross_losses = abs(sum(losers)) if losers else 0

        return {
            "trades": n,
            "hit_rate": round(len(winners) / n, 3) if n > 0 else 0,
            "avg_pnl_after_cost": round(sum(pnls) / n, 5) if n > 0 else 0,
            "expectancy": round(sum(pnls) / n, 5) if n > 0 else 0,
            "avg_winner": round(sum(winners) / len(winners), 5) if winners else 0,
            "avg_loser": round(sum(losers) / len(losers), 5) if losers else 0,
            "profit_factor": (
                round(gross_wins / gross_losses, 2) if gross_losses > 0 else
                (float("inf") if gross_wins > 0 else 0)
            ),
            "total_cost_drag": round(
                sum(t["cost_pct"] for t in trades), 5
            ),
            "avg_hold_hours": round(
                sum(t["hold_hours"] for t in trades) / n, 1
            ) if n > 0 else 0,
            "avg_edge_ratio": round(
                sum(t["edge_ratio"] for t in trades) / n, 2
            ) if n > 0 else 0,
        }

    def classify_symbols(self, min_trades=5):
        """
        Classify each symbol by after-cost performance.

        Args:
            min_trades: Minimum trades to classify (otherwise "insufficient_data").

        Returns:
            Dict of {symbol: {stats, classification, regime_breakdown}}.
        """
        by_symbol = defaultdict(list)
        for t in self._trades:
            by_symbol[t["symbol"]].append(t)

        results = {}
        for sym in sorted(by_symbol.keys()):
            trades = by_symbol[sym]
            stats = self._symbol_stats(trades)

            if len(trades) < min_trades:
                classification = "insufficient_data"
            elif stats["expectancy"] > 0.001 and stats["profit_factor"] >= 1.2:
                classification = "core_earner"
            elif stats["expectancy"] > 0:
                # Check regime dependence
                regime_breakdown = self._regime_breakdown(trades)
                positive_regimes = sum(
                    1 for r in regime_breakdown.values()
                    if r.get("expectancy", 0) > 0
                )
                if positive_regimes > 0:
                    classification = "conditional_earner"
                else:
                    classification = "dead_weight"
            else:
                classification = "dead_weight"

            # Regime breakdown
            regime_breakdown = self._regime_breakdown(trades)

            # Exit type breakdown
            exit_breakdown = defaultdict(int)
            for t in trades:
                exit_breakdown[t["exit_type"]] += 1

            results[sym] = {
                "stats": stats,
                "classification": classification,
                "regime_breakdown": regime_breakdown,
                "exit_breakdown": dict(exit_breakdown),
            }

        return results

    def _regime_breakdown(self, trades):
        """Break down performance by regime phase."""
        by_regime = defaultdict(list)
        for t in trades:
            by_regime[t["regime_phase"]].append(t)

        return {
            regime: self._symbol_stats(regime_trades)
            for regime, regime_trades in by_regime.items()
        }

    def get_culling_recommendations(self, min_trades=5):
        """
        Return symbols recommended for removal from the universe.

        A symbol is recommended for removal if:
        - At least min_trades completed
        - Negative expectancy after costs
        - Profit factor < 1.0
        """
        classified = self.classify_symbols(min_trades=min_trades)
        recommendations = []

        for sym, data in classified.items():
            if data["classification"] == "dead_weight":
                stats = data["stats"]
                recommendations.append({
                    "symbol": sym,
                    "reason": (
                        f"Dead weight: expectancy={stats['expectancy']:.4%}, "
                        f"profit_factor={stats['profit_factor']:.2f}, "
                        f"hit_rate={stats['hit_rate']:.0%} "
                        f"over {stats['trades']} trades"
                    ),
                    "stats": stats,
                })

        return recommendations

    def format_text(self, classification=None, min_trades=5):
        """Format symbol performance report as human-readable text."""
        if classification is None:
            classification = self.classify_symbols(min_trades=min_trades)

        if not classification:
            return "SYMBOL PERFORMANCE: No data.\n"

        lines = []
        lines.append("=" * 85)
        lines.append("SYMBOL PERFORMANCE & CLASSIFICATION")
        lines.append("=" * 85)
        lines.append("")

        # Group by classification
        for cls_name in ["core_earner", "conditional_earner", "dead_weight",
                         "insufficient_data"]:
            members = {
                sym: data for sym, data in classification.items()
                if data["classification"] == cls_name
            }
            if not members:
                continue

            label = cls_name.upper().replace("_", " ")
            lines.append(f"  [{label}]")

            lines.append(
                f"    {'SYMBOL':10s} {'TRADES':>6s} {'HIT%':>5s} {'EXPECT':>8s} "
                f"{'PF':>5s} {'AVG_W':>7s} {'AVG_L':>7s} {'HOLD_H':>6s} "
                f"{'COST':>7s}"
            )
            lines.append("    " + "-" * 72)

            for sym in sorted(members.keys()):
                s = members[sym]["stats"]
                lines.append(
                    f"    {sym:10s} {s['trades']:>6d} {s['hit_rate']:>4.0%} "
                    f"{s['expectancy']:>+8.4%} {s['profit_factor']:>5.2f} "
                    f"{s['avg_winner']:>+7.4%} {s['avg_loser']:>+7.4%} "
                    f"{s['avg_hold_hours']:>6.1f} {s['total_cost_drag']:>7.4%}"
                )

            lines.append("")

        # Culling recommendations
        recs = [
            sym for sym, data in classification.items()
            if data["classification"] == "dead_weight"
        ]
        if recs:
            lines.append("  CULLING RECOMMENDATIONS:")
            for sym in recs:
                s = classification[sym]["stats"]
                lines.append(
                    f"    REMOVE {sym}: expectancy={s['expectancy']:+.4%} "
                    f"after {s['trades']} trades"
                )
            lines.append("")

        return "\n".join(lines)


def load_symbol_history():
    """Load historical symbol performance data."""
    tracker = SymbolTracker()
    if not os.path.exists(SYMBOL_LOG_PATH):
        return tracker

    with open(SYMBOL_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tracker._trades.append(json.loads(line))

    return tracker
