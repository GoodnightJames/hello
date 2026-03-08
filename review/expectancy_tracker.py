"""
Expectancy Tracker — breaks down performance by setup type.

Don't just track "crypto sleeve performance." Break it down by:
- symbol
- regime
- score bucket (weak/medium/strong signal)
- cost bucket (tight/moderate/expensive)
- clamp status (was exit clamped?)
- holding time bucket (quick/medium/long)
- exit type (take_profit/hard_stop/trailing/time_decay/momentum_collapse)

Find: which setups make money, which only look active,
and which ones subsidize the losers.

Usage:
    from review.expectancy_tracker import ExpectancyTracker

    tracker = ExpectancyTracker()
    tracker.add_trade(...)
    report = tracker.by_dimension("symbol")
    full = tracker.full_breakdown()
"""

import json
import os
from collections import defaultdict
from datetime import datetime

from core.logging import get_logger

logger = get_logger("review.expectancy_tracker")

EXPECTANCY_LOG_PATH = "reports/expectancy_log.jsonl"


class ExpectancyTracker:
    """Multi-dimensional expectancy analysis."""

    def __init__(self):
        self._trades = []

    def add_trade(self, symbol, pnl_pct, cost_bps=0,
                  entry_score=0, edge_ratio=0,
                  regime_phase="unknown", hold_hours=0,
                  exit_type="unknown", tp_was_clamped=False,
                  stop_was_clamped=False, timestamp=None):
        """Record one completed trade with all dimensional attributes."""
        cost_pct = cost_bps / 10000.0
        pnl_after_cost = pnl_pct - cost_pct

        entry = {
            "timestamp": (timestamp or datetime.utcnow()).isoformat() + "Z",
            "symbol": symbol,
            "pnl_pct": round(pnl_pct, 6),
            "cost_bps": round(cost_bps, 2),
            "pnl_after_cost": round(pnl_after_cost, 6),
            # Dimensional buckets
            "score_bucket": _bucket_score(entry_score),
            "cost_bucket": _bucket_cost(cost_bps),
            "hold_bucket": _bucket_hold(hold_hours),
            "regime_phase": regime_phase,
            "exit_type": exit_type,
            "clamp_status": (
                "both_clamped" if tp_was_clamped and stop_was_clamped else
                "tp_clamped" if tp_was_clamped else
                "stop_clamped" if stop_was_clamped else
                "unclamped"
            ),
            # Raw values for custom analysis
            "entry_score": round(entry_score, 4),
            "edge_ratio": round(edge_ratio, 2),
            "hold_hours": round(hold_hours, 1),
            "is_win": pnl_after_cost > 0,
        }
        self._trades.append(entry)

        os.makedirs(os.path.dirname(EXPECTANCY_LOG_PATH), exist_ok=True)
        with open(EXPECTANCY_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def by_dimension(self, dimension):
        """
        Compute expectancy broken down by one dimension.

        Args:
            dimension: One of "symbol", "regime_phase", "score_bucket",
                      "cost_bucket", "hold_bucket", "exit_type", "clamp_status".

        Returns:
            Dict of {bucket_value: stats_dict}.
        """
        groups = defaultdict(list)
        for t in self._trades:
            key = t.get(dimension, "unknown")
            groups[key].append(t)

        return {
            bucket: _compute_stats(trades)
            for bucket, trades in sorted(groups.items())
        }

    def full_breakdown(self):
        """Compute expectancy across all dimensions."""
        dimensions = [
            "symbol", "regime_phase", "score_bucket",
            "cost_bucket", "hold_bucket", "exit_type", "clamp_status",
        ]
        return {dim: self.by_dimension(dim) for dim in dimensions}

    def find_profitable_setups(self, min_trades=3):
        """Find setup combinations that consistently make money."""
        profitable = []
        unprofitable = []

        for dim in ["symbol", "regime_phase", "score_bucket", "exit_type"]:
            breakdown = self.by_dimension(dim)
            for bucket, stats in breakdown.items():
                if stats["trades"] < min_trades:
                    continue
                entry = {
                    "dimension": dim,
                    "bucket": bucket,
                    "trades": stats["trades"],
                    "expectancy": stats["expectancy"],
                    "profit_factor": stats["profit_factor"],
                    "hit_rate": stats["hit_rate"],
                }
                if stats["expectancy"] > 0:
                    profitable.append(entry)
                else:
                    unprofitable.append(entry)

        profitable.sort(key=lambda x: x["expectancy"], reverse=True)
        unprofitable.sort(key=lambda x: x["expectancy"])

        return {"profitable": profitable, "unprofitable": unprofitable}

    def format_text(self, breakdown=None):
        """Format full expectancy breakdown as text."""
        if breakdown is None:
            breakdown = self.full_breakdown()

        if not self._trades:
            return "EXPECTANCY TRACKER: No trades recorded.\n"

        lines = []
        lines.append("=" * 90)
        lines.append("EXPECTANCY BY SETUP TYPE")
        lines.append("=" * 90)
        lines.append(f"  Total trades: {len(self._trades)}")
        lines.append("")

        # Overall stats
        overall = _compute_stats(self._trades)
        lines.append("  OVERALL:")
        lines.append(
            f"    Expectancy: {overall['expectancy']:+.4%}  "
            f"Hit rate: {overall['hit_rate']:.0%}  "
            f"PF: {overall['profit_factor']:.2f}  "
            f"Avg W: {overall['avg_winner']:+.4%}  "
            f"Avg L: {overall['avg_loser']:+.4%}"
        )
        lines.append("")

        # Each dimension
        dim_labels = {
            "symbol": "BY SYMBOL",
            "regime_phase": "BY REGIME",
            "score_bucket": "BY SIGNAL STRENGTH",
            "cost_bucket": "BY COST BUCKET",
            "hold_bucket": "BY HOLD TIME",
            "exit_type": "BY EXIT TYPE",
            "clamp_status": "BY CLAMP STATUS",
        }

        for dim, label in dim_labels.items():
            data = breakdown.get(dim, {})
            if not data:
                continue

            lines.append(f"  {label}:")
            lines.append(
                f"    {'BUCKET':15s} {'N':>4s} {'HIT%':>5s} {'EXPECT':>8s} "
                f"{'PF':>5s} {'AVG_W':>7s} {'AVG_L':>7s}"
            )
            lines.append("    " + "-" * 55)

            for bucket in sorted(data.keys()):
                s = data[bucket]
                pf_str = f"{s['profit_factor']:>5.2f}" if s['profit_factor'] != float('inf') else "  inf"
                lines.append(
                    f"    {bucket:15s} {s['trades']:>4d} {s['hit_rate']:>4.0%} "
                    f"{s['expectancy']:>+8.4%} {pf_str} "
                    f"{s['avg_winner']:>+7.4%} {s['avg_loser']:>+7.4%}"
                )
            lines.append("")

        # Profitable / unprofitable setups
        setups = self.find_profitable_setups()
        if setups["profitable"]:
            lines.append("  TOP PROFITABLE SETUPS:")
            for s in setups["profitable"][:5]:
                lines.append(
                    f"    {s['dimension']:15s} = {s['bucket']:15s} "
                    f"expect={s['expectancy']:+.4%} PF={s['profit_factor']:.2f} "
                    f"({s['trades']} trades)"
                )
            lines.append("")

        if setups["unprofitable"]:
            lines.append("  WORST UNPROFITABLE SETUPS:")
            for s in setups["unprofitable"][:5]:
                lines.append(
                    f"    {s['dimension']:15s} = {s['bucket']:15s} "
                    f"expect={s['expectancy']:+.4%} PF={s['profit_factor']:.2f} "
                    f"({s['trades']} trades)"
                )
            lines.append("")

        return "\n".join(lines)


def _compute_stats(trades):
    """Compute standard stats for a group of trades."""
    if not trades:
        return {
            "trades": 0, "hit_rate": 0, "expectancy": 0,
            "avg_winner": 0, "avg_loser": 0, "profit_factor": 0,
        }

    n = len(trades)
    pnls = [t["pnl_after_cost"] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    gross_wins = sum(winners) if winners else 0
    gross_losses = abs(sum(losers)) if losers else 0

    return {
        "trades": n,
        "hit_rate": round(len(winners) / n, 3) if n > 0 else 0,
        "expectancy": round(sum(pnls) / n, 6) if n > 0 else 0,
        "avg_winner": round(gross_wins / len(winners), 6) if winners else 0,
        "avg_loser": round(-gross_losses / len(losers), 6) if losers else 0,
        "profit_factor": (
            round(gross_wins / gross_losses, 2) if gross_losses > 0 else
            (float("inf") if gross_wins > 0 else 0)
        ),
        "total_pnl": round(sum(pnls), 6),
    }


def _bucket_score(score):
    """Bucket entry score into strength categories."""
    if score >= 0.50:
        return "strong"
    elif score >= 0.20:
        return "medium"
    elif score >= 0.10:
        return "weak"
    else:
        return "below_threshold"


def _bucket_cost(cost_bps):
    """Bucket cost into categories."""
    if cost_bps <= 30:
        return "tight"
    elif cost_bps <= 60:
        return "moderate"
    else:
        return "expensive"


def _bucket_hold(hold_hours):
    """Bucket holding time."""
    if hold_hours <= 4:
        return "quick (<4h)"
    elif hold_hours <= 12:
        return "medium (4-12h)"
    elif hold_hours <= 24:
        return "day (12-24h)"
    else:
        return "long (>24h)"


def load_expectancy_history():
    """Load historical expectancy data from JSONL log."""
    tracker = ExpectancyTracker()
    if not os.path.exists(EXPECTANCY_LOG_PATH):
        return tracker

    with open(EXPECTANCY_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tracker._trades.append(json.loads(line))

    return tracker
