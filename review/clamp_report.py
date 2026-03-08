"""
Clamp Hit-Rate Report — tracks how often ATR-scaled exits are clamped.

If clamps dominate (>25-35%), the ATR-scaling is not actually driving exits —
the min/max floors are. That means the exit model needs wider bands or the
universe contains assets whose ATR doesn't fit the clamp range.

Reports:
- Overall TP clamp rate / stop clamp rate
- Per-symbol clamp rates
- Median raw vs clamped TP / stop
- Trend over time (are clamp rates rising?)

Usage:
    from review.clamp_report import ClampTracker, generate_clamp_report

    tracker = ClampTracker()
    tracker.record(symbol, tp_raw, tp_clamped, stop_raw, stop_clamped)
    report = tracker.summarize()
"""

import json
import os
from collections import defaultdict
from datetime import datetime

from core.logging import get_logger

logger = get_logger("review.clamp_report")

CLAMP_LOG_PATH = "reports/clamp_history.jsonl"


class ClampTracker:
    """Accumulates clamp observations across cycles."""

    def __init__(self):
        self._observations = []

    def record(self, symbol, tp_raw, tp_clamped, stop_raw, stop_clamped,
               atr_pct=0.0, timestamp=None):
        """Record one clamp observation (one symbol, one cycle)."""
        obs = {
            "timestamp": (timestamp or datetime.utcnow()).isoformat() + "Z",
            "symbol": symbol,
            "tp_raw": round(tp_raw, 6),
            "tp_clamped": round(tp_clamped, 6),
            "tp_was_clamped": abs(tp_raw - tp_clamped) > 1e-6,
            "stop_raw": round(stop_raw, 6),
            "stop_clamped": round(stop_clamped, 6),
            "stop_was_clamped": abs(stop_raw - stop_clamped) > 1e-6,
            "atr_pct": round(atr_pct, 6),
        }
        self._observations.append(obs)

        # Append to persistent log
        os.makedirs(os.path.dirname(CLAMP_LOG_PATH), exist_ok=True)
        with open(CLAMP_LOG_PATH, "a") as f:
            f.write(json.dumps(obs) + "\n")

    def record_from_diagnostics(self, symbol, diagnostics, timestamp=None):
        """Record from a _score_coins() diagnostics dict."""
        self.record(
            symbol=symbol,
            tp_raw=diagnostics.get("tp_raw", 0),
            tp_clamped=diagnostics.get("tp_clamped", 0),
            stop_raw=diagnostics.get("stop_raw", 0),
            stop_clamped=diagnostics.get("stop_clamped", 0),
            atr_pct=diagnostics.get("atr_pct", 0),
            timestamp=timestamp,
        )

    def summarize(self):
        """
        Compute aggregate clamp statistics.

        Returns:
            Dict with overall and per-symbol clamp rates.
        """
        if not self._observations:
            return {"total_observations": 0}

        total = len(self._observations)
        tp_clamped_count = sum(1 for o in self._observations if o["tp_was_clamped"])
        stop_clamped_count = sum(1 for o in self._observations if o["stop_was_clamped"])

        # Per-symbol breakdown
        by_symbol = defaultdict(lambda: {
            "count": 0, "tp_clamped": 0, "stop_clamped": 0,
            "tp_raw_vals": [], "tp_clamped_vals": [],
            "stop_raw_vals": [], "stop_clamped_vals": [],
        })

        for o in self._observations:
            sym = o["symbol"]
            by_symbol[sym]["count"] += 1
            if o["tp_was_clamped"]:
                by_symbol[sym]["tp_clamped"] += 1
            if o["stop_was_clamped"]:
                by_symbol[sym]["stop_clamped"] += 1
            by_symbol[sym]["tp_raw_vals"].append(o["tp_raw"])
            by_symbol[sym]["tp_clamped_vals"].append(o["tp_clamped"])
            by_symbol[sym]["stop_raw_vals"].append(o["stop_raw"])
            by_symbol[sym]["stop_clamped_vals"].append(o["stop_clamped"])

        symbol_summaries = {}
        for sym, data in by_symbol.items():
            n = data["count"]
            symbol_summaries[sym] = {
                "observations": n,
                "tp_clamp_rate": round(data["tp_clamped"] / n, 3) if n > 0 else 0,
                "stop_clamp_rate": round(data["stop_clamped"] / n, 3) if n > 0 else 0,
                "median_tp_raw": round(_median(data["tp_raw_vals"]), 5),
                "median_tp_clamped": round(_median(data["tp_clamped_vals"]), 5),
                "median_stop_raw": round(_median(data["stop_raw_vals"]), 5),
                "median_stop_clamped": round(_median(data["stop_clamped_vals"]), 5),
            }

        report = {
            "total_observations": total,
            "overall_tp_clamp_rate": round(tp_clamped_count / total, 3),
            "overall_stop_clamp_rate": round(stop_clamped_count / total, 3),
            "tp_clamped_count": tp_clamped_count,
            "stop_clamped_count": stop_clamped_count,
            "median_tp_raw": round(_median([o["tp_raw"] for o in self._observations]), 5),
            "median_tp_clamped": round(_median([o["tp_clamped"] for o in self._observations]), 5),
            "median_stop_raw": round(_median([o["stop_raw"] for o in self._observations]), 5),
            "median_stop_clamped": round(_median([o["stop_clamped"] for o in self._observations]), 5),
            "by_symbol": symbol_summaries,
        }

        return report

    def format_text(self, report=None):
        """Format clamp report as human-readable text."""
        if report is None:
            report = self.summarize()

        if report.get("total_observations", 0) == 0:
            return "CLAMP REPORT: No observations recorded.\n"

        lines = []
        lines.append("=" * 70)
        lines.append("ATR CLAMP HIT-RATE REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Overall
        tp_rate = report["overall_tp_clamp_rate"]
        stop_rate = report["overall_stop_clamp_rate"]
        tp_flag = " *** HIGH" if tp_rate > 0.35 else (" * ELEVATED" if tp_rate > 0.25 else "")
        stop_flag = " *** HIGH" if stop_rate > 0.35 else (" * ELEVATED" if stop_rate > 0.25 else "")

        lines.append(f"  Total observations: {report['total_observations']}")
        lines.append(f"  TP clamp rate:      {tp_rate:.1%} "
                     f"({report['tp_clamped_count']}/{report['total_observations']}){tp_flag}")
        lines.append(f"  Stop clamp rate:    {stop_rate:.1%} "
                     f"({report['stop_clamped_count']}/{report['total_observations']}){stop_flag}")
        lines.append("")
        lines.append(f"  Median TP:   raw={report['median_tp_raw']:.4f}  "
                     f"clamped={report['median_tp_clamped']:.4f}")
        lines.append(f"  Median Stop: raw={report['median_stop_raw']:.4f}  "
                     f"clamped={report['median_stop_clamped']:.4f}")
        lines.append("")

        # Per-symbol table
        lines.append(f"  {'SYMBOL':10s} {'OBS':>4s} {'TP_CL%':>7s} {'ST_CL%':>7s} "
                     f"{'TP_RAW':>8s} {'TP_CL':>8s} {'ST_RAW':>8s} {'ST_CL':>8s}")
        lines.append("  " + "-" * 70)

        for sym in sorted(report["by_symbol"].keys()):
            s = report["by_symbol"][sym]
            lines.append(
                f"  {sym:10s} {s['observations']:>4d} "
                f"{s['tp_clamp_rate']:>6.0%} {s['stop_clamp_rate']:>7.0%} "
                f"{s['median_tp_raw']:>8.4f} {s['median_tp_clamped']:>8.4f} "
                f"{s['median_stop_raw']:>8.4f} {s['median_stop_clamped']:>8.4f}"
            )

        lines.append("")

        # Alert
        if tp_rate > 0.35 or stop_rate > 0.35:
            lines.append(
                "  ALERT: Clamp rate exceeds 35%. The ATR-scaled exit model is "
                "not driving exits — the min/max floors are. Consider widening "
                "clamp bands or reviewing universe composition."
            )
            lines.append("")

        return "\n".join(lines)


def _median(values):
    """Compute median of a list of numbers."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def load_clamp_history():
    """Load historical clamp observations from the JSONL log."""
    tracker = ClampTracker()
    if not os.path.exists(CLAMP_LOG_PATH):
        return tracker

    with open(CLAMP_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                obs = json.loads(line)
                tracker._observations.append(obs)

    return tracker
