"""
Cycle Dashboard — full decision pipeline visibility per cycle.

Shows every decision gate for every candidate in a single table.
One row per symbol per cycle. Columns trace the entire pipeline:

  timestamp | regime | symbol | eligible | z_composite | threshold_pass |
  atr_pct | mom_strength | expected_edge | cost_hurdle | edge_ratio |
  cost_gate_pass | selected | reject_reason | budget_scale |
  pre_scale_notional | post_scale_notional | min_notional_pass

This is the primary diagnostic artifact for auditing whether the
strategy is behaving as designed.

Usage:
    from review.cycle_dashboard import CycleDashboard

    dash = CycleDashboard()
    dash.record_candidate(...)   # called per symbol during scoring
    dash.record_cycle_meta(...)  # called once per cycle
    dash.finalize_cycle()        # writes to disk/log
    dash.get_latest_cycle()      # returns the most recent cycle data
"""

import json
import os
from datetime import datetime

from core.logging import get_logger

logger = get_logger("review.cycle_dashboard")

DASHBOARD_DIR = "reports/dashboards"


class CycleDashboard:
    """Collects per-symbol decision data and renders a full cycle view."""

    def __init__(self):
        self._cycle_id = None
        self._cycle_meta = {}
        self._candidates = []

    def start_cycle(self, cycle_type="crypto"):
        """Begin a new cycle. Resets all candidate data."""
        self._cycle_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._cycle_meta = {
            "cycle_id": self._cycle_id,
            "cycle_type": cycle_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self._candidates = []

    def record_cycle_meta(self, regime=None, eligible_universe=None,
                          budget_result=None):
        """Record cycle-level context (regime, universe, budget state)."""
        if regime:
            self._cycle_meta["regime"] = regime
        if eligible_universe is not None:
            self._cycle_meta["eligible_universe"] = eligible_universe
            self._cycle_meta["eligible_count"] = len(eligible_universe)
        if budget_result:
            self._cycle_meta["budget_scale"] = budget_result.get("position_scale", 1.0)
            self._cycle_meta["budget_within"] = budget_result.get("within_budget", True)
            self._cycle_meta["budget_warnings"] = budget_result.get("warnings", [])

    def record_candidate(self, symbol, diagnostics, selected=False,
                         reject_reason=None, pre_scale_notional=None,
                         post_scale_notional=None, min_notional_pass=None):
        """
        Record one candidate row — called per symbol during scoring.

        Args:
            symbol: Ticker symbol.
            diagnostics: Dict from _score_coins() with z-scores, gates, etc.
            selected: Whether this symbol was actually chosen for trading.
            reject_reason: Why it was not selected (if applicable).
            pre_scale_notional: Dollar amount before budget scaling.
            post_scale_notional: Dollar amount after budget scaling.
            min_notional_pass: Whether post-scale amount clears minimum.
        """
        row = {
            "symbol": symbol,
            "z_12h": diagnostics.get("z_12h", 0),
            "z_1d": diagnostics.get("z_1d", 0),
            "z_3d": diagnostics.get("z_3d", 0),
            "z_vol": diagnostics.get("z_vol", 0),
            "z_ext": diagnostics.get("z_ext", 0),
            "momentum": diagnostics.get("momentum", 0),
            "vol_penalty": diagnostics.get("vol_penalty", 0),
            "ext_penalty": diagnostics.get("ext_penalty", 0),
            "rank_score": diagnostics.get("rank_score", 0),
            "threshold_pass": diagnostics.get("rank_score", 0) >= diagnostics.get(
                "min_score_threshold", 0.10
            ) if "rank_score" in diagnostics else None,
            "atr_pct": diagnostics.get("atr_pct", 0),
            "mom_strength": diagnostics.get("mom_strength", 0),
            "expected_edge": diagnostics.get("expected_edge", 0),
            "cost_hurdle": diagnostics.get("cost_hurdle", 0),
            "edge_ratio": diagnostics.get("edge_ratio", 0),
            "min_edge_ratio": diagnostics.get("min_edge_ratio", 1.5),
            "cost_gate_pass": diagnostics.get("passes_cost_gate", False),
            "cost_bps": diagnostics.get("cost_bps", 0),
            "ret_1d": diagnostics.get("ret_1d", 0),
            "vol": diagnostics.get("vol", 0),
            "continuation_fraction": diagnostics.get("continuation_fraction", 0.30),
            # Exit clamp diagnostics
            "tp_raw": diagnostics.get("tp_raw", 0),
            "tp_clamped": diagnostics.get("tp_clamped", 0),
            "tp_was_clamped": diagnostics.get("tp_was_clamped", False),
            "stop_raw": diagnostics.get("stop_raw", 0),
            "stop_clamped": diagnostics.get("stop_clamped", 0),
            "stop_was_clamped": diagnostics.get("stop_was_clamped", False),
            # Selection outcome
            "selected": selected,
            "reject_reason": reject_reason or "",
            "pre_scale_notional": pre_scale_notional,
            "post_scale_notional": post_scale_notional,
            "min_notional_pass": min_notional_pass,
        }
        self._candidates.append(row)

    def finalize_cycle(self):
        """
        Write the completed cycle to disk and log a summary.

        Saves JSON to reports/dashboards/ and logs a compact table.
        """
        if not self._cycle_id:
            return

        cycle_data = {
            "meta": self._cycle_meta,
            "candidates": self._candidates,
        }

        # Save JSON
        os.makedirs(DASHBOARD_DIR, exist_ok=True)
        cycle_type = self._cycle_meta.get("cycle_type", "crypto")
        path = os.path.join(
            DASHBOARD_DIR,
            f"cycle_{cycle_type}_{self._cycle_id}.json",
        )
        with open(path, "w") as f:
            json.dump(cycle_data, f, indent=2, default=str)

        # Log compact summary
        logger.info(
            "Cycle dashboard saved",
            extra={"extra_data": {
                "cycle_id": self._cycle_id,
                "path": path,
                "candidates": len(self._candidates),
                "selected": sum(1 for c in self._candidates if c["selected"]),
            }},
        )

        # Log the decision table
        self._log_table()

        return cycle_data

    def _log_table(self):
        """Log a compact text table of the cycle for terminal/log review."""
        if not self._candidates:
            return

        header = (
            f"{'SYM':10s} {'RANK':>6s} {'THR':>3s} {'MOM_S':>6s} "
            f"{'EDGE':>7s} {'HRDL':>7s} {'RATIO':>5s} {'COST':>3s} "
            f"{'ATR%':>5s} {'TP_C':>3s} {'ST_C':>3s} {'SEL':>3s} {'REASON'}"
        )
        logger.info(f"CYCLE DASHBOARD [{self._cycle_id}]")
        logger.info(header)
        logger.info("-" * len(header))

        for c in sorted(self._candidates, key=lambda x: x["rank_score"], reverse=True):
            thr = "Y" if c.get("threshold_pass") else "N"
            cst = "Y" if c.get("cost_gate_pass") else "N"
            sel = ">>>" if c.get("selected") else ""
            tp_c = "C" if c.get("tp_was_clamped") else ""
            st_c = "C" if c.get("stop_was_clamped") else ""
            reason = c.get("reject_reason", "")[:30]

            line = (
                f"{c['symbol']:10s} {c['rank_score']:>+6.3f} {thr:>3s} "
                f"{c['mom_strength']:>6.2f} {c['expected_edge']:>7.4f} "
                f"{c['cost_hurdle']:>7.4f} {c['edge_ratio']:>5.1f} {cst:>3s} "
                f"{c['atr_pct']:>5.3f} {tp_c:>3s} {st_c:>3s} {sel:>3s} {reason}"
            )
            logger.info(line)

    def get_latest_cycle(self):
        """Return the most recent cycle data (in-memory)."""
        return {
            "meta": self._cycle_meta,
            "candidates": self._candidates,
        }

    def format_cycle_text(self, cycle_data=None):
        """
        Format a cycle as human-readable text for report inclusion.

        Args:
            cycle_data: Dict from finalize_cycle() or get_latest_cycle().
                       Uses current in-memory data if None.

        Returns:
            Multi-line string.
        """
        if cycle_data is None:
            cycle_data = self.get_latest_cycle()

        meta = cycle_data.get("meta", {})
        candidates = cycle_data.get("candidates", [])

        lines = []
        lines.append("=" * 90)
        lines.append(f"CYCLE DASHBOARD — {meta.get('cycle_type', '?')} "
                      f"@ {meta.get('timestamp', '?')}")
        lines.append("=" * 90)

        # Regime
        regime = meta.get("regime", {})
        if regime:
            lines.append(
                f"  Regime: trend={regime.get('trend', '?')} "
                f"vol={regime.get('vol', '?')} "
                f"phase={regime.get('phase', '?')}"
            )

        # Universe
        lines.append(
            f"  Universe: {meta.get('eligible_count', '?')} eligible symbols"
        )

        # Budget
        budget_scale = meta.get("budget_scale", 1.0)
        lines.append(f"  Budget scale: {budget_scale:.2f}")
        for w in meta.get("budget_warnings", []):
            lines.append(f"    WARNING: {w}")

        lines.append("")

        # Candidate table
        lines.append(
            f"  {'SYMBOL':10s} {'RANK':>7s} {'THR':>4s} {'MOM_S':>6s} "
            f"{'EDGE':>8s} {'HURDLE':>8s} {'RATIO':>6s} {'COST':>4s} "
            f"{'ATR%':>6s} {'TP_CL':>5s} {'ST_CL':>5s} {'SEL':>4s} {'REASON'}"
        )
        lines.append("  " + "-" * 100)

        for c in sorted(candidates, key=lambda x: x["rank_score"], reverse=True):
            thr = "PASS" if c.get("threshold_pass") else "FAIL"
            cst = "PASS" if c.get("cost_gate_pass") else "FAIL"
            sel = " >>>" if c.get("selected") else ""
            tp_cl = "CLAMP" if c.get("tp_was_clamped") else ""
            st_cl = "CLAMP" if c.get("stop_was_clamped") else ""
            reason = c.get("reject_reason", "")[:35]

            lines.append(
                f"  {c['symbol']:10s} {c['rank_score']:>+7.3f} {thr:>4s} "
                f"{c['mom_strength']:>6.2f} {c['expected_edge']:>8.5f} "
                f"{c['cost_hurdle']:>8.5f} {c['edge_ratio']:>6.1f} {cst:>4s} "
                f"{c['atr_pct']:>6.3f} {tp_cl:>5s} {st_cl:>5s} {sel:>4s} {reason}"
            )

        # Sizing detail for selected
        selected = [c for c in candidates if c.get("selected")]
        if selected:
            lines.append("")
            lines.append("  SELECTED SIZING:")
            for c in selected:
                pre = c.get("pre_scale_notional")
                post = c.get("post_scale_notional")
                mn = c.get("min_notional_pass")
                pre_str = f"${pre:.2f}" if pre is not None else "?"
                post_str = f"${post:.2f}" if post is not None else "?"
                mn_str = "PASS" if mn else ("FAIL" if mn is not None else "?")
                lines.append(
                    f"    {c['symbol']:10s} pre={pre_str} → "
                    f"post={post_str} (budget×{budget_scale:.2f}) "
                    f"min_notional={mn_str}"
                )

        lines.append("")
        return "\n".join(lines)
