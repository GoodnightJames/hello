"""
Cycle Instrumentation — wires diagnostic trackers into the trading cycle.

This module provides a single function that instruments a crypto DCA cycle
with all four visibility tools:
1. CycleDashboard (full per-symbol decision table)
2. ClampTracker (ATR exit clamp rates)
3. DeadzoneTracker (signal funnel attrition)
4. UniverseTracker (listing rule stability)

The strategy and risk modules remain clean — instrumentation is
applied at the orchestration layer.

Usage:
    from review.cycle_instrumentation import instrument_crypto_cycle

    # After scoring and signal generation:
    instrument_crypto_cycle(
        scored=scored,             # from _score_coins()
        signals=signals,           # from generate_signals()
        eligible_symbols=eligible, # from universe filter
        removed_symbols=removed,   # from universe filter
        all_symbols=all_symbols,
        budget_result=budget_result,
        regime=regime,
        strategy=strategy,
    )
"""

from core.logging import get_logger
from review.cycle_dashboard import CycleDashboard
from review.clamp_report import ClampTracker
from review.deadzone_report import DeadzoneTracker
from review.universe_report import UniverseTracker

logger = get_logger("review.cycle_instrumentation")

# Module-level singleton trackers (persist across cycles within a process)
_dashboard = CycleDashboard()
_clamp_tracker = ClampTracker()
_deadzone_tracker = DeadzoneTracker()
_universe_tracker = UniverseTracker()


def get_dashboard():
    """Get the singleton CycleDashboard instance."""
    return _dashboard


def get_clamp_tracker():
    """Get the singleton ClampTracker instance."""
    return _clamp_tracker


def get_deadzone_tracker():
    """Get the singleton DeadzoneTracker instance."""
    return _deadzone_tracker


def get_universe_tracker():
    """Get the singleton UniverseTracker instance."""
    return _universe_tracker


def instrument_crypto_cycle(
    scored,
    signals,
    eligible_symbols=None,
    removed_symbols=None,
    all_symbols=None,
    budget_result=None,
    regime=None,
    pre_scale_notional=None,
    post_scale_notional=None,
    min_notional_pass=None,
    min_score_threshold=0.10,
):
    """
    Instrument one complete crypto DCA cycle.

    Populates all four trackers with cycle data.

    Args:
        scored: List of (symbol, score, reason, diagnostics) from _score_coins().
        signals: List of signal dicts from generate_signals() (may be empty).
        eligible_symbols: List of symbols that passed universe filter.
        removed_symbols: List of (symbol, [reasons]) for rejected symbols.
        all_symbols: Full configured universe.
        budget_result: Dict from check_sleeve_risk_budget().
        regime: Dict from tag_current_regime().
        pre_scale_notional: Dollar amount before budget scaling.
        post_scale_notional: Dollar amount after budget scaling.
        min_notional_pass: Whether post-scale clears minimum.
        min_score_threshold: Score threshold for ranking gate.
    """
    selected_symbols = {s["symbol"] for s in signals} if signals else set()

    # 1. Dashboard
    _dashboard.start_cycle(cycle_type="crypto")
    _dashboard.record_cycle_meta(
        regime=regime,
        eligible_universe=eligible_symbols,
        budget_result=budget_result,
    )

    for symbol, score, reason, diag in scored:
        is_selected = symbol in selected_symbols

        # Determine reject reason
        reject = ""
        if not is_selected:
            if score < min_score_threshold:
                reject = "below_score_threshold"
            elif not diag.get("passes_cost_gate", False):
                reject = "cost_gate_fail"
            elif budget_result and not budget_result.get("within_budget", True):
                reject = "budget_blocked"
            elif min_notional_pass is False:
                reject = "below_min_notional"
            else:
                reject = "not_top_ranked"

        _dashboard.record_candidate(
            symbol=symbol,
            diagnostics={**diag, "min_score_threshold": min_score_threshold},
            selected=is_selected,
            reject_reason=reject,
            pre_scale_notional=pre_scale_notional if is_selected else None,
            post_scale_notional=post_scale_notional if is_selected else None,
            min_notional_pass=min_notional_pass if is_selected else None,
        )

    cycle_data = _dashboard.finalize_cycle()

    # 2. Clamp tracker
    for symbol, score, reason, diag in scored:
        if diag.get("atr_pct", 0) > 0:
            _clamp_tracker.record_from_diagnostics(symbol, diag)

    # 3. Dead-zone tracker
    passed_threshold = sum(
        1 for _, s, _, d in scored if s >= min_score_threshold
    )
    passed_cost = sum(
        1 for _, s, _, d in scored
        if s >= min_score_threshold and d.get("passes_cost_gate", False)
    )
    budget_ok = 1 if (not budget_result or budget_result.get("within_budget", True)) else 0
    notional_ok = 1 if min_notional_pass is not False else 0

    blocked_reasons = {}
    if passed_threshold == 0:
        blocked_reasons["below_score_threshold"] = len(scored)
    if passed_threshold > 0 and passed_cost == 0:
        blocked_reasons["cost_gate_fail"] = passed_threshold
    if budget_result and not budget_result.get("within_budget", True):
        for w in budget_result.get("warnings", []):
            blocked_reasons[w] = blocked_reasons.get(w, 0) + 1
    if min_notional_pass is False:
        blocked_reasons["below_min_notional"] = 1

    _deadzone_tracker.record_cycle(
        cycle_type="crypto",
        universe_total=len(all_symbols) if all_symbols else len(scored),
        universe_eligible=len(eligible_symbols) if eligible_symbols else len(scored),
        passed_score_threshold=passed_threshold,
        passed_cost_gate=passed_cost,
        passed_budget=passed_cost if budget_ok else 0,
        passed_min_notional=passed_cost if (budget_ok and notional_ok) else 0,
        executed=len(signals),
        budget_scale=budget_result.get("position_scale", 1.0) if budget_result else 1.0,
        blocked_reasons=blocked_reasons,
    )

    # 4. Universe tracker
    if eligible_symbols is not None and removed_symbols is not None:
        _universe_tracker.record_cycle(
            eligible=eligible_symbols,
            removed=removed_symbols,
            all_symbols=all_symbols,
        )

    logger.info(
        "Cycle instrumented",
        extra={"extra_data": {
            "candidates": len(scored),
            "selected": len(selected_symbols),
            "passed_threshold": passed_threshold,
            "passed_cost": passed_cost,
        }},
    )

    return cycle_data


def generate_visibility_report():
    """
    Generate a combined visibility report from all trackers.

    Returns:
        Multi-line string with all four reports concatenated.
    """
    sections = []

    # Dashboard: show latest cycle
    latest = _dashboard.get_latest_cycle()
    if latest.get("candidates"):
        sections.append(_dashboard.format_cycle_text(latest))

    # Clamp report
    clamp_summary = _clamp_tracker.summarize()
    if clamp_summary.get("total_observations", 0) > 0:
        sections.append(_clamp_tracker.format_text(clamp_summary))

    # Dead-zone report
    dz_summary = _deadzone_tracker.summarize()
    if dz_summary.get("total_cycles", 0) > 0:
        sections.append(_deadzone_tracker.format_text(dz_summary))

    # Universe stability report
    uni_summary = _universe_tracker.summarize()
    if uni_summary.get("total_cycles", 0) > 0:
        sections.append(_universe_tracker.format_text(uni_summary))

    if not sections:
        return "No visibility data collected yet.\n"

    return "\n\n".join(sections)
