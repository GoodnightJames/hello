"""
Sample Cycle Output — generates a realistic demo of the full decision pipeline.

Run this to see what one complete crypto DCA cycle looks like through all
four visibility tools. Uses synthetic data so it works without live APIs.

Usage:
    python -m review.sample_cycle_output
"""

from review.cycle_dashboard import CycleDashboard
from review.clamp_report import ClampTracker
from review.deadzone_report import DeadzoneTracker
from review.universe_report import UniverseTracker


def generate_sample():
    """Generate sample cycle data and print all reports."""

    # Simulated scored output from _score_coins()
    scored = [
        ("BTC/USD", 0.847, "rank=+0.847 edge=0.0081 hurdle=0.0046 ratio=1.8x cost=46bps", {
            "z_12h": 1.12, "z_1d": 0.95, "z_3d": 0.68,
            "z_vol": -0.32, "z_ext": 0.15,
            "momentum": 0.943, "vol_penalty": -0.112, "ext_penalty": 0.038,
            "rank_score": 0.847,
            "mom_strength": 1.85, "expected_edge": 0.00810, "cost_hurdle": 0.00460,
            "edge_ratio": 1.8, "min_edge_ratio": 1.5, "passes_cost_gate": True,
            "continuation_fraction": 0.30,
            "cost_bps": 46, "atr_pct": 0.0146, "ret_1d": 0.028, "vol": 0.0152,
            "tp_raw": 0.0292, "tp_clamped": 0.03, "tp_was_clamped": True,
            "stop_raw": 0.01825, "stop_clamped": 0.03, "stop_was_clamped": True,
        }),
        ("ETH/USD", 0.523, "rank=+0.523 edge=0.0052 hurdle=0.0038 ratio=1.4x cost=38bps COST_FAIL", {
            "z_12h": 0.78, "z_1d": 0.62, "z_3d": 0.41,
            "z_vol": -0.15, "z_ext": -0.08,
            "momentum": 0.612, "vol_penalty": -0.053, "ext_penalty": -0.020,
            "rank_score": 0.523,
            "mom_strength": 1.25, "expected_edge": 0.00520, "cost_hurdle": 0.00380,
            "edge_ratio": 1.4, "min_edge_ratio": 1.5, "passes_cost_gate": False,
            "continuation_fraction": 0.30,
            "cost_bps": 38, "atr_pct": 0.0139, "ret_1d": 0.019, "vol": 0.0148,
            "tp_raw": 0.0278, "tp_clamped": 0.03, "tp_was_clamped": True,
            "stop_raw": 0.01738, "stop_clamped": 0.03, "stop_was_clamped": True,
        }),
        ("SOL/USD", 0.215, "rank=+0.215 edge=0.0035 hurdle=0.0052 ratio=0.7x cost=52bps COST_FAIL", {
            "z_12h": 0.35, "z_1d": 0.22, "z_3d": 0.08,
            "z_vol": 0.45, "z_ext": 0.12,
            "momentum": 0.248, "vol_penalty": 0.158, "ext_penalty": 0.030,
            "rank_score": 0.215,
            "mom_strength": 0.72, "expected_edge": 0.00350, "cost_hurdle": 0.00520,
            "edge_ratio": 0.7, "min_edge_ratio": 1.5, "passes_cost_gate": False,
            "continuation_fraction": 0.30,
            "cost_bps": 52, "atr_pct": 0.0163, "ret_1d": 0.011, "vol": 0.0210,
            "tp_raw": 0.0326, "tp_clamped": 0.0326, "tp_was_clamped": False,
            "stop_raw": 0.02038, "stop_clamped": 0.03, "stop_was_clamped": True,
        }),
        ("AVAX/USD", -0.112, "rank=-0.112 edge=0.0000 hurdle=0.0058 ratio=0.0x cost=58bps COST_FAIL", {
            "z_12h": -0.25, "z_1d": -0.18, "z_3d": -0.52,
            "z_vol": 0.82, "z_ext": -0.35,
            "momentum": -0.294, "vol_penalty": 0.287, "ext_penalty": -0.088,
            "rank_score": -0.112,
            "mom_strength": -0.45, "expected_edge": 0.0, "cost_hurdle": 0.00580,
            "edge_ratio": 0.0, "min_edge_ratio": 1.5, "passes_cost_gate": False,
            "continuation_fraction": 0.30,
            "cost_bps": 58, "atr_pct": 0.0195, "ret_1d": -0.009, "vol": 0.0245,
            "tp_raw": 0.0390, "tp_clamped": 0.0390, "tp_was_clamped": False,
            "stop_raw": 0.02438, "stop_clamped": 0.03, "stop_was_clamped": True,
        }),
        ("LINK/USD", -0.340, "rank=-0.340 edge=0.0000 hurdle=0.0054 ratio=0.0x cost=54bps COST_FAIL", {
            "z_12h": -0.55, "z_1d": -0.42, "z_3d": -0.30,
            "z_vol": 0.65, "z_ext": -0.22,
            "momentum": -0.456, "vol_penalty": 0.228, "ext_penalty": -0.055,
            "rank_score": -0.340,
            "mom_strength": -0.82, "expected_edge": 0.0, "cost_hurdle": 0.00540,
            "edge_ratio": 0.0, "min_edge_ratio": 1.5, "passes_cost_gate": False,
            "continuation_fraction": 0.30,
            "cost_bps": 54, "atr_pct": 0.0180, "ret_1d": -0.015, "vol": 0.0220,
            "tp_raw": 0.0360, "tp_clamped": 0.0360, "tp_was_clamped": False,
            "stop_raw": 0.02250, "stop_clamped": 0.03, "stop_was_clamped": True,
        }),
        ("DOGE/USD", -0.785, "rank=-0.785 edge=0.0000 hurdle=0.0072 ratio=0.0x cost=72bps COST_FAIL", {
            "z_12h": -1.15, "z_1d": -0.92, "z_3d": -0.68,
            "z_vol": 1.35, "z_ext": -0.48,
            "momentum": -0.948, "vol_penalty": 0.473, "ext_penalty": -0.120,
            "rank_score": -0.785,
            "mom_strength": -1.52, "expected_edge": 0.0, "cost_hurdle": 0.00720,
            "edge_ratio": 0.0, "min_edge_ratio": 1.5, "passes_cost_gate": False,
            "continuation_fraction": 0.30,
            "cost_bps": 72, "atr_pct": 0.0320, "ret_1d": -0.032, "vol": 0.0380,
            "tp_raw": 0.0640, "tp_clamped": 0.0640, "tp_was_clamped": False,
            "stop_raw": 0.04000, "stop_clamped": 0.04, "stop_was_clamped": False,
        }),
    ]

    # Simulated context
    signals = [{
        "symbol": "BTC/USD",
        "signal_type": "BUY",
        "score": 0.847,
        "metadata": {"dollar_amount": 2.0},
    }]
    regime = {"trend": "risk_on", "vol": "normal_vol", "phase": "trending"}
    eligible = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "DOGE/USD"]
    removed = [("XRP/USD", ["spread too wide: 35 bps > 30 bps max"])]
    all_symbols = eligible + ["XRP/USD"]
    budget_result = {
        "within_budget": True,
        "position_scale": 0.75,
        "warnings": ["VOLATILITY WARNING: 18.5% approaching limit 25.0%"],
    }

    # ── 1. Dashboard ────────────────────────────────────────────
    dashboard = CycleDashboard()
    dashboard.start_cycle("crypto")
    dashboard.record_cycle_meta(
        regime=regime,
        eligible_universe=eligible,
        budget_result=budget_result,
    )

    for symbol, score, reason, diag in scored:
        is_selected = symbol == "BTC/USD"
        reject = ""
        if not is_selected:
            if score < 0.10:
                reject = "below_score_threshold"
            elif not diag.get("passes_cost_gate"):
                reject = "cost_gate_fail"
            else:
                reject = "not_top_ranked"

        dashboard.record_candidate(
            symbol=symbol,
            diagnostics={**diag, "min_score_threshold": 0.10},
            selected=is_selected,
            reject_reason=reject,
            pre_scale_notional=2.00 if is_selected else None,
            post_scale_notional=1.50 if is_selected else None,
            min_notional_pass=True if is_selected else None,
        )

    dashboard.finalize_cycle()
    print(dashboard.format_cycle_text())

    # ── 2. Clamp report ─────────────────────────────────────────
    clamp = ClampTracker()
    for symbol, _, _, diag in scored:
        clamp.record_from_diagnostics(symbol, diag)
    # Add a second "cycle" of observations for variety
    for symbol, _, _, diag in scored:
        clamp.record_from_diagnostics(symbol, diag)
    print(clamp.format_text())

    # ── 3. Dead-zone report ──────────────────────────────────────
    dz = DeadzoneTracker()
    # Simulate 10 cycles with varying outcomes
    scenarios = [
        (6, 6, 3, 1, 1, 1, 1, 1.0),    # Normal: 1 trade
        (6, 6, 2, 1, 1, 1, 1, 1.0),    # Normal: 1 trade
        (6, 6, 1, 0, 0, 0, 0, 1.0),    # No edge
        (6, 6, 0, 0, 0, 0, 0, 1.0),    # No signal
        (6, 6, 3, 2, 1, 1, 1, 0.75),   # Budget warning
        (6, 6, 2, 1, 0, 0, 0, 0.25),   # Budget blocked
        (6, 6, 3, 1, 1, 0, 0, 0.50),   # Min notional fail
        (6, 5, 2, 1, 1, 1, 1, 1.0),    # Universe removed 1
        (6, 6, 4, 2, 2, 2, 1, 1.0),    # 2 viable, 1 executed
        (6, 6, 0, 0, 0, 0, 0, 1.0),    # No signal
    ]
    for (total, elig, thr, cost, bud, notn, ex, scale) in scenarios:
        dz.record_cycle(
            universe_total=total, universe_eligible=elig,
            passed_score_threshold=thr, passed_cost_gate=cost,
            passed_budget=bud, passed_min_notional=notn,
            executed=ex, budget_scale=scale,
        )
    print(dz.format_text())

    # ── 4. Universe stability ────────────────────────────────────
    uni = UniverseTracker()
    # Simulate 8 cycles of universe filtering
    stable_eligible = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
    for i in range(8):
        if i % 3 == 0:
            # DOGE gets removed every 3rd cycle (flickering)
            eli = stable_eligible + ["LINK/USD"]
            rem = [("DOGE/USD", ["spread too wide: 18 bps > 15 bps max"])]
        else:
            eli = stable_eligible + ["LINK/USD", "DOGE/USD"]
            rem = []

        # XRP always removed
        rem.append(("XRP/USD", ["spread too wide: 35 bps > 30 bps max"]))
        all_s = eli + [s for s, _ in rem]
        uni.record_cycle(eligible=eli, removed=rem, all_symbols=all_s)
    print(uni.format_text())


if __name__ == "__main__":
    generate_sample()
