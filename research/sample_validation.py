"""
Sample Validation Run — generates a realistic demo of the validation report.

Uses synthetic data to show the full output format. Run with:
    python -m research.sample_validation
"""

from research.validation import generate_validation_report


def generate_sample():
    """Generate sample validation data and print report."""

    # Simulated cycle history (from CycleDashboard)
    cycles = []
    regimes = [
        "trending", "trending", "trending", "ranging", "ranging",
        "correction", "correction", "correction", "ranging", "trending",
        "trending", "trending",
    ]

    for i, phase in enumerate(regimes):
        candidates = [
            {
                "symbol": "BTC/USD", "rank_score": 0.85 - i * 0.05,
                "threshold_pass": (0.85 - i * 0.05) >= 0.10,
                "cost_gate_pass": i < 8,
                "selected": i < 7 and phase in ("trending", "ranging"),
                "reject_reason": "" if (i < 7 and phase in ("trending", "ranging"))
                    else "below_threshold" if (0.85 - i * 0.05) < 0.10
                    else "cost_gate_fail" if i >= 8
                    else "not_top_ranked",
            },
            {
                "symbol": "ETH/USD", "rank_score": 0.52 - i * 0.03,
                "threshold_pass": (0.52 - i * 0.03) >= 0.10,
                "cost_gate_pass": i < 6,
                "selected": False,
                "reject_reason": "not_top_ranked" if i < 6 else "cost_gate_fail",
            },
            {
                "symbol": "SOL/USD", "rank_score": 0.21 - i * 0.04,
                "threshold_pass": (0.21 - i * 0.04) >= 0.10,
                "cost_gate_pass": False,
                "selected": False,
                "reject_reason": "cost_gate_fail",
            },
            {
                "symbol": "DOGE/USD", "rank_score": -0.30 - i * 0.02,
                "threshold_pass": False,
                "cost_gate_pass": False,
                "selected": False,
                "reject_reason": "below_threshold",
            },
        ]

        cycles.append({
            "meta": {
                "timestamp": f"2026-03-0{i+1}T08:00:00Z",
                "regime": {"trend": "risk_on" if phase in ("trending", "ranging") else "risk_off",
                           "vol": "normal_vol", "phase": phase},
            },
            "candidates": candidates,
        })

    # Simulated trades
    trades = [
        # Trending — good trades
        {"symbol": "BTC/USD", "pnl_after_cost": 0.018, "regime_phase": "trending",
         "exit_type": "take_profit", "health_score": 0.85,
         "regime_thresholds_active": True, "time_decay_active": True,
         "momentum_collapse_active": True, "health_layer_active": True},
        {"symbol": "BTC/USD", "pnl_after_cost": 0.012, "regime_phase": "trending",
         "exit_type": "trailing_stop", "health_score": 0.80,
         "regime_thresholds_active": True, "time_decay_active": True,
         "momentum_collapse_active": True, "health_layer_active": True},
        {"symbol": "ETH/USD", "pnl_after_cost": -0.008, "regime_phase": "trending",
         "exit_type": "hard_stop", "health_score": 0.75,
         "regime_thresholds_active": True, "time_decay_active": True,
         "momentum_collapse_active": True, "health_layer_active": True},
        {"symbol": "BTC/USD", "pnl_after_cost": 0.025, "regime_phase": "trending",
         "exit_type": "take_profit", "health_score": 0.90,
         "regime_thresholds_active": True, "time_decay_active": True,
         "momentum_collapse_active": True, "health_layer_active": True},

        # Ranging — mixed
        {"symbol": "BTC/USD", "pnl_after_cost": 0.005, "regime_phase": "ranging",
         "exit_type": "take_profit", "health_score": 0.72,
         "regime_thresholds_active": True, "time_decay_active": True,
         "momentum_collapse_active": True, "health_layer_active": True},
        {"symbol": "ETH/USD", "pnl_after_cost": -0.012, "regime_phase": "ranging",
         "exit_type": "time_decay", "health_score": 0.65,
         "regime_thresholds_active": True, "time_decay_active": True,
         "momentum_collapse_active": True, "health_layer_active": True},

        # Correction — mostly losses
        {"symbol": "BTC/USD", "pnl_after_cost": -0.015, "regime_phase": "correction",
         "exit_type": "hard_stop", "health_score": 0.45,
         "regime_thresholds_active": True, "time_decay_active": True,
         "momentum_collapse_active": True, "health_layer_active": True},
        {"symbol": "SOL/USD", "pnl_after_cost": -0.022, "regime_phase": "correction",
         "exit_type": "momentum_collapse", "health_score": 0.35,
         "regime_thresholds_active": True, "time_decay_active": True,
         "momentum_collapse_active": True, "health_layer_active": True},

        # Recovery trades (post-correction)
        {"symbol": "BTC/USD", "pnl_after_cost": 0.020, "regime_phase": "trending",
         "exit_type": "take_profit", "health_score": 0.60,
         "regime_thresholds_active": True, "time_decay_active": True,
         "momentum_collapse_active": True, "health_layer_active": True},
        {"symbol": "ETH/USD", "pnl_after_cost": 0.015, "regime_phase": "trending",
         "exit_type": "trailing_stop", "health_score": 0.68,
         "regime_thresholds_active": True, "time_decay_active": True,
         "momentum_collapse_active": True, "health_layer_active": True},
    ]

    # Health history
    health_history = [
        {"score": 0.85}, {"score": 0.80}, {"score": 0.72},
        {"score": 0.65}, {"score": 0.45}, {"score": 0.35},
        {"score": 0.42}, {"score": 0.55}, {"score": 0.60},
        {"score": 0.68}, {"score": 0.75}, {"score": 0.82},
    ]

    # Symbol classification
    symbol_classification = {
        "BTC/USD": {
            "classification": "core_earner",
            "stats": {"expectancy": 0.0090, "profit_factor": 2.15, "hit_rate": 0.65,
                      "trades": 15, "avg_winner": 0.018, "avg_loser": -0.012},
            "regime_breakdown": {
                "trending": {"expectancy": 0.015, "trades": 8},
                "ranging": {"expectancy": 0.003, "trades": 4},
                "correction": {"expectancy": -0.010, "trades": 3},
            },
        },
        "ETH/USD": {
            "classification": "conditional_earner",
            "stats": {"expectancy": 0.0015, "profit_factor": 1.12, "hit_rate": 0.50,
                      "trades": 10, "avg_winner": 0.014, "avg_loser": -0.011},
            "regime_breakdown": {
                "trending": {"expectancy": 0.008, "trades": 5},
                "ranging": {"expectancy": -0.005, "trades": 3},
                "correction": {"expectancy": -0.012, "trades": 2},
            },
        },
        "SOL/USD": {
            "classification": "dead_weight",
            "stats": {"expectancy": -0.0045, "profit_factor": 0.72, "hit_rate": 0.38,
                      "trades": 8, "avg_winner": 0.012, "avg_loser": -0.014},
            "regime_breakdown": {
                "trending": {"expectancy": -0.001, "trades": 4},
                "ranging": {"expectancy": -0.008, "trades": 3},
                "correction": {"expectancy": -0.015, "trades": 1},
            },
        },
        "DOGE/USD": {
            "classification": "dead_weight",
            "stats": {"expectancy": -0.0082, "profit_factor": 0.55, "hit_rate": 0.30,
                      "trades": 6, "avg_winner": 0.010, "avg_loser": -0.015},
            "regime_breakdown": {
                "trending": {"expectancy": -0.003, "trades": 3},
                "ranging": {"expectancy": -0.012, "trades": 2},
                "correction": {"expectancy": -0.018, "trades": 1},
            },
        },
        "AVAX/USD": {
            "classification": "insufficient_data",
            "stats": {"expectancy": 0.0, "profit_factor": 0, "hit_rate": 0,
                      "trades": 2},
            "regime_breakdown": {},
        },
    }

    _, text = generate_validation_report(
        trades=trades,
        cycles=cycles,
        health_history=health_history,
        symbol_classification=symbol_classification,
    )

    print(text)


if __name__ == "__main__":
    generate_sample()
