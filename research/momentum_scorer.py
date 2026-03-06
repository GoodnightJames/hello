"""
Dual Momentum Scorer — Antonacci (2014) implementation.

Two momentum components:
1. Absolute momentum: Is the asset's 12-month return above T-bills (SHY)?
   If no → hold cash proxy (SHY).
2. Relative momentum: Among risk assets with positive absolute momentum,
   which has the highest 12-month return? Hold that one.

This is a low-turnover strategy: 1-2 trades per month average,
holding periods of 1-6 months.
"""

import pandas as pd
from core.logging import get_logger

logger = get_logger("research.momentum_scorer")


def compute_absolute_momentum(returns_12m, benchmark_symbol="SHY"):
    """
    Check if each asset's 12-month return exceeds the benchmark (T-bills).

    Args:
        returns_12m: DataFrame of 12-month returns (latest row = most recent).
        benchmark_symbol: T-bill proxy symbol.

    Returns:
        Dict of {symbol: bool} — True if absolute momentum is positive.
    """
    if returns_12m.empty:
        logger.warning("No 12-month return data available")
        return {}

    latest = returns_12m.iloc[-1]

    if benchmark_symbol not in latest.index:
        logger.error(
            f"Benchmark {benchmark_symbol} not in return data",
            extra={"extra_data": {"available": list(latest.index)}},
        )
        return {}

    benchmark_return = latest[benchmark_symbol]
    results = {}

    for symbol in latest.index:
        if symbol == benchmark_symbol:
            continue
        has_abs_momentum = latest[symbol] > benchmark_return
        results[symbol] = has_abs_momentum

    logger.info(
        "Absolute momentum computed",
        extra={
            "extra_data": {
                "benchmark_return": round(float(benchmark_return), 4),
                "results": {s: v for s, v in results.items()},
            }
        },
    )
    return results


def compute_relative_momentum(returns_12m, symbols):
    """
    Rank assets by 12-month return (descending).

    Args:
        returns_12m: DataFrame of 12-month returns.
        symbols: List of symbols to rank.

    Returns:
        List of (symbol, return) tuples sorted by return descending.
    """
    if returns_12m.empty:
        logger.warning("No 12-month return data for relative ranking")
        return []

    latest = returns_12m.iloc[-1]
    available = [s for s in symbols if s in latest.index and pd.notna(latest[s])]

    ranked = sorted(available, key=lambda s: latest[s], reverse=True)
    result = [(s, float(latest[s])) for s in ranked]

    logger.info(
        "Relative momentum ranking",
        extra={"extra_data": {"ranking": result}},
    )
    return result


def score_dual_momentum(features, strategy_config):
    """
    Run the full dual momentum scoring pipeline.

    Logic:
    1. Compute 12-month returns for risk assets and benchmark.
    2. Check absolute momentum (return > SHY return).
    3. Rank risk assets by relative momentum.
    4. Select the top-ranked asset with positive absolute momentum.
    5. If no risk asset has positive absolute momentum → signal HOLD (stay in SHY/cash).

    Args:
        features: Dict from feature_store.build_features().
        strategy_config: Dict loaded from momentum_v1.yaml.

    Returns:
        List of signal dicts:
        [
            {
                "symbol": str,
                "signal_type": "BUY" | "SELL" | "HOLD",
                "score": float (12m return),
                "metadata": {
                    "absolute_momentum": bool,
                    "relative_rank": int,
                    "benchmark_return": float,
                    "asset_return": float
                }
            }
        ]
    """
    logger.info("Running dual momentum scoring")

    returns = features.get("returns", {})
    returns_12m = returns.get("12m", pd.DataFrame())

    if returns_12m.empty:
        logger.warning("No 12-month return data — cannot score")
        return []

    # Get strategy instruments
    instruments = strategy_config.get("instruments", {})
    risk_assets = instruments.get("risk_assets", [])
    safe_assets = instruments.get("safe_assets", [])
    benchmark = strategy_config.get("signals", {}).get("benchmark", "SHY")

    all_symbols = risk_assets + safe_assets
    available = [s for s in all_symbols if s in returns_12m.columns]

    if not available:
        logger.warning("No strategy instruments found in data")
        return []

    # Step 1: Absolute momentum
    abs_momentum = compute_absolute_momentum(returns_12m, benchmark)

    # Step 2: Relative momentum among risk assets
    risk_available = [s for s in risk_assets if s in returns_12m.columns]
    relative_ranking = compute_relative_momentum(returns_12m, risk_available)

    # Step 3: Find the best risk asset with positive absolute momentum
    latest_returns = returns_12m.iloc[-1]
    benchmark_return = float(latest_returns.get(benchmark, 0))

    signals = []
    selected_asset = None

    for rank, (symbol, ret) in enumerate(relative_ranking):
        if abs_momentum.get(symbol, False):
            # This asset has positive absolute momentum and is top-ranked
            if selected_asset is None:
                selected_asset = symbol
                signals.append({
                    "symbol": symbol,
                    "signal_type": "BUY",
                    "score": ret,
                    "metadata": {
                        "absolute_momentum": True,
                        "relative_rank": rank + 1,
                        "benchmark_return": benchmark_return,
                        "asset_return": ret,
                    },
                })
            else:
                # Not the top-ranked — signal HOLD (don't buy but don't sell if held)
                signals.append({
                    "symbol": symbol,
                    "signal_type": "HOLD",
                    "score": ret,
                    "metadata": {
                        "absolute_momentum": True,
                        "relative_rank": rank + 1,
                        "benchmark_return": benchmark_return,
                        "asset_return": ret,
                    },
                })
        else:
            # No absolute momentum — should not be held
            signals.append({
                "symbol": symbol,
                "signal_type": "SELL",
                "score": ret,
                "metadata": {
                    "absolute_momentum": False,
                    "relative_rank": rank + 1,
                    "benchmark_return": benchmark_return,
                    "asset_return": ret,
                },
            })

    # If no risk asset has positive absolute momentum → rotate to safe asset
    if selected_asset is None:
        # Signal BUY on the top safe asset (AGG or SHY)
        for safe in safe_assets:
            if safe in returns_12m.columns and safe != benchmark:
                safe_return = float(latest_returns.get(safe, 0))
                signals.append({
                    "symbol": safe,
                    "signal_type": "BUY",
                    "score": safe_return,
                    "metadata": {
                        "absolute_momentum": False,
                        "relative_rank": 0,
                        "benchmark_return": benchmark_return,
                        "asset_return": safe_return,
                        "reason": "No risk asset has positive absolute momentum — rotating to bonds",
                    },
                })
                break

    logger.info(
        "Dual momentum scoring complete",
        extra={
            "extra_data": {
                "selected_asset": selected_asset,
                "signal_count": len(signals),
                "signals": [
                    {"symbol": s["symbol"], "type": s["signal_type"], "score": round(s["score"], 4)}
                    for s in signals
                ],
            }
        },
    )
    return signals
