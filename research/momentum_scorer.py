"""
Dual Momentum Scorer — Antonacci (2014) with alpha optimizations.

Two momentum components:
1. Absolute momentum: Is the asset's 12-month return above T-bills (SHY)?
   If no → hold cash proxy (SHY).
2. Relative momentum: Among risk assets with positive absolute momentum,
   which has the highest 12-month return? Hold that one.

v3.0 optimizations:
- Multi-timeframe confirmation (12m + 6m alignment)
- Minimum edge threshold (skip marginal signals)
- RSI(2) overbought filter (don't buy at the top)
- Composite signal strength score (for position sizing)
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


def check_multi_timeframe(features, symbol, benchmark, config):
    """
    Check if shorter-term returns confirm the 12-month signal.

    Prevents buying into dying trends: a stock can have +20% 12m return
    but be down -10% over 6 months — the trend is reversing.

    Returns:
        (confirmed: bool, details: dict)
    """
    mt_config = config.get("signals", {}).get("multi_timeframe", {})
    if not mt_config.get("enabled", False):
        return True, {"reason": "multi_timeframe disabled"}

    returns = features.get("returns", {})

    details = {}
    confirmed = True

    # 6-month confirmation
    if mt_config.get("require_6m_positive", True):
        returns_6m = returns.get("6m", pd.DataFrame())
        if not returns_6m.empty and symbol in returns_6m.columns and benchmark in returns_6m.columns:
            latest_6m = returns_6m.iloc[-1]
            sym_6m = float(latest_6m[symbol])
            bench_6m = float(latest_6m[benchmark])
            details["return_6m"] = round(sym_6m, 4)
            details["benchmark_6m"] = round(bench_6m, 4)
            if sym_6m <= bench_6m:
                confirmed = False
                details["blocked_by"] = "6m return below benchmark"

    # 3-month confirmation (optional, off by default)
    if mt_config.get("require_3m_positive", False):
        returns_3m = returns.get("3m", pd.DataFrame())
        if not returns_3m.empty and symbol in returns_3m.columns and benchmark in returns_3m.columns:
            latest_3m = returns_3m.iloc[-1]
            sym_3m = float(latest_3m[symbol])
            bench_3m = float(latest_3m[benchmark])
            details["return_3m"] = round(sym_3m, 4)
            details["benchmark_3m"] = round(bench_3m, 4)
            if sym_3m <= bench_3m:
                confirmed = False
                details["blocked_by"] = "3m return below benchmark"

    if confirmed:
        logger.info(f"Multi-timeframe confirmed for {symbol}", extra={"extra_data": details})
    else:
        logger.info(f"Multi-timeframe REJECTED for {symbol}", extra={"extra_data": details})

    return confirmed, details


def check_minimum_edge(return_12m, benchmark_return, config):
    """
    Check if excess return over benchmark meets minimum threshold.

    A 0.5% edge over T-bills isn't worth trading — slippage and spreads
    will eat the alpha. Require a meaningful edge before acting.

    Returns:
        (has_edge: bool, excess_return: float)
    """
    min_edge = config.get("signals", {}).get("min_edge_pct", 0.02)
    excess = return_12m - benchmark_return
    has_edge = excess >= min_edge
    if not has_edge:
        logger.info(
            f"Minimum edge not met: excess {excess:.4f} < threshold {min_edge}",
        )
    return has_edge, float(excess)


def check_rsi_filter(features, symbol, config):
    """
    Check RSI(2) overbought/oversold conditions.

    RSI(2) > 90: asset is at a short-term extreme — wait for pullback.
    RSI(2) < 10: asset is oversold — boost signal strength.

    Returns:
        (allowed: bool, rsi_value: float or None, strength_multiplier: float)
    """
    rsi_config = config.get("signals", {}).get("rsi_filter", {})
    if not rsi_config.get("enabled", False):
        return True, None, 1.0

    rsi_data = features.get("rsi_2", pd.DataFrame())
    if rsi_data.empty or symbol not in rsi_data.columns:
        return True, None, 1.0

    rsi_value = rsi_data[symbol].dropna()
    if rsi_value.empty:
        return True, None, 1.0

    current_rsi = float(rsi_value.iloc[-1])
    overbought = rsi_config.get("overbought_threshold", 90)
    oversold_boost = rsi_config.get("oversold_boost", 1.25)

    if current_rsi > overbought:
        logger.info(
            f"RSI filter BLOCKED {symbol}: RSI(2) = {current_rsi:.1f} > {overbought}",
        )
        return False, current_rsi, 1.0

    # Boost for oversold conditions
    multiplier = 1.0
    if current_rsi < 10:
        multiplier = oversold_boost
        logger.info(
            f"RSI oversold boost for {symbol}: RSI(2) = {current_rsi:.1f}, multiplier = {multiplier}",
        )

    return True, current_rsi, multiplier


def compute_signal_strength(features, symbol, benchmark, config):
    """
    Compute composite signal strength from multi-timeframe returns.

    Stronger momentum = higher confidence = larger position.
    The strength score (0.0 to 1.0+) flows through to position sizing.

    Returns:
        float — composite signal strength score
    """
    ss_config = config.get("signals", {}).get("signal_strength", {})
    if not ss_config.get("enabled", False):
        return 1.0

    returns = features.get("returns", {})
    weight_12m = ss_config.get("weight_12m", 0.50)
    weight_6m = ss_config.get("weight_6m", 0.30)
    weight_3m = ss_config.get("weight_3m", 0.20)

    # Get excess returns over benchmark for each timeframe
    scores = {}
    for label, weight in [("12m", weight_12m), ("6m", weight_6m), ("3m", weight_3m)]:
        ret_df = returns.get(label, pd.DataFrame())
        if not ret_df.empty and symbol in ret_df.columns and benchmark in ret_df.columns:
            latest = ret_df.iloc[-1]
            excess = float(latest[symbol]) - float(latest[benchmark])
            scores[label] = excess * weight
        else:
            scores[label] = 0.0

    # Raw composite score (can be negative, but we only use for BUY signals)
    raw_score = sum(scores.values())

    # Normalize: 10% composite excess return = strength 1.0
    # Higher is stronger, lower is weaker
    strength = max(0.1, min(2.0, raw_score / 0.10)) if raw_score > 0 else 0.1

    # Momentum acceleration bonus: if 3m excess > 6m excess, momentum is
    # accelerating — the trend is getting stronger, not fading.
    # This is one of the strongest predictors of continued momentum.
    ret_3m = returns.get("3m", pd.DataFrame())
    ret_6m = returns.get("6m", pd.DataFrame())
    accelerating = False
    if (not ret_3m.empty and not ret_6m.empty
            and symbol in ret_3m.columns and symbol in ret_6m.columns
            and benchmark in ret_3m.columns and benchmark in ret_6m.columns):
        excess_3m = float(ret_3m.iloc[-1][symbol]) - float(ret_3m.iloc[-1][benchmark])
        excess_6m = float(ret_6m.iloc[-1][symbol]) - float(ret_6m.iloc[-1][benchmark])
        if excess_3m > excess_6m and excess_3m > 0:
            accelerating = True
            strength *= 1.15  # 15% boost for accelerating momentum
            strength = min(2.0, strength)  # Cap at 2.0

    logger.info(
        f"Signal strength for {symbol}: {strength:.2f}",
        extra={"extra_data": {
            "scores": scores, "raw": round(raw_score, 4),
            "accelerating": accelerating,
        }},
    )
    return strength


def score_dual_momentum(features, strategy_config):
    """
    Run the full dual momentum scoring pipeline with alpha optimizations.

    Pipeline:
    1. Compute 12-month returns for risk assets and benchmark.
    2. Check absolute momentum (return > SHY return).
    3. Check multi-timeframe confirmation (6m alignment).
    4. Check minimum edge threshold (2% excess return).
    5. Check RSI(2) overbought filter.
    6. Rank risk assets by relative momentum.
    7. Compute signal strength score for position sizing.
    8. Select the top-ranked qualifying asset.
    9. If no risk asset qualifies → rotate to safe asset (AGG/SHY).

    Args:
        features: Dict from feature_store.build_features().
        strategy_config: Dict loaded from momentum_v1.yaml.

    Returns:
        List of signal dicts with signal_strength for position sizing.
    """
    logger.info("Running dual momentum scoring (v3.0 with optimizations)")

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

    # Step 3: Build qualified candidates (pass all filters)
    latest_returns = returns_12m.iloc[-1]
    benchmark_return = float(latest_returns.get(benchmark, 0))

    signals = []
    selected_asset = None

    for rank, (symbol, ret) in enumerate(relative_ranking):
        has_abs = abs_momentum.get(symbol, False)

        if has_abs:
            # Check multi-timeframe confirmation
            mt_confirmed, mt_details = check_multi_timeframe(
                features, symbol, benchmark, strategy_config
            )

            # Check minimum edge
            has_edge, excess_return = check_minimum_edge(ret, benchmark_return, strategy_config)

            # Check RSI overbought filter
            rsi_allowed, rsi_value, rsi_multiplier = check_rsi_filter(
                features, symbol, strategy_config
            )

            # All filters must pass for BUY
            fully_qualified = mt_confirmed and has_edge and rsi_allowed

            if fully_qualified and selected_asset is None:
                # Compute signal strength for position sizing
                strength = compute_signal_strength(
                    features, symbol, benchmark, strategy_config
                )
                strength *= rsi_multiplier  # Apply oversold boost if applicable

                selected_asset = symbol
                signals.append({
                    "symbol": symbol,
                    "signal_type": "BUY",
                    "score": ret,
                    "signal_strength": strength,
                    "metadata": {
                        "absolute_momentum": True,
                        "relative_rank": rank + 1,
                        "benchmark_return": benchmark_return,
                        "asset_return": ret,
                        "excess_return": excess_return,
                        "multi_timeframe": mt_details,
                        "rsi_2": rsi_value,
                        "signal_strength": round(strength, 3),
                    },
                })
            elif fully_qualified:
                # Qualified but not top-ranked
                strength = compute_signal_strength(
                    features, symbol, benchmark, strategy_config
                )
                signals.append({
                    "symbol": symbol,
                    "signal_type": "HOLD",
                    "score": ret,
                    "signal_strength": strength,
                    "metadata": {
                        "absolute_momentum": True,
                        "relative_rank": rank + 1,
                        "benchmark_return": benchmark_return,
                        "asset_return": ret,
                        "excess_return": excess_return,
                    },
                })
            else:
                # Has absolute momentum but failed a filter — SKIP not SELL
                skip_reasons = []
                if not mt_confirmed:
                    skip_reasons.append(f"6m trend not confirmed")
                if not has_edge:
                    skip_reasons.append(f"edge too small ({excess_return:.2%})")
                if not rsi_allowed:
                    skip_reasons.append(f"RSI(2) overbought ({rsi_value:.0f})")

                signals.append({
                    "symbol": symbol,
                    "signal_type": "HOLD",  # Don't sell — momentum exists, just filtered
                    "score": ret,
                    "signal_strength": 0.0,
                    "metadata": {
                        "absolute_momentum": True,
                        "relative_rank": rank + 1,
                        "benchmark_return": benchmark_return,
                        "asset_return": ret,
                        "filter_blocked": True,
                        "skip_reasons": skip_reasons,
                    },
                })
        else:
            # No absolute momentum — should not be held
            signals.append({
                "symbol": symbol,
                "signal_type": "SELL",
                "score": ret,
                "signal_strength": 0.0,
                "metadata": {
                    "absolute_momentum": False,
                    "relative_rank": rank + 1,
                    "benchmark_return": benchmark_return,
                    "asset_return": ret,
                },
            })

    # If no risk asset qualified → rotate to safe asset
    if selected_asset is None:
        # Rank safe assets by return and pick the best one
        safe_available = [s for s in safe_assets if s in returns_12m.columns and s != benchmark]
        if safe_available:
            safe_ranked = sorted(
                safe_available,
                key=lambda s: float(latest_returns.get(s, 0)),
                reverse=True,
            )
            best_safe = safe_ranked[0]
            safe_return = float(latest_returns.get(best_safe, 0))
            signals.append({
                "symbol": best_safe,
                "signal_type": "BUY",
                "score": safe_return,
                "signal_strength": 0.5,  # Reduced strength for defensive rotation
                "metadata": {
                    "absolute_momentum": False,
                    "relative_rank": 0,
                    "benchmark_return": benchmark_return,
                    "asset_return": safe_return,
                    "reason": "No risk asset qualified — rotating to bonds",
                },
            })

    logger.info(
        "Dual momentum scoring complete",
        extra={
            "extra_data": {
                "selected_asset": selected_asset,
                "signal_count": len(signals),
                "signals": [
                    {
                        "symbol": s["symbol"],
                        "type": s["signal_type"],
                        "score": round(s["score"], 4),
                        "strength": round(s.get("signal_strength", 0), 3),
                    }
                    for s in signals
                ],
            }
        },
    )
    return signals
