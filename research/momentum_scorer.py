"""
Dual Momentum Scorer — Antonacci (2014) with alpha optimizations.

Two momentum components:
1. Absolute momentum: Is the asset's 12-month return above T-bills (SHY)?
   If no → hold cash proxy (SHY).
2. Relative momentum: Among risk assets with positive absolute momentum,
   which has the highest 12-month return? Hold that one.

v4.0 optimizations:
- Multi-timeframe confirmation (12m + 6m alignment)
- Minimum edge threshold (skip marginal signals)
- RSI(2) overbought filter (don't buy at the top)
- Composite signal strength score (for position sizing)
- Multi-position: BUY top N qualified assets (not just #1)
- Fast exit: sell when 3-month momentum breaks down
- Volatility-scaled sizing: low-vol assets get bigger positions
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

    Returns:
        (confirmed: bool, details: dict)
    """
    mt_config = config.get("signals", {}).get("multi_timeframe", {})
    if not mt_config.get("enabled", False):
        return True, {"reason": "multi_timeframe disabled"}

    returns = features.get("returns", {})
    details = {}
    confirmed = True

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

    multiplier = 1.0
    if current_rsi < 10:
        multiplier = oversold_boost
        logger.info(
            f"RSI oversold boost for {symbol}: RSI(2) = {current_rsi:.1f}, multiplier = {multiplier}",
        )

    return True, current_rsi, multiplier


def check_fast_exit(features, symbol, benchmark, config):
    """
    Check if a held position should be exited early due to 3-month breakdown.

    The standard 12-month exit is too slow — a stock can drop 30% before
    the 12m return turns negative. If 3-month return breaks below benchmark,
    the trend is deteriorating. Exit now, don't wait.

    Returns:
        (should_exit: bool, details: dict)
    """
    fast_exit_config = config.get("signals", {}).get("fast_exit", {})
    if not fast_exit_config.get("enabled", False):
        return False, {}

    if not fast_exit_config.get("exit_on_3m_breakdown", True):
        return False, {}

    returns = features.get("returns", {})
    returns_3m = returns.get("3m", pd.DataFrame())

    if returns_3m.empty or symbol not in returns_3m.columns:
        return False, {}

    if benchmark not in returns_3m.columns:
        return False, {}

    latest = returns_3m.iloc[-1]
    sym_3m = float(latest[symbol])
    bench_3m = float(latest[benchmark])

    details = {
        "return_3m": round(sym_3m, 4),
        "benchmark_3m": round(bench_3m, 4),
    }

    if sym_3m < bench_3m:
        logger.warning(
            f"FAST EXIT triggered for {symbol}: 3m return {sym_3m:.4f} < benchmark {bench_3m:.4f}",
            extra={"extra_data": details},
        )
        return True, details

    return False, details


def is_in_cooldown(symbol, exit_log, config):
    """
    Check if a symbol is in cooldown after a fast exit.

    Prevents whipsaw: exit Monday, re-buy Friday, exit again Monday.
    After a fast exit, wait N trading days before allowing re-entry.

    Args:
        symbol: The asset to check.
        exit_log: Dict of {symbol: last_exit_date} (pd.Timestamp or str).
        config: Strategy config dict.

    Returns:
        (in_cooldown: bool, days_remaining: int)
    """
    fast_exit_config = config.get("signals", {}).get("fast_exit", {})
    cooldown_days = fast_exit_config.get("cooldown_days", 5)

    if cooldown_days <= 0 or symbol not in exit_log:
        return False, 0

    last_exit = pd.Timestamp(exit_log[symbol])
    today = pd.Timestamp.now().normalize()
    # Count business days since exit
    bdays = pd.bdate_range(start=last_exit, end=today)
    elapsed = max(0, len(bdays) - 1)  # exclude the exit day itself

    if elapsed < cooldown_days:
        remaining = cooldown_days - elapsed
        logger.info(
            f"Cooldown active for {symbol}: {elapsed}/{cooldown_days} days elapsed, {remaining} remaining",
        )
        return True, remaining

    return False, 0


def run_daily_exit_scan(features, held_positions, strategy_config):
    """
    Daily fast-exit scan — runs every trading day, not just on rebalance.

    Checks each held position for 3-month momentum breakdown.
    This is the "don't wait until Friday" check.

    Args:
        features: Dict from feature_store.build_features().
        held_positions: List of symbols currently held.
        strategy_config: Dict loaded from momentum_v1.yaml.

    Returns:
        List of signal dicts (SELL signals only — daily scan never buys).
    """
    logger.info(f"Running daily exit scan for {len(held_positions)} positions")

    benchmark = strategy_config.get("signals", {}).get("benchmark", "SHY")
    signals = []

    for symbol in held_positions:
        should_exit, details = check_fast_exit(
            features, symbol, benchmark, strategy_config
        )
        if should_exit:
            signals.append({
                "symbol": symbol,
                "signal_type": "SELL",
                "score": 0.0,
                "signal_strength": 0.0,
                "metadata": {
                    "scan_type": "daily_exit",
                    "fast_exit": True,
                    "exit_details": details,
                },
            })

    logger.info(
        f"Daily exit scan complete: {len(signals)} exits triggered",
        extra={"extra_data": {
            "held": held_positions,
            "exits": [s["symbol"] for s in signals],
        }},
    )
    return signals


def compute_volatility_scalar(features, symbol):
    """
    Compute inverse-volatility position size scalar.

    Low volatility assets get bigger positions (less risky per dollar).
    Uses 20-day realized vol, annualized. Target 15%.
    Clamped [0.5, 1.5].

    Returns:
        float — volatility scalar for position sizing
    """
    prices = features.get("prices", pd.DataFrame())
    if prices.empty or symbol not in prices.columns:
        return 1.0

    sym_prices = prices[symbol].dropna()
    if len(sym_prices) < 21:
        return 1.0

    returns = sym_prices.pct_change().dropna().tail(20)
    if len(returns) < 10:
        return 1.0

    realized_vol = float(returns.std()) * (252 ** 0.5)
    if realized_vol <= 0:
        return 1.0

    target_vol = 0.15
    scalar = target_vol / realized_vol
    scalar = max(0.5, min(1.5, scalar))

    logger.info(
        f"Volatility scalar for {symbol}: {scalar:.2f} (realized vol: {realized_vol:.1%})",
    )
    return scalar


def compute_signal_strength(features, symbol, benchmark, config):
    """
    Compute composite signal strength from multi-timeframe returns.

    Includes momentum acceleration bonus and volatility adjustment.

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

    scores = {}
    for label, weight in [("12m", weight_12m), ("6m", weight_6m), ("3m", weight_3m)]:
        ret_df = returns.get(label, pd.DataFrame())
        if not ret_df.empty and symbol in ret_df.columns and benchmark in ret_df.columns:
            latest = ret_df.iloc[-1]
            excess = float(latest[symbol]) - float(latest[benchmark])
            scores[label] = excess * weight
        else:
            scores[label] = 0.0

    raw_score = sum(scores.values())
    strength = max(0.1, min(2.0, raw_score / 0.10)) if raw_score > 0 else 0.1

    # Momentum acceleration bonus
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
            strength *= 1.15
            strength = min(2.0, strength)

    # Volatility adjustment
    vol_scalar = compute_volatility_scalar(features, symbol)
    strength *= vol_scalar
    strength = max(0.1, min(2.0, strength))

    logger.info(
        f"Signal strength for {symbol}: {strength:.2f}",
        extra={"extra_data": {
            "scores": scores, "raw": round(raw_score, 4),
            "accelerating": accelerating,
            "vol_scalar": round(vol_scalar, 2),
        }},
    )
    return strength


def score_dual_momentum(features, strategy_config, exit_log=None):
    """
    Run the full dual momentum scoring pipeline with alpha optimizations.

    v4.0: Multi-position BUY (up to N qualified assets), fast exits,
    volatility-scaled sizing, cooldown enforcement.

    Args:
        features: Dict from feature_store.build_features().
        strategy_config: Dict loaded from momentum_v1.yaml.
        exit_log: Optional dict of {symbol: last_exit_date} for cooldown tracking.

    Returns:
        List of signal dicts with signal_strength for position sizing.
    """
    if exit_log is None:
        exit_log = {}

    logger.info("Running dual momentum scoring (v4.0 multi-position)")

    returns = features.get("returns", {})
    returns_12m = returns.get("12m", pd.DataFrame())

    if returns_12m.empty:
        logger.warning("No 12-month return data — cannot score")
        return []

    instruments = strategy_config.get("instruments", {})
    risk_assets = instruments.get("risk_assets", [])
    safe_assets = instruments.get("safe_assets", [])
    benchmark = strategy_config.get("signals", {}).get("benchmark", "SHY")
    max_buy_signals = strategy_config.get("signals", {}).get("max_buy_signals", 1)

    all_symbols = risk_assets + safe_assets
    available = [s for s in all_symbols if s in returns_12m.columns]

    if not available:
        logger.warning("No strategy instruments found in data")
        return []

    abs_momentum = compute_absolute_momentum(returns_12m, benchmark)
    risk_available = [s for s in risk_assets if s in returns_12m.columns]
    relative_ranking = compute_relative_momentum(returns_12m, risk_available)

    latest_returns = returns_12m.iloc[-1]
    benchmark_return = float(latest_returns.get(benchmark, 0))

    signals = []
    selected_count = 0

    for rank, (symbol, ret) in enumerate(relative_ranking):
        has_abs = abs_momentum.get(symbol, False)

        if has_abs:
            # Fast exit check — overrides everything
            should_exit, exit_details = check_fast_exit(
                features, symbol, benchmark, strategy_config
            )
            if should_exit:
                signals.append({
                    "symbol": symbol,
                    "signal_type": "SELL",
                    "score": ret,
                    "signal_strength": 0.0,
                    "metadata": {
                        "absolute_momentum": True,
                        "relative_rank": rank + 1,
                        "benchmark_return": benchmark_return,
                        "asset_return": ret,
                        "fast_exit": True,
                        "exit_details": exit_details,
                    },
                })
                continue

            # Cooldown check — recently exited symbols can't re-enter yet
            in_cd, cd_remaining = is_in_cooldown(symbol, exit_log, strategy_config)
            if in_cd:
                signals.append({
                    "symbol": symbol,
                    "signal_type": "HOLD",
                    "score": ret,
                    "signal_strength": 0.0,
                    "metadata": {
                        "absolute_momentum": True,
                        "relative_rank": rank + 1,
                        "benchmark_return": benchmark_return,
                        "asset_return": ret,
                        "cooldown": True,
                        "cooldown_days_remaining": cd_remaining,
                    },
                })
                continue

            mt_confirmed, mt_details = check_multi_timeframe(
                features, symbol, benchmark, strategy_config
            )
            has_edge, excess_return = check_minimum_edge(ret, benchmark_return, strategy_config)
            rsi_allowed, rsi_value, rsi_multiplier = check_rsi_filter(
                features, symbol, strategy_config
            )

            fully_qualified = mt_confirmed and has_edge and rsi_allowed

            if fully_qualified and selected_count < max_buy_signals:
                strength = compute_signal_strength(
                    features, symbol, benchmark, strategy_config
                )
                strength *= rsi_multiplier

                selected_count += 1
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
                        "buy_slot": selected_count,
                    },
                })
            elif fully_qualified:
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
                skip_reasons = []
                if not mt_confirmed:
                    skip_reasons.append(f"6m trend not confirmed")
                if not has_edge:
                    skip_reasons.append(f"edge too small ({excess_return:.2%})")
                if not rsi_allowed:
                    skip_reasons.append(f"RSI(2) overbought ({rsi_value:.0f})")

                signals.append({
                    "symbol": symbol,
                    "signal_type": "HOLD",
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
    if selected_count == 0:
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
                "signal_strength": 0.5,
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
                "buy_count": selected_count,
                "max_buy_signals": max_buy_signals,
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
