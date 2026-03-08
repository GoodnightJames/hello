"""
Regime Tagger — labels every trade and decision with market context.

Stamps each trade with:
- trend regime: risk_on / risk_off (SPY vs 200d SMA)
- volatility regime: low_vol / normal_vol / high_vol
- market phase: trending / ranging / crisis

This enables performance analysis by regime:
- "Does the crypto sleeve only work in trending expansion?"
- "Is the equity filter over-defensive in recoveries?"

Usage:
    from risk.regime_tagger import tag_current_regime, tag_trade

    regime = tag_current_regime(features)
    # Returns: {"trend": "risk_on", "vol": "normal_vol", "phase": "trending"}

    tagged_trade = tag_trade(trade_dict, regime)
    # Adds regime labels to trade metadata
"""

import pandas as pd
from core.logging import get_logger

logger = get_logger("risk.regime_tagger")


def tag_current_regime(features=None):
    """
    Classify the current market regime from features.

    Returns a dict of regime labels that can be attached to trades.

    Args:
        features: Dict from feature_store.build_features().
                  If None, returns "unknown" labels.

    Returns:
        Dict with keys: trend, vol, phase
    """
    regime = {
        "trend": "unknown",
        "vol": "unknown",
        "phase": "unknown",
    }

    if features is None:
        return regime

    prices = features.get("prices", pd.DataFrame())

    # Trend: SPY vs 200d SMA
    if not prices.empty and "SPY" in prices.columns:
        spy = prices["SPY"].dropna()
        if len(spy) >= 200:
            sma_200 = spy.rolling(200).mean().iloc[-1]
            current = spy.iloc[-1]
            regime["trend"] = "risk_on" if current > sma_200 else "risk_off"
        elif len(spy) >= 50:
            sma_50 = spy.rolling(50).mean().iloc[-1]
            current = spy.iloc[-1]
            regime["trend"] = "risk_on" if current > sma_50 else "risk_off"

    # Volatility: 20-day realized vol of SPY
    if not prices.empty and "SPY" in prices.columns:
        spy = prices["SPY"].dropna()
        if len(spy) >= 21:
            returns = spy.pct_change().dropna().tail(20)
            vol_20d = float(returns.std()) * (252 ** 0.5)
            if vol_20d < 0.12:
                regime["vol"] = "low_vol"
            elif vol_20d < 0.22:
                regime["vol"] = "normal_vol"
            else:
                regime["vol"] = "high_vol"

    # Phase: combine trend + vol
    if regime["trend"] == "risk_on" and regime["vol"] in ("low_vol", "normal_vol"):
        regime["phase"] = "trending"
    elif regime["trend"] == "risk_off" and regime["vol"] == "high_vol":
        regime["phase"] = "crisis"
    elif regime["trend"] == "risk_off":
        regime["phase"] = "correction"
    else:
        regime["phase"] = "ranging"

    logger.info(
        "Regime tagged",
        extra={"extra_data": regime},
    )
    return regime


def tag_trade(trade_dict, regime):
    """
    Add regime labels to a trade's metadata.

    Args:
        trade_dict: Trade result dict (from execution).
        regime: Dict from tag_current_regime().

    Returns:
        trade_dict with "regime" key added.
    """
    trade_dict["regime"] = regime
    return trade_dict


def summarize_performance_by_regime(trades):
    """
    Group trade performance by regime labels.

    Args:
        trades: List of trade dicts with "regime" key.

    Returns:
        Dict of {regime_label: {trades, wins, pnl, hit_rate}} for each dimension.
    """
    summaries = {}

    for dimension in ("trend", "vol", "phase"):
        groups = {}
        for t in trades:
            regime = t.get("regime", {})
            label = regime.get(dimension, "unknown")
            if label not in groups:
                groups[label] = {"trades": 0, "wins": 0, "pnl": 0.0}
            groups[label]["trades"] += 1
            pnl = t.get("realized_pnl", 0)
            groups[label]["pnl"] += pnl
            if pnl > 0:
                groups[label]["wins"] += 1

        for label, stats in groups.items():
            stats["hit_rate"] = (
                round(stats["wins"] / stats["trades"], 3)
                if stats["trades"] > 0 else 0
            )
            stats["pnl"] = round(stats["pnl"], 2)

        summaries[dimension] = groups

    return summaries
