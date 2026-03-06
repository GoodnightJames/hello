"""
Regime classifier — determines current market regime.

Two regime filters from spec:
1. SPY below 200-day SMA → risk-off (hold cash / reduce positions 50%)
2. VIX above 30 → high volatility (reduce position sizes 50%)

These filters gate the decision engine. When both fire simultaneously,
position sizes are reduced by 50% twice (to 25% of normal).
"""

import yaml
from core.logging import get_logger
from data.feature_store import get_price_history, compute_sma

logger = get_logger("research.regime")

# Regime states
REGIME_RISK_ON = "risk_on"
REGIME_RISK_OFF = "risk_off"
REGIME_HIGH_VOL = "high_volatility"


def load_risk_params(config_path="config/risk_params.yaml"):
    """Load risk parameters from YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def classify_trend_regime(prices_df, sma_200d_df, reference_symbol="SPY"):
    """
    Classify trend regime based on SPY vs 200-day SMA.

    Args:
        prices_df: DataFrame of close prices with symbol columns.
        sma_200d_df: DataFrame of 200-day SMA values.
        reference_symbol: Symbol to check (default SPY).

    Returns:
        Dict with regime info:
        {
            "trend_regime": "risk_on" | "risk_off",
            "spy_price": float,
            "spy_200d_sma": float,
            "above_200d": bool
        }
    """
    if reference_symbol not in prices_df.columns:
        logger.error(
            f"{reference_symbol} not found in price data",
            extra={"extra_data": {"available": list(prices_df.columns)}},
        )
        return {
            "trend_regime": REGIME_RISK_OFF,
            "spy_price": None,
            "spy_200d_sma": None,
            "above_200d": False,
        }

    if reference_symbol not in sma_200d_df.columns:
        logger.error(f"{reference_symbol} not found in SMA data")
        return {
            "trend_regime": REGIME_RISK_OFF,
            "spy_price": None,
            "spy_200d_sma": None,
            "above_200d": False,
        }

    latest_price = prices_df[reference_symbol].dropna().iloc[-1]
    latest_sma = sma_200d_df[reference_symbol].dropna().iloc[-1]

    above_200d = latest_price > latest_sma
    regime = REGIME_RISK_ON if above_200d else REGIME_RISK_OFF

    result = {
        "trend_regime": regime,
        "spy_price": float(latest_price),
        "spy_200d_sma": float(latest_sma),
        "above_200d": above_200d,
    }

    logger.info(
        f"Trend regime: {regime}",
        extra={"extra_data": result},
    )
    return result


def classify_volatility_regime(vix_price=None, vix_threshold=30):
    """
    Classify volatility regime based on VIX level.

    Note: VIX data may come from external source or be proxied.
    In v1, if VIX data is unavailable, default to normal volatility.

    Args:
        vix_price: Current VIX level (float or None).
        vix_threshold: Threshold for high-vol classification (from config).

    Returns:
        Dict with regime info:
        {
            "vol_regime": "normal" | "high_volatility",
            "vix_level": float | None,
            "vix_threshold": float,
            "above_threshold": bool
        }
    """
    if vix_price is None:
        logger.info("VIX data unavailable — defaulting to normal volatility")
        return {
            "vol_regime": "normal",
            "vix_level": None,
            "vix_threshold": vix_threshold,
            "above_threshold": False,
        }

    above = vix_price > vix_threshold
    regime = REGIME_HIGH_VOL if above else "normal"

    result = {
        "vol_regime": regime,
        "vix_level": float(vix_price),
        "vix_threshold": vix_threshold,
        "above_threshold": above,
    }

    logger.info(
        f"Volatility regime: {regime}",
        extra={"extra_data": result},
    )
    return result


def get_position_size_multiplier(trend_regime, vol_regime, risk_params=None):
    """
    Calculate position size multiplier based on combined regime state.

    From spec:
    - SPY below 200d SMA → reduce 50%
    - VIX > 30 → reduce 50%
    - Both → reduce to 25% (multiplicative)

    Args:
        trend_regime: Dict from classify_trend_regime().
        vol_regime: Dict from classify_volatility_regime().
        risk_params: Risk parameters dict (loaded from YAML if not provided).

    Returns:
        Float multiplier (0.0 to 1.0) to apply to position sizes.
    """
    if risk_params is None:
        risk_params = load_risk_params()

    throttles = risk_params.get("regime_throttles", {})
    spy_reduce = throttles.get("spy_below_200d_reduce", 0.5)
    vix_reduce = throttles.get("vix_position_reduce", 0.5)

    multiplier = 1.0

    if trend_regime.get("trend_regime") == REGIME_RISK_OFF:
        multiplier *= (1.0 - spy_reduce)

    if vol_regime.get("vol_regime") == REGIME_HIGH_VOL:
        multiplier *= (1.0 - vix_reduce)

    logger.info(
        "Position size multiplier calculated",
        extra={
            "extra_data": {
                "trend_regime": trend_regime.get("trend_regime"),
                "vol_regime": vol_regime.get("vol_regime"),
                "multiplier": multiplier,
            }
        },
    )
    return multiplier


def classify_regime(features, risk_params=None):
    """
    Full regime classification — convenience function combining all checks.

    Args:
        features: Dict from feature_store.build_features().
        risk_params: Risk parameters dict.

    Returns:
        Dict with complete regime state:
        {
            "trend": {trend_regime_dict},
            "volatility": {vol_regime_dict},
            "position_multiplier": float,
            "allow_new_entries": bool
        }
    """
    if risk_params is None:
        risk_params = load_risk_params()

    prices = features.get("prices", None)
    sma = features.get("sma", {})
    sma_200d = sma.get("200d", None)

    # Trend regime
    if prices is not None and sma_200d is not None and not prices.empty and not sma_200d.empty:
        trend = classify_trend_regime(prices, sma_200d)
    else:
        logger.warning("Insufficient data for trend regime — defaulting to risk_off")
        trend = {
            "trend_regime": REGIME_RISK_OFF,
            "spy_price": None,
            "spy_200d_sma": None,
            "above_200d": False,
        }

    # Volatility regime (VIX — use VIXY or ^VIX proxy if available)
    vol = classify_volatility_regime(
        vix_price=None,  # VIX integration in v1.5
        vix_threshold=risk_params.get("regime_throttles", {}).get("vix_threshold", 30),
    )

    multiplier = get_position_size_multiplier(trend, vol, risk_params)

    # In risk-off trend regime, the strategy should hold cash (no new entries)
    # But existing positions may be held per strategy rules
    allow_new_entries = trend.get("trend_regime") == REGIME_RISK_ON

    result = {
        "trend": trend,
        "volatility": vol,
        "position_multiplier": multiplier,
        "allow_new_entries": allow_new_entries,
    }

    logger.info(
        "Regime classification complete",
        extra={
            "extra_data": {
                "trend": trend["trend_regime"],
                "vol": vol["vol_regime"],
                "multiplier": multiplier,
                "allow_entries": allow_new_entries,
            }
        },
    )
    return result
