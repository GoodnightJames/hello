"""
Transaction Cost Model — realistic fee, spread, and slippage estimation.

Every signal and backtest result should be adjusted for costs.
Without this, paper P&L overstates live performance.

Cost components:
1. Broker fees (Alpaca: 0 for equities, ~0.15% for crypto)
2. Spread cost (bid-ask, varies by asset class and liquidity)
3. Slippage (market impact, worse for illiquid/volatile assets)
4. Bar-close to execution lag (price moved between signal and fill)

Usage:
    from risk.cost_model import estimate_round_trip_cost, estimate_one_way_cost

    cost = estimate_one_way_cost("BTC/USD", notional=100.0, volatility=0.05)
    # Returns: {"total_bps": 35.0, "fee_bps": 15.0, "spread_bps": 10.0, "slippage_bps": 10.0}
"""

import yaml
from core.logging import get_logger

logger = get_logger("risk.cost_model")

# Default cost assumptions (overridden by config)
DEFAULT_COSTS = {
    "crypto": {
        "fee_bps": 15.0,          # Alpaca crypto fee ~0.15% (15 bps)
        "base_spread_bps": 10.0,  # Typical BTC/ETH spread
        "slippage_per_vol_bps": 5.0,  # Additional slippage per 1% realized vol
        "min_spread_bps": 5.0,    # Floor spread for liquid coins
        "max_spread_bps": 50.0,   # Cap for illiquid/meme coins
    },
    "equity": {
        "fee_bps": 0.0,           # Alpaca zero commission equities
        "base_spread_bps": 2.0,   # ETF spreads are tight
        "slippage_per_vol_bps": 1.0,
        "min_spread_bps": 1.0,
        "max_spread_bps": 10.0,
    },
}

# Per-symbol spread overrides for known illiquid/wide-spread assets
SPREAD_OVERRIDES_BPS = {
    "SHIB/USD": 30.0,
    "PEPE/USD": 40.0,
    "DOGE/USD": 15.0,
    "DOT/USD": 15.0,
    "AVAX/USD": 12.0,
    "LINK/USD": 10.0,
    "SOL/USD": 8.0,
    "XRP/USD": 10.0,
    "ETH/USD": 5.0,
    "BTC/USD": 3.0,
}


def _load_cost_config():
    """Load cost model config if available, else use defaults."""
    try:
        with open("config/risk_params.yaml", "r") as f:
            params = yaml.safe_load(f)
        return params.get("cost_model", {})
    except Exception:
        return {}


def _is_crypto(symbol):
    """Check if symbol is crypto (uses slash notation)."""
    return "/" in symbol


def estimate_one_way_cost(symbol, notional=100.0, volatility=0.0, cost_config=None):
    """
    Estimate one-way transaction cost in basis points.

    Args:
        symbol: Ticker symbol (e.g., "BTC/USD" or "SPY").
        notional: Dollar amount of the trade.
        volatility: Recent realized volatility (daily, as decimal e.g. 0.05 = 5%).
        cost_config: Optional override config dict.

    Returns:
        Dict with cost breakdown in basis points:
        {
            "total_bps": float,
            "fee_bps": float,
            "spread_bps": float,
            "slippage_bps": float,
            "total_dollars": float,
        }
    """
    if cost_config is None:
        cost_config = _load_cost_config()

    asset_class = "crypto" if _is_crypto(symbol) else "equity"
    defaults = DEFAULT_COSTS[asset_class]
    class_config = cost_config.get(asset_class, {})

    # Fee
    fee_bps = class_config.get("fee_bps", defaults["fee_bps"])

    # Spread — use per-symbol override if available
    if symbol in SPREAD_OVERRIDES_BPS:
        spread_bps = SPREAD_OVERRIDES_BPS[symbol]
    else:
        spread_bps = class_config.get("base_spread_bps", defaults["base_spread_bps"])

    # Clamp spread
    min_spread = class_config.get("min_spread_bps", defaults["min_spread_bps"])
    max_spread = class_config.get("max_spread_bps", defaults["max_spread_bps"])
    spread_bps = max(min_spread, min(max_spread, spread_bps))

    # Slippage — scales with volatility
    slippage_per_vol = class_config.get("slippage_per_vol_bps", defaults["slippage_per_vol_bps"])
    vol_pct = volatility * 100  # Convert 0.05 → 5.0
    slippage_bps = slippage_per_vol * vol_pct

    total_bps = fee_bps + spread_bps + slippage_bps
    total_dollars = notional * total_bps / 10000.0

    return {
        "total_bps": round(total_bps, 2),
        "fee_bps": round(fee_bps, 2),
        "spread_bps": round(spread_bps, 2),
        "slippage_bps": round(slippage_bps, 2),
        "total_dollars": round(total_dollars, 4),
    }


def estimate_round_trip_cost(symbol, notional=100.0, volatility=0.0, cost_config=None):
    """
    Estimate round-trip (buy + sell) transaction cost.

    Returns:
        Dict with total round-trip cost breakdown.
    """
    one_way = estimate_one_way_cost(symbol, notional, volatility, cost_config)
    return {
        "total_bps": round(one_way["total_bps"] * 2, 2),
        "fee_bps": round(one_way["fee_bps"] * 2, 2),
        "spread_bps": round(one_way["spread_bps"] * 2, 2),
        "slippage_bps": round(one_way["slippage_bps"] * 2, 2),
        "total_dollars": round(one_way["total_dollars"] * 2, 4),
    }


def compute_cost_hurdle(symbol, volatility=0.0, holding_hours=4.0):
    """
    Compute the minimum return needed to break even after costs.

    For a crypto trade with $10 notional, 15 bps fee + 10 bps spread each way,
    you need ~50 bps (0.50%) return just to break even on a round trip.

    Args:
        symbol: Ticker symbol.
        volatility: Daily realized volatility.
        holding_hours: Expected holding period in hours.

    Returns:
        Float — minimum return (as decimal) to break even.
    """
    rt_cost = estimate_round_trip_cost(symbol, notional=100.0, volatility=volatility)
    hurdle = rt_cost["total_bps"] / 10000.0
    return hurdle


def apply_cost_penalty_to_score(symbol, raw_score, volatility=0.0, notional=10.0):
    """
    Subtract estimated round-trip cost from a signal score.

    This is the key integration point: signals should only be acted on
    if their expected edge exceeds transaction costs.

    Args:
        symbol: Ticker symbol.
        raw_score: Raw momentum/signal score.
        volatility: Daily realized volatility.
        notional: Trade size in dollars.

    Returns:
        (adjusted_score, cost_info)
    """
    rt_cost = estimate_round_trip_cost(symbol, notional, volatility)
    cost_penalty = rt_cost["total_bps"] / 10000.0  # Convert bps to decimal

    adjusted_score = raw_score - cost_penalty

    logger.info(
        f"Cost adjustment: {symbol} raw={raw_score:.4f} cost={cost_penalty:.4f} adj={adjusted_score:.4f}",
        extra={"extra_data": {
            "symbol": symbol,
            "raw_score": round(raw_score, 4),
            "cost_penalty": round(cost_penalty, 4),
            "adjusted_score": round(adjusted_score, 4),
            "cost_breakdown": rt_cost,
        }},
    )

    return adjusted_score, rt_cost
