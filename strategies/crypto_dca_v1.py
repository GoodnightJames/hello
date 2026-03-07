"""
Crypto DCA Strategy — dollar-cost averaging into crypto assets.

No momentum filter, no regime filter. Just buy a fixed dollar amount
on schedule, weighted by market-cap allocation.

This is a separate sleeve from equity momentum. Different market,
different time horizon, different logic.
"""

import yaml
from core.logging import get_logger
from strategies.base import StrategyBase

logger = get_logger("strategies.crypto_dca_v1")


def load_strategy_config(config_path="config/strategies/crypto_dca_v1.yaml"):
    """Load crypto DCA strategy configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class CryptoDCAStrategy(StrategyBase):
    """Crypto dollar-cost averaging strategy."""

    def __init__(self, config=None):
        if config is None:
            config = load_strategy_config()
        super().__init__(config)
        self.dollars_per_cycle = config.get("allocation", {}).get("dollars_per_cycle", 2.0)
        self.min_trade_dollars = config.get("allocation", {}).get("min_trade_dollars", 1.0)
        self.sleeve_pct = config.get("allocation", {}).get("sleeve_pct", 0.30)

    def get_instruments(self):
        """Get all crypto instruments this strategy trades."""
        assets = self.config.get("instruments", {}).get("assets", [])
        return [a["symbol"] for a in assets]

    def get_weighted_assets(self):
        """Get list of (symbol, weight) tuples."""
        assets = self.config.get("instruments", {}).get("assets", [])
        return [(a["symbol"], a["weight"]) for a in assets]

    def generate_signals(self, data=None, exit_log=None):
        """
        Generate DCA buy signals for all crypto assets.

        No momentum check, no regime check. Just buy.
        Each signal includes the dollar amount to deploy based on weight.

        Returns:
            List of signal dicts — always BUY signals for each asset.
        """
        logger.info(f"Generating DCA signals for {self.name} v{self.version}")

        weighted_assets = self.get_weighted_assets()
        signals = []

        for symbol, weight in weighted_assets:
            dollar_amount = self.dollars_per_cycle * weight

            if dollar_amount < self.min_trade_dollars:
                logger.info(
                    f"DCA skip {symbol}: ${dollar_amount:.2f} below "
                    f"${self.min_trade_dollars:.2f} minimum",
                )
                continue

            signals.append({
                "symbol": symbol,
                "signal_type": "BUY",
                "score": weight,
                "signal_strength": 1.0,
                "metadata": {
                    "strategy": "crypto_dca",
                    "weight": weight,
                    "dollar_amount": dollar_amount,
                    "reason": f"DCA: ${dollar_amount:.2f} ({weight:.0%} of ${self.dollars_per_cycle:.2f})",
                },
            })

        logger.info(
            "DCA signal generation complete",
            extra={
                "extra_data": {
                    "strategy": self.name,
                    "signal_count": len(signals),
                    "total_deploy": sum(
                        s["metadata"]["dollar_amount"] for s in signals
                    ),
                }
            },
        )
        return signals

    def get_position_size(self, signal, portfolio_state):
        """
        Position size for DCA is the fixed dollar amount from the signal.

        Not percentage-based — just the flat dollar amount per cycle.
        """
        return signal.get("metadata", {}).get("dollar_amount", self.min_trade_dollars)
