"""
Crypto DCA Strategy — dollar-cost averaging into crypto assets.

No momentum filter, no regime filter. Just buy a fixed dollar amount
on schedule, rotating through coins one at a time.

Rotation model: each cycle buys ONE coin, cycling through the list
using a time-based index. This ensures every trade clears the $1
Alpaca minimum even with small weekly budgets.

This is a separate sleeve from equity momentum. Different market,
different time horizon, different logic.
"""

import time
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
        # Support both formats: list of dicts with "symbol" key, or plain strings
        if assets and isinstance(assets[0], dict):
            return [a["symbol"] for a in assets]
        return assets

    def _get_rotation_index(self):
        """
        Deterministic rotation index based on current time.

        Uses epoch hours divided by cycle interval to produce a stable
        index that advances each cycle. Same hour = same coin.
        """
        epoch_hours = int(time.time()) // 3600
        # 8-hour cycles: index advances every 8 hours
        cycle_number = epoch_hours // 8
        return cycle_number

    def generate_signals(self, data=None, exit_log=None):
        """
        Generate ONE DCA buy signal per cycle via rotation.

        Each cycle picks the next coin in the rotation list.
        The full dollars_per_cycle amount goes to that one coin,
        ensuring every trade clears the $1 Alpaca minimum.

        Returns:
            List with one signal dict (BUY for the selected coin).
        """
        instruments = self.get_instruments()
        if not instruments:
            logger.warning("No instruments configured for crypto DCA")
            return []

        # Pick one coin via rotation
        rotation_idx = self._get_rotation_index()
        coin_idx = rotation_idx % len(instruments)
        symbol = instruments[coin_idx]
        dollar_amount = self.dollars_per_cycle

        logger.info(
            f"DCA rotation: cycle #{rotation_idx} → {symbol} "
            f"(index {coin_idx}/{len(instruments)}), ${dollar_amount:.2f}",
        )

        if dollar_amount < self.min_trade_dollars:
            logger.warning(
                f"DCA skip {symbol}: ${dollar_amount:.2f} below "
                f"${self.min_trade_dollars:.2f} minimum",
            )
            return []

        signals = [{
            "symbol": symbol,
            "signal_type": "BUY",
            "score": 1.0,
            "signal_strength": 1.0,
            "metadata": {
                "strategy": "crypto_dca",
                "dollar_amount": dollar_amount,
                "rotation_index": coin_idx,
                "reason": f"DCA rotation: ${dollar_amount:.2f} into {symbol}",
            },
        }]

        logger.info(
            "DCA signal generation complete",
            extra={
                "extra_data": {
                    "strategy": self.name,
                    "signal_count": len(signals),
                    "total_deploy": dollar_amount,
                    "rotation_coin": symbol,
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
