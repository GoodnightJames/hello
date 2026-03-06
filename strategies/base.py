"""
Strategy plugin base class.

All strategies inherit from StrategyBase. To add a new strategy:
1. Create a new file in strategies/ that inherits StrategyBase
2. Implement generate_signals() and get_position_size()
3. Add strategy config YAML to config/strategies/
4. Update config/settings.yaml to reference the new strategy
"""

from abc import ABC, abstractmethod


class StrategyBase(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config):
        """
        Initialize strategy with its configuration.

        Args:
            config: Dict loaded from the strategy's YAML config file.
        """
        self.config = config
        self.name = config.get("name", "unnamed")
        self.version = config.get("version", "0.0.0")

    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals from market data.

        Args:
            data: DataFrame of daily bars for strategy instruments.

        Returns:
            List of signal dicts with keys: symbol, signal_type, score, metadata.
        """
        ...

    @abstractmethod
    def get_position_size(self, signal, portfolio_state):
        """
        Calculate position size for a given signal.

        Args:
            signal: Signal dict from generate_signals().
            portfolio_state: Current portfolio state (cash, equity, positions).

        Returns:
            Float representing number of shares/units to trade.
        """
        ...
