"""
Dual Momentum / Trend Core Strategy — concrete implementation.

Implements StrategyBase using the Antonacci dual momentum approach:
- 12-month return vs T-bills (absolute momentum)
- Relative ranking of risk assets
- 200-day SMA regime filter
- Monthly rebalance cadence

Capital allocation: 60% of portfolio (from config).
"""

import yaml
from core.logging import get_logger
from strategies.base import StrategyBase
from data.feature_store import build_features
from research.momentum_scorer import score_dual_momentum, run_daily_scan
from research.regime import classify_regime

logger = get_logger("strategies.momentum_v1")


def load_strategy_config(config_path="config/strategies/momentum_v1.yaml"):
    """Load momentum strategy configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DualMomentumStrategy(StrategyBase):
    """Dual Momentum / Trend Core strategy implementation."""

    def __init__(self, config=None):
        if config is None:
            config = load_strategy_config()
        super().__init__(config)
        self.lookback_days = config.get("signals", {}).get("lookback_days", 252)
        self.capital_pct = config.get("allocation", {}).get("capital_pct", 0.60)

    def get_instruments(self):
        """Get all instruments this strategy trades."""
        instruments = self.config.get("instruments", {})
        risk = instruments.get("risk_assets", [])
        safe = instruments.get("safe_assets", [])
        return risk + safe

    def generate_signals(self, data=None, exit_log=None):
        """
        Generate dual momentum signals (weekly rebalance).

        Args:
            data: Pre-built features dict, or None to build from DB.
            exit_log: Optional dict of {symbol: last_exit_date} for cooldown.

        Returns:
            List of signal dicts from momentum scorer.
        """
        logger.info(f"Generating signals for {self.name} v{self.version}")

        if data is None:
            symbols = self.get_instruments()
            # Extra buffer for 200-day SMA computation
            data = build_features(symbols, lookback_days=self.lookback_days + 100)

        if data.get("prices") is None or data["prices"].empty:
            logger.warning("No price data available for signal generation")
            return []

        signals = score_dual_momentum(data, self.config, exit_log=exit_log)

        logger.info(
            "Signal generation complete",
            extra={
                "extra_data": {
                    "strategy": self.name,
                    "signal_count": len(signals),
                }
            },
        )
        return signals

    def generate_daily_signals(self, data=None, held_positions=None, exit_log=None):
        """
        Generate daily exit+reallocate signals.

        Runs every trading day. Checks held positions for 3m breakdown,
        immediately finds replacements for freed slots.

        Args:
            data: Pre-built features dict, or None to build from DB.
            held_positions: List of currently held symbols.
            exit_log: Dict of {symbol: last_exit_date} for cooldown.

        Returns:
            List of signal dicts (SELL exits + BUY replacements).
        """
        logger.info(f"Generating daily signals for {self.name} v{self.version}")

        if held_positions is None:
            held_positions = []

        if not held_positions:
            logger.info("No held positions — daily scan skipped")
            return []

        if data is None:
            symbols = self.get_instruments()
            data = build_features(symbols, lookback_days=self.lookback_days + 100)

        if data.get("prices") is None or data["prices"].empty:
            logger.warning("No price data available for daily scan")
            return []

        signals = run_daily_scan(data, held_positions, self.config, exit_log=exit_log)

        logger.info(
            "Daily signal generation complete",
            extra={
                "extra_data": {
                    "strategy": self.name,
                    "signal_count": len(signals),
                    "exits": [s["symbol"] for s in signals if s["signal_type"] == "SELL"],
                    "replacements": [s["symbol"] for s in signals if s["signal_type"] == "BUY"],
                }
            },
        )
        return signals

    def get_regime(self, features=None):
        """
        Get current regime classification for this strategy.

        Args:
            features: Pre-built features dict, or None to build from DB.

        Returns:
            Regime classification dict.
        """
        if features is None:
            # Need SPY in features for regime check
            symbols = list(set(self.get_instruments() + ["SPY"]))
            features = build_features(symbols, lookback_days=self.lookback_days + 100)

        return classify_regime(features)

    def get_position_size(self, signal, portfolio_state):
        """
        Calculate position size for a signal.

        Uses capital allocation from config and respects regime multiplier.

        Args:
            signal: Signal dict from generate_signals().
            portfolio_state: Dict with keys: total_equity, cash, positions.

        Returns:
            Float — dollar amount to allocate (not shares).
        """
        total_equity = portfolio_state.get("total_equity", 0)
        if total_equity <= 0:
            logger.warning("No equity available for position sizing")
            return 0.0

        # Base allocation: strategy's capital percentage
        base_allocation = total_equity * self.capital_pct

        # Regime multiplier is applied by the decision engine, not here
        # This returns the raw strategy-level allocation
        logger.info(
            "Position size calculated",
            extra={
                "extra_data": {
                    "symbol": signal.get("symbol"),
                    "total_equity": total_equity,
                    "capital_pct": self.capital_pct,
                    "base_allocation": base_allocation,
                }
            },
        )
        return base_allocation
