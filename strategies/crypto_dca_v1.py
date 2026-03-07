"""
Crypto DCA + Exit Strategy — v2.0

Buy side: Dollar-cost average into crypto assets on a fixed schedule,
rotating through coins one at a time.

Sell side: Every cycle checks all open crypto positions for exit conditions:
1. Take-profit — sell when position is up X% from cost basis
2. Trailing stop — sell when price drops Y% from peak (after arming)
3. Hard stop — sell when down Z% from entry (unconditional)

Proceeds from sells recycle back to the crypto sleeve for re-deployment,
turning the weekly $30 allocation into a capital recycling engine.
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
    """Crypto dollar-cost averaging strategy with active exit management."""

    def __init__(self, config=None):
        if config is None:
            config = load_strategy_config()
        super().__init__(config)
        self.dollars_per_cycle = config.get("allocation", {}).get("dollars_per_cycle", 2.0)
        self.min_trade_dollars = config.get("allocation", {}).get("min_trade_dollars", 1.0)
        self.sleeve_pct = config.get("allocation", {}).get("sleeve_pct", 0.30)

        # Exit configuration
        exits = config.get("exits", {})

        tp = exits.get("take_profit", {})
        self.take_profit_enabled = tp.get("enabled", False)
        self.take_profit_pct = tp.get("target_pct", 0.08)

        ts = exits.get("trailing_stop", {})
        self.trailing_stop_enabled = ts.get("enabled", False)
        self.trailing_stop_pct = ts.get("stop_pct", 0.05)
        self.trailing_stop_min_gain = ts.get("min_gain_to_activate", 0.03)

        hs = exits.get("hard_stop", {})
        self.hard_stop_enabled = hs.get("enabled", False)
        self.hard_stop_pct = hs.get("stop_pct", 0.10)

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
        ensuring every trade clears the $10 Alpaca minimum.

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

    def generate_exit_signals(self, positions, cost_bases, high_water_marks):
        """
        Check all crypto positions for exit conditions.

        Args:
            positions: Dict of {symbol: {"qty": float, "current_price": float}}
            cost_bases: Dict of {symbol: {"avg_price": float, "qty": float}}
            high_water_marks: Dict of {symbol: {"high_price": float, "entry_price": float}}

        Returns:
            List of exit signal dicts with action=SELL and exit reason.
        """
        instruments = set(self.get_instruments())
        exit_signals = []

        for symbol, pos_data in positions.items():
            # Only check our crypto instruments
            if symbol not in instruments:
                continue

            current_price = pos_data.get("current_price", 0)
            qty = pos_data.get("qty", 0)
            if current_price <= 0 or qty <= 0:
                continue

            basis = cost_bases.get(symbol, {})
            avg_entry = basis.get("avg_price", 0)
            if avg_entry <= 0:
                continue

            gain_from_entry = (current_price - avg_entry) / avg_entry

            # 1. Take-profit check
            if self.take_profit_enabled and gain_from_entry >= self.take_profit_pct:
                exit_signals.append({
                    "symbol": symbol,
                    "signal_type": "SELL",
                    "action": "SELL",
                    "reason": (
                        f"TAKE PROFIT: {symbol} up {gain_from_entry:.1%} "
                        f"(target {self.take_profit_pct:.0%}), "
                        f"entry=${avg_entry:.2f}, now=${current_price:.2f}"
                    ),
                    "exit_type": "take_profit",
                })
                logger.info(
                    f"Take-profit triggered: {symbol} up {gain_from_entry:.1%} from entry",
                    extra={"extra_data": {
                        "symbol": symbol,
                        "current_price": current_price,
                        "avg_entry": avg_entry,
                        "gain_pct": round(gain_from_entry, 4),
                        "target_pct": self.take_profit_pct,
                    }},
                )
                continue  # Don't double-trigger

            # 2. Hard stop check (unconditional — no arming needed)
            if self.hard_stop_enabled and gain_from_entry <= -self.hard_stop_pct:
                exit_signals.append({
                    "symbol": symbol,
                    "signal_type": "SELL",
                    "action": "SELL",
                    "reason": (
                        f"HARD STOP: {symbol} down {abs(gain_from_entry):.1%} "
                        f"(floor {self.hard_stop_pct:.0%}), "
                        f"entry=${avg_entry:.2f}, now=${current_price:.2f}"
                    ),
                    "exit_type": "hard_stop",
                })
                logger.warning(
                    f"Hard stop triggered: {symbol} down {abs(gain_from_entry):.1%} from entry",
                    extra={"extra_data": {
                        "symbol": symbol,
                        "current_price": current_price,
                        "avg_entry": avg_entry,
                        "loss_pct": round(abs(gain_from_entry), 4),
                        "stop_pct": self.hard_stop_pct,
                    }},
                )
                continue

            # 3. Trailing stop check (requires arming via min_gain)
            if self.trailing_stop_enabled:
                hw = high_water_marks.get(symbol, {})
                high_price = hw.get("high_price", 0)

                if high_price <= 0:
                    continue

                gain_from_entry_hw = (high_price - avg_entry) / avg_entry
                if gain_from_entry_hw < self.trailing_stop_min_gain:
                    continue  # Not armed yet

                drop_from_high = (high_price - current_price) / high_price
                if drop_from_high >= self.trailing_stop_pct:
                    exit_signals.append({
                        "symbol": symbol,
                        "signal_type": "SELL",
                        "action": "SELL",
                        "reason": (
                            f"TRAILING STOP: {symbol} dropped {drop_from_high:.1%} from "
                            f"peak ${high_price:.2f} (stop {self.trailing_stop_pct:.0%}), "
                            f"now=${current_price:.2f}"
                        ),
                        "exit_type": "trailing_stop",
                    })
                    logger.warning(
                        f"Trailing stop triggered: {symbol} dropped {drop_from_high:.1%} from peak",
                        extra={"extra_data": {
                            "symbol": symbol,
                            "current_price": current_price,
                            "high_price": high_price,
                            "drop_from_high": round(drop_from_high, 4),
                            "stop_pct": self.trailing_stop_pct,
                            "avg_entry": avg_entry,
                        }},
                    )

        if exit_signals:
            logger.info(
                f"Crypto exit signals: {len(exit_signals)} sell(s)",
                extra={"extra_data": {
                    "exits": [{"symbol": s["symbol"], "type": s["exit_type"]} for s in exit_signals],
                }},
            )
        else:
            logger.info("Crypto exit check: no exits triggered")

        return exit_signals

    def get_position_size(self, signal, portfolio_state):
        """
        Position size for DCA is the fixed dollar amount from the signal.

        Not percentage-based — just the flat dollar amount per cycle.
        """
        return signal.get("metadata", {}).get("dollar_amount", self.min_trade_dollars)
