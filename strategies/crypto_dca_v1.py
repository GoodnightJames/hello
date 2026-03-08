"""
Crypto Active Trading Strategy — v3.0

Buy side: Scores all coins by momentum + volatility, picks the best
candidate that we don't already hold a large position in. Buys the
coin most likely to hit the 5% take-profit target quickly.

Sell side: Every 15 min checks all open crypto positions for exit conditions:
1. Take-profit — sell when position is up X% from cost basis
2. Trailing stop — sell when price drops Y% from peak (after arming)
3. Hard stop — sell when down Z% from entry (unconditional)

Proceeds from sells recycle back to the crypto sleeve for re-deployment.
"""

import yaml
from core.logging import get_logger
from strategies.base import StrategyBase
from data.feature_store import get_price_history

logger = get_logger("strategies.crypto_dca_v1")


def load_strategy_config(config_path="config/strategies/crypto_dca_v1.yaml"):
    """Load crypto DCA strategy configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class CryptoDCAStrategy(StrategyBase):
    """Crypto active trading strategy with momentum-based selection and exits."""

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

    def _score_coins(self, instruments, held_symbols=None):
        """
        Score each coin by momentum + volatility to find the best buy.

        Scoring factors:
        1. Short-term momentum (1-day return) — coins moving up are more likely
           to continue and hit take-profit
        2. Volatility (avg daily range over 3 days) — higher volatility = faster
           moves = more likely to hit 5% target
        3. Position penalty — deprioritize coins we already hold to spread risk

        Returns:
            List of (symbol, score, reason) sorted by score descending.
        """
        if held_symbols is None:
            held_symbols = set()

        # Fetch recent price data (7 days is enough for our signals)
        prices = get_price_history(instruments, lookback_days=7)

        scores = []
        for symbol in instruments:
            if symbol not in prices.columns:
                # No price data — can still buy via live quote, give neutral score
                score = 0.5
                reason = "no history (neutral)"
                # But penalize if already held
                if symbol in held_symbols:
                    score *= 0.3
                    reason += ", already held"
                scores.append((symbol, score, reason))
                continue

            col = prices[symbol].dropna()
            if len(col) < 2:
                score = 0.5
                reason = "insufficient history (neutral)"
                if symbol in held_symbols:
                    score *= 0.3
                    reason += ", already held"
                scores.append((symbol, score, reason))
                continue

            # 1-day return (momentum)
            ret_1d = (col.iloc[-1] - col.iloc[-2]) / col.iloc[-2]

            # 3-day return if available
            if len(col) >= 4:
                ret_3d = (col.iloc[-1] - col.iloc[-4]) / col.iloc[-4]
            else:
                ret_3d = ret_1d

            # Volatility: average daily percentage range over last 3 days
            # Use price swings as a proxy (close-to-close absolute changes)
            recent = col.tail(4)
            daily_changes = recent.pct_change().dropna().abs()
            avg_volatility = daily_changes.mean() if len(daily_changes) > 0 else 0.02

            # Score = momentum component + volatility component
            # Positive momentum is good (coin is moving up)
            # High volatility is good (coin moves enough to hit 5% target)
            momentum_score = ret_1d * 2.0 + ret_3d * 1.0  # weight recent more
            volatility_score = avg_volatility * 10.0  # scale up: 5% daily vol = 0.5

            score = momentum_score + volatility_score

            # Penalize coins we already hold — spread the bets
            if symbol in held_symbols:
                score *= 0.3
                reason = (
                    f"1d:{ret_1d:+.1%} 3d:{ret_3d:+.1%} vol:{avg_volatility:.1%} "
                    f"[HELD, penalized]"
                )
            else:
                reason = f"1d:{ret_1d:+.1%} 3d:{ret_3d:+.1%} vol:{avg_volatility:.1%}"

            scores.append((symbol, score, reason))

        # Sort by score descending — best candidate first
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def generate_signals(self, data=None, exit_log=None, held_symbols=None):
        """
        Generate ONE buy signal for the highest-scoring coin.

        Instead of blind rotation, scores all coins by momentum + volatility
        and picks the one most likely to hit the 5% take-profit target.
        Deprioritizes coins we already hold to spread risk.

        Args:
            data: Unused (kept for interface compatibility).
            exit_log: Unused.
            held_symbols: Set of symbols we currently hold positions in.

        Returns:
            List with one signal dict (BUY for the best-scoring coin).
        """
        instruments = self.get_instruments()
        if not instruments:
            logger.warning("No instruments configured for crypto strategy")
            return []

        if held_symbols is None:
            held_symbols = set()

        # Score all coins and pick the best one
        scored = self._score_coins(instruments, held_symbols)
        dollar_amount = self.dollars_per_cycle

        # Log all scores for transparency
        for symbol, score, reason in scored:
            logger.info(f"  Coin score: {symbol} = {score:.3f} ({reason})")

        best_symbol, best_score, best_reason = scored[0]

        logger.info(
            f"Momentum pick: {best_symbol} (score {best_score:.3f}) — {best_reason}, "
            f"${dollar_amount:.2f}",
        )

        if dollar_amount < self.min_trade_dollars:
            logger.warning(
                f"Skip {best_symbol}: ${dollar_amount:.2f} below "
                f"${self.min_trade_dollars:.2f} minimum",
            )
            return []

        signals = [{
            "symbol": best_symbol,
            "signal_type": "BUY",
            "score": best_score,
            "signal_strength": max(0.0, min(1.0, best_score)),
            "metadata": {
                "strategy": "crypto_momentum",
                "dollar_amount": dollar_amount,
                "momentum_score": best_score,
                "reason": f"Momentum pick: {best_reason}",
            },
        }]

        logger.info(
            "Signal generation complete",
            extra={
                "extra_data": {
                    "strategy": self.name,
                    "signal_count": len(signals),
                    "total_deploy": dollar_amount,
                    "selected_coin": best_symbol,
                    "score": best_score,
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
        Position size is the fixed dollar amount from the signal.
        """
        return signal.get("metadata", {}).get("dollar_amount", self.min_trade_dollars)
