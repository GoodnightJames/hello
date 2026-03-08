"""
Crypto Active Trading Strategy — v5.0

MAJOR CHANGES from v4:
- Z-score based momentum (not raw returns) with vol penalty
- Overextension penalty (distance above 20-period EMA)
- Cost-adjusted scoring (only buy when edge > fees)
- Signal threshold — no forced buys. Cash is a position.
- ATR/volatility-scaled exits instead of flat percentages
- Meme coins treated as speculative sub-sleeve

Buy side: Scores all coins by risk-adjusted momentum, penalizes
volatility and overextension, subtracts transaction cost estimate.
Only buys if best score exceeds threshold.

Sell side: Every 15 min checks all open crypto positions for exit conditions:
1. Take-profit — ATR-scaled (1.5-2x ATR from entry)
2. Trailing stop — ATR-scaled (activates after +1 ATR gain)
3. Hard stop — ATR-scaled (1.0-1.25x ATR below entry)

Proceeds from sells recycle back to the crypto sleeve for re-deployment.
"""

import numpy as np
import yaml
from core.logging import get_logger
from strategies.base import StrategyBase
from data.feature_store import get_price_history
from risk.cost_model import apply_cost_penalty_to_score

logger = get_logger("strategies.crypto_dca_v1")


def load_strategy_config(config_path="config/strategies/crypto_dca_v1.yaml"):
    """Load crypto DCA strategy configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class CryptoDCAStrategy(StrategyBase):
    """Crypto active trading strategy with risk-adjusted momentum scoring."""

    def __init__(self, config=None):
        if config is None:
            config = load_strategy_config()
        super().__init__(config)
        self.dollars_per_cycle = config.get("allocation", {}).get("dollars_per_cycle", 2.0)
        self.min_trade_dollars = config.get("allocation", {}).get("min_trade_dollars", 1.0)
        self.sleeve_pct = config.get("allocation", {}).get("sleeve_pct", 0.30)

        # Signal threshold — minimum score to trigger a buy
        scoring = config.get("scoring", {})
        self.min_score_threshold = scoring.get("min_score_threshold", 0.10)

        # Momentum weights (z-score based)
        self.weight_12h = scoring.get("weight_12h", 0.45)
        self.weight_1d = scoring.get("weight_1d", 0.35)
        self.weight_3d = scoring.get("weight_3d", 0.20)

        # Penalty weights
        self.vol_penalty_weight = scoring.get("vol_penalty_weight", 0.35)
        self.overextension_penalty_weight = scoring.get("overextension_penalty_weight", 0.25)
        self.position_penalty = scoring.get("position_penalty", 0.30)

        # EMA period for overextension check
        self.ema_period = scoring.get("ema_period", 20)

        # Exit configuration — ATR-scaled
        exits = config.get("exits", {})

        tp = exits.get("take_profit", {})
        self.take_profit_enabled = tp.get("enabled", False)
        self.take_profit_atr_mult = tp.get("atr_multiplier", 2.0)
        self.take_profit_min_pct = tp.get("min_pct", 0.03)
        self.take_profit_max_pct = tp.get("max_pct", 0.12)

        ts = exits.get("trailing_stop", {})
        self.trailing_stop_enabled = ts.get("enabled", False)
        self.trailing_stop_atr_mult = ts.get("atr_multiplier", 1.5)
        self.trailing_stop_activation_atr = ts.get("activation_atr_mult", 1.0)

        hs = exits.get("hard_stop", {})
        self.hard_stop_enabled = hs.get("enabled", False)
        self.hard_stop_atr_mult = hs.get("atr_multiplier", 1.25)
        self.hard_stop_min_pct = hs.get("min_pct", 0.03)
        self.hard_stop_max_pct = hs.get("max_pct", 0.10)

        # ATR lookback
        self.atr_period = exits.get("atr_period", 14)

    def get_instruments(self):
        """Get all crypto instruments this strategy trades."""
        assets = self.config.get("instruments", {}).get("assets", [])
        if assets and isinstance(assets[0], dict):
            return [a["symbol"] for a in assets]
        return assets

    def _compute_z_scores(self, series_dict):
        """
        Compute cross-sectional z-scores for a dict of {symbol: value}.

        Z-scoring normalizes across the universe so we compare apples-to-apples.
        """
        values = list(series_dict.values())
        if len(values) < 2:
            return {k: 0.0 for k in series_dict}

        mean = np.mean(values)
        std = np.std(values)
        if std < 1e-10:
            return {k: 0.0 for k in series_dict}

        return {k: (v - mean) / std for k, v in series_dict.items()}

    def _compute_atr(self, prices_series, period=14):
        """
        Compute Average True Range from close prices (close-to-close proxy).

        Returns ATR as a percentage of current price.
        """
        if len(prices_series) < period + 1:
            if len(prices_series) < 3:
                return 0.03
            changes = prices_series.pct_change().dropna().abs()
            return float(changes.mean()) if len(changes) > 0 else 0.03

        changes = prices_series.pct_change().dropna().tail(period).abs()
        return float(changes.mean()) if len(changes) > 0 else 0.03

    def _score_coins(self, instruments, held_symbols=None):
        """
        Score each coin by risk-adjusted momentum with cost penalty.

        Scoring formula (z-score based):
            raw_momentum = 0.45*z(12h_ret) + 0.35*z(1d_ret) + 0.20*z(3d_ret)
            vol_penalty  = -0.35 * z(realized_vol)
            ext_penalty  = -0.25 * z(distance_above_ema20)
            raw_score    = raw_momentum + vol_penalty + ext_penalty
            adjusted     = raw_score - round_trip_cost

        Returns:
            List of (symbol, score, reason, diagnostics) sorted descending.
        """
        if held_symbols is None:
            held_symbols = set()

        prices = get_price_history(instruments, lookback_days=30)

        scores = []
        returns_12h = {}
        returns_1d = {}
        returns_3d = {}
        volatilities = {}
        overextensions = {}
        atrs = {}

        for symbol in instruments:
            if symbol not in prices.columns:
                scores.append((symbol, 0.0, "no history (skip)", {}))
                continue

            col = prices[symbol].dropna()
            if len(col) < 4:
                scores.append((symbol, 0.0, "insufficient history (skip)", {}))
                continue

            ret_1d = (col.iloc[-1] - col.iloc[-2]) / col.iloc[-2]
            returns_1d[symbol] = ret_1d
            returns_12h[symbol] = ret_1d * 0.6  # Proxy from daily bar

            if len(col) >= 4:
                ret_3d = (col.iloc[-1] - col.iloc[-4]) / col.iloc[-4]
            else:
                ret_3d = ret_1d
            returns_3d[symbol] = ret_3d

            recent = col.tail(min(8, len(col)))
            daily_changes = recent.pct_change().dropna().abs()
            vol = float(daily_changes.mean()) if len(daily_changes) > 0 else 0.02
            volatilities[symbol] = vol

            ema_len = min(self.ema_period, len(col))
            if ema_len >= 3:
                ema = col.ewm(span=ema_len, adjust=False).mean()
                dist_from_ema = (col.iloc[-1] - ema.iloc[-1]) / ema.iloc[-1]
            else:
                dist_from_ema = 0.0
            overextensions[symbol] = dist_from_ema

            atrs[symbol] = self._compute_atr(col, self.atr_period)

        # Z-score all metrics cross-sectionally
        z_12h = self._compute_z_scores(returns_12h)
        z_1d = self._compute_z_scores(returns_1d)
        z_3d = self._compute_z_scores(returns_3d)
        z_vol = self._compute_z_scores(volatilities)
        z_ext = self._compute_z_scores(overextensions)

        for symbol in instruments:
            if symbol not in z_1d:
                continue

            momentum = (
                self.weight_12h * z_12h.get(symbol, 0)
                + self.weight_1d * z_1d.get(symbol, 0)
                + self.weight_3d * z_3d.get(symbol, 0)
            )
            vol_pen = self.vol_penalty_weight * z_vol.get(symbol, 0)
            ext_pen = self.overextension_penalty_weight * z_ext.get(symbol, 0)
            raw_score = momentum - vol_pen - ext_pen

            vol_for_cost = volatilities.get(symbol, 0.02)
            adjusted_score, cost_info = apply_cost_penalty_to_score(
                symbol, raw_score, volatility=vol_for_cost, notional=self.dollars_per_cycle
            )

            if symbol in held_symbols:
                adjusted_score *= self.position_penalty
                held_flag = " [HELD]"
            else:
                held_flag = ""

            diagnostics = {
                "z_12h": round(z_12h.get(symbol, 0), 3),
                "z_1d": round(z_1d.get(symbol, 0), 3),
                "z_3d": round(z_3d.get(symbol, 0), 3),
                "z_vol": round(z_vol.get(symbol, 0), 3),
                "z_ext": round(z_ext.get(symbol, 0), 3),
                "momentum": round(momentum, 4),
                "vol_penalty": round(vol_pen, 4),
                "ext_penalty": round(ext_pen, 4),
                "raw_score": round(raw_score, 4),
                "cost_bps": cost_info["total_bps"],
                "atr_pct": round(atrs.get(symbol, 0.03), 4),
                "ret_1d": round(returns_1d.get(symbol, 0), 4),
                "vol": round(volatilities.get(symbol, 0), 4),
            }

            reason = (
                f"mom={momentum:+.3f} vol_pen={vol_pen:.3f} ext_pen={ext_pen:.3f} "
                f"cost={cost_info['total_bps']:.0f}bps{held_flag}"
            )

            scores.append((symbol, adjusted_score, reason, diagnostics))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def generate_signals(self, data=None, exit_log=None, held_symbols=None):
        """
        Generate a buy signal ONLY if the best coin's score exceeds threshold.

        Key change from v4: cash is a position. If no coin scores above
        min_score_threshold after cost adjustment, we do nothing.
        """
        instruments = self.get_instruments()
        if not instruments:
            logger.warning("No instruments configured for crypto strategy")
            return []

        if held_symbols is None:
            held_symbols = set()

        scored = self._score_coins(instruments, held_symbols)
        dollar_amount = self.dollars_per_cycle

        for symbol, score, reason, diag in scored:
            logger.info(f"  Coin score: {symbol} = {score:.4f} ({reason})")

        # Threshold check — no forced buys
        if not scored or scored[0][1] < self.min_score_threshold:
            best_sym = scored[0][0] if scored else "none"
            best_score = scored[0][1] if scored else 0.0
            logger.info(
                f"NO BUY: Best score {best_sym}={best_score:.4f} below "
                f"threshold {self.min_score_threshold}. Holding cash.",
            )
            return []

        best_symbol, best_score, best_reason, best_diag = scored[0]

        logger.info(
            f"Momentum pick: {best_symbol} (score {best_score:.4f}) — {best_reason}, "
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
                "strategy": "crypto_momentum_v5",
                "dollar_amount": dollar_amount,
                "momentum_score": best_score,
                "diagnostics": best_diag,
                "reason": f"Momentum pick: {best_reason}",
                "threshold": self.min_score_threshold,
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
                    "threshold": self.min_score_threshold,
                    "all_scores": {s[0]: round(s[1], 4) for s in scored},
                }
            },
        )
        return signals

    def _get_atr_for_symbol(self, symbol):
        """Get ATR percentage for a symbol from recent price data."""
        try:
            prices = get_price_history([symbol], lookback_days=30)
            if symbol not in prices.columns:
                return 0.03
            col = prices[symbol].dropna()
            return self._compute_atr(col, self.atr_period)
        except Exception:
            return 0.03

    def _clamp(self, value, min_val, max_val):
        """Clamp value between min and max."""
        return max(min_val, min(max_val, value))

    def generate_exit_signals(self, positions, cost_bases, high_water_marks):
        """
        Check all crypto positions for ATR-scaled exit conditions.

        ATR-scaled exits adapt to each coin's actual volatility:
        - BTC with 2% ATR -> take profit at ~4%, stop at ~2.5%
        - PEPE with 8% ATR -> take profit at ~16%, stop at ~10%
        """
        instruments = set(self.get_instruments())
        exit_signals = []

        for symbol, pos_data in positions.items():
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
            atr_pct = self._get_atr_for_symbol(symbol)

            # 1. Take-profit (ATR-scaled)
            if self.take_profit_enabled:
                tp_target = self._clamp(
                    atr_pct * self.take_profit_atr_mult,
                    self.take_profit_min_pct,
                    self.take_profit_max_pct,
                )
                if gain_from_entry >= tp_target:
                    exit_signals.append({
                        "symbol": symbol,
                        "signal_type": "SELL",
                        "action": "SELL",
                        "reason": (
                            f"TAKE PROFIT: {symbol} up {gain_from_entry:.1%} "
                            f"(ATR-target {tp_target:.1%}, ATR={atr_pct:.1%}), "
                            f"entry=${avg_entry:.2f}, now=${current_price:.2f}"
                        ),
                        "exit_type": "take_profit",
                    })
                    logger.info(
                        f"Take-profit triggered: {symbol} up {gain_from_entry:.1%} "
                        f"(ATR-target={tp_target:.1%})",
                        extra={"extra_data": {
                            "symbol": symbol, "gain_pct": round(gain_from_entry, 4),
                            "target_pct": tp_target, "atr_pct": atr_pct,
                        }},
                    )
                    continue

            # 2. Hard stop (ATR-scaled, unconditional)
            if self.hard_stop_enabled:
                stop_level = self._clamp(
                    atr_pct * self.hard_stop_atr_mult,
                    self.hard_stop_min_pct,
                    self.hard_stop_max_pct,
                )
                if gain_from_entry <= -stop_level:
                    exit_signals.append({
                        "symbol": symbol,
                        "signal_type": "SELL",
                        "action": "SELL",
                        "reason": (
                            f"HARD STOP: {symbol} down {abs(gain_from_entry):.1%} "
                            f"(ATR-floor {stop_level:.1%}, ATR={atr_pct:.1%}), "
                            f"entry=${avg_entry:.2f}, now=${current_price:.2f}"
                        ),
                        "exit_type": "hard_stop",
                    })
                    logger.warning(
                        f"Hard stop triggered: {symbol} down {abs(gain_from_entry):.1%} "
                        f"(ATR-floor={stop_level:.1%})",
                        extra={"extra_data": {
                            "symbol": symbol, "loss_pct": round(abs(gain_from_entry), 4),
                            "stop_pct": stop_level, "atr_pct": atr_pct,
                        }},
                    )
                    continue

            # 3. Trailing stop (ATR-scaled, requires arming)
            if self.trailing_stop_enabled:
                hw = high_water_marks.get(symbol, {})
                high_price = hw.get("high_price", 0)
                if high_price <= 0:
                    continue

                activation_gain = atr_pct * self.trailing_stop_activation_atr
                gain_from_entry_hw = (high_price - avg_entry) / avg_entry
                if gain_from_entry_hw < activation_gain:
                    continue

                trail_pct = atr_pct * self.trailing_stop_atr_mult
                drop_from_high = (high_price - current_price) / high_price
                if drop_from_high >= trail_pct:
                    exit_signals.append({
                        "symbol": symbol,
                        "signal_type": "SELL",
                        "action": "SELL",
                        "reason": (
                            f"TRAILING STOP: {symbol} dropped {drop_from_high:.1%} from "
                            f"peak ${high_price:.2f} (ATR-trail {trail_pct:.1%}), "
                            f"now=${current_price:.2f}"
                        ),
                        "exit_type": "trailing_stop",
                    })
                    logger.warning(
                        f"Trailing stop triggered: {symbol} dropped {drop_from_high:.1%} "
                        f"from peak (ATR-trail={trail_pct:.1%})",
                        extra={"extra_data": {
                            "symbol": symbol, "drop_from_high": round(drop_from_high, 4),
                            "trail_pct": trail_pct, "atr_pct": atr_pct,
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
        """Position size is the fixed dollar amount from the signal."""
        return signal.get("metadata", {}).get("dollar_amount", self.min_trade_dollars)
