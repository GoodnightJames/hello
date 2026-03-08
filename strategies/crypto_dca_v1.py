"""
Crypto Active Trading Strategy — v6.0

CHANGES from v5:
- Regime-adaptive thresholds (tighter entry in corrections/crisis)
- Time-decay exits (cut stale trades after 12h without progress)
- Momentum-collapse exits (cut when momentum sharply reverses)
- Sleeve health meta-layer integration (reduce aggression when out of form)

Core architecture (v5):
- Z-score based momentum with vol/overextension penalties
- Two-gate entry: rank threshold + cost gate (edge_ratio >= 1.5x)
- ATR-scaled exits (take-profit, trailing stop, hard stop)
- Signal threshold — no forced buys. Cash is a position.

Exit hierarchy (checked in order):
1. Take-profit — ATR-scaled (harvest winners)
2. Hard stop — ATR-scaled (cut losers)
3. Time decay — exit stale trades (12h without progress)
4. Momentum collapse — exit when thesis breaks
5. Trailing stop — ATR-scaled (protect profits)
"""

import numpy as np
import yaml
from core.logging import get_logger
from strategies.base import StrategyBase
from data.feature_store import get_price_history
from risk.cost_model import compute_cost_hurdle, estimate_round_trip_cost

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

        # Regime-adaptive thresholds: tighten entry requirements in worse regimes.
        # In trending markets, use base threshold. In corrections/crisis, require
        # much stronger signals — one good trade beats five mediocre ones.
        regime_cfg = config.get("regime_overrides", {})
        self.regime_threshold_adjustments = regime_cfg.get("threshold_adjustments", {
            "trending": 0.0,        # Normal: use base threshold
            "ranging": 0.05,        # Add 0.05 in mixed conditions
            "correction": 0.10,     # Add 0.10 in corrections
            "crisis": 0.20,         # Add 0.20 in crisis — very selective
        })
        self.regime_edge_ratio_adjustments = regime_cfg.get("edge_ratio_adjustments", {
            "trending": 0.0,        # Normal: use base min_edge_ratio
            "ranging": 0.25,        # Need 1.75x instead of 1.5x
            "correction": 0.50,     # Need 2.0x instead of 1.5x
            "crisis": 1.0,          # Need 2.5x instead of 1.5x
        })

        # Time-based trade decay: exit stale trades that haven't progressed.
        # If a trade hasn't moved enough after N hours, the thesis is weak.
        decay_cfg = config.get("trade_decay", {})
        self.time_decay_enabled = decay_cfg.get("enabled", True)
        self.stale_hours = decay_cfg.get("stale_hours", 12)
        self.stale_min_progress_pct = decay_cfg.get("min_progress_pct", 0.005)
        self.momentum_collapse_exit = decay_cfg.get("momentum_collapse_exit", True)

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
        Score each coin by risk-adjusted momentum. Separate ranking from entry gate.

        RANKING (z-score space — determines WHICH coin):
            momentum   = 0.45*z(12h) + 0.35*z(1d) + 0.20*z(3d)
            vol_pen    = -0.35 * z(realized_vol)
            ext_pen    = -0.25 * z(distance_above_ema20)
            rank_score = momentum - vol_pen - ext_pen
            (position penalty applied multiplicatively if already held)

        ENTRY GATE (return space — determines WHETHER to buy):
            expected_edge = weighted average of raw returns (same weights)
            cost_hurdle   = estimated round-trip cost as decimal
            passes_cost_gate = expected_edge > cost_hurdle

        This fixes the unit mismatch: z-scores rank coins against each other,
        but the cost check uses real return vs real cost in the same units.

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

            # RANKING: z-score composite (dimensionless, for relative ordering)
            momentum = (
                self.weight_12h * z_12h.get(symbol, 0)
                + self.weight_1d * z_1d.get(symbol, 0)
                + self.weight_3d * z_3d.get(symbol, 0)
            )
            vol_pen = self.vol_penalty_weight * z_vol.get(symbol, 0)
            ext_pen = self.overextension_penalty_weight * z_ext.get(symbol, 0)
            rank_score = momentum - vol_pen - ext_pen

            # Position penalty — deprioritize coins already held
            if symbol in held_symbols:
                rank_score *= self.position_penalty
                held_flag = " [HELD]"
            else:
                held_flag = ""

            # ENTRY GATE: expected forward edge vs cost, both in return space.
            #
            # Why not just use raw past returns as expected edge:
            # 1. ret_12h is ret_1d*0.6 — double-counts the same data
            # 2. Raw returns treat "BTC went up 3% today" as "I expect 3% forward"
            # 3. Mixes time horizons (1d vs 3d) without normalization
            #
            # Better approach: ATR-calibrated momentum continuation estimate.
            # If momentum is positive and coin is not overextended, estimate
            # a conservative forward move as a fraction of ATR — which is in
            # the same return-space units as the cost hurdle.
            #
            # expected_forward = momentum_direction * atr * continuation_fraction
            # where continuation_fraction reflects how much of one ATR move
            # we conservatively expect to capture in the next holding period.
            vol_for_cost = volatilities.get(symbol, 0.02)
            atr_pct = atrs.get(symbol, 0.03)

            # Annualize 1d return for direction, normalize by vol to get
            # momentum strength in "number of daily vol moves" — a dimensionless
            # signal that converts cleanly to expected forward return.
            ret_1d_sym = returns_1d.get(symbol, 0)
            ret_3d_sym = returns_3d.get(symbol, 0)
            daily_vol = volatilities.get(symbol, 0.02)

            if daily_vol > 1e-6:
                # Momentum strength: how many daily vol units is the move?
                # Blends 1d and 3d (3d annualized to daily) for stability.
                mom_strength_1d = ret_1d_sym / daily_vol
                mom_strength_3d = (ret_3d_sym / 3.0) / daily_vol  # Per-day 3d return
                mom_strength = 0.6 * mom_strength_1d + 0.4 * mom_strength_3d
            else:
                mom_strength = 0.0

            # Expected forward move: momentum_strength * ATR * continuation_frac.
            # continuation_fraction = 0.3 is conservative: we expect to capture
            # ~30% of an ATR move in the next holding period. This is deliberately
            # pessimistic — if a strategy can't clear costs at 30% capture, it
            # shouldn't be trading.
            continuation_fraction = self.config.get("scoring", {}).get(
                "continuation_fraction", 0.30
            )
            expected_edge = max(0.0, mom_strength * atr_pct * continuation_fraction)

            cost_hurdle_val = compute_cost_hurdle(symbol, volatility=vol_for_cost)

            # Require edge_ratio >= min_edge_ratio (default 1.5x cost).
            # A tiny edge above cost is not robust enough for live execution.
            min_edge_ratio = self.config.get("scoring", {}).get(
                "min_edge_ratio", 1.5
            )
            edge_ratio = (expected_edge / cost_hurdle_val) if cost_hurdle_val > 0 else 0.0
            passes_cost_gate = edge_ratio >= min_edge_ratio

            # ATR clamp diagnostics — check if clamps are dominating
            tp_raw = atr_pct * self.take_profit_atr_mult
            tp_clamped = self._clamp(tp_raw, self.take_profit_min_pct, self.take_profit_max_pct)
            tp_was_clamped = abs(tp_raw - tp_clamped) > 1e-6
            stop_raw = atr_pct * self.hard_stop_atr_mult
            stop_clamped = self._clamp(stop_raw, self.hard_stop_min_pct, self.hard_stop_max_pct)
            stop_was_clamped = abs(stop_raw - stop_clamped) > 1e-6

            # Get cost breakdown for logging
            cost_info = estimate_round_trip_cost(
                symbol, self.dollars_per_cycle, vol_for_cost,
            )

            diagnostics = {
                "z_12h": round(z_12h.get(symbol, 0), 3),
                "z_1d": round(z_1d.get(symbol, 0), 3),
                "z_3d": round(z_3d.get(symbol, 0), 3),
                "z_vol": round(z_vol.get(symbol, 0), 3),
                "z_ext": round(z_ext.get(symbol, 0), 3),
                "momentum": round(momentum, 4),
                "vol_penalty": round(vol_pen, 4),
                "ext_penalty": round(ext_pen, 4),
                "rank_score": round(rank_score, 4),
                "mom_strength": round(mom_strength, 4),
                "expected_edge": round(expected_edge, 6),
                "cost_hurdle": round(cost_hurdle_val, 6),
                "edge_ratio": round(edge_ratio, 3),
                "min_edge_ratio": min_edge_ratio,
                "passes_cost_gate": passes_cost_gate,
                "continuation_fraction": continuation_fraction,
                "cost_bps": cost_info["total_bps"],
                "atr_pct": round(atr_pct, 4),
                "tp_raw": round(tp_raw, 4),
                "tp_clamped": round(tp_clamped, 4),
                "tp_was_clamped": tp_was_clamped,
                "stop_raw": round(stop_raw, 4),
                "stop_clamped": round(stop_clamped, 4),
                "stop_was_clamped": stop_was_clamped,
                "ret_1d": round(ret_1d_sym, 4),
                "vol": round(daily_vol, 4),
            }

            reason = (
                f"rank={rank_score:+.3f} edge={expected_edge:.4f} "
                f"hurdle={cost_hurdle_val:.4f} ratio={edge_ratio:.1f}x "
                f"cost={cost_info['total_bps']:.0f}bps"
                f"{' COST_FAIL' if not passes_cost_gate else ''}{held_flag}"
            )

            scores.append((symbol, rank_score, reason, diagnostics))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _get_regime_adjusted_thresholds(self, regime=None):
        """
        Adjust entry thresholds based on market regime.

        In trending markets: use base thresholds (normal trading).
        In corrections/crisis: require much stronger signals.
        "Do nothing" bias when conditions are mixed.

        Args:
            regime: Dict from tag_current_regime() with "phase" key.

        Returns:
            (adjusted_threshold, adjusted_min_edge_ratio)
        """
        if regime is None:
            return self.min_score_threshold, self.config.get("scoring", {}).get(
                "min_edge_ratio", 1.5
            )

        phase = regime.get("phase", "trending")

        threshold_bump = self.regime_threshold_adjustments.get(phase, 0.0)
        edge_ratio_bump = self.regime_edge_ratio_adjustments.get(phase, 0.0)

        base_threshold = self.min_score_threshold
        base_edge_ratio = self.config.get("scoring", {}).get("min_edge_ratio", 1.5)

        adjusted_threshold = base_threshold + threshold_bump
        adjusted_edge_ratio = base_edge_ratio + edge_ratio_bump

        if threshold_bump > 0 or edge_ratio_bump > 0:
            logger.info(
                f"Regime [{phase}]: threshold {base_threshold:.2f} → "
                f"{adjusted_threshold:.2f}, edge_ratio {base_edge_ratio:.1f}x → "
                f"{adjusted_edge_ratio:.1f}x",
            )

        return adjusted_threshold, adjusted_edge_ratio

    def generate_signals(self, data=None, exit_log=None, held_symbols=None,
                         regime=None):
        """
        Generate a buy signal using two independent gates with regime adaptation.

        Gate 1 (ranking): z-score rank_score >= regime-adjusted threshold
        Gate 2 (cost): edge_ratio >= regime-adjusted min_edge_ratio

        In worse regimes, both gates tighten — fewer trades, higher conviction.
        Both gates must pass. Cash is a position.

        Args:
            regime: Dict from tag_current_regime(). If provided, thresholds
                    adjust based on market phase (trending/ranging/correction/crisis).
        """
        instruments = self.get_instruments()
        if not instruments:
            logger.warning("No instruments configured for crypto strategy")
            return []

        if held_symbols is None:
            held_symbols = set()

        # Get regime-adjusted thresholds
        effective_threshold, effective_edge_ratio = (
            self._get_regime_adjusted_thresholds(regime)
        )

        scored = self._score_coins(instruments, held_symbols)
        dollar_amount = self.dollars_per_cycle

        for symbol, score, reason, diag in scored:
            logger.info(f"  Coin score: {symbol} = {score:.4f} ({reason})")

        # Gate 1: ranking threshold (z-score space, regime-adjusted)
        if not scored or scored[0][1] < effective_threshold:
            best_sym = scored[0][0] if scored else "none"
            best_score = scored[0][1] if scored else 0.0
            logger.info(
                f"NO BUY: Best rank_score {best_sym}={best_score:.4f} below "
                f"threshold {effective_threshold:.2f}. Holding cash.",
            )
            return []

        # Gate 2: cost hurdle (return space, regime-adjusted edge ratio)
        best_entry = None
        for symbol, score, reason, diag in scored:
            if score < effective_threshold:
                break  # Sorted descending; all remaining below threshold
            # Use regime-adjusted edge ratio instead of the base one
            symbol_edge_ratio = diag.get("edge_ratio", 0)
            if symbol_edge_ratio >= effective_edge_ratio:
                best_entry = (symbol, score, reason, diag)
                break
            else:
                logger.info(
                    f"COST GATE FAIL: {symbol} rank={score:.4f} "
                    f"edge_ratio={symbol_edge_ratio:.1f}x < "
                    f"required={effective_edge_ratio:.1f}x",
                )

        if best_entry is None:
            logger.info(
                "NO BUY: No coin passes both rank threshold and cost hurdle. "
                "Holding cash.",
            )
            return []

        best_symbol, best_score, best_reason, best_diag = best_entry

        logger.info(
            f"Momentum pick: {best_symbol} (rank={best_score:.4f} "
            f"edge={best_diag.get('expected_edge', 0):.4f}) — {best_reason}, "
            f"${dollar_amount:.2f}",
        )

        if dollar_amount < self.min_trade_dollars:
            logger.warning(
                f"Skip {best_symbol}: ${dollar_amount:.2f} below "
                f"${self.min_trade_dollars:.2f} minimum",
            )
            return []

        phase = regime.get("phase", "unknown") if regime else "unknown"

        signals = [{
            "symbol": best_symbol,
            "signal_type": "BUY",
            "score": best_score,
            "signal_strength": max(0.0, min(1.0, best_score)),
            "metadata": {
                "strategy": "crypto_momentum_v5",
                "dollar_amount": dollar_amount,
                "rank_score": best_score,
                "expected_edge": best_diag.get("expected_edge", 0),
                "cost_hurdle": best_diag.get("cost_hurdle", 0),
                "diagnostics": best_diag,
                "reason": f"Momentum pick: {best_reason}",
                "rank_threshold": effective_threshold,
                "effective_edge_ratio": effective_edge_ratio,
                "regime_phase": phase,
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
                    "rank_score": best_score,
                    "expected_edge": best_diag.get("expected_edge"),
                    "cost_hurdle": best_diag.get("cost_hurdle"),
                    "rank_threshold": effective_threshold,
                    "effective_edge_ratio": effective_edge_ratio,
                    "regime_phase": phase,
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

    def generate_exit_signals(self, positions, cost_bases, high_water_marks,
                              entry_times=None):
        """
        Check all crypto positions for exit conditions.

        Exit hierarchy (checked in order, first match wins):
        1. Take-profit — ATR-scaled (harvest winners)
        2. Hard stop — ATR-scaled (cut losers unconditionally)
        3. Time decay — exit stale trades that haven't progressed
        4. Momentum collapse — exit if momentum sharply reverses after entry
        5. Trailing stop — ATR-scaled (protect profits in winning trades)

        Args:
            entry_times: Dict of {symbol: entry_datetime} for time-decay checks.
        """
        from datetime import datetime, timezone

        instruments = set(self.get_instruments())
        exit_signals = []
        if entry_times is None:
            entry_times = {}

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

            # 3. Time decay — exit stale trades that haven't made progress.
            # If a trade hasn't moved enough after N hours, the thesis is weak.
            # This prevents capital getting trapped in mediocre setups.
            if self.time_decay_enabled and symbol in entry_times:
                entry_time = entry_times[symbol]
                now = datetime.now(timezone.utc)
                if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo is None:
                    from datetime import timezone as tz
                    entry_time = entry_time.replace(tzinfo=tz.utc)
                hours_held = (now - entry_time).total_seconds() / 3600.0

                if hours_held >= self.stale_hours:
                    if abs(gain_from_entry) < self.stale_min_progress_pct:
                        exit_signals.append({
                            "symbol": symbol,
                            "signal_type": "SELL",
                            "action": "SELL",
                            "reason": (
                                f"TIME DECAY: {symbol} held {hours_held:.0f}h with only "
                                f"{gain_from_entry:+.2%} progress "
                                f"(need >{self.stale_min_progress_pct:.1%} by "
                                f"{self.stale_hours}h), freeing capital"
                            ),
                            "exit_type": "time_decay",
                        })
                        logger.info(
                            f"Time decay triggered: {symbol} {hours_held:.0f}h "
                            f"stale ({gain_from_entry:+.2%})",
                            extra={"extra_data": {
                                "symbol": symbol, "hours_held": round(hours_held, 1),
                                "gain_pct": round(gain_from_entry, 4),
                                "min_progress": self.stale_min_progress_pct,
                            }},
                        )
                        continue

            # 4. Momentum collapse — if momentum score has sharply reversed
            # since entry, the thesis is broken even if stops haven't triggered.
            if self.momentum_collapse_exit:
                try:
                    current_scores = self._score_coins([symbol])
                    if current_scores:
                        _, current_score, _, current_diag = current_scores[0]
                        mom_strength_now = current_diag.get("mom_strength", 0)
                        # If momentum has flipped strongly negative, exit early
                        if mom_strength_now < -1.0 and gain_from_entry < 0:
                            exit_signals.append({
                                "symbol": symbol,
                                "signal_type": "SELL",
                                "action": "SELL",
                                "reason": (
                                    f"MOMENTUM COLLAPSE: {symbol} momentum={mom_strength_now:.1f} "
                                    f"(strongly negative) while underwater "
                                    f"({gain_from_entry:+.1%})"
                                ),
                                "exit_type": "momentum_collapse",
                            })
                            logger.warning(
                                f"Momentum collapse: {symbol} mom_strength="
                                f"{mom_strength_now:.1f}, gain={gain_from_entry:+.1%}",
                                extra={"extra_data": {
                                    "symbol": symbol,
                                    "mom_strength": round(mom_strength_now, 2),
                                    "gain_pct": round(gain_from_entry, 4),
                                }},
                            )
                            continue
                except Exception:
                    pass  # Non-critical — skip if scoring fails

            # 5. Trailing stop (ATR-scaled, requires arming)
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
