"""Tests for the dual momentum scorer with alpha optimizations."""

import numpy as np
import pandas as pd
import pytest
import yaml

from research.momentum_scorer import (
    compute_absolute_momentum,
    compute_relative_momentum,
    check_multi_timeframe,
    check_minimum_edge,
    check_rsi_filter,
    check_fast_exit,
    is_in_cooldown,
    run_daily_exit_scan,
    compute_volatility_scalar,
    compute_signal_strength,
    score_dual_momentum,
)


@pytest.fixture
def strategy_config():
    with open("config/strategies/momentum_v1.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def returns_12m_positive():
    """12-month returns where SPY beats SHY, QQQ does not."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "SPY": [0.10, 0.12, 0.15, 0.18, 0.20],
            "QQQ": [0.01, 0.02, 0.01, 0.00, 0.02],
            "AGG": [0.03, 0.04, 0.03, 0.04, 0.05],
            "SHY": [0.04, 0.04, 0.04, 0.04, 0.04],
        },
        index=dates,
    )


@pytest.fixture
def returns_12m_all_negative():
    """12-month returns where no risk asset beats SHY."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "SPY": [-0.05, -0.04, -0.03, -0.02, -0.01],
            "QQQ": [-0.10, -0.08, -0.06, -0.04, -0.03],
            "AGG": [0.02, 0.02, 0.03, 0.03, 0.03],
            "SHY": [0.04, 0.04, 0.04, 0.04, 0.04],
        },
        index=dates,
    )


class TestAbsoluteMomentum:
    def test_spy_beats_shy(self, returns_12m_positive):
        result = compute_absolute_momentum(returns_12m_positive, "SHY")
        assert bool(result["SPY"]) is True
        assert bool(result["QQQ"]) is False

    def test_no_asset_beats_shy(self, returns_12m_all_negative):
        result = compute_absolute_momentum(returns_12m_all_negative, "SHY")
        assert bool(result["SPY"]) is False
        assert bool(result["QQQ"]) is False

    def test_missing_benchmark(self, returns_12m_positive):
        result = compute_absolute_momentum(returns_12m_positive, "MISSING")
        assert result == {}

    def test_empty_returns(self):
        result = compute_absolute_momentum(pd.DataFrame(), "SHY")
        assert result == {}


class TestRelativeMomentum:
    def test_ranking_order(self, returns_12m_positive):
        result = compute_relative_momentum(returns_12m_positive, ["SPY", "QQQ"])
        # SPY (0.20) should rank above QQQ (0.02)
        assert result[0][0] == "SPY"
        assert result[1][0] == "QQQ"

    def test_empty_returns(self):
        result = compute_relative_momentum(pd.DataFrame(), ["SPY"])
        assert result == []


class TestMultiTimeframe:
    def test_confirmed_when_6m_also_positive(self, strategy_config):
        """6m return above benchmark = confirmed."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        features = {
            "returns": {
                "6m": pd.DataFrame(
                    {"SPY": [0.10, 0.12, 0.15, 0.18, 0.15], "SHY": [0.02] * 5},
                    index=dates,
                ),
            },
        }
        confirmed, details = check_multi_timeframe(features, "SPY", "SHY", strategy_config)
        assert confirmed is True

    def test_rejected_when_6m_below_benchmark(self, strategy_config):
        """6m return below benchmark = trend is dying."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        features = {
            "returns": {
                "6m": pd.DataFrame(
                    {"SPY": [0.01, 0.00, -0.01, -0.02, 0.01], "SHY": [0.02] * 5},
                    index=dates,
                ),
            },
        }
        confirmed, details = check_multi_timeframe(features, "SPY", "SHY", strategy_config)
        assert confirmed is False
        assert "blocked_by" in details

    def test_disabled_always_confirms(self):
        """When disabled, always returns True."""
        config = {"signals": {"multi_timeframe": {"enabled": False}}}
        confirmed, _ = check_multi_timeframe({}, "SPY", "SHY", config)
        assert confirmed is True


class TestMinimumEdge:
    def test_sufficient_edge(self, strategy_config):
        """20% return vs 4% benchmark = 16% edge > 2% threshold."""
        has_edge, excess = check_minimum_edge(0.20, 0.04, strategy_config)
        assert has_edge is True
        assert excess == pytest.approx(0.16)

    def test_insufficient_edge(self, strategy_config):
        """5% return vs 4% benchmark = 1% edge < 2% threshold."""
        has_edge, excess = check_minimum_edge(0.05, 0.04, strategy_config)
        assert has_edge is False
        assert excess == pytest.approx(0.01)

    def test_above_threshold(self, strategy_config):
        """7% return vs 4% benchmark = 3% edge > 2% threshold."""
        has_edge, excess = check_minimum_edge(0.07, 0.04, strategy_config)
        assert has_edge is True
        assert excess == pytest.approx(0.03)


class TestRSIFilter:
    def test_overbought_blocked(self, strategy_config):
        """RSI(2) > 90 should block BUY."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        features = {
            "rsi_2": pd.DataFrame(
                {"SPY": [50, 60, 70, 85, 95]},
                index=dates,
            ),
        }
        allowed, rsi_val, mult = check_rsi_filter(features, "SPY", strategy_config)
        assert allowed is False
        assert rsi_val == pytest.approx(95.0)

    def test_normal_rsi_passes(self, strategy_config):
        """RSI(2) = 50 should pass."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        features = {
            "rsi_2": pd.DataFrame(
                {"SPY": [30, 40, 45, 50, 50]},
                index=dates,
            ),
        }
        allowed, rsi_val, mult = check_rsi_filter(features, "SPY", strategy_config)
        assert allowed is True
        assert mult == pytest.approx(1.0)

    def test_oversold_gets_boost(self, strategy_config):
        """RSI(2) < 10 should boost signal strength."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        features = {
            "rsi_2": pd.DataFrame(
                {"SPY": [30, 20, 15, 10, 5]},
                index=dates,
            ),
        }
        allowed, rsi_val, mult = check_rsi_filter(features, "SPY", strategy_config)
        assert allowed is True
        assert mult == pytest.approx(1.25)  # oversold boost

    def test_disabled_always_passes(self):
        """When disabled, always returns True."""
        config = {"signals": {"rsi_filter": {"enabled": False}}}
        allowed, _, mult = check_rsi_filter({}, "SPY", config)
        assert allowed is True
        assert mult == pytest.approx(1.0)


class TestSignalStrength:
    def test_strong_momentum_high_strength(self, strategy_config):
        """All timeframes positive and large = high strength."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        features = {
            "returns": {
                "12m": pd.DataFrame({"SPY": [0.20] * 5, "SHY": [0.04] * 5}, index=dates),
                "6m": pd.DataFrame({"SPY": [0.12] * 5, "SHY": [0.02] * 5}, index=dates),
                "3m": pd.DataFrame({"SPY": [0.08] * 5, "SHY": [0.01] * 5}, index=dates),
            },
        }
        strength = compute_signal_strength(features, "SPY", "SHY", strategy_config)
        assert strength > 1.0  # Strong signal

    def test_weak_momentum_low_strength(self, strategy_config):
        """Marginal excess returns = low strength."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        features = {
            "returns": {
                "12m": pd.DataFrame({"SPY": [0.06] * 5, "SHY": [0.04] * 5}, index=dates),
                "6m": pd.DataFrame({"SPY": [0.03] * 5, "SHY": [0.02] * 5}, index=dates),
                "3m": pd.DataFrame({"SPY": [0.015] * 5, "SHY": [0.01] * 5}, index=dates),
            },
        }
        strength = compute_signal_strength(features, "SPY", "SHY", strategy_config)
        assert strength < 1.0  # Weak signal

    def test_disabled_returns_1(self):
        """When disabled, strength is always 1.0."""
        config = {"signals": {"signal_strength": {"enabled": False}}}
        strength = compute_signal_strength({}, "SPY", "SHY", config)
        assert strength == pytest.approx(1.0)


class TestScoreDualMomentum:
    def test_buy_signal_when_momentum_positive(self, strategy_config):
        dates = pd.date_range("2023-01-01", periods=300, freq="B")
        np.random.seed(42)
        prices = pd.DataFrame(
            {
                "SPY": 400 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 300)),
                "QQQ": 350 * np.cumprod(1 + np.random.normal(0.0005, 0.012, 300)),
                "AGG": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.003, 300)),
                "SHY": 85 * np.cumprod(1 + np.random.normal(0.00005, 0.001, 300)),
            },
            index=dates,
        )
        from data.feature_store import compute_returns, compute_rsi

        returns = compute_returns(prices, {"12m": 252, "6m": 126, "3m": 63, "1m": 21})
        rsi_2 = compute_rsi(prices, period=2)
        features = {"prices": prices, "returns": returns, "sma": {}, "rsi_2": rsi_2}

        signals = score_dual_momentum(features, strategy_config)
        assert len(signals) > 0
        # At least one signal should exist
        actions = {s["symbol"]: s["signal_type"] for s in signals}
        assert any(a in ("BUY", "SELL", "HOLD") for a in actions.values())

        # BUY signals should have signal_strength
        buy_signals = [s for s in signals if s["signal_type"] == "BUY"]
        for s in buy_signals:
            assert "signal_strength" in s
            assert s["signal_strength"] > 0

    def test_rotate_to_bonds_when_no_momentum(self, strategy_config):
        """When no risk asset has absolute momentum, should BUY safe asset."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        returns_12m = pd.DataFrame(
            {
                "SPY": [-0.05, -0.04, -0.03, -0.02, -0.01],
                "QQQ": [-0.10, -0.08, -0.06, -0.04, -0.03],
                "AGG": [0.02, 0.02, 0.03, 0.03, 0.06],
                "SHY": [0.04, 0.04, 0.04, 0.04, 0.04],
            },
            index=dates,
        )
        features = {
            "prices": pd.DataFrame(),
            "returns": {"12m": returns_12m, "6m": pd.DataFrame(), "3m": pd.DataFrame()},
            "sma": {},
            "rsi_2": pd.DataFrame(),
        }
        signals = score_dual_momentum(features, strategy_config)
        buy_signals = [s for s in signals if s["signal_type"] == "BUY"]
        # Should buy AGG (safe asset) since no risk asset has momentum
        assert len(buy_signals) == 1
        assert buy_signals[0]["symbol"] == "AGG"
        # Safe asset rotation has reduced signal strength
        assert buy_signals[0]["signal_strength"] == pytest.approx(0.5)

    def test_minimum_edge_filters_marginal_signal(self, strategy_config):
        """Asset barely beating benchmark should be filtered (HOLD not BUY)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        # SPY 5% vs SHY 4% = only 1% edge (below 2% threshold)
        returns_12m = pd.DataFrame(
            {
                "SPY": [0.04, 0.04, 0.04, 0.04, 0.05],
                "QQQ": [0.01, 0.01, 0.01, 0.01, 0.01],
                "AGG": [0.03, 0.03, 0.03, 0.03, 0.03],
                "SHY": [0.04, 0.04, 0.04, 0.04, 0.04],
            },
            index=dates,
        )
        returns_6m = pd.DataFrame(
            {
                "SPY": [0.03, 0.03, 0.03, 0.03, 0.03],
                "QQQ": [0.01, 0.01, 0.01, 0.01, 0.01],
                "SHY": [0.02, 0.02, 0.02, 0.02, 0.02],
            },
            index=dates,
        )
        features = {
            "prices": pd.DataFrame(),
            "returns": {"12m": returns_12m, "6m": returns_6m, "3m": pd.DataFrame()},
            "sma": {},
            "rsi_2": pd.DataFrame(),
        }
        signals = score_dual_momentum(features, strategy_config)
        # SPY has absolute momentum but marginal edge — should NOT be BUY
        buy_signals = [s for s in signals if s["signal_type"] == "BUY"]
        spy_buys = [s for s in buy_signals if s["symbol"] == "SPY"]
        assert len(spy_buys) == 0  # Filtered out by minimum edge

    def test_rsi_overbought_blocks_buy(self, strategy_config):
        """RSI(2) > 90 should prevent BUY even with strong momentum."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        returns_12m = pd.DataFrame(
            {
                "SPY": [0.15, 0.18, 0.20, 0.22, 0.25],
                "QQQ": [0.08, 0.10, 0.12, 0.14, 0.16],
                "AGG": [0.03, 0.03, 0.03, 0.03, 0.03],
                "SHY": [0.04, 0.04, 0.04, 0.04, 0.04],
            },
            index=dates,
        )
        returns_6m = pd.DataFrame(
            {
                "SPY": [0.10, 0.12, 0.14, 0.16, 0.18],
                "QQQ": [0.05, 0.06, 0.07, 0.08, 0.10],
                "SHY": [0.02, 0.02, 0.02, 0.02, 0.02],
            },
            index=dates,
        )
        rsi_data = pd.DataFrame(
            {
                "SPY": [50, 70, 80, 90, 95],  # Overbought
                "QQQ": [40, 50, 55, 60, 65],  # Normal
            },
            index=dates,
        )
        features = {
            "prices": pd.DataFrame(),
            "returns": {"12m": returns_12m, "6m": returns_6m, "3m": pd.DataFrame()},
            "sma": {},
            "rsi_2": rsi_data,
        }
        signals = score_dual_momentum(features, strategy_config)
        # SPY blocked by RSI, QQQ should get the BUY instead
        buy_signals = [s for s in signals if s["signal_type"] == "BUY"]
        if buy_signals:
            # Either QQQ gets BUY or bonds rotation
            assert buy_signals[0]["symbol"] != "SPY"


class TestFastExit:
    def test_exit_when_3m_below_benchmark(self, strategy_config):
        """3-month return below benchmark triggers fast exit."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        features = {
            "returns": {
                "3m": pd.DataFrame(
                    {"SPY": [0.05, 0.03, 0.01, -0.01, -0.02], "SHY": [0.01] * 5},
                    index=dates,
                ),
            },
        }
        should_exit, details = check_fast_exit(features, "SPY", "SHY", strategy_config)
        assert should_exit is True
        assert details["return_3m"] < details["benchmark_3m"]

    def test_no_exit_when_3m_above_benchmark(self, strategy_config):
        """3-month return above benchmark = no exit."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        features = {
            "returns": {
                "3m": pd.DataFrame(
                    {"SPY": [0.05, 0.06, 0.07, 0.08, 0.10], "SHY": [0.01] * 5},
                    index=dates,
                ),
            },
        }
        should_exit, details = check_fast_exit(features, "SPY", "SHY", strategy_config)
        assert should_exit is False

    def test_disabled_never_exits(self):
        """When disabled, never triggers exit."""
        config = {"signals": {"fast_exit": {"enabled": False}}}
        should_exit, details = check_fast_exit({}, "SPY", "SHY", config)
        assert should_exit is False
        assert details == {}

    def test_missing_data_no_exit(self, strategy_config):
        """Missing 3m data = no exit (safe default)."""
        features = {"returns": {}}
        should_exit, _ = check_fast_exit(features, "SPY", "SHY", strategy_config)
        assert should_exit is False


class TestVolatilityScalar:
    def test_low_vol_gets_bigger_scalar(self):
        """Low volatility asset should get scalar > 1.0."""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        # Very stable prices (low vol)
        prices = pd.DataFrame(
            {"SHY": 85.0 + np.cumsum(np.random.normal(0, 0.01, 30))},
            index=dates,
        )
        np.random.seed(10)
        scalar = compute_volatility_scalar({"prices": prices}, "SHY")
        assert scalar >= 1.0  # Low vol -> bigger position

    def test_high_vol_gets_smaller_scalar(self):
        """High volatility asset should get scalar < 1.0."""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        np.random.seed(42)
        # Very volatile prices
        prices = pd.DataFrame(
            {"MEME": 100 * np.cumprod(1 + np.random.normal(0, 0.05, 30))},
            index=dates,
        )
        scalar = compute_volatility_scalar({"prices": prices}, "MEME")
        assert scalar <= 1.0  # High vol -> smaller position

    def test_scalar_clamped_min(self):
        """Scalar should never go below 0.5."""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        np.random.seed(99)
        # Extremely volatile
        prices = pd.DataFrame(
            {"WILD": 100 * np.cumprod(1 + np.random.normal(0, 0.15, 30))},
            index=dates,
        )
        scalar = compute_volatility_scalar({"prices": prices}, "WILD")
        assert scalar >= 0.5

    def test_scalar_clamped_max(self):
        """Scalar should never exceed 1.5."""
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        # Nearly zero vol
        prices = pd.DataFrame(
            {"STABLE": [100.00 + i * 0.001 for i in range(30)]},
            index=dates,
        )
        scalar = compute_volatility_scalar({"prices": prices}, "STABLE")
        assert scalar <= 1.5

    def test_missing_symbol_returns_1(self):
        """Missing symbol returns neutral 1.0."""
        scalar = compute_volatility_scalar({"prices": pd.DataFrame()}, "MISSING")
        assert scalar == 1.0

    def test_insufficient_data_returns_1(self):
        """Less than 21 data points returns neutral 1.0."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"SPY": range(10)}, index=dates)
        scalar = compute_volatility_scalar({"prices": prices}, "SPY")
        assert scalar == 1.0


class TestMultiPositionBuy:
    def test_multiple_buy_signals(self, strategy_config):
        """With max_buy_signals=3, should generate up to 3 BUY signals."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        # Three strong risk assets all beating benchmark
        returns_12m = pd.DataFrame(
            {
                "SPY": [0.20] * 5,
                "QQQ": [0.18] * 5,
                "IWM": [0.15] * 5,
                "XLK": [0.12] * 5,
                "AGG": [0.03] * 5,
                "SHY": [0.04] * 5,
                "TLT": [0.02] * 5,
            },
            index=dates,
        )
        returns_6m = pd.DataFrame(
            {
                "SPY": [0.12] * 5,
                "QQQ": [0.10] * 5,
                "IWM": [0.08] * 5,
                "XLK": [0.07] * 5,
                "SHY": [0.02] * 5,
            },
            index=dates,
        )
        returns_3m = pd.DataFrame(
            {
                "SPY": [0.06] * 5,
                "QQQ": [0.05] * 5,
                "IWM": [0.04] * 5,
                "XLK": [0.03] * 5,
                "SHY": [0.01] * 5,
            },
            index=dates,
        )
        features = {
            "prices": pd.DataFrame(),
            "returns": {"12m": returns_12m, "6m": returns_6m, "3m": returns_3m},
            "sma": {},
            "rsi_2": pd.DataFrame(),
        }
        signals = score_dual_momentum(features, strategy_config)
        buy_signals = [s for s in signals if s["signal_type"] == "BUY"]
        # Should have up to 3 BUY signals (config max_buy_signals=3)
        assert len(buy_signals) <= 3
        assert len(buy_signals) >= 2  # At least SPY and QQQ should qualify

    def test_excess_qualified_become_hold(self, strategy_config):
        """Assets beyond max_buy_signals should get HOLD, not BUY."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        # Five strong risk assets
        returns_12m = pd.DataFrame(
            {
                "SPY": [0.25] * 5,
                "QQQ": [0.22] * 5,
                "IWM": [0.20] * 5,
                "XLK": [0.18] * 5,
                "XLE": [0.15] * 5,
                "AGG": [0.03] * 5,
                "SHY": [0.04] * 5,
                "TLT": [0.02] * 5,
            },
            index=dates,
        )
        returns_6m = pd.DataFrame(
            {
                "SPY": [0.15] * 5,
                "QQQ": [0.13] * 5,
                "IWM": [0.11] * 5,
                "XLK": [0.10] * 5,
                "XLE": [0.09] * 5,
                "SHY": [0.02] * 5,
            },
            index=dates,
        )
        returns_3m = pd.DataFrame(
            {
                "SPY": [0.08] * 5,
                "QQQ": [0.07] * 5,
                "IWM": [0.06] * 5,
                "XLK": [0.05] * 5,
                "XLE": [0.04] * 5,
                "SHY": [0.01] * 5,
            },
            index=dates,
        )
        features = {
            "prices": pd.DataFrame(),
            "returns": {"12m": returns_12m, "6m": returns_6m, "3m": returns_3m},
            "sma": {},
            "rsi_2": pd.DataFrame(),
        }
        signals = score_dual_momentum(features, strategy_config)
        buy_signals = [s for s in signals if s["signal_type"] == "BUY"]
        hold_signals = [s for s in signals if s["signal_type"] == "HOLD"]
        # Max 3 BUYs
        assert len(buy_signals) == 3
        # At least some assets should be HOLD (qualified but beyond limit)
        qualified_holds = [s for s in hold_signals if s["signal_strength"] > 0]
        assert len(qualified_holds) >= 1

    def test_buy_signals_have_slot_metadata(self, strategy_config):
        """BUY signals should include buy_slot in metadata."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        returns_12m = pd.DataFrame(
            {
                "SPY": [0.20] * 5,
                "QQQ": [0.18] * 5,
                "AGG": [0.03] * 5,
                "SHY": [0.04] * 5,
                "TLT": [0.02] * 5,
            },
            index=dates,
        )
        returns_6m = pd.DataFrame(
            {
                "SPY": [0.12] * 5,
                "QQQ": [0.10] * 5,
                "SHY": [0.02] * 5,
            },
            index=dates,
        )
        features = {
            "prices": pd.DataFrame(),
            "returns": {"12m": returns_12m, "6m": returns_6m, "3m": pd.DataFrame()},
            "sma": {},
            "rsi_2": pd.DataFrame(),
        }
        signals = score_dual_momentum(features, strategy_config)
        buy_signals = [s for s in signals if s["signal_type"] == "BUY"]
        for s in buy_signals:
            assert "buy_slot" in s["metadata"]
            assert s["metadata"]["buy_slot"] >= 1

    def test_fast_exit_sell_in_scoring(self, strategy_config):
        """Fast exit should emit SELL with fast_exit metadata in full pipeline."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        returns_12m = pd.DataFrame(
            {
                "SPY": [0.20] * 5,
                "QQQ": [0.15] * 5,
                "AGG": [0.03] * 5,
                "SHY": [0.04] * 5,
                "TLT": [0.02] * 5,
            },
            index=dates,
        )
        returns_6m = pd.DataFrame(
            {
                "SPY": [0.12] * 5,
                "QQQ": [0.08] * 5,
                "SHY": [0.02] * 5,
            },
            index=dates,
        )
        returns_3m = pd.DataFrame(
            {
                # SPY 3m is below benchmark -> fast exit
                "SPY": [0.05, 0.03, 0.01, -0.01, -0.02],
                # QQQ 3m is fine
                "QQQ": [0.04, 0.05, 0.06, 0.07, 0.08],
                "SHY": [0.01] * 5,
            },
            index=dates,
        )
        features = {
            "prices": pd.DataFrame(),
            "returns": {"12m": returns_12m, "6m": returns_6m, "3m": returns_3m},
            "sma": {},
            "rsi_2": pd.DataFrame(),
        }
        signals = score_dual_momentum(features, strategy_config)
        spy_signals = [s for s in signals if s["symbol"] == "SPY"]
        assert len(spy_signals) == 1
        assert spy_signals[0]["signal_type"] == "SELL"
        assert spy_signals[0]["metadata"].get("fast_exit") is True


class TestDailyExitScan:
    def test_exits_broken_positions(self, strategy_config):
        """Daily scan should SELL positions with 3m breakdown."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        features = {
            "returns": {
                "3m": pd.DataFrame(
                    {
                        "SPY": [0.05, 0.03, 0.01, -0.01, -0.02],  # broken
                        "QQQ": [0.04, 0.05, 0.06, 0.07, 0.08],    # fine
                        "SHY": [0.01] * 5,
                    },
                    index=dates,
                ),
            },
        }
        signals = run_daily_exit_scan(features, ["SPY", "QQQ"], strategy_config)
        assert len(signals) == 1
        assert signals[0]["symbol"] == "SPY"
        assert signals[0]["signal_type"] == "SELL"
        assert signals[0]["metadata"]["scan_type"] == "daily_exit"

    def test_no_exits_when_all_healthy(self, strategy_config):
        """No exits when all positions are above benchmark."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        features = {
            "returns": {
                "3m": pd.DataFrame(
                    {
                        "SPY": [0.08] * 5,
                        "QQQ": [0.06] * 5,
                        "SHY": [0.01] * 5,
                    },
                    index=dates,
                ),
            },
        }
        signals = run_daily_exit_scan(features, ["SPY", "QQQ"], strategy_config)
        assert len(signals) == 0

    def test_empty_positions_no_signals(self, strategy_config):
        """No held positions = no signals."""
        signals = run_daily_exit_scan({}, [], strategy_config)
        assert signals == []

    def test_daily_scan_never_buys(self, strategy_config):
        """Daily scan only produces SELL signals, never BUY."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        features = {
            "returns": {
                "3m": pd.DataFrame(
                    {
                        "SPY": [-0.05] * 5,
                        "QQQ": [-0.03] * 5,
                        "XLK": [-0.04] * 5,
                        "SHY": [0.01] * 5,
                    },
                    index=dates,
                ),
            },
        }
        signals = run_daily_exit_scan(features, ["SPY", "QQQ", "XLK"], strategy_config)
        for s in signals:
            assert s["signal_type"] == "SELL"


class TestCooldown:
    def test_recent_exit_blocks_buy(self, strategy_config):
        """Symbol exited 2 days ago should be in cooldown (5-day default)."""
        two_days_ago = (pd.Timestamp.now().normalize() - pd.offsets.BDay(2))
        exit_log = {"SPY": two_days_ago}
        in_cd, remaining = is_in_cooldown("SPY", exit_log, strategy_config)
        assert in_cd is True
        assert remaining > 0

    def test_old_exit_allows_buy(self, strategy_config):
        """Symbol exited 10 days ago should be clear."""
        ten_days_ago = (pd.Timestamp.now().normalize() - pd.offsets.BDay(10))
        exit_log = {"SPY": ten_days_ago}
        in_cd, remaining = is_in_cooldown("SPY", exit_log, strategy_config)
        assert in_cd is False
        assert remaining == 0

    def test_no_exit_log_no_cooldown(self, strategy_config):
        """Symbol not in exit log = no cooldown."""
        in_cd, remaining = is_in_cooldown("QQQ", {}, strategy_config)
        assert in_cd is False

    def test_cooldown_in_scoring_pipeline(self, strategy_config):
        """Cooldown should produce HOLD with cooldown metadata in full pipeline."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        returns_12m = pd.DataFrame(
            {
                "SPY": [0.25] * 5,
                "QQQ": [0.20] * 5,
                "AGG": [0.03] * 5,
                "SHY": [0.04] * 5,
                "TLT": [0.02] * 5,
            },
            index=dates,
        )
        returns_6m = pd.DataFrame(
            {
                "SPY": [0.15] * 5,
                "QQQ": [0.12] * 5,
                "SHY": [0.02] * 5,
            },
            index=dates,
        )
        returns_3m = pd.DataFrame(
            {
                "SPY": [0.08] * 5,
                "QQQ": [0.06] * 5,
                "SHY": [0.01] * 5,
            },
            index=dates,
        )
        features = {
            "prices": pd.DataFrame(),
            "returns": {"12m": returns_12m, "6m": returns_6m, "3m": returns_3m},
            "sma": {},
            "rsi_2": pd.DataFrame(),
        }
        # SPY exited yesterday — should be in cooldown
        yesterday = pd.Timestamp.now().normalize() - pd.offsets.BDay(1)
        exit_log = {"SPY": yesterday}
        signals = score_dual_momentum(features, strategy_config, exit_log=exit_log)
        spy_signals = [s for s in signals if s["symbol"] == "SPY"]
        assert len(spy_signals) == 1
        assert spy_signals[0]["signal_type"] == "HOLD"
        assert spy_signals[0]["metadata"].get("cooldown") is True
        # QQQ should still get BUY (not in cooldown)
        qqq_signals = [s for s in signals if s["symbol"] == "QQQ"]
        assert len(qqq_signals) == 1
        assert qqq_signals[0]["signal_type"] == "BUY"
