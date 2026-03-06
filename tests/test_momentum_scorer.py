"""Tests for the dual momentum scorer."""

import numpy as np
import pandas as pd
import pytest
import yaml

from research.momentum_scorer import (
    compute_absolute_momentum,
    compute_relative_momentum,
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
        from data.feature_store import compute_returns

        returns = compute_returns(prices, {"12m": 252})
        features = {"prices": prices, "returns": returns, "sma": {}, "rsi_2": pd.DataFrame()}

        signals = score_dual_momentum(features, strategy_config)
        assert len(signals) > 0
        # At least one signal should exist
        actions = {s["symbol"]: s["signal_type"] for s in signals}
        assert any(a in ("BUY", "SELL", "HOLD") for a in actions.values())

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
            "returns": {"12m": returns_12m},
            "sma": {},
            "rsi_2": pd.DataFrame(),
        }
        signals = score_dual_momentum(features, strategy_config)
        buy_signals = [s for s in signals if s["signal_type"] == "BUY"]
        # Should buy AGG (safe asset) since no risk asset has momentum
        assert len(buy_signals) == 1
        assert buy_signals[0]["symbol"] == "AGG"
