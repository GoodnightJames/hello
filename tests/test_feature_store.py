"""Tests for the feature store — returns, SMAs, RSI computation."""

import numpy as np
import pandas as pd
import pytest

from data.feature_store import compute_returns, compute_sma, compute_rsi


@pytest.fixture
def sample_prices():
    """Create a simple deterministic price series for testing."""
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "SPY": 400 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 300)),
            "QQQ": 350 * np.cumprod(1 + np.random.normal(0.0005, 0.012, 300)),
            "SHY": 85 * np.cumprod(1 + np.random.normal(0.00005, 0.001, 300)),
        },
        index=dates,
    )


class TestComputeReturns:
    def test_returns_shape(self, sample_prices):
        result = compute_returns(sample_prices, {"12m": 252})
        assert "12m" in result
        assert result["12m"].shape == sample_prices.shape

    def test_returns_nan_for_insufficient_data(self, sample_prices):
        result = compute_returns(sample_prices, {"12m": 252})
        # First 252 rows should be NaN
        assert result["12m"].iloc[0].isna().all()
        # Last row should have values
        assert result["12m"].iloc[-1].notna().all()

    def test_returns_multiple_periods(self, sample_prices):
        result = compute_returns(sample_prices, {"12m": 252, "1m": 21})
        assert "12m" in result
        assert "1m" in result

    def test_returns_empty_for_insufficient_data(self, sample_prices):
        short = sample_prices.iloc[:10]
        result = compute_returns(short, {"12m": 252})
        assert result["12m"].empty


class TestComputeSMA:
    def test_sma_200d(self, sample_prices):
        result = compute_sma(sample_prices, {"200d": 200})
        assert "200d" in result
        # First 199 rows should be NaN
        assert result["200d"].iloc[100].isna().all()
        # Last row should have values
        assert result["200d"].iloc[-1].notna().all()

    def test_sma_value_correctness(self):
        """SMA of constant series should equal that constant."""
        prices = pd.DataFrame(
            {"A": [10.0] * 50},
            index=pd.date_range("2024-01-01", periods=50, freq="B"),
        )
        result = compute_sma(prices, {"10d": 10})
        assert result["10d"]["A"].iloc[-1] == pytest.approx(10.0)

    def test_sma_multiple_windows(self, sample_prices):
        result = compute_sma(sample_prices, {"200d": 200, "50d": 50})
        assert "200d" in result
        assert "50d" in result


class TestComputeRSI:
    def test_rsi_range(self, sample_prices):
        rsi = compute_rsi(sample_prices, period=2)
        valid = rsi.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 100).all().all()

    def test_rsi_shape(self, sample_prices):
        rsi = compute_rsi(sample_prices, period=2)
        assert rsi.shape == sample_prices.shape

    def test_rsi_mostly_up(self):
        """A mostly rising series should have high RSI."""
        np.random.seed(99)
        changes = np.abs(np.random.normal(1.0, 0.3, 50))
        prices = pd.DataFrame(
            {"A": 100 + np.cumsum(changes)},
            index=pd.date_range("2024-01-01", periods=50, freq="B"),
        )
        rsi = compute_rsi(prices, period=14)
        assert rsi["A"].iloc[-1] > 80.0
