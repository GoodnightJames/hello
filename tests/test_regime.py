"""Tests for the regime classifier."""

import pandas as pd
import pytest

from research.regime import (
    classify_trend_regime,
    classify_volatility_regime,
    get_position_size_multiplier,
    REGIME_RISK_ON,
    REGIME_RISK_OFF,
    REGIME_HIGH_VOL,
)


class TestTrendRegime:
    def test_risk_on_above_200d(self):
        prices = pd.DataFrame(
            {"SPY": [400, 410, 420]},
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )
        sma = pd.DataFrame(
            {"SPY": [390, 395, 400]},
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )
        result = classify_trend_regime(prices, sma)
        assert result["trend_regime"] == REGIME_RISK_ON
        assert bool(result["above_200d"]) is True

    def test_risk_off_below_200d(self):
        prices = pd.DataFrame(
            {"SPY": [380, 370, 360]},
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )
        sma = pd.DataFrame(
            {"SPY": [390, 395, 400]},
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )
        result = classify_trend_regime(prices, sma)
        assert result["trend_regime"] == REGIME_RISK_OFF
        assert bool(result["above_200d"]) is False

    def test_missing_spy_defaults_risk_off(self):
        prices = pd.DataFrame(
            {"QQQ": [350, 360, 370]},
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )
        sma = pd.DataFrame(
            {"QQQ": [340, 345, 350]},
            index=pd.date_range("2024-01-01", periods=3, freq="B"),
        )
        result = classify_trend_regime(prices, sma)
        assert result["trend_regime"] == REGIME_RISK_OFF


class TestVolatilityRegime:
    def test_normal_vol(self):
        result = classify_volatility_regime(vix_price=20, vix_threshold=30)
        assert result["vol_regime"] == "normal"
        assert result["above_threshold"] is False

    def test_high_vol(self):
        result = classify_volatility_regime(vix_price=35, vix_threshold=30)
        assert result["vol_regime"] == REGIME_HIGH_VOL
        assert result["above_threshold"] is True

    def test_vix_unavailable_defaults_normal(self):
        result = classify_volatility_regime(vix_price=None)
        assert result["vol_regime"] == "normal"


class TestPositionMultiplier:
    def test_risk_on_normal_vol(self):
        trend = {"trend_regime": REGIME_RISK_ON}
        vol = {"vol_regime": "normal"}
        params = {"regime_throttles": {"spy_below_200d_reduce": 0.5, "vix_position_reduce": 0.5}}
        assert get_position_size_multiplier(trend, vol, params) == 1.0

    def test_risk_off_reduces_50pct(self):
        trend = {"trend_regime": REGIME_RISK_OFF}
        vol = {"vol_regime": "normal"}
        params = {"regime_throttles": {"spy_below_200d_reduce": 0.5, "vix_position_reduce": 0.5}}
        assert get_position_size_multiplier(trend, vol, params) == pytest.approx(0.5)

    def test_high_vol_reduces_50pct(self):
        trend = {"trend_regime": REGIME_RISK_ON}
        vol = {"vol_regime": REGIME_HIGH_VOL}
        params = {"regime_throttles": {"spy_below_200d_reduce": 0.5, "vix_position_reduce": 0.5}}
        assert get_position_size_multiplier(trend, vol, params) == pytest.approx(0.5)

    def test_both_reduces_to_25pct(self):
        trend = {"trend_regime": REGIME_RISK_OFF}
        vol = {"vol_regime": REGIME_HIGH_VOL}
        params = {"regime_throttles": {"spy_below_200d_reduce": 0.5, "vix_position_reduce": 0.5}}
        assert get_position_size_multiplier(trend, vol, params) == pytest.approx(0.25)
