"""
Regression test: generate_signals must only score the instruments it receives.

If generate_signals() ignores its instruments parameter and re-fetches the full
universe internally, universe policy filters become fake — any symbol disabled
or blocked by allocation policy would still get scored and potentially selected.

This test verifies the contract:
    1. Only passed instruments are scored.
    2. Excluded symbols never appear in diagnostics or signals.
    3. Backward compat: omitting instruments= still works (uses full universe).
"""

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from strategies.crypto_dca_v1 import CryptoDCAStrategy


def _make_price_df(symbols, n_days=30, base_prices=None):
    """Build a mock price DataFrame with gentle uptrend."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    data = {}
    for i, sym in enumerate(symbols):
        base = (base_prices or {}).get(sym, 100 + i * 50)
        # Gentle uptrend so momentum is positive
        data[sym] = [base * (1 + 0.005 * d) for d in range(n_days)]
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def strategy():
    """Create a strategy instance with known config."""
    s = CryptoDCAStrategy()
    return s


class TestUniverseFilterContract:
    """generate_signals must respect the instruments parameter."""

    @patch("strategies.crypto_dca_v1.estimate_round_trip_cost")
    @patch("strategies.crypto_dca_v1.get_price_history")
    def test_only_filtered_instruments_are_scored(
        self, mock_prices, mock_cost, strategy
    ):
        """Given 3 filtered symbols, only those 3 should appear in scored output."""
        filtered = ["BTC/USD", "ETH/USD", "SOL/USD"]
        excluded = ["XRP/USD", "LINK/USD", "AVAX/USD", "DOT/USD", "DOGE/USD"]

        # Price data for ALL symbols (simulates DB having everything)
        all_syms = filtered + excluded
        mock_prices.return_value = _make_price_df(all_syms)
        mock_cost.return_value = {"total_bps": 15, "spread_bps": 5, "fee_bps": 10}

        scored = strategy._score_coins(filtered, held_symbols=set())

        scored_symbols = {s[0] for s in scored}
        # Only filtered symbols should be scored
        assert scored_symbols <= set(filtered), (
            f"Excluded symbols leaked into scoring: "
            f"{scored_symbols - set(filtered)}"
        )
        # All filtered symbols should be present
        assert scored_symbols == set(filtered), (
            f"Missing filtered symbols: {set(filtered) - scored_symbols}"
        )

    @patch("strategies.crypto_dca_v1.estimate_round_trip_cost")
    @patch("strategies.crypto_dca_v1.get_price_history")
    def test_excluded_symbols_never_in_signals(
        self, mock_prices, mock_cost, strategy
    ):
        """Signals must only contain symbols from the instruments parameter."""
        filtered = ["BTC/USD", "ETH/USD", "SOL/USD"]
        excluded = ["XRP/USD", "LINK/USD"]

        all_syms = filtered + excluded
        mock_prices.return_value = _make_price_df(all_syms)
        mock_cost.return_value = {"total_bps": 15, "spread_bps": 5, "fee_bps": 10}

        signals = strategy.generate_signals(
            held_symbols=set(),
            regime={"phase": "trending", "confidence": 0.8},
            instruments=filtered,
        )

        for sig in signals:
            assert sig["symbol"] in filtered, (
                f"Signal for excluded symbol: {sig['symbol']}"
            )
            assert sig["symbol"] not in excluded, (
                f"Excluded symbol leaked into signals: {sig['symbol']}"
            )

    @patch("strategies.crypto_dca_v1.estimate_round_trip_cost")
    @patch("strategies.crypto_dca_v1.get_price_history")
    def test_diagnostics_only_contain_filtered_symbols(
        self, mock_prices, mock_cost, strategy
    ):
        """Diagnostics dicts should only exist for filtered symbols."""
        filtered = ["BTC/USD", "ETH/USD"]
        excluded = ["SOL/USD", "XRP/USD", "LINK/USD"]

        all_syms = filtered + excluded
        mock_prices.return_value = _make_price_df(all_syms)
        mock_cost.return_value = {"total_bps": 15, "spread_bps": 5, "fee_bps": 10}

        scored = strategy._score_coins(filtered, held_symbols=set())

        for sym, score, reason, diag in scored:
            assert sym in filtered, (
                f"Diagnostics contain excluded symbol: {sym}"
            )

    @patch("strategies.crypto_dca_v1.estimate_round_trip_cost")
    @patch("strategies.crypto_dca_v1.get_price_history")
    def test_omitting_instruments_uses_full_universe(
        self, mock_prices, mock_cost, strategy
    ):
        """Legacy behavior: omitting instruments= should use get_instruments()."""
        full_universe = strategy.get_instruments()
        mock_prices.return_value = _make_price_df(full_universe)
        mock_cost.return_value = {"total_bps": 15, "spread_bps": 5, "fee_bps": 10}

        # Call without instruments= parameter
        signals = strategy.generate_signals(
            held_symbols=set(),
            regime={"phase": "trending", "confidence": 0.8},
        )

        # Should not crash, and any signal should be from full universe
        for sig in signals:
            assert sig["symbol"] in full_universe

    @patch("strategies.crypto_dca_v1.estimate_round_trip_cost")
    @patch("strategies.crypto_dca_v1.get_price_history")
    def test_score_count_matches_filtered_count(
        self, mock_prices, mock_cost, strategy
    ):
        """Number of scored entries must equal number of filtered instruments."""
        filtered = ["BTC/USD", "SOL/USD", "AVAX/USD"]

        mock_prices.return_value = _make_price_df(filtered)
        mock_cost.return_value = {"total_bps": 15, "spread_bps": 5, "fee_bps": 10}

        scored = strategy._score_coins(filtered, held_symbols=set())

        assert len(scored) == len(filtered), (
            f"Expected {len(filtered)} scored entries, got {len(scored)}"
        )
