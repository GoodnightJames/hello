"""
Tests for execution/alpaca_broker.py — all Alpaca API calls are mocked.

Verifies:
- Client initialization and credential validation
- Account info parsing
- Position fetching
- Order submission
- Order status polling
- Retry logic for transient errors
"""

import os
from unittest.mock import patch, MagicMock

import pytest

# Patch environment before importing broker module
os.environ["ALPACA_API_KEY"] = "test-key"
os.environ["ALPACA_SECRET_KEY"] = "test-secret"


def _reset_client():
    """Reset the module-level singleton so each test gets a fresh client."""
    import execution.alpaca_broker as broker
    broker._client = None


# ── Client initialization ──────────────────────────────────────────────────


class TestClientInit:
    def setup_method(self):
        _reset_client()

    @patch("execution.alpaca_broker.TradingClient")
    def test_creates_client_with_paper_mode(self, mock_cls):
        from execution.alpaca_broker import get_client
        get_client()
        mock_cls.assert_called_once_with("test-key", "test-secret", paper=True)

    @patch("execution.alpaca_broker.TradingClient")
    def test_singleton_returns_same_client(self, mock_cls):
        from execution.alpaca_broker import get_client
        c1 = get_client()
        c2 = get_client()
        assert c1 is c2
        assert mock_cls.call_count == 1

    def test_missing_credentials_raises(self):
        _reset_client()
        with patch.dict(os.environ, {"ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""}):
            from execution.alpaca_broker import get_client
            _reset_client()
            with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
                get_client()


# ── Account info ───────────────────────────────────────────────────────────


class TestGetAccount:
    def setup_method(self):
        _reset_client()

    @patch("execution.alpaca_broker.TradingClient")
    def test_parses_account_fields(self, mock_cls):
        mock_account = MagicMock()
        mock_account.cash = "100000.00"
        mock_account.equity = "100000.00"
        mock_account.buying_power = "200000.00"
        mock_account.status = MagicMock(value="ACTIVE")
        mock_account.currency = "USD"
        mock_account.pattern_day_trader = False

        mock_cls.return_value.get_account.return_value = mock_account

        from execution.alpaca_broker import get_account
        info = get_account()

        assert info["cash"] == 100000.0
        assert info["equity"] == 100000.0
        assert info["buying_power"] == 200000.0
        assert info["status"] == "ACTIVE"
        assert info["currency"] == "USD"


# ── Positions ──────────────────────────────────────────────────────────────


class TestGetPositions:
    def setup_method(self):
        _reset_client()

    @patch("execution.alpaca_broker.TradingClient")
    def test_parses_positions(self, mock_cls):
        mock_pos = MagicMock()
        mock_pos.symbol = "SPY"
        mock_pos.qty = "10"
        mock_pos.market_value = "5700.00"
        mock_pos.avg_entry_price = "560.00"
        mock_pos.unrealized_pl = "100.00"
        mock_pos.current_price = "570.00"

        mock_cls.return_value.get_all_positions.return_value = [mock_pos]

        from execution.alpaca_broker import get_positions
        positions = get_positions()

        assert "SPY" in positions
        assert positions["SPY"]["qty"] == 10.0
        assert positions["SPY"]["avg_entry"] == 560.0
        assert positions["SPY"]["current_price"] == 570.0

    @patch("execution.alpaca_broker.TradingClient")
    def test_empty_positions(self, mock_cls):
        mock_cls.return_value.get_all_positions.return_value = []

        from execution.alpaca_broker import get_positions
        positions = get_positions()
        assert positions == {}


# ── Order submission ───────────────────────────────────────────────────────


class TestSubmitOrder:
    def setup_method(self):
        _reset_client()

    @patch("execution.alpaca_broker.TradingClient")
    def test_submit_buy_order(self, mock_cls):
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.symbol = "SPY"
        mock_order.status = MagicMock(value="accepted")
        mock_order.submitted_at = "2026-03-06T10:00:00Z"

        mock_cls.return_value.submit_order.return_value = mock_order

        from execution.alpaca_broker import submit_market_order
        result = submit_market_order("SPY", 10, "buy")

        assert result["broker_order_id"] == "order-123"
        assert result["symbol"] == "SPY"
        assert result["side"] == "buy"
        assert result["qty"] == 10

    @patch("execution.alpaca_broker.TradingClient")
    def test_submit_sell_order(self, mock_cls):
        mock_order = MagicMock()
        mock_order.id = "order-456"
        mock_order.symbol = "QQQ"
        mock_order.status = MagicMock(value="accepted")
        mock_order.submitted_at = "2026-03-06T10:00:00Z"

        mock_cls.return_value.submit_order.return_value = mock_order

        from execution.alpaca_broker import submit_market_order
        result = submit_market_order("QQQ", 5, "sell")

        assert result["side"] == "sell"
        assert result["qty"] == 5


# ── Order status ───────────────────────────────────────────────────────────


class TestOrderStatus:
    def setup_method(self):
        _reset_client()

    @patch("execution.alpaca_broker.TradingClient")
    def test_filled_order_status(self, mock_cls):
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.symbol = "SPY"
        mock_order.side = MagicMock(value="buy")
        mock_order.qty = "10"
        mock_order.status = MagicMock(value="filled")
        mock_order.filled_qty = "10"
        mock_order.filled_avg_price = "570.50"
        mock_order.submitted_at = "2026-03-06T10:00:00Z"
        mock_order.filled_at = "2026-03-06T10:00:01Z"

        mock_cls.return_value.get_order_by_id.return_value = mock_order

        from execution.alpaca_broker import get_order_status
        status = get_order_status("order-123")

        assert status["status"] == "filled"
        assert status["filled_avg_price"] == 570.50
        assert status["filled_qty"] == 10.0


# ── Retry logic ────────────────────────────────────────────────────────────


class TestRetryLogic:
    def setup_method(self):
        _reset_client()

    @patch("execution.alpaca_broker.time.sleep")
    def test_retries_on_connection_error(self, mock_sleep):
        from requests.exceptions import ConnectionError
        from execution.alpaca_broker import _retry

        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection reset")
            return "success"

        result = _retry(flaky_func, max_retries=3, base_delay=1)
        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2  # 2 retries before success

    @patch("execution.alpaca_broker.time.sleep")
    def test_gives_up_after_max_retries(self, mock_sleep):
        from requests.exceptions import ConnectionError
        from execution.alpaca_broker import _retry

        def always_fail():
            raise ConnectionError("network down")

        with pytest.raises(ConnectionError):
            _retry(always_fail, max_retries=2, base_delay=1)

        assert mock_sleep.call_count == 2

    @patch("execution.alpaca_broker.time.sleep")
    def test_does_not_retry_on_api_rejection(self, mock_sleep):
        from execution.alpaca_broker import _retry

        def api_error():
            raise ValueError("insufficient buying power")

        with pytest.raises(ValueError):
            _retry(api_error, max_retries=3, base_delay=1)

        # Should NOT have retried — ValueError is not a network error
        assert mock_sleep.call_count == 0


# ── Go-live readiness ──────────────────────────────────────────────────────


class TestGoLiveReadiness:
    def test_not_ready_at_start(self):
        from status import check_go_live_readiness

        trade_stats = {"total_trades": 0, "days_active": 0}
        equity_history = {"drawdown_pct": 0, "snapshots": 0}
        risk_events = []

        result = check_go_live_readiness(None, trade_stats, equity_history, risk_events)
        assert result["ready"] is False
        assert result["checks"]["min_days"]["passed"] is False
        assert result["checks"]["min_trades"]["passed"] is False

    def test_ready_after_30_days(self):
        from status import check_go_live_readiness

        trade_stats = {"total_trades": 15, "days_active": 35}
        equity_history = {"drawdown_pct": 3.5, "snapshots": 30}
        risk_events = []

        result = check_go_live_readiness(None, trade_stats, equity_history, risk_events)
        assert result["ready"] is True

    def test_fails_on_high_drawdown(self):
        from status import check_go_live_readiness

        trade_stats = {"total_trades": 15, "days_active": 35}
        equity_history = {"drawdown_pct": 12.0, "snapshots": 30}
        risk_events = []

        result = check_go_live_readiness(None, trade_stats, equity_history, risk_events)
        assert result["ready"] is False
        assert result["checks"]["max_drawdown"]["passed"] is False

    def test_fails_on_too_many_loss_events(self):
        from status import check_go_live_readiness

        trade_stats = {"total_trades": 15, "days_active": 35}
        equity_history = {"drawdown_pct": 3.0, "snapshots": 30}
        risk_events = [
            {"type": "daily_loss_limit", "severity": "CRITICAL", "date": "2026-03-01"},
            {"type": "daily_loss_limit", "severity": "CRITICAL", "date": "2026-03-05"},
            {"type": "daily_loss_limit", "severity": "CRITICAL", "date": "2026-03-10"},
        ]

        result = check_go_live_readiness(None, trade_stats, equity_history, risk_events)
        assert result["ready"] is False
        assert result["checks"]["daily_loss_events"]["passed"] is False
