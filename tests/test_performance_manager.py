"""Tests for the performance manager — risk mode selection and metrics."""

import json
import os
from datetime import datetime, timedelta

import pytest

from data.db import init_db, get_session, PortfolioState, Order, Base, get_engine
from performance.manager import (
    MODE_CONSERVATIVE,
    MODE_NORMAL,
    MODE_AGGRESSIVE,
    compute_equity_curve,
    compute_rolling_return,
    compute_max_drawdown,
    compute_current_drawdown,
    count_trades_this_week,
    select_risk_mode,
    get_mode_params,
    check_trade_throttle,
)
from risk.enforcer import load_risk_params


@pytest.fixture(autouse=True)
def clean_db():
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    engine = init_db("sqlite:///:memory:")
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def session(clean_db):
    return get_session(clean_db)


@pytest.fixture
def risk_params():
    return load_risk_params()


def _add_equity_snapshots(session, values, start_days_ago=None):
    """Helper to add a sequence of portfolio snapshots."""
    now = datetime.utcnow()
    if start_days_ago is None:
        start_days_ago = len(values)
    for i, equity in enumerate(values):
        days_ago = start_days_ago - i
        state = PortfolioState(
            date=now - timedelta(days=days_ago),
            cash=equity * 0.5,
            total_equity=equity,
            positions_json=json.dumps({}),
        )
        session.add(state)
    session.commit()


# ── Equity curve tests ─────────────────────────────────────────────────────


class TestEquityCurve:
    def test_returns_snapshots_oldest_first(self, session):
        _add_equity_snapshots(session, [500, 510, 520])
        curve = compute_equity_curve(session, lookback_days=30)
        assert len(curve) == 3
        assert curve[0][1] == 500
        assert curve[-1][1] == 520

    def test_empty_when_no_data(self, session):
        curve = compute_equity_curve(session)
        assert curve == []


# ── Rolling return tests ────────────────────────────────────────────────────


class TestRollingReturn:
    def test_positive_return(self):
        now = datetime.utcnow()
        curve = [
            (now - timedelta(days=60), 500),
            (now - timedelta(days=30), 520),
            (now, 540),
        ]
        ret = compute_rolling_return(curve, lookback_days=60)
        assert ret == pytest.approx(0.08)  # (540-500)/500

    def test_negative_return(self):
        now = datetime.utcnow()
        curve = [
            (now - timedelta(days=60), 500),
            (now, 450),
        ]
        ret = compute_rolling_return(curve, lookback_days=60)
        assert ret == pytest.approx(-0.10)

    def test_insufficient_data(self):
        assert compute_rolling_return([], lookback_days=60) is None
        assert compute_rolling_return([(datetime.utcnow(), 500)], lookback_days=60) is None


# ── Drawdown tests ──────────────────────────────────────────────────────────


class TestDrawdown:
    def test_no_drawdown(self):
        now = datetime.utcnow()
        curve = [(now - timedelta(days=i), 500 + i) for i in range(10)]
        assert compute_max_drawdown(curve) == 0.0

    def test_10pct_drawdown(self):
        now = datetime.utcnow()
        curve = [
            (now - timedelta(days=3), 100),
            (now - timedelta(days=2), 110),  # peak
            (now - timedelta(days=1), 99),   # 10% from peak
            (now, 105),
        ]
        dd = compute_max_drawdown(curve)
        assert dd == pytest.approx(0.10, abs=0.001)

    def test_current_drawdown(self):
        now = datetime.utcnow()
        curve = [
            (now - timedelta(days=2), 100),
            (now - timedelta(days=1), 110),  # peak
            (now, 99),                        # current (10% from peak)
        ]
        assert compute_current_drawdown(curve) == pytest.approx(0.10, abs=0.001)

    def test_no_drawdown_at_peak(self):
        now = datetime.utcnow()
        curve = [
            (now - timedelta(days=1), 100),
            (now, 110),  # at peak
        ]
        assert compute_current_drawdown(curve) == 0.0

    def test_empty_curve(self):
        assert compute_max_drawdown([]) == 0.0
        assert compute_current_drawdown([]) == 0.0


# ── Trade throttle tests ───────────────────────────────────────────────────


class TestTradeThrottle:
    def test_not_throttled_when_no_trades(self, session, risk_params):
        throttled, details = check_trade_throttle(session, risk_params)
        assert throttled is False
        assert details["trades_this_week"] == 0

    def test_throttled_at_limit(self, session, risk_params):
        now = datetime.utcnow()
        max_per_week = risk_params.get("trade_throttle", {}).get("max_trades_per_week", 5)
        for i in range(max_per_week):
            order = Order(
                symbol="SPY",
                side="buy",
                qty=1,
                order_type="market",
                status="filled",
                filled_at=now - timedelta(hours=i),
            )
            session.add(order)
        session.commit()

        throttled, details = check_trade_throttle(session, risk_params)
        assert throttled is True
        assert details["trades_this_week"] == max_per_week


# ── Mode selection tests ───────────────────────────────────────────────────


class TestSelectRiskMode:
    def test_normal_when_no_data(self, session, risk_params):
        result = select_risk_mode(session, risk_params)
        assert result["mode"] == MODE_NORMAL
        assert "insufficient" in result["reason"].lower()

    def test_conservative_on_drawdown(self, session, risk_params):
        # Create equity curve with 11% drawdown (threshold is 10% for small accounts)
        values = [500, 510, 520, 530, 520, 472]
        _add_equity_snapshots(session, values, start_days_ago=10)

        result = select_risk_mode(session, risk_params)
        assert result["mode"] == MODE_CONSERVATIVE
        assert "drawdown" in result["reason"].lower()

    def test_aggressive_on_strong_performance(self, session, risk_params):
        # Create equity curve: 15% gain, no drawdown (threshold is 12%)
        now = datetime.utcnow()
        start = 500
        end = 575  # 15% gain
        n = 70  # 60-day lookback
        for i in range(n):
            equity = start + (end - start) * i / (n - 1)
            state = PortfolioState(
                date=now - timedelta(days=n - i),
                cash=equity * 0.5,
                total_equity=equity,
                positions_json=json.dumps({}),
            )
            session.add(state)
        session.commit()

        result = select_risk_mode(session, risk_params)
        assert result["mode"] == MODE_AGGRESSIVE

    def test_normal_when_moderate_performance(self, session, risk_params):
        # Create equity curve: 3% gain (below aggressive threshold), low drawdown
        now = datetime.utcnow()
        start = 500
        end = 515  # 3% gain
        n = 70
        for i in range(n):
            equity = start + (end - start) * i / (n - 1)
            state = PortfolioState(
                date=now - timedelta(days=n - i),
                cash=equity * 0.5,
                total_equity=equity,
                positions_json=json.dumps({}),
            )
            session.add(state)
        session.commit()

        result = select_risk_mode(session, risk_params)
        assert result["mode"] == MODE_NORMAL


# ── Mode params tests ──────────────────────────────────────────────────────


class TestGetModeParams:
    def test_conservative_params(self, risk_params):
        params = get_mode_params(risk_params, MODE_CONSERVATIVE)
        assert params["risk_per_trade_pct"] == 0.50  # Deploy 50% of cash
        assert params["max_positions"] == 2
        assert params["deploy_deposits"] is False

    def test_normal_params(self, risk_params):
        params = get_mode_params(risk_params, MODE_NORMAL)
        assert params["risk_per_trade_pct"] == 0.90  # Deploy 90% of cash
        assert params["max_positions"] == 3
        assert params["deploy_deposits"] is True

    def test_aggressive_params(self, risk_params):
        params = get_mode_params(risk_params, MODE_AGGRESSIVE)
        assert params["risk_per_trade_pct"] == 0.95  # Deploy 95% of cash
        assert params["max_positions"] == 4
        assert params["deploy_deposits"] is True

    def test_unknown_mode_uses_base_defaults(self, risk_params):
        params = get_mode_params(risk_params, "unknown_mode")
        # Falls back to base position_limits
        assert params["risk_per_trade_pct"] == 0.90
        assert params["max_positions"] == 3


# ── Integration: risk enforcer with modes ──────────────────────────────────


class TestRiskEnforcerModeIntegration:
    def test_conservative_mode_reduces_trade_size(self, session, risk_params):
        from risk.enforcer import calculate_max_trade_size

        normal_params = get_mode_params(risk_params, MODE_NORMAL)
        conservative_params = get_mode_params(risk_params, MODE_CONSERVATIVE)

        # Use cash-based sizing (small account mode)
        normal_size = calculate_max_trade_size(risk_params, 1000, mode_params=normal_params, cash=200)
        conservative_size = calculate_max_trade_size(risk_params, 1000, mode_params=conservative_params, cash=200)

        assert conservative_size < normal_size
        assert normal_size == pytest.approx(180.0)      # 90% of $200 cash
        assert conservative_size == pytest.approx(100.0)  # 50% of $200 cash

    def test_aggressive_mode_increases_trade_size(self, session, risk_params):
        from risk.enforcer import calculate_max_trade_size

        aggressive_params = get_mode_params(risk_params, MODE_AGGRESSIVE)
        normal_params = get_mode_params(risk_params, MODE_NORMAL)

        agg_size = calculate_max_trade_size(risk_params, 1000, mode_params=aggressive_params, cash=200)
        normal_size = calculate_max_trade_size(risk_params, 1000, mode_params=normal_params, cash=200)

        assert agg_size > normal_size
        assert agg_size == pytest.approx(190.0)  # 95% of $200 cash

    def test_conservative_mode_limits_positions(self, session, risk_params):
        from risk.enforcer import check_position_limits

        conservative_params = get_mode_params(risk_params, MODE_CONSERVATIVE)

        # Fill up to conservative max (2)
        for i in range(2):
            state = PortfolioState(
                date=datetime.utcnow(),
                cash=100,
                total_equity=1000,
                positions_json=json.dumps({f"SYM{j}": 10 for j in range(i + 1)}),
            )
            session.add(state)
        session.commit()

        can_open, _ = check_position_limits(session, risk_params, "NEW", mode_params=conservative_params)
        assert can_open is False

    def test_deposit_buffered_in_conservative(self, session, risk_params):
        from capital.manager import initialize_portfolio, record_deposit

        initialize_portfolio(session, initial_capital=500)

        conservative_params = get_mode_params(risk_params, MODE_CONSERVATIVE)
        result = record_deposit(session, 100, notes="Weekly", mode_params=conservative_params)

        assert result["cash"] == 600  # Cash still increases
        # But the deposit is marked as buffered (via notes on the Deposit record)
