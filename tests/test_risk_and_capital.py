"""Tests for risk enforcer, capital manager, and order manager."""

import json
import os

import pytest

from data.db import init_db, get_session, PortfolioState, Base, get_engine
from risk.enforcer import (
    calculate_max_trade_size,
    check_kill_switch,
    check_position_limits,
    load_risk_params,
    validate_order,
)
from capital.manager import (
    initialize_portfolio,
    get_or_create_portfolio,
    update_position,
    record_deposit,
    calculate_strategy_allocation,
)
from execution.order_manager import calculate_shares, create_order


@pytest.fixture(autouse=True)
def clean_db():
    """Use a fresh in-memory database for each test."""
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


class TestMaxTradeSize:
    def test_500_portfolio(self, risk_params):
        # 90% of $500 = $450 (deploy almost all cash)
        result = calculate_max_trade_size(risk_params, 500)
        assert result == pytest.approx(450.00)

    def test_1000_portfolio(self, risk_params):
        # 90% of $1000 = $900
        result = calculate_max_trade_size(risk_params, 1000)
        assert result == pytest.approx(900.00)

    def test_cash_based_sizing(self, risk_params):
        # When cash is provided, size from cash not equity
        result = calculate_max_trade_size(risk_params, 10000, cash=100)
        assert result == pytest.approx(90.00)  # 90% of $100 cash

    def test_zero_equity(self, risk_params):
        result = calculate_max_trade_size(risk_params, 0)
        assert result == 0.0


class TestKillSwitch:
    def test_kill_switch_off(self):
        os.environ.pop("KILL_SWITCH", None)
        assert check_kill_switch() is False

    def test_kill_switch_on(self):
        os.environ["KILL_SWITCH"] = "true"
        assert check_kill_switch() is True
        os.environ.pop("KILL_SWITCH", None)


class TestPositionLimits:
    def test_can_open_when_no_positions(self, session, risk_params):
        can_open, details = check_position_limits(session, risk_params, "SPY")
        assert can_open is True

    def test_blocked_at_max(self, session, risk_params):
        # Create a portfolio state with max positions
        max_pos = risk_params["position_limits"]["max_concurrent_positions"]
        positions = {f"SYM{i}": 10 for i in range(max_pos)}
        from datetime import datetime
        state = PortfolioState(
            date=datetime(2024, 1, 1),
            cash=100,
            total_equity=1000,
            positions_json=json.dumps(positions),
        )
        session.add(state)
        session.commit()

        can_open, details = check_position_limits(session, risk_params, "NEW")
        assert can_open is False


class TestCapitalManager:
    def test_initialize_portfolio(self, session):
        state = initialize_portfolio(session, initial_capital=500)
        assert state["cash"] == 500
        assert state["total_equity"] == 500
        assert state["positions"] == {}

    def test_get_or_create(self, session):
        state = get_or_create_portfolio(session, initial_capital=750)
        assert state["cash"] == 750

    def test_record_deposit(self, session):
        initialize_portfolio(session, initial_capital=500)
        updated = record_deposit(session, 100, notes="Weekly deposit")
        assert updated["cash"] == 600

    def test_strategy_allocation(self):
        state = {"total_equity": 1000}
        alloc = calculate_strategy_allocation(state, 0.60)
        assert alloc == pytest.approx(600)

    def test_update_position_buy(self, session):
        state = initialize_portfolio(session, initial_capital=1000)
        updated = update_position(session, "SPY", qty_change=2, price=450, current_state=state)
        assert updated["cash"] == pytest.approx(100)  # 1000 - 900
        assert updated["positions"]["SPY"] == 2

    def test_update_position_sell(self, session):
        state = initialize_portfolio(session, initial_capital=100)
        # Manually set a position
        state["positions"] = {"SPY": 2}
        state["cash"] = 100
        updated = update_position(session, "SPY", qty_change=-2, price=450, current_state=state)
        assert updated["cash"] == pytest.approx(1000)  # 100 + 900
        assert "SPY" not in updated["positions"]


class TestCalculateShares:
    def test_basic_calculation(self):
        assert calculate_shares(500, 100) == pytest.approx(5.0)

    def test_fractional_shares(self):
        # $550 / $100 = 5.5 shares (fractional by default)
        assert calculate_shares(550, 100) == pytest.approx(5.5)

    def test_whole_shares_mode(self):
        # With fractional=False, truncates to whole shares
        assert calculate_shares(550, 100, fractional=False) == 5

    def test_small_amount_fractional(self):
        # $100 / $550 = 0.181818 shares — fractional makes this possible
        result = calculate_shares(100, 550)
        assert result == pytest.approx(0.181818, abs=0.001)

    def test_below_min_notional(self):
        # Below $1 minimum notional
        assert calculate_shares(0.50, 450) == 0

    def test_zero_price(self):
        assert calculate_shares(500, 0) == 0

    def test_zero_dollars(self):
        assert calculate_shares(0, 100) == 0


class TestValidateOrder:
    def test_sell_always_approved(self, session, risk_params):
        decision = {"symbol": "SPY", "action": "SELL"}
        state = {"total_equity": 1000}
        result = validate_order(session, risk_params, decision, state)
        assert result["approved"] is True

    def test_skip_passes(self, session, risk_params):
        decision = {"symbol": "SPY", "action": "SKIP"}
        state = {"total_equity": 1000}
        result = validate_order(session, risk_params, decision, state)
        assert result["approved"] is True

    def test_buy_approved_when_clean(self, session, risk_params):
        decision = {"symbol": "SPY", "action": "BUY"}
        state = {"total_equity": 1000, "cash": 200}
        result = validate_order(session, risk_params, decision, state)
        assert result["approved"] is True
        # 90% of $200 cash = $180
        assert result["max_trade_size"] == pytest.approx(180.0)

    def test_kill_switch_blocks(self, session, risk_params):
        os.environ["KILL_SWITCH"] = "true"
        decision = {"symbol": "SPY", "action": "BUY"}
        state = {"total_equity": 1000}
        result = validate_order(session, risk_params, decision, state)
        assert result["approved"] is False
        os.environ.pop("KILL_SWITCH", None)
