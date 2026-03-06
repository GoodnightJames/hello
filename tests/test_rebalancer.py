"""Tests for idle cash deployment / rebalancer."""

import os

import pytest

from data.db import Base, get_session, init_db
from capital.rebalancer import check_idle_cash_deployment
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


class TestIdleCashDeployment:
    def test_deploys_to_held_position(self, session, risk_params):
        """Should deploy idle cash to an existing position with HOLD signal."""
        portfolio = {"cash": 100.0, "total_equity": 200.0, "positions": {"SPY": 0.2}}
        decisions = [
            {"symbol": "SPY", "action": "HOLD", "signal_type": "HOLD", "signal_strength": 1.2},
        ]
        result = check_idle_cash_deployment(session, portfolio, decisions, risk_params)
        assert len(result) == 1
        assert result[0]["symbol"] == "SPY"
        assert result[0]["action"] == "BUY"
        assert result[0]["signal_strength"] == 1.2

    def test_no_deploy_when_buy_exists(self, session, risk_params):
        """Should not deploy if there's already a BUY decision."""
        portfolio = {"cash": 100.0, "total_equity": 200.0, "positions": {"SPY": 0.2}}
        decisions = [
            {"symbol": "QQQ", "action": "BUY", "signal_type": "BUY"},
            {"symbol": "SPY", "action": "HOLD", "signal_type": "HOLD"},
        ]
        result = check_idle_cash_deployment(session, portfolio, decisions, risk_params)
        assert result == []

    def test_no_deploy_when_no_cash(self, session, risk_params):
        """Should not deploy when cash is too low."""
        portfolio = {"cash": 0.50, "total_equity": 100.0, "positions": {"SPY": 0.2}}
        decisions = [
            {"symbol": "SPY", "action": "HOLD", "signal_type": "HOLD"},
        ]
        result = check_idle_cash_deployment(session, portfolio, decisions, risk_params)
        assert result == []

    def test_no_deploy_when_no_positions(self, session, risk_params):
        """Should not deploy when we don't own any HOLD symbols."""
        portfolio = {"cash": 100.0, "total_equity": 200.0, "positions": {}}
        decisions = [
            {"symbol": "SPY", "action": "HOLD", "signal_type": "HOLD"},
        ]
        result = check_idle_cash_deployment(session, portfolio, decisions, risk_params)
        assert result == []

    def test_no_deploy_when_all_skip(self, session, risk_params):
        """Should not deploy when all decisions are SKIP."""
        portfolio = {"cash": 100.0, "total_equity": 200.0, "positions": {"SPY": 0.2}}
        decisions = [
            {"symbol": "SPY", "action": "SKIP", "signal_type": "BUY"},
        ]
        result = check_idle_cash_deployment(session, portfolio, decisions, risk_params)
        assert result == []
