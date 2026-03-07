"""Tests for virtual sleeve accounting and equal-weight allocation."""

import os

import pytest

from data.db import init_db, get_session, Base
from capital.manager import (
    get_or_create_sleeve,
    get_sleeve_cash,
    get_sleeve_deployable_cash,
    deposit_to_sleeves,
    sleeve_spend,
    sleeve_receive,
    get_sleeve_summary,
    record_deposit,
    SLEEVE_EQUITY,
    SLEEVE_CRYPTO,
)
from decision.engine import assign_target_weights


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


class TestSleeveCreation:
    def test_creates_sleeve_on_first_access(self, session):
        sleeve = get_or_create_sleeve(session, SLEEVE_EQUITY)
        assert sleeve.cash == 0
        assert sleeve.total_deposited == 0

    def test_returns_existing_sleeve(self, session):
        s1 = get_or_create_sleeve(session, SLEEVE_EQUITY)
        s1.cash = 50
        session.flush()
        s2 = get_or_create_sleeve(session, SLEEVE_EQUITY)
        assert s2.cash == 50


class TestDepositSplit:
    def test_splits_100_into_70_30(self, session):
        result = deposit_to_sleeves(session, 100.0)
        assert result[SLEEVE_EQUITY] == 70.0
        assert result[SLEEVE_CRYPTO] == 30.0

    def test_sleeve_cash_reflects_deposit(self, session):
        deposit_to_sleeves(session, 100.0)
        assert get_sleeve_cash(session, SLEEVE_EQUITY) == 70.0
        assert get_sleeve_cash(session, SLEEVE_CRYPTO) == 30.0

    def test_multiple_deposits_accumulate(self, session):
        deposit_to_sleeves(session, 100.0)
        deposit_to_sleeves(session, 100.0)
        assert get_sleeve_cash(session, SLEEVE_EQUITY) == 140.0
        assert get_sleeve_cash(session, SLEEVE_CRYPTO) == 60.0


class TestSleeveSpendReceive:
    def test_spend_deducts_from_sleeve(self, session):
        deposit_to_sleeves(session, 100.0)
        remaining = sleeve_spend(session, SLEEVE_CRYPTO, 1.50)
        assert remaining == pytest.approx(28.50)

    def test_receive_credits_sleeve(self, session):
        deposit_to_sleeves(session, 100.0)
        sleeve_spend(session, SLEEVE_EQUITY, 50.0)
        updated = sleeve_receive(session, SLEEVE_EQUITY, 55.0)
        assert updated == pytest.approx(75.0)  # 70 - 50 + 55

    def test_sleeves_are_independent(self, session):
        deposit_to_sleeves(session, 100.0)
        sleeve_spend(session, SLEEVE_CRYPTO, 10.0)
        # Equity sleeve should be untouched
        assert get_sleeve_cash(session, SLEEVE_EQUITY) == 70.0
        assert get_sleeve_cash(session, SLEEVE_CRYPTO) == 20.0


class TestSleeveDeployableCash:
    def test_deployable_subtracts_buffer(self, session):
        deposit_to_sleeves(session, 100.0)
        deployable = get_sleeve_deployable_cash(session, SLEEVE_EQUITY)
        # 70 - min(1.0, 70*0.05=3.50) = 70 - 1.0 = 69.0
        assert deployable == pytest.approx(69.0)

    def test_zero_cash_returns_zero(self, session):
        deployable = get_sleeve_deployable_cash(session, SLEEVE_CRYPTO)
        assert deployable == 0


class TestSleeveSummary:
    def test_summary_tracks_all_flows(self, session):
        deposit_to_sleeves(session, 100.0)
        sleeve_spend(session, SLEEVE_EQUITY, 30.0)
        sleeve_receive(session, SLEEVE_EQUITY, 5.0)

        summary = get_sleeve_summary(session)
        eq = summary[SLEEVE_EQUITY]
        assert eq["cash"] == pytest.approx(45.0)  # 70 - 30 + 5
        assert eq["total_deposited"] == 70.0
        assert eq["total_spent"] == 30.0
        assert eq["total_received"] == 5.0


class TestEqualWeightAllocation:
    def test_three_buys_get_one_third(self):
        decisions = [
            {"symbol": "SPY", "action": "BUY"},
            {"symbol": "QQQ", "action": "BUY"},
            {"symbol": "XLK", "action": "BUY"},
        ]
        result = assign_target_weights(decisions)
        for d in result:
            assert d["target_weight"] == pytest.approx(0.3333, abs=0.001)

    def test_one_buy_gets_capped_at_40pct(self):
        decisions = [{"symbol": "SPY", "action": "BUY"}]
        result = assign_target_weights(decisions)
        assert result[0]["target_weight"] == 0.40

    def test_two_buys_get_capped_at_40pct(self):
        decisions = [
            {"symbol": "SPY", "action": "BUY"},
            {"symbol": "QQQ", "action": "BUY"},
        ]
        result = assign_target_weights(decisions)
        for d in result:
            assert d["target_weight"] == 0.40

    def test_holds_and_skips_unchanged(self):
        decisions = [
            {"symbol": "SPY", "action": "BUY"},
            {"symbol": "QQQ", "action": "HOLD"},
            {"symbol": "IWM", "action": "SKIP"},
        ]
        result = assign_target_weights(decisions)
        assert result[0]["target_weight"] == 0.40
        assert "target_weight" not in result[1]
        assert "target_weight" not in result[2]

    def test_no_buys_returns_unchanged(self):
        decisions = [{"symbol": "SPY", "action": "HOLD"}]
        result = assign_target_weights(decisions)
        assert "target_weight" not in result[0]
