"""Tests for trailing stop-loss logic."""

import os
from datetime import datetime

import pytest

from data.db import (
    Base,
    CostBasis,
    PositionHighWater,
    get_engine,
    get_session,
    init_db,
)
from risk.trailing_stop import (
    check_trailing_stops,
    clear_high_water,
    update_high_water_marks,
)
from risk.enforcer import load_risk_params


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


class TestHighWaterMarks:
    def test_initialize_new_position(self, session):
        """First update creates high-water at current price."""
        update_high_water_marks(session, {"SPY": 550.0})
        session.commit()

        hw = session.query(PositionHighWater).filter_by(symbol="SPY").first()
        assert hw is not None
        assert hw.high_price == pytest.approx(550.0)
        assert hw.entry_price == pytest.approx(550.0)

    def test_update_to_new_high(self, session):
        """Price increase updates high-water mark."""
        update_high_water_marks(session, {"SPY": 550.0})
        session.commit()

        update_high_water_marks(session, {"SPY": 580.0})
        session.commit()

        hw = session.query(PositionHighWater).filter_by(symbol="SPY").first()
        assert hw.high_price == pytest.approx(580.0)
        assert hw.entry_price == pytest.approx(550.0)  # Entry unchanged

    def test_price_drop_does_not_lower_high(self, session):
        """Price decrease does NOT change high-water mark."""
        update_high_water_marks(session, {"SPY": 550.0})
        session.commit()

        update_high_water_marks(session, {"SPY": 520.0})
        session.commit()

        hw = session.query(PositionHighWater).filter_by(symbol="SPY").first()
        assert hw.high_price == pytest.approx(550.0)  # Unchanged

    def test_uses_cost_basis_for_entry_price(self, session):
        """When cost basis exists, uses it as entry price."""
        basis = CostBasis(symbol="SPY", qty=1, avg_price=500.0, total_cost=500.0)
        session.add(basis)
        session.commit()

        update_high_water_marks(session, {"SPY": 550.0})
        session.commit()

        hw = session.query(PositionHighWater).filter_by(symbol="SPY").first()
        assert hw.entry_price == pytest.approx(500.0)  # From cost basis
        assert hw.high_price == pytest.approx(550.0)


class TestTrailingStopCheck:
    def test_no_trigger_when_above_stop(self, session, risk_params):
        """Position above stop level should not trigger."""
        # Entry at 500, high at 550 (10% gain), current at 520 (5.5% below high)
        hw = PositionHighWater(
            symbol="SPY", high_price=550.0, entry_price=500.0,
        )
        session.add(hw)
        session.commit()

        # 8% stop from 550 = 506. Current 520 > 506, no trigger
        triggered = check_trailing_stops(session, {"SPY": 520.0}, risk_params)
        assert triggered == []

    def test_trigger_when_below_stop(self, session, risk_params):
        """Position below stop level should trigger sell."""
        # Entry at 500, high at 550 (10% gain), current at 500 (9.1% below high)
        hw = PositionHighWater(
            symbol="SPY", high_price=550.0, entry_price=500.0,
        )
        session.add(hw)
        session.commit()

        # 8% stop from 550 = 506. Current 500 < 506, trigger!
        triggered = check_trailing_stops(session, {"SPY": 500.0}, risk_params)
        assert "SPY" in triggered

    def test_no_trigger_before_min_gain(self, session, risk_params):
        """Stop should not activate until position has gained min_gain (3%)."""
        # Entry at 500, high at 510 (2% gain — below 3% threshold)
        hw = PositionHighWater(
            symbol="SPY", high_price=510.0, entry_price=500.0,
        )
        session.add(hw)
        session.commit()

        # Even if price drops 8% from high (to 469.2), stop should not trigger
        # because gain from entry never reached 3%
        triggered = check_trailing_stops(session, {"SPY": 469.0}, risk_params)
        assert triggered == []

    def test_disabled_returns_empty(self, session):
        """When trailing_stop is disabled, no triggers."""
        hw = PositionHighWater(
            symbol="SPY", high_price=550.0, entry_price=500.0,
        )
        session.add(hw)
        session.commit()

        params = {"trailing_stop": {"enabled": False}}
        triggered = check_trailing_stops(session, {"SPY": 400.0}, params)
        assert triggered == []

    def test_multiple_positions(self, session, risk_params):
        """Check multiple positions, only triggered ones returned."""
        hw1 = PositionHighWater(symbol="SPY", high_price=550.0, entry_price=500.0)
        hw2 = PositionHighWater(symbol="QQQ", high_price=400.0, entry_price=350.0)
        session.add_all([hw1, hw2])
        session.commit()

        # SPY: 8% stop from 550 = 506. Current 505 → triggered
        # QQQ: 8% stop from 400 = 368. Current 390 → not triggered
        triggered = check_trailing_stops(
            session, {"SPY": 505.0, "QQQ": 390.0}, risk_params
        )
        assert "SPY" in triggered
        assert "QQQ" not in triggered


class TestClearHighWater:
    def test_clear_removes_record(self, session):
        """Clearing high-water removes it from DB."""
        hw = PositionHighWater(symbol="SPY", high_price=550.0, entry_price=500.0)
        session.add(hw)
        session.commit()

        clear_high_water(session, "SPY")
        session.commit()

        assert session.query(PositionHighWater).filter_by(symbol="SPY").first() is None

    def test_clear_nonexistent_is_safe(self, session):
        """Clearing a non-existent symbol doesn't error."""
        clear_high_water(session, "NONEXISTENT")
