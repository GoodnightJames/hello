"""Tests for market calendar — holiday detection and trading day utilities."""

from datetime import date

from core.market_calendar import (
    get_nyse_holidays,
    is_market_open,
    next_market_open,
    trading_days_between,
)


class TestNYSEHolidays:
    def test_new_years_day(self):
        holidays = get_nyse_holidays(2025)
        assert date(2025, 1, 1) in holidays

    def test_mlk_day_2025(self):
        """MLK Day 2025 is Jan 20 (3rd Monday)."""
        holidays = get_nyse_holidays(2025)
        assert date(2025, 1, 20) in holidays

    def test_christmas_on_thursday(self):
        """Christmas 2025 falls on Thursday — observed on Thursday."""
        holidays = get_nyse_holidays(2025)
        assert date(2025, 12, 25) in holidays

    def test_july4_on_saturday_observed_friday(self):
        """When July 4 is Saturday, observed on Friday July 3."""
        # July 4, 2026 is a Saturday
        holidays = get_nyse_holidays(2026)
        assert date(2026, 7, 3) in holidays  # Friday before

    def test_thanksgiving_2025(self):
        """Thanksgiving 2025 is Nov 27 (4th Thursday)."""
        holidays = get_nyse_holidays(2025)
        assert date(2025, 11, 27) in holidays

    def test_good_friday_exists(self):
        """Good Friday should be in the holiday list."""
        holidays = get_nyse_holidays(2025)
        # Good Friday 2025 is April 18
        assert date(2025, 4, 18) in holidays

    def test_reasonable_count(self):
        """NYSE has 9-10 holidays per year."""
        holidays = get_nyse_holidays(2025)
        assert 9 <= len(holidays) <= 11


class TestIsMarketOpen:
    def test_regular_weekday(self):
        """Regular Tuesday should be open."""
        assert is_market_open(date(2025, 3, 4)) is True  # Tuesday

    def test_weekend_saturday(self):
        """Saturday is closed."""
        assert is_market_open(date(2025, 3, 1)) is False

    def test_weekend_sunday(self):
        """Sunday is closed."""
        assert is_market_open(date(2025, 3, 2)) is False

    def test_holiday_closed(self):
        """Christmas is closed."""
        assert is_market_open(date(2025, 12, 25)) is False

    def test_regular_monday(self):
        """Non-holiday Monday is open."""
        assert is_market_open(date(2025, 3, 3)) is True


class TestNextMarketOpen:
    def test_from_friday(self):
        """Next open from Friday is Monday."""
        result = next_market_open(date(2025, 2, 28))  # Friday
        assert result == date(2025, 3, 3)  # Monday

    def test_from_saturday(self):
        """Next open from Saturday is Monday."""
        result = next_market_open(date(2025, 3, 1))  # Saturday
        assert result == date(2025, 3, 3)  # Monday

    def test_from_weekday(self):
        """Next open from Tuesday is Wednesday."""
        result = next_market_open(date(2025, 3, 4))  # Tuesday
        assert result == date(2025, 3, 5)  # Wednesday


class TestTradingDays:
    def test_full_week(self):
        """Monday to Friday = 5 trading days."""
        count = trading_days_between(date(2025, 3, 3), date(2025, 3, 8))
        assert count == 5

    def test_includes_holiday(self):
        """Week with holiday = 4 trading days."""
        # Christmas week 2025: Dec 22-26, Dec 25 is holiday
        count = trading_days_between(date(2025, 12, 22), date(2025, 12, 27))
        assert count == 4
