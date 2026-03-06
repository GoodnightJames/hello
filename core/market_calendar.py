"""
Market Calendar — NYSE/NASDAQ holiday awareness.

Prevents the scheduler from running data ingestion, signals, and execution
on market holidays (wasted API calls, stale data, rejected orders).

Also provides next-open and days-until-close utilities for timing decisions.
"""

from datetime import datetime, timedelta, date

from core.logging import get_logger

logger = get_logger("core.market_calendar")

# NYSE observed holidays (fixed dates + computed dates)
# These are the dates the market is CLOSED.
# Source: NYSE holiday calendar (updated annually)
NYSE_FIXED_HOLIDAYS = {
    # (month, day) — New Year's, Juneteenth, Independence Day, Christmas
    (1, 1),    # New Year's Day
    (6, 19),   # Juneteenth
    (7, 4),    # Independence Day
    (12, 25),  # Christmas Day
}


def _nth_weekday(year, month, weekday, n):
    """Get the nth occurrence of a weekday in a month (1-indexed)."""
    d = date(year, month, 1)
    # Find first occurrence of weekday
    while d.weekday() != weekday:
        d += timedelta(days=1)
    # Move to nth occurrence
    d += timedelta(weeks=n - 1)
    return d


def _last_weekday(year, month, weekday):
    """Get the last occurrence of a weekday in a month."""
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def get_nyse_holidays(year):
    """
    Get all NYSE market holidays for a given year.

    Includes:
    - New Year's Day (Jan 1)
    - Martin Luther King Jr. Day (3rd Monday in Jan)
    - Presidents' Day (3rd Monday in Feb)
    - Good Friday (varies — Friday before Easter)
    - Memorial Day (last Monday in May)
    - Juneteenth (Jun 19)
    - Independence Day (Jul 4)
    - Labor Day (1st Monday in Sep)
    - Thanksgiving Day (4th Thursday in Nov)
    - Christmas Day (Dec 25)

    When a holiday falls on Saturday, the preceding Friday is observed.
    When a holiday falls on Sunday, the following Monday is observed.

    Returns:
        Set of date objects.
    """
    holidays = set()

    # Fixed holidays with weekend adjustment
    for month, day in NYSE_FIXED_HOLIDAYS:
        d = date(year, month, day)
        if d.weekday() == 5:  # Saturday → Friday
            d -= timedelta(days=1)
        elif d.weekday() == 6:  # Sunday → Monday
            d += timedelta(days=1)
        holidays.add(d)

    # MLK Day: 3rd Monday in January
    holidays.add(_nth_weekday(year, 1, 0, 3))

    # Presidents' Day: 3rd Monday in February
    holidays.add(_nth_weekday(year, 2, 0, 3))

    # Good Friday: Friday before Easter
    holidays.add(_easter_date(year) - timedelta(days=2))

    # Memorial Day: last Monday in May
    holidays.add(_last_weekday(year, 5, 0))

    # Labor Day: 1st Monday in September
    holidays.add(_nth_weekday(year, 9, 0, 1))

    # Thanksgiving: 4th Thursday in November
    holidays.add(_nth_weekday(year, 11, 3, 4))

    return holidays


def _easter_date(year):
    """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def is_market_open(check_date=None):
    """
    Check if the market is open on a given date.

    Returns False for weekends and NYSE holidays.

    Args:
        check_date: Date to check (defaults to today).

    Returns:
        bool — True if market is open.
    """
    if check_date is None:
        check_date = date.today()

    if isinstance(check_date, datetime):
        check_date = check_date.date()

    # Weekends
    if check_date.weekday() >= 5:
        return False

    # Holidays
    holidays = get_nyse_holidays(check_date.year)
    if check_date in holidays:
        logger.info(
            f"Market closed: {check_date} is a holiday",
        )
        return False

    return True


def next_market_open(from_date=None):
    """
    Get the next date the market is open.

    Args:
        from_date: Starting date (defaults to today).

    Returns:
        date — next market open date.
    """
    if from_date is None:
        from_date = date.today()

    if isinstance(from_date, datetime):
        from_date = from_date.date()

    d = from_date + timedelta(days=1)
    while not is_market_open(d):
        d += timedelta(days=1)
    return d


def trading_days_between(start, end):
    """Count trading days between two dates (exclusive of end)."""
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()

    count = 0
    d = start
    while d < end:
        if is_market_open(d):
            count += 1
        d += timedelta(days=1)
    return count
