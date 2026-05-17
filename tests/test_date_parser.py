from datetime import date
from photopipe.date_parser import parse_date_from_text, expand_year


def test_expand_two_digit_year_pre_2031():
    assert expand_year("85") == 1985
    assert expand_year("25") == 2025
    assert expand_year("00") == 2000


def test_photo_lab_stamp_month_year_short():
    # "JUN '85"
    results = parse_date_from_text("JUN '85")
    assert results
    assert results[0][0] == date(1985, 6, 15)


def test_full_date():
    results = parse_date_from_text("June 14, 1985")
    assert results
    assert results[0][0] == date(1985, 6, 14)


def test_seasonal():
    results = parse_date_from_text("Summer '92")
    assert results
    assert results[0][0].year == 1992
    assert results[0][0].month == 7  # SEASON_MAP summer


def test_year_only_fallback():
    results = parse_date_from_text("Mom holding the cat 1985")
    # Year-only must still parse
    years = [r[0].year for r in results]
    assert 1985 in years


def test_no_date_returns_empty():
    assert parse_date_from_text("Nothing useful here.") == []


def test_rejects_implausible_year():
    # The year-only regex doesn't match years before 1900 at all
    assert parse_date_from_text("1850") == []
    # And anything it does match must be in range
    results = parse_date_from_text("Mom holding the cat 1985")
    assert results
    assert all(1900 <= r[0].year <= 2100 for r in results)
