from datetime import date
from photopipe.date_parser import parse_date_from_text, expand_year


def test_expand_two_digit_year_pre_2031():
    assert expand_year("85") == 1985
    assert expand_year("25") == 2025
    assert expand_year("00") == 2000


def test_expand_two_digit_year_never_future():
    from datetime import datetime
    assert expand_year("29") == 1929
    assert expand_year(str((datetime.now().year + 1) % 100)) < datetime.now().year


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


def test_four_digit_year_not_truncated():
    # Regression: "Dec 1985" used to parse as "Dec 19" -> 2019
    results = parse_date_from_text("Dec 1985")
    assert results
    assert results[0][0] == date(1985, 12, 15)
    assert all(r[0].year == 1985 for r in results)

    results = parse_date_from_text("JUN 1985")
    assert results
    assert results[0][0] == date(1985, 6, 15)

    results = parse_date_from_text("Jan 15, 1985")
    assert results
    assert results[0][0] == date(1985, 1, 15)


def test_days_29_to_31_not_clamped():
    # Regression: days used to be clamped to min(day, 28)
    results = parse_date_from_text("June 30, 1985")
    assert results
    assert results[0][0] == date(1985, 6, 30)

    results = parse_date_from_text("Dec. 31, 1999")
    assert results
    assert results[0][0] == date(1999, 12, 31)

    results = parse_date_from_text("07/29/85")
    assert results
    assert results[0][0] == date(1985, 7, 29)


def test_invalid_day_rejected():
    results = parse_date_from_text("February 30, 1985")
    assert not any(r[1] == "full_date" for r in results)


def test_pivot_year_wraps_to_past():
    # Regression: "APR '29" used to expand to 2029 and be dropped
    results = parse_date_from_text("APR '29")
    assert results
    assert results[0][0] == date(1929, 4, 15)


def test_single_digit_numeric_dates():
    for text in ("6/15/85", "6-15-85", "6/15/1985"):
        results = parse_date_from_text(text)
        assert results, text
        assert results[0][0] == date(1985, 6, 15), text


def test_full_month_name_short_year():
    results = parse_date_from_text("JUNE '85")
    assert results
    assert results[0][0] == date(1985, 6, 15)


def test_decade():
    for text in ("1950s", "circa 1950s", "1950's"):
        results = parse_date_from_text(text)
        assert results, text
        assert results[0][0] == date(1955, 6, 15), text
        assert results[0][1] == "decade", text


def test_no_date_returns_empty():
    assert parse_date_from_text("Nothing useful here.") == []


def test_rejects_implausible_year():
    # The year-only regex doesn't match years before 1900 at all
    assert parse_date_from_text("1850") == []
    # And anything it does match must be in range
    results = parse_date_from_text("Mom holding the cat 1985")
    assert results
    assert all(1900 <= r[0].year <= 2100 for r in results)
