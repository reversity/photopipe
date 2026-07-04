"""Pure date-string parsing. No I/O, no OCR dependencies."""

import re
from datetime import date, datetime


# Date patterns to recognize (ordered by specificity/reliability)
DATE_PATTERNS = [
    # Photo lab stamps (most common, reliable)
    (r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*['’]?\s*(\d{2})(?![,\s]*\d)", "month_year_short"),
    (r"(January|February|March|April|May|June|July|August|September|October|November|December)\s*['’]?\s*(\d{2})(?![,\s]*\d)", "month_year_short_full"),
    (r"(?<!\d)(\d{2})\s+(\d{2})\s+(\d{2,4})(?!\d)", "numeric_spaced"),
    (r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{2,4})(?!\d)", "numeric_slash"),
    (r"(?<!\d)(\d{1,2})-(\d{1,2})-(\d{2,4})(?!\d)", "numeric_dash"),

    # Full date formats
    (r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})", "full_date"),
    (r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})", "abbrev_date"),

    # Month and year only
    (r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", "month_year_full"),
    (r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{4})", "month_year_abbrev"),

    # Contextual dates
    (r"(Summer|Spring|Fall|Autumn|Winter|Xmas|Christmas|Easter|Thanksgiving)\s*['’]?\s*(\d{2,4})(?!\d)", "seasonal"),

    # Decades ("1950s", "circa 1950s")
    (r"\b(19\d0|20[0-2]0)\s*['’]?s\b", "decade"),

    # Year only (last resort)
    (r"\b(19\d{2}|20[0-2]\d)\b", "year_only"),
]

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

SEASON_MAP = {
    "spring": (3, 15),
    "summer": (7, 15),
    "fall": (10, 15),
    "autumn": (10, 15),
    "winter": (1, 15),
    "xmas": (12, 25),
    "christmas": (12, 25),
    "easter": (4, 15),
    "thanksgiving": (11, 25),
}


def expand_year(year_str: str) -> int:
    """
    Expand 2-digit year to 4-digit year.

    Assumes years 00-30 are 2000s, 31-99 are 1900s, shifting back a
    century if the result would land in the future.
    """
    year = int(year_str)
    if year < 100:
        if year <= 30:
            year += 2000
        else:
            year += 1900
        if year > datetime.now().year:
            year -= 100
    return year


def parse_date_from_text(text: str) -> list[tuple[date, str, str]]:
    """
    Extract dates from OCR text using pattern matching.

    Returns:
        List of (date, pattern_type, raw_match) tuples
    """
    results = []

    for pattern, pattern_type in DATE_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            try:
                parsed_date = None

                if pattern_type in ("month_year_short", "month_year_short_full"):
                    # "JUN '85" or "JUNE '85"
                    month_str = match.group(1).lower()[:3]
                    year = expand_year(match.group(2))
                    month = MONTH_MAP.get(month_str, 6)
                    parsed_date = date(year, month, 15)

                elif pattern_type in ("numeric_spaced", "numeric_slash", "numeric_dash"):
                    # "06 15 85" or "06/15/85" or "06-15-85"
                    parts = [int(match.group(i)) for i in range(1, 4)]
                    # Assume MM DD YY format (American)
                    month, day, year = parts[0], parts[1], expand_year(str(parts[2]))
                    if month > 12:
                        # Might be DD MM YY format
                        month, day = day, month
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        parsed_date = date(year, month, day)

                elif pattern_type == "full_date":
                    # "January 15, 1985"
                    month_str = match.group(1).lower()
                    day = int(match.group(2))
                    year = int(match.group(3))
                    month = MONTH_MAP.get(month_str, 6)
                    parsed_date = date(year, month, day)

                elif pattern_type == "abbrev_date":
                    # "Jan. 15, 1985"
                    month_str = match.group(1).lower()[:3]
                    day = int(match.group(2))
                    year = int(match.group(3))
                    month = MONTH_MAP.get(month_str, 6)
                    parsed_date = date(year, month, day)

                elif pattern_type in ("month_year_full", "month_year_abbrev"):
                    # "January 1985" or "Jan. 1985"
                    month_str = match.group(1).lower()[:3]
                    year = int(match.group(2))
                    month = MONTH_MAP.get(month_str, 6)
                    parsed_date = date(year, month, 15)

                elif pattern_type == "seasonal":
                    # "Summer '85"
                    season = match.group(1).lower()
                    year = expand_year(match.group(2))
                    month, day = SEASON_MAP.get(season, (6, 15))
                    parsed_date = date(year, month, day)

                elif pattern_type == "decade":
                    # "1950s"
                    year = int(match.group(1)) + 5
                    parsed_date = date(year, 6, 15)

                elif pattern_type == "year_only":
                    # "1985"
                    year = int(match.group(1))
                    parsed_date = date(year, 6, 15)

                if parsed_date and 1900 <= parsed_date.year <= datetime.now().year:
                    results.append((parsed_date, pattern_type, match.group(0)))

            except (ValueError, AttributeError):
                continue

    return results
