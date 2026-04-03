#!/usr/bin/env python3
"""Deterministic tools for inference pipeline.

Called by the inference layer when the model issues a tool_use block.
Zero LLM calls, zero external APIs — pure computation.
"""

import re
from datetime import date
from typing import Literal

from dateutil.relativedelta import relativedelta

# ── Date parsing ──

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

SEASON_MAP = {
    "spring": (3, 1),
    "summer": (6, 1),
    "fall": (9, 1),
    "autumn": (9, 1),
    "winter": (12, 1),
}

# early/mid/late modifiers → month
MODIFIER_MAP = {
    "early": 1,
    "mid": 7,
    "late": 10,
}


def _parse_date(s: str) -> date | None:
    """Parse a date string into a date object, or None if unparseable."""
    s = s.strip()

    # Strip leading "approximately" / "approx" / "~"
    s = re.sub(r"^(approximately|approx\.?|~)\s*", "", s, flags=re.IGNORECASE).strip()

    # ISO: 2021-03-15
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))

    # Range: "June-Aug 2023" → take start of range
    m = re.match(r"([A-Za-z]+)\s*[-–]\s*[A-Za-z]+\s+(\d{4})", s)
    if m:
        month_name = m[1].lower()
        year = int(m[2])
        if month_name in MONTH_MAP:
            return date(year, MONTH_MAP[month_name], 1)

    # Season + year: "summer 2023"
    lower = s.lower()
    for season, (month, day) in SEASON_MAP.items():
        m = re.fullmatch(rf"{season}\s+(\d{{4}})", lower)
        if m:
            return date(int(m[1]), month, day)

    # early/mid/late + Month + Year: "early March 2022"
    m = re.match(r"(early|mid|late)\s+([A-Za-z]+)\s+(\d{4})", s, re.IGNORECASE)
    if m:
        month_name = m[2].lower()
        year = int(m[3])
        if month_name in MONTH_MAP:
            return date(year, MONTH_MAP[month_name], 1)

    # early/mid/late + Year: "early 2022"
    m = re.match(r"(early|mid|late)\s+(\d{4})", s, re.IGNORECASE)
    if m:
        modifier = m[1].lower()
        year = int(m[2])
        return date(year, MODIFIER_MAP[modifier], 1)

    # "Month Day, Year": "March 15, 2021"
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", s)
    if m:
        month_name = m[1].lower()
        day_val = int(m[2])
        year = int(m[3])
        if month_name in MONTH_MAP:
            return date(year, MONTH_MAP[month_name], day_val)

    # "Day Month Year": "15 March 2021"
    m = re.match(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
    if m:
        day_val = int(m[1])
        month_name = m[2].lower()
        year = int(m[3])
        if month_name in MONTH_MAP:
            return date(year, MONTH_MAP[month_name], day_val)

    # "Month Year": "March 2021"
    m = re.match(r"([A-Za-z]+)\s+(\d{4})", s)
    if m:
        month_name = m[1].lower()
        year = int(m[2])
        if month_name in MONTH_MAP:
            return date(year, MONTH_MAP[month_name], 1)

    # Year only: "2021"
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return date(int(m[1]), 1, 1)

    return None


# ── Tools ──

def date_diff(
    date1: str,
    date2: str,
    unit: Literal["days", "weeks", "months", "years"],
) -> dict:
    """Return exact difference between two dates.

    Args:
        date1: First date — ISO (2021-03-15) or natural language.
        date2: Second date — same formats.
        unit:  "days" | "weeks" | "months" | "years"

    Returns:
        {"result": float, "unit": str, "date1_parsed": str, "date2_parsed": str}
        or {"error": str} if either date cannot be parsed.

    Always positive — order of arguments does not matter.
    """
    d1 = _parse_date(date1)
    d2 = _parse_date(date2)

    errors = []
    if d1 is None:
        errors.append(f"Cannot parse date: '{date1}'")
    if d2 is None:
        errors.append(f"Cannot parse date: '{date2}'")
    if errors:
        return {"error": "; ".join(errors)}

    # Ensure d1 <= d2 for consistent calculation
    if d1 > d2:
        d1, d2 = d2, d1

    delta_days = (d2 - d1).days

    if unit == "days":
        result = delta_days
    elif unit == "weeks":
        result = round(delta_days / 7, 1)
    elif unit == "months":
        result = round(delta_days / 30.44, 1)
    elif unit == "years":
        result = round(delta_days / 365.25, 2)
    else:
        return {"error": f"Unknown unit: '{unit}'"}

    return {
        "result": result,
        "unit": unit,
        "date1_parsed": d1.isoformat(),
        "date2_parsed": d2.isoformat(),
    }


def count_items(items: list[str]) -> dict:
    """Count a list of items exactly.

    Args:
        items: List of string items.

    Returns:
        {"count": int, "items": list[str]}
    """
    return {"count": len(items), "items": items}
