#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for serializable temporal values across CE and deep history."""

from __future__ import annotations

import re
from datetime import datetime

import cftime

CF_TIME_UNITS = "days since 0001-01-01"
CF_TIME_CALENDAR = "proleptic_gregorian"
CF_HAS_YEAR_ZERO = False

_DATE_REPR_RE = re.compile(r"^([+-]?\d+)-(\d{2})-(\d{2})$")
_MONTH_NAMES = (
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def format_temporal_date_repr(year: int, month: int, day: int) -> str:
    year = int(year)
    if 0 < year <= 9999:
        year_repr = f"{year:04d}"
    elif -9999 <= year < 0:
        year_repr = f"-{abs(year):04d}"
    else:
        year_repr = str(year)
    return f"{year_repr}-{int(month):02d}-{int(day):02d}"


def parse_temporal_date_repr(value: str | None) -> tuple[int, int, int] | None:
    text = str(value or "").strip()
    match = _DATE_REPR_RE.fullmatch(text)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    if year == 0:
        return None
    return (year, month, day)


def temporal_sort_day_from_parts(year: int, month: int, day: int) -> int | None:
    if int(year) == 0:
        return None
    try:
        value = cftime.date2num(
            cftime.DatetimeProlepticGregorian(
                int(year),
                int(month),
                int(day),
                has_year_zero=CF_HAS_YEAR_ZERO,
            ),
            units=CF_TIME_UNITS,
            calendar=CF_TIME_CALENDAR,
            has_year_zero=CF_HAS_YEAR_ZERO,
        )
    except Exception:
        return None
    return int(value)


def temporal_sort_day_from_repr(value: str | None) -> int | None:
    parts = parse_temporal_date_repr(value)
    if parts is None:
        return None
    return temporal_sort_day_from_parts(*parts)


def temporal_sort_day_from_datetime(value: datetime) -> int:
    return int(
        cftime.date2num(
            value,
            units=CF_TIME_UNITS,
            calendar=CF_TIME_CALENDAR,
            has_year_zero=CF_HAS_YEAR_ZERO,
        )
    )


def temporal_date_repr_from_sort_day(sort_day: int | float) -> str | None:
    try:
        dt = cftime.num2date(
            float(sort_day),
            units=CF_TIME_UNITS,
            calendar=CF_TIME_CALENDAR,
            has_year_zero=CF_HAS_YEAR_ZERO,
        )
    except Exception:
        return None
    return format_temporal_date_repr(int(dt.year), int(dt.month), int(dt.day))


def parse_anchor_sort_day(value: str | None) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    sort_day = temporal_sort_day_from_repr(text)
    if sort_day is not None:
        return sort_day
    try:
        return temporal_sort_day_from_datetime(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except Exception:
        pass
    for pattern in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
        "%H:%M on %d %B, %Y",
        "%H:%M on %d %b, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y-%m-%d",
    ):
        try:
            return temporal_sort_day_from_datetime(datetime.strptime(text, pattern))
        except Exception:
            continue
    return None


def shift_anchor_by_days(anchor_value: str | None, delta_days: int) -> tuple[str, int] | None:
    anchor_sort_day = parse_anchor_sort_day(anchor_value)
    if anchor_sort_day is None:
        return None
    resolved_sort_day = int(anchor_sort_day) + int(delta_days)
    resolved_repr = temporal_date_repr_from_sort_day(resolved_sort_day)
    if resolved_repr is None:
        return None
    return (resolved_repr, resolved_sort_day)


def format_temporal_year_label(date_repr: str | None) -> str | None:
    parts = parse_temporal_date_repr(date_repr)
    if parts is None:
        return None
    year = parts[0]
    if year > 0:
        return str(year)
    return f"{abs(year)} BCE"


def format_temporal_month_label(date_repr: str | None) -> str | None:
    parts = parse_temporal_date_repr(date_repr)
    if parts is None:
        return None
    year, month, _day = parts
    month_name = _MONTH_NAMES[month]
    if year > 0:
        return f"{month_name} {year}"
    return f"{month_name} {abs(year)} BCE"


def format_temporal_day_label(date_repr: str | None) -> str | None:
    parts = parse_temporal_date_repr(date_repr)
    if parts is None:
        return None
    year, month, day = parts
    month_name = _MONTH_NAMES[month]
    if year > 0:
        return f"{month_name} {day}, {year}"
    return f"{month_name} {day}, {abs(year)} BCE"
