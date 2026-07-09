"""Deterministic total-experience calculation from work-history dates.

The LLM is unreliable at date math (overlapping roles, "Present"), so total
`years_experience` is computed in Python: parse each role's start/end, merge
overlapping intervals, and report the union duration as {years, months}. This
overrides the model's guess whenever it can be computed from the work history.
"""

from __future__ import annotations

import logging
import re
from datetime import date

logger = logging.getLogger(__name__)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_PRESENT = {"present", "current", "now", "ongoing", "to date", "till date", "date"}
_MONTH_YEAR_RE = re.compile(r"([A-Za-z]{3,})\.?\s*,?\s*(\d{4})")
_YEAR_RE = re.compile(r"\b(\d{4})\b")


def parse_month_year(value: str | None, today: date) -> date | None:
    """Parse 'Nov 2025' / 'September 2024' / 'Present' / '2023' -> date (day=1)."""
    if not value:
        return None
    text = value.strip().lower()
    if any(p in text for p in _PRESENT):
        return today
    m = _MONTH_YEAR_RE.search(text)
    if m:
        mon = _MONTHS.get(m.group(1)[:3])
        if mon:
            return date(int(m.group(2)), mon, 1)
    y = _YEAR_RE.search(text)
    if y:
        return date(int(y.group(1)), 1, 1)  # year only -> assume January
    return None


def _entry_dates(entry: dict, today: date) -> tuple[date, date] | None:
    """Extract (start, end) from a work entry using start/end-ish keys."""
    start_key = next((k for k in entry if "start" in k.lower()), None)
    end_key = next((k for k in entry if "end" in k.lower()), None)
    start = parse_month_year(entry.get(start_key) if start_key else None, today)
    if start is None:
        return None
    end = parse_month_year(entry.get(end_key) if end_key else None, today) or today
    if end < start:
        end = start
    return start, end


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def compute_total_experience(
    work_history: list[dict], today: date
) -> tuple[int, int] | None:
    """Return (years, months) for the union of role date ranges, or None."""
    intervals = [
        d for e in work_history if isinstance(e, dict) and (d := _entry_dates(e, today))
    ]
    if not intervals:
        return None
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:  # overlapping or adjacent -> merge
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    total = sum(_months_between(a, b) for a, b in merged)
    return total // 12, total % 12


def _find_field(field_spec: list[dict], predicate) -> dict | None:
    return next((f for f in field_spec if predicate(f)), None)


def backfill_experience(
    result: dict, field_spec: list[dict], today: date | None = None
) -> dict:
    """Override the experience field with a value computed from work history.

    Detects the experience field (name contains 'experience') and the work-history
    field (array<object> with start/end date properties) from the field_spec.
    Sets {years, months} for an object field, or total years for an integer field.
    No-op if either field or the dates can't be resolved.
    """
    today = today or date.today()
    exp_field = _find_field(field_spec, lambda f: "experience" in f["name"].lower())
    wh_field = _find_field(
        field_spec,
        lambda f: f.get("type") == "array"
        and f.get("items") == "object"
        and any("start" in p["name"].lower() for p in f.get("properties", []))
        and any("end" in p["name"].lower() for p in f.get("properties", [])),
    )
    if not exp_field or not wh_field:
        return result

    work_history = result.get(wh_field["name"])
    if not isinstance(work_history, list) or not work_history:
        return result

    computed = compute_total_experience(work_history, today)
    if computed is None:
        return result

    years, months = computed
    if exp_field.get("type") == "object":
        result[exp_field["name"]] = {"years": years, "months": months}
    else:
        result[exp_field["name"]] = years
    logger.info("computed years_experience from work history: %dy %dm", years, months)
    return result
