from datetime import date

from resume_parser.experience import (
    backfill_experience,
    compute_total_experience,
    parse_month_year,
)

TODAY = date(2026, 7, 1)

# The Abdullah-zahid roles (two overlap in 2023-2024).
WORK_HISTORY = [
    {"company": "NASTP", "title": "RL Intern", "start_date": "Nov 2025", "end_date": "Present"},
    {"company": "self", "title": "Freelance MLE", "start_date": "Aug 2023", "end_date": "Sept 2024"},
    {"company": "Shocks & Stars", "title": "Lead", "start_date": "July 2023", "end_date": "May 2024"},
]

FIELD_SPEC = [
    {"name": "years_experience", "type": "object", "required": True,
     "properties": [
         {"name": "years", "type": "integer", "required": True},
         {"name": "months", "type": "integer", "required": True},
     ]},
    {"name": "work_history", "type": "array", "items": "object", "required": False,
     "properties": [
         {"name": "company", "type": "string", "required": True},
         {"name": "start_date", "type": "string", "required": False},
         {"name": "end_date", "type": "string", "required": False},
     ]},
]


def test_parse_month_year():
    assert parse_month_year("Nov 2025", TODAY) == date(2025, 11, 1)
    assert parse_month_year("September 2024", TODAY) == date(2024, 9, 1)
    assert parse_month_year("Sept 2024", TODAY) == date(2024, 9, 1)
    assert parse_month_year("Present", TODAY) == TODAY
    assert parse_month_year("2023", TODAY) == date(2023, 1, 1)
    assert parse_month_year("", TODAY) is None
    assert parse_month_year(None, TODAY) is None


def test_compute_total_experience_merges_overlaps():
    # Union: Jul2023-Sep2024 (14mo) + Nov2025-Jul2026 (8mo) = 22mo = 1y10m.
    assert compute_total_experience(WORK_HISTORY, TODAY) == (1, 10)


def test_compute_ignores_unparseable_and_empty():
    assert compute_total_experience([], TODAY) is None
    assert compute_total_experience([{"company": "x"}], TODAY) is None


def test_backfill_experience_sets_object():
    result = {"years_experience": {"years": 2, "months": 4}, "work_history": WORK_HISTORY}
    out = backfill_experience(result, FIELD_SPEC, today=TODAY)
    assert out["years_experience"] == {"years": 1, "months": 10}  # overrides model


def test_backfill_experience_integer_field():
    spec = [
        {"name": "years_experience", "type": "integer", "required": True},
        FIELD_SPEC[1],
    ]
    result = {"years_experience": 99, "work_history": WORK_HISTORY}
    out = backfill_experience(result, spec, today=TODAY)
    assert out["years_experience"] == 1  # whole years only for int field


def test_backfill_noop_without_work_history():
    result = {"years_experience": {"years": 5, "months": 0}}
    out = backfill_experience(result, FIELD_SPEC, today=TODAY)
    assert out["years_experience"] == {"years": 5, "months": 0}  # unchanged
