"""Deterministic backfill for contact fields (email / phone).

Regular, regex-extractable fields like email and phone are more reliably found by
a pattern than by the LLM (DeepSeek-R1 consistently drops the phone even though
it is in the source). After the LLM step, `backfill_contacts` fills any contact
field the model left empty, matched by field name from the field_spec so the
pipeline stays schema-driven. It never overwrites a value the model did provide.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# A phone-like run: optional +, digits and common separators. Validated by digit
# count afterwards so stray numbers (years, house numbers) don't match.
_PHONE_RE = re.compile(r"\+?\d[\d\s().\-]{6,}\d")
_MIN_PHONE_DIGITS = 7
_MAX_PHONE_DIGITS = 15


def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None


def extract_phone(text: str) -> str | None:
    for m in _PHONE_RE.finditer(text):
        candidate = m.group(0).strip()
        digits = sum(c.isdigit() for c in candidate)
        if _MIN_PHONE_DIGITS <= digits <= _MAX_PHONE_DIGITS:
            return candidate
    return None


def _is_empty(value: object) -> bool:
    return value is None or value == "" or value == []


def _top_level_field_names(field_spec: list[dict]) -> list[str]:
    return [f["name"] for f in field_spec]


def backfill_contacts(
    result: dict, source_text: str, field_spec: list[dict]
) -> dict:
    """Fill empty email/phone top-level fields from regex matches on the source.

    Only acts on fields that exist in `field_spec`, are string-typed, and were
    left empty by the model. Returns the same dict (mutated) for convenience.
    """
    names = _top_level_field_names(field_spec)
    string_fields = {
        f["name"] for f in field_spec if f.get("type", "string") == "string"
    }

    for name in names:
        if name not in string_fields or not _is_empty(result.get(name)):
            continue
        lname = name.lower()
        value = None
        if "email" in lname:
            value = extract_email(source_text)
        elif any(k in lname for k in ("phone", "mobile", "tel", "contact_number")):
            value = extract_phone(source_text)
        if value:
            result[name] = value
            logger.info("backfilled contact field %r from source", name)

    return result
