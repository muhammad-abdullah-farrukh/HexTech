"""Empirically determine which instructor mode works for the configured model.

Run BEFORE committing to a mode. Builds the response_model from the REAL
config/field_spec.json (so nested work_history/education arrays-of-objects are
exercised — a mode can pass a toy 2-field schema and still choke here), then
walks the mode ladder against the local LLM and reports, per mode:

  * whether the request was accepted and the result validates
  * which fields are populated vs missing/null  (per-field, not just pass/fail)
  * the raw server response on failure
  * a classification of *why* it failed, distinguishing:
      - schema rejected for containing a "default" key   -> strip defaults, no model change
      - schema rejected for required/additionalProperties -> use --strict schema
      - output present but failed validation              -> per-field errors

Usage:
    python scripts/probe_instructor_mode.py [--env .env] [--field-spec config/field_spec.json]

Each accepted+valid call costs one LLM request.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a bare script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import BadRequestError  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from resume_parser.llm_client import make_client, sampling_kwargs  # noqa: E402
from resume_parser.schema_builder import (  # noqa: E402
    build_dynamic_model,
    load_field_spec,
)
from resume_parser.settings import MODE_LADDER, load_settings  # noqa: E402

SYSTEM_PROMPT = (
    "Extract structured resume data strictly according to the provided schema. "
    "Do not invent values. Leave optional fields null if not present."
)

# A small but realistic resume that exercises nested object arrays.
SAMPLE_RESUME = """\
Jane Q. Candidate
jane.candidate@example.com | (555) 123-4567

Summary
Software engineer with 7 years of professional experience.

Skills
Python, Go, Kubernetes, PostgreSQL, AWS

Experience
Acme Corp — Senior Software Engineer
Jan 2021 - Present
- Led migration to Kubernetes.

Globex Inc — Software Engineer
Jun 2017 - Dec 2020
- Built the billing service.

Education
State University — B.S. Computer Science, 2017
"""


def classify_bad_request(exc: BadRequestError) -> str:
    body = str(exc).lower()
    if "default" in body:
        return ("REJECTED: schema contains 'default' -> fix by stripping default "
                "keys before sending (schema_builder.strip_defaults); no model change")
    if "additionalproperties" in body or "required" in body or "strict" in body:
        return ("REJECTED: required/additionalProperties strictness -> fix with the "
                "--strict schema path (build_dynamic_model strict=True)")
    return "REJECTED: provider refused the request (see raw body above)"


def report_fields(model_cls, obj) -> None:
    data = obj.model_dump()
    populated = [k for k, v in data.items() if v not in (None, [], "")]
    empty = [k for k in data if k not in populated]
    print(f"    populated ({len(populated)}): {', '.join(populated) or '-'}")
    print(f"    missing/empty ({len(empty)}): {', '.join(empty) or '-'}")


def probe_mode(settings, mode: str, model_cls, label: str) -> bool:
    print(f"\n=== mode={mode}  [{label}] ===")
    client = make_client(settings, mode=mode)
    try:
        # max_retries=0: we want first-attempt behaviour, not instructor's re-ask.
        result = client.chat.completions.create(
            model=settings.model,
            response_model=model_cls,
            max_retries=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": SAMPLE_RESUME},
            ],
            **sampling_kwargs(settings),
        )
    except BadRequestError as exc:
        print(f"  raw body: {exc}")
        print(f"  -> {classify_bad_request(exc)}")
        return False
    except ValidationError as exc:
        print("  output returned but FAILED validation; per-field errors:")
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            print(f"    - {loc}: {err['msg']}")
        return False
    except Exception as exc:  # noqa: BLE001 - probe reports anything else verbatim
        print(f"  UNEXPECTED {exc.__class__.__name__}: {exc}")
        return False

    print("  OK — accepted and validated.")
    report_fields(model_cls, result)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probe instructor modes for a slug.")
    ap.add_argument("--env", default=None)
    ap.add_argument("--field-spec", default="config/field_spec.json")
    args = ap.parse_args(argv)

    settings = load_settings(args.env)
    field_spec = load_field_spec(args.field_spec)

    lax_model = build_dynamic_model("ProbeResume", field_spec, strict=False)
    strict_model = build_dynamic_model("ProbeResumeStrict", field_spec, strict=True)

    print(f"model: {settings.model}")
    print(f"base_url: {settings.base_url}")
    print(f"ladder: {MODE_LADDER}")

    working: list[str] = []
    for mode in MODE_LADDER:
        if probe_mode(settings, mode, lax_model, "lax schema"):
            working.append(f"{mode} (lax)")
        elif probe_mode(settings, mode, strict_model, "strict schema"):
            working.append(f"{mode} (strict)")

    print("\n" + "=" * 50)
    if working:
        print("WORKING:", ", ".join(working))
        first = working[0]
        print(f"Set INSTRUCTOR_MODE={first.split()[0]} in .env"
              + (" and use --strict" if "strict" in first else ""))
        return 0
    print("NO mode produced conformant output — reassess model choice.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
