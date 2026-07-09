"""LLM normalization via instructor against the runtime-defined schema.

Two stages:

  1. Generate — build the dynamic model and extract an initial JSON, walking the
     mode-fallback ladder (try the configured instructor mode; on a rejected
     request or a validation failure that survives instructor's re-asks, drop one
     rung). Transient/rate-limit errors are retried with backoff a level down in
     `completion_with_backoff`.
  2. Self-verify/refine — up to `settings.refine_passes` passes, the model
     re-checks its own JSON against the source and returns `{approved, corrected}`.
     Stop early on approval or convergence. Total LLM calls = 1 + passes run.

The model sees field *names* (grammar-forced) but not descriptions, so a compact
field guide (from the field_spec) is injected into both prompts to restore that
semantic context. If no mode produces conformant output, we raise loudly — a
half-valid result is never returned.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from openai import BadRequestError
from pydantic import BaseModel, ValidationError

from .contacts import backfill_contacts
from .context import fit_review_content, fit_user_content
from .experience import backfill_experience
from .llm_client import (
    completion_with_backoff,
    make_client,
    sampling_kwargs,
)
from .schema_builder import (
    build_dynamic_model,
    build_review_model,
    order_by_spec,
    render_field_guide,
)
from .settings import Settings, fallback_modes

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You extract structured resume data according to the provided schema. "
    "Capture EVERYTHING present in the source text — do not omit, summarize, or "
    "truncate any item that fits a schema field.\n"
    "\n"
    "Rules:\n"
    "1. Split lists into individual elements. A comma-, semicolon-, slash-, "
    "pipe-, or bullet-separated run of values (e.g. a skills line) must become "
    "one array element per value. Never merge multiple distinct values into a "
    "single string. Strip surrounding labels/headers (e.g. 'Tools:') from the "
    "values themselves.\n"
    "2. For employment/experience fields, gather entries from ANY section that "
    "describes roles or work, regardless of heading — e.g. 'Experience', 'Work "
    "History', 'Employment', 'Professional Experience', 'Career Summary', "
    "'Career History', 'Summary', 'Internships', 'Projects', 'Research', "
    "'Positions'. Each dated role with an employer/organization is one entry; "
    "create a separate entry per distinct role even if several share a heading.\n"
    "3. Each skill MUST be a single atomic technology, tool, language, method, or "
    "competency (e.g. 'Verilog', 'FPGA', 'Software Defined Radio'). Never output "
    "a phrase, sentence fragment, or several skills concatenated into one string. "
    "Deduplicate skills that appear more than once. Pull skills from anywhere they "
    "appear — summary, coursework, and role descriptions included.\n"
    "4. Populate the projects field ONLY from sections whose heading denotes "
    "projects or achievements — e.g. 'Projects', 'Technical Projects', 'Academic "
    "Projects', 'Selected Projects', 'Portfolio', 'Achievements', 'Key "
    "Achievements', 'Academic & Work Achievements'. Each distinct item under such "
    "a heading is its own project entry (name, description, technologies), EVEN IF "
    "it was done during a job — be exhaustive within these sections, one entry per "
    "distinct item, don't collapse. Do NOT create projects from the work-history/"
    "experience entries themselves or from a role's routine responsibilities/"
    "duties — those belong ONLY to work_history, not projects.\n"
    "5. years_experience is an object {years, months} — the TOTAL professional "
    "experience. Compute it from the work-history date ranges (e.g. 1 year 10 "
    "months -> {\"years\": 1, \"months\": 10}); months is 0-11.\n"
    "6. Extract ALL contact details from the header/contact line — phone AND "
    "email (and any others). They are often packed onto one line with irregular "
    "spacing/symbols; parse them out individually, don't skip the phone.\n"
    "7. Populate every field for which the information is present anywhere in the "
    "text, even if it appears far from the relevant heading.\n"
    "8. Do NOT invent values that are not in the source text. Leave an optional "
    "field null only when the information is genuinely absent.\n"
    "9. Preserve original wording for names, titles, companies, and dates."
)

VERIFY_SYSTEM_PROMPT = (
    "You are a strict auditor of an extracted resume JSON against the source text. "
    "Work through this checklist and FIX every problem in `corrected`:\n"
    "- projects: every item under a projects/achievements-type heading "
    "('Projects', 'Technical Projects', 'Academic Projects', 'Achievements', 'Key "
    "Achievements', 'Academic & Work Achievements', etc.) is present as its own "
    "entry, EVEN IF job-related; and NO project merely duplicates a work_history "
    "role or a plain job duty (remove those). Be exhaustive within qualifying "
    "sections.\n"
    "- phone AND email are present (parse them from the contact/header line even if "
    "packed together with odd spacing).\n"
    "- years_experience is an object {years, months} with a computed months (0-11), "
    "derived from the work-history date ranges.\n"
    "- every work_history and education entry from the source is captured (none "
    "dropped).\n"
    "- each skill is a single atomic term (no run-on/concatenated strings); "
    "deduplicated.\n"
    "\n"
    "Set approved=true ONLY when nothing on the checklist needs changing. If "
    "anything is wrong you MUST apply the fix in `corrected` — NEVER return "
    "approved=false with an unchanged `corrected`. Never invent data that is not "
    "present in the source text."
)

# Errors that mean "this mode/schema shape doesn't work here" -> try next rung.
_FALLBACK_ERRORS = (BadRequestError, ValidationError)


def _with_guide(base_prompt: str, field_guide: str) -> str:
    """Append the field guide (names/types/descriptions) to a system prompt."""
    return (
        f"{base_prompt}\n\nSchema fields (name (type[, required]): description):\n"
        f"{field_guide}"
    )


def _call_structured(
    settings: Settings,
    response_model: type[BaseModel],
    system: str,
    user: str,
) -> BaseModel:
    """One structured call, walking the mode-fallback ladder with backoff.

    Returns the validated model instance. Raises RuntimeError if no mode works.
    """
    extra = sampling_kwargs(settings)
    last_exc: Exception | None = None
    for mode in fallback_modes(settings.instructor_mode):
        client = make_client(settings, mode=mode)
        try:
            return completion_with_backoff(
                client.chat.completions.create,
                attempts=settings.ratelimit_attempts,
                model=settings.model,
                response_model=response_model,
                max_retries=settings.max_retries,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **extra,
            )
        except _FALLBACK_ERRORS as exc:
            logger.warning("instructor mode %s failed (%s); trying next rung", mode, exc)
            last_exc = exc
            continue

    raise RuntimeError(
        f"No instructor mode produced schema-conformant output from model "
        f"{settings.model!r} at {settings.base_url!r}. Try the strict-schema path "
        f"(--strict) or a different model. Last error: {last_exc!r}"
    ) from last_exc


def run_refine_loop(
    initial: dict,
    refine_fn: Callable[[dict], tuple[bool, dict]],
    max_passes: int,
) -> dict:
    """Iteratively verify/correct `initial`.

    `refine_fn(current)` returns `(approved, corrected)`. Stops when a pass
    approves, when the corrected output stops changing (converged), or when
    `max_passes` is reached. Returns the final dict.
    """
    current = initial
    for i in range(max_passes):
        approved, corrected = refine_fn(current)
        changed = corrected != current
        logger.info(
            "refine pass %d/%d: approved=%s changed=%s", i + 1, max_passes, approved, changed
        )
        current = corrected
        if approved:
            break
        if not changed:
            logger.warning(
                "refine pass %d stopped: not approved but made no changes — "
                "the check flagged an issue it could not fix (returning best effort)",
                i + 1,
            )
            break
    return current


def extract_structured(
    clean_markdown: str,
    field_spec: list[dict],
    settings: Settings,
    *,
    strict: bool = False,
) -> dict:
    """Normalize cleaned resume text into a schema-conformant dict via the LLM.

    Generates an initial extraction, then runs up to `settings.refine_passes`
    self-verify/refine passes (total LLM calls = 1 + passes run).
    """
    resume_model = build_dynamic_model("DynamicResumeModel", field_spec, strict=strict)
    field_guide = render_field_guide(field_spec)
    gen_system = _with_guide(SYSTEM_PROMPT, field_guide)

    # --- Pass 1: generate ---
    user_content, budget = fit_user_content(gen_system, clean_markdown, settings)
    logger.info(
        "generate | model=%s | prompt~%d tok (sys %d + resume %d), reserve %d, ctx %d%s",
        settings.model,
        budget.system_tokens + budget.user_tokens_after,
        budget.system_tokens,
        budget.user_tokens_after,
        budget.reserved_for_output,
        budget.context_window,
        " [TRUNCATED]" if budget.truncated else "",
    )
    initial = _call_structured(settings, resume_model, gen_system, user_content).model_dump(
        exclude_none=True
    )

    if settings.refine_passes <= 0:
        return _finalize(initial, clean_markdown, field_spec)

    # --- Passes 2..N: self-verify/refine ---
    review_model = build_review_model(resume_model)
    verify_system = _with_guide(VERIFY_SYSTEM_PROMPT, field_guide)

    def refine_fn(current: dict) -> tuple[bool, dict]:
        json_text = json.dumps(current, ensure_ascii=False, indent=2)
        review_user, rbudget = fit_review_content(
            verify_system, clean_markdown, json_text, settings
        )
        logger.info(
            "verify | prompt~%d tok (sys %d + source %d + json %d), reserve %d, ctx %d%s",
            rbudget.system_tokens + rbudget.user_tokens_after + rbudget.protected_tokens,
            rbudget.system_tokens,
            rbudget.user_tokens_after,
            rbudget.protected_tokens,
            rbudget.reserved_for_output,
            rbudget.context_window,
            " [SOURCE TRUNCATED]" if rbudget.truncated else "",
        )
        review = _call_structured(settings, review_model, verify_system, review_user)
        return bool(review.approved), review.corrected.model_dump(exclude_none=True)

    final = run_refine_loop(initial, refine_fn, settings.refine_passes)
    return _finalize(final, clean_markdown, field_spec)


def _finalize(result: dict, clean_markdown: str, field_spec: list[dict]) -> dict:
    """Deterministic post-processing: contact backfill, experience math, ordering.

    Uses the FULL cleaned source (not the possibly-truncated prompt).
    """
    result = backfill_contacts(result, clean_markdown, field_spec)
    result = backfill_experience(result, field_spec)
    return order_by_spec(result, field_spec)
