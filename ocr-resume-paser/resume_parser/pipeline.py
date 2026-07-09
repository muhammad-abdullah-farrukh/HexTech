"""Orchestrator: PDF in -> JSON out.

Flow: triage+extract -> deterministic cleanup -> (optional) artifacts ->
(optional) LLM normalization -> write JSON. With `run_llm=False` it stops after
cleanup/artifacts, which is how extraction quality is validated without an LLM
call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .artifacts import save_artifacts, save_structured
from .cleanup import clean_extraction
from .extract import extract_pdf
from .normalize import extract_structured
from .settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable


def run_pipeline(
    pdf_path: str,
    field_spec: list[dict],
    output_path: str | None = None,
    *,
    settings: Settings | None = None,
    artifacts_dir: str | None = None,
    run_llm: bool = True,
    strict: bool = False,
    ingest_fn: Callable[[dict, str], str] | None = None,
) -> dict | None:
    """Run the full pipeline on one PDF.

    Returns the structured dict when `run_llm` is True, else None (extraction and
    artifacts still run). Requires `settings` when `run_llm` is True. When
    `ingest_fn` is given, the structured dict is persisted via it (e.g. to Postgres)
    after the artifacts are written; the pipeline itself stays database-agnostic.
    """
    engine, pages = extract_pdf(pdf_path)
    clean_text = clean_extraction(pages)

    if artifacts_dir:
        save_artifacts(artifacts_dir, engine, pages, clean_text)

    if not run_llm:
        return None

    if settings is None:
        raise ValueError("settings is required when run_llm=True")

    structured = extract_structured(clean_text, field_spec, settings, strict=strict)

    if artifacts_dir:
        save_structured(artifacts_dir, structured)

    if ingest_fn is not None:
        ingest_fn(structured, pdf_path)

    if output_path:
        Path(output_path).write_text(
            json.dumps(structured, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return structured
