"""Intermediate-artifact writer for inspecting/comparing extractor output.

When enabled, each run writes raw per-engine output, the cleaned text, and a
metadata JSON next to the final result, so Marker vs PyMuPDF4LLM quality can be
compared directly and independently of the final JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

PAGE_BREAK = "\n\n---PAGE BREAK---\n\n"


def save_artifacts(
    artifacts_dir: str, engine: str, pages: list[str], clean_text: str
) -> None:
    """Write raw extraction, cleaned text, and dedup metadata to `artifacts_dir`."""
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_text = PAGE_BREAK.join(pages)
    (out / f"01_raw_{engine}.md").write_text(raw_text, encoding="utf-8")
    (out / "02_cleaned.md").write_text(clean_text, encoding="utf-8")

    metadata = {
        "engine_used": engine,
        "page_count": len(pages),
        "raw_char_count": len(raw_text),
        "cleaned_char_count": len(clean_text),
        "chars_removed_by_dedup": len(raw_text) - len(clean_text),
        "dedup_reduction_pct": round(
            (1 - len(clean_text) / max(len(raw_text), 1)) * 100, 2
        ),
    }
    (out / "03_extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def save_structured(artifacts_dir: str, structured: dict) -> None:
    """Write the final structured JSON alongside the extraction artifacts."""
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "04_structured.json").write_text(
        json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8"
    )
