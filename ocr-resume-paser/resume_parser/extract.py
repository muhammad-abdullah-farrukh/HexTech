"""Extraction fork: triage the PDF, then route to the right engine.

`extract_marker` (and its heavy torch/marker imports) is imported lazily inside
the scanned branch so native-only runs never import the OCR stack.
"""

from __future__ import annotations

from .extract_native import extract_native
from .triage import needs_ocr


def extract_pdf(pdf_path: str) -> tuple[str, list[str]]:
    """Return (engine_used, pages) so downstream code can tag/compare per engine."""
    if needs_ocr(pdf_path):
        from .extract_marker import extract_scanned  # lazy: avoids importing torch

        return "marker", extract_scanned(pdf_path)
    return "pymupdf4llm", extract_native(pdf_path)
