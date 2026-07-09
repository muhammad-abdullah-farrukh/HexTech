"""Scanned/OCR extraction via Marker.

Marker's model weights are expensive to load (`create_model_dict()`), so they are
loaded once and cached at module level. This is a lazy singleton: native-only
runs never pay the cost, and a long-running process (CLI or a future service)
reuses the same converter across documents.

Concurrency note: the singleton is NOT safe to initialise from multiple threads
at once. In a service, call `warm_marker()` during single-threaded startup
(e.g. a FastAPI lifespan hook) before serving requests.
"""

from __future__ import annotations

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

_marker_converter: PdfConverter | None = None


def get_marker_converter() -> PdfConverter:
    """Return the process-wide Marker converter, loading weights on first call."""
    global _marker_converter
    if _marker_converter is None:
        _marker_converter = PdfConverter(artifact_dict=create_model_dict())
    return _marker_converter


def warm_marker() -> None:
    """Eagerly load Marker's weights. Call once during single-threaded startup."""
    get_marker_converter()


def extract_scanned(pdf_path: str) -> list[str]:
    """OCR a scanned PDF to markdown.

    Marker returns a single combined markdown string, so it is wrapped as a
    one-element list. Cross-page boilerplate stripping (cleanup) is then a no-op
    for this branch, but within-page dedup and whitespace normalization still apply.
    """
    converter = get_marker_converter()
    rendered = converter(pdf_path)
    markdown_text, _, _ = text_from_rendered(rendered)
    return [markdown_text]
