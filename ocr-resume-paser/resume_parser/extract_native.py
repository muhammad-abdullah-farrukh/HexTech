"""Native-text extraction via PyMuPDF4LLM.

`page_chunks=True` returns each page as its own dict so cross-page boilerplate
stripping (cleanup step) has page boundaries to work with.
"""

from __future__ import annotations

import pymupdf4llm


def extract_native(pdf_path: str) -> list[str]:
    """Extract a native-text PDF to a list of per-page markdown strings."""
    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    return [chunk["text"] for chunk in chunks]
