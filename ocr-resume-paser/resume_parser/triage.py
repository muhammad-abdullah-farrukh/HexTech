"""Triage: decide whether a PDF needs OCR (scanned) or has a usable text layer.

Inspects the embedded text layer directly — no rendering or OCR — so the check
costs a few milliseconds regardless of document length.
"""

from __future__ import annotations

import fitz  # PyMuPDF


def needs_ocr(pdf_path: str, sample_pages: int = 3, char_threshold: int = 40) -> bool:
    """Return True if the PDF looks scanned/image-only and should go through OCR.

    Heuristics: a page that is mostly image area with almost no extractable text
    is a scan; failing that, a low average character count across the sampled
    pages indicates no usable text layer.
    """
    doc = fitz.open(pdf_path)
    try:
        pages_to_check = min(sample_pages, doc.page_count)
        if pages_to_check == 0:
            return False

        total_chars = 0
        for i in range(pages_to_check):
            page = doc[i]
            text = page.get_text("text")
            total_chars += len(text.strip())

            # get_image_info() returns a list of dicts; the bbox is under "bbox"
            # as an (x0, y0, x1, y1) tuple (not a positional bbox).
            img_area = 0.0
            for info in page.get_image_info():
                x0, y0, x1, y1 = info["bbox"]
                img_area += (x1 - x0) * (y1 - y0)
            page_area = page.rect.width * page.rect.height
            if (
                page_area > 0
                and img_area / page_area > 0.85
                and len(text.strip()) < char_threshold
            ):
                return True

        avg_chars = total_chars / pages_to_check
        return avg_chars < char_threshold
    finally:
        doc.close()
