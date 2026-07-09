"""Triage tests — build tiny synthetic PDFs with PyMuPDF and check routing."""

import fitz
import pytest

from resume_parser.triage import needs_ocr


@pytest.fixture
def native_pdf(tmp_path):
    path = tmp_path / "native.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Jane Candidate\nSenior Software Engineer\n"
        "Experience at Acme Corp building distributed systems for many years.\n"
        "Skills: Python, Go, Kubernetes, PostgreSQL, AWS, Terraform.",
        fontsize=11,
    )
    doc.save(path)
    doc.close()
    return str(path)


@pytest.fixture
def empty_pdf(tmp_path):
    """A page with no extractable text layer -> should be routed to OCR."""
    path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()  # blank, no text
    doc.save(path)
    doc.close()
    return str(path)


@pytest.fixture
def scanned_like_pdf(tmp_path):
    """A page that is almost entirely a single image with no text -> OCR.

    Exercises the get_image_info()/"bbox" path that previously crashed.
    """
    path = tmp_path / "scanned.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # A small solid-gray pixmap stretched to cover the whole page (>85% area),
    # with no text layer.
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8))
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    doc.close()
    return str(path)


def test_native_text_pdf_does_not_need_ocr(native_pdf):
    assert needs_ocr(native_pdf) is False


def test_textless_pdf_needs_ocr(empty_pdf):
    assert needs_ocr(empty_pdf) is True


def test_image_only_pdf_needs_ocr(scanned_like_pdf):
    # Must not raise (regression for the get_image_info "bbox" KeyError) and
    # should route a near-full-page image with no text to OCR.
    assert needs_ocr(scanned_like_pdf) is True
