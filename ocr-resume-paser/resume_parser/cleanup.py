"""Deterministic cleanup — pure Python, runs before any LLM call.

Order matters and is fixed in `clean_extraction`:
  1. boilerplate strip   (page-level: repeated headers/footers/page numbers)
  2. near-duplicate collapse (paragraph-level: dual-read columns, OCR overlaps)
  3. sentence merge      (line-level: repairs column-edge line breaks)
  4. whitespace normalize (final pass; last so it doesn't perturb the similarity
     comparisons above it)
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher


def strip_repeated_boilerplate(
    pages: list[str], min_page_count: int = 2, repetition_ratio: float = 0.5
) -> list[str]:
    """Drop lines that repeat across many pages (headers/footers/page numbers)."""

    def normalize(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    if len(pages) < min_page_count:
        return pages

    line_counts: Counter[str] = Counter()
    page_lines = [p.splitlines() for p in pages]
    for lines in page_lines:
        line_counts.update({normalize(l) for l in lines if l.strip()})

    threshold = max(2, int(len(pages) * repetition_ratio))
    boilerplate = {l for l, c in line_counts.items() if c >= threshold}

    return [
        "\n".join(l for l in lines if normalize(l) not in boilerplate)
        for lines in page_lines
    ]


def dedupe_near_identical_blocks(text: str, similarity_threshold: float = 0.92) -> str:
    """Collapse paragraph blocks that are near-duplicates of a recent block."""
    blocks = [b for b in text.split("\n\n") if b.strip()]
    kept: list[str] = []
    for block in blocks:
        is_dup = any(
            SequenceMatcher(None, block, k).ratio() >= similarity_threshold
            for k in kept[-5:]
        )
        if not is_dup:
            kept.append(block)
    return "\n\n".join(kept)


def merge_split_sentences(text: str) -> str:
    """Re-join lines broken mid-sentence at column edges."""
    lines = text.split("\n")
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if (
            merged
            and stripped
            and stripped[0].islower()
            and merged[-1]
            and merged[-1][-1] not in ".!?:;-•"
            and not merged[-1].startswith(("#", "|", "-", "*"))
        ):
            merged[-1] = merged[-1].rstrip() + " " + stripped
        else:
            merged.append(line)
    return "\n".join(merged)


def normalize_whitespace(text: str) -> str:
    """Normalize bullets and collapse excess blank lines / trailing spaces."""
    text = re.sub(r"[•●▪]", "-", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def clean_extraction(pages: list[str]) -> str:
    """Run the full cleanup pipeline over per-page text and return one string."""
    pages = strip_repeated_boilerplate(pages)
    text = "\n\n".join(pages)
    text = dedupe_near_identical_blocks(text)
    text = merge_split_sentences(text)
    text = normalize_whitespace(text)
    return text
