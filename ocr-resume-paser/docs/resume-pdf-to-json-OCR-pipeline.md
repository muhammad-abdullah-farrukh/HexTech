# Resume PDF → JSON Extraction Pipeline (Option A: Dynamic Pydantic + Instructor)

End-to-end pipeline: PDF in → triaged extraction (PyMuPDF native text or Marker OCR) → deterministic cleanup → LLM normalization against a runtime-defined schema via `pydantic.create_model()` + `instructor` → final JSON file out.

## 1. Triage — decide native text vs. OCR

```python
import fitz  # PyMuPDF

def needs_ocr(pdf_path: str, sample_pages: int = 3, char_threshold: int = 40) -> bool:
    doc = fitz.open(pdf_path)
    pages_to_check = min(sample_pages, doc.page_count)
    total_chars = 0
    for i in range(pages_to_check):
        page = doc[i]
        text = page.get_text("text")
        total_chars += len(text.strip())
        img_area = sum(
            (b[2] - b[0]) * (b[3] - b[1]) for b in page.get_image_info()
        ) if page.get_images() else 0
        page_area = page.rect.width * page.rect.height
        if img_area / page_area > 0.85 and len(text.strip()) < char_threshold:
            return True
    avg_chars = total_chars / pages_to_check
    return avg_chars < char_threshold
```

No rendering or OCR is run at this stage — it inspects the embedded text layer directly, so the check costs a few milliseconds regardless of document length.

## 2. Extraction — native text (PyMuPDF4LLM) and scanned (Marker)

Native-text resumes are extracted with `pymupdf4llm`, using `page_chunks=True` so each page comes back as its own dict (needed for cross-page boilerplate stripping in step 3).

```python
import pymupdf4llm

def extract_native(pdf_path: str) -> list[str]:
    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    return [chunk["text"] for chunk in chunks]
```

Scanned resumes go through Marker, which runs a layout-segmentation model ahead of OCR so multi-column resumes don't get interleaved into garbage, and emits markdown rather than a flat OCR text blob.

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

_marker_converter = None

def get_marker_converter() -> PdfConverter:
    """Lazily load Marker's model weights once and reuse across requests —
    create_model_dict() is expensive and must not run per-document."""
    global _marker_converter
    if _marker_converter is None:
        _marker_converter = PdfConverter(artifact_dict=create_model_dict())
    return _marker_converter

def extract_scanned(pdf_path: str) -> list[str]:
    converter = get_marker_converter()
    rendered = converter(pdf_path)
    markdown_text, _, _ = text_from_rendered(rendered)
    # Marker returns a single combined markdown string rather than a clean
    # per-page dict, so it is wrapped as a one-element list. Cross-page
    # boilerplate stripping (step 3) is a no-op in this branch, but
    # within-page dedup and whitespace normalization still apply.
    return [markdown_text]
```

```python
def extract_pdf(pdf_path: str) -> tuple[str, list[str]]:
    """Returns (engine_used, pages) so downstream code can tag and compare
    raw output per engine — needed for evaluating Marker vs PyMuPDF4LLM."""
    if needs_ocr(pdf_path):
        return "marker", extract_scanned(pdf_path)
    return "pymupdf4llm", extract_native(pdf_path)
```

## 3. Deterministic cleanup (pure Python, runs before any LLM call)

```python
import re
from collections import Counter
from difflib import SequenceMatcher

def strip_repeated_boilerplate(pages: list[str], min_page_count: int = 2,
                                 repetition_ratio: float = 0.5) -> list[str]:
    def normalize(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    if len(pages) < min_page_count:
        return pages

    line_counts = Counter()
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
    blocks = [b for b in text.split("\n\n") if b.strip()]
    kept = []
    for block in blocks:
        is_dup = any(
            SequenceMatcher(None, block, k).ratio() >= similarity_threshold
            for k in kept[-5:]
        )
        if not is_dup:
            kept.append(block)
    return "\n\n".join(kept)

def merge_split_sentences(text: str) -> str:
    lines = text.split("\n")
    merged = []
    for line in lines:
        stripped = line.strip()
        if (merged and stripped and stripped[0].islower()
                and merged[-1] and merged[-1][-1] not in ".!?:;-•"
                and not merged[-1].startswith(("#", "|", "-", "*"))):
            merged[-1] = merged[-1].rstrip() + " " + stripped
        else:
            merged.append(line)
    return "\n".join(merged)

def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[•●▪]", "-", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()

def clean_extraction(pages: list[str]) -> str:
    pages = strip_repeated_boilerplate(pages)
    text = "\n\n".join(pages)
    text = dedupe_near_identical_blocks(text)
    text = merge_split_sentences(text)
    text = normalize_whitespace(text)
    return text
```

Order matters: boilerplate strip (page-level, catches repeated headers/footers/page numbers) → near-duplicate block collapse (paragraph-level, catches dual-read columns and overlapping OCR regions) → sentence merge (line-level, repairs column-edge line breaks) → whitespace normalize (final pass, run last so it doesn't interfere with the similarity comparisons above it).

## 4. Dynamic schema → Pydantic model

The target schema arrives at runtime as a field-spec list — not a hardcoded class:

```python
field_spec = [
    {"name": "candidate_name", "type": "string", "required": True, "description": "Full legal name"},
    {"name": "email", "type": "string", "required": True, "description": "Primary contact email"},
    {"name": "phone", "type": "string", "required": False, "description": "Primary contact phone number"},
    {"name": "years_experience", "type": "integer", "required": True, "description": "Total professional years"},
    {"name": "skills", "type": "array", "items": "string", "required": True, "description": "Technical skills list"},
    {"name": "work_history", "type": "array", "items": "object", "required": False, "description": "Employment entries",
     "properties": [
         {"name": "company", "type": "string", "required": True},
         {"name": "title", "type": "string", "required": True},
         {"name": "start_date", "type": "string", "required": False},
         {"name": "end_date", "type": "string", "required": False},
     ]},
    {"name": "education", "type": "array", "items": "object", "required": False, "description": "Education entries",
     "properties": [
         {"name": "institution", "type": "string", "required": True},
         {"name": "degree", "type": "string", "required": False},
         {"name": "graduation_year", "type": "string", "required": False},
     ]},
]
```

This recursively compiles to nested `BaseModel` classes, so `work_history` becomes `List[WorkHistoryItem]` (a real validated type, not a raw dict):

```python
from pydantic import create_model, Field
from typing import Optional, List

TYPE_MAP = {"string": str, "integer": int, "number": float, "boolean": bool}

def build_dynamic_model(name: str, fields: list[dict]):
    annotations = {}
    for f in fields:
        if f["type"] == "object" and "properties" in f:
            py_type = build_dynamic_model(f"{name}_{f['name']}", f["properties"])
        elif f["type"] == "array":
            if f.get("items") == "object" and "properties" in f:
                nested = build_dynamic_model(f"{name}_{f['name']}_item", f["properties"])
                py_type = List[nested]
            else:
                py_type = List[TYPE_MAP.get(f.get("items", "string"), str)]
        else:
            py_type = TYPE_MAP.get(f["type"], str)

        if not f.get("required", False):
            py_type = Optional[py_type]
            field_def = (py_type, Field(default=None, description=f.get("description", "")))
        else:
            field_def = (py_type, Field(..., description=f.get("description", "")))

        annotations[f["name"]] = field_def

    return create_model(name, **annotations)
```

## 5. LLM normalization via `instructor`

```python
import instructor
from openai import OpenAI

def extract_structured(clean_markdown: str, field_spec: list[dict],
                        model_name: str = "your-vllm-model",
                        base_url: str = "http://localhost:8000/v1") -> dict:
    DynamicResumeModel = build_dynamic_model("DynamicResumeModel", field_spec)

    client = instructor.from_openai(
        OpenAI(base_url=base_url, api_key="not-needed"),
        mode=instructor.Mode.JSON_SCHEMA,  # maps onto vLLM's native response_format json_schema path
    )

    result = client.chat.completions.create(
        model=model_name,
        response_model=DynamicResumeModel,
        max_retries=2,  # instructor auto-reasks on validation failure
        messages=[
            {"role": "system", "content": (
                "Extract structured resume data strictly according to the provided schema. "
                "Do not invent values not present in the source text. Leave optional fields "
                "null if the information is not present."
            )},
            {"role": "user", "content": clean_markdown},
        ],
    )
    return result.model_dump(exclude_none=True)
```

Use `instructor.Mode.JSON_SCHEMA` against locally hosted vLLM models rather than `Mode.TOOLS` — most open-weight chat models served via vLLM don't reliably emit OpenAI-style tool-call envelopes, but they do honor vLLM's native `response_format={"type": "json_schema", ...}` path, which is what `JSON_SCHEMA` mode targets.

## 6. Full pipeline — PDF in, JSON file out

```python
import json
from pathlib import Path

def save_artifacts(artifacts_dir: str, engine: str, pages: list[str], clean_text: str) -> None:
    """Writes intermediate extraction artifacts to disk so raw Marker/PyMuPDF4LLM
    output can be inspected and compared directly, independent of the final JSON."""
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_text = "\n\n---PAGE BREAK---\n\n".join(pages)
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


def run_pipeline(pdf_path: str, field_spec: list[dict], output_path: str,
                  model_name: str = "your-vllm-model",
                  base_url: str = "http://localhost:8000/v1",
                  artifacts_dir: str | None = None) -> dict:
    engine, pages = extract_pdf(pdf_path)
    clean_text = clean_extraction(pages)

    if artifacts_dir:
        save_artifacts(artifacts_dir, engine, pages, clean_text)

    structured = extract_structured(clean_text, field_spec, model_name, base_url)

    Path(output_path).write_text(
        json.dumps(structured, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return structured


if __name__ == "__main__":
    result = run_pipeline(
        pdf_path="resume.pdf",
        field_spec=field_spec,
        output_path="resume_extracted.json",
        artifacts_dir="artifacts",  # set to None to skip artifact writing
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
```

This is the complete path from a raw PDF (native or scanned) to a clean, deduplicated, schema-conformant JSON file — `extract_pdf()` forks on triage, `clean_extraction()` removes layout-induced duplication before any tokens reach the LLM, and `extract_structured()` validates the model's output against the runtime schema, retrying automatically on a validation failure before it's ever written to disk.

## 7. Evaluating Marker vs. PyMuPDF4LLM from the artifacts

When `artifacts_dir` is set, each run produces three files alongside the final JSON, written into the given directory:

- **`01_raw_{engine}.md`** — the untouched output straight from whichever engine ran (`pymupdf4llm` or `marker`), with `---PAGE BREAK---` markers between pages. Open this directly to check for column-interleaving errors, missing sections, garbled OCR characters, or broken table structure, before any cleanup has touched it.
- **`02_cleaned.md`** — the same text after the dedup layer (boilerplate strip, near-duplicate collapse, sentence merge, whitespace normalize). Diffing this against the raw file shows exactly what the dedup layer removed or repaired, which is the fastest way to spot whether an engine is producing duplicate blocks or split sentences in the first place.
- **`03_extraction_metadata.json`** — `engine_used`, `page_count`, raw/cleaned character counts, and `dedup_reduction_pct`. A high reduction percentage on a given resume is a direct signal that the extraction engine produced redundant or fragmented output for that document — useful for comparing engine quality across a batch of test PDFs without manually reading every file.

To run a side-by-side comparison across a folder of test resumes, loop `run_pipeline()` with a distinct `artifacts_dir` per file (e.g. `artifacts/{pdf_stem}/`) and aggregate the metadata JSONs afterward to see average dedup-reduction and page counts per engine.
