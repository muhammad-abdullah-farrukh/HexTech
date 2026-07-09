# Technical Report: Resume PDF-to-JSON Extraction Pipeline

---

## 1. System Overview

This pipeline converts arbitrary resume PDFs into structured, schema-conformant JSON. The architecture is a strict sequential pipeline with three major phases: **extraction** (PDF → markdown text), **cleanup** (deterministic text normalization), and **normalization** (LLM-driven structured extraction with deterministic post-processing). All three phases are independently testable; the LLM phase can be skipped entirely for debugging.

The system is designed around a single local LLM backend (DeepSeek-R1-32B via llama.cpp), a runtime-configurable output schema (`field_spec.json`), and a self-verify/refine loop that caps total LLM calls at three per resume. Deterministic fallbacks (regex, date math) handle fields the LLM reliably drops.

---

## 2. Repository Structure

```
ocr-resume-paser/
├── config/
│   └── field_spec.json          # Runtime output schema definition
├── resume_parser/
│   ├── __init__.py
│   ├── triage.py                # PDF classification: native text vs. scanned
│   ├── extract.py               # Extraction router
│   ├── extract_native.py        # PyMuPDF4LLM path (native text layer)
│   ├── extract_marker.py        # Marker/OCR path (scanned images)
│   ├── cleanup.py               # Deterministic text normalization
│   ├── schema_builder.py        # Dynamic Pydantic model + field guide
│   ├── context.py               # Context-window budgeting
│   ├── settings.py              # Configuration (env-driven)
│   ├── llm_client.py            # LLM client factory + retry logic
│   ├── normalize.py             # LLM extraction + self-verify/refine loop
│   ├── contacts.py              # Deterministic email/phone backfill
│   ├── experience.py            # Deterministic years_experience calculator
│   ├── artifacts.py             # Intermediate artifact writer
│   ├── pipeline.py              # Top-level orchestrator
│   └── cli.py                   # argparse CLI entry point
├── tests/
│   ├── test_cleanup.py
│   ├── test_triage.py
│   ├── test_schema_builder.py
│   ├── test_context.py
│   ├── test_normalize.py
│   ├── test_contacts.py
│   └── test_experience.py
├── resumes/                     # Input PDFs (not committed)
├── artifacts/                   # Per-resume output directory (git-ignored)
├── .env                         # Local secrets and tuning (git-ignored)
├── .env.example                 # Template with empty placeholders
├── conftest.py
└── requirements.txt
```

---

## 3. Configuration Layer

**File:** `resume_parser/settings.py`

All runtime configuration is loaded from environment variables (via `python-dotenv`). The `Settings` dataclass is frozen — it is built once at startup and passed through the pipeline immutably.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | *(required)* | Model name the server serves |
| `LLM_API_KEY` | `not-needed` | Ignored by local servers |
| `INSTRUCTOR_MODE` | `JSON_SCHEMA` | Structured-output decoding mode |
| `LLM_MAX_RETRIES` | `2` | instructor re-ask attempts on validation failure |
| `LLM_RATELIMIT_ATTEMPTS` | `5` | tenacity retry attempts on transient errors |
| `LLM_HEALTH_TIMEOUT` | `2.0` | Seconds for `/models` health check |
| `LLM_REFINE_PASSES` | `2` | Max self-verify/refine passes after generate |
| `LLM_CONTEXT_WINDOW` | `24576` | Total context window (tokens) — must match server `-c` |
| `LLM_CONTEXT_SAFETY_MARGIN` | `256` | Buffer for chat-template overhead |
| `LLM_CHARS_PER_TOKEN` | `3.5` | Token estimate ratio (no tokenizer dependency) |
| `LLM_TEMPERATURE` | `0.6` | DeepSeek-R1's recommended range: 0.5–0.7 |
| `LLM_FREQUENCY_PENALTY` | `0.15` | Suppresses repetition loops |
| `LLM_PRESENCE_PENALTY` | `0.0` | |
| `LLM_MAX_TOKENS` | `4096` | Generation cap — must match server `--n-predict` |

The mode-fallback ladder is `JSON_SCHEMA → TOOLS → JSON → MD_JSON`. If the configured mode fails (due to provider rejection or validation error), the pipeline tries each successive rung before raising.

---

## 4. Runtime Schema: `field_spec.json`

The output schema is defined entirely at runtime as a JSON array of field descriptor objects. This means the output structure can be changed by editing one file — no Python changes needed. The `schema_builder` module compiles this spec into nested Pydantic v2 models at startup.

**Current schema fields:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `candidate_name` | `string` | yes | |
| `email` | `string` | yes | Deterministic regex backfill if LLM drops it |
| `phone` | `string` | no | Deterministic regex backfill if LLM drops it |
| `years_experience` | `object {years, months}` | yes | Always overwritten by deterministic date calculator |
| `skills` | `array<string>` | yes | Each element must be atomic (one tool/language/method) |
| `work_history` | `array<object>` | no | `{company, title, start_date, end_date}` |
| `projects` | `array<object>` | no | `{name, description, technologies[]}` |
| `education` | `array<object>` | no | `{institution, degree, graduation_year}` |

Field descriptor keys: `name`, `type`, `required`, `description`, `items` (for arrays), `properties` (for objects and array-of-objects).

---

## 5. Pipeline Stages

### Stage 1 — Triage (`triage.py`)

**Input:** PDF file path  
**Output:** `bool` (True = needs OCR)

Opens the PDF with PyMuPDF (`fitz`) and samples up to 3 pages. For each page it measures:
- Total extractable character count from the text layer
- Total image area as a fraction of page area (using `get_image_info()` which returns dicts with a `"bbox"` key in PyMuPDF ≥1.26)

A page is flagged as scanned if image area exceeds 85% of page area and text characters are below 40. A document is flagged scanned if average characters per sampled page is below 40. Either condition routes the PDF to OCR.

**Cost:** Milliseconds. No model loading.

---

### Stage 2 — Extraction (`extract.py`, `extract_native.py`, `extract_marker.py`)

**Input:** PDF file path  
**Output:** `(engine_name: str, pages: list[str])`

Two sub-paths exist, selected by triage:

**Native path (PyMuPDF4LLM):** Calls `pymupdf4llm.to_markdown(pdf_path, page_chunks=True)`. Returns each page as a separate markdown string, preserving page boundaries for the cleanup boilerplate stripper. Fast, no model weights.

**OCR path (Marker):** Loads Marker's model weights once (lazy singleton, process-scoped) via `create_model_dict()`. Marker's `PdfConverter` renders the full document to markdown in one call, returned as a single-element list. Model weights are expensive to load (~several seconds on first call); subsequent calls reuse the cached converter. A `warm_marker()` function is exposed for eager loading during server startup.

---

### Stage 3 — Cleanup (`cleanup.py`)

**Input:** `pages: list[str]`  
**Output:** `clean_text: str`

Four deterministic transformations applied in fixed order:

1. **Boilerplate strip** — counts each normalized line across all pages. Lines appearing on ≥50% of pages (minimum 2) are identified as headers/footers/page numbers and dropped from every page. Normalization replaces all digit runs with `#` so "Page 3" and "Page 12" match the same pattern.

2. **Near-duplicate collapse** — splits the joined text into paragraph blocks (double-newline separated). Each block is compared to the 5 most recent kept blocks using `difflib.SequenceMatcher`. Blocks with ≥0.92 similarity to any recent block are dropped. Removes column-layout duplicates and OCR overlaps.

3. **Sentence merge** — re-joins lines that were broken mid-sentence at column edges. A line starting with a lowercase letter, preceded by a line not ending in terminal punctuation or a list marker, is appended to the preceding line.

4. **Whitespace normalize** — standardizes bullet characters to `-`, collapses 3+ consecutive blank lines to 2, and trims trailing spaces from lines.

---

### Stage 4 — Schema Building (`schema_builder.py`)

**Input:** `field_spec: list[dict]`  
**Output:** Pydantic `BaseModel` class (dynamically constructed)

`build_dynamic_model` recursively compiles the field spec to Pydantic v2 models:
- Scalar fields (`string`, `integer`, `number`, `boolean`) map directly via `TYPE_MAP`.
- `object` fields with `properties` become a nested `create_model()` call.
- `array<object>` fields become `List[NestedModel]`.
- `array<scalar>` fields become `List[str/int/...]`.

Required fields use `Field(...)` (no default). Optional fields use `Field(default=None)`. In strict mode, optional fields become `Optional[type]` with `Field(...)` (nullable but still required as a key) and the top-level model sets `extra="forbid"` — producing an OpenAI-strict-compatible schema.

`build_review_model` wraps the resume model in a `ResumeReview` model with two fields: `approved: bool` and `corrected: <resume_model>`. This is the return type for the self-verify pass — one LLM call both judges the extraction and returns the corrected output.

`render_field_guide` converts the field spec to a human-readable block injected into the LLM prompts:
```
- candidate_name (string, required): Full legal name
- years_experience (object, required): Total professional experience ...
    · years (integer, required): Whole years of professional experience
    · months (integer, required): Additional months beyond the whole years (0-11)
```
This is necessary because grammar-constrained decoding (JSON_SCHEMA mode) forces the model to emit the correct field names but strips away the field descriptions — the guide restores that semantic context.

`order_by_spec` reorders a result dict to match the field spec's declaration order. Applied as the final post-processing step so fields added out-of-order by the backfill steps (e.g. `phone`) appear in their schema position rather than appended at the end.

---

### Stage 5 — Context-Window Budgeting (`context.py`)

**Input:** system prompt text, user text, `Settings`  
**Output:** `(user_text: str, BudgetInfo)` — user_text is truncated if needed

Token counts are estimated from character count divided by `chars_per_token` (default 3.5), rounded up. No tokenizer dependency. The safety margin (default 256 tokens) absorbs the estimate's imprecision and chat-template overhead.

**For the generate pass** (`fit_user_content`): available budget = `context_window - max_tokens - safety_margin - system_tokens`. If the resume text exceeds available budget, it is truncated at the character level to fit exactly. The system prompt and generation reserve are never touched. Raises if there is no room for any user text.

**For the verify pass** (`fit_review_content`): the current extracted JSON is protected (never truncated). Available budget excludes the JSON token count. Only the source text is trimmed. Raises if system + JSON + reserve already exceed the window.

The `BudgetInfo` dataclass captures all budget components and is used for the structured log line emitted before each LLM call.

---

### Stage 6 — LLM Normalization (`normalize.py`, `llm_client.py`)

**Overall design:** Up to 3 LLM calls per resume (1 generate + up to 2 verify/refine). The generate pass should be correct enough that verify passes are rarely needed. When verify runs, it must both identify and fix problems in one call.

#### 6a — Generate Pass

System prompt (9 rules) covers:
1. Split all comma/semicolon/pipe/bullet lists into individual array elements.
2. Gather work history from any dated-role section regardless of heading.
3. Skills must be atomic (single technology/tool/language/method); deduplicated.
4. Projects scoped strictly to sections whose heading indicates projects or achievements (e.g. `Projects`, `Technical Projects`, `Key Achievements`, `Academic & Work Achievements`). Items under plain work-history sections are excluded.
5. `years_experience` is `{years, months}` computed from work-history dates.
6. Extract both phone and email from contact lines even with irregular formatting.
7. Populate every field where data is present anywhere in the text.
8. Do not invent data not in the source.
9. Preserve original wording for names, titles, companies, and dates.

The field guide is appended to the system prompt so the model sees field descriptions even though grammar-constrained decoding enforces field names without exposing the schema's description keys.

#### 6b — Self-Verify/Refine Loop

`VERIFY_SYSTEM_PROMPT` defines a strict auditor with an explicit checklist:
- Projects: every item under a qualifying heading is present; no project that merely duplicates a work_history entry.
- Phone and email both present.
- `years_experience` is `{years, months}` with months in range 0–11.
- Every work_history and education entry captured.
- Skills are atomic and deduplicated.

Hard rule: set `approved=true` only when nothing on the checklist needs changing. If anything is wrong, apply the fix in `corrected` — never return `approved=false` with an unchanged `corrected`.

`run_refine_loop` calls the verify function up to `refine_passes` times. Stops early on approval or on convergence (output unchanged between passes). If it exhausts passes without approval or converges unapproved, it logs a warning and returns the best-available output rather than raising.

#### 6c — Mode Fallback Ladder

`_call_structured` walks the mode ladder (`JSON_SCHEMA → TOOLS → JSON → MD_JSON`). On `BadRequestError` (provider rejected the schema) or `ValidationError` (output did not conform after instructor's re-asks), it moves to the next rung. Transient errors (`RateLimitError`, `APIConnectionError`, `APITimeoutError`, `InternalServerError`) are retried with exponential backoff + jitter via tenacity before propagating. If no mode produces conformant output, a `RuntimeError` is raised.

**LLM backend specifics:**  
DeepSeek-R1 is a reasoning model that emits `<think>` blocks before the JSON output. `JSON_SCHEMA` mode is essential: llama.cpp converts the Pydantic-derived JSON schema into a GBNF grammar and enforces it from token 1, preventing the model from emitting a think block that would consume the generation budget before the JSON starts. Temperature must be ~0.6 (the model's official recommendation); temperature 0.0 causes degenerate rambling that fills the token cap without completing the JSON.

---

### Stage 7 — Deterministic Post-Processing

Applied in `_finalize()` after the LLM step, in fixed order:

**Contact backfill (`contacts.py`):** For each field in the spec that is string-typed and whose name contains `email`, `phone`, `mobile`, `tel`, or `contact_number`, if the LLM left the field empty, a regex match on the full cleaned source is tried. Email: `EMAIL_RE` (`[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}`). Phone: `_PHONE_RE` (`\+?\d[\d\s().\-]{6,}\d`) validated by digit count (7–15 digits). Never overwrites a value the LLM provided.

**Experience calculator (`experience.py`):** Parses each work_history entry's start and end dates using a multi-format parser that handles `Nov 2025`, `September 2024`, `Sept 2024`, `Present` / `Current` / `Ongoing`, and bare year `2023`. Builds a list of `(start, end)` date intervals, sorts them, merges overlapping intervals (union-of-intervals), and sums the merged durations in months. Result is `(total_months // 12, total_months % 12)`. Always overwrites the LLM's value whenever a computable result is available. Handles the object `{years, months}` field type as well as a plain integer type.

**Field ordering (`schema_builder.order_by_spec`):** Reorders the result dict's keys to match the field_spec declaration order. Extra keys (not in the spec) are appended after. This ensures the backfilled `phone` field, which was appended at dict-end by the contact backfill, appears in its schema position (after `email`) in the final JSON.

---

### Stage 8 — Artifact Writing (`artifacts.py`)

When `--artifacts-dir` is passed, four files are written to `artifacts/<resume_name>/`:

| File | Content |
|---|---|
| `01_raw_<engine>.md` | Raw per-page extraction output, pages separated by `---PAGE BREAK---` |
| `02_cleaned.md` | Full cleaned text after all cleanup passes |
| `03_extraction_metadata.json` | Engine used, page count, character counts, dedup reduction % |
| `04_structured.json` | Final structured JSON output |

Artifacts 01–03 are written before the LLM call so they are available for debugging even if normalization fails. Artifact 04 is written after.

---

## 6. Data Flow Diagram

```
PDF file
   │
   ▼
[triage.py]
   │  needs_ocr() — samples 3 pages, checks text density + image area ratio
   │
   ├─ native text ──► [extract_native.py]  PyMuPDF4LLM.to_markdown(page_chunks=True)
   │                        │
   └─ scanned ──────► [extract_marker.py]  Marker PdfConverter (lazy singleton)
                            │
   ◄────────────────────────┘
   │  (engine, pages: list[str])
   │
   ▼
[cleanup.py]  clean_extraction()
   │  1. strip_repeated_boilerplate()  — page-level header/footer removal
   │  2. dedupe_near_identical_blocks()  — 0.92 similarity threshold, 5-block window
   │  3. merge_split_sentences()  — repair column-edge line breaks
   │  4. normalize_whitespace()  — bullets, blank lines, trailing spaces
   │
   ▼  clean_text: str
   │
   ├──► [artifacts.py]  save_artifacts()  writes 01_raw, 02_cleaned, 03_metadata
   │
   ▼
[schema_builder.py]  build_dynamic_model() + render_field_guide()
   │  Compiles field_spec.json → Pydantic models at runtime
   │
   ▼
[context.py]  fit_user_content()
   │  Budget: ctx_window − max_tokens − margin − sys_tokens = available for resume
   │  Truncates resume text only if it exceeds budget
   │
   ▼
[llm_client.py + normalize.py]  _call_structured() — GENERATE PASS
   │  instructor JSON_SCHEMA mode → grammar-constrained JSON decoding
   │  Fallback ladder: JSON_SCHEMA → TOOLS → JSON → MD_JSON
   │  tenacity backoff on transient errors; instructor re-ask on validation failure
   │
   ▼  initial: dict
   │
   ▼
[normalize.py]  run_refine_loop()  (up to refine_passes iterations)
   │
   │  ┌── refine_fn(current) ───────────────────────────────────────────────────┐
   │  │  [context.py]  fit_review_content()                                    │
   │  │    Budget: json_text is protected; only source_text may be truncated   │
   │  │  [llm_client.py + normalize.py]  _call_structured() — VERIFY PASS     │
   │  │    Returns {approved: bool, corrected: <resume_model>}                 │
   │  │  Stop if approved=True, or corrected==current (converged)              │
   │  └─────────────────────────────────────────────────────────────────────────┘
   │
   ▼  final: dict
   │
   ▼
[_finalize()]  deterministic post-processing
   │  [contacts.py]  backfill_contacts()  — regex email/phone on full source
   │  [experience.py]  backfill_experience()  — union-of-intervals date math
   │  [schema_builder.py]  order_by_spec()  — reorder keys to spec order
   │
   ▼
   ├──► [artifacts.py]  save_structured()  writes 04_structured.json
   └──► stdout  (JSON, pretty-printed)
```

---

## 7. Testing

42 unit tests across 7 test files. All tests pass with `LLM_MODEL=dummy` (the LLM layer is mocked/bypassed in unit tests).

| Test file | What it covers |
|---|---|
| `test_cleanup.py` | Boilerplate stripping, near-duplicate collapse, sentence merge, whitespace normalization |
| `test_triage.py` | Native-text PDFs stay native; image-only PDFs trigger OCR; mixed cases |
| `test_schema_builder.py` | Dynamic model compilation; field guide rendering with nested properties; `order_by_spec` (spec order, extras appended); `build_review_model` shape |
| `test_context.py` | Token estimation; fit_user_content under/over budget; fit_review_content JSON-protected truncation; raises on no room |
| `test_normalize.py` | `run_refine_loop`: converges, stop-on-approval, stop-on-stuck (no-change warning) |
| `test_contacts.py` | Email and phone backfill; does not overwrite existing values; non-contact fields skipped |
| `test_experience.py` | `parse_month_year` all formats; `compute_total_experience` with overlapping intervals; `backfill_experience` for object and integer field types |

---

## 8. Dependency Stack

**Extraction:**
- `pymupdf` 1.28 — triage + native extraction
- `pymupdf4llm` 1.28 — markdown conversion of native PDFs
- `marker-pdf` 1.10 — OCR for scanned PDFs (pulls `torch`, `surya-ocr`, `transformers`)

**LLM interface:**
- `openai` 2.44 — OpenAI-compatible HTTP client
- `instructor` 1.15 — structured output wrapper (Pydantic validation + mode management)
- `tenacity` 9.1 — retry with exponential backoff

**Schema and validation:**
- `pydantic` 2.13 — model compilation, runtime validation, JSON schema emission

**Configuration:**
- `python-dotenv` 1.2 — `.env` loading

**Testing:**
- `pytest` 9.1

**LLM server (external, not a Python dependency):**
- llama.cpp / llama-server — serves DeepSeek-R1-32B at `http://localhost:8080/v1`
- Config: `-ngl 99 -c 24576 -b 512 -ub 512 --flash-attn on -ctk q8_0 -ctv q8_0 --parallel 1 --n-predict 4096`

---

## 9. Known Limitations and Open Issues

**1. RST Moto work entry date field pollution**  
The model extracted the location (`Sialkot, Pakistan`) into `start_date` for the RST Moto work entry, leaving no end date. This entry is therefore excluded from the union-of-intervals experience calculation, causing an undercount of `years_experience` for that resume.

**2. Skills atomicity drift under refine passes**  
The verify pass can add additional skills scraped from role description text. On one run, the skills list expanded to include phrase-like entries (`tactical aerial coordination`, `shared reward signals`) that are not atomic technology names. The generate prompt explicitly instructs against this, but the verify prompt's checklist doesn't penalize it.

**3. Scanned path not end-to-end tested**  
Marker's OCR path (`extract_marker.py`) has not been exercised on a real scanned resume in the current setup. The code path is correct but the quality of Marker's output on borderline PDFs (mixed native/scanned, complex layouts) is unknown.

**4. Year-only start dates mapped to January**  
`parse_month_year` maps a bare year string (e.g. `2023`) to January 1st of that year. For `Freelance (2023 – present)` this overstates experience by up to 11 months. If multiple roles start in the same year with only year-level precision, merged intervals may absorb noise.

**5. Projects count is heading-driven**  
The number of projects extracted depends entirely on whether the resume uses recognizable achievement-type section headings. A resume with no explicit "Projects" or "Key Achievements" heading will produce an empty projects list even if the candidate has significant project experience described under experience entries.

**6. No batch/parallel processing**  
The pipeline is single-threaded and processes one PDF per invocation. Batching was handled by a shell loop in the CLI, not by the pipeline itself.

---

## 10. Database Ingestion — Recommendation

### Where to Insert

The natural insertion point is in `pipeline.py`, immediately after `extract_structured()` returns the `structured` dict and before (or alongside) `save_structured()`. The pipeline currently returns the dict to the caller; the caller (CLI or a wrapper script) is where the database write should live. This keeps the pipeline itself database-agnostic — it produces a Python dict, and the caller decides what to do with it.

A thin function with this signature is all that is needed:

```python
def ingest_resume(structured: dict, source_pdf_path: str, metadata: dict) -> str:
    """Write one structured resume dict to the database. Returns a record ID."""
    ...
```

### Schema Mapping Strategy

The structured output is a nested JSON document. Two approaches suit different database families:

**Document store (flat JSON ingestion):**  
Insert the `structured` dict directly as a document. The top-level fields become document fields. Arrays (`skills`, `work_history`, `projects`, `education`) are stored as nested arrays within the document. `years_experience` is stored as a sub-document `{years, months}`. Each resume document should carry:
- The `structured` dict as-is
- `source_file`: the original PDF filename
- `ingested_at`: an ISO-8601 timestamp (stamped by the ingest function, not the pipeline)
- `pipeline_version`: a short string identifying the field_spec and model version used

**Relational store (normalized tables):**  
A normalized design splits the nested arrays into joined tables:

```
resumes (id, candidate_name, email, phone, years, months, ingested_at, source_file)
skills  (resume_id, skill)
work_history (resume_id, company, title, start_date, end_date)
projects (resume_id, name, description)
project_technologies (project_id, technology)
education (resume_id, institution, degree, graduation_year)
```

`years_experience.years` and `years_experience.months` become flat integer columns on the `resumes` table. Skills are one row per element. `project_technologies` is a child of `projects`. This structure supports SQL aggregation (skills frequency, experience distribution, graduation year filtering) without JSON traversal.

### Upsert vs. Insert

Because the same PDF may be reprocessed after prompt or schema changes, the ingest function should support upsert keyed on a stable identifier. A good natural key is a hash of the raw PDF bytes (e.g. SHA-256 of the file), stored alongside the record. On reprocess, the existing record is replaced, not duplicated. The CLI already passes the PDF path to `run_pipeline`, so computing the hash there and threading it through to `save_structured` / the ingest function requires minimal change.

### Integration Pattern

```python
# In pipeline.py — minimal change
def run_pipeline(..., ingest_fn=None) -> dict | None:
    ...
    structured = extract_structured(...)
    if artifacts_dir:
        save_structured(artifacts_dir, structured)
    if ingest_fn is not None:
        ingest_fn(structured, pdf_path)
    return structured
```

The `ingest_fn` parameter is optional and defaults to `None`, so existing callers (CLI, tests) are unaffected. The ingest function is injected by the caller — a CLI flag like `--db-uri` can construct and pass it; a web service can pass its own connection-aware closure. This keeps the pipeline layer free of any database import and makes the ingest function independently testable with a mock.

### Connection Management

For a CLI-driven batch use case, open and close the connection in the CLI's `main()` using a context manager, passing a bound method or closure as `ingest_fn` to `run_pipeline`. For a long-running service, maintain a connection pool at the application level and inject a pool-bound write function.

### Schema Versioning

As `field_spec.json` evolves (new fields, type changes), stored records from different versions will have different shapes. Store `field_spec_hash` (SHA-256 of `field_spec.json` contents) alongside each record. This makes it possible to filter records by schema version, reprocess only old-version records, and avoid silently mixing incompatible shapes in queries.
