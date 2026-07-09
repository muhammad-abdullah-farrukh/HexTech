implementation Plan — Resume PDF → JSON Pipeline (Option A) on OpenRouter/Nemotron
Context
This is a greenfield project. The only artifact today is the spec resume-pdf-to-json-option-a-pipeline.md, which describes an end-to-end pipeline: PyMuPDF triage → PyMuPDF4LLM (native) / Marker (scanned) extraction → deterministic dedup cleanup → dynamic Pydantic model (pydantic.create_model) + instructor LLM normalization → JSON out, with an optional artifacts_dir for per-engine inspection.

The one substantive deviation from the spec: the LLM backend is not a local no-auth vLLM server. It is NVIDIA Nemotron 3 Ultra via OpenRouter — an OpenAI-compatible API at https://openrouter.ai/api/v1, model slug nvidia/nemotron-3-ultra-550b-a55b:free, authenticated with an OPENROUTER_API_KEY bearer token. This changes auth, error/rate-limit handling, and — critically — makes the instructor structured-output mode something to verify empirically rather than assume, because OpenRouter routes the same slug across multiple backing providers whose structured-output support varies.

Confirmed choices (from clarifying questions): pip + requirements.txt; CLI now but service-ready (Marker loader as a reusable module-level singleton); external config files (field_spec as JSON, secrets/model slug in .env).

This document is plan only — no implementation code is written yet.

1. Project structure
Flat package at repo root (no pip install -e, run via python -m):

ocr-resume-paser/
├── requirements.txt
├── .env.example                # OPENROUTER_API_KEY=, model slug, base_url
├── .gitignore                  # .env, artifacts/, __pycache__, *.json outputs
├── config/
│   └── field_spec.json         # the runtime schema (the spec's field_spec, as JSON)
├── resume_parser/
│   ├── __init__.py
│   ├── settings.py             # load .env -> typed config (key, slug, base_url, limits)
│   ├── triage.py               # needs_ocr()
│   ├── extract_native.py       # extract_native()  [pymupdf4llm]
│   ├── extract_marker.py       # get_marker_converter() singleton + extract_scanned()
│   ├── extract.py              # extract_pdf() — forks on triage, returns (engine, pages)
│   ├── cleanup.py              # strip_repeated_boilerplate / dedupe / merge / normalize / clean_extraction
│   ├── schema_builder.py       # TYPE_MAP, build_dynamic_model(), load_field_spec()
│   ├── llm_client.py           # OpenRouter client factory + retry/rate-limit wrapper
│   ├── normalize.py            # extract_structured() [instructor]
│   ├── artifacts.py            # save_artifacts()
│   └── pipeline.py             # run_pipeline() orchestrator
├── cli.py                      # argparse entry point -> run_pipeline()
├── scripts/
│   └── probe_instructor_mode.py  # empirical mode verification (see §4)
└── tests/
    ├── samples/                # native + scanned sample PDFs
    ├── test_triage.py
    ├── test_cleanup.py
    └── test_schema_builder.py
Why this split vs. the spec's single file: the spec is one module for readability; splitting by responsibility keeps the Marker singleton, the OpenRouter client, and the deterministic cleanup independently testable and importable by a future service. Pure functions from the spec (triage, cleanup, schema_builder, artifacts) move over essentially verbatim.

Config locations:

OPENROUTER_API_KEY, model slug, base_url, rate-limit knobs → .env (loaded once in settings.py via python-dotenv; never hardcoded).
field_spec → config/field_spec.json, loaded by load_field_spec().
.env and config/ paths overridable via CLI flags.
2. Dependency setup (requirements.txt)
# PDF / extraction
pymupdf            # fitz — triage
pymupdf4llm        # native-text markdown extraction
marker-pdf         # scanned/OCR extraction (pulls torch, surya-ocr, transformers)

# LLM structured output
openai>=1.0        # v1 SDK (OpenAI client class used by the spec)
instructor         # structured output / re-ask
pydantic>=2.5      # create_model with (type, FieldInfo) tuples — v2 required

# config / robustness
python-dotenv      # .env loading
tenacity           # rate-limit backoff (see §4)
Version-compatibility notes worth pinning after a first clean install:

pydantic v2 is mandatory across instructor, marker-pdf, and the spec's create_model(...) tuple syntax. Pin pydantic>=2.5.
marker-pdf is the heavy constraint. It transitively pins torch, transformers, and surya-ocr to fairly tight ranges. Strategy: install marker-pdf first, let it resolve torch/transformers, then install the rest and pin the whole resolved set with pip freeze into requirements.txt. Do not pre-pin torch/transformers yourself — let marker drive them.
The spec's Marker imports (marker.converters.pdf.PdfConverter, marker.models.create_model_dict, marker.output.text_from_rendered) are the marker-pdf v1.x API — pin marker-pdf>=1.0.
openai>=1.0 is required (spec uses the OpenAI(...) client class, not the legacy 0.x module API). instructor recent versions track the v1 SDK.
Python 3.10.12 (system) satisfies marker-pdf's >=3.10 floor. Use a fresh venv; torch + surya weights are large (multi-GB download on first Marker run).
GPU optional: Marker runs on CPU but is much slower; note for testing.
3. Marker model-loading lifecycle
Keep the spec's lazy module-level singleton, isolated in extract_marker.py:

# extract_marker.py  (shape, not final code)
_marker_converter = None
def get_marker_converter():
    global _marker_converter
    if _marker_converter is None:
        _marker_converter = PdfConverter(artifact_dict=create_model_dict())
    return _marker_converter
Service-ready without over-building: because the loader is a plain module function with module-level cached state, a future FastAPI app can call get_marker_converter() once in a startup/lifespan hook to warm it, and the CLI gets the same lazy-load for free. No worker framework needed now.
create_model_dict() runs at most once per process; the CLI loads it only when a PDF actually triages to the scanned branch (native-only runs never pay the cost).
Document the trade-off in code: the singleton is not safe to re-init concurrently; first call should happen during single-threaded startup (warm-up helper warm_marker() provided for the service path).
4. OpenRouter integration
Client factory (llm_client.py):

client = instructor.from_openai(
    OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,          # from .env, required
        default_headers={                              # optional OpenRouter ranking headers
            "HTTP-Referer": settings.app_url,
            "X-Title": "resume-pdf-to-json",
        },
    ),
    mode=settings.instructor_mode,                     # chosen empirically — see below
)
API key handling: settings.py reads OPENROUTER_API_KEY via python-dotenv; raise a clear error at startup if missing. Never log the key. .env is git-ignored; ship .env.example with empty placeholder.

Structured-output mode — verify, don't assume (the key risk): OpenRouter fans the same slug across providers with differing support for tools vs. response_format: json_schema. So:

Add scripts/probe_instructor_mode.py: run against the live slug across an ordered ladder [TOOLS, JSON_SCHEMA, JSON, MD_JSON], printing which modes return a valid, schema-conformant object and which raise/validate-fail. Run this before committing to a mode; record the winner in .env (INSTRUCTOR_MODE).
Probe against the REAL nested schema, not a toy. Build the probe's response_model from the actual config/field_spec.json (so it includes the arrays-of-objects: work_history, education). A mode can pass a trivial 2-field test and still choke once nested object arrays are involved. The probe must report which fields fail, not just pass/fail per mode — dump the raw provider response and the per-field validation errors so a partial/coerced result is visible.
Read the probe errors carefully — two distinct failure modes, two fixes. Pydantic v2 already emits Optional[X]=None as {"anyOf":[{"type":X},{"type":"null"}],"default":null}, so the nullable typing is mostly handled. The two things to tell apart in the probe output: (i) a field dropped/coerced because it wasn't in required → fix in the schema builder (emit strict-compatible all-required schema); vs. (ii) the whole request rejected because the "default" key is present — several hosted backends reject any schema containing default, independent of the required/additionalProperties strictness → fix by stripping default keys from the emitted JSON schema before sending (no model-structure change). Don't conflate them; an unexpected probe error is most likely (ii).
Strict-schema provider risk (explicit check). Some hosted structured-output backends enforce OpenAI-strict semantics under the hood — additionalProperties: false, all properties listed as required, and no implicit optionals. Against such a provider the Optional[X] = None / defaulted fields produced by create_model() can be silently rejected or coerced wrong, even though instructor's default JSON_SCHEMA mode assumes laxer semantics. The probe must surface this specifically (look for optional fields being dropped, defaulted away, or hard-rejected). Mitigations to keep ready if it bites: (a) emit a strict-compatible schema — mark every field required and model "absent" as an explicit nullable type (Optional rendered as anyOf: [..., {"type":"null"}]) rather than a defaulted-omittable field; (b) generate the model with model_config = ConfigDict(extra="forbid") so the emitted schema carries additionalProperties: false to match the provider's expectation; (c) if a strict provider still mangles nested arrays, fall back to a non-strict mode (JSON/MD_JSON) + instructor re-ask. Decide based on what the probe actually shows per field.
Force OpenRouter to only route to providers that honor the requested params: pass extra_body={"provider": {"require_parameters": True}} on the call. This prevents silent routing to a backend that ignores response_format.
Keep the spec's max_retries=2 on instructor for validation re-asks, and in normalize.py add a mode-fallback ladder: if the configured mode raises a provider/validation error, retry the call one rung down the ladder before giving up. This is the robustness answer to "if the chosen mode doesn't reliably produce valid output on the first attempt."
Rate-limit / error handling: free-tier (:free) slugs are aggressively limited (per-minute throttle + a daily request cap tied to account credit).

Wrap the LLM call with tenacity exponential backoff that retries on openai.RateLimitError (HTTP 429) and transient APIError/timeouts, honoring Retry-After when present; cap total attempts so a hard daily-cap failure surfaces a clear message instead of looping.
Distinguish 429-rate-limit (retry w/ backoff) from validation failure (instructor re-ask) from daily-cap exhaustion (fail fast, tell the user). Surface OpenRouter's error body in the message.
Make slug configurable so switching :free → paid is a .env edit.
5. CLI / entry point (cli.py)
argparse-based, the minimal way to run end-to-end on one PDF:

python -m resume_parser.cli RESUME.pdf \
    --field-spec config/field_spec.json \
    --output resume_extracted.json \
    --artifacts-dir artifacts/ \
    [--model nvidia/nemotron-3-ultra-550b-a55b:free] \
    [--env .env]
Loads .env + field_spec.json, calls run_pipeline(), prints the resulting JSON. Defaults for model/base_url come from settings.py so the bare python -m resume_parser.cli RESUME.pdf works once .env is populated.
run_pipeline() keeps the spec's signature/flow (triage → clean → optional artifacts → extract_structured → write JSON), with model_name/base_url defaulted from settings rather than the vLLM literals.
6. Testing strategy
Two independent axes — extraction quality and LLM conformance:

Deterministic units (no network, no models): pytest over cleanup.py (each dedup function + ordering), schema_builder.py (nested/array/optional → correct Pydantic types, required-vs-optional), and triage.py (synthetic native-text vs. image-heavy PDFs hit the right branch). These are the fast-feedback core and need no API budget.

Extraction quality — Marker vs PyMuPDF4LLM via artifacts_dir: assemble a handful of tests/samples/ PDFs (mix of native-text and scanned/multi-column). Loop run_pipeline() with a per-file artifacts/{pdf_stem}/, then inspect 01_raw_{engine}.md (column interleaving, garbled OCR, missing sections), diff against 02_cleaned.md, and aggregate 03_extraction_metadata.json (dedup_reduction_pct, page counts) across the batch to flag engines producing fragmented/duplicated output. Run the LLM step disabled here (add a --no-llm flag) so extraction can be validated without spending OpenRouter quota.

LLM conformance — Nemotron via OpenRouter: after probe_instructor_mode.py picks a mode, run the full pipeline on the cleaned text of several samples and assert the output validates against the dynamic model and that required fields are populated. Budget note: :free daily caps mean batch LLM testing must be small/spaced; do extraction-quality iteration with --no-llm and reserve live calls for conformance checks. Consider a tiny recorded fixture (saved cleaned-text → expected JSON) for a non-network regression check.

Pipeline is "reliable" only when: units green, every sample triages correctly and produces clean artifacts, and the chosen instructor mode returns schema-conformant JSON across the sample set without manual repair.

7. Open questions / assumptions to confirm
Model slug & access: assuming nvidia/nemotron-3-ultra-550b-a55b:free is live and your key has access. The probe script will confirm; if the :free variant is unavailable or too rate-limited to test, fall back to the paid slug (a .env change). Confirm you're OK spending paid credits if needed.
field_spec source of truth: assuming the spec's example field_spec becomes config/field_spec.json as-is. Confirm that's the real target schema (or supply the actual one).
CPU vs GPU for Marker: assuming CPU is acceptable (slower) for now. Confirm if a GPU is available/expected — affects sample-test runtime only.
Mode ladder default: I'll start the probe at TOOLS then JSON_SCHEMA; if you already know which OpenRouter provider/mode works for this slug, tell me and I'll skip straight to it.
Strict-schema behavior is unknown until probed: the chosen mode may pass a simple schema but reject/coerce Optional fields or nested object arrays under a strict provider (see §4). The probe will tell us per-field; if it bites we adopt strict-compatible schema emission (all-required + explicit nullable + additionalProperties:false). Assumption to confirm: acceptable to make the schema builder emit strict-compatible JSON schema if the probe shows the provider needs it.
If no mode is conformant even after strict mitigations: the pipeline fails loud — extract_structured() raises with the last per-field errors and the raw provider response, and we reassess the model/provider choice (pin a specific OpenRouter provider, or switch slug). There is no silent manual-repair fallback; a half-valid JSON is never written. Confirm that's the behavior you want (vs. e.g. writing a best-effort partial with a warning).
Output/artifacts retention: assuming local-disk artifacts are fine and git-ignored; no DB/object-store persistence in scope yet.
Folder name: repo dir is ocr-resume-paser (note: "paser", missing the "r" in parser) — that's the real on-disk path, so the plan uses it as-is. Rename to ocr-resume-parser is trivial but optional; say the word.
Verification (once implemented)
pip install -r requirements.txt in a fresh venv succeeds; pip freeze pins resolved torch/transformers.
python -m pytest — deterministic units green.
python scripts/probe_instructor_mode.py — builds the response_model from the real config/field_spec.json (nested work_history/education), prints a working instructor mode and per-field validation results against the live slug; flags any strict-schema/optional-field coercion. Record the mode in .env.
python -m resume_parser.cli tests/samples/<native>.pdf --artifacts-dir artifacts/ --no-llm → triages to pymupdf4llm, writes clean artifacts.
python -m resume_parser.cli tests/samples/<scanned>.pdf --artifacts-dir artifacts/ --no-llm → triages to marker, loads weights once, writes clean artifacts.
Full run (LLM enabled) on one sample → resume_extracted.json validates against the dynamic model with required fields populated.