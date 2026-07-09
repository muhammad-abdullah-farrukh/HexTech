import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR          = ROOT / "data"
DOCS_DIR          = DATA_DIR / "documents"
WIKIDATA_DIR      = DATA_DIR / "wikidata"
EMBEDDINGS_DIR    = ROOT / "embeddings"
OUTPUTS_DIR       = ROOT / "outputs"

WIKIDATA_RAW        = WIKIDATA_DIR / "properties_raw.json"
WIKIDATA_FILTERED   = WIKIDATA_DIR / "properties_filtered.json"
WIKIDATA_EMBEDDINGS = EMBEDDINGS_DIR / "wikidata_embeddings.npy"

# Gazetteer JSON files (companies, universities, certifications, skills, job_titles)
GAZETTEER_DIR = DATA_DIR / "gazetteers"
# EDC canonical relation store (persists across documents)
CANON_STORE_DIR = DATA_DIR / "canon_store"
# ── LLM ────────────────────────────────────────────────────────────────────
# stages/llm.py talks to {base_url}/chat/completions on one of the endpoints
# below. Independent calls within a stage are fanned out across all of them
# concurrently (call_llm_parallel); single calls use the primary (index 0).
LLM_PROVIDER = "openai-compatible"

# Endpoints for parallel dispatch. Each carries the model id it ACTUALLY
# serves (verify with `curl {base_url}/models`): vLLM validates the request's
# "model" field and 404s a wrong id, and the served-vs-requested mismatch
# check in call_llm keys off it.
#
# `slots` = how many requests to fire CONCURRENTLY at that endpoint — a purely
# client-side knob: call_llm_parallel spawns this many worker threads bound to
# the endpoint (local llama-server was started with -np 3; the cluster vLLM does
# its own continuous batching, so this is just how wide we choose to fan out).
#
# `ctx` = the min-context ROUTING GATE, not a doc string. call_llm_parallel's
# min_ctx filter only dispatches a call to endpoints whose "ctx" >= the prompt's
# token estimate (see stage2_cq_answer). So it MUST be the PER-REQUEST window a
# single call actually gets. Values verified 2026-07-09:
#   - cluster: max_model_len 32768 (the endpoint's manager raised it from the
#     old 8192). This is now the LARGER window, so it's the safe home for any
#     long-prompt call — the min_ctx gate routes big Stage-2 prompts here.
#   - local: 8192 per request. The server is started with `-c 24576 -np 3`, and
#     llama.cpp splits that total KV cache across the 3 parallel slots, so each
#     concurrent request gets 24576/3 = 8192 (confirmed: /props reports
#     total_slots 3, n_ctx 8192). Do NOT put 24576 here — that's the aggregate
#     across slots, not what one prompt can use; a >8192 prompt truncates.
#
# NOTE: these are DIFFERENT models — local DeepSeek-R1-32B (a slow reasoning
# model) vs. cluster Qwen3.5-9B. An independent call may land on either, so a
# stage's per-item decisions come from a mix of both. This is accepted
# deliberately (quality of both judged acceptable) to gain the throughput of
# running them at once. Do NOT route mutually-dependent calls that must agree
# with each other (e.g. both halves of a single A-vs-B comparison) this way.
#
# The cluster endpoint has itself been observed answering as two different
# backends (see stages/llm.py + CLAUDE.md); the per-call llm_calls.jsonl log
# records the model each call was actually served by, so a run's real split is
# always recoverable after the fact.
LLM_ENDPOINTS = [
    {
        "base_url": "http://127.0.0.1:9000/v1",
        "model": "/home/faryal/models/deepseek-r1/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "slots": 3,      # local llama-server: -np 3
        "ctx": 8192,     # -c 24576 / -np 3 slots = 8192 per request (see above)
        "think_suppress": "prefill",  # DeepSeek-R1 has no enable_thinking flag
    },
    {
        "base_url": "http://192.168.3.84:8001/v1",
        "model": "/app/local_model/Qwen3.5-9B",
        "slots": 4,      # cluster vLLM: continuous batching
        "ctx": 32768,    # max_model_len (manager raised from 8192, 2026-07-09)
        "think_suppress": "flag",     # Qwen honours chat_template_kwargs
    },
]

# Set HEXTECH_LLM_SINGLE_ENDPOINT=1 to force everything onto the primary
# endpoint only (disables cross-endpoint fan-out) — e.g. when the cluster is
# unreachable or you want a single-model run for a clean measurement.
if os.environ.get("HEXTECH_LLM_SINGLE_ENDPOINT", "").strip() in ("1", "true", "True"):
    LLM_ENDPOINTS = LLM_ENDPOINTS[:1]

# Primary endpoint (index 0): the default for a bare call_llm() with no
# endpoint override, and the single server the llm_lock + provenance labels
# assume. Kept in sync with LLM_ENDPOINTS[0].
LLM_BASE_URL = LLM_ENDPOINTS[0]["base_url"]
LLM_MODEL    = LLM_ENDPOINTS[0]["model"]

LLM_TEMPERATURE = 0.0

LLM_API_KEY = "not-needed"  # the server ignores this; the client sends it anyway

# ── Database (shared with ocr_resume_parser) ────────────────────────────────
# Same Postgres instance/URL the parser uses (see ocr-resume-paser/.env).
DATABASE_URL = os.environ.get("DATABASE_URL")
# ── Embedding model (Stage 5) ──────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en"   # verbatim from paper
EMBED_DIM   = 384                   # bge-small-en dimension; matches vector(384) columns
# Device for SentenceTransformer(EMBED_MODEL). SentenceTransformer defaults to
# CUDA when available, but the local llama-server (-ngl 99, -np 3, -c 24576,
# q8 KV cache) claims ~22.2 GB of the 24.5 GB card, leaving <1.1 GB free
# (measured 2026-07-09). bge-small's weights + CUDA context + a full-gazetteer
# batch encode (Tier 2) don't fit in that, and loading on CUDA OOMs on the
# first Stage 6 embed (the Session-4 failure). Embeddings are also not on the
# speed-critical path — bge-small encodes in ms on CPU; the wall-clock cost is
# the LLM calls. CPU is the safe, no-speed-penalty choice while a GPU-resident
# LLM server is running. Set to "cuda" only if you shrink that server first.
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu").strip()

# ── EDC verify gate (Step 4) ────────────────────────────────────────────────
# Deterministic pre-filter that replaces most per-candidate reasoning-LLM
# calls. Two tiers, in order (see stages/verify_gate.py):
#
#   1. Token containment + BENIGN_SUFFIXES: catches the "modifier changes
#      meaning" failure mode that dense similarity cannot (e.g. 'employer' vs
#      'current employer' — high cosine, NOT equivalent) — words like
#      'number'/'address' just describe the value's format and don't change
#      what's being asked; anything else extra is treated as meaning-changing.
#      An earlier attempt to solve this with a pretrained NLI cross-encoder
#      failed calibration entirely (merge/reject scores both clustered near
#      zero — entailment is the wrong relationship for "same meaning", and the
#      label:definition input format is out-of-distribution for NLI models).
#   2. bge cosine bands: only reached when containment doesn't apply. Pairs
#      scoring in the ambiguous middle ESCALATE to the LLM (_llm_verify).
#
# VERIFY_TAU_HI/LO below are bootstrap fallbacks only, validated against 70
# backfilled deepseek verdicts (0 false merges, 0 false rejects, 47% coverage
# at these values) — NOT hand-tuned constants to trust indefinitely. The live
# values come from the verify_thresholds table (latest row), recomputed by
# scripts/recompute_thresholds.py only once enough new verdicts accumulate
# (MIN_SAMPLES_FOR_RECOMPUTE) AND the same precision bar is met — see that
# script. These constants are used only if that table is empty (fresh install).
VERIFY_TAU_HI = 0.94                 # cosine ≥ this → MERGE without the LLM
VERIFY_TAU_LO = 0.85                 # cosine ≤ this → REJECT without the LLM
MIN_SAMPLES_FOR_RECOMPUTE = 500       # verify_verdicts rows required before
                                       # recompute_thresholds.py will move
                                       # tau_hi/tau_lo off the current values —
                                       # "hundreds to thousands", not a handful
BENIGN_SUFFIXES = {                   # extra words that don't change meaning
    "address", "number", "name", "id", "code", "type", "value",
}
# Ambiguous-band fallback: "llm" = escalate to _llm_verify (default);
# "reject" = conservative offline mode (never merge on uncertainty) so the
# pipeline stays runnable with the LLM server down.
VERIFY_ESCALATION = "llm"

# ── Stage 1: Competency Question generation ───────────────────────────────
# We don't pre-guess how many CQs a document "should" have — no formula
# (word count, entity count, etc.) actually knows that before the model
# reads the document. The LLM decides count based on what's actually in
# the document. CQ_SAFETY_MAX is purely a ceiling to stop a malformed/huge
# doc from generating an unbounded number of CQs (and unbounded downstream
# LLM calls in stage 2/3) — it is not fed to the model as a target.
CQ_SAFETY_MAX = 40

# ── Pipeline modes ─────────────────────────────────────────────────────────
# True  → no-schema-constraint mode  (new props added if no Wikidata match)
# False → target-schema-constrained  (discard unmatched props)
SCHEMA_EXPANSION = True

# ── Wikidata allowed datatypes (Stage 4) ───────────────────────────────────
ALLOWED_DATATYPES = {
    "wikibase-item",
    "quantity",
    "string",
    "monolingualtext",
    "time",
}

# ── Stage 5/6: top-k Wikidata candidate retrieval ──────────────────────────
# Validate top-k nearest Wikidata neighbours in rank order; return first match.
# Set to 3 (default) or 5 (higher recall, more LLM calls) experimentally.
TOP_K_CANDIDATES = 3

# ── Stage 9: context-window guard ──────────────────────────────────────────
# Rough ceiling on the Stage 9 prompt in characters (~4 chars per token).
# At ~6 k tokens input the 8B model still has headroom for 2 k output tokens
# on a typical 8 k context window.
# DeepSeek-R1-70B served through vLLM.
# Maximum context length reported by server: 6144 tokens.
MAX_PROMPT_CHARS = 15000

# ── EDC relation canonicalization ──────────────────────────────────────────
# Top-k canon store candidates to validate before declaring a relation novel.
CANON_TOP_K = 3

# ── Entity resolution ───────────────────────────────────────────────────────
# Enable entity resolution post-processing on generated KG Turtle (Stage 9).
ENTITY_RESOLUTION_ENABLED = True

# Map Wikidata property labels (PascalCase) → entity type for entity resolver.
# Used to infer the entity type of wd: URIs from the predicate that uses them.
PROPERTY_ENTITY_TYPE_MAP: dict[str, str] = {
    # Employment
    "WorkedAt":        "company",
    "Employer":        "company",
    "EmployedBy":      "company",
    "WorksFor":        "company",
    "WorkPlace":       "company",
    # Education
    "EducatedAt":      "university",
    "AlumniOf":        "university",
    "Education":       "university",
    "DegreeFrom":      "university",
    "StudiedAt":       "university",
    "GraduatedFrom":   "university",
    # Skills
    "HasSkill":        "skill",
    "Skill":           "skill",
    "KnowledgeOf":     "skill",
    "TechnicalSkill":  "skill",
    "ProgrammingLanguage": "skill",
    "Technology":      "skill",
    "Tool":            "skill",
    "Framework":       "skill",
    # Certifications
    "Certification":       "certification",
    "HasCertification":    "certification",
    "Certified":           "certification",
    "License":             "certification",
    # Job titles
    "JobTitle":        "job_title",
    "Occupation":      "job_title",
    "Position":        "job_title",
    "Role":            "job_title",
    "Title":           "job_title",
    # Projects — résumé-unique; the resolver hard-skips this type, this
    # mapping just makes the inferred type truthful instead of "unknown".
    "HasProject":      "project",
    "Project":         "project",
    "BuiltProject":    "project",
}