# Part 2 — Ontogen Database Integration Plan

**Scope:** move the Ontogen knowledge-graph pipeline (`ontogen/`) off the filesystem and onto PostgreSQL, so it (a) reads résumés directly from the `resumes` table built in Part 1, (b) stops treating per-document JSON/Turtle files as pipeline input, (c) stages its entities/relationships in Postgres before loading into Neo4j, and (d) uses two extraction paths — deterministic (from already-structured fields) and CQ-driven (from free-text descriptions only) — instead of running every fact through CQs.

This plan assumes Part 1 is complete and verified (`resumes`, `skills`, `work_history`, `education`, `projects` tables, all confirmed working via `DB_REPORT.md`).

---

## 1. What changes and why (recap of decisions made)

1. **Ontogen's input becomes `resumes.structured`, not `data/documents/*.txt`.** A `render_resume_text()` function replaces reading a file — but it renders *only* free-text fields (descriptions, summary), not the whole resume.
2. **Two relation-extraction paths, merged before Stage 6:**
   - **Path A (deterministic, no LLM):** `structured_to_relations()` turns `work_history[]`, `education[]`, `skills[]` directly into relation dicts — no CQ generation, no CQ answering, no relation-extraction LLM call, since these facts are already clean structured data from Part 1.
   - **Path B (CQ-driven, LLM):** Stages 1–3 run only against free-text fields (`work_history[].description`, `projects[].description`, `summary`) — the content that genuinely isn't already broken into named fields.
   - Both paths produce the same `list[dict]` relation shape and are concatenated immediately before Stage 6 (Wikidata match/validate) — this is the single merge point; nothing downstream needs to know which path a relation came from (though each relation is tagged `"source": "structured"` / `"source": "cq_extracted"` for later debugging/provenance).
3. **File-based caching (`outputs/{cqs,answers,relations,matches,ontology}/`) becomes the `pipeline_runs` table**, keyed by `(document_id, stage)` instead of filename.
4. **Corpus-wide stores move to Postgres:** canon store (`data/canon_store/entries.json`), gazetteers (5 files in `data/gazetteers/`), and Wikidata properties + embeddings (`data/wikidata/properties_filtered.json` + `embeddings/wikidata_embeddings.npy`) all become tables — the last one specifically to fix the documented "brute-force scan won't scale" issue via `pgvector`.
5. **Final KG output is staged in Postgres before Neo4j**, not written straight to `.ttl` files: `stage9_10_kg.py`'s output is parsed into `graph_entities` / `graph_relationships` rows, enabling cross-document entity dedup *before* the Neo4j load — which today doesn't happen at all (each document's Turtle is loaded independently).
6. **Provenance moves from `outputs/provenance/*.json` to a `provenance` table**, same record shape (`doc_id`, `stage`, `timestamp`, `confidence`, `model`, `extra`).
7. **JSON/Turtle files become read-only debug exports**, generated on demand from the database for manual inspection — never read back in by the pipeline.

---

## 2. Schema additions (new Alembic migration, `0003_ontogen_schema.py`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector

-- Per-document, per-stage run tracking — replaces outputs/*/*.{json,ttl} existence checks
CREATE TABLE pipeline_runs (
    document_id  UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL,   -- 'cq_gen' | 'cq_answer' | 'relation_extract' |
                                   -- 'match_validate' | 'edc_canon' | 'ontology' | 'kg_facts'
    status       TEXT NOT NULL DEFAULT 'succeeded',  -- 'pending' | 'succeeded' | 'failed'
    output       JSONB,           -- stage's output payload (list/dict), or {"ttl": "..."} for text output
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, stage)
);

-- EDC cross-document canon store (replaces data/canon_store/entries.json)
CREATE TABLE canon_store (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label       TEXT NOT NULL,
    definition  TEXT NOT NULL,
    turtle      TEXT,
    source_doc  UUID REFERENCES resumes(id),
    embedding   vector(384),      -- bge-small-en dimension, per config.py EMBED_MODEL
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_canon_store_embedding ON canon_store USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ix_canon_store_label ON canon_store (lower(label));

-- Gazetteers — replaces data/gazetteers/*.json (5 files → 1 table)
CREATE TABLE gazetteers (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type  TEXT NOT NULL,   -- 'company' | 'university' | 'certification' | 'skill' | 'job_title'
    alias        TEXT NOT NULL,
    canonical    TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'static',  -- 'static' (seeded) | 'tier3_llm' (learned)
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_gazetteers_alias ON gazetteers (entity_type, lower(alias));

-- Wikidata properties + embeddings — replaces properties_filtered.json + wikidata_embeddings.npy
CREATE TABLE wikidata_properties (
    pid          TEXT PRIMARY KEY,
    label        TEXT NOT NULL,
    description  TEXT,
    embedding    vector(384)
);
CREATE INDEX ix_wikidata_properties_embedding ON wikidata_properties USING hnsw (embedding vector_cosine_ops);

-- Provenance — replaces outputs/provenance/*.json
CREATE TABLE provenance (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    subject     TEXT NOT NULL,
    predicate   TEXT NOT NULL,
    object      TEXT NOT NULL,
    stage       TEXT NOT NULL,
    confidence  FLOAT NOT NULL,
    model       TEXT NOT NULL,
    extra       JSONB,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_provenance_document_id ON provenance (document_id);

-- KG staging — nodes and edges before Neo4j load
CREATE TABLE graph_entities (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type      TEXT NOT NULL,      -- Person, Organization, Skill, ... (from PROPERTY_ENTITY_TYPE_MAP)
    canonical_group  UUID,               -- set by cross-document dedup; NULL until resolved
    properties       JSONB NOT NULL,     -- includes "uri" (wd:slug), plus literal props
    source_doc       UUID REFERENCES resumes(id),
    synced_to_neo4j  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_graph_entities_uri ON graph_entities (((properties->>'uri')));
CREATE INDEX ix_graph_entities_unsynced ON graph_entities (synced_to_neo4j) WHERE synced_to_neo4j = FALSE;

CREATE TABLE graph_relationships (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_entity      UUID NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    to_entity        UUID NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    rel_type         TEXT NOT NULL,
    properties       JSONB,
    source_doc       UUID REFERENCES resumes(id),
    synced_to_neo4j  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_graph_relationships_unsynced ON graph_relationships (synced_to_neo4j) WHERE synced_to_neo4j = FALSE;
```

Note: `resumes` lives in the parser's database/schema from Part 1 — Ontogen's models must reference the *same* database (one Postgres instance, per `WHY_POSTGRES.md`), not a separate one, so these foreign keys resolve.

---

## 3. Python DB layer for Ontogen — `ontogen/db/`

Mirror the structure already established in `resume_parser/db/`:

- `ontogen/db/models.py` — SQLAlchemy models for all tables in §2. Reuses/imports `resume_parser.db.models.Resume` for the FK relationship rather than redefining it.
- `ontogen/db/session.py` — same `make_session_factory(database_url)` pattern as Part 1, same fork-safety documentation for future parallelism.
- `ontogen/db/runs.py` — helpers for `pipeline_runs`: `get_stage_output(session, document_id, stage) -> dict | None` and `save_stage_output(session, document_id, stage, output, status="succeeded")`, replacing the `OUTPUTS_DIR / stage / f"{doc_name}.json"` existence-check pattern throughout `pipeline.py`.
- `ontogen/db/canon.py` — `pgvector`-backed replacement for the canon store's in-memory `_entries` list in `canonicalize.py`'s `EDCBackend`: `search_similar(embedding, top_k) -> list[dict]`, `add_entry(label, definition, turtle, embedding, source_doc)`.
- `ontogen/db/gazetteers.py` — `lookup(entity_type, alias) -> str | None`, `add_alias(entity_type, alias, canonical, source="tier3_llm")` — replaces loading 5 JSON files into memory at startup.
- `ontogen/db/wikidata.py` — `top_k_candidates(embedding, k) -> list[dict]` via `pgvector`, replacing the `.npy` nearest-neighbor scan in Stage 6.
- `ontogen/db/kg_staging.py` — `stage_entity(...)`, `stage_relationship(...)` — parses `stage9_10_kg.py`'s output triples into `graph_entities`/`graph_relationships` rows (literal-object triples → node properties, uri-object triples → edges), replacing the `.ttl` file write.

---

## 4. Pipeline rewrite — `ontogen/pipeline.py`

### 4.1 Resume rendering (new — `ontogen/render.py`)

```python
def render_resume_text(structured: dict) -> str:
    """Free-text fields only — summary and description fields — for CQ-driven
    extraction (Path B). Excludes anything already structured (skills, dates,
    company/institution names), since those are handled deterministically."""
    ...

def structured_to_relations(structured: dict) -> list[dict]:
    """Deterministic relation extraction (Path A) — no LLM call. Maps
    work_history[], education[], skills[] directly to relation dicts,
    tagged source='structured'."""
    ...
```

### 4.2 `process_doc(doc_path)` → `process_resume(resume_id, session_factory)`

- Fetch `Resume` row by `id`; `structured = resume.structured`.
- `narrative_text = render_resume_text(structured)`.
- Stage 1/2/3 run against `narrative_text` only (not the full resume) — CQs scoped to what free text actually contains, since Path A already covers everything else.
- `relations = stage3_relation_extract.run(...) + structured_to_relations(structured)`.
- Each stage checkpoint (`cq_out.exists()` etc.) becomes `get_stage_output(session, resume.id, "cq_gen")`.
- Stage 6 (`stage6_match_validate.run`) queries `wikidata_properties` via `ontogen/db/wikidata.py` instead of the `.npy` array.
- Stage 3.5 EDC canonicalization queries `canon_store` via `ontogen/db/canon.py` instead of `EDCBackend._entries`.
- Stage 7/8 ontology output saved via `save_stage_output(..., "ontology", {"ttl": ontology_text})`.
- Stage 9/10 output is parsed and staged into `graph_entities`/`graph_relationships` via `ontogen/db/kg_staging.py`, **not** written to `outputs/kg/{doc}.ttl`.
- Provenance records (`stages/provenance.py`) write to the `provenance` table instead of `ProvenanceStore.save(Path(...))`.

### 4.3 `main()`

- Replace `DOCS_DIR.glob("*.txt")` with a query: `SELECT id FROM resumes` (optionally `WHERE id NOT IN (SELECT document_id FROM pipeline_runs WHERE stage='kg_facts' AND status='succeeded')` for resume-mode).
- `--resume` flag now means "skip resumes whose `pipeline_runs` already has a succeeded `kg_facts` row" instead of checking file existence.

---

## 5. Stage-by-stage change summary

| File | Change |
|---|---|
| `stages/llm.py` | None. |
| `stages/stage1_cq_gen.py`, `stage2_cq_answer.py` | Input text scoped to free-text fields only (via `render_resume_text`); internals unchanged. |
| `stages/stage3_relation_extract.py` | Output merged with `structured_to_relations()` output at the call site in `pipeline.py`; internals unchanged. |
| `stages/stage4_filter_wikidata.py` | One-time script; final write step (`WIKIDATA_FILTERED.write_text`) becomes a `wikidata_properties` insert. Fetch/filter logic unchanged. |
| `stages/stage5_embed_wikidata.py` | One-time script; final write (`np.save`) becomes an `UPDATE wikidata_properties SET embedding = ...`. Embedding logic unchanged. |
| `stages/stage6_match_validate.py` | Nearest-neighbor lookup source changes from `.npy` array to `ontogen/db/wikidata.py`'s `top_k_candidates()`. |
| `stages/canonicalize.py` | `EDCBackend`'s `_entries` in-memory list and its `GAZETTEER_DIR` JSON loads are replaced with calls into `ontogen/db/canon.py` and `ontogen/db/gazetteers.py`. Tier 1/2/3 resolution logic unchanged. |
| `stages/stage7_8_ontology.py` | Internals unchanged (builds Turtle from `match_results` in memory); only the final write target changes (`pipeline_runs` instead of `outputs/ontology/{doc}.ttl`). |
| `stages/stage9_10_kg.py` | Final Turtle is parsed and staged into `graph_entities`/`graph_relationships` instead of written to `outputs/kg/{doc}.ttl`. |
| `stages/provenance.py` | `ProvenanceStore.save()` writes to the `provenance` table instead of a JSON file. |
| `graphdb/load_to_neo4j.py` | `parse_ttl()` (file-based) replaced with a loader reading `graph_entities`/`graph_relationships` `WHERE synced_to_neo4j = FALSE`, batched `UNWIND ... MERGE`, flips `synced_to_neo4j = TRUE` on success. |
| `graphdb/config.py` | Unchanged (still just Neo4j connection settings) — but rotate the hardcoded default password out before this is ever pushed anywhere public. |
| `config.py` | Add `DATABASE_URL` resolution (same `os.environ.get` pattern as `resume_parser/settings.py`); paths like `CANON_STORE_DIR`, `GAZETTEER_DIR`, `WIKIDATA_EMBEDDINGS` become unused/removed once their tables exist. |

---

## 6. One-time data migration (seed the new tables from existing files)

Before switching the pipeline over, migrate what's already on disk so nothing is lost:

1. `data/gazetteers/*.json` → `gazetteers` rows (`source='static'`).
2. `data/canon_store/entries.json` → `canon_store` rows (re-embed if the stored embeddings aren't recoverable from the JSON as-is).
3. `data/wikidata/properties_filtered.json` + `embeddings/wikidata_embeddings.npy` → `wikidata_properties` rows, matched by array index per the existing file's documented index-alignment (`stage5`'s docstring: "index position in .npy corresponds to same index in properties_filtered.json").

A one-off script, `scripts/migrate_ontogen_files_to_db.py`, run once and then discarded (or kept as a historical record) — not part of the ongoing pipeline.

---

## 7. Neo4j load — batched, incremental

```cypher
CREATE CONSTRAINT entity_uri IF NOT EXISTS FOR (n:Entity) REQUIRE n.uri IS UNIQUE;
```
Loader reads unsynced `graph_entities`/`graph_relationships` in batches (≈500–1000 rows), `UNWIND ... MERGE` keyed on `properties->>'uri'`, then sets `synced_to_neo4j = TRUE` on success — per the incremental-sync design already agreed on, so re-runs only push deltas.

---

## 8. Testing

`tests/test_ontogen_db.py`, following the same real-Postgres pattern as `test_db_ingest.py` (reuse the `ensure_test_db` fixture from Part 1's `conftest.py`):
- `structured_to_relations()` produces expected relation dicts from a known `structured` fixture.
- `render_resume_text()` excludes structured fields, includes only description/summary text.
- `pipeline_runs` round-trip: save stage output, retrieve it, confirm resume-mode skip logic works.
- Gazetteer/canon-store/wikidata-properties lookup functions return expected results against seeded test rows.
- `graph_entities`/`graph_relationships` staging: a known Turtle fragment produces the correct node/edge rows.

---

## 9. Explicitly out of scope for this plan

- Batch/parallel processing across multiple résumés (same deferral as Part 1 §6).
- Encryption (deferred, per earlier discussion).
- Changing the LLM provider/model (`config.py`'s `ollama`/`qwen3:32b` stays as-is — unrelated to this integration).
- Rewriting `stage7_8_ontology.py` or `stage9_10_kg.py`'s actual extraction/ontology-building logic — only their I/O boundaries change.
