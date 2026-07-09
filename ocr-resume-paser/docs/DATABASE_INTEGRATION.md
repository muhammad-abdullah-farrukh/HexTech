# Part 1 — Parser Database Integration Plan

**Scope:** persist the resume parser's (`ocr-resume-paser`) structured JSON output (`04_structured.json`) into PostgreSQL. This covers ingestion, schema, upsert safety, and forward-compatibility with future batch/parallel processing. It does **not** cover the downstream knowledge-graph pipeline (Ontogen) or Neo4j — that is a separate, later phase that reads from this database but is out of scope here.

---

## 1. Goals

1. Replace `artifacts/{name}/04_structured.json` files with durable, queryable Postgres rows.
2. Deduplicate by PDF content (re-submitting the same PDF updates the existing record, never duplicates it).
3. Keep the parser pipeline itself database-agnostic — it returns a Python dict; the caller decides whether/how to persist it.
4. Design the ingestion code so that adding batch/parallel processing later requires no rework of the schema or ingest logic — only a small addition at the process-pool level.
5. Track schema drift explicitly (`field_spec.json` will change over time; stored records must record which version produced them).

---

## 2. Infrastructure

### 2.1 Postgres via Docker Compose

```yaml
# docker-compose.yml (repo root)
services:
  postgres:
    image: postgres:17
    environment:
      POSTGRES_DB: resume_parser
      POSTGRES_USER: resume_parser
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-devpassword}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

### 2.2 Environment

Add to `.env.example` and `.env`:

```
DATABASE_URL=postgresql+psycopg2://resume_parser:devpassword@localhost:5432/resume_parser
```

### 2.3 Dependencies

Add to `requirements.txt`:

```
sqlalchemy>=2.0
psycopg2-binary
alembic
```

---

## 3. Schema

One parent table (`resumes`) holding the raw structured JSON verbatim, plus three child tables that form a **rebuildable projection** for indexed queries (skill search, company search, etc.). The projection is *derived* from `structured` — it is a cache, not a second source of truth. On re-ingest, projection rows are deleted and re-derived rather than diffed.

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto; -- for gen_random_uuid()

CREATE TABLE resumes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pdf_hash        TEXT UNIQUE NOT NULL,       -- SHA-256 of raw PDF bytes; upsert key
    source_file     TEXT,
    structured      JSONB NOT NULL,             -- 04_structured.json, verbatim
    field_spec_hash TEXT NOT NULL,               -- SHA-256 of field_spec.json at ingest time
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ix_resumes_email ON resumes ((structured->>'email'));

CREATE TABLE skills (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resume_id  UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    skill      TEXT NOT NULL
);
CREATE INDEX ix_skills_skill ON skills (skill);
CREATE INDEX ix_skills_resume_id ON skills (resume_id);

CREATE TABLE work_history (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resume_id   UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    company     TEXT,
    title       TEXT,
    start_date  TEXT,
    end_date    TEXT
);
CREATE INDEX ix_work_history_company ON work_history (company);
CREATE INDEX ix_work_history_resume_id ON work_history (resume_id);

CREATE TABLE education (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resume_id        UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    institution      TEXT,
    degree           TEXT,
    graduation_year  TEXT
);
CREATE INDEX ix_education_institution ON education (institution);
CREATE INDEX ix_education_resume_id ON education (resume_id);
```

`ON DELETE CASCADE` is deliberate: it's what makes "wipe and re-derive the projection on re-ingest" a one-line operation (`resume.skills.clear()`, etc.) instead of a diff.

Manage this via **Alembic** migrations from the first commit — `field_spec.json` is runtime-editable, so this schema *will* evolve, and migration discipline should exist before the first change, not after.

---

## 4. Python Layer

### 4.1 Models — `resume_parser/db/models.py`

SQLAlchemy 2.x declarative models mapping 1:1 to the schema in §3: `Resume`, `Skill`, `WorkHistory`, `Education`, with `cascade="all, delete-orphan"` relationships from `Resume` to each child collection.

### 4.2 Session factory — `resume_parser/db/session.py`

A `make_session_factory(database_url: str)` function that builds an `Engine` (`pool_pre_ping=True`) and returns a `sessionmaker` bound to it. **Critical constraint for future parallelization:** this factory must be constructed fresh inside each process/worker — never created once and shared across forked processes (see §6).

### 4.3 Ingest function — `resume_parser/db/ingest.py`

`make_ingest_fn(session_factory, field_spec_path)` returns a closure matching the signature the pipeline already expects:

```python
def ingest_resume(structured: dict, source_pdf_path: str) -> str: ...
```

Responsibilities:
- Compute `pdf_hash` (SHA-256 of the PDF file's raw bytes) and `field_spec_hash` (SHA-256 of `field_spec.json`'s contents).
- **Atomic upsert** on `resumes` keyed by `pdf_hash`, using Postgres's native `INSERT ... ON CONFLICT (pdf_hash) DO UPDATE` (via SQLAlchemy's `postgresql.insert(...).on_conflict_do_update(...)`) — not a manual "check-then-insert-or-update," which has a race condition under concurrent writers.
- Clear and re-populate `skills`, `work_history`, `education` from the `structured` dict's `skills[]`, `work_history[]`, `education[]` arrays.
- Open and close its own session per call — no session is held open across calls or shared across threads/processes.
- Return the resume's `id` (UUID as string).

---

## 5. Pipeline Integration

Minimal, additive change to `pipeline.py` — no new imports of any DB library in the pipeline module itself:

```python
def run_pipeline(..., ingest_fn=None) -> dict | None:
    ...
    structured = extract_structured(...)
    if artifacts_dir:
        save_structured(artifacts_dir, structured)
    if ingest_fn is not None:
        ingest_fn(structured, pdf_path)
    return structured
```

`cli.py` gains a `--db-uri` flag. When present, it constructs the session factory and ingest closure and passes them through; when absent, behavior is identical to today (no DB import happens at all). This keeps `run_pipeline`'s existing callers and tests unaffected.

---

## 6. Forward-Compatibility With Batch/Parallel Processing (do not build yet)

This section documents *why* the current design is already parallel-safe, so that a future change is additive rather than a rewrite. **None of the following should be implemented now** — only the two items in §6.3 are worth doing today.

### 6.1 What already makes this safe to parallelize later

- The ingest function takes a **session factory**, not a live connection or session — each call is self-contained.
- The upsert is **atomic** at the database level (`ON CONFLICT`), so two workers racing on the same `pdf_hash` cannot corrupt state or throw a `UniqueViolation`.

### 6.2 The one real gotcha, when parallelization is added

SQLAlchemy engines are **not fork-safe**. If a `session_factory` is created in the parent process before `multiprocessing.Pool`/`ProcessPoolExecutor` forks workers (default on Linux), child processes inherit the same raw TCP socket to Postgres, which corrupts concurrent queries. The fix at that time: use the pool's `initializer` to construct a fresh engine/session factory **inside each worker process**, after the fork — not before it. This is a ~10-line addition, not a redesign.

### 6.3 Worth doing now (cheap, prevents future pain)

1. Use the atomic `ON CONFLICT` upsert from day one (§4.3) — this is the piece that would be a real bug fix later if skipped now; doing it now is nearly free.
2. Cap `pool_size` on the engine (e.g. `pool_size=5`) so that `future_num_workers × pool_size` stays comfortably under Postgres's `max_connections` (default 100) once parallelism is added.

### 6.4 Note on why parallelizing the Python layer alone won't help yet

The LLM server currently runs `--parallel 1` (one inference slot). Parallel Python workers calling the LLM will queue behind that single slot with no speedup until `--parallel N` is set on the server (which costs VRAM per additional slot). Database-layer parallelism readiness (this section) is independent of that and can be built first without effect until the LLM server side is also changed.

---

## 7. Testing

`tests/test_db_ingest.py`, run against a real Postgres test database (a second Docker Compose service or database, e.g. `resume_parser_test`) rather than mocks or SQLite — the behavior under test (JSONB storage, the `pdf_hash` unique constraint, `ON CONFLICT` upsert semantics) is Postgres-specific and won't be faithfully exercised by a substitute.

Cases to cover:
- Fresh ingest creates one `resumes` row plus correct child rows.
- Re-ingesting the same `pdf_hash` updates the existing row (no duplicate), and projection tables reflect the new data, not a union of old and new.
- `field_spec_hash` is recorded correctly and changes when `field_spec.json` changes.
- Missing optional fields (`work_history`, `projects`, `education` are non-required per the field spec) do not raise.

---

## 8. Out of Scope for Part 1

- Knowledge graph entity/relationship extraction (Ontogen pipeline) and its own staging tables (`pipeline_runs`, `canon_store`, `gazetteers`, `graph_entities`, `graph_relationships`).
- Neo4j loading/sync.
- Encryption (column-level `pgcrypto`, TDE, at-rest via managed hosting) — deferred until closer to actual production deployment.
- Actually implementing batch/parallel processing — §6 documents *readiness* only.
