Postgres Persistence for the Resume Parser
Context
Today the pipeline's structured output lives only as artifacts/{name}/04_structured.json files — not queryable, not deduplicated, easy to lose. DATABASE_INTEGRATION.md (Part 1) specifies persisting that output into PostgreSQL: a parent resumes table holding the JSON verbatim, plus three derived projection tables (skills, work_history, education) for indexed search. The goal is durable, dedup-by-PDF storage while keeping the parser itself DB-agnostic (it returns a dict; the caller decides whether to persist), and keeping the design forward-compatible with future batch/parallel processing without reworking the schema or ingest logic.

This plan implements Part 1 of the MD except the parallel-processing work in §6.1/6.2/6.4 (explicitly future). §6.3 (atomic ON CONFLICT upsert + pool_size cap) is implemented now.

Verification against the real code (done)
run_pipeline — pipeline.py:21-30: keyword-only params after *; structured is produced at line 48 and saved at pipeline.py:50-51. The new ingest_fn hook slots in right there. pdf_path (line 22) is the source-PDF variable to pass to ingest.
cli.py uses argparse (cli.py:22-56) and calls run_pipeline at cli.py:76-84.
Settings are a frozen dataclass loaded with python-dotenv + os.environ.get (settings.py:49-87) — not pydantic-settings. DATABASE_URL resolution should follow the same os.environ.get pattern.
config/field_spec.json matches the MD schema: skills = array of strings; work_history items = company/title/start_date/end_date; education items = institution/degree/graduation_year. (years_experience is an object {years, months} but is not projected, so no conflict.)
Absent today: sqlalchemy / psycopg2 / alembic, docker-compose.yml, alembic/, and any DATABASE_URL in .env.example. python-dotenv and pytest are present.
Conventions to match: from __future__ import annotations at top of every module; relative intra-package imports; narrative Google-style docstrings (no Args/Returns sections); standard exceptions (ValueError/RuntimeError) with guidance-rich messages; tests are plain test_*() functions with inline data; the existing suite needs zero external services.
1. Infrastructure
docker-compose.yml (repo root) — verbatim from MD §2.1: postgres:17, db/user/pass resume_parser / resume_parser / ${POSTGRES_PASSWORD:-devpassword}, port 5432, named volume pgdata. No second service and no docker-entrypoint-initdb.d script (the test DB is created programmatically, per decision below).

requirements.txt — add sqlalchemy, psycopg2-binary, alembic. The file is fully version-pinned, so after pip install pin them to the installed versions to match convention (don't leave bare >= specifiers).

.env.example — add the two lines below. Also append the same two lines to the local .env for dev convenience (the devpassword is a non-secret dev default, and nothing sensitive is written):

DATABASE_URL=postgresql+psycopg2://resume_parser:devpassword@localhost:5432/resume_parser
# Same container/port, separate database used only by tests/test_db_ingest.py:
TEST_DATABASE_URL=postgresql+psycopg2://resume_parser:devpassword@localhost:5432/resume_parser_test
2. Alembic + initial migration
alembic.ini (root) + alembic/env.py, alembic/script.py.mako, alembic/versions/.
env.py: target_metadata = Base.metadata (from resume_parser.db.models); read the URL from os.environ["DATABASE_URL"] (load .env via python-dotenv) rather than hardcoding it in alembic.ini.
Hand-written initial migration (not autogenerate — autogenerate needs a live DB, which needs the container running / your approval). It creates the schema in MD §3 exactly: CREATE EXTENSION IF NOT EXISTS pgcrypto; resumes (UUID pk default gen_random_uuid(), pdf_hash UNIQUE NOT NULL, source_file, structured JSONB NOT NULL, field_spec_hash NOT NULL, ingested_at TIMESTAMPTZ default now()); child tables skills / work_history / education with resume_id FK ON DELETE CASCADE; and every index listed in §3 (incl. the expression index ix_resumes_email ON resumes ((structured->>'email'))).
I will show you this migration file and get approval before running alembic upgrade head, and will not run it (or docker compose up) against anything but local dev without asking.
3. Python DB layer — resume_parser/db/
New package (db/__init__.py empty). All modules use from __future__ import annotations, relative imports, Google-style docstrings.

models.py — SQLAlchemy 2.x declarative (DeclarativeBase, Mapped, mapped_column), 1:1 with §3. Resume has skills / work_history / education relationships with cascade="all, delete-orphan". UUID via postgresql.UUID(as_uuid=True) with server_default=text("gen_random_uuid()"); structured as postgresql.JSONB; ingested_at server_default=text("now()").

session.py — make_session_factory(database_url: str) -> sessionmaker. Builds the Engine with pool_pre_ping=True and pool_size=5 (MD §6.3 cap). Docstring documents the fork-safety constraint (MD §4.2/§6.2): the factory must be built fresh inside each worker process — never shared across a fork — so future parallelism is a small additive change.

ingest.py — make_ingest_fn(session_factory, field_spec_path) returns a closure ingest_resume(structured: dict, source_pdf_path: str) -> str:

pdf_hash = SHA-256 of the PDF file's raw bytes; field_spec_hash = SHA-256 of field_spec.json's bytes (read per call, so it reflects the spec at ingest time).
Atomic upsert via postgresql.insert(Resume).on_conflict_do_update(index_elements=["pdf_hash"], set_={...}) updating structured / source_file / field_spec_hash / ingested_at, with .returning(Resume.id) to get the id for both insert and update. ingested_at is in set_ on purpose: its semantics are "last processed", so re-ingesting after a field_spec.json change refreshes the timestamp — making it easy to spot stale records. (Not "first seen"; the MD schema has no separate column for that.)
Delete existing child rows for that resume_id, then re-insert from structured.get("skills", []) / .get("work_history", []) / .get("education", []) (guarding None), so projections are wiped-and-re-derived, never diffed. Missing optional arrays simply produce no child rows.
One transaction per call: upsert → delete children → insert children all commit once at the end (a single session.commit()), never three separate commits — a crash mid-ingest must not leave a resumes row pointing at stale/half-written child rows. Opens and closes its own Session per call; returns str(id).
4. Pipeline + CLI wiring (additive, non-breaking)
pipeline.py — add keyword-only ingest_fn: Callable[[dict, str], str] | None = None (import Callable under TYPE_CHECKING only; no DB import in this module). After the save_structured block:

if ingest_fn is not None:
    ingest_fn(structured, pdf_path)
Ingestion therefore only happens when structured exists (i.e. run_llm=True).

cli.py — add --db-uri (default None). DB ingestion happens only when --db-uri is explicitly passed — it is never inferred from DATABASE_URL (or any env var) being present. This preserves the non-breaking guarantee: a plain CLI run stays file-only and byte-for-byte identical to today even after DATABASE_URL lands in .env (which it must, for the DB tests). When args.db_uri is set, lazily import from .db.session import make_session_factory and from .db.ingest import make_ingest_fn, build the factory + closure (passing args.field_spec), print a one-line stderr confirmation (Ingesting to database at <uri>) so persistence is never silent, and pass ingest_fn= to run_pipeline. When absent, no DB import happens at all — existing callers/tests unaffected.

5. Tests — tests/test_db_ingest.py + conftest.py fixtures
Real Postgres (MD §7), no mocks/SQLite. Add fixtures to the existing root conftest.py (imports of psycopg2/sqlalchemy kept inside the fixtures so the rest of the suite stays service-free):

Session-scoped ensure_test_db: parse TEST_DATABASE_URL, connect to the postgres admin database via psycopg2 autocommit, and idempotently CREATE DATABASE resume_parser_test (guard with a pg_database existence query / catch psycopg2.errors.DuplicateDatabase). If Postgres is unreachable, warnings.warn(...) that the database isn't working, then pytest.skip. Because this behavior lives in a fixture consumed only by tests/test_db_ingest.py, the skip is scoped to just those DB tests — the other 7 test modules never touch the fixture and keep passing with no DB.
A fixture that builds a session_factory against TEST_DATABASE_URL, runs Base.metadata.create_all, and truncates the tables between tests for isolation.
Cases (MD §7): fresh ingest → one resumes row + correct child rows; re-ingest same pdf_hash → updates in place (no dup) and projections reflect new data (not a union); field_spec_hash recorded and changes when the spec bytes change; missing optional arrays (work_history/projects/education) don't raise.

Files
Create: docker-compose.yml, alembic.ini, alembic/env.py, alembic/script.py.mako, alembic/versions/<rev>_initial.py, resume_parser/db/{__init__,models,session,ingest}.py, tests/test_db_ingest.py. Edit: requirements.txt, .env.example, resume_parser/pipeline.py, resume_parser/cli.py, conftest.py.

Explicitly NOT doing (MD §6.1/6.2/6.4, §8)
No parallel/batch processing, no pool initializer, no Ontogen/Neo4j, no encryption. The session-factory-per-call + atomic upsert design keeps those additive later.

Verification
Show you the initial migration file; on approval (local dev only): docker compose up -d, then alembic upgrade head.
Run one real resume through the CLI with --db-uri "$DATABASE_URL"; confirm a resumes row + child rows via psql, then re-run the same PDF and confirm no duplicate and refreshed projections.
pytest — new DB tests pass against resume_parser_test; full suite still green (and skips-with-warning if Postgres is down).