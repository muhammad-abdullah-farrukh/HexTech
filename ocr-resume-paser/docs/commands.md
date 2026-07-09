# Commands Reference

Everything you need to run, test, ingest, and inspect the resume parser + its Postgres
database. Commands assume you're in the repo root (`~/ocr-resume-paser`).

> **Two recurring gotchas on this machine:**
> - `docker` needs `sudo` (your user isn't in the `docker` group; passwordless sudo works).
> - Postgres is published on host port **5433** (not 5432 — a system Postgres already owns 5432).

---

## 0. One-time / per-shell setup

```bash
# Activate the Python virtualenv (has all deps: sqlalchemy, psycopg2, alembic, ...).
source .venv/bin/activate

# Load the env vars (DATABASE_URL, TEST_DATABASE_URL, LLM_* ) into this shell.
set -a && . ./.env && set +a
```

---

## 1. Postgres container (Docker Compose)

```bash
# Start Postgres in the background (pgvector/pgvector:pg17 — the pgvector-enabled
# image the Ontogen schema needs; host port 5433 -> container 5432).
sudo docker compose up -d

# Check it's running.
sudo docker compose ps

# Wait until it actually accepts connections (not just "started").
sudo docker exec ocr-resume-paser-postgres-1 pg_isready -U resume_parser -d resume_parser

# Follow the server logs.
sudo docker compose logs -f postgres

# Stop the container (keeps the data volume).
sudo docker compose down

# Stop AND delete all stored data (wipes the pgdata volume — fresh DB next time).
sudo docker compose down -v
```

---

## 2. Database schema (Alembic migrations)

```bash
# Apply all migrations to the dev database (creates/updates the schema).
alembic upgrade head

# Show the current migration the DB is on.
alembic current

# List the full migration history.
alembic history

# Roll back the most recent migration.
alembic downgrade -1

# Create a NEW empty migration to hand-edit (after changing models.py).
alembic revision -m "describe your change"
```

> The DB URL comes from `DATABASE_URL` in `.env` (read by `alembic/env.py`), not from
> `alembic.ini`.

---

## 3. Run the parser (CLI)

```bash
# Parse a PDF to JSON on stdout only (no DB, no files) — quick sanity check.
python -m resume_parser.cli "resumes/Riyan Resume.pdf"

# Parse and PERSIST to the database (the --db-uri flag is what enables ingestion).
python -m resume_parser.cli "resumes/Riyan Resume.pdf" --db-uri "$DATABASE_URL"

# Parse, write debug artifacts (raw/cleaned text + 04_structured.json) AND persist.
# Passing BOTH flags guarantees the artifact file and the DB row come from ONE
# extraction, so they always match (avoids the drift we fixed earlier).
python -m resume_parser.cli "resumes/Riyan Resume.pdf" \
  --artifacts-dir "artifacts/Riyan Resume" \
  --db-uri "$DATABASE_URL"

# Also save the JSON to a specific file.
python -m resume_parser.cli "resumes/Riyan Resume.pdf" \
  --output out.json --db-uri "$DATABASE_URL"

# Extraction + cleanup ONLY, skip the LLM (fast; validates PDF text extraction).
python -m resume_parser.cli "resumes/Riyan Resume.pdf" --no-llm --artifacts-dir "artifacts/Riyan Resume"

# Verbose logging (shows pipeline steps / LLM calls).
python -m resume_parser.cli "resumes/Riyan Resume.pdf" --db-uri "$DATABASE_URL" -v

# See every flag.
python -m resume_parser.cli --help
```

### Ingest ALL resumes at once (artifacts + DB, kept in sync)

```bash
for pdf in resumes/*.pdf; do
  echo "=== $pdf ==="
  python -m resume_parser.cli "$pdf" \
    --artifacts-dir "artifacts/$(basename "$pdf" .pdf)" \
    --db-uri "$DATABASE_URL"
done
```

---

## 4. Tests (pytest)

```bash
# Run the whole test suite.
python -m pytest -q

# Run only the database ingest tests.
python -m pytest tests/test_db_ingest.py -q

# Run one test by name.
python -m pytest tests/test_db_ingest.py -k reingest -q

# Verbose, and don't hide print()/warnings.
python -m pytest -v -s
```

> The DB tests create/use a separate `resume_parser_test` database automatically. If
> Postgres isn't running they **skip with a warning** — the rest of the suite still passes.

---

## 5. Verify the database (project script)

```bash
# Introspect the live schema + verify every stored row's projection tables match its
# JSONB. Writes docs/DB_REPORT.md and prints the report. Exit code 0 only if all pass.
python verify_db.py

# Write the report somewhere else.
python verify_db.py --output /tmp/report.md
```

---

## 6. Inspect the database (psql)

### Open an interactive shell
```bash
sudo docker exec -it ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser
```
Inside the `resume_parser=#` prompt (backslash commands need no `;`, SQL does):
```
\dt                 -- list tables
\d resumes          -- describe a table (columns, indexes, FKs)
\di                 -- list indexes
\x                  -- toggle expanded (one-field-per-line) view — good for JSONB
\c resume_parser_test  -- switch to the test database
\q                  -- quit
```

### One-off queries (run and exit, from your normal shell)
```bash
# Everyone stored + when they were ingested.
sudo docker exec ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser -c \
  "SELECT structured->>'candidate_name' AS name, id, ingested_at FROM resumes;"

# Scalar fields that live in the JSONB (email/phone/years_experience).
sudo docker exec ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser -x -c \
  "SELECT structured->>'candidate_name' AS name,
          structured->>'email' AS email,
          structured->>'phone' AS phone,
          structured->'years_experience' AS years_experience
   FROM resumes;"

# Row counts across every table.
sudo docker exec ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser -c \
  "SELECT 'resumes' t, count(*) FROM resumes
   UNION ALL SELECT 'skills', count(*) FROM skills
   UNION ALL SELECT 'work_history', count(*) FROM work_history
   UNION ALL SELECT 'education', count(*) FROM education
   UNION ALL SELECT 'projects', count(*) FROM projects;"

# The child (projection) tables.
sudo docker exec ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser -c "SELECT * FROM skills LIMIT 20;"
sudo docker exec ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser -c "SELECT company, title, start_date, end_date FROM work_history;"
sudo docker exec ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser -c "SELECT name, technologies FROM projects;"

# Full JSON for one candidate, pretty-printed.
sudo docker exec ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser -c \
  "SELECT jsonb_pretty(structured) FROM resumes WHERE structured->>'candidate_name' = 'Abdullah Zahid';"

# Search: who has a skill / used a technology (the point of the projection tables).
sudo docker exec ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser -c \
  "SELECT r.structured->>'candidate_name' FROM resumes r JOIN skills s ON s.resume_id=r.id WHERE s.skill='Python';"
sudo docker exec ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser -c \
  "SELECT r.structured->>'candidate_name', p.name FROM projects p JOIN resumes r ON r.id=p.resume_id WHERE 'FastAPI' = ANY(p.technologies);"
```

### Connect from the host instead (needs a local psql client)
```bash
# Install a client once (the bare 'psql' wrapper alone won't connect).
sudo apt-get install -y postgresql-client
# Then connect on the mapped port 5433 (no docker/sudo needed).
psql "postgresql://resume_parser:devpassword@localhost:5433/resume_parser"
```

---

## 7. Visual GUI (Adminer)

```bash
# Launch a web DB browser on http://localhost:8081 (8080 is taken by the LLM server).
sudo docker run --rm -d --name adminer \
  --network ocr-resume-paser_default \
  -p 8081:8080 adminer

# Stop it when done.
sudo docker stop adminer
```
Login at http://localhost:8081 → System: **PostgreSQL**, Server: **postgres**,
User: **resume_parser**, Password: **devpassword**, Database: **resume_parser**.

---

## 8. Handy maintenance snippets

```bash
# Delete every stored resume (child rows cascade automatically). DESTRUCTIVE.
sudo docker exec ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser -c \
  "TRUNCATE resumes CASCADE;"

# Re-ingest all resumes from their existing artifact JSON (deterministic, no LLM).
# Same rows are updated in place (dedup by pdf_hash), ingested_at refreshes.
python - <<'PY'
import json, os
from pathlib import Path
from dotenv import load_dotenv
from resume_parser.db.session import make_session_factory
from resume_parser.db.ingest import make_ingest_fn
load_dotenv()
sf = make_session_factory(os.environ["DATABASE_URL"])
ingest = make_ingest_fn(sf, "config/field_spec.json")
for pdf in Path("resumes").glob("*.pdf"):
    art = Path("artifacts") / pdf.stem / "04_structured.json"
    if art.exists():
        ingest(json.loads(art.read_text()), str(pdf))
        print("re-ingested", pdf.name)
PY
```

---

## 9. Knowledge graph (Ontogen — Part 2)

Ontogen reads résumés from the same database, extracts a knowledge graph, and
stages it into `graph_entities` / `graph_relationships` before Neo4j. It runs in
**this repo's venv** (it imports `resume_parser` for the shared models/session)
and shares the same LLM (`LLM_BASE_URL` in `.env`).

```bash
# Normal two-step flow — parse every resumes/*.pdf into Postgres, then build the
# KG over what's now in the database. Run from the HexTech repo root.
python run_pipeline.py

# Ontogen on its own, against résumés already parsed into the database.
# (DATABASE_URL must be set in this shell.)
python ../ontogen/pipeline.py                 # all résumés
python ../ontogen/pipeline.py <resume_uuid>   # one résumé
python ../ontogen/pipeline.py --resume        # skip résumés already fully processed
```

> **Shared LLM lock.** Both the parser and Ontogen hit one llama-server running
> `--parallel 1`, so each takes an advisory file lock (`$TMPDIR/hextech_llm.lock`)
> around its LLM work and pre-flights the endpoint first. If a run starts while
> another holds the lock, it aborts with `LLM appears busy [pid …]` instead of
> queuing; a lock left by a dead process is auto-cleaned. `run_pipeline.py`
> acquires/releases the lock **per PDF** (not across the whole batch) so the slot
> isn't held during inter-PDF disk I/O.

```bash
# One-time: seed the corpus stores (gazetteers, canon store, wikidata properties
# + embeddings) from the on-disk files into the new tables. Re-embeds with
# bge-small-en. Run once, from the ontogen dir.
python ../ontogen/scripts/migrate_ontogen_files_to_db.py

# Load the staged KG into Neo4j (incremental — only pushes unsynced rows).
python ../ontogen/graphdb/load_to_neo4j.py
python ../ontogen/graphdb/load_to_neo4j.py --wipe   # clear the graph first
```

> Ontogen's tables are created by the same Alembic chain as the parser's
> (migration `0003_ontogen_schema`), so `alembic upgrade head` (§2) builds them
> too — it needs the pgvector image from §1.

