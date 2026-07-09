# HexTech — Résumé Knowledge Graph

Turn résumé PDFs into a queryable **knowledge graph** in Neo4j, backed by
PostgreSQL. Two subsystems share one database:

- **`ocr-resume-paser/`** — parses each PDF into structured JSON and stores it in
  the `resumes` table (+ projection tables).
- **`ontogen/`** — reads those résumés, extracts entities/relationships, stages
  them in Postgres, and loads them into Neo4j.

```
PDF ──parser──► Postgres (resumes.structured) ──ontogen──► Postgres (graph_*) ──► Neo4j
```

For the full design, data flow, schema, and stage-by-stage details, see
**[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**. For a copy-paste command
reference, see **[ocr-resume-paser/docs/commands.md](ocr-resume-paser/docs/commands.md)**.

## Two extraction paths

- **Path A (deterministic, no LLM):** already-structured fields (employer,
  education, skills, projects, technologies, contact info) become graph triples
  directly — this always produces a graph.
- **Path B (LLM, free text only):** competency-question extraction over project
  descriptions, matched against Wikidata, adding facts the structured fields
  don't capture.

## Prerequisites

- Docker (for Postgres) — `sudo` on this machine.
- The Python virtualenv at `ocr-resume-paser/.venv` (used by **both** subsystems;
  Ontogen imports `resume_parser`). Deps: `sqlalchemy`, `psycopg2`, `alembic`,
  `pgvector`, `sentence-transformers`, `rdflib`, `neo4j`, plus the parser's stack.
- An OpenAI-compatible LLM endpoint (configured in `ocr-resume-paser/.env` for the
  parser and `ontogen/config.py` for Ontogen).
- A running Neo4j (bolt on `:7687`) for the final load.

## Quickstart

```bash
cd ocr-resume-paser
source .venv/bin/activate
set -a && . ./.env && set +a          # DATABASE_URL, LLM_*, ...

# 1. Database (pgvector-enabled Postgres on host port 5433).
sudo docker compose up -d
alembic upgrade head                  # parser + ontogen schema (migration 0003)

# 2. One-time: seed Ontogen's corpus stores from the on-disk files.
python ../ontogen/scripts/migrate_ontogen_files_to_db.py

# 3. Run everything: parse all resumes/*.pdf, then build the KG.
cd ..
python run_pipeline.py

# 4. Load the staged graph into Neo4j.
python ontogen/graphdb/load_to_neo4j.py
```

Then in Neo4j Browser:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 300
```

## Running the pieces separately

```bash
# Parse one résumé into Postgres.
python -m resume_parser.cli "resumes/Riyan Resume.pdf" --db-uri "$DATABASE_URL"

# Ontogen over résumés already in the database.
python ontogen/pipeline.py               # all
python ontogen/pipeline.py <resume_uuid> # one
python ontogen/pipeline.py --resume      # skip already-processed résumés

# Resume one résumé from a specific stage — earlier stages are loaded from
# outputs/stages/<uuid>/ instead of recomputed. Must be --from-stage=<stage>
# (single token, requires a uuid). stage is one of: cq_gen, cq_answer,
# relation_extract, match_validate, edc_canon, ontology, stage9_10.
python ontogen/pipeline.py <resume_uuid> --from-stage=stage9_10

# Neo4j load is incremental (only unsynced rows); --wipe clears the graph first.
python ontogen/graphdb/load_to_neo4j.py --wipe
```

**Gotchas when re-running the pipeline or reloading Neo4j:**
- `kg_staging.py`'s writers are plain `INSERT`s with no dedup/upsert — re-running
  `pipeline.py` for an already-processed résumé duplicates its `graph_entities`/
  `graph_relationships` rows rather than updating them. Clear a résumé's rows
  first (`DELETE FROM graph_relationships/graph_entities WHERE source_doc = '<uuid>'`)
  before re-staging it if you want a clean set.
- `--wipe` only clears Neo4j — it does **not** reset Postgres's `synced_to_neo4j`
  flags. Reloading after a wipe only pushes rows still marked unsynced, so any
  résumé whose rows were already synced beforehand silently disappears from
  Neo4j unless its flags are reset too
  (`UPDATE graph_entities/graph_relationships SET synced_to_neo4j = FALSE`).

## Tests

```bash
cd ocr-resume-paser
python -m pytest tests/test_db_ingest.py -q          # parser DB tests
python -m pytest ../ontogen/tests/test_ontogen_db.py -q   # ontogen DB layer
```

## Notes

- Both subsystems share one llama-server (`--parallel 1`), guarded by an advisory
  lock (`resume_parser/llm_lock.py`) so they never collide on the inference slot.
  Ontogen's LLM endpoint is a local llama-server (`http://127.0.0.1:9000/v1`,
  `ontogen/config.py`) — its structured-output support differs from vLLM's
  `guided_json`, which is requested but **not enforced** on this endpoint
  (`[llm] guided_json was requested but is not enforced` is expected); stages
  validate/repair JSON output downstream instead of relying on grammar-constrained
  decoding.
- The Neo4j default password is currently hard-coded in
  `ontogen/graphdb/config.py` — move it to an env var before any non-local use.
- Every pipeline stage's output is also mirrored to
  `outputs/stages/<resume_uuid>/stageN_*.json|.ttl` (in addition to Postgres), for
  inspection and for `--from-stage` resume (see above).
- Entity resolution (`ResumeEntityResolver` in `ontogen/stages/canonicalize.py`)
  has 3 tiers: gazetteer exact-match, gazetteer embedding match, LLM
  normalization. Cosine-similarity matches (Tier 2) and free-form LLM matches
  (Tier 3) are both LLM-verified / lexically guarded before being accepted —
  earlier versions could confidently substitute a wrong-but-similar entity
  (e.g. matching one university to a completely different one on embedding
  similarity alone). See `ontogen/logs_for_session/2026-07-07.md` for the
  root-cause writeup.
- See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) §10 for known limitations
  (notably: Path B's LLM stages need larger token budgets with the reasoning
  model to contribute facts; Path A carries the graph today).
```
