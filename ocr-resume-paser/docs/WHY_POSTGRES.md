# Why PostgreSQL — Database Choice Rationale

**Project:** Hextech — resume PDF → structured JSON → Wikidata-aligned knowledge graph
**Scope of this document:** justify PostgreSQL (with the `pgvector` extension) as the single database underpinning both the parser and the Ontogen knowledge-graph pipeline, staged ahead of Neo4j.

---

## 1. Summary

The system has three distinct data-storage needs, arising from its two pipelines:

1. **Parser output** — structured, semi-uniform JSON per résumé (`04_structured.json`), needing durable storage, dedup, and indexed lookup.
2. **Knowledge-graph staging** — cross-document vector-searchable stores (EDC canon store, Wikidata property embeddings), run/state tracking across a 10-stage pipeline, and staged nodes/edges before loading into Neo4j.
3. **Eventual production concerns** — encryption, access control, and enough operational simplicity that one person can run and reason about the whole system.

PostgreSQL, with the `pgvector` extension, is the only option evaluated that satisfies all three without requiring a second database engine to be run alongside it.

---

## 2. The core argument: one engine, three roles

The three storage needs above look, on the surface, like they belong to three different kinds of database — a document store (flexible JSON), a vector database (embeddings for dedup/similarity), and a relational store (indexed lookups, run/state tracking, referential integrity). Historically, that would mean running three separate systems.

Postgres collapses this into one engine because it natively supports all three access patterns as first-class column types, not bolted-on features:

| Need | Postgres feature | What it replaces |
|---|---|---|
| Flexible, semi-structured JSON | `JSONB` column type | A document database (MongoDB) |
| Similarity search over embeddings | `pgvector` extension, HNSW index | A dedicated vector database (Pinecone, Weaviate) |
| Structured lookups, joins, constraints | Native relational tables | What a document/vector store would *lack* |

This isn't a compromise where each role is served worse than a specialized tool would serve it — for the *scale* this project operates at (hundreds to low-thousands of résumés), each of Postgres's implementations of these features is more than sufficient, and the operational savings of running one database instead of three are substantial for a project maintained by one or two people.

---

## 3. Why not a document database (e.g. MongoDB)

The parser's structured output is a natural fit for document storage on the surface — nested arrays (`skills[]`, `work_history[]`, `education[]`), no rigid uniform shape across every résumé. But the actual shape of the data is an **80/20 split**: most fields are always present and simply structured (`candidate_name`, `email`, `years_experience`), and only a minority genuinely vary in shape. Postgres's `JSONB` column type is built exactly for this split — structured columns for what's fixed, a JSONB column for what's flexible — without requiring a second database.

What MongoDB offers beyond this — sharded multi-region writes, massive insert throughput, a document-native aggregation pipeline — are strengths that matter at a scale and write-throughput this project does not operate at. Adopting it would mean paying its operational cost (a separate service to run, monitor, and secure) for capabilities that go unused.

---

## 4. Why not a separate vector database

The Ontogen pipeline's EDC canon store and Wikidata property matching both need embedding similarity search. The pipeline's own documentation already flags that the current brute-force in-memory cosine scan over `.npy` files will not scale past a small corpus.

`pgvector` solves exactly this — HNSW-indexed similarity search — as a Postgres extension, not a separate system. Running a dedicated vector database (Pinecone, Weaviate, Qdrant) alongside Postgres would mean:

- A second connection, second set of credentials, second thing to keep running.
- Cross-database joins (or application-level stitching) every time a vector search result needs to be correlated with a structured record — which is *every* dedup and canonicalization operation in this pipeline.
- No meaningful performance gain at this project's data volume — HNSW indexing in Postgres handles low-thousands to low-millions of vectors comfortably; a dedicated vector DB's advantages appear at a scale far beyond what this system processes.

Keeping vectors in the same database as the records they describe means a dedup or canonicalization query is a single SQL statement, not a distributed operation.

---

## 5. Why this matters specifically for the two-pipeline architecture

The parser and Ontogen pipelines are sequential stages of one system, not independent services:

- Ontogen's input **is** a row in the parser's `resumes` table.
- Every entity/relationship Ontogen stages needs to trace back to the résumé it came from (`document_id` foreign keys, `entity_uri_map`).
- Cross-document entity resolution (the EDC canon store, the gazetteers) needs to query across *all* résumés processed by the parser, not just the one currently being extracted.

A single Postgres database makes these relationships plain foreign keys and joins. Splitting parser storage and KG staging across two databases would turn every one of these — which are core to how the pipeline works, not edge cases — into a cross-database operation, for no corresponding benefit: nothing in this system needs independent scaling, independent uptime, or independent failure isolation between the two pipelines.

---

## 6. Why this matters for the eventual Neo4j handoff

Neo4j remains the right tool for what it's good at — graph traversal, pattern matching across the final knowledge graph — and this design doesn't compete with that. Postgres's role is upstream and complementary:

- `graph_entities` / `graph_relationships` tables stage nodes and edges *before* they're loaded into Neo4j, giving a place to run corpus-wide entity resolution (merging duplicate "TeseraVR" mentions across different résumés) before the graph is built — something that's far more naturally expressed as SQL `UPDATE`s on staging rows than as in-place graph mutations.
- A `synced_to_neo4j` flag on each staged row means only *changed* records get re-loaded into Neo4j, not the whole graph on every run — a plain boolean column, not a feature Neo4j itself needs to provide.
- The `uri` join key ties a Postgres row, a Neo4j node, and the original extracted Turtle together, so any node in the final graph can be traced back to its source résumé.

Postgres is the system of record and staging layer; Neo4j is the queryable graph built from it. Each does the job it's actually suited for.

---

## 7. Why this matters for production concerns (encryption, scale)

- **Encryption at rest** in production Postgres deployments comes from the hosting layer (managed providers like RDS/Aurora encrypt automatically via KMS) rather than a Postgres-specific feature — the same pattern used by virtually every production relational or document database in the cloud. This isn't a Postgres weakness; it's the standard architecture, and it means the database choice doesn't constrain the encryption story.
- **Column-level encryption** for specific sensitive fields (`pgcrypto`) is available natively as a core-adjacent extension, without a vendor fork.
- **Concurrency safety** (`ON CONFLICT` atomic upserts, connection pooling, `SELECT ... FOR UPDATE SKIP LOCKED` for future parallel workers) is built into standard Postgres — no additional system needed to make batch/parallel processing safe later.

---

## 8. What this design deliberately does *not* need

- No sharding, read replicas, or horizontal scale-out — the data volume (low-thousands of résumés, tens of thousands of extracted triples) doesn't approach the scale where these matter.
- No separate message broker/task queue database — Celery's broker (if introduced later) is a different concern from structured data storage and doesn't change this reasoning.
- No multi-region or high-availability requirements — this is a single-operator research/internship-scale system, not a multi-tenant SaaS product.

Choosing a more "scalable" or "specialized" database now would be solving problems this project doesn't have, at the cost of running and securing more than one system.

---

## 9. Conclusion

PostgreSQL with `pgvector` is not a compromise between three specialized databases — it is the single tool whose native feature set (JSONB, vector indexing, relational integrity, constraint-based concurrency safety) covers what both pipelines actually need, at the scale they actually operate at, while keeping the parser and the knowledge-graph pipeline joinable through plain foreign keys instead of cross-database operations. The result is one database to run, secure, back up, and reason about — for a system built and maintained by a small team.
