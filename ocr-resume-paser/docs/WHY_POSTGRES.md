# Why PostgreSQL — Database Choice Rationale

**Project:** Hextech — resume PDF → structured JSON → Wikidata-aligned knowledge graph
**Scope of this document:** justify PostgreSQL (with the `pgvector` extension) as the single database underpinning both the parser and the Ontogen knowledge-graph pipeline, staged ahead of Neo4j — and, more generally, evaluate PostgreSQL as an RDBMS choice against its main alternatives, including where it stops being the right answer and how to substitute or extend it if that happens.

*Last updated: July 2026. Benchmark figures cited below are from sources published in late 2025–mid 2026; re-verify before relying on exact numbers for a production decision, as database performance claims shift with each major release.*

---

## 1. Summary

The system has three distinct data-storage needs, arising from its two pipelines:

1. **Parser output** — structured, semi-uniform JSON per résumé (`04_structured.json`), needing durable storage, dedup, and indexed lookup.
2. **Knowledge-graph staging** — cross-document vector-searchable stores (EDC canon store, Wikidata property embeddings), run/state tracking across a 10-stage pipeline, and staged nodes/edges before loading into Neo4j.
3. **Eventual production concerns** — encryption, access control, and enough operational simplicity that one person can run and reason about the whole system.

PostgreSQL, with the `pgvector` extension, is the only option evaluated that satisfies all three without requiring a second database engine to be run alongside it. Sections 2–9 make that case for this project specifically. Sections 10–12 broaden out to PostgreSQL as a general RDBMS choice — including the concrete scenarios where it is *not* the best tool, and what to do about it without discarding the work already built on it.

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

The parser's structured output is a natural fit for document storage on the surface — nested arrays (`skills[]`, `work_history[]`, `education[]`), no rigid uniform shape across every résumé. But the actual shape of the data is an **80/20 split**: most fields are always present and simply structured (`candidate_name`, `email`, `years_experience`), and only a minority genuinely vary in shape. This is close to a textbook case for JSONB rather than a document store: JSONB is a pragmatic middle ground — structured columns for fields that need indexing and relational integrity, with JSONB for flexible, schema-less sub-documents — and industry guidance converges on the same 80/20 heuristic used here: when most data is structured and only a minority is genuinely flexible, JSONB tends to beat a pure document model [1][2].

Where MongoDB has a real, measured edge is on **sustained partial-document updates under high concurrency** — a January 2026 MongoDB-published benchmark (13M documents, 256 concurrent users) showed steadier throughput and lower tail latency than Postgres JSONB for that specific workload, attributable to MongoDB's document-native storage engine versus Postgres's row-and-version (MVCC) model, which can accumulate dead tuples and index churn under repeated partial rewrites of large JSONB blobs [3]. That's not this project's workload — résumés are written once and occasionally re-ingested wholesale, not repeatedly patched field-by-field — so the one place MongoDB demonstrably wins doesn't apply here.

What MongoDB offers beyond this — sharded multi-region writes, massive insert throughput, a document-native aggregation pipeline, and embedded-document reads with no joins required — are strengths that matter at a scale and access pattern this project does not operate at [4][5]. Adopting it would mean paying its operational cost (a separate service to run, monitor, and secure) for capabilities that go unused. Worth noting separately: MongoDB's Server Side Public License (SSPL) is not OSI-approved open source, which matters if this project — or a spin-off of it — were ever offered as a hosted service; PostgreSQL's license carries no such restriction [4].

---

## 4. Why not a separate vector database

The Ontogen pipeline's EDC canon store and Wikidata property matching both need embedding similarity search. The pipeline's own documentation already flags that the current brute-force in-memory cosine scan over `.npy` files will not scale past a small corpus.

`pgvector` solves exactly this — HNSW-indexed similarity search — as a Postgres extension, not a separate system. Independent 2026 benchmarking against dedicated vector databases (Pinecone, Qdrant, Weaviate) is consistent on the tradeoff: pgvector is genuinely production-grade — companies including Supabase, Neon, and Instacart run it at meaningful scale — and for under roughly 10 million vectors with HNSW indexing it's fast enough that you additionally get joins, transactions, and row-level access control for free, which a standalone vector database doesn't provide [6][7]. The gap opens up specifically at very large scale (tens to hundreds of millions of vectors), where dedicated engines maintain high recall without manual tuning while pgvector needs careful HNSW parameter tuning to keep up, and single-node Postgres has a practical ceiling around ~50M vectors on well-provisioned hardware [7].

This project's corpus (low-thousands of résumés, a few thousand Wikidata properties, tens of canon-store entries) is orders of magnitude below where that gap becomes relevant. Running a dedicated vector database alongside Postgres here would mean:

- A second connection, second set of credentials, second thing to keep running.
- Cross-database joins (or application-level stitching) every time a vector search result needs to be correlated with a structured record — which is *every* dedup and canonicalization operation in this pipeline.
- No meaningful performance gain at this project's data volume — one direct benchmark reports pgvector HNSW query latency of 5–8ms at production scale, with embedding *generation* (not the database) as the actual bottleneck in the overall pipeline [8].

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

- **Encryption at rest** in production Postgres deployments comes from the hosting layer, not a Postgres-specific feature. AWS RDS/Aurora encrypt storage, automated backups, snapshots, and read replicas using AES-256, managed through KMS — and as of February 18, 2026, new Aurora clusters are encrypted by default using an AWS-owned key even without explicit configuration, with the option to specify a customer-managed key for compliance needs [9][10]. This is the standard architecture used by virtually every production relational or document database in the cloud, so the database engine choice doesn't constrain the encryption story.
- **Column-level encryption** for specific sensitive fields (`pgcrypto`) is available natively as a core-adjacent extension, without a vendor fork.
- **Concurrency safety** (`ON CONFLICT` atomic upserts, connection pooling, `SELECT ... FOR UPDATE SKIP LOCKED` for future parallel workers) is built into standard Postgres — no additional system needed to make batch/parallel processing safe later.
- **Transactional DDL** — PostgreSQL 17 allows DDL statements (schema changes) to be included in multi-statement transaction blocks alongside DML, committed or rolled back as one unit, with savepoint support [11]. This is a genuine correctness advantage relevant to this project's Alembic-managed migration history: a bad migration can be rolled back cleanly mid-transaction, which is not universally true of every RDBMS (MySQL's DDL support is atomic per-statement but has historically been more limited in multi-statement transactional grouping [11]).

---

## 8. What this design deliberately does *not* need

- No sharding, read replicas, or horizontal scale-out — the data volume (low-thousands of résumés, tens of thousands of extracted triples) doesn't approach the scale where these matter.
- No separate message broker/task queue database — Celery's broker (if introduced later) is a different concern from structured data storage and doesn't change this reasoning.
- No multi-region or high-availability requirements — this is a single-operator research/internship-scale system, not a multi-tenant SaaS product.

Choosing a more "scalable" or "specialized" database now would be solving problems this project doesn't have, at the cost of running and securing more than one system.

---

## 9. Conclusion (project-specific)

PostgreSQL with `pgvector` is not a compromise between three specialized databases — it is the single tool whose native feature set (JSONB, vector indexing, relational integrity, constraint-based concurrency safety) covers what both pipelines actually need, at the scale they actually operate at, while keeping the parser and the knowledge-graph pipeline joinable through plain foreign keys instead of cross-database operations. The result is one database to run, secure, back up, and reason about — for a system built and maintained by a small team.

---

## 10. PostgreSQL as a general RDBMS choice

Beyond this project, PostgreSQL is widely regarded as the strongest general-purpose default among open-source relational databases as of 2026 — multiple independent 2026 comparisons converge on recommending it as the default for new projects absent a specific reason otherwise [1][12][13][14]. The reasons generalize beyond this project:

- **Standards compliance.** PostgreSQL is considered the most SQL-standard-compliant open-source database available, with native support for window functions, CTEs, lateral joins, recursive queries, and full-text search [13].
- **MVCC concurrency.** PostgreSQL was the first DBMS to implement multi-version concurrency control, letting readers and writers proceed without blocking each other [15]. This is a materially different model from lock-based concurrency and tends to perform better under mixed concurrent read/write load.
- **Full ACID compliance unconditionally.** PostgreSQL is ACID-compliant across all its storage mechanisms with no exceptions. MySQL is only ACID-compliant when using the InnoDB storage engine — its older MyISAM engine is not transactional, and it's possible to accidentally create a non-transactional table if a developer isn't careful [12].
- **Query planner sophistication.** Benchmarks reported in 2026 show PostgreSQL completing complex transactional workloads (multi-table joins, aggregations, window functions, subqueries) meaningfully faster than MySQL in several independent tests — figures cited range from roughly 2x on TPC-C-style complex transactions to over 10x on analytical queries with heavy aggregation, though exact multipliers vary by benchmark methodology and should be treated as directional, not exact [12].
- **Extension ecosystem.** Beyond `pgvector`, the same "one engine, multiple roles" pattern used in this project extends to PostGIS (geospatial), TimescaleDB (time-series), and 300+ other extensions, letting Postgres absorb specialized workloads that would otherwise require separate database systems [12][14].
- **Licensing.** PostgreSQL's license is fully open source and permissive, with no company able to unilaterally change licensing terms the way MongoDB's SSPL or (historically) MySQL's Oracle ownership have raised questions for some organizations [4][14].

---

## 11. Where PostgreSQL is *not* the recommended choice

Being the strongest general default does not mean PostgreSQL is correct for every workload. The following are the concrete scenarios where the honest answer is a different tool — along with what actually changes, and how to get there without discarding a Postgres-based system already in production.

### 11.1 Very high-throughput, simple, single-table reads → MySQL

**When this applies:** read-heavy workloads with simple queries against one table at a time — content management systems, WordPress-style sites, simple API endpoints with predictable access patterns. Benchmark data collected across three sources (in-house Sysbench tests, Percona's 2025 benchmark suite, and community pgbench results) consistently shows MySQL outperforming PostgreSQL on simple single-table SELECTs by roughly 20–35%, attributable to MySQL's query cache, InnoDB's clustered index design, and a lighter-weight connection model [14]. PostgreSQL's advantage — a more sophisticated query planner — is wasted overhead when queries are simple and don't need it; PostgreSQL also forks a new OS process per connection, which uses more memory per connection than MySQL's thread-based model, and needs a connection pooler like PgBouncer to stay efficient beyond roughly 100 concurrent connections [12].

**How this would apply here, and the substitution path:** none of this project's actual access patterns are simple single-table reads at high concurrency — every meaningful query in this pipeline (résumé → skills → entity resolution → graph staging) involves joins or JSONB lookups. If a future feature genuinely needed this pattern (e.g., a public-facing "browse candidates" page under heavy simple traffic), the practical path is **not** a full migration — it's introducing a read-optimized cache or read replica in front of the existing Postgres database (see 11.5), or, only if truly justified by measured load, standing up a narrow MySQL-backed read service for that one feature while Postgres remains the system of record. A full migration off Postgres for this reason alone would be solving a problem this project doesn't have.

### 11.2 Massive horizontal write scale / true multi-region sharding → Citus (stay on Postgres) or a natively distributed database

**When this applies:** tables reaching hundreds of millions of rows or beyond, where a single Postgres node's CPU, memory, and I/O become the bottleneck even after vertical scaling and query tuning [16].

**The good news for an already-Postgres system:** the first substitution to reach for isn't a different database at all. **Citus is an open-source PostgreSQL extension, not a fork** — it transforms a single Postgres instance into a distributed cluster while keeping the exact same SQL interface, tooling, and client libraries [17][18]. A Citus cluster has one coordinator node (query routing/planning) and multiple worker nodes (shard storage); your application continues talking to Postgres exactly as before, with Citus handling shard distribution transparently [16][17]. Reported benchmarks show up to ~40x speedups on analytical queries with an 8-node cluster versus a single node on ~100GB datasets [19]. As of Citus 12, schema-based sharding removes the need to even choose a distribution key for many multi-tenant patterns — each tenant simply gets its own schema, and Citus handles placement and rebalancing automatically [20].

**How to integrate it here if this project ever reached that scale:** because Citus is an extension rather than a different engine, the migration path is additive, not a rewrite — `CREATE EXTENSION citus;`, then designating a distribution column (for this project, `resumes.id` or a tenant-equivalent column would be the natural choice, since nearly every table already carries a foreign key back to it) and calling `create_distributed_table()` on the large tables. Foreign keys, JSONB columns, and `pgvector` all continue to work as before, provided joins and updates include the distribution column [16]. This is explicitly *not* recommended prematurely: Citus is described by its own maintainers as unnecessary if a workload will never exceed a single node's capacity, and it adds real operational complexity — a coordinator node is a new single point of failure unless made highly available, and cross-shard joins that don't route through the distribution column suffer from slow broadcast joins [16][17]. This project's current and foreseeable scale (low-thousands of résumés) is nowhere near where Citus becomes necessary.

**If Citus genuinely isn't enough** (true multi-region active-active writes, planet-scale distribution), the honest answer moves outside the Postgres ecosystem entirely — to something like CockroachDB or a managed globally-distributed service. At that scale, a full data migration is unavoidable regardless of starting database, since no single-region relational system natively solves multi-region active-active writes.

### 11.3 Pure document/nested-data workloads with high write throughput → MongoDB

**When this applies:** data that is genuinely and permanently schemaless — content catalogs with wildly different attributes per item, IoT telemetry, logs — where every document has a meaningfully different shape and imposing a schema adds no value, and where documents are read/written as a whole embedded unit with no relational joins needed [4][14][21]. MongoDB's advantage compounds under sustained high-concurrency partial-document updates specifically, per the benchmark cited in Section 3 [3].

**How this would apply here, and the integration path:** this project's data is the *opposite* case — an explicit 80/20 structured/flexible split with `resumes.structured` deliberately backed by relational projection tables (`skills`, `work_history`, `education`, `projects`) precisely because most fields are queried relationally. If a future extension of this system introduced a genuinely document-shaped workload unrelated to résumé structure (e.g., ingesting raw scraped job postings with wildly inconsistent shapes, purely for archival, never queried relationally), the recommended pattern is **not** replacing Postgres — multiple 2026 sources converge on running both side by side: Postgres for transactional/relational data where relationships and consistency matter, MongoDB for the specific content/log/catalog workload where schema flexibility and write throughput dominate [4][14]. This project's `resumes` table would remain in Postgres; only the new, genuinely-document-shaped collection would live in MongoDB, joined at the application layer via a shared identifier — not a wholesale migration.

### 11.4 Vector search at very large scale (tens to hundreds of millions of vectors) → a dedicated vector database

**When this applies:** as covered in Section 4, pgvector's practical ceiling is roughly 50M vectors on a well-provisioned single Postgres node before recall and query latency require increasingly careful manual tuning that dedicated engines handle automatically [7]. This project's corpus is many orders of magnitude below that threshold and isn't expected to approach it.

**The integration/substitution path if it ever did:** this is the *easiest* of the scenarios in this section to migrate, specifically because of how this project's schema is already structured. `canon_store.embedding` and `wikidata_properties.embedding` are isolated `vector(384)` columns, not entangled with the relational data around them. Migrating to a dedicated vector database (Qdrant is the most commonly recommended self-hosted option for teams wanting to keep infrastructure ownership similar to the current Postgres-only setup, given its Rust-based performance and strong filtered-search support [6][7][22]) means: (1) keep the *rows* — `label`, `definition`, `source_doc`, `pid` — in Postgres exactly as now, since that's still relational metadata; (2) export just the `embedding` vectors plus a foreign-key-equivalent ID into the vector database's collection; (3) at query time, search the vector database for nearest-neighbor IDs, then join those IDs back against the Postgres rows for the actual record data. This is a **dual-store pattern**, not a full migration — Postgres remains the system of record for everything except the vector index itself, and the schema changes required are additive (drop the `embedding` column and its HNSW index once confirmed working elsewhere), not a rewrite of the surrounding tables.

### 11.5 Sub-millisecond caching / extremely hot read paths → Redis (alongside, not instead of, Postgres)

**When this applies:** this isn't really a "PostgreSQL vs. X" scenario — no relational database, including Postgres, is designed to be a sub-millisecond key-value cache under extreme read concurrency. Redis exists specifically for this: an in-memory store for hot-path lookups (session data, rate limiting, frequently-requested query results) that would otherwise repeatedly hit the same database rows.

**The integration path here:** purely additive — Redis sits in front of Postgres as a cache layer, not a replacement for any part of the current schema. Nothing in this project currently has a hot-path read pattern that would justify this (the pipelines are batch/CLI-driven, not serving live request traffic), but if this system were ever wrapped in a web-facing query API (e.g., "search the knowledge graph"), a Redis cache in front of frequent Neo4j/Postgres queries would be the natural next step — added alongside the existing database layer, with Postgres/Neo4j remaining the source of truth and Redis holding disposable, regenerable cached results.

---

## 12. Decision summary

| Scenario | Recommended tool | Migration effort from this project's current Postgres setup |
|---|---|---|
| Current scale (low-thousands of résumés, batch pipelines) | **PostgreSQL + pgvector** (current choice) | — |
| Simple, high-concurrency single-table reads | MySQL, or a Postgres read replica/cache first | Low if solved via caching; only relevant for a narrow future feature |
| Table(s) reach hundreds of millions of rows | **Citus** (Postgres extension, not a new engine) | Low–medium — additive, same SQL interface, no rewrite |
| True multi-region active-active writes at planet scale | CockroachDB or managed distributed service | High — genuine migration, but only relevant far beyond this project's scope |
| Genuinely schemaless, high-write document data unrelated to résumés | MongoDB, run alongside Postgres | Low — new workload only, existing tables untouched |
| Vector search beyond ~50M vectors | Qdrant (or similar), dual-store with Postgres | Low–medium — additive dual-store, existing relational rows untouched |
| Sub-millisecond hot-path caching | Redis, alongside Postgres | Low — purely additive cache layer |

The pattern across every row of this table is deliberate: PostgreSQL's extension ecosystem and JSONB flexibility mean that in nearly every scenario where a specialized tool becomes genuinely necessary, it can be **added alongside** the existing Postgres database rather than requiring a full replacement of it. The one scenario requiring a true migration — planet-scale multi-region writes — is far beyond any plausible future scope of this project, and would require a full migration regardless of which database this system started on.

---

## References

[1] Webyot Technologies, "PostgreSQL vs MongoDB: The Honest Database Comparison for 2026," April 2026. https://webyot.in/learning/postgresql-vs-mongodb.html

[2] tech-insider.org, "MongoDB vs PostgreSQL: When NoSQL Actually Wins," 2026. https://tech-insider.org/mongodb-vs-postgresql-2026/

[3] techbytes.app, "PostgreSQL 17 JSON vs MongoDB: Benchmark Reality Check," April 2026. https://techbytes.app/posts/postgresql-17-json-vs-mongodb-benchmark-reality-check/

[4] TechPlained, "PostgreSQL vs MongoDB (2026): Benchmarks & Use Cases," April 2026. https://www.techplained.com/postgresql-vs-mongodb

[5] AI2SQL, "PostgreSQL vs MongoDB (2026): Speed, Scale & Cost — Which Wins?," March 2026. https://builder.ai2sql.io/blog/postgresql-vs-mongodb

[6] Kalvium Labs, "pgvector vs Pinecone vs Qdrant vs Weaviate (2026): Which We Actually Use in Production," April 2026. https://www.kalviumlabs.ai/blog/vector-databases-compared-pgvector-pinecone-qdrant-weaviate/

[7] Groovyweb, "Pinecone vs pgvector vs Chroma vs Weaviate (2026): Best Vector DB by Use Case," 2026. https://www.groovyweb.co/blog/vector-database-comparison-2026

[8] Vecstore, "pgvector vs Pinecone vs Qdrant: 2026 Benchmarks," April 2026. https://vecstore.app/blog/vector-database-performance-compared

[9] AWS Database Blog, "Use default encryption at rest for new Amazon Aurora clusters," February 2026. https://aws.amazon.com/blogs/database/use-default-encryption-at-rest-for-new-amazon-aurora-clusters/

[10] AWS Documentation, "Encrypting Amazon Aurora resources." https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/Overview.Encryption.html

[11] Bytebase, "Postgres vs. MySQL: DDL Transaction Difference." https://www.bytebase.com/blog/postgres-vs-mysql-ddl-transaction/

[12] techsy.io, "PostgreSQL vs MySQL in 2026: The Definitive Comparison," April 2026. https://techsy.io/en/blog/postgresql-vs-mysql

[13] ZeonEdge, "PostgreSQL vs MySQL in 2026: The Definitive Comparison for Modern Applications," December 2025. https://zeonedge.com/blog/postgresql-vs-mysql-comparison-2026

[14] tech-insider.org, "PostgreSQL vs MySQL 2026: 3.7x JSON Gap and 300 Extensions [Tested]," April 2026. https://tech-insider.org/postgresql-vs-mysql-2026-2/

[15] Integrate.io, "PostgreSQL vs MySQL: The Critical Differences," January 2026. https://www.integrate.io/blog/postgresql-vs-mysql-which-one-is-better-for-your-use-case/

[16] Stormatics, "A Beginner's Guide to Sharding PostgreSQL with Citus," December 2025. https://stormatics.tech/blogs/a-beginners-guide-to-sharding-postgresql-with-citus

[17] Citus Data, "Get Started with Citus – Distributed PostgreSQL At Any Scale." https://www.citusdata.com/getting-started/

[18] GitHub, "citusdata/citus: Distributed PostgreSQL as an extension." https://github.com/citusdata/citus

[19] Citus Data, benchmark chart referenced on the Citus getting-started page (8-node cluster vs. single node, ~100GB GitHub archive dataset). https://www.citusdata.com/getting-started/

[20] Citus Data, "Citus 12: Schema-based sharding for PostgreSQL," July 2023 (feature current as of 2026). https://www.citusdata.com/blog/2023/07/18/citus-12-schema-based-sharding-for-postgres/

[21] Medium (Krpsanthoshkumar), "MongoDB vs PostgreSQL JSONB: Which One Should You Choose for Storing JSON Data?," December 2025. https://medium.com/@krpsanthoshkumar/mongodb-vs-postgresql-jsonb-which-one-should-you-choose-for-storing-json-data-628aa21cf599

[22] callmissed.com, "Vector Databases 2026: Pinecone vs Qdrant vs Weaviate vs pgvector," May 2026. https://www.callmissed.com/en/blog/vector-database-comparison-2026
