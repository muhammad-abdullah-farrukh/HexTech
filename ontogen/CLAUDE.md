# CLAUDE.md

## Known Gotchas / Environment Notes

- CUDA builds on this machine must pin **both** `-DCMAKE_CUDA_COMPILER` and
  `-DCUDAToolkit_ROOT` explicitly (e.g. to `/usr/local/cuda-12.4`). The
  `/usr/local/cuda` symlink points to a different, mismatched toolkit
  version — pinning only the compiler still lets CMake resolve the toolkit
  root through that symlink and pull in an incompatible target.
- This GPU (RTX 4500 Ada, compute capability 8.9) only needs
  `-DCMAKE_CUDA_ARCHITECTURES=89` — no need to build for other architectures.
- `pipeline.py` mirrors every stage's output to
  `outputs/stages/<resume_uuid>/stageN_*.json|.ttl` in addition to Postgres.
  `python pipeline.py <uuid> --from-stage=<stage>` resumes from it, loading
  everything before `<stage>` from disk instead of recomputing — valid keys:
  `cq_gen, cq_answer, relation_extract, match_validate, edc_canon, ontology,
  stage9_10`. Must be `--from-stage=x` (single token) — the CLI parsing has
  no `argparse` and a space-separated value gets swallowed as the résumé UUID.
- `db/kg_staging.py`'s writers (`stage_entity`, `stage_relationship`) are
  plain `INSERT`s with **no dedup/upsert**. Re-running `pipeline.py` for an
  already-processed résumé duplicates its `graph_entities`/
  `graph_relationships` rows rather than updating them in place. Clear a
  résumé's rows first (`DELETE FROM graph_relationships/graph_entities WHERE
  source_doc = '<uuid>'`) before re-staging it if you want a clean set.
- `graphdb/load_to_neo4j.py --wipe` clears Neo4j but does **not** reset
  Postgres's `synced_to_neo4j` flags. After any manual Neo4j wipe
  (`cypher-shell ... MATCH (n) DETACH DELETE n`), reloading only pushes rows
  still marked `synced_to_neo4j = FALSE` — any résumé whose rows were already
  synced before the wipe silently disappears from Neo4j and won't come back
  without resetting its flags too
  (`UPDATE graph_entities/graph_relationships SET synced_to_neo4j = FALSE
  WHERE source_doc = '<uuid>'`).
- Entity resolution (`ResumeEntityResolver` in `stages/canonicalize.py`) has
  3 tiers: gazetteer exact-match, gazetteer embedding match, LLM normalization.
  Tier 2 (embedding) now requires an LLM to confirm a match before accepting
  it (mirroring `stage6_match_validate.py`'s design for relationship/property
  matching) — cosine similarity alone was observed accepting wrong entities
  (e.g. "Air University" → "Brown University" at 0.884) because generic
  shared words inflate similarity between short, otherwise-unrelated names.
  Wikidata's own embedding space (`wikidata_properties`) is used only for
  *relationship* matching in `stage6_match_validate.py`, never for entities —
  entity Tier 2 matches against the local gazetteer's own canonical values,
  not Wikidata directly.
- No `psql` client on the host. Postgres runs in Docker
  (`ocr-resume-paser-postgres-1`) — query it via `sudo docker exec
  ocr-resume-paser-postgres-1 psql -U resume_parser -d resume_parser -c
  "..."`. Neo4j: `cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p
  5053811238 "..."`.
- When root-causing a resolver/embedding-matching bug, reproduce against the
  exact production embedding model name (`config.EMBED_MODEL`) and the full
  real candidate set/gazetteer — a smaller hand-picked candidate list or a
  slightly different model variant can show a similarity score safely under
  threshold when the real path is actually over it.
- **There are THREE `_slugify` functions minting `wd:` entity URIs** —
  `db/kg_staging.py` (Path A), `stages/stage9_10_kg.py` (Path B
  `facts_to_graph`), and `stages/canonicalize.py` (`resolve_kg_entities`
  rewrite). They MUST stay in lockstep (all lowercase, same sentinel):
  URIs are Neo4j MERGE keys, labels/`name` carry display casing. Fixing one
  and not the others splits every entity into per-path duplicate nodes
  (2026-07-08 regression: 57 case-duplicate nodes, two person nodes).
- `resolve_kg_entities` must rewrite `rdfs:label` together with the URI —
  rewriting only the URI carries the old mention's label onto the canonical
  node, and `load_to_neo4j`'s `SET e += row` then renames the legitimate
  node (observed: `wd:Machine_Learning` named "Predictive Optimization
  Models").
- Entity types `project`, `person`, `unknown` are hard-gated out of
  canonicalization (`_NEVER_RESOLVE_TYPES` in `stages/canonicalize.py`);
  the any-type gazetteer fallbacks only fire for
  `_ANY_TYPE_FALLBACK_TYPES` (tech-ish types). Don't loosen either without
  re-reading the Session 3 post-mortem in `logs_for_session/2026-07-07.md`.
- `resume_parser/llm_lock.py` is a **strict single-holder** file lock shared
  between the parser and Ontogen — its docstring literally says the server
  runs `--parallel 1`. If the LLM server is ever restarted with `-np > 1`
  (multiple parallel slots) to run résumés concurrently, this lock still
  only lets 1 process through at a time and becomes the bottleneck, not the
  server. A counting-semaphore rewrite (env-var slot cap,
  `flock`-guarded mutex against the TOCTOU race, atomic writes) was
  designed and verified working in Session 4 but was **reverted** — it is
  not in the codebase. See `logs_for_session/2026-07-07.md` Session 4 for
  the design if concurrency is revisited.
- Before raising the LLM server's `-np`/`-c`/parallelism flags: check
  `nvidia-smi` free VRAM first. `-ngl 99` (full GPU offload) on the 32B
  model plus a large `-c`/`-np` config can leave near-zero VRAM free, and
  `SentenceTransformer(EMBED_MODEL)` (`stage6_match_validate.py`,
  `canonicalize.py` — 3 call sites) defaults to CUDA with no device
  override, so it will `CUDA error: out of memory` on load — even running
  one résumé at a time, not just under concurrency. Not yet fixed; pin
  `EMBED_DEVICE=cpu` there before attempting `-np > 1` again
  (`bge-small-en` is small enough CPU inference isn't a bottleneck).
- `stages/stage7_8_ontology.py`'s `build_ontology()` dedups new-property LLM
  attempts by `attempted_new_props` (marked *before* calling
  `_new_prop_turtle()`), not just on success. A property with many relation
  instances (e.g. Path A's `hasSkill`, one per skill listed — 41 for one
  résumé) will retry the *same* LLM call once per instance if you ever
  change this back to marking "seen" only after success — a model that
  fails deterministically for that property then retries dozens of times
  for zero possible benefit (observed, Session 4).
- `pipeline.py`'s `relations` pool feeding Stage 6/EDC/ontology is
  **Path B (`cq_rels`) only** — Path A's `struct_rels` used to be merged in
  too, but Path A's own graph write (`kg_staging.stage_structured_relations`)
  never reads `match_results`/`ontology`; it maps property names to
  relationship types via a static regex and resolves entity values via
  `ResumeEntityResolver` independently. Don't re-add `struct_rels` to that
  pool without re-reading the Session 4 log — it was pure re-validation
  waste (roughly half of Zahid's runtime) for zero downstream effect.
- `config.py`'s `LLM_MODEL`/`LLM_BASE_URL` point at whatever server is
  actually running on `127.0.0.1:9000` — this has changed multiple times
  across sessions (llama-server/DeepSeek-R1-32B, then vLLM/Qwen3.5-9B).
  Check `curl 127.0.0.1:9000/v1/models` before assuming which model/context
  window (`max_model_len`) is live rather than trusting the comment, which
  can lag a manual server restart.
- **The cluster endpoint (`http://192.168.3.84:8001/v1`) can silently route
  a single request to more than one backend model.** Observed 2026-07-09:
  polling `/v1/models` 20x at 0.5s intervals returned vLLM `Qwen3.5-9B`
  (`max_model_len` 8192) ~80% of the time and a different llama.cpp-served
  model, `models/qwen3-8b-bf16.gguf` (`n_ctx` 32768), ~20% of the time —
  same URL, no error, no distinguishing header, only the response body's own
  `"model"` field tells you which one answered. This means any single-model
  assumption about that endpoint (timing, quality, `max_tokens` headroom vs
  context window) can be wrong for a fraction of calls with zero
  client-visible signal at request time.
- `stages/llm.py:call_llm()` now logs **every** attempt (success or failure)
  to `outputs/logs/llm_calls.jsonl` — timestamp, elapsed seconds, the model
  requested (`config.LLM_MODEL`) AND the model the server actually reports
  answering with (response body's `"model"` field), and a `model_mismatch`
  flag when they differ. A live warning prints on mismatch too. This exists
  specifically because of the cluster endpoint's routing behavior above — no
  per-stage timing/quality claim about "the pipeline on Qwen3.5-9B" is
  trustworthy without checking this log first. Aggregate a run's calls with
  `python scripts/summarize_llm_calls.py --run <resume_uuid>` (reads that
  résumé's own `pipeline_<uuid>.log` start/finish timestamps to scope the
  window automatically) or `--since`/`--until` for an arbitrary window.
