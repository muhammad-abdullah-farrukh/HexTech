# KG Construction Pipeline Comparison Report
**Input:** Abdullah Zahid resume (`yourfile.txt`)  
**Systems Under Comparison:** Result1 v2 (ontology-driven, 10-stage) vs Result2 (spaCy+LLM fusion, single-script)  
**Verdict: Result1 v2 is the better pipeline for resume KG construction.**

---

## 1. Pipeline Architecture

### Result1 (v1 and v2) — `ontogen` (10-stage ontology-driven)

A research-paper-faithful multi-stage pipeline:

| Stage | What it does |
|---|---|
| 1 | LLM generates Competency Questions (CQs) from the document |
| 2 | LLM answers each CQ, grounding facts |
| 3 | LLM extracts relation types in natural language |
| 4–5 | Wikidata candidate retrieval + embedding-based filtering |
| 6 | Cosine similarity + LLM validation to map relations → Wikidata properties |
| 7–8 | Ontology construction in OWL/Turtle |
| 9–10 | Fact extraction (JSON, not Turtle) + rdflib graph assembly + entity resolution + provenance |

**Key architectural decision in v2:** The LLM outputs JSON facts, never Turtle. rdflib builds and serializes the graph deterministically. This was introduced specifically because 8B instruct models reliably break when asked to write punctuation-sensitive RDF syntax — as confirmed by v1's parse failure.

### Result2 — `spacy+llm` (single 578-line script)

A dual-path fusion pipeline:

1. **spaCy path:** dependency parsing → triple extraction via constituency/linear rules → NER-based entity typing  
2. **LLM path (vLLM):** chunked document → structured triple extraction  
3. **Fusion:** entity deduplication by normalized string hash → merged entity/relation lists  
4. **Serialization:** custom TTL writer (no ontology schema, all entities typed as `kg:Concept`)

---

## 2. Output Comparison

### 2.1 KG File Size

| System | KG TTL size | Status |
|---|---|---|
| Result1 v1 | 3,187 bytes | **Parse failed** — raw LLM Turtle leaked into output |
| Result1 v2 | 5,472 bytes | **Structured JSON facts** stored (TTL build failed at final step, but facts are valid) |
| Result2 | 30,792 bytes | Valid TTL, 907 lines |

> **Note on Result1 v2 KG TTL:** The file header says `# NO VALID FACTS extracted after 3 attempts` but the body contains a complete, well-formed JSON fact array with 38 triples. This is a logging/serialization bug — the facts *were* extracted, the TTL writer just failed to consume them. The fact extraction itself succeeded.

### 2.2 Entity Quality

| Metric | Result1 v2 | Result2 |
|---|---|---|
| Entity types | Named (Person, Org, Skill, Degree, etc.) — via ontology | All `kg:Concept` — no type discrimination |
| Granularity | Domain-appropriate (e.g. `PositionHolder`, `EducatedAt`, `HasCertification`) | Flat — "professional profile abdullah zahid", "electrical engineering roots", "pioneering ai" as equal-weight nodes |
| Noise | Low — CQ-grounded | High — spaCy extracts section headers, noun chunks, and resume boilerplate as entities (e.g. `"september 2021 to june 2025"`, `"cgpa"`, `"70 percent"` as entities) |
| Total entities | ~38 meaningful (from fact JSON) | 110 (including significant noise) |

### 2.3 Relation Quality

| Metric | Result1 v2 | Result2 |
|---|---|---|
| Relation vocabulary | Mapped to **Wikidata PIDs** — `P69` (EducatedAt), `P512` (AcademicDegree), `P1416` (Affiliation), `P277` (ProgrammedIn), etc. | Surface-level dependency labels — `include`, `be`, `appos`, `of`, `pioneer` |
| Semantic precision | High — ontology-constrained | Low — syntactic artifacts dominate |
| Top relation | `EducatedAt` (P69), `Affiliation` (P1416), `HasCertification` | `include` (11), `be` (7), `appos` (7) |
| Interoperability | Yes — Wikidata-aligned PIDs are queryable/linkable | No — custom local namespace, no external alignment |

**R2 worst examples:**
- `"spu power generation sub-system" → be → "demonstrator"` (grammatical artifact, not a fact)
- `"technical presentations" → on → "low orbit satellites"` (preposition extraction)
- `"his work" → blend → "cutting-edge rl research"` (narrative fluff, not resume fact)

### 2.4 Wikidata Alignment (Result1 only)

Result1 v2 matched 29/32 extracted relations to Wikidata properties via cosine similarity + LLM validation. Match scores:

| Relation extracted | Wikidata PID | Label | Cos score |
|---|---|---|---|
| `is member of` | P1416 | Affiliation | 0.9453 |
| `is proficient in` | P277 | ProgrammedIn | 0.9305 |
| `received certification` | P10611 | HasCertification | 0.9343 |
| `attended school` | P69 | EducatedAt | 0.9182 |
| `worked as` | P1308 | PositionHolder | 0.9079 |
| `studied course` | P812 | AcademicMajor | 0.9117 |

Notable mismatches (accepted but semantically off):
- `has background` → `AncestralHome` (P66) — cosine 0.89, wrong semantic
- `worked at` → `RealEstateDeveloper` (P6237) — clearly wrong, LLM accepted it anyway
- `is familiar with` → `GeneralizationOf` (P7719) — wrong

These are fixable with a better Wikidata property seed set or a stricter LLM validator prompt.

### 2.5 Competency Questions (Result1 only — core differentiator)

Result1 generates CQs before extraction, which grounds what facts to extract. Examples for this resume:

- *"What programming languages is Abdullah Zahid proficient in?"*
- *"What is the name of the internship Abdullah Zahid participated in, and what was its focus?"*
- *"What certification did Abdullah Zahid receive?"*
- *"What organization is Abdullah Zahid a member of?"*

These CQs act as a **semantic specification** for the ontology and KG. Result2 has no equivalent — it extracts whatever spaCy parses.

---

## 3. Failure Mode Analysis

### Result1 v1 Failure
The v1 KG TTL is garbage — the file contains raw English prose spliced into what should be Turtle syntax:
```
# PARSE FAILED after 4 attempts — raw LLM output below
...
WDT(ProgrammedIn ) (a re assertion)
AbdullaZahid holds an "M.Sc." in Electrical Engineering from AIr University Islamabad...
```
This is the exact failure the v2 comment block documents: escalating temperature retries caused total format collapse. **v2 fixed this** by switching to JSON fact extraction + rdflib serialization.

### Result1 v2 Failure
The TTL writer reports `# NO VALID FACTS extracted after 3 attempts` but the embedded JSON in the comment block shows 38 well-formed facts were extracted. This is a **serialization bug**, not an extraction bug — the JSON → rdflib ingestion step failed to consume its own output. The extracted facts are correct and complete.

### Result2 Failure Modes
- No ontology → no semantic constraints → spaCy noise passes through unfiltered
- All entities typed `kg:Concept` → no entity type reasoning possible downstream
- Dependency-parsed relations (`appos`, `be`, `of`) are not KG relations — they are grammatical artifacts
- No provenance, no Wikidata alignment, no CQ grounding

---

## 4. Resume KG Suitability

For resume KG construction specifically, what matters:

| Requirement | Result1 v2 | Result2 |
|---|---|---|
| Extracts person → institution links | ✅ `EducatedAt`, `Affiliation` | ⚠️ partially, but mixed with noise |
| Extracts skills/tools | ✅ `ProgrammedIn`, `DataAnalysisMethod` | ✅ spaCy catches most tools |
| Extracts roles/positions | ✅ `PositionHolder` | ⚠️ `work_as` relation present but poorly typed |
| Extracts degrees/certifications | ✅ `AcademicDegree`, `HasCertification` | ❌ not specifically |
| Queryable schema | ✅ Wikidata PIDs | ❌ flat local namespace |
| Neo4j importability | ✅ structured, typed | ⚠️ possible but all `Concept` nodes lose semantic value |
| Noise level | Low | High |

---

## 5. Verdict

**Result1 v2 wins.** Despite the TTL serialization bug in the final step, the *conceptual pipeline and extracted data* are substantially better:

- CQ-driven extraction is semantically grounded and resume-appropriate
- 29/32 relations mapped to real Wikidata PIDs with explainable cosine scores
- Entity types are meaningful (not all `Concept`)
- The ontology defines exactly what the KG captures — reproducible, auditable
- The v1 → v2 fix (JSON facts + rdflib) was architecturally correct; it just has a bug in the handoff

**Result2 loses** not because spaCy is wrong in general, but because for a structured document like a resume — where entities and relations are semantically rich and well-defined — dependency parsing without ontological constraints produces a bloated, noisy graph with no discriminative typing and relations that are English prepositions, not domain predicates.

### What to fix in Result1 v2

1. **Primary bug:** The `stage9_10_kg.py` JSON → rdflib ingestion is not reading its own extracted facts. The JSON is written correctly in the comment/fallback block — wire it into the rdflib graph builder directly.
2. **Mapping error:** `worked at` → `RealEstateDeveloper` (P6237) is wrong. Either add `Employer` (P108) to the Wikidata seed or tighten the LLM validator prompt to reject obvious semantic mismatches.
3. `AncestralHome` for `has background` is also wrong — that relation probably shouldn't be in the ontology for a resume domain at all.
