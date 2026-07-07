"""
canonicalize.py — Relation canonicalization (EDC) + Entity resolution

─── Relation canonicalization ───────────────────────────────────────────────
Implements EDC (Extract, Define, Canonicalise) from EMNLP 2024:
  https://arxiv.org/abs/2401.09280

Key departure from naive embedding matching:
  - A natural-language *definition* is generated for each extracted relation
    (the "Define" step).  The definition is embedded, NOT the raw surface
    label.  Two relations with different labels but identical meanings produce
    nearly identical definition embeddings, enabling correct merging.
  - An LLM verification step gates every proposed merge.  Embedding distance
    alone never triggers a merge.
  - Merge confidence scores are stored on every accepted/rejected result.

The canonical store persists between documents so relations extracted from
document N are compared against all relations canonicalized from documents
1…N-1.  This enables progressive schema consolidation across a corpus.

─── Entity resolution ────────────────────────────────────────────────────────
3-tier strategy suited for large-scale resume KGs:

  Tier 1 – Gazetteer (O(1)):  curated JSON lookup tables for companies,
            universities, certifications, skills, and job titles.  Handles
            ~80 % of resume entity mentions with high precision.

  Tier 2 – Embedding similarity: embed mention + context, search canonical
            entity store.  Catches variants and abbreviations not in the
            gazetteer.

  Tier 3 – LLM normalization: only reached for genuinely ambiguous or novel
            entities.  Expensive; kept to ~5 % of mentions.

─── Plugin interface ─────────────────────────────────────────────────────────
Both canonicalization and entity resolution are exposed through abstract base
classes so alternative backends (e.g. OPIEC, CESI replacement, fine-tuned
models) can be swapped without changing call sites.

─── Fix log ──────────────────────────────────────────────────────────────────
- resolve_kg_entities(): the per-node default entity_type used to be
  hardcoded "company" for BOTH the subject and the object whenever the
  predicate wasn't in PROPERTY_ENTITY_TYPE_MAP. The subject is always the
  resume owner (a person), never a company — that single fallback was why
  every entity in the KG came out typed "company". Subject now defaults to
  "person"; object defaults to "unknown" instead of silently guessing
  "company".
- ResumeEntityResolver._tier3(): confidence used to be a hardcoded 0.70
  constant regardless of what the LLM actually returned. The prompt now
  asks for an explicit "<canonical form> | <score>" response (same pattern
  already used by EDCBackend._llm_verify), parsed into a real confidence.
- ResumeEntityResolver._tier3(): added a guard against the model leaking
  its own reasoning into the canonical form (e.g. "Machine Learning\n\nNote:
  The canonical form is..."). Only the first line is ever used, and
  suspiciously long / multi-line / "note"-prefixed output falls back to the
  original mention with low confidence instead of polluting the KG.
"""
from __future__ import annotations

import json
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from stages.llm import call_llm_answer
from db import canon as canon_db, gazetteers as gaz_db
from config import (
    EMBED_MODEL,
    CANON_TOP_K,
    PROPERTY_ENTITY_TYPE_MAP,
    LLM_MODEL,
    LLM_TOKENS_CLASSIFY,
    LLM_TOKENS_DEFINE,
    LLM_TOKENS_RETRY,
)

# The canon store and gazetteers now live in Postgres, so the backends need a
# session factory. The pipeline calls configure() once at startup; the lazy
# singletons below read it.
_session_factory = None


def configure(session_factory) -> None:
    """Point the EDC + entity-resolution backends at the shared database."""
    global _session_factory
    _session_factory = session_factory


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RelationCanonResult:
    original_property:    str
    original_description: str
    definition:           str          # EDC "Define" step output
    canonical_label:      Optional[str]
    canonical_description: Optional[str]
    confidence:           float        # 0.0 if no match
    was_merged:           bool
    rejected_candidates:  list[dict] = field(default_factory=list)


@dataclass
class EntityResolutionResult:
    original_mention: str
    canonical_form:   str
    entity_type:      str
    resolution_tier:  str              # "gazetteer" | "embedding" | "llm" | "unresolved"
    confidence:       float
    wikidata_qid:     Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Abstract backends
# ══════════════════════════════════════════════════════════════════════════════

class RelationCanonicalizationBackend(ABC):
    """Swap EDC for any future canonicalization strategy without touching call sites."""

    @abstractmethod
    def canonicalize(
        self, relation: dict, context_cqs: list[str]
    ) -> RelationCanonResult: ...

    @abstractmethod
    def register_new_property(
        self, label: str, definition: str, turtle: str, source_doc: str
    ) -> None: ...

    @abstractmethod
    def flush(self) -> None:
        """Persist in-memory state (e.g. new canonical store entries)."""
        ...


class EntityResolutionBackend(ABC):
    """Swap 3-tier gazetteer for any future resolution strategy."""

    @abstractmethod
    def resolve(
        self, mention: str, entity_type: str, context: str = ""
    ) -> EntityResolutionResult: ...


# ══════════════════════════════════════════════════════════════════════════════
# EDC backend — relation canonicalization
# ══════════════════════════════════════════════════════════════════════════════

_DEFINE_PROMPT = """\
You are building a knowledge graph schema for a large resume corpus.

Given a relation label and its initial description extracted from text, write a
single precise canonical definition of this relation.

Requirements for the definition:
- Capture the semantic meaning, not just rephrase the surface label.
- Be phrasing-independent: two relations with the same meaning should produce
  nearly identical definitions even if their labels differ.
- Implicitly describe the expected domain (subject type) and range (object type).
- Be one sentence, maximum 40 words.

Relation label: {label}
Initial description: {description}
Context questions: {cqs}

Definition:"""

_VERIFY_PROMPT = """\
Decide whether two knowledge graph relations express the same semantic
relationship and should be canonicalized to a single property.

Relation A:
  Label:      {label_a}
  Definition: {definition_a}

Relation B:
  Label:      {label_b}
  Definition: {definition_b}

Rules:
- Answer "yes" if they are semantically equivalent, or if one is the inverse
  of the other (e.g. "worksAt" / "employs").
- Answer "no" if they have different domains, ranges, or meanings.
- Append a confidence integer 0–100 after your answer.

Format exactly: "yes 87"  or  "no 12"

Answer:"""


class EDCBackend(RelationCanonicalizationBackend):
    """
    EDC: Extract → Define → Canonicalise (EMNLP 2024).

    The canon store lives in Postgres now (db.canon): search_similar() does the
    cosine nearest-neighbour lookup via pgvector's HNSW index instead of an
    in-memory numpy scan over entries.json/embeddings.npy, and add_entry()
    persists a new property immediately (no flush step).
    """

    def __init__(
        self,
        session_factory,
        embed_model_name: str,
        top_k: int,
    ) -> None:
        self._session_factory = session_factory
        self._model_name      = embed_model_name
        self._top_k           = top_k
        self._st_model        = None          # lazy: SentenceTransformer

    # ── Embedding model (lazy) ──────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self._model_name)

    def _embed(self, text: str) -> np.ndarray:
        self._load_model()
        vec = self._st_model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )
        return vec[0]

    def flush(self) -> None:
        # No-op: the canon store is written straight to Postgres by add_entry().
        return

    # ── Top-k retrieval ─────────────────────────────────────────────────────

    def _top_k_candidates(
        self, vec: np.ndarray
    ) -> list[tuple[dict, float]]:
        """Return (entry, cosine_score) pairs for the k nearest canon entries."""
        with self._session_factory() as session:
            entries = canon_db.search_similar(session, vec.tolist(), self._top_k)
        return [(entry, entry["cos_score"]) for entry in entries]

    # ── LLM steps ───────────────────────────────────────────────────────────

    def _generate_definition(
        self, label: str, description: str, cqs: list
    ) -> str:
        # cqs may be the current {"subject": ..., "question": ...} dict
        # shape from stage1_cq_gen.py, or a legacy plain string — never
        # interpolate the item directly (that silently stringifies a dict
        # into the "Define" prompt, e.g. "- {'subject': 'person', ...}").
        # Same fix pattern as stage3_relation_extract.py's _cq_text().
        questions = [
            str(cq.get("question", "")).strip() if isinstance(cq, dict)
            else str(cq).strip()
            for cq in cqs
        ]
        questions = [q for q in questions if q]
        cq_block = "\n".join(f"- {q}" for q in questions[:5]) if questions else "(none)"
        prompt   = _DEFINE_PROMPT.format(
            label=label, description=description, cqs=cq_block
        )
        raw, truncated = call_llm_answer(
            prompt, LLM_TOKENS_DEFINE, retry_budget=LLM_TOKENS_RETRY
        )
        if truncated:
            print(f"  [EDC] ⚠ definition truncated for '{label}' (reasoning exceeded "
                  f"budget) — using best-effort text", flush=True)
        # Strip any "Definition:" echo the model might prepend
        definition = re.sub(r"^definition[:\s]*", "", raw.strip(), flags=re.IGNORECASE)
        return definition.strip()

    def _llm_verify(
        self,
        label_a: str, definition_a: str,
        label_b: str, definition_b: str,
    ) -> tuple[bool, float]:
        """
        Returns (accepted, confidence 0-1).
        Parses the structured "yes 87" / "no 12" format.
        Never treats a parse failure as an acceptance.
        """
        prompt = _VERIFY_PROMPT.format(
            label_a=label_a, definition_a=definition_a,
            label_b=label_b, definition_b=definition_b,
        )
        raw_text, truncated = call_llm_answer(
            prompt, LLM_TOKENS_CLASSIFY, retry_budget=LLM_TOKENS_RETRY
        )
        if truncated:
            # Indeterminate: the model was still reasoning at the cap. Don't
            # merge, but flag it rather than silently reading "" as "no".
            print("  [EDC] ⚠ verify truncated (reasoning exceeded budget) — "
                  "INDETERMINATE, no merge", flush=True)
            return False, 0.0
        raw    = raw_text.strip().lower()
        m      = re.match(r"^(yes|no)\s*(\d{1,3})?", raw)
        if not m:
            print(f"  [EDC] ⚠ unparseable verify response: {repr(raw)} — treating as no", flush=True)
            return False, 0.0

        accepted   = m.group(1) == "yes"
        raw_score  = int(m.group(2)) if m.group(2) else (70 if accepted else 30)
        confidence = min(100, max(0, raw_score)) / 100.0
        return accepted, confidence

    # ── Public API ──────────────────────────────────────────────────────────

    def canonicalize(
        self, relation: dict, context_cqs: list[str]
    ) -> RelationCanonResult:
        label       = relation["property"]
        description = relation.get("description", "")

        # EDC Step 1 — Define
        t0         = time.time()
        definition = self._generate_definition(label, description, context_cqs)
        print(
            f"  [EDC] '{label}' → definition in {time.time()-t0:.1f}s: "
            f"{definition[:80]}{'…' if len(definition)>80 else ''}",
            flush=True,
        )

        # EDC Step 2 — Embed definition
        vec = self._embed(definition)

        # EDC Step 3 — Retrieve top-k
        candidates     = self._top_k_candidates(vec)
        rejected: list[dict] = []

        # EDC Step 4 — LLM-verify each candidate in rank order
        for entry, cos_score in candidates:
            accepted, confidence = self._llm_verify(
                label, definition,
                entry["label"], entry["definition"],
            )
            print(
                f"  [EDC]   vs '{entry['label']}' (cos={cos_score:.3f}) → "
                f"{'✓ merged' if accepted else '✗ rejected'} (conf={confidence:.2f})",
                flush=True,
            )
            if accepted:
                return RelationCanonResult(
                    original_property=label,
                    original_description=description,
                    definition=definition,
                    canonical_label=entry["label"],
                    canonical_description=entry["definition"],
                    confidence=confidence,
                    was_merged=True,
                    rejected_candidates=rejected,
                )
            rejected.append({
                "label": entry["label"],
                "cos_score": cos_score,
                "confidence": confidence,
            })

        # No match found
        return RelationCanonResult(
            original_property=label,
            original_description=description,
            definition=definition,
            canonical_label=None,
            canonical_description=None,
            confidence=0.0,
            was_merged=False,
            rejected_candidates=rejected,
        )

    def register_new_property(
        self, label: str, definition: str, turtle: str, source_doc: str
    ) -> None:
        """
        Add a genuinely new property to the canon store so future documents
        can merge against it.  Call this after Stage 7/8 generates Turtle for
        an unmatched property. add_entry() dedups by canonical label and
        commits immediately. source_doc is the résumé UUID (or None).
        """
        vec = self._embed(definition)
        with self._session_factory() as session:
            canon_db.add_entry(
                session,
                label=label,
                definition=definition,
                turtle=turtle,
                embedding=vec.tolist(),
                source_doc=source_doc,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Resume entity resolver — 3-tier
# ══════════════════════════════════════════════════════════════════════════════

_ENTITY_LLM_PROMPT = """\
You are normalizing entity mentions for a resume knowledge graph.

Entity mention: "{mention}"
Entity type: {entity_type}
Context: {context}

Normalize this mention to its most widely recognized canonical form.
Examples of canonical forms: "Google", "Massachusetts Institute of Technology",
"AWS Certified Solutions Architect - Associate", "Python", "Software Engineer".

If the mention is already canonical or you are uncertain, return it unchanged.

Return ONLY the canonical form followed by a confidence score from 0-100,
nothing else — no explanation, no reasoning, no extra lines.
Format exactly: <canonical form> | <score>

Answer:"""


# Entity types where fuzzy canonicalization (embedding + LLM) is safe and
# valuable — collapsing "Google Inc." → "Google", "MIT" → "Massachusetts
# Institute of Technology". For every OTHER type (skill, technology, language,
# person, organization, activity, reference, unknown, …) fuzzy matching does
# more harm than good: bge-small-en snaps distinct skills onto the nearest
# "famous" gazetteer neighbour ("OpenAI Gym"→"OpenAI API", "AWS (EC2, ECR,
# IAM)"→"AWS Lambda"), and the Tier 3 LLM rewrites/hallucinates mentions with no
# gazetteer to anchor them (e.g. a reference's name). Those types get Tier 1
# (exact alias) only, else verbatim. This is an allowlist, not a denylist, so any
# new entity type is safe-by-default.
CANONICALIZE_TYPES = {"company", "university", "certification", "job_title"}


class ResumeEntityResolver(EntityResolutionBackend):
    """
    Tier 1 — Gazetteer lookup (O(1), curated JSON files)
    Tier 2 — Embedding similarity against canonical entity store (variants)
    Tier 3 — LLM normalization (novel/ambiguous entities)

    Tiers 2 and 3 run only for CANONICALIZE_TYPES; all other types resolve via
    Tier 1 (exact alias) or fall through verbatim, preserving the surface form.
    """

    def __init__(self, session_factory, embed_model_name: str) -> None:
        self._session_factory = session_factory
        self._model_name      = embed_model_name
        self._st_model        = None  # lazy

    # ── Gazetteer lookups (Postgres — db.gazetteers) ─────────────────────────

    def _get_qid(self, entity_type: str, canonical: str) -> Optional[str]:
        with self._session_factory() as session:
            return gaz_db.get_qid(session, entity_type, canonical)

    # ── Tier 1: Gazetteer ───────────────────────────────────────────────────

    def _tier1(
        self, mention: str, entity_type: str
    ) -> Optional[EntityResolutionResult]:
        with self._session_factory() as session:
            canonical = gaz_db.lookup(session, entity_type.lower(), mention)
            if canonical is None:
                return None
            qid = gaz_db.get_qid(session, entity_type.lower(), canonical)
        return EntityResolutionResult(
            original_mention=mention,
            canonical_form=canonical,
            entity_type=entity_type,
            resolution_tier="gazetteer",
            confidence=1.0,
            wikidata_qid=qid,
        )

    # ── Tier 2: Embedding ───────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self._model_name)

    def _tier2(
        self, mention: str, entity_type: str, context: str
    ) -> Optional[EntityResolutionResult]:
        """
        Check if the mention closely matches any gazetteer canonical value
        by embedding similarity.  Avoids a separate entity store; the
        canonical values in the gazetteer are the embedding targets.
        """
        self._load_model()
        with self._session_factory() as session:
            canonical_values = gaz_db.canonical_values(session, entity_type.lower())
        if not canonical_values:
            return None

        query_text = f"{mention} ({context})" if context else mention
        query_vec  = self._st_model.encode(
            [query_text], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        canon_vecs = self._st_model.encode(
            canonical_values, normalize_embeddings=True, convert_to_numpy=True
        )
        scores = canon_vecs @ query_vec
        best_i = int(np.argmax(scores))
        best_score = float(scores[best_i])

        if best_score >= 0.94:
            canonical = canonical_values[best_i]
            return EntityResolutionResult(
                original_mention=mention,
                canonical_form=canonical,
                entity_type=entity_type,
                resolution_tier="embedding",
                confidence=best_score,
                wikidata_qid=self._get_qid(entity_type, canonical),
            )
        return None

    # ── Tier 3: LLM ────────────────────────────────────────────────────────

    def _tier3(
        self, mention: str, entity_type: str, context: str
    ) -> EntityResolutionResult:
        prompt = _ENTITY_LLM_PROMPT.format(
            mention=mention, entity_type=entity_type, context=context or "N/A"
        )
        raw_text, truncated = call_llm_answer(
            prompt, LLM_TOKENS_CLASSIFY, retry_budget=LLM_TOKENS_RETRY
        )
        if truncated:
            # Reasoning ran past the cap — keep the mention verbatim rather than
            # trusting a cut-off canonical form. Flagged, not silently accepted.
            print(f"  [entity] ⚠ Tier-3 truncated for {mention!r} — keeping mention "
                  f"verbatim (low confidence)", flush=True)
            return EntityResolutionResult(
                original_mention=mention,
                canonical_form=mention,
                entity_type=entity_type,
                resolution_tier="llm",
                confidence=0.3,
                wikidata_qid=None,
            )
        raw = raw_text.strip()

        # Only the first line is ever trusted — anything past it is the model
        # ignoring "nothing else" and adding reasoning/explanation, which must
        # never leak into the KG.
        first_line = raw.splitlines()[0].strip() if raw else ""

        if "|" in first_line:
            canon_part, _, score_part = first_line.partition("|")
            canonical = canon_part.strip().strip('"').strip("'")
            digits = re.sub(r"\D", "", score_part)
            try:
                confidence = min(100, max(0, int(digits))) / 100.0 if digits else 0.5
            except ValueError:
                confidence = 0.5
        else:
            canonical, confidence = first_line.strip('"').strip("'"), 0.5

        if not canonical:
            canonical = mention

        # Guard against leaked reasoning: canonical forms are short, single
        # values. If this looks like an explanation instead, discard it and
        # fall back to the original mention with low confidence rather than
        # writing garbage into the graph.
        looks_like_leak = (
            len(canonical) > 80
            or "\n" in raw.strip().split("|")[0]
            or canonical.lower().startswith("note")
            or canonical.lower().startswith("the canonical")
        )
        if looks_like_leak:
            canonical, confidence = mention, 0.3

        return EntityResolutionResult(
            original_mention=mention,
            canonical_form=canonical,
            entity_type=entity_type,
            resolution_tier="llm",
            confidence=confidence,
            wikidata_qid=None,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def _verbatim(self, mention: str, entity_type: str) -> EntityResolutionResult:
        """Keep the mention exactly as written — no fuzzy match, no LLM rewrite."""
        return EntityResolutionResult(
            original_mention=mention,
            canonical_form=mention,
            entity_type=entity_type,
            resolution_tier="verbatim",
            confidence=1.0,
            wikidata_qid=None,
        )

    def resolve(
        self, mention: str, entity_type: str, context: str = ""
    ) -> EntityResolutionResult:
        # Tier 1 (exact gazetteer alias) applies to every type.
        result = self._tier1(mention, entity_type)
        if result:
            return result

        # Fuzzy Tiers 2/3 only for types where canonicalization is safe; every
        # other type keeps its surface form (see CANONICALIZE_TYPES).
        if entity_type.lower() not in CANONICALIZE_TYPES:
            return self._verbatim(mention, entity_type)

        result = self._tier2(mention, entity_type, context)
        if result:
            return result

        return self._tier3(mention, entity_type, context)


# ══════════════════════════════════════════════════════════════════════════════
# KG entity post-processor
# ══════════════════════════════════════════════════════════════════════════════

def _slugify(text: str) -> str:
    """'Google Inc.' → 'Google_Inc'"""
    return re.sub(r"[^\w]", "_", text).strip("_")


def _unslugify(slug: str) -> str:
    """'Google_Inc' → 'Google Inc'"""
    return slug.replace("_", " ").strip()


def resolve_kg_entities(
    turtle_str: str,
    resolver: EntityResolutionBackend,
) -> tuple[str, dict[str, EntityResolutionResult]]:
    """
    Parse the generated Turtle, resolve all wd: entity URIs to canonical
    forms using the resolver, rewrite the graph, and serialize back.

    Returns (rewritten_turtle, {original_uri: EntityResolutionResult}).
    """
    import rdflib
    from rdflib import URIRef, Graph
    from rdflib.namespace import RDF

    WD  = "http://www.wikidata.org/entity/"
    WDT = "http://www.wikidata.org/prop/direct/"

    g = Graph()
    try:
        g.parse(data=turtle_str, format="turtle")
    except Exception:
        # Can't resolve what we can't parse; caller handles parse errors
        return turtle_str, {}

    # Build predicate → entity-type map from config
    prop_type_map = {
        WDT + k: v for k, v in PROPERTY_ENTITY_TYPE_MAP.items()
    }

    # Collect all wd: entity URIs (subjects + objects) and infer their type.
    #
    # FIX: the subject of a resume KG triple is, by construction, always the
    # person the resume is about — it must never silently default to
    # "company" just because the predicate wasn't in PROPERTY_ENTITY_TYPE_MAP.
    # The object's type is genuinely unknown without a predicate match, so
    # its fallback is "unknown" rather than a guessed type — better to flag
    # it for review than mislabel it.
    uri_type: dict[str, str] = {}
    for s, p, o in g:
        p_str = str(p)
        for node, default_type in [(s, "person"), (o, "unknown")]:
            uri = str(node)
            if uri.startswith(WD) and not uri.startswith(WDT):
                local = uri[len(WD):]
                if "_" in local or local[0].isupper():
                    inferred = prop_type_map.get(p_str, default_type)
                    uri_type.setdefault(uri, inferred)

    # Resolve each unique entity
    resolution_map: dict[str, EntityResolutionResult] = {}
    uri_rewrite:    dict[str, str] = {}

    for uri, etype in uri_type.items():
        local   = uri[len(WD):]
        mention = _unslugify(local)
        result  = resolver.resolve(mention, etype)
        resolution_map[uri] = result

        if result.resolution_tier != "unresolved":
            if result.wikidata_qid:
                new_uri = WD + result.wikidata_qid
            else:
                new_uri = WD + _slugify(result.canonical_form)
            if new_uri != uri:
                uri_rewrite[uri] = new_uri

    if not uri_rewrite:
        return turtle_str, resolution_map

    # Rewrite the graph with canonical URIs
    g2 = Graph()
    # Copy namespace bindings
    for prefix, ns in g.namespaces():
        g2.bind(prefix, ns)

    def _remap(node):
        if isinstance(node, URIRef):
            return URIRef(uri_rewrite.get(str(node), str(node)))
        return node

    for s, p, o in g:
        g2.add((_remap(s), _remap(p), _remap(o)))

    rewritten = g2.serialize(format="turtle")
    return rewritten, resolution_map


# ══════════════════════════════════════════════════════════════════════════════
# Module-level singletons (lazy-initialised)
# ══════════════════════════════════════════════════════════════════════════════

_edc_backend:     Optional[EDCBackend]           = None
_entity_resolver: Optional[ResumeEntityResolver] = None


def get_edc_backend() -> EDCBackend:
    global _edc_backend
    if _edc_backend is None:
        if _session_factory is None:
            raise RuntimeError(
                "canonicalize.configure(session_factory) must be called before "
                "using the EDC backend (the canon store lives in Postgres now)."
            )
        _edc_backend = EDCBackend(
            session_factory=_session_factory,
            embed_model_name=EMBED_MODEL,
            top_k=CANON_TOP_K,
        )
    return _edc_backend


def get_entity_resolver() -> ResumeEntityResolver:
    global _entity_resolver
    if _entity_resolver is None:
        if _session_factory is None:
            raise RuntimeError(
                "canonicalize.configure(session_factory) must be called before "
                "using the entity resolver (gazetteers live in Postgres now)."
            )
        _entity_resolver = ResumeEntityResolver(
            session_factory=_session_factory,
            embed_model_name=EMBED_MODEL,
        )
    return _entity_resolver


# ══════════════════════════════════════════════════════════════════════════════
# Public convenience API (used by pipeline stages)
# ══════════════════════════════════════════════════════════════════════════════

def canonicalize_relation(
    relation: dict, context_cqs: list[str]
) -> RelationCanonResult:
    return get_edc_backend().canonicalize(relation, context_cqs)


def resolve_entity(
    mention: str, entity_type: str, context: str = ""
) -> EntityResolutionResult:
    return get_entity_resolver().resolve(mention, entity_type, context)


def flush_canon_store() -> None:
    get_edc_backend().flush()