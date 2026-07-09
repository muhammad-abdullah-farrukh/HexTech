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
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from stages.llm import call_llm
from stages import verify_gate
from db import canon as canon_db, gazetteers as gaz_db
from config import (
    EMBED_MODEL,
    EMBED_DEVICE,
    CANON_TOP_K,
    PROPERTY_ENTITY_TYPE_MAP,
    LLM_MODEL,
    OUTPUTS_DIR,
)

# The canon store and gazetteers now live in Postgres, so the backends need a
# session factory. The pipeline calls configure() once at startup; the lazy
# singletons below read it.
_session_factory = None


def configure(session_factory) -> None:
    """Point the EDC + entity-resolution backends at the shared database."""
    global _session_factory
    _session_factory = session_factory


def _edc_log(doc_name: str, event: dict) -> None:
    """Append one JSON line to outputs/logs/edc_{doc_name}.jsonl — every
    Define-step LLM call, every verify decision (whichever tier decided it —
    containment, cosine, or an actual LLM call), and every final per-relation
    outcome. Lets the whole EDC phase be watched live (`tail -f`) without
    touching Postgres, same pattern as pipeline.py's plain-text stage log."""
    log_path = OUTPUTS_DIR / "logs" / f"edc_{doc_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _record_verdict(
    session_factory, label_a, definition_a, label_b, definition_b,
    accepted, confidence, source, doc_name,
) -> None:
    """Persist every verify decision — gate or LLM — to verify_verdicts.
    This is what lets recompute_thresholds.py's evidence grow from ordinary
    pipeline runs, not just from a one-off backfill."""
    from db.models import VerifyVerdict

    with session_factory() as session:
        session.add(VerifyVerdict(
            label_a=label_a, definition_a=definition_a,
            label_b=label_b, definition_b=definition_b,
            accepted=accepted, confidence=confidence,
            source=source, resume_id=doc_name if _looks_like_uuid(doc_name) else None,
        ))
        session.commit()


def _looks_like_uuid(s: str) -> bool:
    import re as _re
    return bool(_re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", s
    ))


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
        # EDC labels are canonicalized concurrently (pipeline fans them out
        # across endpoints); the one shared SentenceTransformer is guarded so
        # concurrent .encode() calls don't race. Embeds are ms, so serializing
        # them costs nothing next to the LLM calls that dominate.
        self._embed_lock      = threading.Lock()

    # ── Embedding model (lazy) ──────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self._model_name, device=EMBED_DEVICE)

    def _embed(self, text: str) -> np.ndarray:
        self._load_model()
        with self._embed_lock:
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
        self, label: str, description: str, cqs: list, endpoint: dict | None = None
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
        # max_tokens=800 (was 60 — too tight once reasoning is counted against
        # the budget; measured completion_tokens=319 for a real relation on
        # this prompt shape, ~2.5x margin).
        raw = call_llm(prompt, max_tokens=800, endpoint=endpoint)
        # Strip any "Definition:" echo the model might prepend
        definition = re.sub(r"^definition[:\s]*", "", raw.strip(), flags=re.IGNORECASE)
        return definition.strip()

    def _llm_verify(
        self,
        label_a: str, definition_a: str,
        label_b: str, definition_b: str,
        endpoint: dict | None = None,
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
        # max_tokens=1000 (was 10 — reasoning consumed the whole budget
        # before the yes/no answer; measured completion_tokens=571 for a
        # real relation pair, same shape/value as validate_match()).
        raw    = call_llm(prompt, max_tokens=1000, endpoint=endpoint).strip().lower()
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
        self, relation: dict, context_cqs: list[str], doc_name: str = "unknown",
        endpoint: dict | None = None,
    ) -> RelationCanonResult:
        label       = relation["property"]
        description = relation.get("description", "")

        # EDC Step 1 — Define
        t0         = time.time()
        definition = self._generate_definition(label, description, context_cqs, endpoint=endpoint)
        dt = time.time() - t0
        print(
            f"  [EDC] '{label}' → definition in {dt:.1f}s: "
            f"{definition[:80]}{'…' if len(definition)>80 else ''}",
            flush=True,
        )
        _edc_log(doc_name, {
            "event": "define", "label": label, "description": description,
            "definition": definition, "seconds": round(dt, 2),
        })

        # EDC Step 2 — Embed definition
        vec = self._embed(definition)

        # EDC Step 3 — Retrieve top-k
        candidates     = self._top_k_candidates(vec)
        rejected: list[dict] = []

        # EDC Step 4 — verify gate first (deterministic, ~ms); only pairs it
        # can't decide (ESCALATE) fall through to the reasoning LLM.
        for entry, cos_score in candidates:
            method, accepted, _ = verify_gate.decide(
                label, entry["label"], cos_score, self._session_factory
            )
            llm_seconds = 0.0
            if method == verify_gate.ESCALATE:
                t1 = time.time()
                accepted, confidence = self._llm_verify(
                    label, definition, entry["label"], entry["definition"],
                    endpoint=endpoint,
                )
                llm_seconds = time.time() - t1
                method = "llm"
            else:
                # Deterministic tiers don't produce a graded confidence the
                # way the LLM does — record the decision as fully confident,
                # since both tiers are precision-gated to ~0.95+ before use.
                confidence = 1.0 if accepted else 0.0

            print(
                f"  [EDC]   vs '{entry['label']}' (cos={cos_score:.3f}, method={method}) → "
                f"{'✓ merged' if accepted else '✗ rejected'} (conf={confidence:.2f})",
                flush=True,
            )
            _edc_log(doc_name, {
                "event": "verify", "label_a": label, "label_b": entry["label"],
                "method": method, "accepted": accepted, "confidence": round(confidence, 2),
                "cos_score": round(cos_score, 3), "seconds": round(llm_seconds, 2),
            })
            _record_verdict(
                self._session_factory, label, definition,
                entry["label"], entry["definition"], accepted, confidence,
                source="deepseek" if method == "llm" else "gate", doc_name=doc_name,
            )

            if accepted:
                _edc_log(doc_name, {"event": "outcome", "label": label, "merged_with": entry["label"]})
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
        _edc_log(doc_name, {"event": "outcome", "label": label, "merged_with": None})
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

Only fix spelling, punctuation, capitalization, or formatting of THIS SAME
entity — e.g. expand a known acronym, or standardize how its name is written.
Never substitute a different, unrelated entity, even if it seems more famous
or more likely to appear on a resume — a wrong answer is worse than no answer.
If you do not recognize this exact entity, or are at all uncertain, return
the mention completely unchanged.

Examples of correct normalization: "MIT" -> "Massachusetts Institute of
Technology", "google inc" -> "Google", "aws saa" -> "AWS Certified Solutions
Architect - Associate".

Return ONLY the canonical form followed by a confidence score from 0-100,
nothing else — no explanation, no reasoning, no extra lines.
Format exactly: <canonical form> | <score>

Answer:"""

# Generic words carry no distinguishing information for the lexical-overlap
# guard below (every university has "University" in it) — stripped before
# checking whether an LLM "normalization" actually still refers to the same
# entity. _STOPWORDS is the narrower set also excluded when checking for an
# acronym-expansion (those still count toward an acronym's initials).
_STOPWORDS = {"of", "the", "and", "at", "in", "for"}
_GENERIC_ENTITY_WORDS = _STOPWORDS | {
    "university", "college", "institute", "institution", "school",
    "technology", "technologies", "science", "sciences",
    "international", "national", "state",
}


def _content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _GENERIC_ENTITY_WORDS}


def _is_acronym_expansion(mention: str, canonical: str) -> bool:
    """True if `mention` looks like an acronym whose initials match
    `canonical` (e.g. "MIT" -> "Massachusetts Institute of Technology") — the
    one legitimate case where a match has no lexical token overlap with the
    mention but is still a correct, same-entity normalization."""
    m = mention.strip()
    if not re.fullmatch(r"[A-Za-z0-9]{2,8}", m) or m.lower() == m:
        return False
    words = [w for w in re.findall(r"[A-Za-z0-9]+", canonical) if w.lower() not in _STOPWORDS]
    initials = "".join(w[0] for w in words).upper()
    return initials == m.upper()


# Mirrors stage6_match_validate.py's VALIDATE_PROMPT / ACCEPT_CONFIDENCE_
# THRESHOLD design: embedding similarity retrieves a candidate, but an LLM
# must confirm it's actually the same real-world entity before it's trusted
# — cosine similarity alone was observed accepting "Air University" ->
# "Brown University" at 0.88 (shared "University", nothing else), the exact
# same class of false-positive stage6 already had to guard against for
# relationship/property matching.
_ENTITY_VERIFY_PROMPT = """\
You are checking whether a candidate entity is the CORRECT match for a
mention extracted from a resume — not merely whether the two sound similar.

Mention 1 is the entity as it was actually written in the resume.
Entity 2 is a candidate match retrieved by embedding similarity, which can
surface plausible-sounding but wrong matches (e.g. two different
universities that share only the generic word "University").

Answer "yes" ONLY if Entity 2 is genuinely the same real-world entity as
Mention 1 (a typo, abbreviation, or formatting variant of the same thing) —
not just a similar category of thing.

Answer "no" if Entity 2 is a different, unrelated entity, even if the words
overlap. For example:
- "Air University" is NOT "Brown University"
- "Pakistan Intl. School" is NOT "Birla Institute of Technology and Science"
Shared generic words like "University", "Institute", or "College" are not
evidence of a correct match by themselves.

Append a confidence integer 0-100 after your answer.
Format exactly: "yes 87" or "no 12"

Mention 1: {mention}
Entity 2: {candidate}

Answer:"""

_ENTITY_ACCEPT_CONFIDENCE_THRESHOLD = 0.65


def _parse_yes_no_confidence(raw_answer: str) -> tuple[bool, float]:
    """Parse "yes 87" / "no 12" into (said_yes, confidence 0-1). A parse
    failure is never treated as an acceptance."""
    m = re.match(r"^(yes|no)\s*(\d{1,3})?", raw_answer.strip().lower())
    if not m:
        return False, 0.0
    said_yes = m.group(1) == "yes"
    raw_score = int(m.group(2)) if m.group(2) else (70 if said_yes else 30)
    return said_yes, min(100, max(0, raw_score)) / 100.0


def _llm_confirms_entity_match(mention: str, candidate: str) -> bool:
    """LLM verification gate for a Tier 2 embedding candidate — same
    retrieve-then-verify shape as stage6_match_validate's validate_match(),
    applied only to the single top candidate Tier 2 already retrieved (not
    re-run per entity; only for matches that already cleared the cosine
    threshold)."""
    prompt = _ENTITY_VERIFY_PROMPT.format(mention=mention, candidate=candidate)
    # 600, not the ~20 tokens the answer itself needs — matches Tier 3's
    # max_tokens (see its comment): this reasoning model burns an
    # unpredictable amount of budget on <think> reasoning before the answer.
    raw_answer = call_llm(prompt, max_tokens=1000)
    said_yes, confidence = _parse_yes_no_confidence(raw_answer)
    return said_yes and confidence >= _ENTITY_ACCEPT_CONFIDENCE_THRESHOLD


# Types that must never be canonicalized: projects are résumé-unique
# artifacts, person names must never be gazetteer-matched, and "unknown"
# gives no basis for a cross-type match. Observed failures without this gate:
# project "Predictive Optimization Models" → skill "Machine Learning"
# (Tier 2, cosine ≥0.88 + LLM confirm), project "Reinforcement Learning
# Suite" → skill "Deep Learning" (Tier 3, shared token "learning").
_NEVER_RESOLVE_TYPES = {"project", "person", "unknown"}

# The any-type gazetteer fallbacks (lookup_any_type / canonical_values_any_type
# / get_qid_any_type) may only fire for tech-ish types where cross-type
# aliasing is legitimate — e.g. "technology" has no scoped gazetteer rows and
# must be allowed to match the skills list (the scikit-learn fix). Letting
# company/university/certification/job_title/person fall back was how
# unrelated entities got matched.
_ANY_TYPE_FALLBACK_TYPES = {"skill", "technology", "tool", "library", "framework"}


class ResumeEntityResolver(EntityResolutionBackend):
    """
    Tier 1 — Gazetteer lookup (O(1), curated JSON files)
    Tier 2 — Embedding similarity against canonical entity store (variants)
    Tier 3 — LLM normalization (novel/ambiguous entities)
    """

    def __init__(self, session_factory, embed_model_name: str) -> None:
        self._session_factory = session_factory
        self._model_name      = embed_model_name
        self._st_model        = None  # lazy

    # ── Gazetteer lookups (Postgres — db.gazetteers) ─────────────────────────

    def _get_qid(self, entity_type: str, canonical: str) -> Optional[str]:
        with self._session_factory() as session:
            qid = gaz_db.get_qid(session, entity_type, canonical)
            if qid is None and entity_type.lower() in _ANY_TYPE_FALLBACK_TYPES:
                qid = gaz_db.get_qid_any_type(session, canonical)
            return qid

    # ── Tier 1: Gazetteer ───────────────────────────────────────────────────

    def _tier1(
        self, mention: str, entity_type: str
    ) -> Optional[EntityResolutionResult]:
        with self._session_factory() as session:
            canonical = gaz_db.lookup(session, entity_type.lower(), mention)
            if canonical is None and entity_type.lower() in _ANY_TYPE_FALLBACK_TYPES:
                # entity_type labels (e.g. "skill" vs "technology") aren't a
                # reliable partition of the same real-world term — fall back
                # to a type-agnostic lookup before giving up on Tier 1.
                canonical = gaz_db.lookup_any_type(session, mention)
            if canonical is None:
                return None
            qid = (
                gaz_db.get_qid(session, entity_type.lower(), canonical)
                or gaz_db.get_qid_any_type(session, canonical)
            )
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
            self._st_model = SentenceTransformer(self._model_name, device=EMBED_DEVICE)

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
            if not canonical_values and entity_type.lower() in _ANY_TYPE_FALLBACK_TYPES:
                # Same type-agnostic fallback as Tier 1 (see _tier1).
                canonical_values = gaz_db.canonical_values_any_type(session)
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

        if best_score >= 0.88:
            canonical = canonical_values[best_i]
            # Cosine similarity alone isn't trustworthy for full entity names
            # — generic shared words ("University", "Institute") inflate the
            # score between otherwise-unrelated entities. Same-string and
            # acronym cases skip the LLM call (unambiguous); anything else
            # must be confirmed before being accepted, mirroring stage6's
            # retrieve-then-verify design for relationship matching.
            if (
                canonical.strip().lower() == mention.strip().lower()
                or _is_acronym_expansion(mention, canonical)
                or _llm_confirms_entity_match(mention, canonical)
            ):
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
        # max_tokens=600 (was 40 — same truncation risk from reasoning
        # overhead; measured completion_tokens=295 for a real mention,
        # ~2x margin).
        raw = call_llm(prompt, max_tokens=1000).strip()

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

        # Guard against hallucinated substitution: a "normalization" that
        # shares no distinguishing words with the mention isn't a
        # normalization at all — it's the model swapping in a different,
        # unrelated entity it happens to know well (e.g. "Air University" ->
        # "Brown University"). Legitimate acronym expansions ("MIT" ->
        # "Massachusetts Institute of Technology") are exempted since they
        # never share lexical tokens with their own expansion.
        if (
            canonical.strip().lower() != mention.strip().lower()
            and not _is_acronym_expansion(mention, canonical)
        ):
            mention_tokens = _content_tokens(mention)
            canonical_tokens = _content_tokens(canonical)
            if mention_tokens and canonical_tokens and mention_tokens.isdisjoint(canonical_tokens):
                canonical, confidence = mention, 0.3

        # Persist a successful, changed normalization so repeat mentions of
        # the same string resolve identically via Tier 1 across paths/runs
        # (Tier 3 is otherwise non-deterministic — the source of FPGA vs
        # Field-Programmable Gate Array style splits). add_alias dedups on
        # (entity_type, lower(alias)); best-effort, never fails resolution.
        if (
            canonical.strip().lower() != mention.strip().lower()
            and confidence >= 0.7
        ):
            try:
                with self._session_factory() as session:
                    gaz_db.add_alias(
                        session, entity_type.lower(), mention, canonical,
                        source="tier3_llm",
                    )
            except Exception:
                pass

        return EntityResolutionResult(
            original_mention=mention,
            canonical_form=canonical,
            entity_type=entity_type,
            resolution_tier="llm",
            confidence=confidence,
            wikidata_qid=None,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def resolve(
        self, mention: str, entity_type: str, context: str = ""
    ) -> EntityResolutionResult:
        if entity_type.strip().lower() in _NEVER_RESOLVE_TYPES:
            return EntityResolutionResult(
                original_mention=mention,
                canonical_form=mention,
                entity_type=entity_type,
                resolution_tier="unresolved",
                confidence=0.0,
                wikidata_qid=None,
            )
        result = self._tier1(mention, entity_type)
        if result:
            return result

        result = self._tier2(mention, entity_type, context)
        if result:
            return result

        result = self._tier3(mention, entity_type, context)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# KG entity post-processor
# ══════════════════════════════════════════════════════════════════════════════

def _slugify(text: str) -> str:
    """'Google Inc.' → 'google_inc' — lowercased because URIs are Neo4j MERGE
    keys (see db/kg_staging._slugify); labels carry the display casing."""
    return re.sub(r"[^\w]", "_", text.lower()).strip("_")


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
    from rdflib import URIRef, Graph, Literal
    from rdflib.namespace import RDF, RDFS

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
                # Treat every local name as a resolvable slug except real
                # Wikidata QIDs. (The old "has underscore or starts uppercase"
                # heuristic silently skipped single-word lowercase slugs like
                # "python" once slugs became lowercase.)
                if not re.fullmatch(r"Q\d+", local):
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

    # When a URI is rewritten to a canonical identity, its rdfs:label must
    # follow — otherwise the old mention's label rides along and, via
    # stage_graph's label→name mapping plus Neo4j's `SET e += row`,
    # overwrites the canonical node's name (observed: wd:Machine_Learning
    # ending up named "Predictive Optimization Models").
    canonical_label = {
        old_uri: resolution_map[old_uri].canonical_form
        for old_uri in uri_rewrite
        if resolution_map.get(old_uri)
        and resolution_map[old_uri].canonical_form
    }

    for s, p, o in g:
        if p == RDFS.label and str(s) in canonical_label:
            g2.add((_remap(s), p, Literal(canonical_label[str(s)], lang="en")))
            continue
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
    relation: dict, context_cqs: list[str], doc_name: str = "unknown",
    endpoint: dict | None = None,
) -> RelationCanonResult:
    return get_edc_backend().canonicalize(relation, context_cqs, doc_name, endpoint=endpoint)


def resolve_entity(
    mention: str, entity_type: str, context: str = ""
) -> EntityResolutionResult:
    return get_entity_resolver().resolve(mention, entity_type, context)


def flush_canon_store() -> None:
    get_edc_backend().flush()