"""Regression tests for the case-duplicate / mis-canonicalization graph bugs.

Covers the three failure modes found in the 2026-07-07 session-2/3 post-mortems:
  1. The three wd:-URI slugifiers drifting out of lockstep (case-duplicate
     Neo4j nodes: wd:Python vs wd:python, two person nodes, …).
  2. ResumeEntityResolver canonicalizing résumé-unique / unmatched types
     (project "Predictive Optimization Models" → skill "Machine Learning").
  3. resolve_kg_entities rewriting a URI to its canonical identity but
     leaving the old mention's rdfs:label attached (wd:Machine_Learning
     ending up named "Predictive Optimization Models" in Neo4j).

None of these tests need Postgres, the embedding model, or the LLM server.
"""
import pytest

from db.kg_staging import _slugify as slug_path_a
from stages.stage9_10_kg import _slugify as slug_path_b
from stages.canonicalize import (
    EntityResolutionResult,
    ResumeEntityResolver,
    _slugify as slug_resolver,
    resolve_kg_entities,
)


# ── 1. Slugifier parity / lowercasing ────────────────────────────────────────

def test_slugifiers_lowercase_and_agree():
    for s in ("Scikit-Learn", "scikit-learn", "SCIKIT LEARN"):
        assert slug_path_a(s) == slug_path_b(s) == "scikit_learn"
        assert slug_resolver(s) == "scikit_learn"
    assert slug_path_a("Python") == slug_path_b("Python") == "python"
    assert slug_resolver("Python") == "python"


def test_slugifier_empty_sentinel_agrees():
    assert slug_path_a("") == slug_path_b("...") == "unknown"


# ── 2. Hard type gate ────────────────────────────────────────────────────────

@pytest.mark.parametrize("etype", ["project", "person", "unknown", "Project", "PERSON"])
def test_never_resolve_types_return_unresolved(etype):
    # The gate must return before any DB/model access, so a resolver with a
    # dummy session factory and model name is safe to construct.
    r = ResumeEntityResolver(session_factory=None, embed_model_name="unused")
    result = r.resolve("Reinforcement Learning Suite", etype)
    assert result.resolution_tier == "unresolved"
    assert result.canonical_form == "Reinforcement Learning Suite"
    assert result.wikidata_qid is None


# ── 3. Label rewritten alongside the URI ─────────────────────────────────────

class _StubResolver:
    """Rewrites 'sklearn' to 'scikit-learn'; leaves everything else alone."""

    def resolve(self, mention, entity_type, context=""):
        if mention == "sklearn":
            return EntityResolutionResult(
                original_mention=mention,
                canonical_form="scikit-learn",
                entity_type=entity_type,
                resolution_tier="gazetteer",
                confidence=1.0,
                wikidata_qid=None,
            )
        return EntityResolutionResult(
            original_mention=mention,
            canonical_form=mention,
            entity_type=entity_type,
            resolution_tier="unresolved",
            confidence=0.0,
            wikidata_qid=None,
        )


TURTLE = """\
@prefix wd:   <http://www.wikidata.org/entity/> .
@prefix wdt:  <http://www.wikidata.org/prop/direct/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

wd:jane_doe wdt:HasSkill wd:sklearn .
wd:sklearn rdfs:label "sklearn"@en .
"""


def test_resolve_kg_entities_rewrites_label_with_uri():
    rewritten, resolution_map = resolve_kg_entities(TURTLE, _StubResolver())

    import rdflib
    from rdflib.namespace import RDFS

    g = rdflib.Graph()
    g.parse(data=rewritten, format="turtle")

    new_uri = rdflib.URIRef("http://www.wikidata.org/entity/scikit_learn")
    old_uri = rdflib.URIRef("http://www.wikidata.org/entity/sklearn")

    labels = [str(o) for o in g.objects(new_uri, RDFS.label)]
    assert labels == ["scikit-learn"], (
        "rewritten URI must carry the canonical label, not the old mention's"
    )
    assert list(g.objects(old_uri, RDFS.label)) == [], "old URI must be gone"
    # The skill edge must point at the rewritten URI.
    wdt_has_skill = rdflib.URIRef("http://www.wikidata.org/prop/direct/HasSkill")
    assert (None, wdt_has_skill, new_uri) in g
