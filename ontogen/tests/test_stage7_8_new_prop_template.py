"""Tests for the deterministic new-property Turtle template in Stage 7/8.

_new_prop_turtle() used to be a freeform LLM call (~40s each); it's now a
fixed template because nothing downstream ever consumed anything beyond the
URI local name, rdfs:label, and schema:description (see the module's PATCH
NOTE). These tests pin that contract: no LLM involvement, rdflib-valid
output, local name derived from the label, and safe literal escaping.

None of these tests need Postgres, the embedding model, or the LLM server.
"""
import rdflib

from stages.stage7_8_ontology import (
    PREFIXES,
    _camel_slug,
    _new_prop_turtle,
    _validate_turtle_block,
    build_ontology,
)
from stages.stage9_10_kg import _ontology_predicate_map


def _parse(block: str) -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(data=PREFIXES + "\n" + block, format="turtle")
    return g


# ── Template output shape ────────────────────────────────────────────────────

def test_template_is_valid_turtle_with_expected_triples():
    block = _new_prop_turtle(
        {"property": "duration of role", "description": "How long a role was held."}
    )
    valid, diag = _validate_turtle_block(block, "duration of role")
    assert valid, diag

    g = _parse(block)
    wdt = "http://www.wikidata.org/prop/direct/"
    subj = rdflib.URIRef(wdt + "DurationOfRole")
    labels = list(g.objects(subj, rdflib.RDFS.label))
    descs = list(g.objects(subj, rdflib.URIRef("http://schema.org/description")))
    assert [str(l) for l in labels] == ["duration of role"]
    assert [str(d) for d in descs] == ["How long a role was held."]


def test_local_name_is_camel_slug_of_label_never_a_bare_pid():
    # The old LLM path once minted wdt:P39 for "duration" — the template
    # can only ever produce the slug of the label.
    block = _new_prop_turtle({"property": "duration", "description": ""})
    assert "wdt:Duration " in block
    assert _camel_slug("duration") == "Duration"


def test_edc_definition_preferred_over_extracted_description():
    block = _new_prop_turtle(
        {"property": "hasSkill", "description": "raw description"},
        definition="A person possesses a specific skill.",
    )
    assert "A person possesses a specific skill." in block
    assert "raw description" not in block


def test_description_falls_back_to_label_when_empty():
    block = _new_prop_turtle({"property": "used tool", "description": ""})
    g = _parse(block)
    descs = [
        str(o)
        for o in g.objects(None, rdflib.URIRef("http://schema.org/description"))
    ]
    assert descs == ["used tool"]


def test_quotes_and_newlines_in_description_are_escaped():
    block = _new_prop_turtle(
        {
            "property": "quote holder",
            "description": 'Said "hello" \\ world\nsecond line',
        }
    )
    valid, diag = _validate_turtle_block(block, "quote holder")
    assert valid, diag
    g = _parse(block)
    descs = [
        str(o)
        for o in g.objects(None, rdflib.URIRef("http://schema.org/description"))
    ]
    assert descs == ['Said "hello" \\ world second line']


# ── No-LLM + downstream-contract guarantees ──────────────────────────────────

def test_build_ontology_new_props_make_no_llm_calls(monkeypatch):
    # call_llm must not even be imported by the module anymore; belt and
    # braces, also assert the pipeline path works with the LLM stack down.
    import stages.stage7_8_ontology as mod

    assert not hasattr(mod, "call_llm")

    match_results = [
        {
            "extracted": {"property": "used tool", "description": "Tools used."},
            "wikidata_match": None,
            "canon_match": None,
            "edc_definition": "The tools used in a project or system.",
        },
        # Repeat instance of the same property — must be minted exactly once.
        {
            "extracted": {"property": "used tool", "description": "Tools used."},
            "wikidata_match": None,
            "canon_match": None,
            "edc_definition": "The tools used in a project or system.",
        },
        {
            "extracted": {"property": "used algorithm", "description": "Algorithms."},
            "wikidata_match": None,
            "canon_match": None,
        },
    ]
    ontology, new_prop_map = build_ontology(match_results)

    assert set(new_prop_map) == {"used tool", "used algorithm"}
    assert ontology.count("wdt:UsedTool ") == 1

    # Stage 9 consumes only the predicate local names — pin that contract.
    predicate_map = _ontology_predicate_map(ontology)
    assert set(predicate_map.values()) == {"UsedTool", "UsedAlgorithm"}
