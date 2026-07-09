"""
Stages 7 & 8 — Ontology creation logic + OWL/Turtle formatting.

Stage 7: decide which properties make it into the ontology
  - Wikidata match         → add Wikidata property
  - EDC canon store match  → reuse previously generated Turtle (no new LLM call)
  - No match + SCHEMA_EXPANSION=True  → deterministic Turtle template,
                                         registered in EDC store
  - No match + SCHEMA_EXPANSION=False → discard

Stage 8: format into Turtle (OWL)
  - Matched props (Wikidata or canon): use known Turtle
  - New props: deterministic template (label + description), no LLM

Ontology quality validation:
  - Each Turtle block is validated with rdflib before being added; invalid
    blocks are dropped with a diagnostic instead of silently corrupting
    Stage 9's input.
  - Full assembled ontology is validated before returning; the pipeline
    fails fast with a clear error if the result is not parseable.

Dedup note (fixed):
  Stage 6 intentionally returns one match result per ORIGINAL relation
  occurrence (e.g. if 13 different CQ-derived relations all resolved to the
  same Wikidata property, that property appears 13 times in match_results —
  this is correct and documented behaviour for stage 6, since callers need
  a 1:1 mapping back to their source relations).
  Previously, build_ontology() only deduplicated the SCHEMA_EXPANSION
  "new property" branch (via seen_new_props) and NOT the Wikidata-match or
  canon-match branches, so a property matched 13 times would get 13
  duplicate `wdt:X a wikibase:Property ; ...` blocks written into the
  ontology. This bloats the ontology text fed into Stage 9's prompt with
  pure noise and increases the chance of the LLM echoing a property name
  back incorrectly. Fixed by deduplicating the Wikidata-match branch by
  `pid` and the canon-match branch by the rendered Turtle text, mirroring
  the existing seen_new_props pattern.

AUDIT NOTE (this pass):
  Verified against an actual pipeline run (farrukh_result1v2.zip):
  outputs/ontology/yourfile.ttl — the file this module's run() actually
  writes — came out with exactly 9 unique property blocks, correctly
  deduped by pid, matching the 9 real facts in the source CV (the CV also
  produced 31 duplicate "project name" relations that all correctly
  collapsed into a single WorkingTitle block). The dedup logic in this file
  is working as intended.

  However, a SEPARATE file, outputs/ontology/ontology.ttl, was found in the
  same run with 124 lines and properties like DataAnalysisMethod duplicated
  14 times. Nothing in this module writes to a file called "ontology.ttl"
  (only "{doc_name}.ttl") — that file must be produced by something else in
  the pipeline (a corpus-level merge step, a different stage, or a stale
  artifact from before this dedup fix existed). The duplication bug
  reported earlier is real, but it is not in this file — track it down in
  whatever script writes outputs/ontology/ontology.ttl (pipeline.py is the
  likely candidate).

PATCH NOTE — new properties are now minted deterministically, not by LLM:
  _new_prop_turtle() used to make a freeform LLM call (~40s each on a
  reasoning model) that invented the whole Turtle block, including the URI
  local name and speculative rdfs:domain/rdfs:range. Traced consumption
  showed none of that extra content is ever read: stage9_10_kg's
  _ontology_predicate_map() extracts only the URI local names, the Stage 9
  prompt receives only that name list, load_to_neo4j/render never look at
  domain/range/comment, and the EDC canon store just re-emits the stored
  block text verbatim. So the block is now rendered from a fixed template
  (local name = _camel_slug(label), plus schema:description and
  rdfs:label) — byte-equivalent downstream, zero LLM calls. This also
  makes two whole failure classes impossible by construction: LLM-minted
  bare-PID local names (wdt:P39 for "duration" — silently dropped every
  matching fact in Stage 9), and invalid/truncated Turtle blocks whose
  properties vanished from the ontology when validation dropped them.
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUTS_DIR, SCHEMA_EXPANSION

PREFIXES = """\
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix wikibase: <http://wikiba.se/ontology#> .
@prefix schema: <http://schema.org/> .
@prefix wd: <http://www.wikidata.org/entity/> .
@prefix wdt: <http://www.wikidata.org/prop/direct/> .
"""

# ── Slug helpers ────────────────────────────────────────────────────────────

def _camel_slug(label: str) -> str:
    """
    'duration of role' -> 'DurationOfRole'
    'course name'       -> 'CourseName'
    Strips anything that isn't alphanumeric before casing each word.
    """
    words = re.findall(r"[A-Za-z0-9]+", label)
    if not words:
        return "Property"
    return "".join(w[:1].upper() + w[1:] for w in words)


def _ttl_escape(text: str) -> str:
    """Escape a value for use inside a double-quoted Turtle literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


# ── Turtle block validation ────────────────────────────────────────────────

def _validate_turtle_block(block: str, label: str = "") -> tuple[bool, str]:
    """
    Return (is_valid, diagnostic) for a single Turtle property block.
    Wraps the block with prefix declarations before parsing so the test
    matches what rdflib will see in the assembled ontology.

    Rejects empty blocks (e.g. a truncated LLM response) and blocks that
    parse but yield zero triples — bare prefix declarations are valid
    Turtle syntax on their own, so parseability alone isn't proof of
    substantive content.
    """
    import rdflib
    if not block or not block.strip():
        diag = f"Turtle validation failed for '{label}': empty block (likely a truncated LLM response)"
        print(f"[Stage 7/8]   ⚠ {diag}", flush=True)
        return False, diag
    try:
        g = rdflib.Graph()
        g.parse(data=PREFIXES + "\n" + block, format="turtle")
        if len(g) == 0:
            diag = f"Turtle validation failed for '{label}': parsed but produced 0 triples"
            print(f"[Stage 7/8]   ⚠ {diag}", flush=True)
            return False, diag
        return True, ""
    except Exception as e:
        diag = f"Turtle validation failed for '{label}': {e}"
        print(f"[Stage 7/8]   ⚠ {diag}", flush=True)
        return False, diag


def _validate_full_ontology(ontology_turtle: str) -> tuple[bool, str]:
    """
    Validate the assembled ontology.  Returns (is_valid, diagnostic).
    Called before returning from run() — fail fast if the full graph is broken
    OR contains zero triples (e.g. every property was rejected/unmatched).
    """
    import rdflib
    try:
        g = rdflib.Graph()
        g.parse(data=ontology_turtle, format="turtle")
        triple_count = len(g)
        if triple_count == 0:
            return False, "ontology parsed as valid Turtle but contains 0 triples — every property was rejected or matched nothing usable"
        return True, f"{triple_count} triples parsed successfully"
    except Exception as e:
        return False, str(e)


# ── Turtle formatters ──────────────────────────────────────────────────────

def _wikidata_turtle(prop: dict) -> str:
    label = prop["label"]
    desc  = prop.get("description", "")
    return (
        f"wdt:{label} a wikibase:Property ;\n"
        f'    schema:description "{_ttl_escape(desc)}" ;\n'
        f'    rdfs:label "{_ttl_escape(label)}"@en .\n'
    )


def _new_prop_turtle(extracted: dict, definition: str = "") -> str:
    """Render the Turtle block for a newly-minted (non-Wikidata) property.

    Deterministic template, same shape as _wikidata_turtle() but with the
    local name slugged from the label (Wikidata labels are already
    PascalCase; extracted labels are free text like "duration of role").
    The EDC definition is preferred as the description when available —
    it's the canonical phrasing the canon store embeds and matches against.
    """
    label = extracted["property"]
    desc  = definition.strip() or str(extracted.get("description", "")).strip() or label
    return (
        f"wdt:{_camel_slug(label)} a wikibase:Property ;\n"
        f'    schema:description "{_ttl_escape(desc)}" ;\n'
        f'    rdfs:label "{_ttl_escape(label)}"@en .\n'
    )


# ── Core ontology builder ──────────────────────────────────────────────────

def build_ontology(match_results: list[dict]) -> tuple[str, dict[str, str]]:
    """
    Returns (ontology_turtle, new_prop_turtle_map).

    new_prop_turtle_map: {property_label → turtle_block} for all genuinely
    new properties — used by the pipeline to register them in the EDC canon
    store after this function returns.
    """
    turtle_blocks  = [PREFIXES]
    seen_wikidata_pids: set[str] = set()   # dedup Wikidata matches by pid
    seen_canon_blocks:  set[str] = set()   # dedup canon-store reuses by exact text
    seen_new_props: dict[str, str] = {}    # lower(label) → turtle block (successes only)
    # Every new-property key attempted this run, success or fail — a property
    # with many relation instances (e.g. 'hasSkill', once per skill listed)
    # must still be minted exactly once. Kept even now that minting is a
    # deterministic template (no LLM): it guards against duplicate blocks and
    # repeated failure logging, not just wasted calls.
    attempted_new_props: set[str] = set()
    new_prop_map:   dict[str, str] = {}    # label → turtle block (for EDC)

    for item in match_results:
        extracted = item["extracted"]
        match     = item.get("wikidata_match")
        canon     = item.get("canon_match")   # EDC canon store match

        if match is not None:
            pid = match.get("pid")
            if pid in seen_wikidata_pids:
                continue  # this Wikidata property was already emitted once
            seen_wikidata_pids.add(pid)
            turtle_blocks.append(_wikidata_turtle(match))

        elif canon is not None:
            # Reuse turtle from EDC canon store — no new LLM call needed
            existing_turtle = canon.get("turtle", "")
            if existing_turtle:
                normalized = existing_turtle.strip()
                if normalized in seen_canon_blocks:
                    continue  # already emitted this exact canon block once
                seen_canon_blocks.add(normalized)
                turtle_blocks.append(existing_turtle)
            else:
                # Canon entry predates turtle storage — fall through to generate
                pass

        else:
            if not SCHEMA_EXPANSION:
                continue  # target-schema-constrained mode: discard

            key = extracted["property"].strip().lower()
            if key in attempted_new_props:
                continue  # already attempted this run (success or fail) — never retry
            attempted_new_props.add(key)

            block = _new_prop_turtle(extracted, definition=item.get("edc_definition", ""))
            valid, diag = _validate_turtle_block(block, extracted["property"])
            if not valid:
                print(
                    f"[Stage 7/8]   dropping invalid Turtle block for "
                    f"'{extracted['property']}': {diag}",
                    flush=True,
                )
                continue

            seen_new_props[key] = block
            new_prop_map[extracted["property"]] = block

    turtle_blocks.extend(seen_new_props.values())
    return "\n".join(turtle_blocks), new_prop_map


def run(source_doc, doc_name: str, match_results: list[dict]) -> str:
    """Build + validate the ontology Turtle and return it.

    The pipeline persists the result to pipeline_runs (stage 'ontology'); this
    function no longer writes outputs/ontology/{doc}.ttl. `source_doc` is the
    résumé UUID recorded on any newly-minted EDC canon-store entries.
    """
    ontology_turtle, new_prop_map = build_ontology(match_results)

    # ── Ontology quality validation (fail fast) ────────────────────────────
    valid, diag = _validate_full_ontology(ontology_turtle)
    if not valid:
        raise ValueError(
            f"[Stage 7/8] FATAL: assembled ontology is not valid Turtle — "
            f"Stage 9 would receive broken input.\n  Diagnostic: {diag}"
        )
    print(f"[Stage 7/8] ✓ ontology validation passed — {diag}", flush=True)

    # Register genuinely new properties in the EDC canon store so subsequent
    # documents can merge against them.
    if new_prop_map:
        try:
            from stages.canonicalize import get_edc_backend
            edc = get_edc_backend()
            for label, turtle_block in new_prop_map.items():
                # The definition stored in match_results came from the EDC
                # "Define" step in canonicalize.  Retrieve it if present.
                definition = ""
                for item in match_results:
                    if item["extracted"]["property"] == label:
                        definition = item.get("edc_definition", item["extracted"].get("description", ""))
                        break
                edc.register_new_property(
                    label=label,
                    definition=definition,
                    turtle=turtle_block,
                    source_doc=source_doc,
                )
            edc.flush()
            print(
                f"[Stage 7/8] {len(new_prop_map)} new propert(ies) registered in EDC canon store",
                flush=True,
            )
        except Exception as e:
            print(f"[Stage 7/8] ⚠ EDC registration skipped: {e}", flush=True)

    return ontology_turtle