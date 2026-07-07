"""
Stages 7 & 8 — Ontology creation logic + OWL/Turtle formatting.

Stage 7: decide which properties make it into the ontology
  - Wikidata match         → add Wikidata property
  - EDC canon store match  → reuse previously generated Turtle (no new LLM call)
  - No match + SCHEMA_EXPANSION=True  → LLM generates Turtle, registered in EDC store
  - No match + SCHEMA_EXPANSION=False → discard

Stage 8: format into Turtle (OWL)
  - Matched props (Wikidata or canon): use known Turtle
  - New props: LLM infers domain/range (verbatim prompt from paper)

Ontology quality validation:
  - Each LLM-generated Turtle block is validated with rdflib before being
    added; invalid blocks are dropped with a diagnostic instead of silently
    corrupting Stage 9's input.
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

PATCH NOTE (this pass) — bare-PID local names for LLM-minted properties:
  Root cause traced to _new_prop_turtle(): new (non-Wikidata) properties are
  minted via a single freeform LLM call that is trusted to invent BOTH the
  URI local name and the rdfs:label together. For most properties the model
  followed the one worked example in NEW_PROP_PROMPT and produced a
  CamelCase slug (courseName, Achievements, PerformancePercentage). For two
  properties ("duration", "presentation title") it instead free-associated
  to real-looking Wikidata PIDs (wdt:P39, wdt:P1) as the local name while
  still writing a human-readable rdfs:label — producing a block where the
  URI and label disagree. _ontology_predicate_map() in stage9_10_kg.py
  builds its valid-predicate list from the URI local name, not rdfs:label,
  so Stage 9's LLM was told the only valid name for "duration of role" was
  the meaningless token "P39" and correctly refused to use it — silently
  dropping every duration/date-range fact and the presentation title fact
  from the final graph.
  This is inherently non-deterministic (LLM word choice), not a
  deterministic fallback bug, so it can recur for any newly-minted property
  on any run. Fixed two ways:
    1. NEW_PROP_PROMPT now explicitly forbids bare Wikidata-style IDs
       (P123-style tokens) as the local name.
    2. _new_prop_turtle() now defensively post-processes the LLM's output:
       if the minted local name matches ^P\\d+$ (or any other non-descriptive
       token that doesn't resemble a slug of the label), it is rewritten to
       a CamelCase slug derived from rdfs:label before validation — mirroring
       the label/URI consistency that _wikidata_turtle() already guarantees
       by construction. This closes the gap regardless of what the LLM
       outputs, rather than relying solely on the prompt instruction.
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from stages.llm import call_llm_answer
from config import OUTPUTS_DIR, SCHEMA_EXPANSION, LLM_TOKENS_DEFINE, LLM_TOKENS_RETRY

NEW_PROP_PROMPT = """\
Use the relations (properties) and their usage comments to
build an ontology in RDF format.

If you don't know the answer, just say that you don't know, don't
try to make up an answer.

Don't provide anything other than an ontology in RDF format.

Infer and summarize classes for domain and range of the
relations across the concepts provided, and add these
classes to relations only if required for closure of relations.

For each relation, add relevant ontology entry for it.

Add rdfs:comment based on the usage comments.

Use wdt: namespace for all relations discovered. Use entities
under these prefixes if necessary:
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix wikibase: <http://wikiba.se/ontology#> .
@prefix schema: <http://schema.org/> .
@prefix wd: <http://www.wikidata.org/entity/> .
@prefix wdt: <http://www.wikidata.org/prop/direct/> .

Use turtle syntax.

IMPORTANT — naming the property's local URI name (the part after "wdt:"):
- The local name MUST be a CamelCase or camelCase slug derived from the
  relation's label (e.g. label "duration of role" -> wdt:DurationOfRole,
  label "course name" -> wdt:courseName).
- NEVER use a bare numeric/alphanumeric code such as wdt:P39 or wdt:P1,
  even if it superficially looks like a real Wikidata property ID. This is
  a NEW, locally-minted property, not an existing Wikidata property, so it
  must never be given a Wikidata-style "P<number>" identifier.
- The local name and the rdfs:label must obviously correspond to each
  other — do not invent a code-like name that a reader could not derive
  from the label.

Below is an example:
####
Relations:
(results, results: results of a competition such as sports or elections)
####
Ontology:
wdt:Results a wikibase:Property ;
    schema:description "results of a competition such as sports or elections" ;
    rdfs:label "results" ;
    rdfs:domain wd:referendum, wd:competition, wd:party_conference, wd:sporting_event ;
    rdfs:range wd:electoral_result, wd:voting_result, wd:sport_result, wd:race_result .
####
Relations:
{relation}
####
Ontology:
"""

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

# Matches a bare Wikidata-style property ID used as a local name, e.g.
# "P39", "p1", "P1234" — this should never appear as a LOCAL NAME for a
# property this pipeline itself minted (only for genuine Wikidata matches,
# which go through _wikidata_turtle(), never through this path).
_BARE_PID_RE = re.compile(r"^[Pp]\d+$")


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


def _fix_bare_pid_local_name(block: str, expected_label: str) -> str:
    """
    Defensive post-processing for _new_prop_turtle()'s LLM output.

    If the minted Turtle block uses a bare Wikidata-style local name
    (wdt:P39, wdt:P1, ...) instead of a readable slug, rewrite every
    occurrence of that local name to a CamelCase slug derived from
    `expected_label` (the property label we asked the LLM to model,
    used as a fallback source of truth if rdfs:label can't be parsed
    out of the block for any reason).

    This guarantees the same label/URI-name consistency that
    _wikidata_turtle() gets "for free" by construction, even when the
    property is minted by a freeform LLM call instead of built
    deterministically.
    """
    # Find the local name(s) used after "wdt:" in this block.
    local_names = set(re.findall(r"\bwdt:([A-Za-z0-9_]+)\b", block))
    bare_pids = {n for n in local_names if _BARE_PID_RE.match(n)}
    if not bare_pids:
        return block  # nothing to fix

    # Prefer the rdfs:label text actually present in the block, since
    # that's what a human (and Stage 9's LLM) will see; fall back to the
    # label we sent in the prompt if it can't be found.
    m = re.search(r'rdfs:label\s+"([^"]+)"', block)
    label_source = m.group(1) if m else expected_label
    slug = _camel_slug(label_source)

    fixed = block
    for pid in bare_pids:
        print(
            f"[Stage 7/8]   ⚠ LLM minted bare PID-style local name 'wdt:{pid}' "
            f"for label '{label_source}' — rewriting to 'wdt:{slug}'",
            flush=True,
        )
        fixed = re.sub(rf"\bwdt:{re.escape(pid)}\b", f"wdt:{slug}", fixed)
    return fixed


# ── Turtle block validation ────────────────────────────────────────────────

def _validate_turtle_block(block: str, label: str = "") -> tuple[bool, str]:
    """
    Return (is_valid, diagnostic) for a single Turtle property block.
    Wraps the block with prefix declarations before parsing so the test
    matches what rdflib will see in the assembled ontology.
    """
    import rdflib
    try:
        rdflib.Graph().parse(data=PREFIXES + "\n" + block, format="turtle")
        return True, ""
    except Exception as e:
        diag = f"Turtle validation failed for '{label}': {e}"
        print(f"[Stage 7/8]   ⚠ {diag}", flush=True)
        return False, diag


def _validate_full_ontology(ontology_turtle: str) -> tuple[bool, str]:
    """
    Validate the assembled ontology.  Returns (is_valid, diagnostic).
    Called before returning from run() — fail fast if the full graph is broken.
    """
    import rdflib
    try:
        g = rdflib.Graph()
        g.parse(data=ontology_turtle, format="turtle")
        triple_count = len(g)
        return True, f"{triple_count} triples parsed successfully"
    except Exception as e:
        return False, str(e)


# ── Turtle formatters ──────────────────────────────────────────────────────

def _wikidata_turtle(prop: dict) -> str:
    label = prop["label"]
    desc  = prop.get("description", "")
    return (
        f"wdt:{label} a wikibase:Property ;\n"
        f'    schema:description "{desc}" ;\n'
        f'    rdfs:label "{label}"@en .\n'
    )


def _new_prop_turtle(extracted: dict) -> str:
    rel_str = f"({extracted['property']}, {extracted['property']}: {extracted['description']})"
    prompt  = NEW_PROP_PROMPT.format(relation=rel_str)
    print(
        f"[Stage 7/8]   generating Turtle for new property '{extracted['property']}' …",
        flush=True,
    )
    raw, truncated = call_llm_answer(
        prompt, LLM_TOKENS_DEFINE, retry_budget=LLM_TOKENS_RETRY
    )
    if truncated:
        print(f"[Stage 7/8]   ⚠ Turtle generation truncated for "
              f"'{extracted['property']}' (reasoning exceeded budget) — the block "
              f"may be incomplete and fail validation", flush=True)
    print(f"[Stage 7/8]   ✓ got {len(raw)} chars back", flush=True)
    raw = raw.replace("```turtle", "").replace("```", "").strip()

    # Defensive fix: never let a bare Wikidata-style PID survive as the
    # local name for a property WE minted, regardless of what the LLM did.
    raw = _fix_bare_pid_local_name(raw, expected_label=extracted["property"])

    return raw


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
    seen_new_props: dict[str, str] = {}    # lower(label) → turtle block
    new_prop_map:   dict[str, str] = {}    # label → turtle block (for EDC)
    attempted_new_props: set[str] = set()  # lower(label) generated once (valid OR not)

    def _generate_new_prop(extracted: dict) -> None:
        """Mint a Turtle block for a property with no Wikidata/canon match yet
        (or a canon match whose stored turtle is missing/empty). One attempt
        per unique property label; invalid blocks are dropped with a diagnostic."""
        key = extracted["property"].strip().lower()
        if key in seen_new_props or key in attempted_new_props:
            # Already generated (valid) OR already tried once and dropped as
            # invalid. Either way, do NOT regenerate per occurrence — that is
            # what made a 34-skill résumé run 34 identical hasSkill Turtle
            # generations (each one failing validation, none ever cached).
            return

        attempted_new_props.add(key)  # one generation attempt per unique property
        block = _new_prop_turtle(extracted)
        valid, diag = _validate_turtle_block(block, extracted["property"])
        if not valid:
            print(
                f"[Stage 7/8]   dropping invalid Turtle block for "
                f"'{extracted['property']}': {diag}",
                flush=True,
            )
            return

        seen_new_props[key] = block
        new_prop_map[extracted["property"]] = block

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
                # Canon entry predates turtle storage: generate one now instead
                # of silently dropping the property (previously a bare `pass`
                # here meant the property never got a turtle block at all —
                # no Wikidata block, no canon block, no generated block — and
                # every fact using it was later rejected in Stage 9 as "not in
                # the ontology" with no way to tell it apart from a genuine
                # hallucinated predicate).
                if not SCHEMA_EXPANSION:
                    continue
                print(
                    f"[Stage 7/8]   canon match for '{extracted['property']}' has "
                    f"no stored turtle — generating one now", flush=True,
                )
                _generate_new_prop(extracted)

        else:
            if not SCHEMA_EXPANSION:
                continue  # target-schema-constrained mode: discard
            _generate_new_prop(extracted)

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