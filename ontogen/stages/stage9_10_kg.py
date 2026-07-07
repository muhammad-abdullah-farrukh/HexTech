"""
Stages 9 & 10 — KG Construction + RDF Parsing + Entity Resolution + Provenance.

Stage 9: LLM generates a list of facts from (doc, QA pairs, ontology)
Stage 10: facts are validated against the ontology and assembled into a
          graph deterministically; rdflib serializes the final Turtle.

WHY THIS VERSION LOOKS DIFFERENT FROM EVERY PREVIOUS ATTEMPT:
  Every previous failure (Douglas Adams leak, `a wikibase:Property`
  redeclaration, "wikibase : Property" spacing bugs, undeclared prefixes,
  prose leaking into Turtle on retry) was a symptom of the same root cause:
  asking an 8B instruct model to hand-write punctuation-sensitive RDF/Turtle
  syntax. Tightening the prompt and adding retries made it *somewhat*
  better, then a hotter retry made it catastrophically worse (full English
  sentences spliced into the "Turtle" output).

  The fix is not a better prompt. It's removing the model's ability to
  emit syntax at all:
    - The model now outputs a JSON array of plain facts: subject,
      predicate (must be one of the ontology's property labels), object,
      and whether the object is an entity or a literal. JSON is a task
      8B instruct models handle far more reliably than Turtle.
    - Every fact is validated against the ontology's actual predicate set
      IN CODE before it is ever allowed into the graph. A predicate that
      isn't declared in the ontology is dropped, not "discouraged by
      prompt wording". `a wikibase:Property` style redeclaration is
      structurally impossible now — there's no Turtle syntax for the
      model to misuse, because the model never writes Turtle.
    - rdflib builds the graph and serializes it. The model never
      generates @prefix lines, never writes a colon, never produces a
      malformed statement — it just lists facts.
    - Retries stay at LOW, FIXED temperature with specific error feedback
      (e.g. "predicate X is not in the ontology, use one of: Y, Z").
      Escalating temperature is exactly what caused the worst failure
      seen so far, so it's gone.

  UPDATE (Ollama migration): guided_json is NOT enforced under Ollama's
  native /api/chat endpoint the way it was under vLLM — Ollama silently
  ignores the request (see the "[llm] guided_json ... ignoring" warning
  printed at call time). This means the model is now free to wrap its
  JSON in markdown fences, add prose before/after it, or get truncated
  mid-array if max_tokens is too small — none of which json.loads()
  tolerates on its own. _parse_facts() now strips markdown wrapping and
  attempts to repair a truncated array before giving up, instead of
  assuming the raw response is always clean JSON.

  UPDATE (chunking — root cause #1 follow-up): raising CQ_SAFETY_MAX in
  Stage 1 (40 -> 100) stopped CQs 41-77 being thrown away before Stage 2
  ever ran, but it made the *existing* problem here worse: this stage was
  already truncating the QA block via _guard_context() to fit one 12K-ish
  char prompt, and doubling the QA volume just meant more of it got cut
  before generate_facts() ever saw it. On top of that, even QA pairs that
  DID survive were all extracted in a single flat LLM call with a single
  implicit subject — which produced ambiguous graphs (e.g. four
  undifferentiated start dates with no way to tell which job/degree each
  one belongs to), since the model had to invent and re-remember subject
  names across one huge context.

  Fix: instead of one generate_facts() call over the whole QA list, QA
  pairs are now grouped into chunks by the "subject" field Stage 1
  attaches to every CQ (and Stage 2 passes through unchanged onto every
  QA pair) — see stage1_cq_gen.py's fix log. Each chunk gets its own
  generate_facts() call with a small, focused QA block that comfortably
  fits in context, plus a subject_hint telling the model which entity
  every fact in that batch is about — so it doesn't have to infer or
  re-remember subject identity across a large flat prompt. Per-chunk
  graphs are merged with rdflib's Graph.__add__ before serialization.
  _guard_context() is kept as a per-chunk safety net (chunks are sized to
  rarely need it, but a single subject with an unusually large number of
  QA pairs could still exceed MAX_PROMPT_CHARS on its own).

  An earlier version of this fix tried to *infer* the grouping by
  regexing over each question's wording (e.g. matching "... at X"). That
  was rejected: it only worked as well as the model's phrasing habits
  happened to cooperate, and degraded silently on any document phrased
  differently. Grouping now reads a field that was decided once, at the
  source, using a controlled output format — no natural-language guessing
  anywhere in this stage. QA pairs with subject="_unlabeled" (Stage 1
  failed to tag them, or the QA file predates tagging) are each treated
  as their own singleton chunk rather than guessed into a group — see
  _chunk_qa_pairs() below.

Post-generation pipeline (unchanged from before):
  1. Entity resolution — every wd: entity URI is resolved through the 3-tier
     resolver (gazetteer → embedding → LLM) and rewritten to its canonical form.
  2. Provenance — every accepted triple gets a ProvenanceRecord stored to
     outputs/provenance/{doc_name}.json with source doc, stage, timestamp,
     confidence, and model version.
"""
import json, re, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from stages.llm import call_llm
from config import (
    OUTPUTS_DIR, MAX_PROMPT_CHARS, ENTITY_RESOLUTION_ENABLED, LLM_MODEL,
    LLM_TOKENS_FACTS, LLM_TOKENS_FACTS_RETRY,
)

# ── Fact-extraction prompt (JSON, not Turtle) ───────────────────────────────

KG_FACTS_PROMPT = """\
Your task is to extract facts from the document below and express
them using ONLY the properties listed in the ontology.
{subject_hint}
Output a JSON array. Each element is one fact, in this exact shape:
{{
  "subject": "<plain name of the person/thing the fact is about>",
  "predicate": "<one of the property names from the ontology below, exact spelling>",
  "object": "<plain name or value>",
  "object_type": "entity" or "literal"
}}

Rules:
- "predicate" MUST be copied exactly from the ontology list below. If no
  property in the ontology fits a fact, skip that fact — do not invent
  a new predicate name and do not use a property name that isn't listed.
- "subject" and "object" are plain human-readable text (e.g. "Air
  University Islamabad"), never URIs, never prefixed names, never RDF
  syntax of any kind.
- object_type is "entity" when the object is a named thing (a person,
  organization, place, institution, tool, framework, project) and
  "literal" when it's a free-text value, date, level, or description
  (e.g. a degree title, a language proficiency level, a description).
- Only include facts that are actually supported by the document and
  the question/answer pairs below. Do not invent facts.
- Output ONLY the JSON array. No explanation, no markdown fences, no
  text before or after it.

Ontology — these are the ONLY valid values for "predicate":
{ont_labels}

Document:
{doc}
####
Questions and Answer pairs:
{qa}
####
{feedback}
Output the JSON array now:
"""

# Injected into KG_FACTS_PROMPT's {subject_hint} slot when a chunk's
# subject is a specific entity. Left blank for subject == "person" (Stage
# 1's label for identity/contact info — see stage1_cq_gen.py) where
# forcing every fact onto one subject would be wrong, and for
# "_unlabeled" chunks where no confirmed subject exists at all.
SUBJECT_HINT_BLOCK = """
The facts below are all about: "{subject}"
Use this (or the person's name, for facts that are about the person
directly rather than this entity) as the "subject" field. Only use a
different subject when a fact is clearly about some other named entity
mentioned in this batch (e.g. the location of an employer).
"""

FEEDBACK_BLOCK = """
Your previous attempt had these specific problems — fix them and
re-output the full corrected JSON array:
{issues}
"""

_FACTS_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "subject":     {"type": "string"},
            "predicate":   {"type": "string"},
            "object":      {"type": "string"},
            "object_type": {"type": "string", "enum": ["entity", "literal"]},
        },
        "required": ["subject", "predicate", "object", "object_type"],
        "additionalProperties": False,
    },
}

_UNKNOWN_PHRASES = frozenset({
    "i don't know",
    "i do not know",
    "don't know",
    "do not know",
    "no information",
    "not provided",
    "not mentioned",
    "not stated",
    "not available",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "no answer",
})


def _is_unanswered(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in _UNKNOWN_PHRASES)


def _format_qa(qa_pairs: list[dict]) -> str:
    lines = []
    for pair in qa_pairs:
        lines.append(f"Q: {pair['question']}")
        lines.append(f"A: {pair['answer']}")
    return "\n".join(lines)


def _guard_context(
    doc_text: str, qa_pairs: list[dict], ontology_labels: str, prompt_template: str
) -> tuple[str, str]:
    """Truncate doc and/or QA block so the Stage 9 prompt stays within
    MAX_PROMPT_CHARS. Kept as a per-chunk safety net: chunks are sized to
    rarely need this, but a single subject with an unusually large
    number of QA pairs (or an unusually long subject_hint) could still
    exceed the ceiling on its own."""
    qa_block = _format_qa(qa_pairs)
    total    = len(prompt_template) + len(ontology_labels) + len(doc_text) + len(qa_block)
    if total <= MAX_PROMPT_CHARS:
        return doc_text, qa_block

    headroom = MAX_PROMPT_CHARS - len(prompt_template) - len(ontology_labels)
    if headroom < 500:
        print(
            f"[Stage 9/10]   ⚠ ontology alone is near the context limit — "
            f"output quality may be poor",
            flush=True,
        )
        headroom = 500

    doc_budget = int(headroom * 0.60)
    qa_budget  = headroom - doc_budget

    if len(doc_text) > doc_budget:
        print(
            f"[Stage 9/10]   ⚠ truncating document {len(doc_text)} → {doc_budget} chars",
            flush=True,
        )
        doc_text = doc_text[:doc_budget] + " ...[truncated]"

    if len(qa_block) > qa_budget:
        print(
            f"[Stage 9/10]   ⚠ truncating QA block {len(qa_block)} → {qa_budget} chars",
            flush=True,
        )
        qa_block = qa_block[:qa_budget]

    return doc_text, qa_block


# ── Ontology → allowed predicate map ────────────────────────────────────────

_WDT_NS = "http://www.wikidata.org/prop/direct/"


def _ontology_predicate_map(ontology: str) -> dict[str, str]:
    """
    Parse the ontology and return {lowercased label: exact label}.
    Lowercased keys give the model a little slack on casing without
    allowing it to invent new predicates outright.
    """
    import rdflib
    g = rdflib.Graph()
    try:
        g.parse(data=ontology, format="turtle")
    except Exception:
        return {}
    labels: dict[str, str] = {}
    for s in g.subjects():
        s_str = str(s)
        if s_str.startswith(_WDT_NS):
            label = s_str[len(_WDT_NS):]
            labels[label.lower()] = label
    return labels


# ── Entity-name slugification (for building wd: URIs) ──────────────────────

def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w]+", "_", name.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "Unknown"


# ── Section chunking (root cause #1 follow-up) ──────────────────────────────
#
# Grouping is a plain lookup on the "subject" field Stage 1 attaches to
# every CQ and Stage 2 passes through unchanged — see stage1_cq_gen.py's
# fix log for why this replaced an earlier regex-based approach that
# guessed the grouping from question wording. There is no NLP/regex
# inference in this stage: the field either exists (normal case) or is
# "_unlabeled" (Stage 1 failed to tag it, or the QA file predates
# tagging), and "_unlabeled" pairs are each kept as their own singleton
# chunk rather than merged into a guessed group.

_UNLABELED_SUBJECT = "_unlabeled"


def _chunk_qa_pairs(
    qa_pairs: list[dict], max_chunk_chars: int
) -> list[tuple[str, list[dict]]]:
    """
    Group QA pairs into chunks by their explicit "subject" field so each
    generate_facts() call gets a small, focused, single-subject QA block
    instead of the full flat list.

    - QA pairs sharing the same subject are grouped together. Stage 1
      emits CQs in document order, so same-subject pairs are already
      adjacent by construction — grouping is done as a single forward
      pass, no reordering needed.
    - A chunk is force-split if it would exceed max_chunk_chars even
      under the same subject, so one unusually large section can't blow
      the per-chunk budget.
    - Pairs with subject == "_unlabeled" are NEVER merged with each
      other or with any other subject, even if adjacent — each becomes
      its own singleton chunk. Grouping untagged pairs together would be
      exactly the kind of guess this design is meant to avoid (two
      untagged pairs in a row have no confirmed relationship).
    - Missing the "subject" key entirely (e.g. a QA file from before
      this change) is treated the same as "_unlabeled", not as a crash.
    """
    chunks: list[tuple[str, list[dict]]] = []
    current_subject: str | None = None
    current: list[dict] = []
    current_len = 0

    def flush():
        if current:
            chunks.append((current_subject, list(current)))

    for pair in qa_pairs:
        subject = str(pair.get("subject") or _UNLABELED_SUBJECT).strip() or _UNLABELED_SUBJECT
        pair_len = len(pair["question"]) + len(pair["answer"])

        if subject == _UNLABELED_SUBJECT:
            # Always its own chunk — flush whatever was building, emit
            # this pair alone, reset.
            flush()
            chunks.append((subject, [pair]))
            current, current_len, current_subject = [], 0, None
            continue

        same_subject = (subject == current_subject)
        fits = (current_len + pair_len) <= max_chunk_chars or not current

        if current and (not same_subject or not fits):
            flush()
            current, current_len, current_subject = [], 0, None

        if current_subject is None:
            current_subject = subject

        current.append(pair)
        current_len += pair_len

    flush()
    return chunks


# ── JSON parsing ────────────────────────────────────────────────────────────

def _strip_markdown_wrapper(raw: str) -> str:
    """Ollama's native API doesn't enforce guided_json — the model is free
    to wrap the array in markdown fences or add a sentence of preamble,
    e.g. '```json\\n[...]\\n```' or 'Here is the JSON:\\n[...]'. Strip that
    off before attempting to parse."""
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    # No fence — but there may still be prose before the array. Find the
    # first '[' and assume the array starts there.
    start = text.find("[")
    if start > 0:
        return text[start:]
    return text


def _repair_truncated_array(text: str) -> str | None:
    """If the response was cut off mid-object (max_tokens reached), find
    the last complete top-level object in the array and close it there,
    rather than discarding the whole batch. Returns repaired text or None
    if no complete object could be found."""
    if not text.lstrip().startswith("["):
        return None
    depth = 0
    last_complete_end = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete_end = i + 1
    if last_complete_end is None:
        return None
    return text[:last_complete_end] + "]"


def _parse_facts(raw: str) -> tuple[list[dict] | None, str]:
    """Returns (facts, error). facts is None on unrecoverable failure.

    NOTE: guided_json is NOT enforced under Ollama's native /api/chat
    endpoint (vLLM guaranteed it; Ollama silently ignores the request —
    see the [llm] warning printed at call time). So this has to tolerate
    markdown-wrapped output and mid-array truncation, not just call
    json.loads() directly."""
    cleaned = _strip_markdown_wrapper(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Try to repair a truncated array (response cut off by max_tokens)
        repaired = _repair_truncated_array(cleaned)
        if repaired is not None:
            try:
                data = json.loads(repaired)
                return data, ""  # repaired successfully, no error to report
            except json.JSONDecodeError:
                pass
        return None, f"JSON did not parse: {e}"

    if not isinstance(data, list):
        return None, "top-level JSON value was not an array"
    return data, ""


# ── Fact validation + deterministic graph building ─────────────────────────

def build_graph_from_facts(
    facts: list[dict],
    predicate_map: dict[str, str],
):
    """
    Validate each fact against the ontology's predicate set and build an
    rdflib graph deterministically — the model never writes Turtle, so
    syntax errors and inline property redeclaration are both impossible
    here by construction.

    Returns (graph, accepted_count, violations).
    """
    import rdflib
    from rdflib import Graph, Literal, Namespace, RDFS

    WD  = Namespace("http://www.wikidata.org/entity/")
    WDT = Namespace("http://www.wikidata.org/prop/direct/")

    g = Graph()
    g.bind("wd", WD)
    g.bind("wdt", WDT)
    g.bind("rdfs", RDFS)

    violations: list[str] = []
    accepted = 0
    seen_labels: set[str] = set()
    bad_predicates_seen: set[str] = set()

    for i, fact in enumerate(facts):
        if not isinstance(fact, dict):
            violations.append(f"fact #{i} was not a JSON object — skipped")
            continue

        subject = str(fact.get("subject", "")).strip()
        predicate = str(fact.get("predicate", "")).strip()
        obj = str(fact.get("object", "")).strip()
        obj_type = str(fact.get("object_type", "entity")).strip().lower()

        if not subject or not predicate or not obj:
            violations.append(f"fact #{i} was missing subject/predicate/object — skipped")
            continue

        exact_predicate = predicate_map.get(predicate.lower())
        if exact_predicate is None:
            if predicate.lower() not in bad_predicates_seen:
                bad_predicates_seen.add(predicate.lower())
                violations.append(
                    f"predicate '{predicate}' is not in the ontology — "
                    f"valid predicates are: {', '.join(sorted(set(predicate_map.values())))[:300]}"
                )
            continue

        subj_uri = WD[_slugify(subject)]
        if subject not in seen_labels:
            g.add((subj_uri, RDFS.label, Literal(subject, lang="en")))
            seen_labels.add(subject)

        pred_uri = WDT[exact_predicate]

        if obj_type == "literal":
            g.add((subj_uri, pred_uri, Literal(obj)))
        else:
            obj_uri = WD[_slugify(obj)]
            if obj not in seen_labels:
                g.add((obj_uri, RDFS.label, Literal(obj, lang="en")))
                seen_labels.add(obj)
            g.add((subj_uri, pred_uri, obj_uri))

        accepted += 1

    return g, accepted, violations


def _format_feedback(violations: list[str]) -> str:
    if not violations:
        return ""
    issues = "\n".join(f"- {v}" for v in violations[:8])
    return FEEDBACK_BLOCK.format(issues=issues)


# ── Fact generation (one LLM call) ──────────────────────────────────────────

def generate_facts(
    doc_text: str,
    qa_pairs: list[dict],
    ontology_labels: str,
    feedback: str = "",
    temperature: float | None = None,
    subject_hint: str = "",
) -> tuple[str, bool]:
    answered  = [p for p in qa_pairs if not _is_unanswered(p["answer"])]
    n_dropped = len(qa_pairs) - len(answered)
    if n_dropped:
        print(f"[Stage 9/10]   filtered {n_dropped} unanswered QA pair(s)", flush=True)

    # Template used only for length-budgeting in _guard_context — the
    # subject_hint slot is already filled in below, so pass the template
    # with that slot resolved so the char budget accounts for its length.
    template_for_budget = KG_FACTS_PROMPT.format(
        subject_hint=subject_hint,
        ont_labels="{ont_labels}",
        doc="{doc}",
        qa="{qa}",
        feedback="{feedback}",
    )
    doc_text, qa_block = _guard_context(
        doc_text, answered, ontology_labels, template_for_budget
    )

    prompt = KG_FACTS_PROMPT.format(
        subject_hint=subject_hint,
        ont_labels=ontology_labels,
        doc=doc_text,
        qa=qa_block,
        feedback=feedback,
    )
    print(
        f"[Stage 9/10]   generating facts as JSON "
        f"({len(prompt)} char prompt, temperature={temperature}"
        f"{', with retry feedback' if feedback else ''}) …",
        flush=True,
    )

    def _call(max_tokens: int) -> tuple[str, str]:
        return call_llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            guided_json=_FACTS_JSON_SCHEMA,  # kept for vLLM compatibility;
                                              # Ollama silently ignores this,
                                              # which is why _parse_facts no
                                              # longer assumes clean JSON back
            return_finish_reason=True,
        )

    raw, finish = _call(LLM_TOKENS_FACTS)
    # DeepSeek-R1 spends its budget reasoning before ever writing the JSON
    # array; if that reasoning alone exceeds LLM_TOKENS_FACTS, _strip_think
    # returns "" and finish_reason == "length". Without this check that empty
    # string reaches _parse_facts, which reports a generic "JSON did not
    # parse" — indistinguishable from the model genuinely emitting garbage —
    # and the caller retries at the SAME budget, guaranteeing the same
    # failure every attempt and silently dropping the whole chunk. Retry
    # once at a larger budget instead, mirroring call_llm_answer's contract.
    truncated = (finish == "length") or (not raw)
    if truncated:
        print(
            f"[Stage 9/10]   ⚠ fact generation truncated/empty at "
            f"max_tokens={LLM_TOKENS_FACTS} (reasoning likely exceeded budget) "
            f"— retrying at max_tokens={LLM_TOKENS_FACTS_RETRY}",
            flush=True,
        )
        raw, finish = _call(LLM_TOKENS_FACTS_RETRY)
        truncated = (finish == "length") or (not raw)

    print(f"[Stage 9/10]   ✓ got {len(raw)} chars back", flush=True)
    return raw, truncated


def parse_rdf(turtle_str: str):
    """Parse arbitrary Turtle text with rdflib; return (graph, triples) or
    (None, []) on failure. Kept for compatibility with any other caller
    that hands this module raw Turtle directly (e.g. re-parsing the final
    serialized output)."""
    import rdflib
    g = rdflib.Graph()
    try:
        g.parse(data=turtle_str, format="turtle")
        return g, list(g)
    except Exception as e:
        print(f"  [Stage 10] RDF parse failed: {e}")
        return None, []


# ── Entity resolution ──────────────────────────────────────────────────────

def _apply_entity_resolution(
    turtle_str: str, doc_name: str
) -> tuple[str, dict]:
    """
    Resolve all wd: entity URIs in turtle_str to canonical forms.
    Returns (rewritten_turtle, resolution_summary_dict).
    Skips silently if ENTITY_RESOLUTION_ENABLED is False or import fails.
    """
    if not ENTITY_RESOLUTION_ENABLED:
        return turtle_str, {}

    try:
        from stages.canonicalize import resolve_kg_entities, get_entity_resolver
        resolver = get_entity_resolver()
        rewritten, resolution_map = resolve_kg_entities(turtle_str, resolver)

        if resolution_map:
            tiers = {}
            for r in resolution_map.values():
                tiers[r.resolution_tier] = tiers.get(r.resolution_tier, 0) + 1
            print(
                f"[Stage 9/10]   entity resolution: {len(resolution_map)} entities "
                f"— {tiers}",
                flush=True,
            )
        return rewritten, {
            str(uri): {
                "canonical":        r.canonical_form,
                "entity_type":      r.entity_type,
                "resolution_tier":  r.resolution_tier,
                "confidence":       r.confidence,
                "wikidata_qid":     r.wikidata_qid,
            }
            for uri, r in resolution_map.items()
        }
    except Exception as e:
        print(f"[Stage 9/10]   ⚠ entity resolution skipped: {e}", flush=True)
        return turtle_str, {}


# ── Provenance tracking ────────────────────────────────────────────────────

def _attach_provenance(
    g,                  # rdflib.Graph
    doc_name: str,
    confidence: float = 0.80,
) -> "ProvenanceStore":
    """
    Create a ProvenanceRecord for every triple in g and return the store.
    Confidence is uniform at 0.80 (LLM-generated triples from Stage 9);
    the field exists for future calibration.
    """
    from stages.provenance import ProvenanceStore, make_record
    store = ProvenanceStore()
    for s, p, o in g:
        rec = make_record(
            doc_id=doc_name,
            stage="stage9_kg_construction",
            confidence=confidence,
            model=LLM_MODEL,
        )
        store.add(str(s), str(p), str(o), rec)
    return store


# ── Single-chunk extraction (retry loop, unchanged logic, now per-chunk) ───

MAX_ATTEMPTS = 3
RETRY_TEMPERATURE = 0.1


def _run_single_chunk(
    doc_text: str,
    qa_pairs: list[dict],
    ontology_labels: str,
    predicate_map: dict[str, str],
    subject_hint: str,
    chunk_label: str,
):
    """
    Runs the fixed-low-temperature retry loop (unchanged from the
    pre-chunking version) against a single chunk of QA pairs. Returns
    (best_graph_or_None, accepted_count, violations, raw_last_response).
    """
    g, accepted, violations, raw = None, 0, [], ""
    feedback = ""
    best_g, best_accepted, best_clean = None, -1, False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        temp = None if attempt == 1 else RETRY_TEMPERATURE
        raw, truncated = generate_facts(
            doc_text, qa_pairs, ontology_labels,
            feedback=feedback, temperature=temp,
            subject_hint=subject_hint,
        )
        facts, parse_err = _parse_facts(raw)

        if facts is None:
            print(
                f"[Stage 9/10]   [{chunk_label}] attempt {attempt}: {parse_err}"
                f"{' (truncated even after budget retry)' if truncated else ''}",
                flush=True,
            )
            if truncated:
                # generate_facts() already retried once at a bigger budget and
                # still came back empty/cut-off — telling the model to "output
                # valid JSON" won't fix a budget problem, so don't pretend
                # this was a formatting mistake. Retry unchanged (still worth
                # another attempt — reasoning length varies run to run) but
                # without misleading corrective feedback.
                feedback = ""
            else:
                feedback = _format_feedback([
                    f"Your last output could not be read as a JSON array ({parse_err}). "
                    f"Output ONLY a valid JSON array, nothing else."
                ])
            continue

        g, accepted, violations = build_graph_from_facts(facts, predicate_map)
        print(
            f"[Stage 9/10]   [{chunk_label}] attempt {attempt}: {len(facts)} fact(s) "
            f"proposed, {accepted} accepted, {len(violations)} issue(s)",
            flush=True,
        )

        # Prefer a clean (no-violation) attempt over a dirtier one with more
        # raw accepted facts — a later low-temperature retry that fixed every
        # violation is a better result than an earlier noisier one, even if
        # it happened to accept fewer facts.
        is_clean = not violations
        if not best_clean and (is_clean or accepted > best_accepted):
            best_g, best_accepted, best_clean = g, accepted, is_clean

        if not violations:
            break  # clean result — stop here for this chunk

        feedback = _format_feedback(violations)
        if attempt < MAX_ATTEMPTS:
            print(
                f"[Stage 9/10]   [{chunk_label}] retrying with corrective feedback …",
                flush=True,
            )

    return best_g, max(best_accepted, 0), violations, raw


# ── Top-level run ────────────────────────────────────────────────────────────

def run(session, source_doc, doc_name: str, doc_text: str, qa_pairs: list[dict], ontology: str):
    """Build the Path-B KG and stage it to Postgres.

    `session` / `source_doc` (the résumé UUID) are the DB I/O boundary — the
    extraction/graph-building logic below is unchanged; only the final write
    target moved from outputs/kg/{doc}.ttl and outputs/provenance/*.json to the
    graph_entities/graph_relationships and provenance tables.
    """
    predicate_map = _ontology_predicate_map(ontology)
    if not predicate_map:
        print(f"[Stage 9/10] ⚠ ontology has no usable predicates — aborting", flush=True)
        return None, []

    ontology_labels = "\n".join(f"- {label}" for label in sorted(predicate_map.values()))

    # Chunk QA pairs by the explicit "subject" field carried from Stage 1
    # (via Stage 2) instead of sending the whole flat list to one
    # generate_facts() call. Budget each chunk's QA block at roughly a
    # third of MAX_PROMPT_CHARS, leaving room for the doc excerpt,
    # ontology labels, and prompt scaffolding within the same overall
    # ceiling _guard_context() enforces per chunk.
    chunk_char_budget = max(MAX_PROMPT_CHARS // 3, 1000)
    chunks = _chunk_qa_pairs(qa_pairs, max_chunk_chars=chunk_char_budget)
    print(
        f"[Stage 9/10] {len(qa_pairs)} QA pair(s) split into {len(chunks)} chunk(s): "
        f"{[subject for subject, _ in chunks]}",
        flush=True,
    )

    combined_g = None
    total_accepted = 0
    all_violations: list[str] = []
    last_raw = ""

    for idx, (subject, chunk_qa) in enumerate(chunks, start=1):
        chunk_label = f"chunk {idx}/{len(chunks)}: {subject}"
        # No hint for "person" (Stage 1's label for identity/contact
        # facts not tied to one entity) or "_unlabeled" (no confirmed
        # subject) — forcing either onto a single subject would be wrong.
        subject_hint = (
            SUBJECT_HINT_BLOCK.format(subject=subject)
            if subject not in ("person", _UNLABELED_SUBJECT)
            else ""
        )

        g_chunk, accepted, violations, raw = _run_single_chunk(
            doc_text, chunk_qa, ontology_labels, predicate_map,
            subject_hint=subject_hint, chunk_label=chunk_label,
        )
        last_raw = raw
        all_violations.extend(f"[{subject}] {v}" for v in violations)

        if g_chunk is None or len(g_chunk) == 0:
            print(
                f"[Stage 9/10]   ⚠ {chunk_label}: no valid facts after "
                f"{MAX_ATTEMPTS} attempts — skipping this chunk",
                flush=True,
            )
            continue

        combined_g = g_chunk if combined_g is None else (combined_g + g_chunk)
        total_accepted += accepted

    g = combined_g

    if g is None or len(g) == 0:
        print(
            f"[Stage 9/10] No valid facts from any chunk after {MAX_ATTEMPTS} "
            f"attempts each — nothing staged. Last raw model output: "
            f"{last_raw[:200]!r}",
            flush=True,
        )
        return None, []

    turtle_str = g.serialize(format="turtle")
    triples = list(g)

    # ── Entity resolution ────────────────────────────────────────────────
    pre_resolution_turtle = turtle_str
    resolved_turtle, resolution_summary = _apply_entity_resolution(turtle_str, doc_name)

    if resolution_summary:
        g2, triples2 = parse_rdf(resolved_turtle)
        if g2 is not None:
            g, triples = g2, triples2
            turtle_str = resolved_turtle
        else:
            print(
                "[Stage 9/10]   ⚠ entity-resolved Turtle failed to re-parse — "
                "keeping pre-resolution version",
                flush=True,
            )
            turtle_str = pre_resolution_turtle
            resolution_summary = {}
    else:
        turtle_str = resolved_turtle

    # ── Provenance → provenance table ─────────────────────────────────────
    prov_store = _attach_provenance(g, doc_name)
    n_prov = prov_store.save_to_db(session, source_doc)
    print(
        f"[Stage 9/10] {len(triples)} triples ({len(chunks)} chunk(s), "
        f"{total_accepted} fact(s) accepted pre-resolution) — "
        f"{n_prov} provenance row(s) written",
        flush=True,
    )

    # ── Stage the KG (Path B) into graph_entities/graph_relationships ─────
    from db import kg_staging
    kg_staging.stage_graph(session, source_doc, turtle_str)
    print(f"[Stage 9/10] KG staged → graph_entities/graph_relationships", flush=True)
    return g, triples