"""
Stage 6 — Embedding nearest-neighbour retrieval (top-k) + LLM yes/no validation.

- Top-k retrieval (TOP_K_CANDIDATES = 3 or 5) instead of top-1; validates in
  rank order and returns the first confirmed match.
- Deduplicates extracted relations by property name before embedding — each
  unique name is validated exactly once and the result is broadcast back to
  all occurrences.
- The validation LLM call goes through call_llm_answer with a reasoning-sized
  budget (config.LLM_TOKENS_CLASSIFY, retry at LLM_TOKENS_RETRY). deepseek-r1
  emits <think> before its "yes 87"/"no 12"; the old max_tokens=10 was consumed
  entirely by reasoning, returned "", and was misread as a rejection — which is
  why every candidate was rejected and Path B produced no matches. A truncated
  answer is now logged (truncated=True) and skipped as INDETERMINATE, never
  counted as a "no".
- Structured mapping log written to outputs/logs/stage6_{doc_name}.jsonl —
  one line per (relation, candidate) pair, with accepted flag and scores.

Fix log:
- The old VALIDATE_PROMPT only asked "are these two properties similar in
  an ontology", which a small instruct model will answer "yes" to almost
  any time there's lexical overlap — observed accepting "current location"
  → "LocatedInOrNextToBodyOfWater" (cos 0.88), "email address" →
  "Addressee", "phone number" → "EmergencyPhoneNumber", "university name"
  → "CarnegieClassificationOfInstitutionsOfHigherEducation". All four are
  topically adjacent but factually wrong matches, and the old prompt had no
  way to tell "similar" apart from "correct".
- VALIDATE_PROMPT now asks specifically whether assigning candidate
  property 2 would be ACCURATE for fact 1, explicitly calls out that
  lexical overlap is not sufficient, and gives contrastive examples drawn
  from the actual false accepts above.
- The model is now also asked to append a 0-100 confidence score (same
  "yes 87" / "no 12" format already used in canonicalize.py's EDC verify
  step), and acceptance requires both "yes" AND confidence above
  ACCEPT_CONFIDENCE_THRESHOLD — a bare "yes" with no real conviction no
  longer auto-accepts.
- This does not fix cases where the *correct* Wikidata property simply
  isn't present in your filtered properties set (data/wikidata/
  properties_filtered.json) — if that's missing entries like a generic
  "location" or "email address" property, tightening the prompt will
  correctly make those fall through to "no match" instead of accepting a
  wrong one, but it can't produce a match that doesn't exist in the data.
  Worth checking that file if rejections go up a lot after this change.
"""
import json, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from stages.llm import call_llm_answer
from db import wikidata
from config import (
    EMBED_MODEL,
    TOP_K_CANDIDATES,
    OUTPUTS_DIR,
    LLM_MODEL,
    LLM_TOKENS_CLASSIFY,
    LLM_TOKENS_RETRY,
)

VALIDATE_PROMPT = """\
You are checking whether a candidate Wikidata property is the CORRECT
property to use for a specific extracted fact — not merely whether the two
sound topically related.

Property 1 describes the fact as it was actually extracted from the source
document.
Property 2 is a candidate Wikidata property retrieved by embedding
similarity, which can surface plausible-sounding but wrong matches.

Answer "yes" ONLY if assigning Property 2 to Property 1's fact would be
accurate and unambiguous — i.e. a person reading the resulting triple would
not be misled about what the value actually represents.

Answer "no" if Property 2 has a different real-world meaning than Property
1, even when the words overlap. For example:
- "current location" is NOT "location next to a body of water"
- "email address" is NOT "addressee of a letter"
- "phone number" is NOT "emergency phone number"
- "university name" is NOT "Carnegie classification of higher ed institutions"
Lexical overlap between the two property descriptions is not evidence of a
correct match by itself.

Append a confidence integer 0–100 after your answer, reflecting how sure
you are this is the correct property (not how similar the words look).

Format exactly: "yes 87" or "no 12"

Property 1: {p1}
Property 2: {p2}

Answer:"""

# A bare "yes" with low stated confidence no longer auto-accepts.
ACCEPT_CONFIDENCE_THRESHOLD = 0.65

_model = None


def _load():
    global _model
    if _model is None:
        print("[Stage 6]   Loading embedding model …", flush=True)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)


def _embed(text: str) -> np.ndarray:
    vec = _model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vec[0]


def _top_k(session, vec: np.ndarray, k: int) -> list[tuple[dict, float]]:
    """Return (property, cosine_score) pairs for the k nearest Wikidata
    properties — now a pgvector lookup (db.wikidata) instead of a .npy scan."""
    candidates = wikidata.top_k_candidates(session, vec.tolist(), k)
    return [(c, c["cos_score"]) for c in candidates]


def _parse_validation(raw_answer: str) -> tuple[bool, float]:
    """
    Parse the "yes 87" / "no 12" format into (passes_yes_no, confidence 0-1).
    A parse failure is never treated as an acceptance.
    """
    m = re.match(r"^(yes|no)\s*(\d{1,3})?", raw_answer.strip().lower())
    if not m:
        return False, 0.0
    said_yes  = m.group(1) == "yes"
    raw_score = int(m.group(2)) if m.group(2) else (70 if said_yes else 30)
    confidence = min(100, max(0, raw_score)) / 100.0
    return said_yes, confidence


def validate_match(
    session,
    extracted: dict,
    idx: int,
    total: int,
    log_entries: list[dict],
) -> dict | None:
    """
    Try top-k nearest Wikidata candidates in rank order.
    Appends one log entry per candidate to log_entries.
    Returns the first confirmed match dict, or None.
    """
    _load()

    t0 = time.time()
    print(f"  [{idx}/{total}] '{extracted['property']}' — embedding …", flush=True)
    query_vec = _embed(extracted["description"])
    t1 = time.time()
    print(f"  [{idx}/{total}]   embed done in {t1-t0:.2f}s", flush=True)

    candidates = _top_k(session, query_vec, TOP_K_CANDIDATES)
    t2 = time.time()
    labels = [c["label"] for c, _ in candidates]
    print(f"  [{idx}/{total}]   top-{TOP_K_CANDIDATES} in {t2-t1:.2f}s → {labels}", flush=True)

    p1 = f"{extracted['property']}: {extracted['description']}"

    for rank, (candidate, cos_score) in enumerate(candidates, start=1):
        p2     = f"{candidate['label']}: {candidate['description']}"
        prompt = VALIDATE_PROMPT.format(p1=p1, p2=p2)

        print(
            f"  [{idx}/{total}]   rank {rank}: validating '{candidate['label']}' "
            f"({candidate['pid']}) cos={cos_score:.3f} …",
            flush=True,
        )
        t_call = time.time()
        try:
            raw_answer, truncated = call_llm_answer(
                prompt, LLM_TOKENS_CLASSIFY, retry_budget=LLM_TOKENS_RETRY
            )
        except Exception as e:
            print(f"  [{idx}/{total}]   ✗ LLM call FAILED: {type(e).__name__}: {e}", flush=True)
            raise
        t_done = time.time()

        log_entry = {
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "relation":        extracted["property"],
            "description":     extracted.get("description", ""),
            "candidate_label": candidate["label"],
            "candidate_pid":   candidate["pid"],
            "rank":            rank,
            "cos_score":       round(cos_score, 4),
            "llm_raw":         raw_answer,
            "model":           LLM_MODEL,
        }

        # A truncated/empty answer means the model was still reasoning when it hit
        # the token cap — it is INDETERMINATE, not a "no". Flag it, skip this
        # candidate, and move on rather than silently rejecting (the exact bug that
        # left Path B producing zero matches).
        if truncated:
            print(
                f"  [{idx}/{total}]   ⚠ {t_done-t_call:.2f}s — INDETERMINATE "
                f"(truncated/empty; budget too tight): {repr(raw_answer)}",
                flush=True,
            )
            log_entries.append({**log_entry, "llm_confidence": 0.0,
                                "accepted": False, "truncated": True})
            continue

        said_yes, confidence = _parse_validation(raw_answer)
        accepted = said_yes and confidence >= ACCEPT_CONFIDENCE_THRESHOLD
        print(
            f"  [{idx}/{total}]   ✓ {t_done-t_call:.2f}s — "
            f"{'accepted' if accepted else 'rejected'} "
            f"(said_yes={said_yes}, conf={confidence:.2f}): {repr(raw_answer)}",
            flush=True,
        )

        log_entries.append({**log_entry, "llm_confidence": confidence,
                            "accepted": accepted, "truncated": False})

        if accepted:
            print(
                f"  [{idx}/{total}]   matched at rank {rank}: "
                f"{candidate['label']} ({candidate['pid']}) conf={confidence:.2f}",
                flush=True,
            )
            return candidate

    print(f"  [{idx}/{total}]   no match in top-{TOP_K_CANDIDATES}", flush=True)
    return None


def run(session, relations: list[dict], doc_name: str = "unknown") -> list[dict]:
    _load()

    # Dedup by normalised property name — validate each unique name exactly once.
    unique: dict[str, dict] = {}
    for rel in relations:
        key = rel["property"].strip().lower()
        if key not in unique:
            unique[key] = rel

    unique_list = list(unique.values())
    print(
        f"[Stage 6] {len(relations)} relations → {len(unique_list)} unique "
        f"property name(s) to validate",
        flush=True,
    )

    log_entries: list[dict] = []

    # Validate unique relations
    unique_matches: dict[str, dict | None] = {}
    total = len(unique_list)
    for i, rel in enumerate(unique_list, start=1):
        match = validate_match(session, rel, i, total, log_entries)
        key   = rel["property"].strip().lower()
        unique_matches[key] = match
        status = f"✓ {match['label']} ({match['pid']})" if match else "✗ no match"
        print(f"  [{rel['property']}] → {status}", flush=True)

    # Persist structured mapping log
    log_path = OUTPUTS_DIR / "logs" / f"stage6_{doc_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        for entry in log_entries:
            fh.write(json.dumps(entry) + "\n")
    accepted_count = sum(1 for e in log_entries if e["accepted"])
    print(
        f"[Stage 6] mapping log → {log_path}  "
        f"({accepted_count}/{len(log_entries)} accepted)",
        flush=True,
    )

    # Expand back to original list order (one result per original relation entry)
    results = []
    for rel in relations:
        key = rel["property"].strip().lower()
        results.append({"extracted": rel, "wikidata_match": unique_matches[key]})

    return results