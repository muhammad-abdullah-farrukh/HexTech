"""
Stage 6 — Embedding nearest-neighbour retrieval (top-k) + LLM yes/no validation.

- Top-k retrieval (TOP_K_CANDIDATES = 3 or 5) instead of top-1; validates in
  rank order and returns the first confirmed match.
- Deduplicates extracted relations by property name before embedding — each
  unique name is validated exactly once and the result is broadcast back to
  all occurrences.
- max_tokens=1000 on the validation LLM call (was 10, itself a bump from 5
  for the confidence digits — but the server returns reasoning in a
  separate `reasoning_content` field that still counts against max_tokens
  before `content` is generated, so a small cap starved the answer
  entirely; measured completion_tokens=419-577 for real candidate pairs,
  ~1.7x margin over the max observed).
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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from stages.llm import call_llm
from db import wikidata
from config import (
    EMBED_MODEL,
    EMBED_DEVICE,
    TOP_K_CANDIDATES,
    OUTPUTS_DIR,
    LLM_MODEL,
    LLM_ENDPOINTS,
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
        _model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)


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


def _retrieve_candidates(session, extracted: dict, idx: int, total: int):
    """Embed the property description and fetch its top-k Wikidata candidates.
    Fast (ms), uses the DB session + embedding model — kept OFF the worker
    threads (one shared session isn't thread-safe; the embed model is loaded
    once). Returns the candidates list."""
    _load()
    t0 = time.time()
    query_vec = _embed(extracted["description"])
    candidates = _top_k(session, query_vec, TOP_K_CANDIDATES)
    labels = [c["label"] for c, _ in candidates]
    print(
        f"  [{idx}/{total}] '{extracted['property']}' — top-{TOP_K_CANDIDATES} "
        f"in {time.time()-t0:.2f}s → {labels}",
        flush=True,
    )
    return candidates


def _validate_candidates(
    extracted: dict,
    candidates: list,
    idx: int,
    total: int,
    endpoint: dict | None = None,
) -> tuple[dict | None, list[dict]]:
    """LLM-validate pre-retrieved candidates in rank order, returning the first
    confirmed match (short-circuit) and the per-candidate log entries.

    Pure LLM work — no DB, no embedding — so it's safe to run concurrently for
    different properties. Candidates WITHIN a property stay sequential because
    of the first-accept short-circuit; `endpoint` pins this property's calls to
    one server. Returns (matched_candidate_or_None, log_entries)."""
    log_entries: list[dict] = []
    p1 = f"{extracted['property']}: {extracted['description']}"

    for rank, (candidate, cos_score) in enumerate(candidates, start=1):
        p2     = f"{candidate['label']}: {candidate['description']}"
        prompt = VALIDATE_PROMPT.format(p1=p1, p2=p2)

        t_call = time.time()
        try:
            raw_answer = call_llm(prompt, max_tokens=1000, endpoint=endpoint)
        except Exception as e:
            print(f"  [{idx}/{total}]   ✗ LLM call FAILED: {type(e).__name__}: {e}", flush=True)
            raise
        t_done = time.time()

        said_yes, confidence = _parse_validation(raw_answer)
        accepted = said_yes and confidence >= ACCEPT_CONFIDENCE_THRESHOLD
        print(
            f"  [{idx}/{total}]   rank {rank} '{candidate['label']}' "
            f"({candidate['pid']}) cos={cos_score:.3f} → "
            f"{'accepted' if accepted else 'rejected'} "
            f"({t_done-t_call:.2f}s, conf={confidence:.2f})",
            flush=True,
        )

        log_entries.append({
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "relation":        extracted["property"],
            "description":     extracted.get("description", ""),
            "candidate_label": candidate["label"],
            "candidate_pid":   candidate["pid"],
            "rank":            rank,
            "cos_score":       round(cos_score, 4),
            "llm_raw":         raw_answer,
            "llm_confidence":  confidence,
            "accepted":        accepted,
            "model":           LLM_MODEL,
        })

        if accepted:
            return candidate, log_entries

    print(f"  [{idx}/{total}]   no match in top-{TOP_K_CANDIDATES}", flush=True)
    return None, log_entries


def run(session, relations: list[dict], doc_name: str = "unknown") -> list[dict]:
    _load()

    # Dedup by normalised property name — validate each unique name exactly once.
    unique: dict[str, dict] = {}
    for rel in relations:
        key = rel["property"].strip().lower()
        if key not in unique:
            unique[key] = rel

    unique_list = list(unique.values())
    total = len(unique_list)
    print(
        f"[Stage 6] {len(relations)} relations → {total} unique "
        f"property name(s) to validate",
        flush=True,
    )

    # Phase 1 — retrieval (sequential: shares the one DB session + embed model,
    # both fast). Build each property's candidate list up front.
    retrieved = [
        (rel, _retrieve_candidates(session, rel, i, total))
        for i, rel in enumerate(unique_list, start=1)
    ]

    # Phase 2 — LLM validation (the slow part) fanned out across endpoints, one
    # property per task, round-robin endpoint. Properties are independent; the
    # short-circuit over a property's own candidates stays inside its task.
    total_slots = sum(max(1, int(e.get("slots", 1))) for e in LLM_ENDPOINTS)
    val_results: list = [None] * total
    with ThreadPoolExecutor(max_workers=max(1, min(total_slots, total or 1))) as pool:
        futs = {
            pool.submit(
                _validate_candidates, rel, cands, i + 1, total,
                LLM_ENDPOINTS[i % len(LLM_ENDPOINTS)],
            ): i
            for i, (rel, cands) in enumerate(retrieved)
        }
        for f in futs:
            val_results[futs[f]] = f.result()

    log_entries: list[dict] = []
    unique_matches: dict[str, dict | None] = {}
    for (rel, _cands), (match, entries) in zip(retrieved, val_results):
        log_entries.extend(entries)
        key = rel["property"].strip().lower()
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