"""Stage 2 — CQ Answering (verbatim prompt from paper, one CQ per call).

Fix log:
- SUBJECT TAGGING passthrough (root cause #1 follow-up): Stage 1 now emits
  CQs as list[dict] ({"subject": ..., "question": ...}) instead of
  list[str], so that Stage 9/10 can group QA pairs by the entity each
  question is actually about instead of guessing from question phrasing.
  This stage is the pass-through point — it must carry that "subject"
  field into each QA pair unchanged, or the tag never reaches Stage
  9/10 and the whole point of tagging at the source is lost.
- Backward compatibility: if this is ever run against an older CQ file
  that's still a flat list[str] (e.g. output/cqs/*.json generated before
  this change), each item is treated as an untagged question with
  subject="_unlabeled" rather than crashing. This is intentionally
  degraded (not grouped downstream) rather than silently guessed — see
  stage9_10_kg.py's fix log for why guessing was rejected as the
  approach here.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from stages.llm import call_llm_parallel
from config import OUTPUTS_DIR

# Rough chars→tokens divisor for context-window budgeting (bge/Qwen-ish);
# only used to keep an over-long Stage 2 prompt off a small-context endpoint.
_CHARS_PER_TOKEN = 3.5

PROMPT_TEMPLATE = """\
Use the provided document to answer user query. If you don't
know the answer, just say that you don't know, don't try to
make up an answer.

Passage: {doc}

Query: {query}
"""


def _normalize_cq(cq) -> dict:
    """Accept either the current {"subject": ..., "question": ...} shape
    or a legacy plain string, and always return the dict shape. Legacy
    strings get subject="_unlabeled" so they're visibly ungrouped
    downstream instead of silently mis-grouped."""
    if isinstance(cq, dict):
        subject = str(cq.get("subject") or "_unlabeled").strip() or "_unlabeled"
        question = str(cq.get("question", "")).strip()
        return {"subject": subject, "question": question}
    # legacy: plain string
    return {"subject": "_unlabeled", "question": str(cq).strip()}


def answer_cqs(doc_text: str, cqs: list) -> list[dict]:
    # One LLM call per CQ, all independent — fan them out across endpoints
    # instead of a sequential loop. Order is preserved by call_llm_parallel,
    # so each answer still lines up with its CQ (and its carried subject).
    normalized = [_normalize_cq(c) for c in cqs]
    normalized = [c for c in normalized if c["question"]]
    if not normalized:
        return []

    prompts = [
        PROMPT_TEMPLATE.format(doc=doc_text, query=c["question"])
        for c in normalized
    ]
    # The whole résumé rides in every prompt — keep the batch off any endpoint
    # whose per-request window can't hold it (local 8192 vs cluster 32768).
    min_ctx = int(max(len(p) for p in prompts) / _CHARS_PER_TOKEN) + 512
    tasks = [{"prompt": p} for p in prompts]
    answers = call_llm_parallel(tasks, min_ctx=min_ctx)

    return [
        {"question": c["question"], "answer": a, "subject": c["subject"]}
        for c, a in zip(normalized, answers)
    ]


def run(doc_name: str, doc_text: str, cqs: list) -> list[dict]:
    qa_pairs = answer_cqs(doc_text, cqs)
    out = OUTPUTS_DIR / "answers" / f"{doc_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(qa_pairs, indent=2))
    n_subjects = len({p["subject"] for p in qa_pairs})
    print(f"[Stage 2] {len(qa_pairs)} QA pairs across {n_subjects} subject(s) → {out}")
    return qa_pairs