"""Deterministic pre-filter for EDC Step 4 — decides most candidate pairs
without calling the reasoning LLM.

History: the first attempt here was a pretrained NLI cross-encoder scoring
bidirectional entailment. It failed calibration completely — merge and
reject verdicts both scored near zero with heavy overlap (see
scripts/calibrate_thresholds.py, now removed) — because (a) entailment
("does A logically imply B") is a stricter, different relationship than
"do A and B mean the same thing", and (b) synthetic "label: definition"
input is out-of-distribution for a model trained on natural sentence pairs.
A follow-up check showed even a bare bge-cosine floor isn't safe either: on
70 backfilled deepseek verdicts, 'employer' vs 'current employer' scored
cos=0.958 — HIGHER than several genuine merges — because label-containment
pairs differing by a meaning-changing modifier ("current", "location of")
look identical, cosine-wise, to pairs differing by a meaning-preserving
value-type suffix ("number", "address") that genuinely are the same thing.

So this gate is two tiers, in order:

  1. Token containment + BENIGN_SUFFIXES (config.py): if one label's words
     are a strict subset of the other's, the *extra* words decide it — if
     they're all "benign" (describe value format, not meaning), MERGE; any
     other extra word, REJECT. Zero training; a short, auditable word list.
  2. bge cosine bands (only when containment doesn't apply): thresholds come
     from the verify_thresholds table (latest row = current best evidence),
     recomputed only once enough data justifies moving them — see
     scripts/recompute_thresholds.py. Falls back to config.VERIFY_TAU_HI/LO
     if that table is empty (fresh install, no accumulated verdicts yet).

Anything neither tier can decide ESCALATES — the caller falls back to the
existing _llm_verify(). Every decision, whichever tier made it, should be
logged by the caller to verify_verdicts (source='gate') and to the EDC jsonl
log — this module only decides, it doesn't persist anything itself.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BENIGN_SUFFIXES, VERIFY_TAU_HI, VERIFY_TAU_LO

CONTAINMENT, COSINE, ESCALATE = "containment", "cosine", "escalate"


def _tokens(label: str) -> set[str]:
    """camelCase/whitespace/punctuation-split, lowercased word set."""
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    return set(re.findall(r"[a-z]+", s.lower()))


def containment_extra(label_a: str, label_b: str) -> set[str] | None:
    """If one label's tokens are a strict subset of the other's, return the
    extra (symmetric-difference) words; else None (containment doesn't apply)."""
    ta, tb = _tokens(label_a), _tokens(label_b)
    if ta and tb and (ta < tb or tb < ta):
        return ta ^ tb
    return None


def get_live_thresholds(session_factory) -> tuple[float, float, str]:
    """Latest (tau_hi, tau_lo) from verify_thresholds, or the config.py
    fallback if no row exists yet. Third element is the source, for logging."""
    from sqlalchemy import select
    from db.models import VerifyThreshold

    with session_factory() as session:
        row = session.execute(
            select(VerifyThreshold).order_by(VerifyThreshold.computed_at.desc()).limit(1)
        ).scalars().first()
    if row is not None:
        return row.tau_hi, row.tau_lo, "verify_thresholds"
    return VERIFY_TAU_HI, VERIFY_TAU_LO, "config_default"


def decide(
    label_a: str, label_b: str, cos_score: float, session_factory,
) -> tuple[str, bool | None, float | None]:
    """Return (method, accepted, cos_score_used).

    method ∈ {CONTAINMENT, COSINE, ESCALATE}; accepted is None when method is
    ESCALATE (caller must ask the LLM). cos_score_used echoes the input
    cosine for logging convenience (already computed by top-k retrieval —
    this function never re-embeds).
    """
    extra = containment_extra(label_a, label_b)
    if extra is not None:
        accepted = extra.issubset(BENIGN_SUFFIXES)
        return CONTAINMENT, accepted, cos_score

    tau_hi, tau_lo, _source = get_live_thresholds(session_factory)
    if cos_score >= tau_hi:
        return COSINE, True, cos_score
    if cos_score <= tau_lo:
        return COSINE, False, cos_score
    return ESCALATE, None, cos_score
