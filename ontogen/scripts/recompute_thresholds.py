"""Recompute VERIFY_TAU_HI / VERIFY_TAU_LO from accumulated verify_verdicts,
inserting a new verify_thresholds row ONLY if both gates pass:

  1. Sample-size gate: at least config.MIN_SAMPLES_FOR_RECOMPUTE non-
     containment pairs (containment-handled pairs are excluded from this
     computation entirely — they're decided by the separate token/
     BENIGN_SUFFIXES tier in stages/verify_gate.py, not by cosine).
  2. Precision gate: the selected tau_hi/tau_lo must hit merge-precision and
     reject-precision >= PRECISION_BAR on the full accumulated set (not just
     the newest data) — same bar used in the original manual calibration.

If either gate fails, nothing is written — the current (possibly
config.py-fallback) thresholds keep governing until more/better data exists.
This is the mechanism, run periodically (cron/schedule, or by hand after
processing a batch of résumés), that lets the system's auto-decide coverage
grow over time without ever moving on a whim of a handful of examples.

Threshold search: smallest tau_hi (maximizes MERGE coverage) that still
clears merge-precision >= PRECISION_BAR; largest tau_lo (maximizes REJECT
coverage) that still clears reject-precision >= PRECISION_BAR.

Run:  python ontogen/scripts/recompute_thresholds.py   (DATABASE_URL set)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select

from config import DATABASE_URL, MIN_SAMPLES_FOR_RECOMPUTE
from db.session import make_session_factory
from db.models import VerifyVerdict, VerifyThreshold
from stages.verify_gate import containment_extra

PRECISION_BAR = 0.95
GRID = [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0.50 .. 0.99


def main() -> None:
    if not DATABASE_URL:
        print("✗ DATABASE_URL is not set"); sys.exit(1)

    session_factory = make_session_factory(DATABASE_URL)
    with session_factory() as session:
        rows = session.execute(select(VerifyVerdict)).scalars().all()

    non_containment = [v for v in rows if containment_extra(v.label_a, v.label_b) is None]
    n = len(non_containment)
    print(f"{len(rows)} total verdicts, {len(rows) - n} containment-handled "
          f"(excluded), {n} eligible for cosine-band recomputation")

    if n < MIN_SAMPLES_FOR_RECOMPUTE:
        print(f"✗ GATE FAILED: {n} < MIN_SAMPLES_FOR_RECOMPUTE ({MIN_SAMPLES_FOR_RECOMPUTE}). "
              f"Leaving current thresholds unchanged — keep processing résumés.")
        return

    from sentence_transformers import SentenceTransformer
    import numpy as np
    from config import EMBED_MODEL

    model = SentenceTransformer(EMBED_MODEL)
    texts_a = [f"{v.label_a}: {v.definition_a}" for v in non_containment]
    texts_b = [f"{v.label_b}: {v.definition_b}" for v in non_containment]
    emb_a = model.encode(texts_a, normalize_embeddings=True, convert_to_numpy=True)
    emb_b = model.encode(texts_b, normalize_embeddings=True, convert_to_numpy=True)
    cos = np.sum(emb_a * emb_b, axis=1)
    labels = [v.accepted for v in non_containment]

    best_hi, best_hi_prec = None, None
    for t in GRID:  # ascending — first one that clears the bar is the smallest (max coverage)
        merged = [l for c, l in zip(cos, labels) if c >= t]
        if not merged:
            continue
        prec = sum(merged) / len(merged)
        if prec >= PRECISION_BAR:
            best_hi, best_hi_prec = t, prec
            break

    best_lo, best_lo_prec = None, None
    for t in reversed(GRID):  # descending — first one that clears the bar is the largest
        rejected = [l for c, l in zip(cos, labels) if c <= t]
        if not rejected:
            continue
        prec = sum(1 for l in rejected if not l) / len(rejected)
        if prec >= PRECISION_BAR:
            best_lo, best_lo_prec = t, prec
            break

    if best_hi is None or best_lo is None:
        print(f"✗ GATE FAILED: no threshold on the full {n}-pair set clears "
              f"merge/reject-precision >= {PRECISION_BAR}. Leaving current "
              f"thresholds unchanged.")
        return

    with session_factory() as session:
        session.add(VerifyThreshold(
            tau_hi=best_hi, tau_lo=best_lo, n_pairs_used=n,
            merge_precision=best_hi_prec, reject_precision=best_lo_prec,
        ))
        session.commit()

    print(f"✓ GATE PASSED — new thresholds recorded: tau_hi={best_hi} "
          f"(merge-prec {best_hi_prec:.3f}), tau_lo={best_lo} "
          f"(reject-prec {best_lo_prec:.3f}), backed by {n} pairs")


if __name__ == "__main__":
    main()
