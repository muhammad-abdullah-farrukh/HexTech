"""Backfill verify_verdicts from edc_canon pipeline_runs checkpoints.

Recovers the deepseek verify judgments already paid for during checkpointed
EDC runs and materializes them as labeled (pair → accepted) rows, so the
cross-encoder has calibration/training data before it is ever enabled.

Sources per checkpointed relation (asdict(RelationCanonResult)):
  - a merged pair:   (original vs canonical_label/description, accepted=True)
  - rejected pairs:  (original vs each rejected_candidates entry, accepted=False)
    rejected_candidates only stores the candidate's *label*, so its definition
    is looked up from canon_store; pairs whose canon entry no longer exists
    (e.g. deleted in the empty-row cleanup) are skipped and counted.

Idempotent: an exact (label_a, definition_a, label_b, accepted, source) match
already in the table is not inserted again.

Deliberately NOT used as a source: the stage6_*.jsonl logs — files written
before today's max_tokens fix have empty llm_raw (truncated responses whose
"no" was a parser fallback, not a judgment), i.e. garbage labels.

Run:  python ontogen/scripts/backfill_verdicts.py   (DATABASE_URL must be set)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, text

from config import DATABASE_URL
from db.session import make_session_factory
from db import canon as canon_db
from db.models import VerifyVerdict
from db.runs import EDC_CANON

SOURCE = "deepseek"


def _exists(session, label_a, definition_a, label_b, accepted) -> bool:
    return session.execute(
        select(VerifyVerdict.id).where(
            VerifyVerdict.label_a == label_a,
            VerifyVerdict.definition_a == definition_a,
            VerifyVerdict.label_b == label_b,
            VerifyVerdict.accepted == accepted,
            VerifyVerdict.source == SOURCE,
        )
    ).first() is not None


def main() -> None:
    if not DATABASE_URL:
        print("✗ DATABASE_URL is not set"); sys.exit(1)

    session_factory = make_session_factory(DATABASE_URL)
    inserted = skipped_dupe = skipped_no_def = 0

    with session_factory() as session:
        rows = session.execute(
            text("SELECT document_id, output FROM pipeline_runs WHERE stage = :s"),
            {"s": EDC_CANON},
        ).all()

        for resume_id, checkpoint in rows:
            for result in (checkpoint or {}).values():
                label_a = result["original_property"]
                def_a = result["definition"]
                if not (def_a or "").strip():
                    skipped_no_def += 1
                    continue

                pairs = []  # (label_b, definition_b, accepted, confidence)
                if result["was_merged"] and result["canonical_label"]:
                    pairs.append((
                        result["canonical_label"],
                        result.get("canonical_description") or "",
                        True,
                        result["confidence"],
                    ))
                for rej in result.get("rejected_candidates", []):
                    entry = canon_db.find_by_label(session, rej["label"])
                    if entry is None or not (entry["definition"] or "").strip():
                        skipped_no_def += 1
                        continue
                    pairs.append((rej["label"], entry["definition"], False, rej["confidence"]))

                for label_b, def_b, accepted, confidence in pairs:
                    if not (def_b or "").strip():
                        skipped_no_def += 1
                        continue
                    if _exists(session, label_a, def_a, label_b, accepted):
                        skipped_dupe += 1
                        continue
                    session.add(VerifyVerdict(
                        label_a=label_a, definition_a=def_a,
                        label_b=label_b, definition_b=def_b,
                        accepted=accepted, confidence=confidence,
                        source=SOURCE, resume_id=resume_id,
                    ))
                    inserted += 1
        session.commit()

    print(f"✓ backfill done: {inserted} inserted, "
          f"{skipped_dupe} duplicates skipped, {skipped_no_def} skipped (missing definition)")


if __name__ == "__main__":
    main()
