"""
Provenance and confidence tracking for KG triples.

Every accepted triple carries a ProvenanceRecord that stores the source document,
extraction stage, UTC timestamp, confidence score (0–1), and the model version
that produced it.  The ProvenanceStore serialises to / deserialises from JSON so
records survive between pipeline runs and can be queried offline.

Usage
-----
from stages.provenance import ProvenanceStore, make_record

store = ProvenanceStore()
rec   = make_record(doc_id="resume_01", stage="stage9_kg", confidence=0.85,
                    model="llama-3.1-8b-instruct", qa_pair_index=3)
store.add("wd:Alice", "wdt:WorkedAt", "wd:Google", rec)
store.save(Path("outputs/provenance/resume_01.json"))
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ProvenanceRecord:
    doc_id:     str
    stage:      str
    timestamp:  str    # ISO-8601 UTC
    confidence: float  # 0.0 – 1.0
    model:      str
    extra:      dict = field(default_factory=dict)


class ProvenanceStore:
    """
    Maps (subject, predicate, object) string triples → ProvenanceRecord.

    Keys use the null-byte separator (\x00) so URIs containing any printable
    delimiter remain unambiguous.
    """

    _SEP = "\x00"

    def __init__(self) -> None:
        self._records: dict[str, ProvenanceRecord] = {}

    # ── Key helpers ────────────────────────────────────────────────────────

    def _key(self, s: str, p: str, o: str) -> str:
        return f"{s}{self._SEP}{p}{self._SEP}{o}"

    # ── Mutation ───────────────────────────────────────────────────────────

    def add(self, subject: str, predicate: str, obj: str,
            record: ProvenanceRecord) -> None:
        self._records[self._key(subject, predicate, obj)] = record

    def remove(self, subject: str, predicate: str, obj: str) -> None:
        self._records.pop(self._key(subject, predicate, obj), None)

    # ── Query ──────────────────────────────────────────────────────────────

    def get(self, subject: str, predicate: str,
            obj: str) -> Optional[ProvenanceRecord]:
        return self._records.get(self._key(subject, predicate, obj))

    def all_records(self) -> dict[str, ProvenanceRecord]:
        return dict(self._records)

    def __len__(self) -> int:
        return len(self._records)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: asdict(v) for k, v in self._records.items()}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def save_to_db(self, session, document_id) -> int:
        """Persist every (s, p, o) → record to the `provenance` table.

        Replaces the JSON-file save in the DB pipeline: one row per triple, keyed
        to the résumé UUID `document_id`. Returns the number of rows written.
        Commits on success.
        """
        from db.models import Provenance  # local import: keeps this module usable without a DB

        rows = []
        for key, rec in self._records.items():
            subject, predicate, obj = key.split(self._SEP)
            rows.append(
                Provenance(
                    document_id=document_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    stage=rec.stage,
                    confidence=rec.confidence,
                    model=rec.model,
                    extra=rec.extra or None,
                )
            )
        if rows:
            session.add_all(rows)
            session.commit()
        return len(rows)

    @classmethod
    def load(cls, path: Path) -> "ProvenanceStore":
        store = cls()
        payload = json.loads(path.read_text())
        for k, raw in payload.items():
            store._records[k] = ProvenanceRecord(**raw)
        return store

    # ── Merge (for cross-document accumulation) ────────────────────────────

    def merge(self, other: "ProvenanceStore") -> None:
        """Merge another store into this one; existing keys are NOT overwritten."""
        for k, rec in other._records.items():
            self._records.setdefault(k, rec)


# ── Factory ────────────────────────────────────────────────────────────────

def make_record(
    doc_id: str,
    stage: str,
    confidence: float,
    model: str,
    **extra,
) -> ProvenanceRecord:
    """Create a ProvenanceRecord with the current UTC timestamp."""
    return ProvenanceRecord(
        doc_id=doc_id,
        stage=stage,
        timestamp=datetime.now(timezone.utc).isoformat(),
        confidence=min(1.0, max(0.0, confidence)),
        model=model,
        extra=extra or {},
    )
