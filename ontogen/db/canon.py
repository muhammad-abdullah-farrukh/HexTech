"""pgvector-backed EDC canon store — replaces canonicalize.EDCBackend's in-memory
_entries list + embeddings.npy scan.

search_similar() does the cosine-nearest lookup in Postgres (HNSW index) instead
of a full numpy dot-product over every entry; add_entry() persists a new
canonical property immediately (no flush step).
"""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import CanonStoreEntry


def search_similar(session: Session, embedding, top_k: int) -> list[dict]:
    """Return the top_k canon entries nearest `embedding` by cosine distance.

    Each dict mirrors the old entries.json shape (label/definition/turtle/
    source_doc) plus a `cos_score` (1 - cosine distance) so callers can log it
    like the old numpy path did.
    """
    distance = CanonStoreEntry.embedding.cosine_distance(embedding)
    rows = session.execute(
        select(CanonStoreEntry, distance.label("distance"))
        .where(CanonStoreEntry.embedding.isnot(None))
        .order_by(distance)
        .limit(top_k)
    ).all()
    return [
        {
            "id": str(entry.id),
            "label": entry.label,
            "definition": entry.definition,
            "turtle": entry.turtle,
            "source_doc": str(entry.source_doc) if entry.source_doc else None,
            "cos_score": 1.0 - float(dist),
        }
        for entry, dist in rows
    ]


def find_by_label(session: Session, label: str) -> dict | None:
    """Fetch a single canon entry by exact (case-insensitive) label."""
    entry = session.execute(
        select(CanonStoreEntry).where(func.lower(CanonStoreEntry.label) == label.lower())
    ).scalars().first()
    if entry is None:
        return None
    return {
        "id": str(entry.id),
        "label": entry.label,
        "definition": entry.definition,
        "turtle": entry.turtle,
        "source_doc": str(entry.source_doc) if entry.source_doc else None,
    }


def add_entry(
    session: Session,
    label: str,
    definition: str,
    turtle: str,
    embedding,
    source_doc=None,
) -> None:
    """Add a new canonical property (dedup by lowercased label) and commit.

    Refuses to persist entries with empty definition/turtle (e.g. from a
    truncated reasoning-model response) — junk here poisons future EDC
    canonicalization/merge quality for other résumés.
    """
    if not (definition or "").strip() or not (turtle or "").strip():
        print(f"[canon_store] ⚠ refusing to store entry '{label}': empty definition or turtle", flush=True)
        return
    exists = session.execute(
        select(CanonStoreEntry.id).where(func.lower(CanonStoreEntry.label) == label.lower())
    ).first()
    if exists is not None:
        return
    session.add(
        CanonStoreEntry(
            label=label,
            definition=definition,
            turtle=turtle,
            embedding=embedding,
            source_doc=source_doc,
        )
    )
    session.commit()
