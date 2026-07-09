"""Wikidata property nearest-neighbour lookup — replaces the .npy brute-force
scan in Stage 6.

top_k_candidates() runs the cosine search in Postgres (HNSW index), returning
the same {pid, label, description} dicts Stage 6's validator already expects.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import WikidataProperty


def top_k_candidates(session: Session, embedding, k: int) -> list[dict]:
    """Return the k Wikidata properties nearest `embedding`, in rank order."""
    distance = WikidataProperty.embedding.cosine_distance(embedding)
    rows = session.execute(
        select(WikidataProperty, distance.label("distance"))
        .where(WikidataProperty.embedding.isnot(None))
        .order_by(distance)
        .limit(k)
    ).all()
    return [
        {
            "pid": prop.pid,
            "label": prop.label,
            "description": prop.description or "",
            "cos_score": 1.0 - float(dist),
        }
        for prop, dist in rows
    ]
