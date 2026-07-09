"""
Stage 5 — Embed Wikidata property descriptions with bge-small-en (run ONCE).

Manual step (after stage 4):
    python stages/stage5_embed_wikidata.py

Reads pid/label/description from the wikidata_properties table and writes each
row's `embedding` (vector(384)) column. Replaces the old .npy save; the
pgvector HNSW index then serves Stage 6's nearest-neighbour lookup directly.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBED_MODEL, EMBED_DEVICE, DATABASE_URL


def run():
    print(f"[Stage 5] Model:  {EMBED_MODEL}")
    if not DATABASE_URL:
        print("[Stage 5] ✗ DATABASE_URL is not set — cannot read/update wikidata_properties.")
        sys.exit(1)

    from sqlalchemy import select
    from db.models import WikidataProperty
    from db.session import make_session_factory

    factory = make_session_factory(DATABASE_URL)

    # ── 1. Load rows from the DB ──────────────────────────────────────────
    with factory() as session:
        rows = session.execute(
            select(WikidataProperty.pid, WikidataProperty.label, WikidataProperty.description)
        ).all()
    if not rows:
        print("[Stage 5] ✗ wikidata_properties is empty — run stage 4 first.")
        sys.exit(1)
    print(f"[Stage 5]   Loaded {len(rows)} properties from wikidata_properties")

    pids = [r.pid for r in rows]
    descriptions = [(r.description or r.label) for r in rows]

    # ── 2. Load model + smoke test ────────────────────────────────────────
    print(f"[Stage 5] Loading SentenceTransformer model '{EMBED_MODEL}' …")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    test_vec = model.encode(["test"], normalize_embeddings=True, convert_to_numpy=True)
    if test_vec.shape[1] != 384:
        print(f"[Stage 5] ⚠ Unexpected embedding dim {test_vec.shape[1]} (expected 384)")

    # ── 3. Encode ─────────────────────────────────────────────────────────
    print(f"[Stage 5] Encoding {len(descriptions)} descriptions (batch_size=256) …")
    embeddings = model.encode(
        descriptions,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    nan_count = int(np.isnan(embeddings).sum()) + int(np.isinf(embeddings).sum())
    if nan_count:
        print(f"[Stage 5] ⚠ {nan_count} NaN/Inf values in embeddings — model output may be corrupt")

    # ── 4. Write embeddings back per pid ──────────────────────────────────
    print(f"[Stage 5] Updating {len(pids)} embedding column(s) …")
    with factory() as session:
        for pid, vec in zip(pids, embeddings):
            session.execute(
                WikidataProperty.__table__.update()
                .where(WikidataProperty.pid == pid)
                .values(embedding=vec.tolist())
            )
        session.commit()

    print(f"[Stage 5] ✓ Done. Embedded {len(pids)} wikidata_properties rows.")


if __name__ == "__main__":
    run()
