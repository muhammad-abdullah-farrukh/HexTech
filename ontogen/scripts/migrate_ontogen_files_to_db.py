"""
One-time migration: seed Ontogen's new Postgres tables from the on-disk files.

Run ONCE (then this script can be discarded or kept as a record):

    python scripts/migrate_ontogen_files_to_db.py

Seeds three corpus-wide stores:
  1. data/gazetteers/*.json          → gazetteers   (source='static', with QIDs)
  2. data/canon_store/entries.json   → canon_store   (re-embedding each definition)
  3. data/wikidata/properties_filtered.json → wikidata_properties (re-embedding descriptions)

Why re-embed rather than copy .npy: there are no embedding .npy files on disk
(none were ever committed), so the vectors are recomputed with bge-small-en
(EMBED_MODEL), which is the same model Stage 6 / EDC query with — dimensions
match the vector(384) columns.

Idempotent: gazetteers' static rows are cleared first, canon entries dedup by
label, wikidata properties upsert by pid.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ontogen root

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import DATABASE_URL, EMBED_MODEL, GAZETTEER_DIR, CANON_STORE_DIR, WIKIDATA_FILTERED
from db.models import CanonStoreEntry, Gazetteer, WikidataProperty
from db.session import make_session_factory

# gazetteer file stem → entity_type stored in the gazetteers table (singular,
# matching the values ResumeEntityResolver / structured_to_relations use).
GAZ_TYPES = {
    "companies": "company",
    "universities": "university",
    "certifications": "certification",
    "skills": "skill",
    "job_titles": "job_title",
}


def _embedder():
    from sentence_transformers import SentenceTransformer
    print(f"[seed] loading embedding model {EMBED_MODEL} …", flush=True)
    return SentenceTransformer(EMBED_MODEL)


def _embed_all(model, texts: list[str]):
    return model.encode(
        texts, batch_size=256, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )


def seed_gazetteers(session) -> int:
    session.execute(delete(Gazetteer).where(Gazetteer.source == "static"))
    rows = []
    for stem, entity_type in GAZ_TYPES.items():
        path = GAZETTEER_DIR / f"{stem}.json"
        if not path.exists():
            print(f"[seed]   ⚠ missing {path.name}, skipping")
            continue
        raw = json.loads(path.read_text())
        # Fail loudly on shape drift rather than silently seeding 0 rows: every
        # gazetteer file is expected to have a non-empty "aliases" dict (QIDs are
        # optional — only companies/universities carry them).
        aliases = raw.get("aliases")
        if not isinstance(aliases, dict) or not aliases:
            raise ValueError(
                f"{path.name}: expected a non-empty 'aliases' object, got "
                f"{type(aliases).__name__} (top-level keys: {list(raw)}). "
                f"Fix the key mapping in seed_gazetteers() if the format changed."
            )
        qids = raw.get("wikidata_qid") or {}
        for alias, canonical in aliases.items():
            rows.append(
                {
                    "entity_type": entity_type,
                    "alias": alias,
                    "canonical": canonical,
                    "wikidata_qid": qids.get(canonical),
                    "source": "static",
                }
            )
    if rows:
        session.execute(pg_insert(Gazetteer), rows)
    session.commit()
    print(f"[seed] gazetteers: {len(rows)} alias row(s)")
    return len(rows)


def seed_canon_store(session, model) -> int:
    path = CANON_STORE_DIR / "entries.json"
    if not path.exists():
        print(f"[seed]   ⚠ {path} not found, skipping canon store")
        return 0
    entries = json.loads(path.read_text())
    if not entries:
        return 0

    # source_doc in the JSON is a filename string ("yourfile"), not a résumé
    # UUID, so it maps to NULL — the provenance pointer is simply unknown here.
    definitions = [e.get("definition") or e.get("label", "") for e in entries]
    vectors = _embed_all(model, definitions)

    added = 0
    for entry, vec in zip(entries, vectors):
        label = entry.get("label", "").strip()
        if not label:
            continue
        exists = session.execute(
            select(CanonStoreEntry.id).where(func.lower(CanonStoreEntry.label) == label.lower())
        ).first()
        if exists is not None:
            continue
        session.add(
            CanonStoreEntry(
                label=label,
                definition=entry.get("definition", ""),
                turtle=entry.get("turtle"),
                embedding=vec.tolist(),
                source_doc=None,
            )
        )
        added += 1
    session.commit()
    print(f"[seed] canon_store: {added} entr(ies) (of {len(entries)})")
    return added


def seed_wikidata(session, model) -> int:
    if not WIKIDATA_FILTERED.exists():
        print(f"[seed]   ⚠ {WIKIDATA_FILTERED} not found, skipping wikidata properties")
        return 0
    props = json.loads(WIKIDATA_FILTERED.read_text())
    if not props:
        return 0

    descriptions = [p.get("description") or p.get("label", "") for p in props]
    print(f"[seed] embedding {len(descriptions)} wikidata descriptions …")
    vectors = _embed_all(model, descriptions)

    for p, vec in zip(props, vectors):
        stmt = pg_insert(WikidataProperty).values(
            pid=p["pid"],
            label=p["label"],
            description=p.get("description", ""),
            embedding=vec.tolist(),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["pid"],
            set_={
                "label": stmt.excluded.label,
                "description": stmt.excluded.description,
                "embedding": stmt.excluded.embedding,
            },
        )
        session.execute(stmt)
    session.commit()
    total = session.execute(select(func.count()).select_from(WikidataProperty)).scalar_one()
    print(f"[seed] wikidata_properties: {len(props)} upserted, {total} total")
    return len(props)


def main():
    if not DATABASE_URL:
        sys.exit("DATABASE_URL is not set (export it or add to ocr-resume-paser/.env).")

    factory = make_session_factory(DATABASE_URL)
    model = _embedder()

    with factory() as session:
        seed_gazetteers(session)
        seed_canon_store(session, model)
        seed_wikidata(session, model)

    print("[seed] ✓ done.")


if __name__ == "__main__":
    main()
