"""
load_to_neo4j.py — incrementally load staged KG rows from Postgres into Neo4j.

Usage:
    python graphdb/load_to_neo4j.py            # push everything unsynced
    python graphdb/load_to_neo4j.py --wipe     # clear the graph first, then push
    python graphdb/load_to_neo4j.py --batch 1000

What it does (replaces the old file-based parse_ttl path):
    Reads graph_entities / graph_relationships WHERE synced_to_neo4j = FALSE in
    batches, MERGEs them into Neo4j keyed on properties->>'uri', and flips
    synced_to_neo4j = TRUE on success — so re-runs push only deltas and
    cross-document rows sharing a uri collapse onto one node.

    Every entity becomes a single :Entity node (uri + name + literal props);
    relationships use rel_type as the Neo4j relationship type.
"""
import argparse
import sys
from pathlib import Path

from neo4j import GraphDatabase
from sqlalchemy import select, text, update

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_URL
from db.models import GraphEntity, GraphRelationship
from db.session import make_session_factory
from graphdb.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

DEFAULT_BATCH = 500


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def load_entities(driver, session, batch: int) -> int:
    """MERGE unsynced graph_entities into Neo4j; flip synced on success."""
    rows = session.execute(
        select(GraphEntity.id, GraphEntity.properties).where(
            GraphEntity.synced_to_neo4j.is_(False)
        )
    ).all()
    if not rows:
        print("  no unsynced entities.")
        return 0

    synced_ids: list = []
    with driver.session(database=NEO4J_DATABASE) as neo:
        for chunk in _batched(rows, batch):
            node_rows = []
            chunk_ids = []
            for ent_id, props in chunk:
                uri = (props or {}).get("uri")
                if not uri:
                    continue  # can't MERGE without the dedup key
                node_rows.append(dict(props))
                chunk_ids.append(ent_id)
            if not node_rows:
                continue
            neo.run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {uri: row.uri})
                SET e += row
                """,
                rows=node_rows,
            )
            synced_ids.extend(chunk_ids)
            print(f"  merged {len(node_rows)} node(s)")

    if synced_ids:
        session.execute(
            update(GraphEntity)
            .where(GraphEntity.id.in_(synced_ids))
            .values(synced_to_neo4j=True)
        )
        session.commit()
    return len(synced_ids)


def load_relationships(driver, session, batch: int) -> int:
    """MERGE unsynced graph_relationships into Neo4j; flip synced on success."""
    rows = session.execute(
        text(
            """
            SELECT r.id AS id,
                   r.rel_type AS rel_type,
                   ef.properties->>'uri' AS from_uri,
                   et.properties->>'uri' AS to_uri
            FROM graph_relationships r
            JOIN graph_entities ef ON r.from_entity = ef.id
            JOIN graph_entities et ON r.to_entity = et.id
            WHERE r.synced_to_neo4j = FALSE
            """
        )
    ).all()
    if not rows:
        print("  no unsynced relationships.")
        return 0

    synced_ids: list = []
    with driver.session(database=NEO4J_DATABASE) as neo:
        for chunk in _batched(rows, batch):
            # Cypher can't parameterize relationship types — group by type.
            by_type: dict[str, list[dict]] = {}
            chunk_ids = []
            for row in chunk:
                if not row.from_uri or not row.to_uri:
                    continue
                by_type.setdefault(row.rel_type, []).append(
                    {"s": row.from_uri, "o": row.to_uri}
                )
                chunk_ids.append(row.id)
            for rtype, pairs in by_type.items():
                neo.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (a:Entity {{uri: row.s}})
                    MATCH (b:Entity {{uri: row.o}})
                    MERGE (a)-[:`{rtype}`]->(b)
                    """,
                    rows=pairs,
                )
            synced_ids.extend(chunk_ids)
            print(f"  merged {sum(len(v) for v in by_type.values())} relationship(s)")

    if synced_ids:
        session.execute(
            update(GraphRelationship)
            .where(GraphRelationship.id.in_(synced_ids))
            .values(synced_to_neo4j=True)
        )
        session.commit()
    return len(synced_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wipe", action="store_true", help="clear the whole Neo4j DB before loading")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="rows per batch")
    args = ap.parse_args()

    if not DATABASE_URL:
        print("✗ DATABASE_URL is not set.")
        return

    factory = make_session_factory(DATABASE_URL)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database=NEO4J_DATABASE) as neo:
            if args.wipe:
                print("wiping existing graph …")
                neo.run("MATCH (n) DETACH DELETE n")
            neo.run(
                "CREATE CONSTRAINT entity_uri IF NOT EXISTS "
                "FOR (n:Entity) REQUIRE n.uri IS UNIQUE"
            )

        with factory() as session:
            print("loading entities …")
            n_nodes = load_entities(driver, session, args.batch)
            print("loading relationships …")
            n_edges = load_relationships(driver, session, args.batch)
    finally:
        driver.close()

    print(f"done. synced {n_nodes} node(s), {n_edges} relationship(s).")
    print("Open Neo4j Browser and run:\n  MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 300")


if __name__ == "__main__":
    main()
