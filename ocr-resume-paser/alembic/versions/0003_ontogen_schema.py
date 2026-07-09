"""ontogen knowledge-graph schema

Adds the Ontogen (Part 2) tables to the same database as the parser: per-stage
run tracking, the EDC canon store, gazetteers, Wikidata properties, provenance,
and the graph_entities/graph_relationships staging tables. Requires the pgvector
extension (the compose image is pgvector/pgvector:pg17).

Two deliberate choices beyond the plan's §2 SQL:
  - gazetteers.wikidata_qid — the gazetteer JSON carries a canonical→QID map
    used to build wd:Q… URIs; without a column it would be lost.
  - FK delete behaviour is split by ownership, not left to default:
      * pipeline_runs.document_id / provenance.document_id → ON DELETE CASCADE
        (genuinely per-document data, owned by the résumé like skills in Part 1).
      * canon_store.source_doc / graph_entities.source_doc /
        graph_relationships.source_doc → ON DELETE SET NULL. These are
        provenance pointers to the *originating* résumé, but the rows may still
        be in use for canonicalizing / graphing other documents, so deleting a
        résumé nulls the pointer instead of cascading away shared data (and
        instead of RESTRICT blocking the delete outright).

Revision ID: 0003_ontogen_schema
Revises: 0002_add_projects
Create Date: 2026-07-02
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003_ontogen_schema"
down_revision: Union[str, None] = "0002_add_projects"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBED_DIM = 384  # bge-small-en


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── pipeline_runs (per-document → CASCADE) ─────────────────────────────
    op.create_table(
        "pipeline_runs",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'succeeded'")),
        sa.Column("output", postgresql.JSONB(), nullable=True),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["resumes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("document_id", "stage"),
    )

    # ── canon_store (source_doc = provenance pointer → SET NULL) ───────────
    op.create_table(
        "canon_store",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("definition", sa.Text(), nullable=False),
        sa.Column("turtle", sa.Text(), nullable=True),
        sa.Column("source_doc", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=True),
        sa.Column(
            "added_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["source_doc"], ["resumes.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        "CREATE INDEX ix_canon_store_embedding ON canon_store "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("CREATE INDEX ix_canon_store_label ON canon_store (lower(label))")

    # ── gazetteers ─────────────────────────────────────────────────────────
    op.create_table(
        "gazetteers",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("entity_type", sa.Text(), nullable=False),
        sa.Column("alias", sa.Text(), nullable=False),
        sa.Column("canonical", sa.Text(), nullable=False),
        sa.Column("wikidata_qid", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=False, server_default=sa.text("'static'")),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        "CREATE INDEX ix_gazetteers_alias ON gazetteers (entity_type, lower(alias))"
    )

    # ── wikidata_properties ────────────────────────────────────────────────
    op.create_table(
        "wikidata_properties",
        sa.Column("pid", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=True),
        sa.PrimaryKeyConstraint("pid"),
    )
    op.execute(
        "CREATE INDEX ix_wikidata_properties_embedding ON wikidata_properties "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    # ── provenance (per-document → CASCADE) ────────────────────────────────
    op.create_table(
        "provenance",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("predicate", sa.Text(), nullable=False),
        sa.Column("object", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("extra", postgresql.JSONB(), nullable=True),
        sa.Column(
            "ts",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["resumes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_provenance_document_id", "provenance", ["document_id"])

    # ── graph_entities (source_doc = provenance pointer → SET NULL) ────────
    op.create_table(
        "graph_entities",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("entity_type", sa.Text(), nullable=False),
        sa.Column("canonical_group", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("properties", postgresql.JSONB(), nullable=False),
        sa.Column("source_doc", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "synced_to_neo4j", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["source_doc"], ["resumes.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        "CREATE INDEX ix_graph_entities_uri ON graph_entities (((properties->>'uri')))"
    )
    op.execute(
        "CREATE INDEX ix_graph_entities_unsynced ON graph_entities (synced_to_neo4j) "
        "WHERE synced_to_neo4j = FALSE"
    )

    # ── graph_relationships (endpoints CASCADE; source_doc → SET NULL) ─────
    op.create_table(
        "graph_relationships",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("from_entity", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("to_entity", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rel_type", sa.Text(), nullable=False),
        sa.Column("properties", postgresql.JSONB(), nullable=True),
        sa.Column("source_doc", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "synced_to_neo4j", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["from_entity"], ["graph_entities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["to_entity"], ["graph_entities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_doc"], ["resumes.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        "CREATE INDEX ix_graph_relationships_unsynced ON graph_relationships "
        "(synced_to_neo4j) WHERE synced_to_neo4j = FALSE"
    )


def downgrade() -> None:
    op.drop_table("graph_relationships")
    op.drop_table("graph_entities")
    op.drop_table("provenance")
    op.drop_table("wikidata_properties")
    op.drop_table("gazetteers")
    op.drop_table("canon_store")
    op.drop_table("pipeline_runs")
