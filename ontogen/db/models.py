"""SQLAlchemy models for Ontogen's Postgres tables (Part 2 schema, §2).

Declared on the *same* ``Base`` as ocr_resume_parser's models, so everything
lives in one registry: the FKs to ``resumes`` resolve, and one
``Base.metadata`` covers both projects' tables. ``vector(384)`` columns use
pgvector; the dimension matches config.EMBED_DIM (bge-small-en).
"""
from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import TIMESTAMP, Boolean, Float, ForeignKey, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

# Reuse the parser's declarative Base and Resume so FKs share one metadata.
from resume_parser.db.models import Base, Resume  # noqa: F401  (Resume re-exported for FK targets)

EMBED_DIM = 384  # bge-small-en; kept local so models import without config


class PipelineRun(Base):
    """Per-document, per-stage checkpoint — replaces outputs/*/*.{json,ttl}."""

    __tablename__ = "pipeline_runs"

    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    stage: Mapped[str] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(nullable=False, server_default=text("'succeeded'"))
    output: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )


class CanonStoreEntry(Base):
    """EDC cross-document canon store — replaces data/canon_store/entries.json."""

    __tablename__ = "canon_store"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    label: Mapped[str] = mapped_column(nullable=False)
    definition: Mapped[str] = mapped_column(nullable=False)
    turtle: Mapped[str | None] = mapped_column(nullable=True)
    # source_doc is a provenance pointer, not ownership: a canon entry stays
    # useful for canonicalizing *other* documents after its originating résumé
    # is gone, so deleting that résumé nulls the pointer rather than the entry.
    source_doc: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="SET NULL"), nullable=True
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)
    added_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )


class VerifyVerdict(Base):
    """One relation-equivalence judgment — the labeled corpus that trains and
    audits the EDC verify classifier.

    Every verdict is logged regardless of who decided (``source``:
    'deepseek' teacher LLM vs 'cross-encoder' student), so the table is both
    the fine-tuning dataset and a full audit trail of merge decisions.
    ``resume_id`` is a provenance pointer (SET NULL on résumé delete), same
    rationale as canon_store.source_doc.
    """

    __tablename__ = "verify_verdicts"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    label_a: Mapped[str] = mapped_column(nullable=False, index=True)
    definition_a: Mapped[str] = mapped_column(nullable=False)
    label_b: Mapped[str] = mapped_column(nullable=False, index=True)
    definition_b: Mapped[str] = mapped_column(nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(nullable=False)  # 'deepseek' | 'cross-encoder'
    resume_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )


class VerifyThreshold(Base):
    """Current live cosine thresholds for the EDC verify gate — a single
    growing row history, latest row (by computed_at) wins.

    Deliberately NOT config.py constants: thresholds must only move once
    accumulated verify_verdicts support them (see
    scripts/recompute_thresholds.py's sample-size + precision gates), so the
    "current" value is data the system computes for itself over time, not a
    file an operator edits. n_pairs_used records how much evidence backed
    this value, so a low-confidence early recomputation is visibly weaker
    than a later one — auditable, not just a number.
    """

    __tablename__ = "verify_thresholds"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tau_hi: Mapped[float] = mapped_column(Float, nullable=False)
    tau_lo: Mapped[float] = mapped_column(Float, nullable=False)
    n_pairs_used: Mapped[int] = mapped_column(nullable=False)
    merge_precision: Mapped[float] = mapped_column(Float, nullable=False)
    reject_precision: Mapped[float] = mapped_column(Float, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )


class Gazetteer(Base):
    """Alias → canonical lookup — replaces data/gazetteers/*.json (5 files → 1).

    ``wikidata_qid`` is a deliberate addition beyond the plan's §2 schema: the
    gazetteer JSON carries a canonical→QID map that ResumeEntityResolver uses to
    build ``wd:Q…`` URIs, and dropping it would silently degrade resolution to
    slug URIs.
    """

    __tablename__ = "gazetteers"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    entity_type: Mapped[str] = mapped_column(nullable=False)
    alias: Mapped[str] = mapped_column(nullable=False)
    canonical: Mapped[str] = mapped_column(nullable=False)
    wikidata_qid: Mapped[str | None] = mapped_column(nullable=True)
    source: Mapped[str] = mapped_column(nullable=False, server_default=text("'static'"))
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )


class WikidataProperty(Base):
    """Filtered Wikidata property + embedding — replaces properties_filtered.json
    + wikidata_embeddings.npy."""

    __tablename__ = "wikidata_properties"

    pid: Mapped[str] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str | None] = mapped_column(nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)


class Provenance(Base):
    """Per-triple provenance — replaces outputs/provenance/*.json."""

    __tablename__ = "provenance"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    subject: Mapped[str] = mapped_column(nullable=False)
    predicate: Mapped[str] = mapped_column(nullable=False)
    object: Mapped[str] = mapped_column(nullable=False)
    stage: Mapped[str] = mapped_column(nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model: Mapped[str] = mapped_column(nullable=False)
    extra: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ts: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )


class GraphEntity(Base):
    """KG node staged before Neo4j load."""

    __tablename__ = "graph_entities"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    entity_type: Mapped[str] = mapped_column(nullable=False)
    canonical_group: Mapped[str | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    properties: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Provenance pointer, not ownership: a canonical_group-merged node may be
    # pointed at by other résumés, so a résumé delete nulls source_doc rather
    # than cascading away graph data still in use elsewhere.
    source_doc: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="SET NULL"), nullable=True
    )
    synced_to_neo4j: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )


class GraphRelationship(Base):
    """KG edge staged before Neo4j load."""

    __tablename__ = "graph_relationships"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    from_entity: Mapped[str] = mapped_column(
        UUID(as_uuid=True), ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False
    )
    to_entity: Mapped[str] = mapped_column(
        UUID(as_uuid=True), ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False
    )
    rel_type: Mapped[str] = mapped_column(nullable=False)
    properties: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # from_entity/to_entity cascade (an edge dies with its node); source_doc is
    # a provenance pointer, so a résumé delete nulls it rather than cascading.
    source_doc: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="SET NULL"), nullable=True
    )
    synced_to_neo4j: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )
