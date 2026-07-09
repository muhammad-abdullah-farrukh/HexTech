"""SQLAlchemy models mapping the structured resume output to Postgres.

`Resume` holds the `04_structured.json` payload verbatim in a JSONB column and is
keyed for dedup by `pdf_hash`. The `skills`, `work_history`, and `education`
child tables are a rebuildable projection derived from that JSON — a cache for
indexed queries, not a second source of truth — so they are wiped and re-derived
on every re-ingest via `ON DELETE CASCADE`.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ARRAY, TIMESTAMP, ForeignKey, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base; `Base.metadata` is Alembic's `target_metadata`."""


class Resume(Base):
    """One resume, keyed by the SHA-256 of its source PDF for dedup."""

    __tablename__ = "resumes"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    pdf_hash: Mapped[str] = mapped_column(unique=True, nullable=False)
    source_file: Mapped[str | None] = mapped_column(nullable=True)
    structured: Mapped[dict] = mapped_column(JSONB, nullable=False)
    field_spec_hash: Mapped[str] = mapped_column(nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )

    skills: Mapped[list[Skill]] = relationship(
        back_populates="resume", cascade="all, delete-orphan"
    )
    work_history: Mapped[list[WorkHistory]] = relationship(
        back_populates="resume", cascade="all, delete-orphan"
    )
    education: Mapped[list[Education]] = relationship(
        back_populates="resume", cascade="all, delete-orphan"
    )
    projects: Mapped[list[Project]] = relationship(
        back_populates="resume", cascade="all, delete-orphan"
    )


class Skill(Base):
    """A single skill string, projected from `structured['skills']`."""

    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    resume_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    skill: Mapped[str] = mapped_column(nullable=False, index=True)

    resume: Mapped[Resume] = relationship(back_populates="skills")


class WorkHistory(Base):
    """One employment entry, projected from `structured['work_history']`."""

    __tablename__ = "work_history"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    resume_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    company: Mapped[str | None] = mapped_column(nullable=True, index=True)
    title: Mapped[str | None] = mapped_column(nullable=True)
    start_date: Mapped[str | None] = mapped_column(nullable=True)
    end_date: Mapped[str | None] = mapped_column(nullable=True)

    resume: Mapped[Resume] = relationship(back_populates="work_history")


class Education(Base):
    """One education entry, projected from `structured['education']`."""

    __tablename__ = "education"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    resume_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    institution: Mapped[str | None] = mapped_column(nullable=True, index=True)
    degree: Mapped[str | None] = mapped_column(nullable=True)
    graduation_year: Mapped[str | None] = mapped_column(nullable=True)

    resume: Mapped[Resume] = relationship(back_populates="education")


class Project(Base):
    """One project entry, projected from `structured['projects']`.

    `technologies` is a string array (Postgres `text[]`) mirroring the field spec's
    `array<string>`, so it can be queried with `= ANY(technologies)`.
    """

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    resume_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str | None] = mapped_column(nullable=True, index=True)
    description: Mapped[str | None] = mapped_column(nullable=True)
    technologies: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)

    resume: Mapped[Resume] = relationship(back_populates="projects")
