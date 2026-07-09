"""Persist a structured resume dict into Postgres.

`make_ingest_fn` returns a closure matching the `ingest_fn(structured, pdf_path)`
signature the pipeline expects, so the pipeline never imports anything from this
package directly. The upsert is keyed on `pdf_hash` and uses Postgres's native
`INSERT ... ON CONFLICT DO UPDATE` — atomic, so two workers racing on the same
PDF cannot duplicate a row or raise a unique-violation. The projection tables are
wiped and re-derived on every call rather than diffed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import delete, insert, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import sessionmaker

from .models import Education, Project, Resume, Skill, WorkHistory


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_ingest_fn(
    session_factory: sessionmaker, field_spec_path: str
) -> Callable[[dict, str], str]:
    """Return a closure that upserts `structured` into the database.

    The closure computes `field_spec_hash` from `field_spec_path` on each call, so
    a stored record always records the spec version that produced it.
    """

    def ingest_resume(structured: dict, source_pdf_path: str) -> str:
        """Upsert one resume + its projection rows; return the resume id as a string."""
        pdf_hash = _sha256(Path(source_pdf_path).read_bytes())
        field_spec_hash = _sha256(Path(field_spec_path).read_bytes())

        session = session_factory()
        try:
            stmt = pg_insert(Resume).values(
                pdf_hash=pdf_hash,
                source_file=source_pdf_path,
                structured=structured,
                field_spec_hash=field_spec_hash,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["pdf_hash"],
                set_={
                    "source_file": stmt.excluded.source_file,
                    "structured": stmt.excluded.structured,
                    "field_spec_hash": stmt.excluded.field_spec_hash,
                    "ingested_at": text("now()"),
                },
            ).returning(Resume.id)
            resume_id = session.execute(stmt).scalar_one()

            # Wipe and re-derive the projection so it never unions old and new data.
            session.execute(delete(Skill).where(Skill.resume_id == resume_id))
            session.execute(
                delete(WorkHistory).where(WorkHistory.resume_id == resume_id)
            )
            session.execute(delete(Education).where(Education.resume_id == resume_id))
            session.execute(delete(Project).where(Project.resume_id == resume_id))

            skill_rows = [
                {"resume_id": resume_id, "skill": s}
                for s in (structured.get("skills") or [])
                if s
            ]
            if skill_rows:
                session.execute(insert(Skill), skill_rows)

            work_rows = [
                {
                    "resume_id": resume_id,
                    "company": e.get("company"),
                    "title": e.get("title"),
                    "start_date": e.get("start_date"),
                    "end_date": e.get("end_date"),
                }
                for e in (structured.get("work_history") or [])
            ]
            if work_rows:
                session.execute(insert(WorkHistory), work_rows)

            education_rows = [
                {
                    "resume_id": resume_id,
                    "institution": e.get("institution"),
                    "degree": e.get("degree"),
                    "graduation_year": e.get("graduation_year"),
                }
                for e in (structured.get("education") or [])
            ]
            if education_rows:
                session.execute(insert(Education), education_rows)

            project_rows = [
                {
                    "resume_id": resume_id,
                    "name": e.get("name"),
                    "description": e.get("description"),
                    "technologies": e.get("technologies"),
                }
                for e in (structured.get("projects") or [])
            ]
            if project_rows:
                session.execute(insert(Project), project_rows)

            # One commit for the upsert + projection rebuild: a crash mid-ingest must
            # never leave a resume row pointing at stale or half-written children.
            session.commit()
            return str(resume_id)
        finally:
            session.close()

    return ingest_resume
