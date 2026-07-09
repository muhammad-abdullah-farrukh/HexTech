"""pipeline_runs helpers — the DB replacement for the outputs/<stage>/<doc>.json
existence checks that used to drive resume mode.

Keyed by (document_id, stage); output is the stage's JSON payload (a list/dict,
or {"ttl": "..."} for Turtle-producing stages).
"""
from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from db.models import PipelineRun

# Stage keys used across the pipeline (see plan §2).
CQ_GEN = "cq_gen"
CQ_ANSWER = "cq_answer"
RELATION_EXTRACT = "relation_extract"
MATCH_VALIDATE = "match_validate"
EDC_CANON = "edc_canon"
ONTOLOGY = "ontology"
KG_FACTS = "kg_facts"


def get_stage_output(session: Session, document_id, stage: str) -> dict | list | None:
    """Return a succeeded stage's stored output, or None if absent/unsuccessful."""
    row = session.execute(
        select(PipelineRun.output, PipelineRun.status).where(
            PipelineRun.document_id == document_id,
            PipelineRun.stage == stage,
        )
    ).first()
    if row is None or row.status != "succeeded":
        return None
    return row.output


def save_stage_output(
    session: Session,
    document_id,
    stage: str,
    output,
    status: str = "succeeded",
) -> None:
    """Upsert a stage's output on (document_id, stage) and commit."""
    stmt = pg_insert(PipelineRun).values(
        document_id=document_id,
        stage=stage,
        status=status,
        output=output,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["document_id", "stage"],
        set_={
            "status": stmt.excluded.status,
            "output": stmt.excluded.output,
            "updated_at": text("now()"),
        },
    )
    session.execute(stmt)
    session.commit()


def has_succeeded(session: Session, document_id, stage: str) -> bool:
    """True if (document_id, stage) has a succeeded run — used by --resume."""
    status = session.execute(
        select(PipelineRun.status).where(
            PipelineRun.document_id == document_id,
            PipelineRun.stage == stage,
        )
    ).scalar_one_or_none()
    return status == "succeeded"
