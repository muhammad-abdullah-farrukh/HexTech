"""verify_verdicts — labeled relation-equivalence judgments for the EDC
verify classifier (teacher LLM verdicts + cross-encoder decisions).

Serves two purposes at once:
  - training corpus for distilling the deepseek verify judgment into a
    local cross-encoder (Phase B fine-tune);
  - audit trail of every merge/reject decision, whoever made it.

resume_id → ON DELETE SET NULL: a verdict about two relation *definitions*
stays valid training data after its originating résumé is deleted, same
rationale as canon_store.source_doc in 0003.

Revision ID: 0004_verify_verdicts
Revises: 0003_ontogen_schema
Create Date: 2026-07-06
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0004_verify_verdicts"
down_revision: Union[str, None] = "0003_ontogen_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "verify_verdicts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("label_a", sa.Text(), nullable=False),
        sa.Column("definition_a", sa.Text(), nullable=False),
        sa.Column("label_b", sa.Text(), nullable=False),
        sa.Column("definition_b", sa.Text(), nullable=False),
        sa.Column("accepted", sa.Boolean(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column(
            "resume_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("resumes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_verify_verdicts_label_a", "verify_verdicts", ["label_a"])
    op.create_index("ix_verify_verdicts_label_b", "verify_verdicts", ["label_b"])


def downgrade() -> None:
    op.drop_index("ix_verify_verdicts_label_b", table_name="verify_verdicts")
    op.drop_index("ix_verify_verdicts_label_a", table_name="verify_verdicts")
    op.drop_table("verify_verdicts")
