"""verify_thresholds — adaptive, data-backed cosine thresholds for the EDC
verify gate (replaces hand-picked constants in config.py).

Append-only history; the latest row (by computed_at) is the active
threshold. recompute_thresholds.py only inserts a new row once accumulated
verify_verdicts pass both a minimum-sample-size gate and the existing
merge/reject-precision bar — see that script's docstring.

Revision ID: 0005_verify_thresholds
Revises: 0004_verify_verdicts
Create Date: 2026-07-07
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0005_verify_thresholds"
down_revision: Union[str, None] = "0004_verify_verdicts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "verify_thresholds",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("tau_hi", sa.Float(), nullable=False),
        sa.Column("tau_lo", sa.Float(), nullable=False),
        sa.Column("n_pairs_used", sa.Integer(), nullable=False),
        sa.Column("merge_precision", sa.Float(), nullable=False),
        sa.Column("reject_precision", sa.Float(), nullable=False),
        sa.Column(
            "computed_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_verify_thresholds_computed_at", "verify_thresholds", ["computed_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_verify_thresholds_computed_at", table_name="verify_thresholds")
    op.drop_table("verify_thresholds")
