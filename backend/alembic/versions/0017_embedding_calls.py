"""The embedding metering table (RMD9-T04; RE2E-DEF-029).

``embedding_calls`` holds one row per embedding provider call, at ingestion or at query time: its
project and exactly one of the document or the run it served, the vendor and model, the input
count, the prompt tokens the response reported (NULL when it reported none), the latency, the
outcome with the failure kind, and the time. No input text. The scope columns cascade with their
project, document and run.

Additive. The downgrade drops exactly this table, so every other row survives it.

Revision ID: 0017
Revises: 0016
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "embedding_calls",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "project_id",
            sa.Uuid(),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "query_run_id",
            sa.Uuid(),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("input_count", sa.Integer(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("failure_kind", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(document_id IS NULL) <> (query_run_id IS NULL)", name="ck_embedding_calls_one_scope"
        ),
    )
    op.create_index("ix_embedding_calls_project_id", "embedding_calls", ["project_id"])
    op.create_index("ix_embedding_calls_document_id", "embedding_calls", ["document_id"])
    op.create_index("ix_embedding_calls_query_run_id", "embedding_calls", ["query_run_id"])


def downgrade() -> None:
    op.drop_index("ix_embedding_calls_query_run_id", table_name="embedding_calls")
    op.drop_index("ix_embedding_calls_document_id", table_name="embedding_calls")
    op.drop_index("ix_embedding_calls_project_id", table_name="embedding_calls")
    op.drop_table("embedding_calls")
