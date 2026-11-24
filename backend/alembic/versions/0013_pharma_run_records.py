"""The pharmaceutical run record: one row per pharmaceutical query run.

``pharma_run_records`` keeps what a pharmaceutical run was asked and what it showed (REQ-PER-01
to REQ-PER-04): the intent, the incident and the filters with the request's correlation id and
the fixed "who initiated" statement, written when the run is created; then the evidence
snapshot, the sufficiency with its inputs and rule version, the conflicts, the policy result,
the escalation and the policy-checked response, written when the run finishes. Model,
provider and routing profile stay on ``query_runs`` and ``agent_runs`` (REQ-PER-05).

The row's key is the run's id and both foreign keys cascade, so deleting a run or its project
deletes the record (REQ-PER-08). The snapshot is a copy, not a reference: it holds each
passage's hash and a bounded excerpt, so later changes to the documents do not change it
(REQ-PER-03). General runs never write the table.

Additive. The downgrade drops the table and its index and nothing else, so every row of a
pre-existing table survives it.

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json(name: str, default: str | None) -> sa.Column[object]:
    if default is None:
        return sa.Column(name, postgresql.JSONB(), nullable=True)
    return sa.Column(
        name, postgresql.JSONB(), nullable=False, server_default=sa.text(f"'{default}'::jsonb")
    )


def upgrade() -> None:
    op.create_table(
        "pharma_run_records",
        sa.Column(
            "query_run_id",
            sa.Uuid(),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "project_id",
            sa.Uuid(),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("intent", sa.String(length=30), nullable=False),
        _json("incident", None),
        _json("filters", "{}"),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.Column("initiated_by", sa.Text(), nullable=False),
        sa.Column("rule_version", sa.Text(), nullable=True),
        _json("evidence_snapshot", "[]"),
        _json("sufficiency", None),
        _json("conflicts", "[]"),
        _json("policy", None),
        _json("escalation", None),
        _json("response", None),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "intent IN ('procedure_question', 'operational_incident', 'readiness_review')",
            name="ck_pharma_run_records_intent",
        ),
    )
    op.create_index("ix_pharma_run_records_project_id", "pharma_run_records", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_pharma_run_records_project_id", table_name="pharma_run_records")
    op.drop_table("pharma_run_records")
